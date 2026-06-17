"""
LSTM PPO Trainer.

Key differences from standard PPO:
- Maintains (h, c) hidden state across steps within a rollout.
- Resets (h, c) = 0 at episode boundaries (done=True).
- Update replays the entire sequence (no random minibatches) using
  policy.evaluate_sequence(), which handles the done-based resets.
- n_epochs=4 epochs of the full sequence per update.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from env            import OBS_DIM, N_ACTIONS   # noqa: E402
from oracle_encoder import GOAL_DIM               # noqa: E402


class LSTMRolloutBuffer:
    def __init__(self, n_steps: int):
        self.n_steps  = n_steps
        self.obs      = np.zeros((n_steps, OBS_DIM),  dtype=np.float32)
        self.goals    = np.zeros((n_steps, GOAL_DIM), dtype=np.float32)
        self.actions  = np.zeros(n_steps,              dtype=np.int64)
        self.rewards  = np.zeros(n_steps,              dtype=np.float32)
        self.dones    = np.zeros(n_steps,              dtype=np.float32)
        self.values   = np.zeros(n_steps,              dtype=np.float32)
        self.log_probs= np.zeros(n_steps,              dtype=np.float32)
        self.pos      = 0

    def add(self, obs, goal_z, action, reward, done, value, log_prob):
        i = self.pos
        self.obs[i]       = obs
        self.goals[i]     = goal_z
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.pos          = i + 1

    def reset(self):
        self.pos = 0

    def full(self) -> bool:
        return self.pos >= self.n_steps

    def compute_returns(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        adv  = np.zeros(self.pos, dtype=np.float32)
        last = 0.0
        for t in reversed(range(self.pos)):
            nv   = last_value if t == self.pos - 1 else self.values[t + 1]
            done = 0.0       if t == self.pos - 1 else self.dones[t + 1]
            delta  = self.rewards[t] + gamma * nv * (1 - done) - self.values[t]
            adv[t] = last = delta + gamma * gae_lambda * (1 - done) * last
        self.advantages = adv
        self.returns    = adv + self.values[: self.pos]


class LSTMPPOTrainer:
    def __init__(
        self,
        policy,
        lr:            float = 3e-4,
        clip_eps:      float = 0.2,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.001,
        max_grad_norm: float = 0.5,
        n_epochs:      int   = 4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        n_steps:       int   = 1024,
    ):
        self.policy   = policy
        self.clip_eps = clip_eps
        self.vf_coef  = vf_coef
        self.ent_coef = ent_coef
        self.max_grad = max_grad_norm
        self.n_epochs = n_epochs
        self.gamma    = gamma
        self.gae_lambda = gae_lambda
        self.n_steps  = n_steps
        self.optimizer = Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.buffer   = LSTMRolloutBuffer(n_steps)

    def collect_rollout(self, env, current_obs, current_goal_attrs):
        self.buffer.reset()
        self.policy.eval()
        ep_stats  = []
        ep_reward = 0.0
        ep_len    = 0
        obs       = current_obs
        gc, gs, gsh = current_goal_attrs

        h, c = self.policy.init_hidden()

        with torch.no_grad():
            for _ in range(self.n_steps):
                obs_t  = torch.from_numpy(obs).unsqueeze(0)
                goal_t = self.policy.encode_goal([gc], [gs], [gsh])

                action, log_prob, value, h, c = self.policy.act(obs_t, goal_t, h, c)
                a  = int(action.item())
                lp = float(log_prob.item())
                v  = float(value.item())
                gz = goal_t.squeeze(0).numpy()

                next_obs, reward, done, info = env.step(a)
                ep_reward += reward
                ep_len    += 1

                self.buffer.add(obs, gz, a, reward, done, v, lp)
                obs = next_obs

                if done:
                    ep_stats.append({
                        "success":   info["success"],
                        "ep_len":    ep_len,
                        "ep_reward": ep_reward,
                    })
                    ep_reward = 0.0
                    ep_len    = 0
                    obs, info = env.reset()
                    gc  = info["goal_combo"][0]
                    gs  = info["goal_combo"][1]
                    gsh = info["goal_combo"][2]
                    h, c = self.policy.init_hidden()

            # Bootstrap value for last state
            obs_t  = torch.from_numpy(obs).unsqueeze(0)
            goal_t = self.policy.encode_goal([gc], [gs], [gsh])
            _, _, last_v, _, _ = self.policy.act(obs_t, goal_t, h, c)

        self.buffer.compute_returns(float(last_v.item()), self.gamma, self.gae_lambda)
        return obs, (gc, gs, gsh), ep_stats

    def update(self) -> dict:
        self.policy.train()
        T      = self.buffer.pos
        obs    = torch.from_numpy(self.buffer.obs[:T])
        goals  = torch.from_numpy(self.buffer.goals[:T])
        acts   = torch.from_numpy(self.buffer.actions[:T])
        dones  = torch.from_numpy(self.buffer.dones[:T])
        rets   = torch.from_numpy(self.buffer.returns)
        advs   = torch.from_numpy(self.buffer.advantages)
        old_lp = torch.from_numpy(self.buffer.log_probs[:T])

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        pg_losses, vf_losses, ent_losses = [], [], []
        for _ in range(self.n_epochs):
            lp, val, ent = self.policy.evaluate_sequence(obs, goals, acts, dones)

            ratio   = torch.exp(lp - old_lp)
            sur1    = ratio * advs
            sur2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advs
            pg_loss = -torch.min(sur1, sur2).mean()
            vf_loss = F.mse_loss(val, rets)
            ent_loss = -ent.mean()
            loss    = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
            self.optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent_loss.item())

        return {
            "pg_loss":  float(np.mean(pg_losses)),
            "vf_loss":  float(np.mean(vf_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }
