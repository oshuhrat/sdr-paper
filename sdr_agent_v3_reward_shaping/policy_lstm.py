"""
LSTM Actor-Critic policy for goal-conditioned GridWorld.

LSTMPolicy:
  obs (225-dim) → obs_encoder → obs_z (64-dim)
  [obs_z, goal_z (32)] → LSTMCell → h_t (128-dim)
  h_t → actor  → logits (5)
  h_t → critic → value  (1)

LSTMPolicySim (diagnostic binding module):
  Adds GoalObjectSim features before LSTM:
    goal_z → key (9-dim) → dot-product with each cell (25 cells × 9 features)
    → top-3 similarity scores + argmax (row, col) = 5-dim sim_feats
  [obs_z (64), goal_z (32), sim_feats (5)] → LSTMCell (101-dim input)
  Tests whether explicit goal↔object matching resolves holdout generalization.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent / "sdr_agent_v3_transfer_action"))

from env            import OBS_DIM, N_ACTIONS, VIEW_SIZE, N_FEATURES   # noqa: E402
from oracle_encoder import GOAL_DIM                                       # noqa: E402

LSTM_HIDDEN = 128
SIM_DIM     = 5   # GoalObjectSim output: top3_sims (3) + argmax_row (1) + argmax_col (1)


class LSTMPolicy(nn.Module):
    def __init__(self, goal_encoder, obs_hidden: int = 128, lstm_hidden: int = LSTM_HIDDEN,
                 extra_lstm_in: int = 0):
        super().__init__()
        self.goal_encoder = goal_encoder
        self.lstm_hidden  = lstm_hidden

        self.obs_encoder = nn.Sequential(
            nn.Linear(OBS_DIM, obs_hidden),
            nn.LayerNorm(obs_hidden),
            nn.ReLU(),
            nn.Linear(obs_hidden, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(64 + GOAL_DIM + extra_lstm_in, lstm_hidden)

        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_goal(self, colors, sizes, shapes):
        return self.goal_encoder(colors, sizes, shapes)

    def init_hidden(self):
        h = torch.zeros(1, self.lstm_hidden)
        c = torch.zeros(1, self.lstm_hidden)
        return h, c

    def _cell(self, obs, goal_z, h, c):
        """One LSTM step. All inputs/outputs are (B, *). Returns logits, value, h', c'."""
        obs_z   = self.obs_encoder(obs)
        h, c    = self.lstm(torch.cat([obs_z, goal_z], dim=-1), (h, c))
        logits  = self.actor(h)
        value   = self.critic(h).squeeze(-1)
        return logits, value, h, c

    # ── Interfaces expected by PPO trainer ──────────────────────────────────────

    def act(self, obs, goal_z, h, c):
        """Stochastic action sampling for rollout collection."""
        logits, value, h, c = self._cell(obs, goal_z, h, c)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, h, c

    def evaluate_sequence(self, obs_T, goals_T, actions_T, dones_T):
        """
        Recompute log_probs, values, entropy for a full rollout sequence.
        Episode boundaries (done_T[t] == 1) reset (h, c) for the NEXT step.

        obs_T:     (T, OBS_DIM)
        goals_T:   (T, GOAL_DIM)
        actions_T: (T,)  int64
        dones_T:   (T,)  float  — 1.0 means episode ended at step t

        Returns log_probs (T,), values (T,), entropy (T,)
        """
        T = obs_T.shape[0]
        h, c = self.init_hidden()

        obs_z_all   = self.obs_encoder(obs_T)                       # (T, 64)
        lstm_in_all = torch.cat([obs_z_all, goals_T], dim=-1)       # (T, 96)

        log_probs, values, entropies = [], [], []
        for t in range(T):
            h, c   = self.lstm(lstm_in_all[t:t+1], (h, c))
            logits = self.actor(h)
            value  = self.critic(h).squeeze(-1)
            dist   = Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions_T[t:t+1]))
            values.append(value)
            entropies.append(dist.entropy())
            if dones_T[t] > 0.5 and t < T - 1:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)

        return (
            torch.cat(log_probs),
            torch.cat(values),
            torch.cat(entropies),
        )

    def forward(self, obs, goal_z, h=None, c=None):
        """Compatibility shim for evaluate.py (greedy with fresh hidden state)."""
        if h is None:
            h, c = self.init_hidden()
        logits, _, _, _ = self._cell(obs, goal_z, h, c)
        return logits, None


# ── Goal-Object Binding Diagnostic Module ─────────────────────────────────────

class GoalObjectSim(nn.Module):
    """
    Per-cell compatibility between goal_z and observed cell features.

    goal_z (32-dim) → Linear(32, 9) → key (9-dim)
    obs (225-dim)   → reshape → (25, 9) cells
    sim_i = dot(cell_i, key)  for i in 0..24     → scores (25,)
    Output: [top3_sims (3), argmax_row (1), argmax_col (1)] = 5-dim

    The argmax cell is the observation cell most compatible with the goal.
    top-3 similarities capture rank ordering of candidate objects.
    """

    def __init__(self, cell_dim: int = N_FEATURES, goal_dim: int = GOAL_DIM):
        super().__init__()
        self.key_proj = nn.Linear(goal_dim, cell_dim, bias=False)

    def forward(self, obs: torch.Tensor, goal_z: torch.Tensor) -> torch.Tensor:
        B       = obs.shape[0]
        n_cells = VIEW_SIZE * VIEW_SIZE                                    # 25
        cells   = obs.view(B, n_cells, N_FEATURES)                        # (B, 25, 9)
        g_key   = self.key_proj(goal_z)                                   # (B, 9)
        scores  = torch.bmm(cells, g_key.unsqueeze(-1)).squeeze(-1)       # (B, 25)
        top3, top3_idx = scores.topk(3, dim=1)                            # (B, 3), (B, 3)
        argmax  = top3_idx[:, 0]                                           # (B,)
        row     = (argmax // VIEW_SIZE).float() / max(VIEW_SIZE - 1, 1)   # (B,) in [0,1]
        col     = (argmax  % VIEW_SIZE).float() / max(VIEW_SIZE - 1, 1)   # (B,) in [0,1]
        return torch.cat([top3, row.unsqueeze(1), col.unsqueeze(1)], dim=1)  # (B, 5)


class LSTMPolicySim(LSTMPolicy):
    """
    LSTMPolicy + GoalObjectSim diagnostic binding module.
    LSTM input: [obs_z (64), goal_z (32), sim_feats (5)] = 101-dim.
    Tests whether explicit goal↔object dot-product resolves holdout generalization.
    """

    def __init__(self, goal_encoder, obs_hidden: int = 128, lstm_hidden: int = LSTM_HIDDEN):
        super().__init__(goal_encoder, obs_hidden, lstm_hidden, extra_lstm_in=SIM_DIM)
        self.sim_module = GoalObjectSim()

    def _cell(self, obs, goal_z, h, c):
        obs_z  = self.obs_encoder(obs)
        sim    = self.sim_module(obs, goal_z)
        h, c   = self.lstm(torch.cat([obs_z, goal_z, sim], dim=-1), (h, c))
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value, h, c

    def evaluate_sequence(self, obs_T, goals_T, actions_T, dones_T):
        T = obs_T.shape[0]
        h, c = self.init_hidden()
        obs_z_all = self.obs_encoder(obs_T)
        sim_all   = self.sim_module(obs_T, goals_T)
        lstm_in   = torch.cat([obs_z_all, goals_T, sim_all], dim=-1)

        log_probs, values, entropies = [], [], []
        for t in range(T):
            h, c   = self.lstm(lstm_in[t:t+1], (h, c))
            logits = self.actor(h)
            value  = self.critic(h).squeeze(-1)
            dist   = Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions_T[t:t+1]))
            values.append(value)
            entropies.append(dist.entropy())
            if dones_T[t] > 0.5 and t < T - 1:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)

        return torch.cat(log_probs), torch.cat(values), torch.cat(entropies)
