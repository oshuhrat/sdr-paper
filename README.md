# Sparse Distributed Representations Enable Compositional Generalisation

Code for the paper:

> **Sparse Distributed Representations Enable Compositional Generalisation**
> [arXiv placeholder]

## Overview

We test the SDR hypothesis chain across three levels:

| Level | Hypothesis | Experiment |
|-------|-----------|------------|
| SDR-L1 | Stable forms preserve structural invariants | Cellular automaton SR metrics |
| SDR-L3 | New concepts = compositions of existing concepts | Contrastive concept composition (CCS) |
| SDR-L4 | SDR composition enables action transfer to novel goal combinations | Goal-conditioned GridWorld agent |

## Results

### Table 1 — Structural Retention (SR_shift at τ=200)

| Object | SR_shift (mean ± std, 5 seeds) |
|--------|-------------------------------|
| Block | 1.0000 ± 0.0000 |
| Glider | 1.0000 ± 0.0000 |
| Random | 0.0017 ± 0.0000 |

SR_shift = shift-invariant NMI via FFT cross-correlation. Structured patterns (block, glider) retain perfect spatial coherence; random noise does not.

### Table 2 — Concept Composition Score (CCS, 5 seeds, 5 epochs)

| Method | CCS (mean ± std) |
|--------|-----------------|
| Learned Linear | **0.877 ± 0.028** |
| Weighted Mean | 0.748 ± 0.031 |
| Simple Mean | 0.429 ± 0.030 |
| Random Baseline | −0.123 ± 0.004 |

CCS = cosine_similarity(z_real_unseen, Compose(C_color, C_size, C_shape)).
Linear probing achieves 100% attribute accuracy at both epoch 5 and epoch 30,
confirming that CCS measures additive geometric structure, not linear separability.

### Table 3 — Agent Goal-Object Binding (GridWorld, 800 rollouts)

| Encoder | Holdout SR ± 95CI | GR |
|---------|-------------------|----|
| Oracle (frozen random projection) | 0.320 ± 0.075 | 0.972 |
| SDR (ConceptComposer) | 0.320 ± 0.075 | 0.916 |
| Random | 0.267 ± 0.071 | 0.797 |

CG = Oracle − SDR = 0.000. SDR encoder matches oracle on holdout generalisation.

## Replication

```bash
pip install -r requirements.txt
```

### Table 1: Structural Retention Test

```bash
cd sdr_structural_retention_test
for seed in 42 43 44 45 46; do
    python -X utf8 main.py --seed $seed
done
```

### Table 2: Concept Composition Score

```bash
cd sdr_agent_v2_concept_discovery
for seed in 42 43 44 45 46; do
    python -X utf8 main.py --fast --seed $seed --no-tsne
done
```

### Table 3: Agent Goal-Object Binding

```bash
cd sdr_agent_v3_reward_shaping
for seed in 42 43 44 45 46; do
    python -X utf8 main.py --seed $seed --modes cg --rollouts 800
done
```

## Environment

- Python 3.10+
- PyTorch ≥ 2.0
- GridWorld: 3×3 grid, 5×5 observation window (full visibility), 5 actions (↑↓←→ + pickup)
- Dense reward: α·Δdist − γ + β·pickup (α=0.10, β=1.00, γ=0.005); wrong pickup terminates episode

## Structure

```
sdr_structural_retention_test/   # Table 1: SDR-L1 structural invariants
sdr_agent_v2_concept_discovery/  # Table 2: SDR-L3 concept composition
sdr_agent_v3_reward_shaping/     # Table 3: SDR-L4 agent binding
results/                         # Raw numbers (CSV)
config.py                        # Shared set_seed() utility
requirements.txt
```
