# Results — recurrent-planner

Post-hoc **thinking-step sweep** (extra thinking ticks 0→12) on the 200M-step checkpoints,
evaluated on:
- **valid_medium** — held-out validation, medium difficulty (the generalization benchmark; standard "Boxoban medium")
- **train_unfiltered** — the training distribution (success on levels from the training split)

1000 levels/tick. Data: [`data/thinking.csv`](data/thinking.csv) · Figures: [`figures/`](figures) · Pipeline: `parse_pkls.py` (node→CSV), `plot.py` (CSV→figures). `local_max` is evaluated at its **120M peak** (its 200M run oscillated/collapsed — see handoff). `attn_masked` excluded (collapsed at ~50M).

## Cellwise matrix (mask × aggregation) — success rate

| arm | train @0 | valid @0 | valid @12 | thinking lift (@12−@0) |
|---|---|---|---|---|
| local_mean | 0.824 | **0.163** | 0.167 | +0.004 |
| local_sum  | 0.819 | 0.151 | 0.172 | +0.021 |
| local_max (peak) | 0.773 | 0.130 | 0.133 | +0.003 |
| dense_mean | 0.774 | 0.142 | 0.168 | +0.026 |
| dense_max  | 0.726 | 0.105 | 0.114 | +0.009 |
| dense_sum  | 0.019 | 0.000 | 0.000 | — (diverged) |

(attn arms: valid eval pending; train @0 so far — `attn_softmax` 0.944, `attn_dense` 0.861. During training, masked attn reached ~41% valid, far above the cellwise arms — attention is a stronger planner here.)

## Read (honest, preliminary)

1. **Large train→valid generalization gap** (~80% train → ~10–17% valid). The agents largely fit the (easier, seen) training distribution; valid-medium (harder, held-out) is much lower.
2. **The mask helps on solve-rate — modestly for max/mean, decisively for sum.** local ≥ dense for every aggregation (mean .163 vs .142, max .130 vs .105), and **`dense_sum` totally fails (0%)**: the unnormalized sum over all ~100 cells blows up, while masked `local_sum` (8 neighbours) trains fine (.151). So the locality mask is load-bearing — most starkly for sum.
3. **The test-time thinking benefit is small and noisy** (~±1–3 pp at 1000 levels ≈ 1.5σ) and does **not** cleanly favour local — `dense_mean` has the *largest* valid lift (+0.026). So the strong behavioural signature "more thinking → much better, local ≫ dense" is **not** borne out at 200M.

**Caveat / why the mechanism still matters:** a faint *behavioural* thinking-lift does not settle the *mechanism*. Local arms could still compute via a localized, spatially-propagating plan/value frontier even if the net behavioural gain is small. That is exactly what the probe experiments test next (concept decodability → frontier propagation → causal patching). The behavioural result here just means the frontier, if present, isn't translating into a large success-vs-thinking slope at this training budget.

## Pending
- Attn arms valid eval (the 2nd, stronger non-conv instance).
- Mechanism probes: concept decodability, frontier-per-tick, causal patching (see `AGENT_HANDOFF.md`).
