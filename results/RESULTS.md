# Results — recurrent-planner

Post-hoc **thinking-step sweep** (extra thinking ticks 0→12) on the 200M-step checkpoints,
evaluated on **valid_medium** (held-out, medium difficulty — the generalization benchmark) and
**train_unfiltered** (training distribution). 1000 levels/tick. `local_max` is evaluated at its
**120M peak** (its 200M run oscillated); `attn_masked` excluded (collapsed ~50M).

Data: [`data/thinking.csv`](data/thinking.csv) · Figures: [`figures/`](figures) · Pipeline: `parse_pkls.py`, `plot.py`.
Raw artifacts (gitignored): `checkpoints/all_arms/all_ckpts.tar` (274 ckpts, all arms), `data/eval_pkls.tar` (22 metric pkls).

## Cellwise matrix (mask × aggregation) — valid_medium success

| arm | valid @0 | @12 | thinking lift |
|---|---|---|---|
| local_mean | 0.163 | 0.167 | +0.004 |
| local_sum | 0.151 | 0.172 | +0.021 |
| local_max (peak) | 0.130 | 0.133 | +0.003 |
| dense_mean | 0.142 | 0.168 | +0.026 |
| dense_max | 0.105 | 0.114 | +0.009 |
| dense_sum | **0.000** | 0.000 | — (diverged) |

## Attention arms — valid_medium success

| arm | valid @0 | @12 | thinking lift |
|---|---|---|---|
| attn_softmax (masked) | **0.458** | 0.507 | +0.049 |
| attn_plain (masked) | 0.377 | 0.424 | +0.047 |
| attn_vn (masked, von Neumann) | 0.319 | 0.368 | +0.049 |
| attn_dense (dense) | 0.215 | 0.228 | +0.013 |
| attn_shallow (depth-1) | 0.183 | 0.247 | +0.064 |

## Read (honest)

1. **Attention ≫ cellwise here.** Masked attention reaches ~38–46% valid vs the cellwise arms' ~13–16%. The cellwise message-passing core is just a weak planner at 200M.
2. **The mask claim holds *cleanly in attention*.** The matched contrast `attn_plain` (masked) vs `attn_dense` (dense) — identical except the locality mask — gives **masked 0.377 vs dense 0.215 (+16 pp)**, and masked's thinking-lift is **+4.7 pp vs dense's +1.3 pp (~3.6×)**. So the mask delivers *both* more solving *and* a larger test-time-thinking benefit — the local-vs-dense thinking-scaling signature.
3. **In the cellwise core the signal is weak/noisy.** local ≥ dense modestly on solve-rate, and decisively for sum (`dense_sum` diverges — unnormalized sum over ~100 cells blows up — while `local_sum` is fine). But the thinking lifts are ~±1–3 pp (≈1.5σ at 1000 levels) and not cleanly local-favoring. The cellwise arms are too weak to resolve the thinking-scaling claim behaviourally.
4. **Large train→valid gap** (~80% → ~15% cellwise): agents fit the training distribution, generalize modestly to held-out medium.

**Takeaway:** the masked-vs-dense thesis is supported behaviourally **in the (strong) attention instance**, not in the (weak) cellwise one. The decisive *mechanism* test — does masked compute a localized, propagating plan/value frontier that dense lacks — is the probe experiments (decodability → frontier-per-tick → causal patching), next.

## Mechanism probes (done)
The decisive mechanism tests — does the core compute a localized, propagating value/plan frontier? — are in
the **D=3 entmax interpretability suite**: see
[`../experiments/interp/README.md`](../experiments/interp/README.md). Headline: planning emerges as
**amortized policy evaluation (value propagation)** along a fixed, transition-respecting attention graph.
