# n100 dense-attention core — interp experiments & takeaways

_Run + analyzed 2026-06-24/25. Raw probe logs live next to this file (`*.log`); the
checkpoints are committed under [`checkpoints/n100_fixed4/`](../../checkpoints/n100_fixed4)._

## What this run is and why it exists

This is the **n100 anchor** for the cell-count study, and a **fixed-depth replication** of the
planning chain proved on the variable-depth thesis run.

- **Config:** `cleanba.config:sokoban_drc_attn_d3_fixed4_nomask_mb4` — state-indexed attention-LSTM
  core, **dense** attention (`use_attention_mask=False`), **1.5-entmax** normalizer
  (`attn_norm="entmax15"`, convex-average aggregation, no max), depth **D=3**, **4 heads**, **C=32**,
  **N=100** cells (the 10×10 board). The one architectural difference from the thesis run
  (`sokoban_drc_attn_vardepth_entmax_d3`) is the thinking-tick schedule: **fixed 4 ticks** here vs.
  variable `d∼U{1..6}` there. Recipe matched to the slot cell-count sweep (mb4, 4-GPU 2a+2l).
- **Training:** IMPALA, Boxoban unfiltered-train, **300M steps, FINISHED**, γ=0.97. Final
  `avg_episode_returns ≈ 8.0–8.3` (took off cleanly past the usual −7 floor).
- **Checkpoint analyzed:** `cp_299996160` (300M final).
- **Role in the paper:** dense attention with **no locality mask** must *discover* binding (σ) and the
  transition graph (𝒩). The slot core (content-addressed) binds but broadcasts (no 𝒩 recovery); the
  dense core recovers 𝒩. This run is the dense, N=100 point of that contrast and a robustness check
  that the planning story is not an artifact of variable-depth training.

## Checkpoints stored in the repo

| run | location | ladder |
|---|---|---|
| vardepth-K6 (thesis run) | `checkpoints/cp_*` | full 18-ckpt ladder, 2M→300M (already committed) |
| n100 fixed-4 (this run) | `checkpoints/n100_fixed4/cp_*` | 6 points: 2M / 20M / 80M / 160M / 260M / 300M |

The n100 subset (2/20/80/160/260/300M) is the spread for an emergence-over-training sweep plus the
300M final. The full 18-ckpt n100 ladder remains on the training node if more points are ever needed.

## Reproduce

```sh
CP=checkpoints/n100_fixed4/cp_299996160        # or the on-node path
ENV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 WANDB_MODE=offline"
$ENV python -m experiments.interp.binding_balanced --ckpt $CP --boards 256   # binding.log
$ENV python -m experiments.interp.e4   --ckpt $CP --boards 128               # e4.log
$ENV python -m experiments.interp.wall --ckpt $CP --boards 256               # wall.log
$ENV python -m results.thinking_curve_vardepth --ckpt $CP --depths 0,1,2,3,4,5,6,8   # thinking.log
$ENV python -m experiments.interp.e1  --ckpt $CP --boards 128 --ticks 12     # e1.log
$ENV python -m experiments.interp.e5  --ckpt $CP --boards 256 --dmax 9 --ticks 12    # e5.log
$ENV python -m experiments.interp.e10 --ckpt $CP --boards 200 --ticks 12     # e10.log
$ENV python -m experiments.interp.e13 --ckpt $CP --boards 512 --ticks 8      # e13.log
```

Each probe recomputes the D=3 entmax stack crash-free (`recompute_d3`); see
[`experiments/interp/README.md`](../../experiments/interp/README.md).

## Headline: the full chain replicates on the fixed-4 core

| step | measurement | **n100 fixed-4** | vardepth-K6 (thesis) |
|---|---|---|---|
| thinking helps | solve rate, ticks 1→4 | **0.199 → 0.316**, then flat | 0.258 → 0.336 peak, droops 0.319 |
| 1 — binding | decode wall/box/target/agent (chance 0.50) | **0.81 / 0.78 / 0.80 / 0.85** | 0.74 / 0.73 / 0.72 / 0.78 |
| 2 — recover 𝒩 | ρ_graph; mass >1 hop; thru-wall; goal anchor | **0.57; 0.40; 0.00; +0.08** | 0.66; 0.34; 0.00; +0.07 |
| 2 — through-walls | blocked/open influence; partial corr(·\|euclid) | **0.31; −0.34** | 0.38; −0.32 |
| 3a — operator | cos(Aₜ,Aₜ₋₁); contraction ρ; max in loop? | **→0.998; 0.86; no** | →0.995; 0.89; no |
| 3a — reach | propagation horizon by tick | **1→4→8→9** (~3.4 hops/tick) | 0→2→8→9 (~3 hops/tick) |
| 3c — re-plan | ‖Δh(agent)‖ on/off-path; action change on/off | **2.7×; 5.5×** | 2.2×; 3× |
| 3 — decision | optimality over ticks; % decisions changed | **0.51→0.59 (+8.0pp); 33%** | 0.52→0.605 (+8.6pp); 35% |

**Verdict:** dense attention with no mask, trained at a fixed thinking depth, still binds cells to
states and *learns* the transition graph (geometric decay in graph distance, exactly zero leakage
through walls, a small goal anchor), and the thinking loop is the same amortized, graph-respecting
policy-evaluation-plus-head-lookahead. Binding is, if anything, sharper than the variable-depth run.

## Per-experiment detail

**Thinking curve (`thinking.log`) — does thinking improve performance? YES.**
Whole episode run at each inner depth. valid-medium: d1 0.199 → d4 **0.316**, then exactly flat
(d5=d6=d8=0.316). train-unfiltered: 0.810 → **0.925**, flat from d4. d0 (no recurrence) = 0.0,
degenerate. The plateau lands precisely at the **trained depth of 4** — the fixed-4 core uses its full
budget and neither degrades nor keeps gaining past it (contrast the vardepth run, which peaks at d4 and
slightly droops by d6).

**Step 1 — binding (`binding.log`).** Per-object balanced linear decode from a single cell, settled
tick, chance 0.50: wall **0.806**, box **0.780**, target **0.795**, agent **0.854**. Flat across the 4
ticks (e.g. agent [0.92 0.83 0.86 0.85]) → state content is written by the encoder and carried by the
loop, not built in it. The latents are the states.

**Step 2 — recovery of 𝒩 (`e4.log`).** Mean attention mass by graph-distance shell:
`[0.018, 0.217, 0.213, 0.054, …]`; geometric decay **ρ_graph = 0.568** (reach ~2.3 hops); **0.395** of
mass is beyond the immediate neighbour (real propagation, not a 1-ring blur); **0.000** mass to
wall-unreachable keys (no through-wall leakage); **+0.080** anchor on the target/goal cell (the reward
source). Mild goal-ward asymmetry (toward/away 1.13). The routing is graph-shaped, not pixel-shaped.

**Step 2 — through-walls, causal (`wall.log`).** At matched pixel distance (band 3.5–7), a cell
separated from the agent by a wall shifts the agent latent **0.149** vs **0.484** for an open cell —
ratio **0.31**; partial corr(influence, graph-dist | euclid) = **−0.34**. Information routes around
walls, along 𝒩, not through Euclidean space.

**Step 3a — operator (`e1.log`).** The attention operator is **stationary**: top-1 entmax mass flat
~0.157 over ~62 active keys, cos(Aₜ,Aₜ₋₁) rises 0.961 → **0.998** (no sharpening toward an argmax).
Aggregation is a convex average (no max). Hidden contracts at **ρ = 0.857** (effective horizon ~7
ticks ≪ the γ=0.97 task horizon ⇒ the long solve is amortized into the weights; the loop runs a few
refinement sweeps). Own-operator regression vₜ ≈ a·(Aₜ₋₁vₜ₋₁) + b·vₜ₋₁ settles to a≈+0.02, R²=0.94 —
a damped propagation coefficient consistent with c ← γ_eff·A·c + r_eff. So: **policy evaluation / SR
propagation, not in-loop value iteration.**

**Step 3a — reach compounds (`e5.log`).** Perturb a floor cell at graph-distance d from the agent;
the agent-cell shift arrives **later for larger d** (arrival-tick vs d slope +0.30, ~3.4 hops/tick).
The horizon grows **1→4→8→9** over ticks 1→4: cells 7–9 hops away still reach the agent, after 3–4
ticks. Because a single tick spans only ~3 hops, the pulled-from cells must themselves be updated each
tick ⇒ genuine **multi-step** propagation (iterative deepening), not a one-shot blur.

**Step 3c — re-plan (`e10.log`).** A path-blocking wall placed outside the agent's conv view (so any
effect arrives by recurrent propagation): the agent-cell shift builds to **0.497 on-path vs 0.185
off-path** (ratio 2.68), and the greedy action changes **0.183 on-path vs 0.033 off-path** at full
depth (5.5×; at the trained depth K=4: 0.083 vs 0.000). The re-formed value reaches the agent's own
cell and re-plans its move — causal value→action.

**Step 3 — decision over ticks (`e13.log`).** Reading the model's own actor/critic head at each tick:
the action **changes on 33.2%** of boards (tick 1 vs settled) and **gets more goalward**, overall
optimality 0.510 → **0.590** (+8.0pp, chance 0.25). The state-value contracts (|ΔV| 1.02 → 0.21) and
the decision margin sharpens (1.06 → 3.59). Value propagation drives action selection — the behavioural
counterpart of the thinking-curve gains.

## The one honest divergence — E13 distance stratification is reversed

On the **vardepth** run the thinking improvement was *largest on the farthest goals* (+0.15 at graph-
distance ≥13). On this **fixed-4** run it is the **opposite** — largest on near goals, smallest on far:

| band | tick 1 → settled | Δ |
|---|---|---|
| d1–3 | 0.46 → 0.58 | **+0.12** |
| d4–7 | 0.50 → 0.58 | +0.08 |
| d8–12 | 0.56 → 0.61 | +0.06 |
| d13+ | 0.58 → 0.61 | **+0.03** |

**Most likely cause (recipe, not a bug):** the loop here was trained at a **fixed 4 ticks**, so far
goals (>4 hops) need more propagation than the loop ever learned to use — the model banks the easy
near-goal gains and cannot deepen for the far ones. The vardepth model (d∼U{1..6}) was trained to
exploit depth and kept paying off on far boards. Two secondary factors: the far bands already start
higher at tick 1 (0.56–0.58, less headroom), and d13+ has only **n=33** boards (noisy). The overall
"thinking improves the decision" result is unaffected; only the *where* differs. This is a clean
illustration that **the horizon the thinking can reach is bounded by the trained depth** — consistent
with the amortized-evaluation reading. Cite E13's distance breakdown from the **vardepth** checkpoint,
not this one.

## Bottom line

The n100 dense-attention anchor is confirmed: binding (Step 1) and learned recovery of 𝒩 (Step 2,
ρ=0.57, 0 through walls) hold, and the loop runs amortized graph-respecting GPI (Step 3) with thinking
improving both solve rate (+11.7pp) and decision optimality (+8.0pp). The only deviation from the
thesis run is the depth-bounded reversal of E13's far-vs-near improvement, explained by fixed-4 training.
