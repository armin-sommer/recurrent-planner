# Is the n100 dense core doing decision-time planning? (tick-resolved causal probes + adversarial check)

_Run 2026-06-25 on the 300M n100 fixed-4 checkpoint (`checkpoints/n100_fixed4/cp_299996160`). Probes:
`experiments/interp/{value_ticks,plan_onset,plan_front}.py`; logs next to this file. An independent
5-skeptic adversarial verification was run over the probe code + numbers; its verdict is folded in below._

## Question

Does the thinking loop perform **decision-time planning by inward value propagation** — pulling
information from cells one-more-graph-hop-out each tick, refining a value estimate, so that a change `d`
hops from the agent only alters the decision after ~`d` thinking steps (a marching front)? Or is the
graph structure realized some other way?

## Answer (calibrated)

**No marching front.** The discriminating test (`plan_front`) refutes the step-by-step inward-propagation /
"~1 hop per tick" picture. The dense core's influence reaches every distance **near-simultaneously**; what
is graph-structured is the **magnitude** of the response, not its **timing**. This is consistent with the
theory paper's own prediction that a *dense* core propagates globally each tick with **no** spatial
wavefront (the wavefront is the signature of *local/convolutional* routing). What survives is real and
publishable: graph-respecting, **decision-relevant**, **global** value-shaping per tick.

## The three probes

### 1. `value_ticks` — does the value refine over ticks? (mostly amortized)
Fixed probe fit at the trained depth, applied to every tick; model's own scalar critic value.
- `corr(V_t, V_settled)` across boards: **0.66 → 0.89 → 0.96 → 1.0** over ticks 1→4, then **drifts** (→0.91)
  past the trained depth. ~⅔ of the value ranking is set by tick 1; it finishes settling by the trained
  depth (4) and is OOD beyond it.
- The per-cell value **field** probe does **not** generalize across boards (held-out R² ≈ **−2.07**), so
  "value" here is the scalar critic ranking, not a validated propagating field.
- **Read: the value is amortized (set early, settles by trained depth), not iteratively propagated.** (This
  corrected an earlier over-claim in both directions — it is neither "fixed from tick 0" nor "refined by a
  DP loop"; it is amortized with a short settling tail.)

### 2. `plan_onset` — action vs (perturbation distance × tick), on-minus-off control
Perturb a path-lengthening wall `d` hops from the agent (≥4px away); measure per tick whether the action
differs from baseline, minus a distance-matched cosmetic (off-path) control. (768 boards, K=12.)
- Latent arrival at the agent staggers weakly with `d` (onset 2→4 ticks over d=4→7).
- Decision-relevant action change is individually significant at **d=5 (z=2.9)** and **d=6 (z=3.1)**,
  p<0.005; off-path control flat.
- **But the "staggered action onset ~1 tick/hop" headline did not survive review:** it rested on 3 distance
  points; the early anchor (d=4 → tick 1) is forced by the onset metric and sits at the conv-RF edge; the
  "~6-hop horizon" is an underpowered null (d≥7: |z|<1.2, n=48–91, and the d7 cutoff hinges on an arbitrary
  0.04 gate it misses by <1 SE). Reported as **suggestive, not a fitted slope.**

### 3. `plan_front` — the discriminating test (front vs global), the decisive result
Perturb a path-lengthening wall at a **far** on-path source (Chebyshev ≥4 from agent); measure `‖Δh‖` at
each intermediate on-path cell, binned by hops-from-source `j`, **excluding** cells within Chebyshev 4 of
the source (no conv leakage), arrival anchored **within the trained depth**, bootstrap CIs, off-path
cosmetic control. (768 boards → ~180 valid, K=8.)
- **Arrival tick is flat in distance: slope +0.00 ticks/hop, 95% CI [−0.03, +0.13].** Cells at j=4 and j=12
  both begin rising at ~tick 2. A front would give a rising arrival; this is the **global** signature.
- **Magnitude decays with graph distance** (settled `‖Δh‖`: j=4 → 0.52, j=12 → 0.32) and is 0 through walls
  (E4) — the graph lives in the kernel weights.
- **Decision-relevance is in the magnitude:** agent-cell `‖Δh‖` is **2×** larger for the on-path
  (path-lengthening) source than the cosmetic off-path source (0.35 vs 0.18), with the **same onset tick**.
- Verdict: **inward propagation front NOT present** → rules out the slow-marching-search reading and the
  "generic settling" alternative is the better description of the *timing* (global), while *magnitude* is
  graph-structured and decision-relevant.

## Adversarial verification (5 skeptics + synthesis)

Verdict: **supported with caveats.** Safe to claim / must drop:

**Safe:** decision-relevant, graph-respecting information reaches the agent's decision latent within the
trained depth; the influence is recurrent (≥4px, off-path control flat, recompute validated 1.8e-7),
graph-structured in magnitude, 2× stronger for decision-relevant than cosmetic changes; corroborated by
E4/E5/E10 and binding 0.78–0.85.

**Dropped / not supported:** (a) a clean ~1 tick/hop **action** stagger (3 points, artifact-anchored); (b)
a demonstrated **~6-hop horizon** (underpowered null); (c) **iterated inward value propagation** ("value
refines as info arrives" — the field probe doesn't generalize, value is amortized); (d) the far/staggered
**action** response as an operating-point property (it's read OOD, past the trained depth, from an
un-converged loop). One code note: the eu≥4px gate is looser than the 7px Chebyshev conv RF; `plan_front`
fixes this with a Chebyshev gate.

## Bottom line for the writeup

The dense n100 core realizes its graph structure as **global, graph-weighted, decision-relevant value
shaping each tick**, not as an inward-marching decision-time search. "Planning content" (binding + a
graph-respecting, wall-blocked, goal-anchored kernel + causal decision-relevant influence on the agent) is
present; the *spatial-temporal mechanism* is global refinement, exactly as predicted for a dense
(non-local) relational core. The marching-wavefront picture belongs to the local/convolutional instance.

## Reproduce
```sh
CP=checkpoints/n100_fixed4/cp_299996160
python -m experiments.interp.value_ticks --ckpt $CP --boards 256 --ticks 8      # value_ticks.log
python -m experiments.interp.plan_onset  --ckpt $CP --boards 768 --dmax 9 --ticks 12 --rf_px 4   # plan_onset.log
python -m experiments.interp.plan_front  --ckpt $CP --boards 768 --ticks 8 --rf 4 # plan_front.log
```
