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
publishable: graph-respecting, **decision-relevant**, **global** shaping of the agent's **latent and
action** per tick.

> **Latent vs value (important):** `plan_front` and `plan_onset` measure the **latent** `‖Δh‖` and the
> **action** (actor argmax) — they do **not** decode the critic, so they say nothing directly about the
> *value function*. The value channel is probed by a **separate** suite (see "What measures what" below);
> do not read `plan_front`'s `‖Δh‖` as "value." The earlier "value-shaping/propagation" phrasing here was a
> conflation and is corrected throughout.

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

## Which is localized & which refines — ACTION vs VALUE (the disambiguation)

`plan_front`/`plan_onset` only show the *latent* `‖Δh‖` moves; they don't say whether the cell carries the
**action** or the **value**, nor which one refines over ticks. The per-cell decode probes (run on n100)
settle it:

| quantity | localized per-cell? | changes over ticks? |
|---|---|---|
| **action** (per-node greedy dir) | **YES — decode 0.50** (chance 0.25), 66% goalward (`e11`) | **NO** — flat 0.53→0.50; every distance band onsets at tick 1, no frontier (`e12`) |
| **value** (per-node) | **weakly** — `e1` R²≈0.22; `value_ticks` held-out field R²≈**−2.07** (doesn't generalize) | **NO** (per-cell) — flat (`e1`) |
| **executed action a₀** (global readout) | n/a (global) | **YES — sharpens 0.59→0.99** over ticks 1→4 (`planq`) |
| **scalar value** (global critic) | n/a (global) | **YES — settles** corr→1.0 by tick 4, then drifts (`value_ticks`) |
| **multi-step plan a₁..a₅** | — | **NO** — at chance, not stored/refined (`planq`) |

**Conclusion (with a caveat the adversarial panel caught).** The **per-cell action field** is robustly
localized (0.50 / 66% goalward) and **flat over ticks**; the **per-cell value field doesn't transfer across
boards** — but that probe regresses the *agent→target navigation geodesic* (γ^BFS-dist), a **proxy** for the
box-push task value, and its target is the same field the action probe takes the argmax of, so "value not
localized" is a **probe/target artifact, not a property of the representation**. What is solid: **neither
per-cell field refines over ticks** (both amortized), and what refines is the **global readout** — `a₀`
sharpens 0.59→0.99 and the scalar value settles by trained depth. Crucially, the *global* value is **not**
weak: re-run on n100 it is Bellman-consistent and greedy-read (next section). So `plan_front`'s
`‖Δh‖`-over-ticks = the **readout integrating distal info**, riding on a genuine (amortized) value — not a
value-free action statistic, and not a per-cell field marching.

## Value channel re-run on n100 (closes the "vardepth-only" gap)

The value pillars were previously only on the vardepth run; re-run on `cp_299996160` they **replicate**, so
the value is real and central on this core too:

| probe | result on n100 | reading |
|---|---|---|
| `bellman` | optimality residual/std = [0.15, 0.17, 0.19, 0.18] over successors 0–3 | converged value is a **Bellman fixed point** (multi-step consistent) |
| `lookahead` | policy = argmaxₐ[r+γV(s′)] over own critic at **0.44** (chance 0.25), rises with thinking | action **is greedy over its own value** |
| `e9` | out-of-view path-block wall: ΔV **−1.17σ** vs −0.34σ off-path (95% drop), **builds over ticks** (−0.19→−1.17) | value **causally adopts new physics**, integrated over ticks |
| `e6b` | goal-move corr +0.39, box-move +0.43; \|dV\| box 1.43 > goal 0.86 | value **tracks task**, box-push-aware |
| `e2` | reward-move propagates **2.4×** a transition-change | **policy-evaluation** (goal baked into the field), not successor-rep |

So on n100 the value is amortized **and** Bellman-consistent, policy-evaluation-like, causally
physics/task-sensitive, and greedy-read — the value half of the claim is now n100-established, not transferred.

## What measures what (latent vs action vs value)

- **Latent** (`‖Δh‖` of the hidden, not decoded into anything): `plan_front`, `plan_onset`, `e10`, `wall`,
  `perturb`. — tells us a change *reaches* a cell, not *what* it encodes.
- **Action** (actor-head argmax): `plan_onset`, `e10`, `e13`, `lookahead`, the thinking-curve. — the decision.
- **Value** (critic head / decoded value field): `value_ticks` (scalar V over ticks → **amortized**, settles
  by trained depth then drifts), `e1` (probed value field), `e6`/`e6b` (V shifts in the resolvent-predicted
  direction under goal/box moves), `e9` (a path-blocking wall lowers V), `bellman` (V is Bellman-consistent),
  `lookahead` (policy ≈ 1-step lookahead over its own V), `e2` (eval vs successor-rep). — these, **not**
  `plan_front`, are what license any statement about the value.

## Bottom line for the writeup — decision-time planning vs DP, resolved

The honest resolution is **amortized value-based GPI with decision-time information integration** — three
separable claims, each scoped:

1. **No in-loop value-field DP** (n100-supported): the operator is stationary (no max), the loop copies its
   value forward rather than iterating it (own-op coeff +0.02 vs +0.98 identity), and the per-cell value
   field does not refine over ticks. There is no per-tick Bellman-backup of a value field in the loop.
2. **There IS a genuine value function** (n100-supported, this session): Bellman-consistent (residual
   0.15–0.19), policy-evaluation-like (goal baked in, e2 2.4×), causally physics/task-sensitive (e9/e6b),
   and the policy is **greedy over it** (lookahead 0.44). It is **amortized** — compiled into the weights /
   settled by trained depth — not absent.
3. **The loop's per-tick job is decision-time information integration**: it folds distal, out-of-view,
   graph-respecting, decision-relevant structure into a settling **global readout** (the value response to a
   hidden wall builds over ticks, e9 −0.19→−1.17; the action readout sharpens, planq 0.59→0.99). This is
   causal (e10: an out-of-view wall flips the action) — content a memoryless policy cannot produce.

So the clean dichotomy "decision-time planning **instead of** value/DP" is **not** the right framing: it is
value-based. Correct statement — **the evaluation is amortized into the weights (not re-iterated per tick),
the loop integrates decision-relevant information into a settling readout at decision time, and the action is
the greedy improvement over that amortized, Bellman-consistent value** (one step of GPI, with the loop on the
integration/improvement side). "Not DP" holds only as "no in-loop value-field iteration"; "action not value"
is **wrong** — the value is the substrate the decision rides on. Spatially it is global refinement, not a
wavefront (the wavefront is the local/conv instance). [theory_empirical.tex describes the *vardepth* N=100
run; its 0.44/0.41/Bellman numbers are that run's — the fixed-4 numbers here (0.50/0.44/0.15–0.19) corroborate
the same picture on a second core.]

## Reproduce
```sh
CP=checkpoints/n100_fixed4/cp_299996160
python -m experiments.interp.value_ticks --ckpt $CP --boards 256 --ticks 8      # value_ticks.log
python -m experiments.interp.plan_onset  --ckpt $CP --boards 768 --dmax 9 --ticks 12 --rf_px 4   # plan_onset.log
python -m experiments.interp.plan_front  --ckpt $CP --boards 768 --ticks 8 --rf 4 # plan_front.log
# action localized? + value channel (the disambiguation + value re-run on n100):
python -m experiments.interp.e11 --ckpt $CP --boards 192 --ticks 8     # e11.log  (per-cell action)
python -m experiments.interp.e12 --ckpt $CP --boards 192 --ticks 8     # e12.log  (action frontier)
python -m experiments.interp.planq --ckpt $CP --boards 512 --horizon 6 # planq.log (multi-step plan)
python -m experiments.interp.bellman   --ckpt $CP --boards 256 --depth 4   # bellman.log
python -m experiments.interp.lookahead --ckpt $CP --boards 256             # lookahead.log
python -m experiments.interp.e9  --ckpt $CP --boards 200 --ticks 12        # e9.log
python -m experiments.interp.e6b --ckpt $CP --boards 200 --ticks 12        # e6b.log
python -m experiments.interp.e2  --ckpt $CP --boards 256                   # e2.log
```
