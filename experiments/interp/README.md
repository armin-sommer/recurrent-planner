# Interpretability experiments — recurrent planner (D=3 entmax core)

Probes behind the claim that **planning emerges as amortized policy evaluation (value propagation)** in a
trained Sokoban agent. Full writeup + figures:
[`../../writeup/planning_emergence.tex`](../../writeup/planning_emergence.tex) (build with `tectonic`; the PDF
is gitignored — rebuild locally).

The experiment code lives here (`experiments/interp/`); the weights it runs on are at
[`../../checkpoints/`](../../checkpoints); the older behavioral thinking-curve sweep is in
[`../../results/`](../../results).

## The model

A state-indexed attention-LSTM (config `cleanba.config:sokoban_drc_attn_vardepth_entmax_d3`): each latent is
the `C=32` vector at a board square over `S=10×10` tokens; per "thinking" tick every cell attends over the
whole board (dense, **no** locality mask; 4 heads; sparse 1.5-entmax normalizer), aggregates neighbour
values (a convex average — no max), and applies an LSTM gate. Depth `D=3`, up to `K=6` ticks; trained with
IMPALA on Boxoban unfiltered-train (variable thinking depth `d~U{1..6}`, `γ=0.97`).

## Reproduce

1. **Environment:** `bash ../../build_env_pod2.sh` (or `make local-install` from the repo root) — py3.10 venv
   (uv), jax cuda12, gym-sokoban submodule. Needs the Boxoban levels (see the top-level
   [`../../README.md`](../../README.md)) and a Sokoban cache.
2. **Checkpoints:** the full 18-checkpoint training ladder (≈2M→300M steps) is at
   [`../../checkpoints/`](../../checkpoints) as `cp_001996800/ … cp_299996160/` (each `model` + `cfg.json`,
   all md5-distinct). The **300M final** is
   [`../../checkpoints/cp_299996160/`](../../checkpoints/cp_299996160) and the **140M** is
   [`../../checkpoints/cp_139991040/`](../../checkpoints/cp_139991040). Load via
   `cleanba.cleanba_impala.load_train_state(Path(cp_dir), env_cfg=…)`. (Probes below were run on the 300M.)
3. **Run a probe** from the repo root (GPU; cap memory with
   `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`):
   ```sh
   python -m experiments.interp.<name> --ckpt <cp_dir> [--boards N] [--ticks K]
   ```
   Each script loads the checkpoint, **recomputes the core crash-free** (`recompute_d3` — the model's own
   forward trips a tracer error on the offset-tied `rel_bias` gather under `nn.scan`), and prints its result.
   Validate the recompute on CPU: `python -m experiments.interp.planning --self-test` (matches the model's
   hidden state to `1.8e-7`). Each script's module docstring has its exact command and what it measures.

## Experiments (by thesis step)

**Step 1 — binding (the latents are the states)**
- `planning.py` — linear probe `h(s)→tile type` + recovery-of-`N` (routing) + per-tick plan probes.
  `--self-test` validates the recompute. (Its Q1 prints *raw* 5-class accuracy, chance ≈0.69.)
- `binding_balanced.py` — the headline **per-object balanced** binding accuracy (chance 0.50), the metric the
  writeup reports: wall/box/target/agent `= 0.74/0.73/0.72/0.78` at the 300M checkpoint, flat across ticks —
  every object class, including the rare movable ones, is decodable from a single cell.
  (`bind_bal.py`, `bind_bal2.py` are earlier precursor drafts of this probe, kept for provenance.)

**Step 2 — the learned graph (connectivity along `N`)**
- `e4.py` — the learned kernel: attention mass vs **graph** distance (geometric decay `ρ≈0.66`,
  ~0 mass through walls, `+0.07` anchor on the goal).
- `wall.py` — through-walls (causal): blocked- vs open-cell influence at matched pixel distance.
- `perturb.py`, `topo.py` — causal influence radius vs ticks; graph- vs euclidean-distance.

**Step 3 — planning by value propagation**
- `e1.py` — per-tick operator: attention stationarity vs sharpening + contraction rate (value
  iteration vs evaluation vs null).
- `box_prop_attn.py` — **the decisive amortization test**: decode the per-cell **box-push** value field at
  every tick and regress it on the model's own operator applied to the previous tick. The own-operator
  coefficient is nonzero only on tick 1 (`0.23→0.01`) and `‖Δv‖→0` thereafter — the value field is computed
  amortized (set by tick 1), **not** re-propagated over thinking ticks. (Imports `_propcommon.py`.)
- `e5.py` — reach vs ticks: does propagation **compound** (are the pulled-from cells themselves updated)?
- `bellman.py` — Bellman optimality residual at the agent + greedy successors (recursion depth).
- `lookahead.py` — policy vs one-step lookahead over the model's **own** value, by thinking depth.
- `planq.py` — is a multi-step action plan decodable per tick? (no — only the immediate action).
- `e2.py` — reward-relabel invariance: policy-evaluation vs successor-representation.
- `e6.py`, `e6b.py` — goal/box interventions: does the value shift in the
  resolvent-predicted direction? (`e6b`: far-side relocation, readout value + magnitude, box > goal).
- `e7.py`, `e7b.py` — does the attention operator re-route under edits? (local `‖ΔA‖`
  by distance + neighbour mass; it does **not**).
- `e8.py` — value-projection norm by tile + floor→wall flip (are wall cells "dead"? no).
- `e9.py` — does the value field adopt new physics? (a path-blocking wall lowers `V`).
- `e10.py` — does a wall's value-change reach the agent's square + re-plan its action?
- `e11.py` — the **decodable plan as a policy field**: per-node greedy action `h(s)→dir` is
  decodable (~0.44, chance 0.25), ~61% of nodes point goalward, and the decoded action **equals the value
  gradient** (move toward the higher-value neighbour) — the plan is the value's gradient, present wherever
  value is, not a stored action sequence (contrast `planq`, which finds the executed trajectory is not stored).
- `e12.py` — does the *per-cell decoded* field **expand outward** as a spatial frontier over ticks?
  **No** — per-node decodability is flat across all 8 ticks at every distance band (onset simultaneous, not
  staggered near→far; 192 and 512 boards). This is the **expected** signature for a *dense* core (every cell
  attends globally each tick → no ring-by-ring wavefront; ticks **sharpen** value, they don't spread it). Note
  this measures *decodability of the field refit per tick* — **not** the model's decision; for that see `e13`.
- `e13.py` — **the model's own decision over ticks** (the direct value→action test). Applying the
  actual actor/critic head to each tick's hidden state: thinking **changes** the decision on **~35%** of
  boards (tick-1 action ≠ settled) and **improves** it toward goalward, **+8.6pp** overall and **most on the
  farthest boards** (d13+: **+0.15**) — exactly the planning prediction (far goals need more propagation). The
  state-value contracts (`|ΔV|` 0.72→0.23) and the decision margin sharpens (1.6→3.3). So value propagation
  **does** drive action selection — reconciling `e12` (flat field *decodability*, not the decision) with `e10`
  (causal: injected value-change flips the agent's move) and the behavioral thinking-curve.
- `e3.py` — superposition / no-kinks check (behavioral confirmation of no-max).
- [`../../results/thinking_curve_vardepth.py`](../../results/thinking_curve_vardepth.py) — solve rate vs inner
  thinking depth (the `n_active` sweep; behavioral, lives with the sweep in `results/`).

**Cross-check on the convolutional planner (DRC(3,3), masked/local routing).** Does the prior-work
ConvLSTM (Guez/Bush/Taufeeque) also show *cellwise* GPI? Loads the pretrained DRC(3,3) 2B checkpoint
(`AlignmentResearch/learned-planner :: drc33/bkynosqi/cp_2002944000`, via `eval_pretrained.py`) and probes
the per-cell ConvLSTM hidden over thinking ticks (extracted by a manual `apply_cells_once` loop).
- `drc_box_gpi.py` — **box-centric** cellwise GPI (the faithful Sokoban quantity): per-cell box-push value
  (BFS on the push graph) + optimal push-direction. Result on the 2B DRC: push-direction decodable **0.68**
  (chance 0.25), the decoded value's **gradient is the optimal push 0.75**, policy ≈ value-gradient **0.55** —
  the cellwise GPI fixed-point — with far cells refining over ticks (d6–8: 0.46→0.69). Mostly amortized by
  tick 1.
- `drc_causal.py` — causal box-value reformation (E9/E10 analog): an on-path wall (lengthens the box-push
  path) moves the decoded cellwise value **~2.5–5× more** than a cosmetic wall, in the predicted direction —
  physics-sensitive, but front-loaded (largest at tick 1, not building over ticks).
- `drc_gpi.py` — the agent-navigation baseline (weak: nav-to-target is the wrong value for a box-pushing
  task; value R² 0.14, no frontier). Kept to show why the box-centric quantity is the right one.
- `box_prop_drc.py` — the DRC analog of `box_prop_attn.py`: the convolutional planner's per-cell box-push
  value is likewise amortized (front-loaded, `‖Δv‖→0` over ticks). (Imports `_propcommon.py`.)
- `drc_propagation.py` — earlier DRC propagation probe (per-cell value reach vs ticks); superseded by
  `drc_box_gpi.py`/`drc_causal.py`, kept for provenance.

**Shared helpers (imported by the probes):** `plan.py` (BFS distance fields, ridge/linear probes,
`analyse_plan`), `slots.py` (`decode_tiles`, slot-core utilities), `search.py`,
`_propcommon.py` (box-push BFS `bfs_box`, masked-mean, ridge `fit`/`pred`/`reg` — used by the `box_prop_*`
amortization tests).

**Separate core:** `attn.py` is the earlier slot/attention-core recovery-of-`N` probe (a different core; not
part of the D=3 suite above), kept here for provenance.

**Older behavioral sweep** (masked-vs-dense thinking-curve at 200M): see
[`../../results/RESULTS.md`](../../results/RESULTS.md).
