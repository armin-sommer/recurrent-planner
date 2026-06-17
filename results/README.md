# Interpretability experiments — recurrent planner (D=3 entmax core)

Probes behind the claim that **planning emerges as amortized policy evaluation (value propagation)** in a
trained Sokoban agent. Full writeup + figures:
[`../writeup/planning_emergence.tex`](../writeup/planning_emergence.tex) (build with `tectonic`; the PDF is
gitignored — rebuild locally).

## The model

A state-indexed attention-LSTM (config `cleanba.config:sokoban_drc_attn_vardepth_entmax_d3`): each latent is
the `C=32` vector at a board square over `S=10×10` tokens; per "thinking" tick every cell attends over the
whole board (dense, **no** locality mask; 4 heads; sparse 1.5-entmax normalizer), aggregates neighbour
values (a convex average — no max), and applies an LSTM gate. Depth `D=3`, up to `K=6` ticks; trained with
IMPALA on Boxoban unfiltered-train (variable thinking depth `d~U{1..6}`, `γ=0.97`).

## Reproduce

1. **Environment:** `bash build_env_pod2.sh` (or `make local-install`) — py3.10 venv (uv), jax cuda12,
   gym-sokoban submodule. Needs the Boxoban levels (see the top-level `README.md`) and a Sokoban cache.
2. **Checkpoints:** the full 18-checkpoint training ladder (≈2M→300M steps) is included under
   [`checkpoints/`](.) as `cp_001996800/ … cp_299996160/` (each `model` + `cfg.json`, all md5-distinct). The
   **300M final** is [`checkpoints/cp_299996160/`](checkpoints/cp_299996160) and the **140M** is
   [`checkpoints/cp_139991040/`](checkpoints/cp_139991040). Load via
   `cleanba.cleanba_impala.load_train_state(Path(cp_dir), env_cfg=…)`. (Probes below were run on the 300M.)
3. **Run a probe** (GPU; cap memory with
   `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`):
   ```sh
   python -m results.interp_<name>_d3 --ckpt <cp_dir> [--boards N] [--ticks K]
   ```
   Each script loads the checkpoint, **recomputes the core crash-free** (`recompute_d3` — the model's own
   forward trips a tracer error on the offset-tied `rel_bias` gather under `nn.scan`), and prints its result.
   Validate the recompute on CPU: `python -m results.interp_planning_d3 --self-test` (matches the model's
   hidden state to `1.8e-7`). Each script's module docstring has its exact command and what it measures.

## Experiments (by thesis step)

**Step 1 — binding (the latents are the states)**
- `interp_planning_d3.py` — linear probe `h(s)→tile type` (per-object balanced accuracy) + recovery-of-`N`
  + per-tick plan probes. `--self-test` validates the recompute.

**Step 2 — the learned graph (connectivity along `N`)**
- `interp_e4_d3.py` — the learned kernel: attention mass vs **graph** distance (geometric decay `ρ≈0.66`,
  ~0 mass through walls, `+0.07` anchor on the goal).
- `interp_wall_d3.py` — through-walls (causal): blocked- vs open-cell influence at matched pixel distance.
- `interp_perturb_d3.py`, `interp_topo_d3.py` — causal influence radius vs ticks; graph- vs euclidean-distance.

**Step 3 — planning by value propagation**
- `interp_e1_d3.py` — per-tick operator: attention stationarity vs sharpening + contraction rate (value
  iteration vs evaluation vs null).
- `interp_e5_d3.py` — reach vs ticks: does propagation **compound** (are the pulled-from cells themselves updated)?
- `interp_bellman_d3.py` — Bellman optimality residual at the agent + greedy successors (recursion depth).
- `interp_lookahead_d3.py` — policy vs one-step lookahead over the model's **own** value, by thinking depth.
- `interp_planq_d3.py` — is a multi-step action plan decodable per tick? (no — only the immediate action).
- `interp_e2_d3.py` — reward-relabel invariance: policy-evaluation vs successor-representation.
- `interp_e6_d3.py`, `interp_e6b_d3.py` — goal/box interventions: does the value shift in the
  resolvent-predicted direction? (`e6b`: far-side relocation, readout value + magnitude, box > goal).
- `interp_e7_d3.py`, `interp_e7b_d3.py` — does the attention operator re-route under edits? (local `‖ΔA‖`
  by distance + neighbour mass; it does **not**).
- `interp_e8_d3.py` — value-projection norm by tile + floor→wall flip (are wall cells "dead"? no).
- `interp_e9_d3.py` — does the value field adopt new physics? (a path-blocking wall lowers `V`).
- `interp_e10_d3.py` — does a wall's value-change reach the agent's square + re-plan its action?
- `interp_e11_d3.py` — the **decodable plan as a policy field**: per-node greedy action `h(s)→dir` is
  decodable (~0.44, chance 0.25), ~61% of nodes point goalward, and the decoded action **equals the value
  gradient** (move toward the higher-value neighbour) — the plan is the value's gradient, present wherever
  value is, not a stored action sequence (contrast `interp_planq`, which finds the executed trajectory is not stored).
- `interp_e12_d3.py` — does the *per-cell decoded* field **expand outward** as a spatial frontier over ticks?
  **No** — per-node decodability is flat across all 8 ticks at every distance band (onset simultaneous, not
  staggered near→far; 192 and 512 boards). This is the **expected** signature for a *dense* core (every cell
  attends globally each tick → no ring-by-ring wavefront; ticks **sharpen** value, they don't spread it). Note
  this measures *decodability of the field refit per tick* — **not** the model's decision; for that see E13.
- `interp_e13_d3.py` — **the model's own decision over ticks** (the direct value→action test). Applying the
  actual actor/critic head to each tick's hidden state: thinking **changes** the decision on **~35%** of
  boards (tick-1 action ≠ settled) and **improves** it toward goalward, **+8.6pp** overall and **most on the
  farthest boards** (d13+: **+0.15**) — exactly the planning prediction (far goals need more propagation). The
  state-value contracts (`|ΔV|` 0.72→0.23) and the decision margin sharpens (1.6→3.3). So value propagation
  **does** drive action selection — reconciling E12 (flat field *decodability*, not the decision) with E10
  (causal: injected value-change flips the agent's move) and the behavioral thinking-curve.
- `interp_e3_d3.py` — superposition / no-kinks check (behavioral confirmation of no-max).
- `thinking_curve_vardepth.py` — solve rate vs inner thinking depth (the `n_active` sweep).

**Shared helpers (imported by the probes):** `interp_plan.py` (BFS distance fields, ridge/linear probes,
`analyse_plan`), `interp_slots.py` (`decode_tiles`, slot-core utilities), `interp_search_d3.py`.

**Older behavioral sweep** (masked-vs-dense thinking-curve at 200M): see [`RESULTS.md`](RESULTS.md).
