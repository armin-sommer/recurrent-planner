# AGENT HANDOFF — recurrent-planner

_Single source of truth. Rewritten 2026-06-15. Read top to bottom._

---

## 0. THESIS (the only thing that matters)

Model-free RL on a **relational recurrent core** induces planning, demonstrated as a three-step chain
on Sokoban — and the structure must **emerge from RL**, not be architecturally imposed:

1. **Binding** — latent cells bind to environment states.
2. **Transition-respecting relations** — the learned routing between latent cells respects the
   environment's one-step transition graph $\mathcal{N}$ (information flows along $\mathcal{N}$).
3. **Planning** — the recurrent "thinking" loop implements planning as a **two-step generalized policy
   iteration**: (3a) amortized, graph-respecting **policy EVALUATION** — goal-directed value propagation
   along $\mathcal{N}$ — in the loop, plus (3b) a **one-step lookahead policy IMPROVEMENT** at the actor
   head. (NOT value iteration — no max in the loop; NOT successor representation — the goal is baked into
   the iterate.) **(3c)** Causally, the value field **tracks task (goal/box) and physics (wall) changes,
   reaches the agent, and re-plans** — via propagated value *content* through a **fixed** $\mathcal{N}$-kernel,
   not by re-routing. (Corrects the earlier "amortized value iteration" wording; see §2 Step 3.)

**Status: all three shown on the trained model (§2). The central result is that (1)–(2) are EMERGENT
— dense attention with NO locality mask discovers them.** Imposing a king-mask would hard-code
$\mathcal{N}$ and make the claim circular — **do NOT mask** (this was considered and rejected).

---

## 1. THE MODEL / RUN (the one and only run we analyze)

- **Config:** `cleanba.config:sokoban_drc_attn_vardepth_entmax_d3` — state-indexed attention-LSTM core,
  **dense** attention (`use_attention_mask=False`) with a **1.5-entmax** sparse normalizer
  (`attn_norm="entmax15"`), depth **D=3** (`n_recurrent`), up to **K=6** thinking ticks, 4 heads,
  C=32. **Variable thinking depth** d∼U{1..6} per rollout (learner replays same depth).
- **Training:** IMPALA, Boxoban unfiltered-train, **300M steps, FINISHED.** Online train solve ≈83%,
  `valid_medium` ≈31% (greedy full depth), `var_explained`=0.99, entropy 1.37→0.29.
- **Checkpoints (node):** `/workspace/runs/vd_d3_entmax/default/wandb/offline-run-20260613_081013-er45zk8y/local-files/cp_*`
  at 2/5/20/40/.../300M (every 20M). Final = `cp_299996160`. Backup of 140M at
  `/workspace/saved_ckpts/cp_139991040`.
- **Pulled local + committed:** `results/checkpoints/cp_139991040/` (140M, SHA-verified), commit
  `b641917`. Code committed in `2c9d074` (branch `recurrent-attention-core`, pushed).
- **Three training requirements (all verified):** (1) random depth; (2) gradients flow through exactly
  d weight-tied ticks — **bit-exact** (`tests/check_vardepth_grad.py`); (3) entmax sparsity is
  architectural, not a loss penalty (`tests/check_entmax.py`, FD-verified). `cleanba/entmax.py` =
  sparsemax + 1.5-entmax with closed-form `custom_vjp`.

---

## 2. WHAT WE'VE SHOWN (evidence per step)

### Step 1 — Binding ✓
Per-object **linear probe** $h(s)\to$ tile type (balanced accuracy, chance 0.5), 160M ckpt:
wall **0.77**, box **0.76**, target **0.70**, agent **0.74** — all well above chance incl. the 1–4%
rare objects. Flat over ticks (read from the input embedding, not built by the loop). The latents are
the states.
- Tool: `results/interp_planning_d3.py` (Q1).
- Caveat learned: the floor-dominated 5-way aggregate probe is uninformative; use **per-object balanced**.

### Step 2 — Routing respects $\mathcal{N}$ ✓ (emergent, causal)
- **Recovery of $\mathcal{N}$ (correlational, 160M):** with no mask, attention **mass** concentrates on
  the wall-masked one-step neighbours at **7–13×** chance (cell0 13.0×, cell1 11.8×, cell2 7.0×; chance
  0.027). Entmax support is broad (~44–88 keys) → mass-concentration, not hard sparsity.
  Tool: `interp_planning_d3.py` (Q2).
- **Through-walls (CAUSAL, 300M) — the key test:** flip a floor cell to a wall at matched pixel
  distance and measure agent-latent shift. **Blocked-by-wall influence is 0.38× the open one**
  (open 0.857 vs blocked 0.323; partial corr(influence, graph-dist | Euclid) = **−0.32**). Information
  **routes around walls, along $\mathcal{N}$**, not through Euclidean space.
  Tool: `results/interp_wall_d3.py`. (Onset/timing is distance-independent ~2 ticks — fast/parallel —
  but *magnitude* is graph-gated; `interp_topo_d3.py`, `interp_perturb_d3.py`.)

### Step 3 — Planning = TWO-STEP generalized policy iteration ✓ (CORRECTED — was mislabeled "value iteration")
The thinking loop does **amortized, graph-respecting policy EVALUATION**; the policy **IMPROVEMENT**
(the max / lookahead) happens once at the **actor head**. It plans by *propagating values* (DP-style),
not by searching trajectories. Evidence:

**(3a) Evaluation in the loop — goal-directed value propagation; NOT value iteration, NOT successor rep.**
- **Thinking helps:** faithful depth sweep `valid_medium`: d1 0.258 → d4 **0.336** (peak) → d6 0.319.
  Tool: `results/thinking_curve_vardepth.py`.
- **No max in the loop (architectural + mech-interp).** Trained config is `readout="softmax"`,
  `attn_norm="entmax15"`, `directional_value=False` (`config.py:346`): the per-tick aggregation
  $z=(A\,v)W_{out}$ is a **convex average** $\mathbb{E}_{s'\sim A}[\cdot]$, *not* a max. The codebase's
  soft-Bellman `maxplus` operator is **OFF** for this checkpoint. **E1 confirms behaviorally:** the
  attention operator $A$ is **stationary and does NOT sharpen** (top-1 mass flat ~0.16, support ~85/100
  keys, $\cos(A_t,A_{t-1})\to0.995$). Tool: `results/interp_e1_d3.py`.
- **Math model of one tick:** linearized, the value channel is $c\leftarrow\gamma_{eff}\,A\,c + r_{eff}$
  ⇒ fixed point $(I-\gamma_{eff}A)^{-1}r_{eff}$ = the discounted **policy-evaluation resolvent**.
  $A\leftrightarrow\gamma P^\pi$ (entmax routing on $\mathcal{N}$), $\sigma(g_f)\leftrightarrow$ discount,
  the $[x;\text{prev}]W_{in}$ injection $\leftrightarrow$ reward source; the $\max_a$ is absent.
- **It's policy EVALUATION, not successor representation (E2).** Relocating the goal (move reward, keep
  $P$) shifts the propagated latent $h_2$ at far cells by **0.35** (relative) — *more* than a matched far
  wall-flip (0.22; ratio **1.55**) — and the agent cell by 0.27 (ratio 1.34), while the readout value
  shifts $0.84\,\sigma$. SR predicts ~0 (reward-agnostic features); the goal is **baked into the iterate**.
  Tool: `results/interp_e2_d3.py`.
- **Genuine but amortized iteration (E1).** The hidden contracts at $\rho\approx0.89$ (effective horizon
  ~9 ticks), only ~60% converged at the trained K=6 — real inference-time iteration, but effective
  horizon (~9) ≪ the task's $\gamma{=}0.97$ horizon, so the long-range solve is **amortized into the
  weights** and the loop does a few refinement sweeps.
- **The learned kernel (E4):** where each cell pulls value from — attention mass decays geometrically in
  **graph** distance ($\rho\approx0.66$, reach ~3 hops), **34% of mass beyond 1 hop**, **0.00 through walls**
  (graph-unreachable), and a **+0.07 direct anchor on the goal cell**. A board-general, graph-respecting,
  goal-anchored propagation kernel. Tool: `results/interp_e4_d3.py`.
- **Reach COMPOUNDS over ticks (E5) — multi-step iterative deepening.** A perturbation $d$ hops away
  reaches the agent only after *more* ticks for larger $d$ (arrival rises with $d$; horizon grows 0→2→8→9
  over ticks 1→4). One tick spans ~3 hops, yet cells 7–9 away still arrive — so the cells a latent pulls
  from are **themselves updated** each tick and reach compounds (~3 hops/tick, board fills in ~3–4 ticks).
  Genuine multi-step propagation (not a slow 1-ring-per-tick crawl, not a one-shot blur).
  Tool: `results/interp_e5_d3.py`. (Corrects an earlier "parallel/distance-independent onset" read.)
- **The converged value IS a recursive Bellman fixed point (Bellman test, 300M).** Optimality residual
  $|V-\max_a(r+\gamma V(s'))|/\mathrm{std}(V)$ along the greedy trajectory at depths 0–3 =
  **[0.19 0.25 0.25 0.22]** — low + flat (recursion holds = multi-step), with a constant *negative*
  soft-Bellman gap ($V<\max Q$, entropy-regularized). Tool: `results/interp_bellman_d3.py`.
  **NB: this is depth-agnostic** (a property of the converged value, not the per-tick process) — it does
  *not by itself* show the loop does the backups; **E1/E2 establish the mechanism**.

**(3b) Improvement at the head — one-step lookahead.**
- **Lookahead consistency (300M):** the policy agrees with $\arg\max_a[r_a+\gamma V(s'_a)]$ over its
  **own value** at **0.41** (chance 0.25), and consistency **rises with thinking depth** (0.34→0.41):
  as the loop sharpens the evaluated value, the head's greedy action increasingly matches the
  value-lookahead choice; thinking flips 13.7% of actions to it. Tool: `results/interp_lookahead_d3.py`.
- **No explicit multi-step plan is stored:** decoding executed actions from the readout, only $a_0$ is
  decodable (0.70→0.97 over ticks); $a_1..a_5$ at chance. Tool: `results/interp_planq_d3.py`.

**(3c) The value field tracks TASK & PHYSICS changes, reaches the agent, and re-plans (the strengthening).**
- **Tracks the task (E6b):** relocating the goal/box to the far side shifts the readout value in the
  resolvent-predicted direction — $\mathrm{corr}(\Delta V,-\Delta\text{dist})=+0.18$ (goal) / $+0.16$ (box);
  a box move moves $V$ *more* than a goal move ($1.5\sigma$ vs $1.0\sigma$). Tool: `results/interp_e6b_d3.py`.
- **Adopts new PHYSICS (E9):** a wall that lengthens the agent's route lowers $V$ (**94%** of cases,
  $-1.3\sigma$) vs a matched off-path wall ($-0.35\sigma$); the effect **builds over ticks**.
  Tool: `results/interp_e9_d3.py`.
- **Reaches the agent + re-plans (E10):** a path-blocking wall *outside the agent's conv view* shifts the
  agent's **own** latent **2.2×** an off-path wall (building over ticks) and changes its greedy action
  **3×** more often. So: new physics → value drops → propagates back to the agent → re-plans.
  Tool: `results/interp_e10_d3.py`.
- **Mechanism — via CONTENT, not re-routing (E7/E7b/E8).** The attention operator $A$ does **not** re-route
  for injected edits (row-change near/far ratio 1.1×; neighbours of a new wall keep attending to it —
  `interp_e7_d3.py`/`interp_e7b_d3.py`), and wall cells are **not** dead-norm ($\|v\|$ wall/floor 0.94, flip
  1.06 — `interp_e8_d3.py`). So $A\leftrightarrow\gamma P$ is a **fixed, board-conditioned** graph-kernel;
  new physics enters as **propagated value content**, not by re-wiring routing. A *functional* world model
  (the value adopts the board's physics) realized by a fixed kernel + content, not a dynamically re-routing one.

**One-liner:** loop = amortized, graph-respecting **policy evaluation** — value propagation along a
**fixed, board-conditioned** $\mathcal{N}$-kernel to a recursive Bellman fixed point; the
**improvement/lookahead** is a single argmax at the actor head (one step of generalized policy iteration).
Interventions confirm the value field **tracks task (goal/box) & physics (wall) changes, reaches the agent,
and re-plans** — via propagated value *content*, not by re-routing. Corrects "value iteration" (no max ⇒
evaluation) and "successor representation" (goal in the iterate).

### Methodology notes (important — two probes were wrong then corrected)
- **Faithful recompute:** the model's own forward (`get_action`/`get_logits_and_value`) hits a
  `TracerArrayConversionError` on the offset-tied `rel_bias` gather under `nn.scan` (jit/GPU/eager-GPU).
  **ALL interp recomputes the D=3 entmax stack in plain JAX from params** (`recompute_d3` in
  `interp_planning_d3.py`) + head matmuls for logits/value — crash-free, GPU-safe, validated to
  reproduce the model's hidden to **1.8e-7** (`--self-test`).
- Corrected probes: (i) flat "steps-to-think" eval is the wrong axis → use the `n_active` sweep;
  (ii) decoding the plan from the *initial* state can't detect planning → use **successor-value
  lookahead** (Step 3). Both corrections flipped conclusions; keep them in mind.

---

## 3. WHAT'S NEXT (prioritized)

1. **Per-tick mechanism + interventions (DONE this session — `interp_{bellman,e1,e2,e4,e5,e6b,e7,e7b,e8,e9,e10}_d3.py`).**
   The loop is **amortized policy evaluation**: value propagation along a fixed, graph-respecting kernel
   (E4: $\rho{=}0.66$, 0 through walls, +0.07 goal-anchor) that **compounds over ticks** (E5), no max (E1
   stationary operator + `readout="softmax"`), converging to a recursive Bellman fixed point. The value
   field **tracks task & physics changes, reaches the agent, and re-plans**: goal/box relocation moves $V$
   in-direction (E6b, box>goal); a path-blocking wall lowers $V$ (E9: 94%, $-1.3\sigma$, builds over ticks);
   the change reaches the agent's own cell (E10: $2.2\times$ off-path) and re-plans the action ($3\times$).
   This is via **content propagation, not routing re-wire** (E7/E7b: $A$ near-invariant to edits; E8: walls
   not dead-norm). **See §2 Step 3.** *Optional nits:* the value↔distance correlation is coarse ($+0.15$); a
   box-push-aware $V^\*$ target would sharpen the per-cell field; E3 no-kinks is a belt-and-suspenders no-max check.
2. **Emergence-over-training sweep (the key remaining FIGURE):** run the Step-1/2/3 probes
   (binding, through-walls, lookahead-consistency, thinking-curve) at checkpoints **2M / 20M / 80M /
   300M**. If all three start near-chance and rise together, that's the causal "RL *induced* the chain"
   result — the heart of the thesis. (Pure inference on existing ckpts; GPU is free.)
3. Optional sharpening: nonlinear probes; box-push-aware value ground truth (vs the navigation proxy);
   re-run interp at the final 300M ckpt (most numbers above are 160M).
4. Deeper-planning lever (separate experiment, optional): if we want planning beyond shallow ~2–3-tick
   propagation, try a depth curriculum or a convergence-promoting "deep-thinking" objective — NOT a
   mask.

**Writeup:** `writeup/planning_emergence.tex` + `planning_emergence.pdf` (4 pages: 2-page body with the
3-step results + 4 figures, then **App. A** architecture/per-tick equations/training and **App. B** every
experiment with method + key number. Build with `tectonic`; PDF gitignored). Reflects the full E1–E10 arc.

---

## 4. INTERP TOOLING (all in `results/`, run on the node)

All load a checkpoint, **recompute** the core (crash-free), and analyse. Run with a small GPU slice
(`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.1`) or CPU.

| script | what it measures | step |
|---|---|---|
| `interp_planning_d3.py` | binding (per-object) + recovery-of-$\mathcal{N}$ + nav-plan probes; `--self-test` validates recompute | 1,2 |
| `interp_wall_d3.py` | through-walls causal: blocked vs open influence at matched pixel dist | 2 |
| `interp_perturb_d3.py`, `interp_topo_d3.py` | causal influence-radius vs ticks; graph-vs-Euclidean onset | 2 |
| `interp_lookahead_d3.py` | policy vs 1-step lookahead over own value, by thinking depth (step 3b) | 3 |
| `interp_planq_d3.py` | multi-step plan decodability vs tick (is a trajectory stored?) | 3 |
| `interp_bellman_d3.py` | Bellman residual at agent + successors (recursion depth) | 3 |
| `interp_e1_d3.py` | per-tick mechanism: attention stationarity vs sharpening, contraction rate, value field (VI vs eval vs null) | 3a |
| `interp_e2_d3.py` | reward-relabel invariance: policy-evaluation vs successor-representation | 3a |
| `interp_e4_d3.py` | the learned kernel: attention mass vs graph distance (decay $\rho$, 0-through-walls, +goal-anchor) | 2,3a |
| `interp_e5_d3.py` | reach vs ticks: does propagation compound (are the pulled-from cells updated)? | 3a |
| `interp_e6b_d3.py` | far-side goal/box interventions: readout-value direction + magnitude (box>goal) | 3c |
| `interp_e7_d3.py`, `interp_e7b_d3.py` | does the operator $A$ re-route under edits? (it does **not**; local $\|\Delta A\|$ + neighbour mass) | 3c |
| `interp_e8_d3.py` | value-projection norm by tile + floor→wall flip (are walls "dead"? **no**) | 3c |
| `interp_e9_d3.py` | does the value field adopt new physics? (path-blocking vs off-path wall lowers $V$) | 3c |
| `interp_e10_d3.py` | does the wall's value-change reach the agent + re-plan its action? | 3c |
| `thinking_curve_vardepth.py` | solve-rate vs inner thinking depth `n_active` | 3 |

Checkpoint path: `/workspace/runs/vd_d3_entmax/default/wandb/offline-run-*/local-files/cp_<step>`
(`ls -dv … | tail -1` for the latest). Load via
`cleanba.cleanba_impala.load_train_state(Path(cp), env_cfg=…)`.

---

## 5. INFRA / HOW TO DRIVE THE NODE

- **Node:** H200, `ssh kz362uma1m94cz-64412132@ssh.runpod.io -i ~/.ssh/id_ed25519`. Repo
  `/workspace/recurrent-planner`, venv `.venv` (jax cuda12 0.4.34), sokoban cache `/opt/sokoban_cache`.
- **PTY-only proxy:** `ssh host 'cmd'` hangs. Pipe a heredoc to `ssh -tt … BatchMode=yes` ending with
  `exit`; filter echo with `tr -d '\r' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g'`. Needs Bash
  `dangerouslyDisableSandbox: true` (outbound SSH). **NB: that flag is gated by an auto-mode safety
  classifier; if it returns "temporarily unavailable", just wait/retry — not a code problem.**
- **File transfer (scp/sftp don't work):** for small files, base64 → `fold -w700` → `printf '%s' 'CHUNK'
  >> /tmp/f.b64` lines → `base64 -d > target`, verify SHA. For a ~14MB checkpoint, stream it OUT:
  `base64 -w0 file; echo` over the SSH stdout, then strip ANSI + decode locally (verify SHA).
- **Local CPU diag venv** (`/tmp/attn-diag`, ARM Mac, no envpool) for `--self-test`s; rebuild via
  `uv venv /tmp/attn-diag --python 3.10 && uv pip install … "jax==0.4.34" "jaxlib==0.4.34" "flax~=0.8.0"
  "optax~=0.1.4" "numpy==1.26.4" gymnasium rlax "setuptools<81" gym_sokoban` (it gets wiped by /tmp
  cleanup).
- **Local LaTeX:** `tectonic` (self-contained) — `cd writeup && tectonic planning_emergence.tex`.

---

## 6. STALE / DON'T
- **CORRECTED (do not revert):** the loop is **policy EVALUATION + head-side lookahead improvement**, NOT
  "value iteration" and NOT "successor representation." No $\max_a$ exists in the trained loop
  (`readout="softmax"`, `maxplus` off, convex-average aggregation; E1 stationary operator); the goal is in
  the iterate (E2). Use the §2 Step 3 wording, not the old "amortized value iteration."
- **The operator $A$ is a FIXED, board-conditioned kernel (do not claim it dynamically re-routes).** Under
  injected edits it does **not** re-wire (E7/E7b) and walls are not dead-norm (E8); the value field still
  adopts new physics (E9/E10) via **propagated value content**, not routing changes. It's a *functional*
  world model, not a re-routing one — phrase §2 Step 2 as "routes along the **given** board's $\mathcal{N}$."
- **Do NOT add a hard attention mask** (`use_attention_mask=True`) as the main run — it hard-codes
  $\mathcal{N}$ and defeats the emergence claim. Dense + entmax is the substrate.
- **Do NOT call the model's own `get_action`/`get_logits_and_value` for interp** — `rel_bias`-gather
  tracer crash. Use `recompute_d3` + head matmuls.
- Old runs (slot cores `s_sm_*`/`s_mp_*`, cellwise, the 200M `vardepth`, the 600M D=1 `vardepth600`,
  smoke tests) are **superseded** and being deleted from the node. The slot core
  (`cleanba/slot_lstm.py`) remains in the source as the documented negative control (free slots do NOT
  recover $\mathcal{N}$); keep the code, not the run dirs.
- Training `avg_episode_returns` is misleading (averaged over depths + partial-box credit); trust the
  post-hoc eval / interp.
