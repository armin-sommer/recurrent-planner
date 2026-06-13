# AGENT HANDOFF — recurrent-planner

_Single source of truth for resuming. Rewritten 2026-06-13. Read top to bottom. Supersedes the
auto-loaded `~/.claude/.../memory/` notes where they conflict (old RunPod IPs are dead — see §7)._

---

## 0. TL;DR / CURRENT STATE

> **UPDATE 2026-06-13 (session: entmax + D=3).** The §0 bullets below about "RUNNING NOW: the 600M"
> and the "sparsity TODO / path-B entropy penalty" are SUPERSEDED — read this block first:
> - **Sparsity is DONE, architecturally, via α-entmax — NOT the entropy penalty.** New module
>   `cleanba/entmax.py` (sparsemax α=2 + 1.5-entmax, closed-form `custom_vjp`, FD-verified). New cell
>   flag `AttentionCellConfig.attn_norm ∈ {softmax,entmax15,sparsemax}` routes BOTH softmax sites; the
>   `softmax` path is byte-identical (regression-checked) so every other config is unchanged. The
>   reverted sow/nested-scan entropy penalty (old §2) is ABANDONED — entmax needs no loss term, no
>   scan plumbing (sidesteps that bug entirely), and gives EXACT-zero weights = a learned sparse
>   support. Dead `attn_entropy_coef` removed from `impala_loss.py`.
> - **The 600M D=1 run was KILLED** at ~36.5M/600M to free the GPU (its 30M ckpt remains at
>   `/workspace/runs/vardepth600/…`). Superseded by the run below.
> - **RUNNING NOW: `sokoban_drc_attn_vardepth_entmax_d3`** — dense attn + **entmax15** (learns its own
>   hard sparse support) + **n_recurrent=3** (D=3; the suspected fix for the flat thinking-curve) +
>   variable depth d∼U{1..6}, **300M** steps. `tmux vd_d3_entmax`, run dir
>   `/workspace/runs/vd_d3_entmax`, log `/workspace/logs/vd_d3_entmax.log`. Confirmed stepping, 0
>   errors; steady SPS ~2.7–2.9k (D=3 ≈ 3× D=1), ETA ~30h. Checkpoints at 2/5/20/40…300M.
>   **Monitor:** `grep 'update=' /workspace/logs/vd_d3_entmax.log | tail` (returns should rise from
>   ~−7, ep-length fall).
> - **Gradient-through-every-tick (#2) is PROVEN bit-exact** — `tests/check_vardepth_grad.py`: cell-grad
>   == 0 at d=0, grows with d, and gated(n_active=d) grad == native d-tick grad to 0.0 (no leakage).
>   `tests/check_entmax.py` validates the normalizers (sum-to-1, true zeros, grad == finite-diff ~1e-10).
> - **NEXT:** when checkpoints land, run the interp (§5): does the thinking-curve now rise past d=1
>   (D=3 working)? does the entmax support land on 𝒩 as a HARD sparse mask (sharper than the soft
>   3.1×)? Compare against the 200M D=1 softmax vardepth.

- **Paper thesis:** model-free RL on a *relational recurrent core* induces planning when (i) latent
  cells **bind** to environment states and (ii) the content-based routing recovers the transition
  graph 𝒩. Then the "thinking" loop = K-step policy improvement. (`paper` draft, abstract.)
- **This session built** three things on top of the prior attn/cellwise cores: (a) a **free-slot
  core** (`slot_lstm.py`) — attention between non-spatial latent slots; (b) **variable-thinking-depth
  training** (gate the recurrence to a sampled depth d per rollout) — validated + trained; (c) an
  **attention-sparsity entropy penalty** (§2) — fully wired + validated in isolation, but **reverted**
  before the long run after a node smoke test caught a nested-scan bug (see §0 status + §2). So the
  slot core + variable-depth are trained; **sparsity is the one open TODO.**
- **Headline results (§3):**
  - **Spatial dense attention LEARNS 𝒩** (the abstract's central novel claim): the unmasked
    square×square attention concentrates **3.1× over chance** on grid-neighbours + the learned
    positional bias up-weights the neighbour ring. The **free-slot core does NOT** (1.03×, diffuse) —
    it's the spatial structure that enables recovery-of-𝒩; the free-slot run is the negative control.
  - **But recovering 𝒩 ≠ strong planning here.** The variable-depth dense model is a weak solver
    (28% train / ~1% valid) and its thinking-curve is **flat beyond d=1** (only the d=0→1 jump
    matters — running the core at all). Most likely cause: `n_recurrent=1` (single layer, too little
    capacity to *use* the recovered graph).
- **RUNNING NOW: the plain 600M (no sparsity yet).** `sokoban_drc_attn_vardepth_600m` launched on the
  H200 at 06:53Z — `tmux vd600`, run dir `/workspace/runs/vardepth600`, log
  `/workspace/logs/vardepth600.log`. Spatial dense attn, `n_recurrent=1`, `K_max=6`, softmax,
  **variable depth d∈{1..6}** (always ≥1 tick), 600M steps, checkpoints every 30M (~18h).
  **FIRST THING NEXT SESSION: confirm it's still stepping** —
  `grep 'update=' /workspace/logs/vardepth600.log | tail` (expect SPS≈9k, returns rising, ep-length
  falling). It is byte-for-byte the code that ran the 200M vardepth, so it should be healthy — but it
  was launched right at the end of the session and **was not yet confirmed stepping**, so verify.
- **⚠️ SPARSITY IS NOT IN THIS RUN — it hit a bug and was reverted.** The user wanted an attention
  entropy-sparsity penalty (§2). It was fully wired and validated in isolation (micro-test + local
  forward + gradient), but the **node smoke test caught a real bug**:
  `scan got values with different leading axis sizes: 1, 6, 6` in `_apply_cells`'s inner tick-scan.
  Root cause: the sow-extraction mechanism `variable_axes={'aux':0}` **does not compose across the
  NESTED scans** (inner ticks × outer time) — it validated in a single scan but not nested. Rather
  than risk an 18h run on buggy code, I **reverted all 4 wiring files** (`cleanba_impala.py`,
  `impala_loss.py`, `convlstm.py`, `attn_lstm.py`) to the known-working state and launched the plain
  600M above. The only vestige left (harmless, dormant): `ImpalaLossConfig.attn_entropy_coef`
  (default 0) + `config.py` line setting it to 0.01 — `impala_loss` no longer reads it.
- **TO ADD SPARSITY (the open TODO — path B):** re-apply §2's 5 pieces but make the sow survive the
  nested scans. The pieces that were CORRECT (reuse them): explicit-softmax to expose the weights;
  `if self.is_mutable_collection('aux'): self.sow('aux','attn_entropy', mean H)` so only the learner
  sows; `mutable=['aux']` on the learner's `get_logits_and_value` partial; the loss unpack + penalty +
  `attn_entropy_coef`. **ONLY the nested-scan stacking failed.** Fixes to try: (i) `variable_carry='aux'`
  instead of `variable_axes={'aux':0}` on both scans + `sow(..., reduce_fn=jnp.add, init_fn=lambda:0.0)`
  — carry a scalar entropy accumulator through each scan instead of stacking a per-tick axis; or
  (ii) thread the entropy out as an explicit scan `y`-output rather than a sown collection. **Then
  RE-SMOKE** (`--from-py-fn=cleanba.config:sokoban_drc_attn_vardepth_600m total_timesteps=1000000`) —
  it MUST step with 0 errors before relaunching the full run. λ=0.01 start; watch `attn_entropy` fall.
- **When the 600M finishes (or at a 30M ckpt): run the mechinterp (§5)** — does ≥1-tick depth + the
  longer run give a rising thinking-curve / better valid generalization than the 200M, or is
  `n_recurrent=1` still the ceiling (→ next experiment: bump to `n_recurrent=3`)? Add sparsity (path B)
  and compare recovery-of-𝒩 (does it sharpen past 3.1×?).
- **Repo:** branch `recurrent-attention-core`. Local edits this session are **NOT committed/pushed**
  (deployed to the node via base64, §6). Commit when ready.

---

## 1. WHAT THIS SESSION ADDED (architecture + mechanisms)

All in the cleanba JAX/flax codebase. Three additions, each isolated + backward-compatible:

### (a) Free-slot core — `cleanba/slot_lstm.py` (`sokoban_drc_slots_*` in config.py)
`BaseLSTM` subclass whose latent cells are **N free slots (no spatial position)**. Binding = a
slot-attention competition over the H·W board tokens (softmax over slots); routing = dense slot↔slot
attention; per-slot learnable identity `slot_mu` (added each tick — distinct slots even from a zeroed
carry; the "start at μ" stabilizer). `skip_final=False` required (slots can't take the spatial skip).
Carry shape `(B, N, d)`. **Verdict: trains but overfits + routing stays diffuse (§3) — the purest test
of "discovered binding+𝒩", and it failed, which is the informative negative control.**

### (b) Variable thinking depth — `convlstm.py` + threaded through IMPALA
The recurrence depth (inner thinking ticks = `repeats_per_step`) is **gated to a per-rollout sampled
depth `n_active`**: `_apply_cells` always scans the static `K_max` ticks, but each tick `k` does
`carry = where(k < n_active, updated, carry)` — ticks past the budget are identity. **Gradient-exact:
verified `grad(gated d) == grad(true d-tick model)` bit-for-bit, and cell-param grad scales with d
(0 at d=0 → grows to d=6). So at d ticks the core is backpropped exactly d times.** `n_active` is
sampled once per rollout (uniform in `Args.variable_thinking_depth=(lo,hi)`), **stored in the
`Rollout` tuple (`n_active_t`)** so the actor and learner replay the *same* depth (V-trace stays
consistent), threaded through `get_action`/`get_logits_and_value`/`step`/`scan`/`_apply_cells`.
Default `n_active=None` → full depth → other cores unchanged.

### (c) Attention-sparsity entropy penalty — see §2 for the full explanation
`Args.loss.attn_entropy_coef · (mean attention entropy)` added to the loss → pushes cells to attend
to *few* sources. Wired via a `sow`-under-scan + `is_mutable_collection('aux')` trick (§2).

---

## 2. THE SPARSITY PENALTY (design — currently REVERTED, see §0)

> **STATUS:** this design was implemented + validated in isolation but **reverted** before the 600M
> run — the node smoke caught a nested-scan bug (the `variable_axes={'aux':0}` step does not compose
> across the inner-tick × outer-time scans). The code below is the *intended* design; everything
> except the nested-scan stacking was correct. This section is the spec for re-doing it (path B in
> §0): keep all the pieces, just swap the extraction mechanism (`variable_carry`+`reduce_fn`, or an
> explicit scan `y`-output) and re-smoke before launching.

**Goal.** Stop each cell from "accruing a little from everyone" (the diffuse/over-smoothing failure
we measured in the free-slot core: attention entropy ≈ ln(100), uniform). Push the attention to be
**sparse/peaked** (attend to a few — ideally the reachable neighbours).

**Mechanism = soft entropy penalty.** Each cell's attention is a softmax distribution `w` over the S
keys. Its entropy `H(w) = −Σ wⱼ log wⱼ` is high when spread out, low when peaked. We add
`λ · mean_cells H(w)` to the training loss; minimizing it pressures the attention to sharpen — soft
(traded against the RL reward) and tunable via `λ` (`attn_entropy_coef`, currently 0.01).

**The hard part + the fix.** The attention `w` is computed *inside* two nested scans (thinking-ticks
× time), so getting its entropy out to the loss is the challenge (naive `capture_intermediates`
crashes — it's exactly what broke the interp). Solution, 5 pieces:
1. **`attn_lstm.py`** — compute the softmax weights *explicitly* (an einsum + `jax.nn.softmax`, ==
   `nn.dot_product_attention`) so `w` is in hand; compute `H(w)`; **`self.sow('aux','attn_entropy',
   mean H)` — but ONLY `if self.is_mutable_collection('aux')`.** This is the key trick: the cell sows
   the entropy *only when the caller asked for it* (made `aux` mutable). The **learner** does; the
   **actor and eval do not** → they skip the sow entirely (no error, no plumbing on their side).
2. **`convlstm.py`** — both scans (tick-scan in `_apply_cells`, time-scan in `scan`) declare
   `variable_axes={'aux':0}` so the sown scalar is stacked through the scans. Verified safe for cores
   that never sow `aux` (conv/slot) — the collection is just empty.
3. **`cleanba_impala.py`** — the learner's `get_logits_and_value` partial gets `mutable=['aux']`, so
   the cell sows and `policy.apply` returns `(output, aux_collection)`.
4. **`impala_loss.py`** — unpacks `(…), _aux = get_logits_and_value(…)`, computes
   `attn_entropy = mean(all sown values)`, and adds `args.attn_entropy_coef * attn_entropy` to
   `total_loss` (+ logs it as a metric). Gradient flows through the sown entropy to the params
   (verified).
5. **`config.py`** — `sokoban_drc_attn_vardepth_600m` sets `attn_entropy_coef=0.01`.

**Validation done:** standalone flax micro-test (sow-under-scan extracts + stacks + gradient flows +
unused-collection-safe); local model test (learner extracts entropy 4.51@randinit + gradient flows;
actor/eval return the plain 4-tuple with no sow; conv core unaffected); node smoke ran with 0 errors.
**To tune:** raise `λ` for more sparsity (too high kills useful attention). Watch the `attn_entropy`
metric in wandb — it should fall over training.

---

## 3. RESULTS (this session + prior context)

### Free-slot core `s_sm_0` (sokoban_drc_slots_softmax, 200M) — NEGATIVE CONTROL
- Behavioural: train_unfiltered **0.67**, valid_medium **0.065**, thinking-curve **flat**.
- Interp (`interp_slots.py`): binding is **object/salience-biased** (bound squares 1.7× enriched for
  box/target/agent, targets 2.1×; avoids walls 46% vs 69%) but **not injective** (18.9 distinct
  squares, per-slot type-consistency 0.52); routing is **diffuse** (recovery-of-𝒩 **1.03×**, entropy
  4.35 ≈ uniform 4.61); ablating routing → **90% of actions unchanged** (routing ~10% load-bearing).
- Read: free slots overfit + plan via diffuse binding-refinement, **not** graph propagation.

### Spatial dense-attention, variable-depth `vardepth` (200M) — the headline + the caveat
- Config: `sokoban_drc_attn_vardepth` = dense attn (no mask), softmax, `n_recurrent=1`, `K_max=6`,
  d∈{0..6}. Returns −7→−0.18 (avg over depths), ep-length 75→62.
- **Recovery-of-𝒩 (`interp_attn.py`) — POSITIVE:** square×square attention concentrates **3.1×** on
  grid-neighbours (per-head 4.7/0.3/4.3/3.1 — 3 local heads + 1 global), entropy 3.85<4.61; learned
  positional bias ring-1 **+0.27** vs center/far ≈0. **Dense attention LEARNED the local stencil with
  no mask** — the abstract's central claim, and what the free-slot (1.03×) couldn't do.
- **Thinking-curve (`thinking_curve_vardepth.py`, sweep inner depth d=0..6) — NEGATIVE:** valid
  d=0:0.000→d=1:0.008→…→d=6:0.005 (~1% flat); train d=0:0.000→d=1:0.253→d=2:**0.286**(peak)→d=6:0.274.
  Only the **d=0→1 jump** matters (running the core at all → validates the (1,6) range). Beyond d=1
  thinking is flat. Weak solver (28% train, ~1% valid).
- Read: **recovering 𝒩 was necessary but not sufficient.** Likely culprit for weak planning:
  `n_recurrent=1` (too shallow to *use* the graph). → motivates the 600M with a hope that ≥1-tick
  depth + sparsity (+ maybe later `n_recurrent=3`) helps.

### Prior attention/cellwise arms (in `all_ckpts.tar`, pre-session — context)
thinking-curve (axis-2, valid_medium): attn_softmax(masked) 0.458 +0.049 · attn_plain(masked maxplus)
0.377 +0.047 · attn_dense(maxplus) 0.215 +0.013. **Aggregation contrast:** maxplus ≈ softmax ≈ mean
for thinking-lift (Bellman op not special); the **locality MASK** drives thinking-scaling far more
than the aggregator; `dense_sum` diverges (dense needs a *normalized* aggregator).

---

## 4. THE 600M RUN (running — pick this up first)
- `sokoban_drc_attn_vardepth_600m`: spatial dense attn, softmax, `n_recurrent=1`, `K_max=6`,
  **d∈{1..6}** (always runs the core ≥1 tick — d=0 was useless), **+ entropy penalty λ=0.01**, 600M
  steps, checkpoints every 30M. `tmux vd600`, `/workspace/runs/vardepth600`, log
  `/workspace/logs/vardepth600.log`.
- **Monitor:** `grep 'update=' /workspace/logs/vardepth600.log | tail` (SPS≈9k solo; returns;
  ep-length). Also pull the **`attn_entropy`** metric from the wandb offline files — it should DROP
  over training (sparsity working). Watch for collapse (returns stuck) / NaN.
- **When done (or at a good ckpt):** run the mechinterp (§5) on the final ckpt: (i) `interp_attn`
  recovery-of-𝒩 (did sparsity sharpen it past 3.1×? entropy lower?), (ii) `thinking_curve_vardepth`
  (does the depth sweep now rise past d=1?), (iii) compare to the 200M vardepth + the free-slot.
- **If planning is still weak:** the prime suspect is `n_recurrent=1`. Next experiment: bump to
  `n_recurrent=3` (match the prior attn arms' capacity) + d∈{1..6} + sparsity, short re-test first.

---

## 5. MECH-INTERP TOOLS (all in `results/`, run on the node)
Approach: load a checkpoint, run/recompute the core, extract attention + per-cell states, analyse.
All validated to reproduce the model's own readout. Run with the node venv + a ckpt dir (a `Path`).
- **`thinking_curve_vardepth.py --ckpt <cp_dir>`** — sweeps the *inner* depth `n_active`=0..6 (bakes
  it into `get_action`, `steps_to_think=[0]`), reports valid_medium + train solve-rate per depth.
  This is the faithful thinking-test for variable-depth models (the eval harness's `steps_to_think`
  is a *different* axis — extra full-depth steps on the initial obs).
- **`interp_attn.py --ckpt <cp_dir> [--king]`** — SPATIAL attention recovery-of-𝒩: recomputes the
  square×square attention, measures mass on grid-neighbours vs chance (lift), per-head, entropy, and
  the learned `rel_bias` by offset. Gets the embed via a method-apply of `_compress_input` (NOT
  `capture_intermediates`, which crashes on the rel_bias gather).
- **`interp_slots.py --ckpt <cp_dir>`** — slot-core: binding (slot→square, object enrichment),
  recovery-of-𝒩 over slots, routing ablation, per-tick.
- Checkpoint dirs: `/workspace/runs/<run>/default/wandb/offline-run-*/local-files/cp_<step>` (find the
  latest with `ls -dv … | tail -1`). Load via `cleanba.cleanba_impala.load_train_state(Path(cp), env_cfg=…)`
  → returns `(policy, _, cp_cfg, train_state, _)`.

---

## 6. INFRA / HOW TO DRIVE THE NODE
- **Node:** H200 (144GB, 192 cores). `ssh kz362uma1m94cz-64412132@ssh.runpod.io -i ~/.ssh/id_ed25519`.
  Repo `/workspace/recurrent-planner`, venv `.venv` (jax 0.4.34 cuda12), sokoban cache
  `/opt/sokoban_cache/boxoban-levels-master`.
- **PTY-only proxy:** `ssh host 'cmd'` hangs. Pipe a heredoc to `ssh -tt … ` ending with `exit`;
  filter echo with `tr -d '\r' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g'`. Needs Bash
  `dangerouslyDisableSandbox: true` for outbound SSH. The PTY occasionally *displays* doubled chars
  (echo only) — verify by SHA, don't trust the echo.
- **File transfer (scp/sftp DON'T work):** gzip → base64 → `fold -w700` → `printf '%s' 'CHUNK' >>`
  lines → `base64 -d | gunzip > target` → **verify `sha256sum` == local `shasum -a 256`, retry on
  mismatch.** (See any transfer block in this session's history.) Inline the `ssh` command in the
  loop — don't put it in a shell var (zsh won't word-split it).
- **Launch:** `tmux new-session -d -s NAME "cd /workspace/recurrent-planner && PYTHONUNBUFFERED=1
  WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python -m cleanba.cleanba_impala
  --from-py-fn=cleanba.config:FN <overrides> > /workspace/logs/NAME.log 2>&1"`. Overrides are bare
  `key=value` (farconf): `total_timesteps=…`, `seed=…`, `base_run_dir=…`. Kill: `tmux kill-session` +
  `pkill -9 -f cleanba_impala` (tmux-kill leaves the python alive).
- **Long waits:** local `sleep N` in a foreground Bash is blocked; use `run_in_background: true`.
  Remote `sleep` inside the SSH heredoc drops on the proxy idle-timeout past ~6 min. A node-side
  `tmux` logger that appends every 5 min survives; poll the log.
- **Build env (if node is reset — `/opt` & uv wiped, `/workspace` persists):** `build_env_pod2.sh`
  pattern (uv venv py3.10, `--no-deps -r requirements.txt`, `-e . -e ./third_party/gym-sokoban`,
  envpool cp310 fork wheel, then `jax[cuda12]==0.4.34`); clone `google-deepmind/boxoban-levels` →
  `/opt/sokoban_cache/boxoban-levels-master`.

---

## 7. STALE / DON'T
- Old RunPod IPs in the `runpod-*` memory notes are DEAD. Only the node in §6 is live.
- `capture_intermediates=True` crashes the attention forward (rel_bias gather → TracerArrayConversion).
  Use the recompute / method-apply approach (§5), or the `sow('aux',…)+mutable` approach (§2).
- The training `avg_episode_returns` is **misleading** — it's averaged over sampled depths and
  includes partial-box credit, so it overstates the greedy solve-rate. Trust the post-hoc eval.

## 8. REFERENCES
DRC (Guez 2019, 1901.03559); Sokoban interp (Bush 2504.01871, Taufeeque 2407.15421); VIN (Tamar
1602.02867); sparse attention (Peters 2019, entmax/sparsemax); multi-step PI (Efroni 2018);
deep-thinking/extrapolation (Schwarzschild 2021), PonderNet (Banino 2021).
