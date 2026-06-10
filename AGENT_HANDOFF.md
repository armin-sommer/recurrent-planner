# AGENT HANDOFF — recurrent-planner: "planning = lookahead policy improvement in a relational backbone"

_Last updated 2026-06-10 ~21:30Z by Claude (Opus), mid-project. This file is the single source of truth for resuming — read it top to bottom. The user's local `~/.claude/.../memory/MEMORY.md` (+ linked notes) also auto-loads and overlaps with this._

---
## 0. ONE-PARAGRAPH STATE
We are building the empirical case that a neural backbone split into **state-indexed latent cells**, updated each "thinking step" by a **local relational rule `F_φ`** over the MDP's transition graph, learns **lookahead policy improvement** (GPI-style) — and that this is a property of the **relational structure + aggregation operator**, NOT of convolution specifically. Established: harness validated (DRC solves); **pure-local conv (no pool-and-inject) solves** ~12%; a **NON-conv max-aggregation GNN also learns & solves** ~7% with a thinking-time benefit (the generality result); the cellwise deadlock was diagnosed & fixed (it needed the **live obs folded into the message source** = `attend_inputs`). **NOW:** running the **mask × aggregation ablation matrix at 80M steps, split across two H100 pods**, plus an attn re-run to come.

> **⚡ INFRA UPDATE (2026-06-10 ~22:30Z): moved to a single 6×H100 node, full matrix runs in PARALLEL at 200M.**
> The two single-H100 pods ran the matrix *sequentially* at 80M. We now run the **full 6-arm mask×aggregation matrix at 200M, one arm per GPU**, on a fresh 6×H100 node → ~one arm's wall-clock (~5–6h) vs ~20h sequential. These are **fresh reruns** (matched LR schedule), NOT resumes — checkpoints DO exist for the 80M runs (`…/wandb/offline-run-*/local-files/cp_<step>/`, save_model on by default), but resuming the already-annealed runs would warm-restart the LR.
> - **SSH (direct TCP, supports scp/heredocs):** `ssh -p 10285 -i ~/.ssh/id_ed25519 root@157.66.254.38` (PTY proxy alt: `cj0gb8s6i2bu40-644119d0@ssh.runpod.io`). 224 cores, ~2 TB RAM (kills the old host-mem SIGKILL constraint → parallel arms are fine).
> - **Layout:** repo `/workspace/recurrent-planner` (branch `recurrent-attention-core`, built via `build_env_pod2.sh`); experiments `/workspace/claude_experiments/{logs,...}`; boxoban cache `/workspace/sokoban_cache` → symlinked `/opt/sokoban_cache`. Container disk is only 50G → **everything lives on the `/workspace` network volume.**
> - **Launcher:** `/workspace/claude_experiments/driver_matrix_200m.sh` → arms `{local,dense}_{max,mean,sum}_200m`. **GOTCHA: pin `CUDA_VISIBLE_DEVICES=0..5` (one GPU per arm)** or each JAX process grabs all 6 and they collide.
> - Commit `8e0d120` forces an exact-final checkpoint (the `eval_at_steps` schedule otherwise stops ~5% short of the last update).
> - The old 2 pods (PTY pod 1 + TCP pod 2 `157.66.254.33:19373`) are being **retired** once the new matrix is confirmed on-GPU; their 80M data stays as preliminary.
> - **Next after matrix:** standalone planning eval (`python -m cleanba.load_and_eval load_other_run=<run> only_last_checkpoint=true`, extended `EvalConfig.steps_to_think`), then a version-controlled `results/` dir (data CSV/JSON + `plot.py` + figures + RESULTS.md).
>
> **OVERNIGHT RUNBOOK (start 2026-06-10 ~23:20Z; matrix ETA ~late-evening PT). Eval was trimmed (commit `71c6f17`: cellwise `steps_to_think=[0,8]` + ~every-10M eval points) → ~8k SPS/arm. Autonomous sequence once the 6 cellwise arms exit:**
> 1. **Detect done:** `pgrep -cf cleanba_impala` → 0 (per-arm: no proc with its `base_run_dir`). Each leaves a final ckpt at `…/<arm>/default/wandb/offline-run-*/local-files/cp_<~200M>/` (commit `8e0d120` forces the exact-final save).
> 2. **Planning eval** per final ckpt: `cd /workspace/recurrent-planner && .venv/bin/python -m cleanba.load_and_eval load_other_run=<arm_run_dir> only_last_checkpoint=true` with eval `steps_to_think=[0,2,4,8,12,16,24,32,48,64]` + higher `n_episode_multiple` (verify flag style: bare `k=v` farconf vs `--k`). Writes `*_metrics_dict.pkl`.
> 3. **Build `results/`** (repo): metrics → `results/data/*.{csv,json}`; `results/plot.py` → success-vs-thinking-ticks per arm + local-vs-dense bars + step-curves → `results/figures/`; `results/RESULTS.md`. Commit + push.
> 4. **Launch attn (2nd non-conv instance), one arm/GPU, 200M, `n_global=0`, eval-trimmed:** masked `sokoban_drc_attn_3_3` vs dense `sokoban_drc_attn_3_3_nomask` (= the masked-vs-dense primary test in attention); fill spare GPUs with `_plain`/`_softmax` ablations. Apply the same eval trim to `sokoban_drc_attn` first. Write `/workspace/claude_experiments/driver_attn_200m.sh` (mirror `driver_matrix_200m.sh`).
> 5. After attn: eval + fold into `results/`. (Old 2 pods TERMINATED — historical attn runs gone, not needed.)
> - **`dense_sum_200m` caveat/FINDING:** with matched init it diverged to nan (~step 8M) — dense + unnormalized sum over ~100 cells blows up. Relaunched on GPU5 with `net.recurrent.zero_init_message=true` (the ONLY arm differing; other 5 use `false`) and it trains fine. Masked `local_sum` is stable with standard init. Report as evidence the dense graph makes sum-aggregation ill-posed; the primary mask claim (local-vs-dense) stays clean via the fully-matched MAX and MEAN pairings.

---
## 1. THE CLAIM
- **Assumption 1 (latent-state graph):** cells `h_t(s) ∈ R^n` indexed by state `s`; edges `N` where MDP transition `s→s'` is physically possible.
- **Assumption 2 (recurrent update):** each thinking step `k`: `h_t^{k+1}(s') = AGG_{(s,s')∈N} F_φ(h_t^k(s))`; `φ` = time-invariant edge params; `AGG ∈ {max, mean}`; `K` thinking steps/env-step; carry `h_t^0 = h_{t-1}^K`.
- **Claim:** under RL this (i) **localizes** decision statistics to cells `h(s)`, and (ii) does **lookahead policy improvement** — each `F_φ` ≈ one hop of a Bellman/greedy backup; `K` steps = `K`-step lookahead; more thinking → better policy.
- **★ PRIMARY CLAIM (the paper's thesis): `F_φ` aggregating over the local NEIGHBOURHOOD `N` (masked) learns lookahead policy improvement; the SAME `F_φ` over a DENSE all-to-all graph does NOT.** The neighbourhood mask is the load-bearing inductive bias: it makes one `F_φ` step = one transition-respecting lookahead hop, so iterated `F_φ` localizes a plan/value field that propagates as a frontier. Dense `F_φ` lets information teleport across the board in one step, so no localized frontier/lookahead forms. **The `dense_*` arms (pod 2) ARE this test.** Prediction: dense fails to localize decision statistics + fails the frontier-propagation/lookahead signatures (measure via the probes in §8.3 — `1×1≈3×3` localization, frontier-per-tick, thinking-time scaling, OOD board-size generalization) — *even if its raw solve-rate ends up closer*. Scope the claim to MECHANISM + generalization, not in-distribution success.
- **Secondary contrast:** **AGGREGATION** `max` (Bellman-optimality = improvement) vs `mean`/`sum` (expectation = evaluation). NOTE: preliminary 80M data shows `mean` *also* learns and shows a thinking-time benefit, so do NOT overclaim "max ≫ mean" — see §6.
- **Generality goal:** show the PRIMARY claim holds in non-conv instances (GNN, attention) so it's about relational structure, not convolution; strongest = a **non-grid** transition graph (conv can't apply at all).

---
## 2. TASK
Sokoban (Boxoban), 10×10, `cleanba` (IMPALA, JAX/flax). Eval set `valid_medium`. The eval ALREADY sweeps extra thinking ticks `{00,02,04,08,12,16,24,32}` → keys `valid_medium/NN_episode_{successes,zero_boxes,one_box,...}`. **Success rising with `NN` = the lookahead signature.**

---
## 3. CODEBASE  (branch `claude/attn-experiments`; commits are LOCAL on the pods, not pushed to GitHub)
- `cleanba/convlstm.py` — DRC ConvLSTM. `BaseLSTM.scan`/`_apply_cells` does the inner `nn.scan` over `repeats_per_step` (the K thinking steps). Proven planner.
- `cleanba/cellwise_lstm.py` — **THE PAPER VEHICLE**: GNN message-passing core = Assumption 2 (shared message MLP `F_φ`, offset-tied edge weights, `aggregation ∈ {mean,sum,max}`, `use_neighbor_mask` toggle). My added knobs: efficient masked/dense `max`; `per_offset_message` (per-offset conv kernel — **conv-like, DON'T use for the non-conv claim**); `zero_init_message`; **`attend_inputs`** (fold live obs into message source = `conv_ih` analog — THIS broke the deadlock).
- `cleanba/attn_lstm.py` — attention core (content-weighted; my edit `out_init_scale`). Deadlocked historically; re-run at ≥80M as 2nd non-conv instance.
- `cleanba/config.py` — `sokoban_drc33_59` (proven DRC recipe), `sokoban_drc_cellwise_3_3{,_sum,_max}`, `sokoban_drc_attn_3_3*`. Cellwise/attn inherit `sokoban_drc33_59`.
- `cleanba/cleanba_impala.py` — train entry (farconf CLI).
- `build_env_pod2.sh` — reproducible from-scratch env build for a fresh pod (uv py3.10 + deps + envpool fork + gym-sokoban + **CUDA jax** + cache).

---
## 4. HOW TO RUN
**Pod 1 (PTY-only proxy):** drive by piping commands to an interactive shell (`ssh host 'cmd'` HANGS):
```
{ printf '%s\n' 'cmd1' 'cmd2' 'exit'; } | ssh -tt -o BatchMode=yes -i ~/.ssh/id_ed25519 2v20rpm3ma1frj-6441223f@ssh.runpod.io 2>&1 | tr -d '\r' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g'
```
**Pod 2 (direct TCP — NORMAL ssh, supports scp/heredocs):**
```
ssh -p 19373 -o BatchMode=yes -i ~/.ssh/id_ed25519 root@157.66.254.33 'commands'
scp -P 19373 -i ~/.ssh/id_ed25519 localfile root@157.66.254.33:/remote/path
```
File transfer to pod1 (no scp): base64 | `fold -w700` | `printf %s 'CHUNK' >> /tmp/f.b64` lines | `base64 -d` | verify sha256.
Launch a run (SOLO, full GPU). pod1 uses tmux; **pod2 has NO tmux** → use `setsid bash driver.sh </dev/null >log 2>&1 &`:
```
.venv/bin/python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:<FN> total_timesteps=80000000 <overrides> base_run_dir=/workspace/claude_experiments/<NAME>
  env: PYTHONUNBUFFERED=1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```
Sequential drivers: pod1 `matrix_driver.sh`/`driver_80m.sh`; pod2 `driver_dense_80m.sh`.

### CRITICAL run rules (each cost hours)
1. **`num_minibatches` = recipe default 8.** Do NOT pass `num_minibatches=4` (silently broke every early run).
2. **SEQUENTIAL per GPU.** Multiple arms on one GPU → host-memory SIGKILL (logs stop, NO traceback). One arm at a time per pod.
3. **CLI bools lowercase:** `net.recurrent.attend_inputs=true` (capital `True` → farconf parse error).
4. **CUDA jax:** `-e .` installs CPU-only `jaxlib` → jax runs on CPU (~85× slow, GPU 0%). Must `uv pip install "jax[cuda12]==0.4.34"`; verify `jax.default_backend()=='gpu'`.
5. **`pkill -f` self-kill (pod2/direct-ssh only):** `ssh host 'pkill -f cleanba_impala'` matches the ssh shell's own argv → kills the session silently. Use bracket trick: `pkill -9 -f '[c]leanba_impala'`.
6. Between runs: kill, then POLL `nvidia-smi` until mem < 3 GB before relaunch (teardown lag → OOM collision).
7. `net.recurrent.n_global=0` (no-pool-and-inject constraint — user requirement).
8. Local Bash: outbound ssh needs `dangerouslyDisableSandbox: true`. Foreground `sleep` is blocked but `sleep` in a `run_in_background` Bash IS fine (used for polling).

### Monitor
- pod1: `bash /workspace/claude_experiments/snap.sh` (reads offline wandb `.wandb` via `wandb.sdk.internal.datastore`; prints per-arm zero_box/success/entropy/var_explained). pod2: `read_eval.py <base_run_dir>` (same reader; snap.sh not copied there).
- Eval metrics are NOT in stdout (only "Evaluating" + timings) — only in the offline wandb binary.

---
## 5. THE METRIC THAT MATTERS
`var_explained`→~0.98 + `losses/entropy` dropping off max (~−1.39 for 4 actions) = learning; stuck `var_explained`~0.4 + entropy pinned = deadlock. Track `valid_medium/NN_episode_{successes,zero_boxes}` over ticks `NN`. **TRAIN `avg_episode_returns` is a RED HERRING** (flat ~−7 even when solving).

---
## 6. RESULTS (valid_medium; success @0 ticks → @32; pool-free unless noted)
| run | arch | mask | AGG | steps | var_expl | succ@0→@32 | note |
|---|---|---|---|---|---|---|---|
| drc_ctrl | conv | local | sum-conv | ~20M | 0.98 | 0.137→0.148 | pooled reference |
| drc_nopool | conv | local | sum-conv | 22M | 0.988 | 0.120→0.135 | pool-free conv works |
| cw_max_ai | GNN | local | **max** | 40M | 0.979 | 0.072→0.077 | **NON-CONV learns + thinking-benefit (key result)** |
| cw_fix | GNN | local | sum/per-offset | 40M | 0.983 | 0.017→0.018 | per_offset = conv-like |
| local_mean_80m | GNN | local | mean | ~64M (→80M) | 0.982 | **0.051→0.062** | mean ALSO learns + thinking-benefit (caught up; was 0.8% mid-ramp) |
| local_sum_80m | GNN | local | sum | queued (pod1) | — | — | |
| dense_max/mean/sum_80m | GNN | dense | max/mean/sum | running/queued (pod2) | — | — | claim-1 (mask) arm |
| local_max_80m | GNN | local | max | **TODO (not yet scheduled)** | — | — | needed for matched-80M max-vs-mean |

**Interpretation so far (PRELIMINARY, pre-80M):** (1) pool-and-inject NOT required; (2) lookahead-PI emerges in a non-conv GNN; (3) `attend_inputs` was the unlock; (4) **max vs mean is closer than first thought** — `local_mean` caught up to ~5–6% as it trained and shows the thinking-time benefit too, so do NOT overclaim "max≫mean" until the matched-80M numbers (`local_max_80m` vs `local_mean_80m`) are in.

---
## 7. RUNNING NOW (2026-06-10 ~21:30Z)
- **Pod 1** (tmux `m80`, driver `/workspace/claude_experiments/logs/driver_80m.log`): `local_mean_80m` (~64M/80M, learning) → then `local_sum_80m`. Its driver ALSO lists `dense_*` — **TODO: those must NOT run on pod1** (pod2 owns dense). When pod1 finishes `local_sum_80m`, kill its driver and instead run **`local_max_80m`** then the **attn re-run**.
- **Pod 2** (`driver_dense_80m.sh`, logs `driver_dense.log`, launched via `setsid`): `dense_max_80m` (running, GPU-confirmed ~6.7k SPS) → `dense_mean_80m` → `dense_sum_80m`, all 80M.
- A ~45-min two-pod monitor was running in the PREVIOUS Claude harness — **it does NOT carry over to a new session.** Re-arm your own polling (see §4 Monitor: `run_in_background` Bash with `sleep` + ssh `snap.sh`/`read_eval.py`). The training runs themselves persist (tmux on pod1, `setsid` on pod2) — only my pollers stop.
- **Most urgent action for the new agent:** watch pod 1; when `local_mean_80m` → `local_sum_80m` finishes, its driver would next start the `dense_*` arms (which pod 2 already owns) — kill the pod1 driver at that point and launch `local_max_80m` then the attn re-run instead.

---
## 8. NEXT EXPERIMENTS (priority order)
1. **Finish the 2×3 matrix at 80M** + add **`local_max_80m`** → the two headline contrasts (mask: local vs dense; AGG: max vs mean/sum). Fill the table + regen `results/plot.py`.
2. **attn_lstm re-run to ≥80M** (user recalls it learned ~50M) — 2nd non-conv instance. FIRST mine old `/workspace/cleanba_runs_attn*` dirs to find which attn config learned. Use `sokoban_drc_attn_3_3*` + correct recipe (num_minibatches 8, n_global=0).
3. **Mechanism probes** on a learning core (`cw_max_ai`/`drc_nopool`): per-cell linear probes for value→V* (Sokoban solver for ground truth), plan-direction (agent/box), `1×1≈3×3` (localization); probe across thinking steps `k` → frontier propagates 1 hop/k; causal interventions (clamp a cell's plan-direction → behavior changes). Template: Bush et al. 2025 (arXiv 2504.01871), Taufeeque et al. 2024 (2407.15421); reuse AlignmentResearch `learned-planner` probe+solver infra.
4. **Non-grid transition graph** (conv can't apply) = strongest non-conv argument.

---
## 9. INFRA / LOCATIONS
- **Pod 1** SSH (PTY proxy): `2v20rpm3ma1frj-6441223f@ssh.runpod.io -i ~/.ssh/id_ed25519`. Pod1's template has the repo + env + cache preinstalled.
- **Pod 2** SSH (direct TCP, supports scp): `ssh -p 19373 -i ~/.ssh/id_ed25519 root@157.66.254.33` (proxy form `hz84eqdtmj6ome-644119d8@ssh.runpod.io` is PTY-only). BLANK `runpod/pytorch:2.4.0-py3.11` image — env built from scratch via `build_env_pod2.sh`.
- repo `/workspace/recurrent-planner` (branch `claude/attn-experiments`); venv `.venv` (jax 0.4.34 + cuda12, py3.10); cache `/opt/sokoban_cache/boxoban-levels-master` (= `git clone github.com/deepmind/boxoban-levels`, 204M).
- experiments `/workspace/claude_experiments/{logs/, results/ (RESULTS.md, plot.py, *.png), snap.sh, read_eval.py, *driver*.sh}` (pod1 has all; pod2 has read_eval.py + drivers, NOT snap.sh/results — scp them if needed).
- gym-sokoban submodule = `github.com/AlignmentResearch/gym-sokoban`; envpool fork wheel = `github.com/AlignmentResearch/envpool/releases/download/v0.1.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl`.
- Key refs: VIN (Tamar 2016, 1602.02867), DRC (Guez 2019, 1901.03559), Sokoban interp (Bush 2504.01871; Taufeeque 2407.15421), attn≈conv (Cordonnier 1911.03584), max-agg DP (Veličković 1910.10593).
- **Keep both pods running** (a RunPod restart wipes `/opt` + uv + venv → full rebuild).
