# AGENT HANDOFF — recurrent-planner

_Single source of truth for resuming. Last rewritten 2026-06-11 after training + eval finished and the GPU node was torn down. Read top to bottom. Supersedes the auto-loaded `~/.claude/.../memory/` notes where they conflict (those describe deleted RunPod pods — see §6)._

---

## 0. TL;DR / CURRENT STATE

- **Project thesis:** model-free RL on a *relational recurrent core* induces **planning** when (i) distinct latent cells **bind** to environment states and (ii) the recurrence's routing recovers the transition graph $\mathcal N$. Under those two emergent conditions the "thinking" loop = K-step policy improvement (policy iteration unrolled in latent space). See §1.
- **Status:** training DONE (cellwise mask×aggregation matrix + attention arms, all 200M steps). Post-hoc thinking-step eval DONE. **The 6×H100 node is DELETED.** All weights + eval data are pulled locally; results pushed to GitHub.
- **Headline result:** the masked-vs-dense thesis holds **cleanly in the attention instance** (masked 0.377 vs dense 0.215 valid; masked gets a ~3.6× bigger thinking-time lift); the cellwise core is too weak (~15%) to resolve it behaviourally. See §2.
- **Next:** the decisive test is **mechanistic interpretability** (the behavioural signal is only strong in attention) — run it on a **cheap single GPU** from the pulled weights. See §4.
- **Repo:** branch `recurrent-attention-core` (pushed to `github.com/armin-sommer/recurrent-planner`). Results in `results/`. Latest commits: `73400e1` (this handoff lineage), `764a509` (full results), `2b57a95` (eval config).

---

## 1. GOALS / THE THESIS (revised)

**Central claim.** A generic relational recurrent backbone (Assumption below), trained only by policy gradient + TD, comes to *plan* — not because planning is built in, but because two properties of the *learned representation* emerge:

1. **Binding** — distinct latent cells come to encode distinct environment states (an approx. injective map $\sigma:\mathcal S\to[N]$, cell content a function of that state).
2. **Transition-respecting routing** — the content-based routing connects a cell to exactly the cells of one-step-reachable states, so the *effective graph over cells = the transition graph $\mathcal N$*.

Under (1)+(2), each thinking step is a one-hop Bellman backup of the critic's value field, so K thinking steps = K-step lookahead policy improvement. The loop is the **improvement** half of policy iteration; TD across env-steps is the **evaluation** half. (Formal: `Proposition (propagated policy improvement)` in the paper draft.)

**THE KEY REFRAME (post-results) — do not regress to "masked good / dense fails".** The operative condition is the **binding + effective-graph = $\mathcal N$**, NOT a hand-coded local mask. With attention as the routing, $\mathcal N$ is **recovered by training**: imposed under a hard local mask, **learned** under dense all-to-all attention. **Planning emerges in both** because dense attention *learns to attend to the transition-feasible neighbours*. The mask is a sample-efficiency convenience, not a prerequisite. This is the framing in the rewritten LaTeX theory (binding as the thesis; masked/dense both recover $\mathcal N$).

**The architecture (Assumption — recurrent relational core).** $N$ latent cells $h_t(i)$; each thinking step $k$: $h_t^{k+1}(i)=\bigoplus_j \alpha_\phi(h^k(j),h^k(i))\,F_\phi(h^k(j))$, with $F_\phi$ a content map, $\alpha_\phi$ a learned routing (attention) score, $\oplus$ edge-invariant aggregation. Time-invariant, weight-tied across cells, permutation-equivariant → the binding $\sigma$ must be *learned* (symmetry-breaking). Masked routing = fixed support; dense = unconstrained.

**What mech-interp must SHOW (the paper's empirical section):** in an attention agent — (i) cells bind to states; (ii) the learned attention recovers $\mathcal N$ (attends to transition-feasible neighbours) **in both the masked and dense cases**; (iii) the plan field propagates one hop/tick = propagated policy improvement. These are predictions of the theory, not surprises.

---

## 2. EMPIRICAL RESULTS (what we have)

**Setup.** Sokoban (Boxoban, 10×10), `cleanba` IMPALA (JAX/flax). Two relational cores: cellwise message-passing (`cleanba/cellwise_lstm.py`) and attention (`cleanba/attn_lstm.py`). Trained the cellwise mask×aggregation matrix + the attention arms, all to **200M steps**. Eval = post-hoc **thinking-step sweep (extra ticks 0→12)** via `cleanba.load_and_eval` on **valid_medium** (held-out, medium difficulty) and **train_unfiltered** (train distribution), 1000 levels/tick. Data: `results/data/thinking.csv`; figures `results/figures/`; write-up `results/RESULTS.md`.

**Arms.** cellwise: `local_{max,mean,sum}` + `dense_{max,mean,sum}` (local = king-mask, dense = all-to-all). attention: `attn_{masked,dense,softmax,plain,vn,shallow}` (`sokoban_drc_attn_3_3{,_nomask,_softmax,_plain,_vn}`, `_1_1`).

**valid_medium success (@0 thinking / thinking-lift @0→12):**

| arm | @0 | lift | note |
|---|---|---|---|
| attn_softmax (masked) | 0.458 | +0.049 | strongest planner |
| attn_plain (**masked**) | 0.377 | +0.047 | clean mask test (masked side) |
| attn_vn (masked, vonN) | 0.319 | +0.049 | |
| **attn_dense** (dense) | 0.215 | +0.013 | clean mask test (dense side) |
| attn_shallow (depth-1) | 0.183 | +0.064 | |
| local_mean | 0.163 | +0.004 | cellwise weak |
| local_sum | 0.151 | +0.021 | |
| dense_mean | 0.142 | +0.026 | |
| local_max (120M peak) | 0.130 | +0.003 | |
| dense_max | 0.105 | +0.009 | |
| dense_sum | 0.000 | — | DIVERGES |

**Findings (honest):**
1. **Attention ≫ cellwise** (~38–46% vs ~13–16% valid). The cellwise message-passing core is a weak planner at 200M.
2. **Mask claim holds cleanly in attention.** Matched contrast `attn_plain` (masked) vs `attn_dense` (dense) — identical except the mask: **0.377 vs 0.215 (+16 pp)**, masked thinking-lift **+4.7 pp vs +1.3 pp (~3.6×)**. (Under the §1 reframe, this is "masked recovers $\mathcal N$ for free; dense must learn it" — mech-interp should show dense *does* learn it; the gap is sample-efficiency.)
3. **Cellwise signal is weak/noisy.** local ≥ dense modestly; thinking-lifts ~±1–3 pp (≈1.5σ at 1000 levels), not cleanly local-favouring. **`dense_sum` diverges** (unnormalized sum over ~100 cells blows up; masked `local_sum` is fine — clean mask-matters-for-sum point).
4. **Large train→valid gap** (~80% → ~15% cellwise): agents fit the training distribution, generalize modestly.
5. **Two max-aggregation instabilities (supporting a sparse-argmax-gradient story):** cellwise `local_max` **oscillated** in late training (use its **120M peak**, not the 200M final); attention `attn_masked` (king + maxplus + directional) **hard-collapsed at ~50M** (var_explained went negative — abandoned). Their softmax / von-Neumann / plain ablations are fine → hard max over many competing neighbours starves the gradient; softening (softmax) or fewer neighbours fixes it.

---

## 3. WHERE THE DATA IS (for mech-interp; node is gone)

- **All weights:** `results/checkpoints/all_arms/all_ckpts.tar` (3.7 GB, **274 checkpoints**, all 12 arms incl. full training ladders, `local_max` peak, and the collapsed `attn_masked`). Gitignored (`*.tar`, `results/checkpoints/`). Each ckpt = `cp_<step>/{model, cfg.json}`, load with `cleanba.cleanba_impala.load_train_state(cp_dir, env_cfg)` (JAX/flax).
- **`local_max` peak** also extracted at `results/checkpoints/local_max_peak/` (cp 110/120/130M).
- **Eval pkls:** `results/data/eval_pkls.tar` (22 metric dicts; each has `NN_episode_successes/zero_boxes/num_noops_per_eps/...` for NN=00–12, + `NN_all_episode_info`). Parsed scalars in `results/data/thinking.csv`. Pipeline: `results/parse_pkls.py` (pkls→CSV), `results/plot.py` (CSV→figures).
- **Repo / configs / cores:** branch `recurrent-attention-core`. Configs `cleanba/config.py` (`sokoban_drc_cellwise_*`, `sokoban_drc_attn_*`; eval `planning_eval_envs` in `cleanba/load_and_eval.py`). Cores `cleanba/cellwise_lstm.py`, `cleanba/attn_lstm.py` (has `use_attention_mask`, `readout∈{maxplus,softmax}`, `directional_value`, `relative_key`, `n_global`).
- **learned-planner interp toolkit:** cloned at `/Users/arminsommer/Downloads/learned-planner-ref` (and was on the node). `learned_planner/interp/`: `collect_dataset.py`, `save_ds.py`, `train_probes.py`, **`act_patch_utils.py`** (causal patching), **`value_mse.py`** (value probe), `offset_fns.py`, `channel_group.py`. Probe concepts (`--labels_type`): `agents_future_position_map`, `agents_future_direction_map`, `boxes_future_direction_map`, `next_box`, `next_target`, `true_value` (γ). **Ground-truth = A\* solutions, HF dataset `AlignmentResearch/boxoban-astar-solutions`** (no solver to write). NB their `load_jax_model_to_torch` is ConvLSTM-specific → we extract activations from our JAX cores directly and reuse their labels + probe/patch code.

---

## 4. MECH-INTERP PLAN (the decisive test)

Behavioural thinking-scaling is strong only in attention; the **mechanism** probes are the real local-vs-dense (and masked-vs-dense-routing) test. **Arms:** local = `local_{max-peak,mean,sum}` + `attn_plain`; dense = `dense_{max,mean,sum}` + `attn_dense`. (`attn_masked` excluded — collapsed.)

**Central question:** does masked compute a localized, propagating plan/value frontier — and (per §1) does **dense attention recover the same $\mathcal N$ and plan the same way**?

**Tiers (run in order):**
1. **Behavioural** (cheap, partly in hand): thinking-scaling (done, `results/`); **pacing** (`num_noops_per_eps` in the pkls — does the agent cycle to buy compute? Taufeeque); **OOD board-size** (remap the king-kernel — `edge_logits`/`rel_bias` are size-specific, `(2H-1)(2W-1)` — onto larger `SokobanConfig` boards; local should transfer, dense is size-locked).
2. **Concept probes** (decodability): linear probes per cell for agent future path/direction, box push-direction, next box/target, value/V\* (A\* ground-truth), via `save_ds`/`train_probes`. **Build JAX per-cell/per-tick activation extraction for our cores** (hook the `cellwise_lstm`/`attn_lstm` forward). Compare decodability: local vs dense, cellwise vs attn.
3. **Frontier propagation (HEADLINE):** re-probe the concept maps at *each thinking tick* — does the confident/solved region grow ~1 hop/tick (value-iteration / Bush's bidirectional search)? **And: does attention recover $\mathcal N$ (attend to transition-feasible neighbours) in masked AND dense?** (the §1 reframe's empirical claim.)
4. **Causal patching** (`act_patch_utils`): edit a cell's plan (flip push-direction / clamp value) → behaviour changes, locally → proves the plan is used.
5. **Cross-architecture:** cellwise-local vs attn-local share the algorithm? Anchor against the DRC findings in the papers.

**Anchor papers:** Bush et al. 2025 (arXiv 2504.01871) — concept probes, "parallelized bidirectional search", activation patching. Taufeeque et al. 2024 (arXiv 2407.15421) — causal plan ~50 steps ahead, pacing, OOD generalization.

---

## 5. HOW TO RUN (next session)

**Mech-interp env (cheap single GPU — NOT 6×H100):** clone `recurrent-attention-core` + `AlignmentResearch/learned-planner`. Build env (see `build_env_pod2.sh` pattern): `uv venv --python 3.10`, `uv pip install "jax[cuda12]==0.4.34"`, `--no-deps -r requirements.txt`, the envpool cp310 fork wheel (`github.com/AlignmentResearch/envpool/releases/download/v0.1.0/...`), `-e . -e ./third_party/gym-sokoban`; boxoban cache → `/opt/sokoban_cache/boxoban-levels-master` (`github.com/deepmind/boxoban-levels`). Upload + extract `all_ckpts.tar`. Verify `jax.default_backend()=='gpu'`.

**Eval re-run (if needed):** `cleanba.load_and_eval load_other_run=<dir-with-ONE-cp> only_last_checkpoint=true` — **use an isolated dir holding a single `cp_*`** (avoids the eval's checkpoint-finder bug: wandb makes `latest-run`+`develop` symlinks and restarted arms have a 2nd empty `offline-run` dir, both of which break `recursive_find_checkpoint`'s `assert len(parents)==1`). Run **low concurrency + `XLA_PYTHON_CLIENT_MEM_FRACTION` cap + stagger launches** (6 simultaneous evals thrash the CPU, loadavg→891; the eval loads 2×500-env sets per proc). The `results/parse_pkls.py` + `plot.py` are the parse/plot pipeline.

**Recipe gotchas:** `net.recurrent.n_global=0` (no pool-and-inject, a project constraint); CLI bools lowercase (`...=true`); `num_minibatches=8`; `dense_sum` needs `zero_init_message=true` or it diverges. The DRC `fence_pad` (boundary marker) is NOT in our cores — we use plain padding at the boundary; cellwise `mean` row-normalizes at the edge while `sum`/per-offset zero-pad like conv-SAME (a potential future ablation: make `mean` conv-faithful + add boundary padding — discussed, deprioritized).

---

## 6. INFRA STATUS

- **ALL RunPod nodes from this project are DELETED.** Do NOT ssh `157.66.254.38` (the 6×H100) or the old proxies / `157.66.254.33` — the memory notes `runpod-ssh-access` / `runpod-training-setup` are STALE. Everything needed is pulled locally (§3) + on GitHub.
- Mech-interp is checkpoint-based → spin a fresh cheap GPU when ready.

## 7. REFERENCES
VIN (Tamar 2016, 1602.02867); DRC (Guez 2019, 1901.03559); Sokoban interp (Bush 2504.01871, Taufeeque 2407.15421); attn≈conv (Cordonnier 1911.03584); max-agg DP (Veličković 1910.10593); multi-step policy improvement (Efroni 2018).
