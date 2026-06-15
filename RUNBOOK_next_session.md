# Runbook — push this session's work + run the Bellman test

All files below are **already saved locally** (uncommitted). Bash was blocked this session by a
transient auto-mode safety-classifier outage; run these in a session where Bash works. Easiest path:
open a fresh Claude session and say *"commit + push the working tree and run the Bellman test on the
node"* — it will do all of this. Manual commands below as backup.

## 1. Commit + push (local)
```bash
cd ~/Downloads/recurrent-planner
git add AGENT_HANDOFF.md RUNBOOK_next_session.md \
  writeup/planning_emergence.tex writeup/planning_emergence.pdf \
  results/interp_planning_d3.py results/interp_search_d3.py results/interp_perturb_d3.py \
  results/interp_topo_d3.py results/interp_wall_d3.py results/interp_lookahead_d3.py \
  results/interp_planq_d3.py results/interp_bellman_d3.py
# do NOT add results/checkpoints/local_max_peak/  (42MB stale checkpoints, intentionally excluded)
git commit -m "Interp suite + 3-step planning writeup; handoff rewrite"
git push origin recurrent-attention-core
```

## 2. Run the Bellman self-consistency test on the node
Get `results/interp_bellman_d3.py` onto the node (a fresh Claude session base64-deploys it; or, after
the push above, `cd /workspace/recurrent-planner && git stash && git pull` if the node tracks origin),
then:
```bash
ssh kz362uma1m94cz-64412132@ssh.runpod.io -i ~/.ssh/id_ed25519
CK=$(ls -dv /workspace/runs/vd_d3_entmax/default/wandb/offline-run-*/local-files/cp_* | tail -1)
tmux new-session -d -s bell "cd /workspace/recurrent-planner && PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  .venv/bin/python -m results.interp_bellman_d3 --ckpt $CK --boards 128 --depth 4 \
  > /workspace/logs/ib.log 2>&1"
# watch:  tail -f /workspace/logs/ib.log   (wait for "BELLMAN SELF-CONSISTENCY")
```
**Interpretation** — optimality residual `|V(s) - max_a(r_a + γ·V(s'_a))|` at depths 0..3:
- **low & flat across depth** → V is a value-iteration fixed point at every state ⇒ the weight-tied
  backup recurses (successors also back up) ⇒ **multi-step value propagation** (not depth-1).
- **low at depth 0, growing with depth** → fixed point only near the agent ⇒ **bounded** effective
  lookahead depth (≈ the depth where the residual blows up).

## 3. (Optional) node cleanup — frees disk, keeps the run we analyze
```bash
ssh kz362uma1m94cz-64412132@ssh.runpod.io -i ~/.ssh/id_ed25519
rm -rf /workspace/runs/{vardepth600,vardepth,smoke,smoke2,s_sm_0,s_sm_1,s_sm_2,s_mp_0,s_mp_1,s_mp_2}
# keep: /workspace/runs/vd_d3_entmax (+ checkpoints) and /workspace/saved_ckpts
```

## Context
See `AGENT_HANDOFF.md` for the full state. One-line: dense D=3 entmax core, 300M, trained; shown
**binding ✓ → transition-respecting routing ✓ (emergent, causal through-walls) → value-propagation
planning ✓**. The Bellman test (step 2 above) pins whether the planning is multi-step (recursive) or
bounded-depth.
