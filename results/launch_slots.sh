#!/bin/bash
# Launch the slot-core seed x routing sweep on the H200. 3 seeds x {softmax, maxplus} = 6 runs,
# each in its own tmux session, 200M steps (matches the attn arms), distinct run dirs, GPU-mem capped
# so all 6 share the one H200. Pure RL (no aux loss). Run once: bash launch_slots.sh
set -u
cd /workspace/recurrent-planner
mkdir -p /workspace/logs /workspace/runs /training/cleanba
MF=0.12                 # XLA mem fraction per run (6 * 0.12 = 0.72 < 1.0; model is tiny)
TS=200000000            # 200M timesteps, matching the attn/cellwise arms

launch () {  # $1=config fn  $2=seed  $3=session/name
  tmux kill-session -t "$3" 2>/dev/null || true
  tmux new-session -d -s "$3" \
    "cd /workspace/recurrent-planner && PYTHONUNBUFFERED=1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=$MF \
     .venv/bin/python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:$1 \
     seed=$2 total_timesteps=$TS base_run_dir=/workspace/runs/$3 > /workspace/logs/$3.log 2>&1"
  echo "launched $3 ($1 seed=$2)"
}

launch sokoban_drc_slots_softmax 0 s_sm_0
launch sokoban_drc_slots_softmax 1 s_sm_1
launch sokoban_drc_slots_softmax 2 s_sm_2
launch sokoban_drc_slots_maxplus 0 s_mp_0
launch sokoban_drc_slots_maxplus 1 s_mp_1
launch sokoban_drc_slots_maxplus 2 s_mp_2

tmux ls
