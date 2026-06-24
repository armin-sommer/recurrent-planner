#!/usr/bin/env bash
# MiniWorld pod setup (RunPod, Ubuntu container, uv venv at /workspace/recurrent-planner/.venv).
#
# Re-run this after a pod restart: /workspace (network volume) keeps the venv + the miniworld/pyglet pip
# installs, but the CONTAINER apt packages (GL/EGL loaders, xvfb) are wiped on restart and must be reinstalled.
#
# Why these: MiniWorld renders via pyglet 1.5.x (legacy OpenGL). NVIDIA EGL renders fine in the MAIN
# process but FAILS in spawned AsyncVectorEnv workers (eglChooseConfig NoSuchConfig). The working
# multiprocess path is a shared xvfb virtual X display (X11), so the run is launched under `xvfb-run`.
#   - libegl1/libglu1-mesa/freeglut3-dev/mesa: the GL/EGL loaders pyglet 1.5 needs.
#   - xvfb: the virtual X server for headless multiprocess GL (~3.7k env-steps/s on MazeS3Fast).
# numpy 1.26.4 / gymnasium 0.29.1 MUST stay pinned (cleanba needs gym.vector.make), so miniworld is
# installed --no-deps and pyglet pinned to 1.5.31 (pyglet 2.x removed the legacy glBegin miniworld uses).
set -euo pipefail
REPO=/workspace/recurrent-planner
UV=/root/.local/bin/uv

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libgles2 libopengl0 libglvnd0 libglu1-mesa freeglut3-dev xvfb

# pip installs persist on the network volume; (re)install only if missing.
if ! "$REPO/.venv/bin/python" -c "import miniworld" 2>/dev/null; then
  VIRTUAL_ENV="$REPO/.venv" "$UV" pip install --no-deps miniworld
fi
VIRTUAL_ENV="$REPO/.venv" "$UV" pip install "pyglet==1.5.31"

echo "=== verify ==="
"$REPO/.venv/bin/python" -c "import gymnasium,numpy; print('gym',gymnasium.__version__,'numpy',numpy.__version__)"
cd "$REPO" && PYGLET_HEADLESS=1 .venv/bin/python - <<'PY'
import pyglet; pyglet.options['headless']=True
import gymnasium as gym, miniworld
e=gym.make('MiniWorld-MazeS3Fast-v0',view='agent',render_mode=None); e.reset(seed=0); e.step(0)
print('headless render OK')
PY

cat <<'EOF'

=== launch (single H100; xvfb-run gives the env workers a shared X display; JAX uses the GPU) ===
cd /workspace/recurrent-planner && source .venv/bin/activate
export WANDB_MODE=offline
setsid bash -c 'xvfb-run -a -s "-screen 0 1024x768x24" \
  python -m cleanba.cleanba_impala \
  --from-py-fn=cleanba.config:miniworld_poolinject_d3_fixed4 \
  base_run_dir=/workspace/cb-miniworld > /workspace/mw.log 2>&1' < /dev/null & disown
EOF
