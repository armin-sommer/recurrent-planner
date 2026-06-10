#!/bin/bash
# Build the recurrent-planner env on a fresh RunPod pod (py3.10 venv via uv).
# Mirrors the Makefile `local-install` recipe (minus the nonexistent dev-local extra).
set -e
export PATH=$HOME/.local/bin:$PATH
cd /workspace/recurrent-planner

echo "[build] gym-sokoban submodule"
rm -rf third_party/gym-sokoban
git clone --depth 1 https://github.com/AlignmentResearch/gym-sokoban third_party/gym-sokoban

echo "[build] uv venv (python 3.10)"
uv venv --python 3.10 .venv

echo "[build] requirements.txt (--no-deps; jax is disabled in it, comes via -e .)"
uv pip install --python .venv/bin/python --no-deps -r requirements.txt

echo "[build] editable project + gym-sokoban (pulls jax==0.4.34)"
uv pip install --python .venv/bin/python -e . -e ./third_party/gym-sokoban

echo "[build] envpool 0.8.4 fork wheel (cp310 linux x86_64)"
uv pip install --python .venv/bin/python https://github.com/AlignmentResearch/envpool/releases/download/v0.1.0/envpool-0.8.4-cp310-cp310-linux_x86_64.whl

echo "[build] CUDA jaxlib -- CRITICAL: '-e .' installs a CPU-only jaxlib, so jax silently runs on CPU (~85x slower). This makes jax use the GPU."
uv pip install --python .venv/bin/python "jax[cuda12]==0.4.34"

echo "[build] verify imports + GPU backend"
.venv/bin/python -c 'import jax,envpool,flax,gym_sokoban,rlax,optax; b=jax.default_backend(); print("BUILD_OK", "jax", jax.__version__, "envpool", envpool.__version__, "backend", b); assert b=="gpu", "JAX NOT ON GPU -- check CUDA jaxlib install"'
echo "[build] BUILD_DONE"
