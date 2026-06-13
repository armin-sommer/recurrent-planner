"""Axis-1 thinking-curve for a variable-thinking-depth model.

The eval harness's `steps_to_think` is axis-2 (extra recurrence applications on the INITIAL obs).
Our vardepth model was trained varying the INNER depth (`n_active`, the gated thinking-tick count)
per step, so the faithful test sweeps `n_active` and runs the WHOLE episode at that depth.

Trick: bake `n_active=v` into the action fn and set `steps_to_think=[0]` — then EvalConfig.run()'s
exact solve-detection/metrics machinery measures solve rate at inner depth v, with no harness edits.

Usage (node):  python -m results.thinking_curve_vardepth --ckpt <cp_dir> [--depths 0,1,2,3,4,5,6]
"""
from __future__ import annotations
import argparse, dataclasses
from functools import partial
from pathlib import Path

import jax
import numpy as np


def main(cp_dir: str, depths):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs

    evalset = planning_eval_envs()                      # valid_medium + train_unfiltered, 500x2=1000 levels
    env_cfg = evalset["valid_medium"].env
    policy, _, cp_cfg, train_state, _ = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = train_state.params

    print(f"loaded {cp_dir}")
    print("depth |  valid_medium  |  train_unfiltered   (solve rate; whole episode run at inner depth d)")
    rows = {}
    for name in ["valid_medium", "train_unfiltered"]:
        base = dataclasses.replace(evalset[name], steps_to_think=[0])   # axis-2 off; sweep axis-1 below
        rows[name] = {}
        for v in depths:
            # bake the inner thinking-depth budget into get_action; whole episode runs at depth v
            gaf = jax.jit(
                partial(policy.apply, method=policy.get_action, n_active=int(v)),
                static_argnames="temperature",
            )
            m = base.run(policy, gaf, params, key=jax.random.PRNGKey(1234))
            rows[name][v] = float(m["00_episode_successes"])

    for v in depths:
        print(f"  d={v}:    {rows['valid_medium'][v]:.3f}          {rows['train_unfiltered'][v]:.3f}")
    vm = rows["valid_medium"]
    base_d, best_d = vm[depths[0]], max(vm.values())
    print(f"\nvalid_medium thinking-lift: d={depths[0]}={base_d:.3f} -> best={best_d:.3f}  (+{best_d-base_d:.3f})")
    print("=> thinking HELPS" if best_d - base_d > 0.01 else "=> thinking flat/no benefit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depths", default="0,1,2,3,4,5,6")
    a = ap.parse_args()
    main(a.ckpt, [int(x) for x in a.depths.split(",")])
