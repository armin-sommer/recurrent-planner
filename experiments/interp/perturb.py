"""DECISIVE search test: causal influence-radius vs thinking ticks (D=3 entmax core).

If the thinking loop does iterative graph propagation (search), a perturbation to a cell at graph-distance
d from the agent should reach the AGENT's latent only after ~d ticks (information travels ~1 hop/tick).
If instead distant info is globally available (non-local pathway / "just a deep net"), the agent latent
is perturbed immediately (tick 1) regardless of d.

Method: take real boards; for each, flip one FLOOR cell at BFS-distance d (over the navigable graph) from
the agent to a WALL (a clean, strongly-bound env-state change). Recompute the K thinking ticks on the
original and perturbed boards (faithful recompute_d3) and measure, per tick, the agent-cell latent shift
||h_pert[k,agent] - h_orig[k,agent]||. Bin by d; normalize each bin by its final-tick value to get the
ONSET shape; the arrival tick (first k reaching 50% of final) vs d is the signature:
  arrival ~ d (slope ~1)  => finite-speed propagation = SEARCH.
  arrival ~ 1 (flat)      => immediate/global = NOT search.
NB: the conv embed has a ~7x7 receptive field, so for d>=5 the agent's embed is UNAFFECTED by the flip --
any agent-cell influence there must arrive via attention propagation. Focus the verdict on d>=5.

  python -m experiments.interp.perturb --ckpt <cp_dir> --boards 256 --dmax 9
"""
from __future__ import annotations

import argparse
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, AGENT


def main(cp_dir, n_boards, dmax):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)        # (B,3,H,W)
    B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)                                                    # (B,S)

    # pick, per board, a floor cell at a target BFS-distance d from the agent (cycle d=1..dmax)
    obs_pert = obs0.copy()
    agent_sq = np.full(B, -1); d_used = np.full(B, -1); cell_used = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]
        if not len(ag):
            continue
        a = int(ag[0]); agent_sq[b] = a
        dist = bfs_from([a], tiles[b], H, W)                                      # nav distance from agent
        target_d = (b % dmax) + 1
        floor = np.where(tiles[b] == FLOOR)[0]
        floor = floor[np.isfinite(dist[floor])]
        if not len(floor):
            continue
        c = int(floor[np.argmin(np.abs(dist[floor] - target_d))])                # floor cell nearest target_d
        cell_used[b] = c; d_used[b] = int(dist[c])
        r, cc = divmod(c, W)
        obs_pert[b, :, r, cc] = 0                                                 # flip to wall (black)

    valid = (agent_sq >= 0) & (cell_used >= 0)
    embed0 = get_embed(policy, params, jnp.asarray(obs0))
    embedp = get_embed(policy, params, jnp.asarray(obs_pert))
    h0, _ = recompute_d3([params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)], embed0, K)
    hp, _ = recompute_d3([params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)], embedp, K)
    h0 = np.asarray(h0); hp = np.asarray(hp)                                      # (K,B,S,C)

    infl_agent = np.zeros((B, K)); infl_cell = np.zeros((B, K))
    for b in range(B):
        if not valid[b]:
            continue
        infl_agent[b] = np.linalg.norm(hp[:, b, agent_sq[b]] - h0[:, b, agent_sq[b]], axis=-1)
        infl_cell[b] = np.linalg.norm(hp[:, b, cell_used[b]] - h0[:, b, cell_used[b]], axis=-1)

    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print(f"\n===== CAUSAL INFLUENCE-RADIUS vs TICKS (step={step}, boards={int(valid.sum())}, K={K}, D={D}) =====")
    print("  flip one floor->wall at BFS-distance d from agent; measure agent-cell latent shift per tick.")
    print(f"  {'d':>3} {'n':>4}  {'arrival':>7}   agent-cell ||Δh|| normalized to final tick (onset shape)")
    arr_d, arr_t = [], []
    for d in range(1, dmax + 1):
        m = valid & (d_used == d)
        if m.sum() < 3:
            continue
        a = infl_agent[m].mean(0)                                                 # (K,) mean influence at tick k
        rel = a / (a[-1] + 1e-9)
        arrival = next((k + 1 for k in range(K) if rel[k] >= 0.5), K)             # first tick reaching 50% of final
        arr_d.append(d); arr_t.append(arrival)
        print(f"  {d:>3} {int(m.sum()):>4}  {arrival:>7}   {f(rel)}   (final ||Δh||={a[-1]:.3f})")
    # sanity: perturbed cell registers immediately (tick-1 local binding); slope of arrival vs d (d>=5 clean)
    far = [(d, t) for d, t in zip(arr_d, arr_t) if d >= 5]
    if len(far) >= 2:
        dd = np.array([d for d, _ in far]); tt = np.array([t for _, t in far])
        slope = float(np.polyfit(dd, tt, 1)[0])
    else:
        slope = float("nan")
    cell_t1 = infl_cell[valid].mean(0)[0]
    print(f"  perturbed-cell ||Δh|| @tick1 (should be large = local registers now) : {cell_t1:.3f}")
    print(f"  arrival-tick vs d slope (d>=5, clean of conv RF) : {slope:.2f}  "
          f"({'~1/hop => PROPAGATION/SEARCH' if slope > 0.4 else 'flat => immediate/global, NOT search'})")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--dmax", type=int, default=9)
    a = ap.parse_args()
    main(a.ckpt, a.boards, a.dmax)
