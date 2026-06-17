"""E5 -- does effective REACH grow with ticks? (iterative deepening / do pulled-from cells get updated?)

E4 showed one tick pulls value ~3 hops (rho_graph~0.66). The open question: with MORE ticks, does a cell
integrate info from FURTHER horizons -- which requires that the cells it pulls from are THEMSELVES updated
(carry forward their own propagated value). Test causally: perturb one floor cell at graph-distance d from
the agent (flip->wall) and measure the agent-cell latent shift ||dh(agent)|| at EACH tick. A single ~3-hop
kernel cannot reach d=8 in one tick; so if far perturbations influence the agent and that influence ARRIVES
LATER for larger d, intermediate cells must relay it across ticks => compounding => answer is YES.

  arrival-tick rises with d  (horizon grows ~k hops/tick)   -> REACH COMPOUNDS: pulled-from cells get updated
  far d (>3) never influences / arrives at tick 1 flat       -> fixed single-tick blur, no deepening

  python -m experiments.interp.e5 --ckpt <cp_dir> --boards 256 --dmax 9 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, AGENT


def main(cp_dir, n_boards, dmax, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs0 = np.asarray(envs.reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    obs_p = obs0.copy(); ag = np.full(B, -1); d_used = np.full(B, -1)
    for b in range(B):
        a = np.where(tiles[b] == AGENT)[0]
        if not len(a):
            continue
        a = int(a[0]); ag[b] = a
        dist = bfs_from([a], tiles[b], H, W)
        floor = np.where(tiles[b] == FLOOR)[0]; floor = floor[np.isfinite(dist[floor])]
        if not len(floor):
            continue
        td = (b % dmax) + 1
        c = int(floor[np.argmin(np.abs(dist[floor] - td))])
        d_used[b] = int(dist[c]); r, cc = divmod(c, W); obs_p[b, :, r, cc] = 0       # flip floor -> wall

    valid = (ag >= 0) & (d_used >= 1)
    h0 = np.asarray(recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs0)), K)[0])   # (K,B,S,C)
    hp = np.asarray(recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs_p)), K)[0])
    infl = np.zeros((B, K))
    for b in range(B):
        if valid[b]:
            infl[b] = np.linalg.norm(hp[:, b, ag[b]] - h0[:, b, ag[b]], axis=-1)            # (K,) per tick

    f = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== E5: REACH vs TICKS (step={step}, boards={int(valid.sum())}, K={K}, D={D}) =====")
    print(f"  perturb a floor cell at graph-dist d from agent; agent-cell ||dh|| per tick.")
    print(f"  {'d':>3} {'n':>4} {'arr':>4} {'final':>7}   normalized onset (||dh||_t / ||dh||_final)")
    arr_d, arr_t, mat = [], [], {}
    for d in range(1, dmax + 1):
        m = valid & (d_used == d)
        if m.sum() < 3:
            continue
        a = infl[m].mean(0)                                                                 # (K,) abs influence
        mat[d] = a
        rel = a / (a[-1] + 1e-9)
        arrival = next((t + 1 for t in range(K) if rel[t] >= 0.5), K)
        arr_d.append(d); arr_t.append(arrival)
        print(f"  {d:>3} {int(m.sum()):>4} {arrival:>4} {a[-1]:7.3f}   {f(rel)}")

    # horizon(tick): largest d whose signal has arrived (>=50% of its own final) by that tick
    print(f"\n  horizon(tick) = max d with signal >=50% arrived:")
    horizon = []
    for t in range(K):
        hd = 0
        for d in mat:
            if mat[d][t] >= 0.5 * (mat[d][-1] + 1e-9):
                hd = max(hd, d)
        horizon.append(hd)
    print(f"     tick:    " + " ".join(f"{t+1:>3}" for t in range(K)))
    print(f"     horizon: " + " ".join(f"{horizon[t]:>3}" for t in range(K)))

    # absolute influence matrix (rows=d, cols=tick) -- does far-d influence appear only late?
    print(f"\n  absolute ||dh(agent)|| by (d x tick):")
    print(f"     {'d/t':>4} " + " ".join(f"{t+1:>5}" for t in range(K)))
    for d in sorted(mat):
        print(f"     {d:>4} " + " ".join(f"{mat[d][t]:5.2f}" for t in range(K)))

    if len(arr_d) >= 3:
        far = [(d, t) for d, t in zip(arr_d, arr_t) if d >= 2]
        dd = np.array([d for d, _ in far]); tt = np.array([t for _, t in far])
        slope = float(np.polyfit(dd, tt, 1)[0])
        hops_per_tick = (1.0 / slope) if slope > 1e-6 else float("inf")
        print(f"\n  arrival-tick vs d slope = {slope:+.2f}  (~{hops_per_tick:.1f} hops/tick); "
              f"horizon {horizon[0]}->{horizon[-1]} over ticks")
        verdict = ("REACH COMPOUNDS over ticks -> pulled-from cells are updated (iterative deepening / multi-step PE)"
                   if (slope > 0.15 or horizon[-1] > horizon[0] + 1) else
                   "reach ~flat -> single wide kernel, little compounding")
        print(f"  --> {verdict}")
    # machine-readable
    print("PLOT_ARRIVAL=" + repr({int(d): int(t) for d, t in zip(arr_d, arr_t)}))
    print("PLOT_HORIZON=" + repr([int(x) for x in horizon]))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--dmax", type=int, default=9); ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.dmax, a.ticks)
