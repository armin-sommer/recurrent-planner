"""TOPOLOGY test: does influence propagate along the navigable GRAPH (around walls) = relational search,
or through EUCLIDEAN space (through walls) = spatial smoothing?  (D=3 entmax core.)

Distinguishes parallel/relational search from distance-decaying spatial mixing -- both are distance-
dependent, so the perturbation onset alone can't separate them; the topology can. For each board we flip
one floor cell to a wall and measure the agent-cell latent ONSET (center-of-mass tick of the influence,
lower=faster). We record BOTH the navigable graph-distance (BFS around walls) and the Euclidean (pixel)
distance from the agent to that cell, then ask which one governs the onset (standardized regression +
binned curves). If onset tracks GRAPH-distance controlling for Euclidean -> propagation respects 𝒩 =
relational search over the bound states. If it tracks EUCLIDEAN -> spatial smoothing.

We deliberately sample cells that are "walled off" (graph >> euclid) as well as "open" (graph≈euclid) so
the two distances are decorrelated enough to separate. Also reports the influence-vs-graph-distance onset
matrix (the propagation LAW: ~d serial, ~log d parallel-doubling, or flat global).

  python -m experiments.interp.topo --ckpt <cp_dir> --boards 256
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


def main(cp_dir, n_boards):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    rng = np.random.default_rng(0)

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)
    B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    obs_pert = obs0.copy()
    agent_sq = np.full(B, -1); cell = np.full(B, -1); gd = np.full(B, np.nan); ed = np.full(B, np.nan)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]
        if not len(ag):
            continue
        a = int(ag[0]); agent_sq[b] = a; ar, ac = divmod(a, W)
        dist = bfs_from([a], tiles[b], H, W)                                       # graph dist around walls
        floor = np.where(tiles[b] == FLOOR)[0]; floor = floor[np.isfinite(dist[floor])]
        if len(floor) < 2:
            continue
        rr, cc = floor // W, floor % W
        euc = np.hypot(rr - ar, cc - ac)
        gdist = dist[floor]
        # half the boards: maximize graph-minus-euclid (walled-off); half: open (graph≈euclid). Both span euclid.
        score = (gdist - euc) if (b % 2 == 0) else -np.abs(gdist - euc)
        valid_pick = euc >= 2.5                                                    # beyond conv RF
        if valid_pick.any():
            cand = np.where(valid_pick)[0]
        else:
            cand = np.arange(len(floor))
        c = int(floor[cand[np.argmax(score[cand])]])
        cell[b] = c; gd[b] = float(dist[c]); ed[b] = float(euc[np.where(floor == c)[0][0]])
        r, cq = divmod(c, W); obs_pert[b, :, r, cq] = 0

    valid = (agent_sq >= 0) & (cell >= 0) & np.isfinite(gd) & np.isfinite(ed)
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    h0, _ = recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs0)), K)
    hp, _ = recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs_pert)), K)
    h0 = np.asarray(h0); hp = np.asarray(hp)

    centroid = np.full(B, np.nan)                                                  # onset center-of-mass tick
    for b in range(B):
        if not valid[b]:
            continue
        a = np.linalg.norm(hp[:, b, agent_sq[b]] - h0[:, b, agent_sq[b]], axis=-1)  # (K,)
        rel = a / (a[-1] + 1e-9)
        inc = np.diff(np.concatenate([[0.0], rel]))                                 # per-tick arrival increments
        centroid[b] = float(np.sum(np.arange(1, K + 1) * inc) / (inc.sum() + 1e-9))

    v = valid & np.isfinite(centroid)
    g = gd[v]; e = ed[v]; y = centroid[v]
    zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    Z = np.stack([zs(g), zs(e), np.ones_like(g)], 1)
    beta, *_ = np.linalg.lstsq(Z, zs(y), rcond=None)                               # standardized partial effects
    cg = float(np.corrcoef(g, y)[0, 1]); ce = float(np.corrcoef(e, y)[0, 1])
    cge = float(np.corrcoef(g, e)[0, 1])

    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print(f"\n===== TOPOLOGY: graph vs Euclidean propagation (step={step}, boards={int(v.sum())}, K={K}, D={D}) =====")
    print(f"  onset center-of-mass tick (lower=faster arrival at agent). corr(graph,euclid)={cge:.2f}")
    print(f"  standardized partial effect on onset:  GRAPH-dist beta={beta[0]:+.3f}   EUCLID-dist beta={beta[1]:+.3f}")
    print(f"  simple corr(onset, graph)={cg:+.2f}   corr(onset, euclid)={ce:+.2f}")
    winner = ("GRAPH (respects walls) => RELATIONAL SEARCH" if beta[0] > beta[1] + 0.1
              else "EUCLIDEAN (through walls) => spatial smoothing" if beta[1] > beta[0] + 0.1
              else "AMBIGUOUS (graph≈euclid effect)")
    print(f"  --> onset is governed by: {winner}")
    # onset vs graph-distance (the propagation law) and vs euclid, binned
    print("  -- onset center-of-mass by distance bin (n>=4) --")
    for label, dvar in [("graph", g), ("euclid", e)]:
        row = []
        for lo in range(1, 11, 2):
            m = (dvar >= lo) & (dvar < lo + 2)
            row.append((lo, int(m.sum()), float(y[m].mean()) if m.sum() >= 4 else np.nan))
        s = "  ".join(f"d{lo}-{lo+1}:{v_:.1f}(n{n})" for lo, n, v_ in row if n >= 4)
        print(f"     by {label:<6}: {s}")
    # walled vs open contrast (matched-ish euclid)
    wmask = (g / (e + 1e-9) > 1.6); omask = (g / (e + 1e-9) < 1.2)
    if wmask.sum() >= 4 and omask.sum() >= 4:
        print(f"  -- walled (graph/euclid>1.6, n={int(wmask.sum())}) vs open (<1.2, n={int(omask.sum())}) --")
        print(f"     onset walled={y[wmask].mean():.2f}  open={y[omask].mean():.2f}  "
              f"euclid walled={e[wmask].mean():.1f} open={e[omask].mean():.1f}  "
              f"({'walled SLOWER => respects walls' if y[wmask].mean() > y[omask].mean() + 0.15 else 'similar => Euclidean-ish'})")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args()
    main(a.ckpt, a.boards)
