"""E4 -- WHERE does each cell pull value FROM? Characterize the learned attention operator A. (D=3 entmax.)

The per-tick update is z=(A v)W_out: cell q aggregates neighbour value v weighted by its attention row
A[q,:]. So A[q,:] literally says "where q pulls value from." We know A is broad (~85 keys), NOT just the
1-hop neighbours -- so what stencil did RL learn? We characterize A (top cell, settled tick, head-avg) by:

  (1) mass vs GRAPH distance d(q,k) over N (BFS around walls): does mass decay ~rho^d? to how many hops?
      fraction of mass beyond 1-hop (the "not immediate neighbours" share); mass leaking to graph-
      UNREACHABLE cells (through-wall leakage).
  (2) mass vs EUCLIDEAN distance (contrast: graph- vs pixel-locality).
  (3) GOAL-WARD asymmetry: does q pull from keys CLOSER to the goal (lower BFS-to-target = higher value,
      the value source direction) more than from farther keys? (real value propagation pulls value
      from toward the reward.)
  (4) LANDMARK mass: on the target cell, the agent cell, and self.

This says what algorithm the loop implements: 1-hop Jacobi PE (mass at d=1 only) vs a broad multi-hop
goal-ward diffusion kernel (geometric decay, biased toward the goal) vs a global broadcast-from-source.

  python -m experiments.interp.e4 --ckpt <cp_dir> --boards 128 --tick -1
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
from collections import deque
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def all_pairs_graph_dist(tiles_b, H, W):
    """(S,S) BFS distance over non-wall cells; inf if unreachable, nan for wall sources."""
    S = H * W
    D = np.full((S, S), np.inf)
    nav = tiles_b != WALL
    for src in range(S):
        if not nav[src]:
            D[src] = np.nan; continue
        d = np.full(S, np.inf); d[src] = 0; q = deque([src])
        while q:
            s = q.popleft(); r, c = divmod(s, W)
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    t = nr * W + nc
                    if nav[t] and d[t] == np.inf:
                        d[t] = d[s] + 1; q.append(t)
        D[src] = d
    return D


def main(cp_dir, n_boards, tick):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs = np.asarray(envs.reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)
    emb = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs))))
    attn = recompute_d3(cps, emb, K)[1]                                   # (K,D,B,nh,S,S)
    A = np.asarray(attn[tick, D - 1].mean(1))                             # head-avg top-cell, settled tick (B,S,S)

    rc = np.arange(S); RR, CC = rc // W, rc % W
    maxd = 12
    gmass = np.zeros(maxd + 1); gcnt = np.zeros(maxd + 1)                 # mass by graph-dist shell (0..maxd)
    unreach = []                                                          # mass to graph-unreachable keys
    emass = np.zeros(maxd + 1); ecnt = np.zeros(maxd + 1)                 # mass by euclid-dist shell (rounded)
    toward = []; away = []; equal = []                                    # goal-ward decomposition
    m_self = []; m_target = []; m_agent = []; beyond1 = []
    rho_rows = []

    for b in range(B):
        nav = tiles[b] != WALL
        dG = all_pairs_graph_dist(tiles[b], H, W)                          # (S,S)
        tgt = np.where(tiles[b] == TARGET)[0]; agt = np.where(tiles[b] == AGENT)[0]
        dT = bfs_from(tgt, tiles[b], H, W) if len(tgt) else np.full(S, np.nan)
        for q in range(S):
            if not nav[q]:
                continue
            w = A[b, q]                                                    # (S,) attention row, sums to 1
            dq = dG[q]                                                     # graph dist q->k
            fin = np.isfinite(dq)
            unreach.append(float(w[~fin & nav].sum()))                     # mass to nav-but-unreachable (through-wall)
            for d in range(maxd + 1):
                sel = fin & (dq == d)
                if sel.any():
                    gmass[d] += w[sel].sum(); gcnt[d] += 1
            eu = np.round(np.hypot(RR - RR[q], CC - CC[q])).astype(int)
            for d in range(maxd + 1):
                sel = (eu == d)
                if sel.any():
                    emass[d] += w[sel].sum(); ecnt[d] += 1
            m_self.append(float(w[q])); beyond1.append(float(w[fin & (dq > 1)].sum()))
            if len(tgt):
                m_target.append(float(w[tgt].sum()))
            if len(agt):
                m_agent.append(float(w[agt].sum()))
            if np.isfinite(dT[q]):
                tw = np.isfinite(dT) & (dT < dT[q]); aw = np.isfinite(dT) & (dT > dT[q]); eq = np.isfinite(dT) & (dT == dT[q])
                toward.append(float(w[tw].sum())); away.append(float(w[aw].sum())); equal.append(float(w[eq].sum()))
        # per-board geometric decay fit on shells 1..6 (mass per shell averaged)

    gprof = gmass / np.maximum(gcnt, 1)                                    # mean mass at each graph-dist shell
    eprof = emass / np.maximum(ecnt, 1)
    # geometric decay rate on graph shells 1..6
    dd = np.arange(1, 7); yy = gprof[1:7]
    pos = yy > 0
    rho_g = float(np.exp(np.polyfit(dd[pos], np.log(yy[pos]), 1)[0])) if pos.sum() >= 2 else float("nan")

    f3 = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== E4: WHERE VALUE IS PULLED FROM (step={step}, boards={B}, tick={tick}, top cell, head-avg) =====")
    print(f"  graph-dist shell d:        " + " ".join(f"{d:>5}" for d in range(0, 9)))
    print(f"  mean attention mass at d:  " + " ".join(f"{gprof[d]:5.3f}" for d in range(0, 9)))
    print(f"  euclid-dist shell mass:    " + f3(eprof[:9]))
    print(f"  geometric decay rho_graph (shells 1-6): {rho_g:.3f}   (mass ~ rho^d; reach 1/(1-rho) ~ {1/(1-rho_g+1e-9):.1f} hops)")
    print(f"  mass at d=0 (self):        {np.mean(m_self):.3f}")
    print(f"  mass at d=1 (immediate nb):{gprof[1]:.3f}")
    print(f"  mass BEYOND 1 hop (d>1):   {np.mean(beyond1):.3f}   <-- value is pulled from here, not just neighbours")
    print(f"  mass to UNREACHABLE (wall-blocked) keys: {np.mean(unreach):.3f}   (through-wall leakage)")
    print(f"  -- goal-ward asymmetry (BFS-to-target) --")
    tw, aw, eq = np.mean(toward), np.mean(away), np.mean(equal)
    print(f"     mass TOWARD goal (lower dT): {tw:.3f}   AWAY (higher dT): {aw:.3f}   equal: {eq:.3f}   toward/away ratio {tw/(aw+1e-9):.2f}")
    print(f"  -- landmark mass --")
    print(f"     on TARGET cell: {np.mean(m_target):.3f}   on AGENT cell: {np.mean(m_agent):.3f}")
    # machine-readable block for plotting
    print("PLOT_GRAPH_PROFILE=" + repr([round(float(gprof[d]), 4) for d in range(0, 9)]))
    print("PLOT_EUCLID_PROFILE=" + repr([round(float(eprof[d]), 4) for d in range(0, 9)]))
    print("PLOT_GOALWARD=" + repr([round(tw, 4), round(eq, 4), round(aw, 4)]))
    print("PLOT_SCALARS=" + repr(dict(rho_g=round(rho_g, 4), self=round(float(np.mean(m_self)), 4),
          d1=round(float(gprof[1]), 4), beyond1=round(float(np.mean(beyond1)), 4),
          unreach=round(float(np.mean(unreach)), 4), target=round(float(np.mean(m_target)), 4),
          agent=round(float(np.mean(m_agent)), 4))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=128); ap.add_argument("--tick", type=int, default=-1)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.tick)
