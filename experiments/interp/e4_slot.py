"""E4 (SLOT CORE) -- WHERE does each slot ROUTE value FROM? Recover the transition graph N.

This is the headline structural result, ported from the spatial ATTENTION core (experiments/interp/e4.py)
to the learnable-SLOT core (cleanba.slot_lstm). The attn core assumed cell i == board square i, so its
attention row A[q,:] could be binned by board-graph distance directly. The slot core has N FREE slots NOT
tied to board position; the binding sigma (slot <-> board square) is LEARNED. So everything must go through
sigma: decode_sigma maps each slot -> the board position it binds, and a slot pair (i,j) is at "graph
distance d" iff the BFS board distance between pos[i] and pos[j] is d.

The per-tick update routes value over the slot<->slot graph: slot q aggregates source-slot value
weighted by its ROUTING row route[q,:] (slot<->slot self-attention). So route[q,:] literally says "which
slots q pulls value from" -- the recovered graph N. At the settled tick (head-avg) we characterize route:

  (1) routing mass vs GRAPH distance d(pos[i],pos[j]) over board squares (BFS around walls): does mass
      decay ~rho^d, and to how many hops? fraction of mass beyond 1-hop (the "not immediate neighbours"
      share); mass routed to graph-UNREACHABLE (through-wall) slot pairs (both slots bind nav squares but
      no path connects them) -- spurious edges in the recovered N.
  (2) routing mass vs EUCLIDEAN distance between bound squares (contrast: graph- vs pixel-locality).
  (3) GOAL-WARD asymmetry: does q route from slots whose bound square is CLOSER to the goal (lower
      BFS-to-target = the value source direction) more than from farther ones? (real value propagation
      pulls value from toward the reward.)
  (4) LANDMARK mass: routing onto the slot bound to the target square, the slot bound to the agent
      square, and self-routing.

This says what algorithm the slot loop implements: a 1-hop Jacobi PE on the recovered graph (mass at d=1
only) vs a broad multi-hop goal-ward diffusion kernel (geometric decay, biased toward the goal) vs a
global broadcast. The structural recovery of N is the difference between "the slots learned the board
adjacency from RL alone" and "the routing is uninformative".

Only slots with a CONFIDENT binding (top binding mass above a threshold) are used, so a slot's claimed
board position is meaningful; slots that bind nothing crisply are dropped from the analysis.

  python -m experiments.interp.e4_slot --ckpt <cp_dir> --boards 256 --tick -1
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
from collections import deque
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def all_pairs_graph_dist(tiles_b, H, W):
    """(S,S) BFS distance over non-wall board squares; inf if unreachable, nan for wall sources."""
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


def main(cp_dir, n_boards, tick, bind_thresh):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step

    envs = env_cfg.make(); obs = np.asarray(envs.reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)

    # Slot core: per-tick top-slot hidden + binding (slot->board square) + routing (slot<->slot).
    h, bind, route = slot_per_tick(policy, params, jnp.asarray(obs), K)        # (K,B,N,d),(K,B,nh,N,S),(K,B,nh,N,N)
    N = bind.shape[-2]
    R = route[tick].mean(1)                                                    # (B,N,N) head-avg routing, settled tick
    pos, mass = decode_sigma(bind, tick=tick)                                  # pos:(B,N) slot->board-pos; mass:(B,N) confidence

    rc = np.arange(S); RR, CC = rc // W, rc % W
    maxd = 12
    gmass = np.zeros(maxd + 1); gcnt = np.zeros(maxd + 1)                      # mass by graph-dist shell (0..maxd)
    unreach = []                                                              # mass routed to graph-unreachable pairs
    emass = np.zeros(maxd + 1); ecnt = np.zeros(maxd + 1)                      # mass by euclid-dist shell (rounded)
    toward = []; away = []; equal = []                                        # goal-ward decomposition
    m_self = []; m_target = []; m_agent = []; beyond1 = []
    n_used_slots = []

    for b in range(B):
        nav = tiles[b] != WALL
        # confident slots whose bound board square is navigable (a meaningful claimed position)
        ok = (mass[b] >= bind_thresh) & nav[pos[b]]                            # (N,) usable source/query slots
        n_used_slots.append(int(ok.sum()))
        if ok.sum() < 2:
            continue
        dG = all_pairs_graph_dist(tiles[b], H, W)                              # (S,S) BFS board distance
        tgt = np.where(tiles[b] == TARGET)[0]; agt = np.where(tiles[b] == AGENT)[0]
        dT = bfs_from(tgt, tiles[b], H, W) if len(tgt) else np.full(S, np.nan)
        # slots whose bound square IS the target / agent square (the landmark slots)
        tgt_slot = ok & np.isin(pos[b], tgt) if len(tgt) else np.zeros(N, bool)
        agt_slot = ok & np.isin(pos[b], agt) if len(agt) else np.zeros(N, bool)

        for q in range(N):
            if not ok[q]:
                continue
            w = R[b, q].copy()                                                 # (N,) routing row: where slot q pulls from
            w[~ok] = 0.0                                                       # restrict to usable source slots
            tot = w.sum()
            if tot <= 0:
                continue
            w = w / tot                                                        # renormalize over the usable graph N
            pq = pos[b, q]                                                     # board square slot q binds
            # graph distance from q's bound square to every (source) slot's bound square
            dq = dG[pq][pos[b]]                                                # (N,) d(pos[q], pos[j]) for source slot j
            fin = np.isfinite(dq)
            unreach.append(float(w[~fin].sum()))                              # mass to nav-but-unreachable (through-wall) slots
            for d in range(maxd + 1):
                sel = fin & (dq == d)
                if sel.any():
                    gmass[d] += w[sel].sum(); gcnt[d] += 1
            eu = np.round(np.hypot(RR[pos[b]] - RR[pq], CC[pos[b]] - CC[pq])).astype(int)  # euclid dist between bound squares
            for d in range(maxd + 1):
                sel = (eu == d)
                if sel.any():
                    emass[d] += w[sel].sum(); ecnt[d] += 1
            m_self.append(float(w[q])); beyond1.append(float(w[fin & (dq > 1)].sum()))
            if tgt_slot.any():
                m_target.append(float(w[tgt_slot].sum()))
            if agt_slot.any():
                m_agent.append(float(w[agt_slot].sum()))
            # goal-ward asymmetry: compare BFS-to-target of each source slot's bound square to q's
            dTq = dT[pq]
            if np.isfinite(dTq):
                dTj = dT[pos[b]]                                               # (N,) BFS-to-target of each source slot
                tw = np.isfinite(dTj) & (dTj < dTq)
                aw = np.isfinite(dTj) & (dTj > dTq)
                eq = np.isfinite(dTj) & (dTj == dTq)
                toward.append(float(w[tw].sum())); away.append(float(w[aw].sum())); equal.append(float(w[eq].sum()))

    gprof = gmass / np.maximum(gcnt, 1)                                        # mean mass at each graph-dist shell
    eprof = emass / np.maximum(ecnt, 1)
    # geometric decay rate on graph shells 1..6
    dd = np.arange(1, 7); yy = gprof[1:7]
    posy = yy > 0
    rho_g = float(np.exp(np.polyfit(dd[posy], np.log(yy[posy]), 1)[0])) if posy.sum() >= 2 else float("nan")

    f3 = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== E4 (slot core): RECOVERY OF THE TRANSITION GRAPH N (step={step}, boards={B}, "
          f"slots={N}, tick={tick}, head-avg) =====")
    print(f"  slots used per board (conf binding >= {bind_thresh:.2f}, bound sq navigable): "
          f"{np.mean(n_used_slots):.1f} / {N}")
    print(f"  graph-dist shell d:        " + " ".join(f"{d:>5}" for d in range(0, 9)))
    print(f"  mean routing mass at d:    " + " ".join(f"{gprof[d]:5.3f}" for d in range(0, 9)))
    print(f"  euclid-dist shell mass:    " + f3(eprof[:9]))
    print(f"  geometric decay rho_graph (shells 1-6): {rho_g:.3f}   (mass ~ rho^d; reach 1/(1-rho) ~ {1/(1-rho_g+1e-9):.1f} hops)")
    print(f"  mass at d=0 (self-route):  {np.mean(m_self) if m_self else float('nan'):.3f}")
    print(f"  mass at d=1 (graph-adjacent slots): {gprof[1]:.3f}   <-- the recovered one-step transition edges")
    print(f"  mass BEYOND 1 hop (d>1):   {np.mean(beyond1) if beyond1 else float('nan'):.3f}   <-- value routed from here, not just neighbours")
    print(f"  mass to graph-UNREACHABLE (through-wall) slot pairs: {np.mean(unreach) if unreach else float('nan'):.3f}   (spurious edges in N)")
    print(f"  -- goal-ward asymmetry (BFS-to-target of bound square) --")
    tw = np.mean(toward) if toward else float('nan')
    aw = np.mean(away) if away else float('nan')
    eq = np.mean(equal) if equal else float('nan')
    print(f"     mass TOWARD goal (lower dT): {tw:.3f}   AWAY (higher dT): {aw:.3f}   equal: {eq:.3f}   toward/away ratio {tw/(aw+1e-9):.2f}")
    print(f"  -- landmark mass --")
    print(f"     on TARGET slot: {np.mean(m_target) if m_target else float('nan'):.3f}   "
          f"on AGENT slot: {np.mean(m_agent) if m_agent else float('nan'):.3f}")
    # machine-readable block for plotting
    print("PLOT_GRAPH_PROFILE=" + repr([round(float(gprof[d]), 4) for d in range(0, 9)]))
    print("PLOT_EUCLID_PROFILE=" + repr([round(float(eprof[d]), 4) for d in range(0, 9)]))
    print("PLOT_GOALWARD=" + repr([round(float(tw), 4), round(float(eq), 4), round(float(aw), 4)]))
    print("PLOT_SCALARS=" + repr(dict(rho_g=round(rho_g, 4),
          self=round(float(np.mean(m_self)) if m_self else float('nan'), 4),
          d1=round(float(gprof[1]), 4),
          beyond1=round(float(np.mean(beyond1)) if beyond1 else float('nan'), 4),
          unreach=round(float(np.mean(unreach)) if unreach else float('nan'), 4),
          target=round(float(np.mean(m_target)) if m_target else float('nan'), 4),
          agent=round(float(np.mean(m_agent)) if m_agent else float('nan'), 4),
          slots_used=round(float(np.mean(n_used_slots)), 2))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--tick", type=int, default=-1)
    ap.add_argument("--bind-thresh", type=float, default=0.05,
                    help="min top-1 binding mass for a slot's claimed board position to count")
    a = ap.parse_args(); main(a.ckpt, a.boards, a.tick, a.bind_thresh)