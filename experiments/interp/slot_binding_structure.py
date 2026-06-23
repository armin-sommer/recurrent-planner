"""What binding structure does an UNDER-COMPLETE slot core use? (n50: 50 slots, 100 board squares.)

Injective slot<->square binding is impossible when num_slots < S. But Sokoban boards are ~half walls,
so the *navigable* (non-wall) region is ~50 cells -- the hypothesis is that the model spends its scarce
slots on the cells that matter (navigable / reachable / object cells) and ignores walls, getting an
~injective binding over the RELEVANT subgraph rather than the whole board.

This decodes the learned binding sigma (slot -> board position, via the slot-attention map) and measures
the allocation structure:
  - wall avoidance:     do slots bind walls, or skip them?
  - navigable coverage: of the non-wall cells, how many are bound by some slot?
  - reachable coverage: of the agent-reachable cells (BFS), how many are bound?
  - object coverage:    are the agent / boxes / targets always bound?
  - concentration:      is each slot's binding peaked on one square (sharp) or diffuse?
  - injectivity:        how many DISTINCT positions do the N slots cover (vs N)?

  python -m experiments.interp.slot_binding_structure --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

TILE = {WALL: "wall", FLOOR: "floor", BOX: "box", TARGET: "target", AGENT: "agent"}


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; N = net.recurrent.num_slots
    envs = env_cfg.make(); obs = np.asarray(envs.reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)                                                # (B,S)

    _h, bind, _route = slot_per_tick(policy, params, jnp.asarray(obs), K)    # bind:(K,B,nh,N,S)
    pos, mass = decode_sigma(bind, tick=-1)                                  # pos:(B,N) slot->square; mass:(B,N)
    # full head-averaged binding distribution at the settled tick, for concentration/entropy
    w = bind[-1].mean(1)                                                     # (B,N,S) per-slot read distribution

    n_nav = []; n_reach = []
    wall_frac, conc, ent = [], [], []
    nav_cov, reach_cov = [], []
    obj_cov = {"box": [], "target": [], "agent": []}
    distinct = []; slots_per_pos = []
    tilebind = {k: [] for k in TILE}                                         # frac of slots binding each tile type

    for b in range(B):
        t = tiles[b]
        nav = np.where(t != WALL)[0]                                         # navigable (non-wall) squares
        ag = np.where(t == AGENT)[0]
        reach = nav
        if len(ag):
            d = bfs_from([int(ag[0])], t, H, W); reach = np.where(np.isfinite(d) & (t != WALL))[0]
        n_nav.append(len(nav)); n_reach.append(len(reach))
        p = pos[b]                                                          # (N,) bound square per slot
        # wall avoidance + per-tile binding fractions
        bt = t[p]                                                           # tile type each slot binds
        wall_frac.append(np.mean(bt == WALL))
        for k in TILE: tilebind[k].append(np.mean(bt == k))
        # concentration of each slot's binding (top-1 mass) and entropy
        conc.append(mass[b].mean())
        pw = np.clip(w[b], 1e-9, 1); ent.append(np.mean(-(pw * np.log(pw)).sum(1)))
        # coverage of navigable / reachable cells
        boundset = set(int(x) for x in p)
        nav_cov.append(len(boundset & set(int(x) for x in nav)) / max(1, len(nav)))
        reach_cov.append(len(boundset & set(int(x) for x in reach)) / max(1, len(reach)))
        # object coverage: is each object's square bound by SOME slot?
        for name, tid in [("box", BOX), ("target", TARGET), ("agent", AGENT)]:
            objs = np.where(t == tid)[0]
            obj_cov[name].append(1.0 if len(objs) and all(int(o) in boundset for o in objs) else (np.nan if not len(objs) else 0.0))
        # injectivity: distinct positions / N ; many-to-one degree
        distinct.append(len(boundset) / N)
        slots_per_pos.append(N / max(1, len(boundset)))

    m = lambda x: float(np.nanmean(x))
    print(f"\n===== SLOT BINDING STRUCTURE (slot core, step={step}, boards={B}, slots N={N}, board S={S}) =====")
    print(f"  board: mean walls={S - m(n_nav):.1f}  navigable(non-wall)={m(n_nav):.1f}  agent-reachable={m(n_reach):.1f}   (vs N={N} slots)")
    print(f"  -- WHAT SLOTS BIND (fraction of the {N} slots, by bound-tile type) --")
    for k in [WALL, FLOOR, BOX, TARGET, AGENT]:
        print(f"       {TILE[k]:>7}: {m(tilebind[k]):.3f}")
    print(f"  -- ALLOCATION STRUCTURE --")
    print(f"       wall-binding slots         : {m(wall_frac):.3f}   (low => slots AVOID walls)")
    print(f"       navigable-cell coverage    : {m(nav_cov):.3f}   (frac of non-wall cells bound by >=1 slot)")
    print(f"       reachable-cell coverage    : {m(reach_cov):.3f}   (frac of agent-reachable cells bound)")
    print(f"       object coverage  box/target/agent : {m(obj_cov['box']):.3f} / {m(obj_cov['target']):.3f} / {m(obj_cov['agent']):.3f}")
    print(f"       binding concentration (top-1 mass): {m(conc):.3f}   (1.0 => one square; low => diffuse)")
    print(f"       binding entropy (nats)            : {m(ent):.2f}   (0 => delta; ln(S)={np.log(S):.2f} => uniform)")
    print(f"       distinct positions / N            : {m(distinct):.3f}   (1.0 => injective over slots)")
    print(f"       slots per covered position        : {m(slots_per_pos):.2f}   (1.0 => 1:1; >1 => redundancy)")
    verdict = ("binds the RELEVANT subgraph (avoids walls, covers navigable/reachable + objects), ~injective over it"
               if m(wall_frac) < 0.2 and m(reach_cov) > 0.7 else "binding structure unclear -- inspect above")
    print(f"  --> {verdict}")
    print("PLOT_BINDSTRUCT=" + repr(dict(N=int(N), navigable=round(m(n_nav), 1), reachable=round(m(n_reach), 1),
          wall_frac=round(m(wall_frac), 3), nav_cov=round(m(nav_cov), 3), reach_cov=round(m(reach_cov), 3),
          obj=[round(m(obj_cov['box']), 3), round(m(obj_cov['target']), 3), round(m(obj_cov['agent']), 3)],
          conc=round(m(conc), 3), distinct=round(m(distinct), 3))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
