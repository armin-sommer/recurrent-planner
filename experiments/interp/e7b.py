"""E7b -- does the operator A re-route LOCALLY around a new block? (E7's all-query average was diluted.)

E7 averaged the attention mass over ALL queries and saw ~no change. But any re-routing is LOCAL to the
edited cell, so a global average washes it out. E7b measures it locally:
  (1) ||dA(q)||_1 = row change of the attention operator, BINNED by graph-distance of query q from the
      edited cell. Re-routing -> large near the edit, decaying with distance. Globally-fixed -> ~0 everywhere.
  (2) NEIGHBOUR mass onto the blocked cell: do the cells bordering X stop attending to it once X becomes a
      wall / a box? (the sharpest local re-route signal). And do the goal's neighbours re-anchor?

Interventions (as in E7): goal->far side, obstacle on the agent->goal geodesic, box->adjacent floor.

  python -m experiments.interp.e7b --ckpt <cp_dir> --boards 160
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_RGB = np.array([254, 126, 125], np.uint8)
WALL_RGB = np.array([0, 0, 0], np.uint8)
BOX_RGB = np.array([142, 121, 56], np.uint8)


def geodesic_mid(agent, dT, H, W):
    s = agent; path = [s]
    while np.isfinite(dT[s]) and dT[s] > 0:
        r, c = divmod(s, W); best, bd = None, dT[s]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                t = nr * W + nc
                if np.isfinite(dT[t]) and dT[t] < bd:
                    bd = dT[t]; best = t
        if best is None:
            break
        s = best; path.append(s)
    return path[len(path) // 2] if len(path) >= 3 else None


def neighbors(cell, tiles_b, H, W):
    r, c = divmod(cell, W); out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and tiles_b[nr * W + nc] != WALL:
            out.append(nr * W + nc)
    return out


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W

    def far_floor(b, frm, banned):
        fl = np.array([s for s in np.where(tiles[b] == FLOOR)[0] if s not in banned])
        return int(fl[np.argmax(np.hypot(RR[fl] - RR[frm], CC[fl] - CC[frm]))]) if len(fl) else -1

    obs_g = obs0.copy(); obs_o = obs0.copy(); obs_b = obs0.copy()
    G0 = np.full(B, -1); G1 = np.full(B, -1); A0 = np.full(B, -1); OB = np.full(B, -1); BX0 = np.full(B, -1); BX1 = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]; fl0 = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl0)):
            continue
        a = int(ag[0]); A0[b] = a; G0[b] = int(tg[0])
        fr, fc = divmod(int(fl0[0]), W); floor_rgb = obs0[b, :, fr, fc]; dT = bfs_from([G0[b]], tiles[b], H, W)
        g1 = far_floor(b, G0[b], {a})
        if g1 >= 0:
            G1[b] = g1; obs_g[b, :, RR[G0[b]], CC[G0[b]]] = floor_rgb; obs_g[b, :, RR[g1], CC[g1]] = TARGET_RGB
        if np.isfinite(dT[a]):
            mid = geodesic_mid(a, dT, H, W)
            if mid is not None and tiles[b, mid] == FLOOR:
                OB[b] = mid; obs_o[b, :, RR[mid], CC[mid]] = WALL_RGB
        boxes = [s for s in np.where(tiles[b] == BOX)[0] if np.array_equal(obs0[b, :, RR[s], CC[s]], BOX_RGB)]
        for s in boxes:
            br, bc = divmod(int(s), W); moved = False
            for dr, dc in DIRS:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < H and 0 <= nc < W and tiles[b, nr * W + nc] == FLOOR:
                    obs_b[b, :, br, bc] = floor_rgb; obs_b[b, :, nr, nc] = BOX_RGB; BX0[b] = int(s); BX1[b] = nr * W + nc; moved = True; break
            if moved:
                break
    tg_g = decode_tiles(obs_g); tg_o = decode_tiles(obs_o); tg_b = decode_tiles(obs_b)
    okg = np.array([G1[b] >= 0 and tg_g[b, G1[b]] == TARGET and tg_g[b, G0[b]] == FLOOR for b in range(B)])
    oko = np.array([OB[b] >= 0 and tg_o[b, OB[b]] == WALL for b in range(B)])
    okb = np.array([BX1[b] >= 0 and tg_b[b, BX1[b]] == BOX and tg_b[b, BX0[b]] == FLOOR for b in range(B)])

    def A_settled(o):
        emb = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(o))))
        return np.asarray(recompute_d3(cps, emb, K)[1][-1, D - 1].mean(1))             # (B,S,S)
    A0a = A_settled(obs0); Aga = A_settled(obs_g); Aoa = A_settled(obs_o); Aba = A_settled(obs_b)

    bins = [(0, 1), (2, 3), (4, 5), (6, 99)]

    def dA_by_dist(Aint, ok, edit_cells_fn):
        acc = {i: [] for i in range(len(bins))}
        for b in range(B):
            if not ok[b]:
                continue
            edits = edit_cells_fn(b)
            de = np.minimum.reduce([bfs_from([e], tiles[b], H, W) for e in edits])       # graph dist to nearest edit
            dA = np.abs(Aint[b] - A0a[b]).sum(1)                                          # (S,) L1 row change
            q = tiles[b] != WALL
            for i, (lo, hi) in enumerate(bins):
                m = q & np.isfinite(de) & (de >= lo) & (de <= hi)
                if m.sum():
                    acc[i].append(float(dA[m].mean()))
        return [float(np.mean(acc[i])) if acc[i] else float("nan") for i in range(len(bins))]

    dA_goal = dA_by_dist(Aga, okg, lambda b: [G0[b], G1[b]])
    dA_obs = dA_by_dist(Aoa, oko, lambda b: [OB[b]])
    dA_box = dA_by_dist(Aba, okb, lambda b: [BX0[b], BX1[b]])

    # neighbour mass onto the blocked / vacated cell
    def nbr_mass(Aint, ok, cell_fn):
        o, a = [], []
        for b in range(B):
            if not ok[b]:
                continue
            cell = cell_fn(b); nb = neighbors(cell, tiles[b], H, W)
            if nb:
                o.append(float(A0a[b][nb, cell].mean())); a.append(float(Aint[b][nb, cell].mean()))
        return float(np.mean(o)), float(np.mean(a))
    nbX_o, nbX_a = nbr_mass(Aoa, oko, lambda b: OB[b])
    nbBn_o, nbBn_a = nbr_mass(Aba, okb, lambda b: BX1[b])
    nbBo_o, nbBo_a = nbr_mass(Aba, okb, lambda b: BX0[b])
    nbG1_o, nbG1_a = nbr_mass(Aga, okg, lambda b: G1[b])

    f3 = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== E7b: LOCAL operator change ||dA(q)||_1 by graph-distance from the edit (step={step}, boards={B}) =====")
    print(f"  distance bin:        " + " ".join(f"{lo}-{hi if hi<99 else '+':>2}" for lo, hi in bins))
    print(f"  goal move (n={int(okg.sum())}):     {f3(dA_goal)}")
    print(f"  add obstacle (n={int(oko.sum())}):  {f3(dA_obs)}")
    print(f"  move box (n={int(okb.sum())}):      {f3(dA_box)}")
    loc = lambda v: (v[0] / (v[-1] + 1e-9)) if np.isfinite(v[0]) and np.isfinite(v[-1]) else float("nan")
    print(f"  near/far ratio (bin0 / bin3):  goal {loc(dA_goal):.1f}x   obstacle {loc(dA_obs):.1f}x   box {loc(dA_box):.1f}x")
    print(f"  -- neighbour mass ONTO the edited cell (sharp local re-route) --")
    print(f"     obstacle cell: floor {nbX_o:.3f} -> wall {nbX_a:.3f}   ({'DROPS -> re-routes' if nbX_a < 0.7*nbX_o else 'no drop'})")
    print(f"     box new cell:  floor {nbBn_o:.3f} -> box  {nbBn_a:.3f}   ({'DROPS -> blocks' if nbBn_a < 0.7*nbBn_o else 'no drop'})")
    print(f"     box old cell:  box   {nbBo_o:.3f} -> floor {nbBo_a:.3f}  ({'RISES -> re-opens' if nbBo_a > 1.2*nbBo_o else 'no rise'})")
    print(f"     new-goal cell: floor {nbG1_o:.3f} -> goal {nbG1_a:.3f}   ({'RISES -> re-anchors' if nbG1_a > 1.2*nbG1_o else 'no rise'})")
    print("PLOT_E7B=" + repr(dict(dA_goal=[round(x,4) for x in dA_goal], dA_obs=[round(x,4) for x in dA_obs], dA_box=[round(x,4) for x in dA_box],
                                  obsX=[round(nbX_o,4),round(nbX_a,4)], boxnew=[round(nbBn_o,4),round(nbBn_a,4)],
                                  boxold=[round(nbBo_o,4),round(nbBo_a,4)], newgoal=[round(nbG1_o,4),round(nbG1_a,4)])))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=160)
    a = ap.parse_args(); main(a.ckpt, a.boards)
