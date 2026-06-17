"""E9 -- does the value field incorporate NEW PHYSICS? (a new wall that lengthens the path to the goal)

The question: it need not re-route its attention; it only needs the new wall's CONSEQUENCE (a longer
geodesic to the goal -> lower value) to propagate into the value field. We test the OUTPUT (the readout
value), not the mechanism:
  * ON-PATH wall  (placed on the agent->goal shortest path): lengthens the geodesic by Delta d >= 0.
                  Theory: the agent's value DROPS, by more when the detour is longer.
  * OFF-PATH wall (far from the path): Delta d ~ 0. Control: value should barely move.
We give EXTENDED thinking (K=16) and read the critic value at every tick (a longer path is a multi-hop
consequence that may need propagation time). corr(dV, -Delta d) > 0 and dV_on << dV_off => the value
field DOES adopt the new transition structure.

  python -m experiments.interp.e9 --ckpt <cp_dir> --boards 200 --ticks 16
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_RGB = np.array([0, 0, 0], np.uint8)
DCAP = 25.0   # cap for "disconnected" (inf) geodesic change


def readout_all_ticks(top_h, emb, Wd, bd, Wc, bc, hs):
    K, B, S, C = top_h.shape; embf = emb.reshape(B, S, C); out = np.zeros((K, B))
    for t in range(K):
        mlp = np.maximum((top_h[t] + embf).reshape(B, S * C) @ Wd + bd, 0.0)
        out[t] = (mlp @ Wc + bc)[..., 0] * hs
    return out


def geodesic_path(agent, dT, H, W):
    s = agent; path = [s]
    while np.isfinite(dT[s]) and dT[s] > 0:
        r, c = divmod(s, W); best, bd = None, dT[s]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[nr * W + nc]) and dT[nr * W + nc] < bd:
                bd = dT[nr * W + nc]; best = nr * W + nc
        if best is None:
            break
        s = best; path.append(s)
    return path


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    hs = getattr(net, "head_scale", 1.0); NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wc = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bc = np.asarray(params["params"]["critic_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W

    obs_on = obs0.copy(); obs_off = obs0.copy()
    A0 = np.full(B, -1); Xon = np.full(B, -1); Xoff = np.full(B, -1); dd_on = np.full(B, np.nan); dd_off = np.full(B, np.nan)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]; fl = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl)):
            continue
        a = int(ag[0]); A0[b] = a; dT = bfs_from([int(tg[0])], tiles[b], H, W)
        if not np.isfinite(dT[a]):
            continue
        path = geodesic_path(a, dT, H, W)
        midcells = [c for c in path if tiles[b, c] == FLOOR]
        if not midcells:
            continue
        xon = midcells[len(midcells) // 2]; Xon[b] = xon
        pathset = set(path)
        cand = [c for c in fl if c not in pathset and min(np.hypot(RR[c] - RR[p], CC[c] - CC[p]) for p in path) >= 2.5]
        if not cand:
            continue
        xoff = int(cand[np.argmax([min(np.hypot(RR[c] - RR[p], CC[c] - CC[p]) for p in path) for c in cand])]); Xoff[b] = xoff
        obs_on[b, :, RR[xon], CC[xon]] = WALL_RGB; obs_off[b, :, RR[xoff], CC[xoff]] = WALL_RGB

    ton = decode_tiles(obs_on); toff = decode_tiles(obs_off)
    ok = np.array([Xon[b] >= 0 and Xoff[b] >= 0 and ton[b, Xon[b]] == WALL and toff[b, Xoff[b]] == WALL for b in range(B)])
    for b in range(B):
        if not ok[b]:
            continue
        d0 = bfs_from([int(np.where(tiles[b] == TARGET)[0][0])], tiles[b], H, W)[A0[b]]
        d_on = bfs_from([int(np.where(ton[b] == TARGET)[0][0])], ton[b], H, W)[A0[b]]
        d_off = bfs_from([int(np.where(toff[b] == TARGET)[0][0])], toff[b], H, W)[A0[b]]
        dd_on[b] = (DCAP if not np.isfinite(d_on) else d_on) - d0
        dd_off[b] = (DCAP if not np.isfinite(d_off) else d_off) - d0

    def perTickV(o):
        emb = np.asarray(get_embed(policy, params, jnp.asarray(o)))
        return readout_all_ticks(np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0]), emb, Wd, bd, Wc, bc, hs)
    V0 = perTickV(obs0); Von = perTickV(obs_on); Voff = perTickV(obs_off)
    stdV = float(np.std(V0[-1]))

    m = ok & np.isfinite(dd_on) & np.isfinite(dd_off)
    print(f"\n===== E9: DOES THE VALUE FIELD ADOPT NEW PHYSICS (a path-blocking wall)? (step={step}, boards={B}, n={int(m.sum())}, K={K}) =====")
    print(f"  geodesic lengthening: on-path mean Delta d = {np.nanmean(dd_on[m]):.2f}   off-path mean Delta d = {np.nanmean(dd_off[m]):.2f}")
    dVon = Von[-1] - V0[-1]; dVoff = Voff[-1] - V0[-1]
    print(f"  final-tick value change dV/std:  on-path {np.nanmean(dVon[m])/stdV:+.3f}   off-path {np.nanmean(dVoff[m])/stdV:+.3f}   (theory: on-path more NEGATIVE)")
    print(f"  corr(dV_on, -Delta d_on) = {corr(dVon[m], -dd_on[m]):+.3f}   (theory >0: longer detour -> lower value)")
    # by-tick: does the effect grow with thinking?
    ag_t = [corr(Von[t][m] - V0[t][m], -dd_on[m]) for t in range(K)]
    dv_t = [float(np.nanmean((Von[t] - V0[t])[m]) / stdV) for t in range(K)]
    f2 = lambda xs: "[" + " ".join(("%+.2f" % x if np.isfinite(x) else " nan") for x in xs) + "]"
    print(f"  by thinking tick:")
    print(f"    corr(dV_on,-dd) : {f2(ag_t)}")
    print(f"    mean dV_on/std  : {f2(dv_t)}")
    # blocking subset
    blk = m & (dd_on > 0.5)
    print(f"  on-path walls that lengthened the path (n={int(blk.sum())}): mean dV/std = {np.nanmean(dVon[blk])/stdV:+.3f}, "
          f"fraction with V dropped = {float((dVon[blk] < 0).mean()):.2f}")
    verdict = (corr(dVon[m], -dd_on[m]) > 0.2 and np.nanmean(dVon[blk]) < np.nanmean(dVoff[m]))
    print(f"  --> {'VALUE ADOPTS THE NEW PHYSICS: it drops in proportion to the added path length' if verdict else 'value does NOT clearly track the new path length'}")
    print("PLOT_E9=" + repr(dict(dV_on=round(float(np.nanmean(dVon[m])/stdV),3), dV_off=round(float(np.nanmean(dVoff[m])/stdV),3),
                                  corr=round(corr(dVon[m], -dd_on[m]),3), dd_on=round(float(np.nanmean(dd_on[m])),2),
                                  agt=[round(x,3) if np.isfinite(x) else None for x in ag_t])))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=200); ap.add_argument("--ticks", type=int, default=16)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
