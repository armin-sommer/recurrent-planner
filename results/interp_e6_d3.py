"""E6 -- INTERVENTION test: do the propagated values change as the resolvent theory predicts? (D=3 entmax)

Theory: V = (I - gamma P)^{-1} r  (policy-evaluation resolvent).
  * MOVE GOAL G0->G1  => the value field re-centers on G1: it should track gamma^{d(.,G1)}, not d(.,G0).
  * ADD OBSTACLE      => P loses edges => value DROPS behind it, exactly where the geodesic-to-goal
                         lengthens: corr(dV(s), -Delta d(s,goal)) > 0.

We decode a per-cell value field with a linear probe fit on ORIGINAL boards to gamma^{d(s,goal)}, apply
the SAME probe under each intervention, and measure agreement at cells OUTSIDE the ~7x7 conv RF of the
edits (so the change is PROPAGATED, not local pixels). Crash-free recompute_d3 + head-free value probe.

  python -m results.interp_e6_d3 --ckpt <cp_dir> --boards 192 --examples 2
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
from collections import deque
import numpy as np
import jax, jax.numpy as jnp

from results.interp_planning_d3 import recompute_d3, get_embed
from results.interp_slots import decode_tiles
from results.interp_plan import bfs_from, WALL, FLOOR, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_RGB = np.array([254, 126, 125], np.uint8)
WALL_RGB = np.array([0, 0, 0], np.uint8)


def topfield(cps, policy, params, obs, K):
    emb = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs))))
    return np.asarray(recompute_d3(cps, emb, K)[0])[-1]                       # (B,S,C) settled top hidden


def fit_probe(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, float(y.mean())


def decode(h, p):
    mu, sd, w, b = p
    return ((h - mu) / sd) @ w + b                                           # (B,S)


def corr(a, b, m):
    a, b = a[m], b[m]
    if len(a) < 6 or a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def geodesic_mid(agent, dT, H, W):
    s = agent; path = [s]
    while dT[s] > 0:
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


def main(cp_dir, n_boards, n_examples):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    gamma = cp_cfg.loss.gamma
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs0 = np.asarray(envs.reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    rc = np.arange(S); RR, CC = rc // W, rc % W
    G0 = np.full(B, -1); A0 = np.full(B, -1)
    dT0 = np.full((B, S), np.nan)
    for b in range(B):
        g = np.where(tiles[b] == TARGET)[0]; a = np.where(tiles[b] == AGENT)[0]
        if len(g) and len(a):
            G0[b] = int(g[0]); A0[b] = int(a[0]); dT0[b] = bfs_from([int(g[0])], tiles[b], H, W)
    V0star = gamma ** dT0                                                      # nan at walls/unreachable

    h0 = topfield(cps, policy, params, obs0, K)
    # fit value probe on original boards (pooled non-wall finite cells)
    idx = [(b, s) for b in range(B) for s in range(S) if np.isfinite(V0star[b, s])]
    Xp = np.stack([h0[b, s] for b, s in idx]); yp = np.array([V0star[b, s] for b, s in idx])
    probe = fit_probe(Xp, yp)
    V0 = decode(h0, probe)

    # ---------- INTERVENTION 1: move the goal G0 -> G1 (far floor, clean of RF) ----------
    obs_g = obs0.copy(); G1 = np.full(B, -1)
    for b in range(B):
        if G0[b] < 0:
            continue
        fl = np.where(tiles[b] == FLOOR)[0]
        eu_a = np.hypot(RR[fl] - RR[A0[b]], CC[fl] - CC[A0[b]]); eu_g = np.hypot(RR[fl] - RR[G0[b]], CC[fl] - CC[G0[b]])
        cand = fl[(eu_a >= 4) & (eu_g >= 4)]
        if not len(cand):
            continue
        g1 = int(cand[np.argmax(np.hypot(RR[cand] - RR[A0[b]], CC[cand] - CC[A0[b]]))]); G1[b] = g1
        gr, gc = divmod(G0[b], W); fr, fc = divmod(int(fl[0]), W); nr, nc = divmod(g1, W)
        obs_g[b, :, gr, gc] = obs0[b, :, fr, fc]                               # vacate old goal
        obs_g[b, :, nr, nc] = TARGET_RGB                                       # plant new goal
    tg_g = decode_tiles(obs_g)
    okg = np.array([G1[b] >= 0 and tg_g[b, G1[b]] == TARGET and tg_g[b, G0[b]] == FLOOR for b in range(B)])
    hg = topfield(cps, policy, params, obs_g, K); V1 = decode(hg, probe)
    dT1 = np.full((B, S), np.nan)
    for b in range(B):
        if okg[b]:
            dT1[b] = bfs_from([int(G1[b])], tg_g[b], H, W)
    V1star = gamma ** dT1
    track_new, track_old, base = [], [], []
    for b in range(B):
        if not okg[b]:
            continue
        eu0 = np.hypot(RR - RR[G0[b]], CC - CC[G0[b]]); eu1 = np.hypot(RR - RR[G1[b]], CC - CC[G1[b]])
        far = (tiles[b] != WALL) & (eu0 >= 4) & (eu1 >= 4) & np.isfinite(dT1[b]) & np.isfinite(dT0[b])
        track_new.append(corr(V1[b], V1star[b], far)); track_old.append(corr(V1[b], V0star[b], far))
        base.append(corr(V0[b], V0star[b], far))

    # ---------- INTERVENTION 2: add an obstacle on the agent->goal geodesic ----------
    obs_o = obs0.copy(); obcell = np.full(B, -1)
    for b in range(B):
        if G0[b] < 0 or A0[b] < 0 or not np.isfinite(dT0[b, A0[b]]):
            continue
        mid = geodesic_mid(A0[b], dT0[b], H, W)
        if mid is None or tiles[b, mid] != FLOOR:
            continue
        obcell[b] = mid; mr, mc = divmod(mid, W); obs_o[b, :, mr, mc] = WALL_RGB
    tg_o = decode_tiles(obs_o)
    oko = np.array([obcell[b] >= 0 and tg_o[b, obcell[b]] == WALL for b in range(B)])
    ho = topfield(cps, policy, params, obs_o, K); Vo = decode(ho, probe)
    agree_obs, frac_drop = [], []
    for b in range(B):
        if not oko[b]:
            continue
        dT0n = bfs_from([int(G0[b])], tg_o[b], H, W)                           # new geodesic-to-goal with obstacle
        dd = dT0n - dT0[b]                                                     # >=0 where path lengthened
        eu_o = np.hypot(RR - RR[obcell[b]], CC - CC[obcell[b]])
        far = (tiles[b] != WALL) & (eu_o >= 4) & np.isfinite(dd) & np.isfinite(dT0[b])
        if far.sum() >= 6:
            agree_obs.append(corr((Vo[b] - V0[b]), -dd, far))                 # value drops where dd>0 -> positive corr
            shadow = far & (dd > 0)
            if shadow.sum() >= 3:
                frac_drop.append(float((Vo[b][shadow] < V0[b][shadow]).mean()))

    f = lambda x: np.nanmean(x)
    print(f"\n===== E6: INTERVENTION vs RESOLVENT THEORY (step={step}, boards={B}, gamma={gamma}) =====")
    print(f"  probe: linear h(s)->gamma^d(s,goal); fit R^2 (orig field vs goal) = {f(base):.3f}  (n_clean_goalmove={int(okg.sum())})")
    print(f"  -- (1) MOVE GOAL G0->G1 (far cells, outside conv RF of both) --")
    print(f"     corr(V_after, gamma^d to NEW goal) = {f(track_new):+.3f}   <- theory: high")
    print(f"     corr(V_after, gamma^d to OLD goal) = {f(track_old):+.3f}   <- theory: low/negative")
    print(f"     => field {'RE-CENTERS on the new goal (AGREES)' if f(track_new) > f(track_old) + 0.1 else 'does NOT clearly re-center'}")
    print(f"  -- (2) ADD OBSTACLE on agent->goal path (n={len(agree_obs)}) --")
    print(f"     corr(dV(s), -Delta geodesic-to-goal) = {f(agree_obs):+.3f}   <- theory: positive (value drops behind obstacle)")
    print(f"     fraction of 'shadowed' cells (Delta d>0) whose value DROPPED = {f(frac_drop):.3f}   <- theory: ~1")
    ok1 = f(track_new) > f(track_old) + 0.1; ok2 = f(agree_obs) > 0.1
    print(f"  --> {'BOTH interventions agree with the resolvent theory' if (ok1 and ok2) else 'partial/again-check'}")

    # ---------- export example fields for heatmaps ----------
    ex = [b for b in range(B) if okg[b] and oko[b]][:n_examples]
    rnd = lambda a: [round(float(x), 3) if np.isfinite(x) else None for x in a]
    for b in ex:
        print(f"EX_b={b} G0={int(G0[b])} G1={int(G1[b])} A0={int(A0[b])} OB={int(obcell[b])}")
        print(f"EX_tiles={list(map(int, tiles[b]))}")
        print(f"EX_V0={rnd(V0[b])}")
        print(f"EX_V1={rnd(V1[b])}")
        print(f"EX_Vo={rnd(Vo[b])}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=192); ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.examples)
