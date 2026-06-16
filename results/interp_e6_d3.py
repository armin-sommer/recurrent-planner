"""E6 -- INTERVENTION test: do the propagated values change as the resolvent theory predicts? (D=3 entmax)

Theory: V = (I - gamma P)^{-1} r  (policy-evaluation resolvent).
  * MOVE GOAL G0->G1  => the value field re-centers on G1: it should track gamma^{d(.,G1)}, not d(.,G0).
  * ADD OBSTACLE      => P loses edges => value DROPS behind it, exactly where the geodesic-to-goal
                         lengthens: corr(dV(s), -Delta d(s,goal)) > 0.
  * MOVE A BOX        => a box is BOTH an obstacle (changes P: clears its old cell, blocks the new one)
                         AND the reward-relevant object. Same geodesic agreement as an obstacle, but the
                         READOUT value (box-aware critic) should move MORE than for a goal/wall move.

We decode a per-cell value field with a linear probe fit on ORIGINAL boards to gamma^{d(s,goal)}, apply
the SAME probe under each intervention, and measure agreement at cells OUTSIDE the ~7x7 conv RF of the
edits (propagated, not local). We also read the model's own critic value V (head) to compare the
MAGNITUDE of impact across the three edits. Crash-free recompute_d3.

  python -m results.interp_e6_d3 --ckpt <cp_dir> --boards 192 --examples 2
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from results.interp_planning_d3 import recompute_d3, get_embed
from results.interp_slots import decode_tiles
from results.interp_plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_RGB = np.array([254, 126, 125], np.uint8)
WALL_RGB = np.array([0, 0, 0], np.uint8)
BOX_RGB = np.array([142, 121, 56], np.uint8)


def field(cps, policy, params, obs, K):
    emb = np.asarray(get_embed(policy, params, jnp.asarray(obs)))                 # (B,H,W,C)
    h = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])[-1]                  # (B,S,C) settled top hidden
    return h, emb


def readout_value(h, emb, Wd, bd, Wc, bc, hs):
    B, S, C = h.shape
    mlp = np.maximum((h + emb.reshape(B, S, C)).reshape(B, S * C) @ Wd + bd, 0.0)
    return (mlp @ Wc + bc)[..., 0] * hs                                           # (B,) critic value


def fit_probe(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, float(y.mean())


def decode(h, p):
    mu, sd, w, b = p
    return ((h - mu) / sd) @ w + b                                               # (B,S)


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
    gamma = cp_cfg.loss.gamma; hs = getattr(net, "head_scale", 1.0)
    NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wc = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bc = np.asarray(params["params"]["critic_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs0 = np.asarray(envs.reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    rc = np.arange(S); RR, CC = rc // W, rc % W
    G0 = np.full(B, -1); A0 = np.full(B, -1); dT0 = np.full((B, S), np.nan)
    for b in range(B):
        g = np.where(tiles[b] == TARGET)[0]; a = np.where(tiles[b] == AGENT)[0]
        if len(g) and len(a):
            G0[b] = int(g[0]); A0[b] = int(a[0]); dT0[b] = bfs_from([int(g[0])], tiles[b], H, W)
    V0star = gamma ** dT0

    h0, emb0 = field(cps, policy, params, obs0, K)
    Vr0 = readout_value(h0, emb0, Wd, bd, Wc, bc, hs); stdV = float(np.std(Vr0))
    idx = [(b, s) for b in range(B) for s in range(S) if np.isfinite(V0star[b, s])]
    Xp = np.stack([h0[b, s] for b, s in idx]); yp = np.array([V0star[b, s] for b, s in idx])
    probe = fit_probe(Xp, yp); V0 = decode(h0, probe)

    def far_field_dV_dec(Vk, V0, b, cells):                                       # mean |dV_decoded| at far cells
        m = tiles[b] != WALL
        for c in cells:
            m = m & (np.hypot(RR - RR[c], CC - CC[c]) >= 4)
        return float(np.abs(Vk[b] - V0[b])[m].mean()) if m.sum() >= 3 else np.nan

    # ---------- (1) MOVE GOAL ----------
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
        obs_g[b, :, gr, gc] = obs0[b, :, fr, fc]; obs_g[b, :, nr, nc] = TARGET_RGB
    tg_g = decode_tiles(obs_g)
    okg = np.array([G1[b] >= 0 and tg_g[b, G1[b]] == TARGET and tg_g[b, G0[b]] == FLOOR for b in range(B)])
    hg, embg = field(cps, policy, params, obs_g, K); V1 = decode(hg, probe); Vrg = readout_value(hg, embg, Wd, bd, Wc, bc, hs)
    dT1 = np.full((B, S), np.nan)
    for b in range(B):
        if okg[b]:
            dT1[b] = bfs_from([int(G1[b])], tg_g[b], H, W)
    V1star = gamma ** dT1
    track_new, track_old, base, mag_g, dvr_g = [], [], [], [], []
    for b in range(B):
        if not okg[b]:
            continue
        eu0 = np.hypot(RR - RR[G0[b]], CC - CC[G0[b]]); eu1 = np.hypot(RR - RR[G1[b]], CC - CC[G1[b]])
        far = (tiles[b] != WALL) & (eu0 >= 4) & (eu1 >= 4) & np.isfinite(dT1[b]) & np.isfinite(dT0[b])
        track_new.append(corr(V1[b], V1star[b], far)); track_old.append(corr(V1[b], V0star[b], far)); base.append(corr(V0[b], V0star[b], far))
        mag_g.append(far_field_dV_dec(V1, V0, b, [G0[b], G1[b]])); dvr_g.append(abs(Vrg[b] - Vr0[b]) / (stdV + 1e-9))

    # ---------- (2) ADD OBSTACLE on agent->goal geodesic ----------
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
    ho, embo = field(cps, policy, params, obs_o, K); Vo = decode(ho, probe); Vro = readout_value(ho, embo, Wd, bd, Wc, bc, hs)
    agree_obs, frac_drop, mag_o, dvr_o = [], [], [], []
    for b in range(B):
        if not oko[b]:
            continue
        dT0n = bfs_from([int(G0[b])], tg_o[b], H, W); dd = dT0n - dT0[b]
        eu_o = np.hypot(RR - RR[obcell[b]], CC - CC[obcell[b]])
        far = (tiles[b] != WALL) & (eu_o >= 4) & np.isfinite(dd) & np.isfinite(dT0[b])
        if far.sum() >= 6:
            agree_obs.append(corr((Vo[b] - V0[b]), -dd, far))
            shadow = far & (dd > 0)
            if shadow.sum() >= 3:
                frac_drop.append(float((Vo[b][shadow] < V0[b][shadow]).mean()))
        mag_o.append(far_field_dV_dec(Vo, V0, b, [obcell[b]])); dvr_o.append(abs(Vro[b] - Vr0[b]) / (stdV + 1e-9))

    # ---------- (3) MOVE A BOX (plain box -> adjacent floor) ----------
    obs_b = obs0.copy(); bx0 = np.full(B, -1); bx1 = np.full(B, -1)
    for b in range(B):
        fl0 = np.where(tiles[b] == FLOOR)[0]
        if not len(fl0):
            continue
        fr, fc = divmod(int(fl0[0]), W); floor_rgb = obs0[b, :, fr, fc]
        boxes = [s for s in np.where(tiles[b] == BOX)[0] if np.array_equal(obs0[b, :, RR[s], CC[s]], BOX_RGB)]
        for s in boxes:
            br, bc2 = divmod(int(s), W); moved = False
            for dr, dc in DIRS:
                nr, nc = br + dr, bc2 + dc
                if 0 <= nr < H and 0 <= nc < W and tiles[b, nr * W + nc] == FLOOR:
                    obs_b[b, :, br, bc2] = floor_rgb; obs_b[b, :, nr, nc] = BOX_RGB
                    bx0[b] = int(s); bx1[b] = nr * W + nc; moved = True; break
            if moved:
                break
    tg_b = decode_tiles(obs_b)
    okb = np.array([bx1[b] >= 0 and tg_b[b, bx1[b]] == BOX and tg_b[b, bx0[b]] == FLOOR for b in range(B)])
    hb, embb = field(cps, policy, params, obs_b, K); Vb = decode(hb, probe); Vrb = readout_value(hb, embb, Wd, bd, Wc, bc, hs)
    agree_box, mag_b, dvr_b = [], [], []
    for b in range(B):
        if not okb[b] or G0[b] < 0:
            continue
        dT0n = bfs_from([int(G0[b])], tg_b[b], H, W); dd = dT0n - dT0[b]
        eu0 = np.hypot(RR - RR[bx0[b]], CC - CC[bx0[b]]); eu1 = np.hypot(RR - RR[bx1[b]], CC - CC[bx1[b]])
        far = (tiles[b] != WALL) & (eu0 >= 4) & (eu1 >= 4) & np.isfinite(dd) & np.isfinite(dT0[b])
        if far.sum() >= 6:
            agree_box.append(corr((Vb[b] - V0[b]), -dd, far))
        mag_b.append(far_field_dV_dec(Vb, V0, b, [bx0[b], bx1[b]])); dvr_b.append(abs(Vrb[b] - Vr0[b]) / (stdV + 1e-9))

    f = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    print(f"\n===== E6: VALUE-FIELD INTERVENTIONS vs RESOLVENT THEORY (step={step}, boards={B}, gamma={gamma}, std(V)={stdV:.2f}) =====")
    print(f"  value probe: linear h(s)->gamma^d(s,goal); orig fit corr = {f(base):.3f}")
    print(f"  -- (1) MOVE GOAL  (n={int(okg.sum())}) --")
    print(f"     corr(V_after, gamma^d to NEW goal)={f(track_new):+.3f}  vs OLD goal={f(track_old):+.3f}"
          f"  => {'RE-CENTERS on new goal (AGREES)' if f(track_new) > f(track_old)+0.1 else 'no clear re-center'}")
    print(f"  -- (2) ADD OBSTACLE on agent->goal path (n={len(agree_obs)}) --")
    print(f"     corr(dV(s), -Delta geodesic)={f(agree_obs):+.3f}  (theory >0)   shadowed cells that DROPPED={f(frac_drop):.2f} (theory ~1)")
    print(f"  -- (3) MOVE BOX  (n={int(okb.sum())}) --")
    print(f"     corr(dV(s), -Delta geodesic)={f(agree_box):+.3f}  (theory >0: box is an obstacle that re-routes value)")
    print(f"  -- IMPACT MAGNITUDE (how much the value moves) --")
    print(f"     {'edit':<14}{'decoded far |dV|':>18}{'readout |dV|/std':>18}")
    print(f"     {'move goal':<14}{f(mag_g):>18.3f}{f(dvr_g):>18.3f}")
    print(f"     {'add obstacle':<14}{f(mag_o):>18.3f}{f(dvr_o):>18.3f}")
    print(f"     {'move box':<14}{f(mag_b):>18.3f}{f(dvr_b):>18.3f}")
    biggest = max([("goal", f(dvr_g)), ("obstacle", f(dvr_o)), ("box", f(dvr_b))], key=lambda t: t[1])[0]
    print(f"     -> largest critic-value impact: MOVE {biggest.upper()}")
    print("PLOT_E6=" + repr(dict(
        goal=dict(track_new=round(f(track_new), 3), track_old=round(f(track_old), 3), dec=round(f(mag_g), 4), readout=round(f(dvr_g), 3)),
        obstacle=dict(agree=round(f(agree_obs), 3), dec=round(f(mag_o), 4), readout=round(f(dvr_o), 3)),
        box=dict(agree=round(f(agree_box), 3), dec=round(f(mag_b), 4), readout=round(f(dvr_b), 3)))))

    # ---------- export example fields for heatmaps ----------
    ex = [b for b in range(B) if okg[b] and oko[b] and okb[b]][:n_examples]
    rnd = lambda a: [round(float(x), 3) if np.isfinite(x) else None for x in a]
    for b in ex:
        print(f"EX_b={b} G0={int(G0[b])} G1={int(G1[b])} A0={int(A0[b])} OB={int(obcell[b])} BX0={int(bx0[b])} BX1={int(bx1[b])}")
        print(f"EX_tiles={list(map(int, tiles[b]))}")
        print(f"EX_V0={rnd(V0[b])}"); print(f"EX_Vgoal={rnd(V1[b])}"); print(f"EX_Vobs={rnd(Vo[b])}"); print(f"EX_Vbox={rnd(Vb[b])}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=192); ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.examples)
