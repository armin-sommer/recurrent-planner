"""E6b -- pointed interventions on the READOUT value, with enough thinking time. (D=3 entmax)

Fixes E6: (1) relocate the goal / a box to the FAR side of the board (large displacement -> the probe has
signal), (2) give the loop EXTENDED thinking (K=16) and read the critic value V at EVERY tick, so we can
see whether it ADAPTS to the new board as thinking proceeds (a far goal is ~10-15 hops away; at the
trained K=6 the value field has not had time to re-form). Theory (policy-evaluation value field):

  * MOVE GOAL to far side  -> V tracks -Delta(agent->goal geodesic): closer goal => higher V.
  * MOVE BOX  to far side  -> if the value is box-PUSH-aware, V tracks -Delta(box->nearest-target
                              geodesic): box nearer a target => higher V.

We report corr(dV_t, -Delta dist) at each tick t; theory predicts it is positive and STRENGTHENS with
thinking (the field re-forms over ticks). Readout value = trained head; edits are re-decoded to verify.

  python -m experiments.interp.e6b --ckpt <cp_dir> --boards 256 --ticks 16
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

TARGET_RGB = np.array([254, 126, 125], np.uint8)
BOX_RGB = np.array([142, 121, 56], np.uint8)


def readout_all_ticks(top_h, emb, Wd, bd, Wc, bc, hs):
    K, B, S, C = top_h.shape
    embf = emb.reshape(B, S, C)
    out = np.zeros((K, B))
    for t in range(K):
        mlp = np.maximum((top_h[t] + embf).reshape(B, S * C) @ Wd + bd, 0.0)
        out[t] = (mlp @ Wc + bc)[..., 0] * hs
    return out                                                                   # (K,B)


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
    tiles = decode_tiles(obs0); rc = np.arange(S); RR, CC = rc // W, rc % W

    def far_floor(b, frm, banned):
        fl = np.where(tiles[b] == FLOOR)[0]
        fl = np.array([s for s in fl if s not in banned])
        if not len(fl):
            return -1
        return int(fl[np.argmax(np.hypot(RR[fl] - RR[frm], CC[fl] - CC[frm]))])   # opposite side

    obs_g = obs0.copy(); obs_b = obs0.copy()
    G0 = np.full(B, -1); A0 = np.full(B, -1); G1 = np.full(B, -1); BX0 = np.full(B, -1); BX1 = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]
        tgts = set(map(int, tg)); fl0 = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl0)):
            continue
        a = int(ag[0]); A0[b] = a; G0[b] = int(tg[0])
        fr, fc = divmod(int(fl0[0]), W); floor_rgb = obs0[b, :, fr, fc]
        # goal -> far side (not on agent)
        g1 = far_floor(b, G0[b], {a})
        if g1 >= 0:
            G1[b] = g1; obs_g[b, :, RR[G0[b]], CC[G0[b]]] = floor_rgb; obs_g[b, :, RR[g1], CC[g1]] = TARGET_RGB
        # a plain box -> far side (not on agent/target)
        boxes = [s for s in np.where(tiles[b] == BOX)[0] if np.array_equal(obs0[b, :, RR[s], CC[s]], BOX_RGB)]
        if boxes:
            bx = int(boxes[0]); b1 = far_floor(b, bx, {a} | tgts)
            if b1 >= 0:
                BX0[b] = bx; BX1[b] = b1; obs_b[b, :, RR[bx], CC[bx]] = floor_rgb; obs_b[b, :, RR[b1], CC[b1]] = BOX_RGB

    tg_g = decode_tiles(obs_g); tg_b = decode_tiles(obs_b)
    okg = np.array([G1[b] >= 0 and tg_g[b, G1[b]] == TARGET and tg_g[b, G0[b]] == FLOOR for b in range(B)])
    okb = np.array([BX1[b] >= 0 and tg_b[b, BX1[b]] == BOX and tg_b[b, BX0[b]] == FLOOR for b in range(B)])

    def perTickV(o):
        emb = np.asarray(get_embed(policy, params, jnp.asarray(o)))
        top_h = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])
        return readout_all_ticks(top_h, emb, Wd, bd, Wc, bc, hs)                  # (K,B)
    V0 = perTickV(obs0); Vg = perTickV(obs_g); Vb = perTickV(obs_b)
    stdV = float(np.std(V0[-1]))

    # distance changes (geodesic over non-wall graph)
    dd_goal = np.full(B, np.nan); dd_box = np.full(B, np.nan)
    for b in range(B):
        if okg[b]:
            d_new = bfs_from([int(G1[b])], tg_g[b], H, W); d_old = bfs_from([int(G0[b])], tiles[b], H, W)
            dd_goal[b] = d_new[A0[b]] - d_old[A0[b]]                              # agent->goal distance change
        if okb[b]:
            dt_new = bfs_from(list(np.where(tg_b[b] == TARGET)[0]), tg_b[b], H, W)
            dt_old = bfs_from(list(np.where(tiles[b] == TARGET)[0]), tiles[b], H, W)
            dd_box[b] = dt_new[int(BX1[b])] - dt_old[int(BX0[b])]                 # box->nearest-target distance change

    print(f"\n===== E6b: POINTED INTERVENTIONS, READOUT VALUE, EXTENDED THINKING (step={step}, boards={B}, trained K={Ktr}, run K={K}, std(V)={stdV:.2f}) =====")
    print(f"  clean: goal-move n={int(okg.sum())}, box-move n={int(okb.sum())};  goal relocated to far side, box relocated to far side")
    ag_g = np.array([corr(Vg[t] - V0[t], -dd_goal) for t in range(K)])
    ag_b = np.array([corr(Vb[t] - V0[t], -dd_box) for t in range(K)])
    f3 = lambda xs: "[" + " ".join(("%+.2f" % x if np.isfinite(x) else "  nan") for x in xs) + "]"
    print(f"  -- corr(dV_t, -Delta dist) by thinking tick (theory: positive, strengthening) --")
    print(f"     tick:                 " + " ".join(f"{t+1:>5}" for t in range(K)))
    print(f"     MOVE GOAL (agent->goal): " + " ".join(f"{ag_g[t]:+5.2f}" if np.isfinite(ag_g[t]) else "  nan" for t in range(K)))
    print(f"     MOVE BOX  (box->target): " + " ".join(f"{ag_b[t]:+5.2f}" if np.isfinite(ag_b[t]) else "  nan" for t in range(K)))
    mg = float(np.nanmean(np.abs(Vg[-1] - V0[-1])[okg])) / (stdV + 1e-9)
    mb = float(np.nanmean(np.abs(Vb[-1] - V0[-1])[okb])) / (stdV + 1e-9)
    print(f"  -- final-tick |dV|/std --   move goal: {mg:.3f}    move box: {mb:.3f}")
    print(f"  -- agreement: trained K={Ktr} vs full K={K} --")
    print(f"     goal: {ag_g[Ktr-1]:+.2f} -> {ag_g[-1]:+.2f}    box: {ag_b[Ktr-1]:+.2f} -> {ag_b[-1]:+.2f}")
    gg = np.nanmax(ag_g); gb = np.nanmax(ag_b)
    print(f"  --> goal value tracks distance: {'YES' if gg > 0.15 else 'no'} (peak {gg:+.2f}); "
          f"box value box-push-aware: {'YES' if gb > 0.15 else 'no'} (peak {gb:+.2f})")
    print("PLOT_E6B=" + repr(dict(goal=[round(float(x), 3) if np.isfinite(x) else None for x in ag_g],
                                   box=[round(float(x), 3) if np.isfinite(x) else None for x in ag_b],
                                   mg=round(mg, 3), mb=round(mb, 3), Ktr=int(Ktr))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256); ap.add_argument("--ticks", type=int, default=16)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
