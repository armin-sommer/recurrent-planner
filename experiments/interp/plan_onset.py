"""plan_onset -- IS IT DECISION-TIME PLANNING? tick x distance causal ACTION-onset probe.

The decisive test for decision-time planning by inward value propagation: a board change d graph-hops
from the agent should only alter the agent's ACTION after enough thinking steps for that information to
propagate inward. So the ONSET tick at which the action diverges from baseline should RISE with the
perturbation distance d (a change 4 hops out shows up in the action later than one 1 hop out), tracking
the latent arrival (E5). If instead the action diverges at the same tick for all d -- or distal changes
never reach the action -- it is not pulling distal info inward over ticks to decide.

We perturb a floor cell -> wall at graph-distance d from the agent, ALWAYS >=4 px away (outside the conv
receptive field, so any effect at the agent is RECURRENT propagation, not local pixels), in two flavours:
  on-path  : the geodesic cell at hop d (lengthens the agent->goal path  => decision-relevant)
  off-path : a non-geodesic cell at ~hop d whose removal does NOT change the agent->goal distance (cosmetic control)
For each tick t we record (vs the unperturbed board, matched tick): agent-cell latent shift, and whether
the model's argmax action differs. Onset(d) = first tick reaching 50% of that curve's settled value.

  python -m experiments.interp.plan_onset --ckpt <cp_dir> --boards 384 --dmax 9 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_RGB = np.array([0, 0, 0], np.uint8)


def geodesic_path(agent, dT, H, W):
    s = agent; path = [s]
    while np.isfinite(dT[s]) and dT[s] > 0:
        r, c = divmod(s, W); best, bd = None, dT[s]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            j = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[j]) and dT[j] < bd:
                bd = dT[j]; best = j
        if best is None:
            break
        s = best; path.append(s)
    return path


def onset(curve, frac=0.5):
    """first 1-indexed tick reaching frac * settled(final) value; nan if curve is ~flat-zero."""
    curve = np.asarray(curve, float); fin = curve[-1]
    if not np.isfinite(fin) or abs(fin) < 1e-6:
        return np.nan
    thr = frac * fin
    hit = np.where(curve >= thr)[0]
    return float(hit[0] + 1) if len(hit) else np.nan


def main(cp_dir, n_boards, dmax, K, rf_px):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wa = np.asarray(params["params"]["actor_params"]["Output"]["kernel"]); ba = np.asarray(params["params"]["actor_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W

    def emb_of(o): return np.asarray(get_embed(policy, params, jnp.asarray(o)))

    def th_of(o, chunk=64):                                   # (K, n, S, C), chunked to fit the GPU (recompute_d3 also builds the big attn tensor)
        outs = []
        for i in range(0, o.shape[0], chunk):
            e = emb_of(o[i:i + chunk])
            outs.append(np.asarray(recompute_d3(cps, jnp.asarray(e), K)[0]))
        return np.concatenate(outs, axis=1)

    def actions(th, emb):                                     # th (K,n,S,C), emb (n,H,W,C) -> (K,n)
        n = th.shape[1]; embr = emb.reshape(n, S, th.shape[-1])
        out = np.zeros((K, n), int)
        for t in range(K):
            mlp = np.maximum((th[t] + embr).reshape(n, S * th.shape[-1]) @ Wd + bd, 0.0)
            out[t] = (mlp @ Wa + ba).argmax(-1)
        return out

    # baseline forward (all boards; chunked recompute so the attn tensor fits)
    e0 = emb_of(obs0); th0 = th_of(obs0); a0 = actions(th0, e0)

    # build the perturbation list: (board b, distance d, kind, cell c)
    perts = []                                                # each: dict(b,d,kind,c)
    eu = lambda c, a: float(np.hypot(RR[c] - RR[a], CC[c] - CC[a]))
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]
        if not (len(ag) and len(tg)):
            continue
        a = int(ag[0]); t = int(tg[0])
        dT = bfs_from([t], tiles[b], H, W)
        if not np.isfinite(dT[a]):
            continue
        dA = bfs_from([a], tiles[b], H, W)                    # graph dist FROM agent
        path = geodesic_path(a, dT, H, W)
        pathset = set(path)
        for d in range(2, dmax + 1):
            # on-path: geodesic cell at hop d, floor, outside conv RF, lengthens the path
            if d < len(path):
                c = path[d]
                if tiles[b, c] == FLOOR and eu(c, a) >= rf_px:
                    t2 = tiles[b].copy(); t2[c] = WALL
                    if bfs_from([t], t2, H, W)[a] > dT[a]:
                        perts.append(dict(b=b, d=d, kind="on", c=c))
            # off-path: non-geodesic floor cell at graph-dist ~d, outside RF, removal does NOT lengthen path
            cand = [int(c) for c in np.where(tiles[b] == FLOOR)[0]
                    if c not in pathset and abs(dA[c] - d) <= 1 and eu(c, a) >= rf_px]
            for c in cand:
                t2 = tiles[b].copy(); t2[c] = WALL
                if bfs_from([t], t2, H, W)[a] == dT[a]:        # cosmetic: path length unchanged
                    perts.append(dict(b=b, d=d, kind="off", c=c)); break

    if not perts:
        print("NO valid perturbations found"); return
    # build perturbed observations
    Pn = len(perts)
    obsp = np.stack([obs0[p["b"]].copy() for p in perts])
    for i, p in enumerate(perts):
        obsp[i, :, RR[p["c"]], CC[p["c"]]] = WALL_RGB
    ep = np.concatenate([emb_of(obsp[i:i + 256]) for i in range(0, Pn, 256)], axis=0)
    thp = th_of(obsp)                                          # (K, Pn, S, C)
    ap = actions(thp, ep)                                      # (K, Pn)

    bidx = np.array([p["b"] for p in perts]); aidx = np.array([int(np.where(tiles[b] == AGENT)[0][0]) for b in bidx])
    dvec = np.array([p["d"] for p in perts]); kind = np.array([p["kind"] for p in perts])
    # per-pert latent shift at agent cell, per tick
    dh = np.zeros((K, Pn))
    for i in range(Pn):
        h0 = th0[:, bidx[i], aidx[i]]; hp = thp[:, i, aidx[i]]
        dh[:, i] = np.linalg.norm(hp - h0, axis=-1) / (np.linalg.norm(h0, axis=-1) + 1e-9)
    # per-pert action difference vs baseline (matched tick)
    adiff = (ap != a0[:, bidx]).astype(float)                 # (K, Pn)

    f = lambda xs: "[" + " ".join(("%.2f" % x if np.isfinite(x) else " . ") for x in xs) + "]"
    print(f"\n===== PLAN-ONSET: action divergence vs (distance d x tick) (step={step}, B={B}, perts={Pn}, K={K}, K_train={Ktr}, RF>={rf_px}px) =====")
    rows = {}
    for kd in ("on", "off"):
        print(f"\n  --- {kd}-path perturbations ---")
        print(f"    d   n   lat_onset  act_onset   act_change_rate_by_tick (1..K)")
        per_d = {}
        for d in range(2, dmax + 1):
            m = (dvec == d) & (kind == kd)
            n = int(m.sum())
            if n < 8:
                continue
            lat = dh[:, m].mean(1); act = adiff[:, m].mean(1)
            lo, ao = onset(lat), onset(act)
            per_d[d] = dict(n=n, lat=[round(float(x), 3) for x in lat], act=[round(float(x), 3) for x in act],
                            lat_onset=lo, act_onset=ao, act_final=float(act[-1]),
                            act_at_Ktr=float(act[Ktr - 1]))
            print(f"    {d:>2} {n:>3}   {('%.1f'%lo) if np.isfinite(lo) else ' . ':>6}     {('%.1f'%ao) if np.isfinite(ao) else ' . ':>6}     {f(act)}")
        rows[kd] = per_d
    # headline: do BOTH onsets rise with d (on-path)?
    on = rows.get("on", {})
    ds = sorted(on)
    lat_on = [on[d]["lat_onset"] for d in ds]; act_on = [on[d]["act_onset"] for d in ds]
    def slope(xs, ys):
        xs = np.array(xs, float); ys = np.array(ys, float); ok = np.isfinite(xs) & np.isfinite(ys)
        return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 3 else np.nan
    sl_lat, sl_act = slope(ds, lat_on), slope(ds, act_on)
    print(f"\n  on-path: distances tested = {ds}")
    print(f"    latent onset(d)  = {f(lat_on)}   slope vs d = {sl_lat:+.2f} ticks/hop")
    print(f"    action onset(d)  = {f(act_on)}   slope vs d = {sl_act:+.2f} ticks/hop")
    print(f"    action change @ settled, by d = {f([on[d]['act_final'] for d in ds])}")
    off = rows.get("off", {})
    if off:
        offd = sorted(off); print(f"    (control) off-path action change @ settled, by d = {f([off[d]['act_final'] for d in offd])}")
    # DECISION-RELEVANT onset: on-path action change ABOVE the matched off-path control (isolates planning from noise)
    diff_onset = {}; diff_curve = {}
    print(f"  on-minus-off action-change (decision-relevant excess) + its onset:")
    for d in ds:
        if d in off:
            diff = np.array(on[d]["act"]) - np.array(off[d]["act"])
            diff_curve[d] = [round(float(x), 3) for x in diff]
            o = onset(np.maximum.accumulate(diff), 0.5) if diff[-3:].mean() > 0.04 else np.nan
            diff_onset[d] = o
            print(f"    d={d:>2}: excess {f(list(diff))}  onset={('%.1f'%o) if np.isfinite(o) else ' . '}")
    dd = [d for d in ds if d in diff_onset and np.isfinite(diff_onset[d])]
    sl_diff = slope(dd, [diff_onset[d] for d in dd])
    print(f"    decision-relevant onset slope vs d = {sl_diff:+.2f} ticks/hop  (over d={dd})")
    verdict = (np.isfinite(sl_diff) and sl_diff > 0.4 and len(dd) >= 3)
    print(f"  --> action onset RISES with distance (staggered inward arrival = decision-time planning): {'YES' if verdict else 'NOT clearly'}")
    print("PLOT_ONSET=" + repr(dict(ds=ds, lat_onset=[round(x,2) if np.isfinite(x) else None for x in lat_on],
                                     act_onset=[round(x,2) if np.isfinite(x) else None for x in act_on],
                                     act_rate_on={d: on[d]["act"] for d in ds}, lat_on={d: on[d]["lat"] for d in ds},
                                     act_rate_off={d: off[d]["act"] for d in sorted(off)},
                                     diff_curve=diff_curve, diff_onset={d: (round(diff_onset[d],2) if np.isfinite(diff_onset[d]) else None) for d in diff_onset},
                                     act_final_on=[round(on[d]["act_final"],3) for d in ds],
                                     act_final_off=([round(off[d]["act_final"],3) for d in sorted(off)] if off else []),
                                     off_ds=(sorted(off) if off else []),
                                     Ktr=int(Ktr), slope_lat=round(sl_lat,3), slope_act=round(sl_act,3), slope_diff=round(sl_diff,3) if np.isfinite(sl_diff) else None)))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=384)
    ap.add_argument("--dmax", type=int, default=9); ap.add_argument("--ticks", type=int, default=12)
    ap.add_argument("--rf_px", type=float, default=4.0)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.dmax, a.ticks, a.rf_px)
