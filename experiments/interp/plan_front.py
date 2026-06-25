"""plan_front -- the DISCRIMINATING decision-time-planning test: a latent-change FRONT propagating inward.

Rigorous version addressing the adversarial review of plan_onset. Generic hidden-settling makes every cell's
||dh|| rise together (arrival tick flat in distance); a genuine inward PROPAGATION makes the change reach
cells near the perturbation first and the agent LAST -- a moving front whose arrival tick rises with
graph-distance-from-source. We test exactly that:

 - perturb a path-lengthening wall at a FAR on-path 'source' cell, Chebyshev>=4 from the agent (its effect on
   the agent is therefore recurrent, not local conv);
 - measure normalized latent shift ||dh|| at every on-path cell between agent and source, per thinking tick,
   EXCLUDING any cell within Chebyshev `rf` of the source (drop conv leakage near the source);
 - bin cells by graph-hops-from-source j; arrival_tick(j) = first tick reaching max(0.5*value@trained-depth,
   floor), measured WITHIN the trained depth Ktr; a positive arrival-vs-j slope = inward front;
 - bootstrap 95% CIs over boards on the slope; compare to an OFF-path cosmetic source (does NOT lengthen the
   path) which should not drive a decision-relevant front (smaller magnitude / flatter slope).

  python -m experiments.interp.plan_front --ckpt <cp> --boards 768 --ticks 8 --rf 4
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
            nr, nc = r + dr, c + dc; j = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[j]) and dT[j] < bd:
                bd = dT[j]; best = j
        if best is None:
            break
        s = best; path.append(s)
    return path


def arrival(curve, Ktr, frac=0.5, floor=0.05):
    """first 1-indexed tick within Ktr reaching max(frac*value@Ktr, floor); nan if never."""
    c = np.asarray(curve, float); ref = c[Ktr - 1]
    if ref < floor:
        return np.nan
    thr = max(frac * ref, floor)
    for t in range(Ktr):
        if c[t] >= thr:
            return float(t + 1)
    return np.nan


def main(cp_dir, n_boards, K, rf):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W
    cheb = lambda a, b: max(abs(RR[a] - RR[b]), abs(CC[a] - CC[b]))

    def emb_of(o): return np.asarray(get_embed(policy, params, jnp.asarray(o)))
    def th_of(o, chunk=64):
        return np.concatenate([np.asarray(recompute_d3(cps, jnp.asarray(emb_of(o[i:i+chunk])), K)[0])
                               for i in range(0, o.shape[0], chunk)], axis=1)   # (K,n,S,C)

    th0 = th_of(obs0)

    # pick, per board: a FAR on-path source (path-lengthening, cheb>=4 from agent) + an off-path cosmetic source
    recs = []   # dict(b, src_on, src_off, path, s, agent)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]; fl = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl)):
            continue
        a = int(ag[0]); t = int(tg[0]); dT = bfs_from([t], tiles[b], H, W)
        if not np.isfinite(dT[a]):
            continue
        path = geodesic_path(a, dT, H, W); pathset = set(path)
        # on-path source: farthest path cell that is floor, cheb>=4 from agent, and lengthens dT[a]
        src_on = -1; s_hop = -1
        for k in range(len(path) - 1, rf, -1):
            c = path[k]
            if tiles[b, c] == FLOOR and cheb(c, a) >= rf:
                t2 = tiles[b].copy(); t2[c] = WALL
                if bfs_from([t], t2, H, W)[a] > dT[a]:
                    src_on = c; s_hop = k; break
        if src_on < 0 or s_hop < 6:
            continue
        # off-path cosmetic source near the on-path source's pixel, not on path, non-lengthening
        eu = lambda c1, c2: np.hypot(RR[c1] - RR[c2], CC[c1] - CC[c2])
        cand = [int(c) for c in fl if c not in pathset and cheb(c, a) >= rf and eu(c, src_on) <= 3.0]
        src_off = -1
        for c in cand:
            t2 = tiles[b].copy(); t2[c] = WALL
            if bfs_from([t], t2, H, W)[a] == dT[a]:
                src_off = c; break
        recs.append(dict(b=b, src_on=src_on, src_off=src_off, path=path, s=s_hop, a=a))

    if not recs:
        print("NO valid boards"); return

    def perturbed(field):                                  # field: 'src_on' or 'src_off'
        idx = [r for r in recs if r[field] >= 0]
        o = np.stack([obs0[r["b"]].copy() for r in idx])
        for i, r in enumerate(idx):
            c = r[field]; o[i, :, RR[c], CC[c]] = WALL_RGB
        return idx, th_of(o)

    # collect per-(board,j) latent-shift curves; j = graph hops from source (s_hop - k), cells with cheb>=rf from source
    def collect(field):
        idx, thp = perturbed(field)
        per_j = {}        # j -> list of per-board ||dh|| curves (K,)
        agent_curves = []
        for i, r in enumerate(idx):
            b = r["b"]; path = r["path"]; s = r["s"]; src = r[field]
            for k, c in enumerate(path[:s + 1]):
                if cheb(c, src) < rf:                      # drop conv-leak neighbourhood of the source
                    continue
                h0 = th0[:, b, c]; hp = thp[:, i, c]
                dh = np.linalg.norm(hp - h0, axis=-1) / (np.linalg.norm(h0, axis=-1) + 1e-9)
                j = s - k
                per_j.setdefault(j, []).append(dh)
                if k == 0:
                    agent_curves.append(dh)
        return per_j, np.array(agent_curves), len(idx)

    f = lambda xs: "[" + " ".join(("%.2f" % x if np.isfinite(x) else " . ") for x in xs) + "]"

    def analyse(per_j, label):
        js = sorted([j for j in per_j if len(per_j[j]) >= 20])
        means = {j: np.stack(per_j[j]).mean(0) for j in js}
        arr = {j: arrival(means[j], Ktr) for j in js}
        # slope of arrival vs j (front => positive), bootstrap CI over boards
        def slope_from(sample_per_j):
            jj, aa = [], []
            for j in js:
                cur = np.stack(sample_per_j[j]).mean(0); av = arrival(cur, Ktr)
                if np.isfinite(av):
                    jj.append(j); aa.append(av)
            return float(np.polyfit(jj, aa, 1)[0]) if len(jj) >= 3 else np.nan
        sl = slope_from(per_j)
        rng = np.random.RandomState(0); boots = []
        # bootstrap: resample the per-j curve lists independently (board-level approx)
        for _ in range(400):
            samp = {}
            for j in js:
                arrs = per_j[j]; ridx = rng.randint(0, len(arrs), len(arrs))
                samp[j] = [arrs[ii] for ii in ridx]
            s2 = slope_from(samp)
            if np.isfinite(s2):
                boots.append(s2)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        print(f"\n  --- {label} ---")
        print(f"    j(hops from source)  n     arrival(<=Ktr)   ||dh|| by tick (1..K)")
        for j in js:
            print(f"      {j:>2}  {len(per_j[j]):>4}      {('%.1f'%arr[j]) if np.isfinite(arr[j]) else ' . ':>5}        {f(means[j])}")
        print(f"    arrival-vs-j slope = {sl:+.2f} ticks/hop   95% CI [{lo:+.2f}, {hi:+.2f}]   (positive + CI>0 => inward FRONT, not settling)")
        return dict(js=js, arr={int(j): (round(float(arr[j]),2) if np.isfinite(arr[j]) else None) for j in js},
                    means={int(j): [round(float(x),3) for x in means[j]] for j in js},
                    slope=round(float(sl),3) if np.isfinite(sl) else None,
                    ci=[round(float(lo),3) if np.isfinite(lo) else None, round(float(hi),3) if np.isfinite(hi) else None])

    print(f"\n===== PLAN-FRONT: inward latent propagation front (step={step}, boards~{len(recs)}, K={K}, Ktr={Ktr}, RF-gate cheb>={rf}) =====")
    on_pj, on_ag, n_on = collect("src_on")
    off_pj, off_ag, n_off = collect("src_off")
    RON = analyse(on_pj, "on_path source (path-lengthening, decision-relevant)")
    ROFF = analyse(off_pj, "off_path source (cosmetic control)")
    # agent-cell arrival on vs off
    ag_on = on_ag.mean(0) if len(on_ag) else np.zeros(K); ag_off = off_ag.mean(0) if len(off_ag) else np.zeros(K)
    print(f"\n  agent-cell ||dh|| (the front's destination):")
    print(f"     on-path : {f(ag_on)}   arrival={arrival(ag_on,Ktr)}")
    print(f"     off-path: {f(ag_off)}   arrival={arrival(ag_off,Ktr)}")
    front = (RON["slope"] is not None and RON["ci"][0] is not None and RON["ci"][0] > 0)
    print(f"\n  --> inward FRONT present (arrival rises with hops-from-source, CI>0): {'YES' if front else 'NOT clearly'}")
    print(f"      discriminates propagation from generic settling (which gives slope~0 / flat arrival).")
    print("PLOT_FRONT=" + repr(dict(on=RON, off=ROFF, n_on=n_on, n_off=n_off, Ktr=int(Ktr),
                                    agent_on=[round(float(x),3) for x in ag_on], agent_off=[round(float(x),3) for x in ag_off])))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=768)
    ap.add_argument("--ticks", type=int, default=8); ap.add_argument("--rf", type=int, default=4)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks, a.rf)
