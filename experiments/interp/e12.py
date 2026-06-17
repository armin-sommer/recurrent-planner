"""E12 -- does the decoded PLAN (policy field) expand OUTWARD from the goal over thinking ticks?

E11 found the per-node action is decodable (~0.44) and flat in aggregate across ticks. But that average
hides spatial structure: if the plan grows like a frontier, near-goal nodes have a correct action early and
far nodes only once value has propagated to them (their accuracy rises at LATER ticks). If the field is set
at once, all distance bands are flat. We fit a per-tick direction probe h_t(s)->greedy-move and measure
test accuracy + goalward-fraction binned by graph-distance-to-goal, per tick.

  expanding frontier : near bands high early; far bands rise, with LATER onset  (accuracy front moves out)
  static field       : all bands ~flat across ticks

  python -m experiments.interp.e12 --ckpt <cp_dir> --boards 192 --ticks 8
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, greedy_dir, WALL, TARGET

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def fit_dir(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ np.eye(4)[y])
    return mu, sd, W


def pred_dir(h, p):
    mu, sd, W = p
    return (((h - mu) / sd) @ W).argmax(-1)


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs); RR, CC = np.arange(S) // W, np.arange(S) % W
    top_h = np.asarray(recompute_d3(cps, jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs)))), K)[0])

    dT = np.full((B, S), np.nan); gdir = np.full((B, S), -1, int)
    for b in range(B):
        tg = np.where(tiles[b] == TARGET)[0]
        if len(tg):
            dT[b] = bfs_from([int(tg[0])], tiles[b], H, W); gdir[b] = greedy_dir(tiles[b], dT[b], H, W)

    rng = np.random.default_rng(0); perm = rng.permutation(B); tr_b, te_b = perm[:int(0.8 * B)], perm[int(0.8 * B):]
    tr = [(b, s) for b in tr_b for s in range(S) if gdir[b, s] >= 0]
    te = [(b, s) for b in te_b for s in range(S) if gdir[b, s] >= 0]
    ytr = np.array([gdir[b, s] for b, s in tr])
    yte = np.array([gdir[b, s] for b, s in te]); dte = np.array([dT[b, s] for b, s in te])
    bands = [(1, 2), (3, 5), (6, 8), (9, 99)]
    bmask = [(dte >= lo) & (dte <= hi) for lo, hi in bands]

    acc = np.zeros((K, len(bands))); gw = np.zeros((K, len(bands)))
    for t in range(K):
        p = fit_dir(np.stack([top_h[t, b, s] for b, s in tr]), ytr)
        pr = pred_dir(np.stack([top_h[t, b, s] for b, s in te]), p)
        # goalward per test cell: predicted dir moves to a lower-dT neighbour
        gwc = np.zeros(len(te), bool)
        for i, (b, s) in enumerate(te):
            r, c = divmod(s, W); dr, dc = DIRS[pr[i]]; nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[b, nr * W + nc]):
                gwc[i] = dT[b, nr * W + nc] < dT[b, s]
        for j, m in enumerate(bmask):
            if m.sum() >= 20:
                acc[t, j] = (pr[m] == yte[m]).mean(); gw[t, j] = gwc[m].mean()
            else:
                acc[t, j] = np.nan; gw[t, j] = np.nan

    f = lambda xs: "[" + " ".join(("%.2f" % x if np.isfinite(x) else " . ") for x in xs) + "]"
    bl = [f"d{lo}-{hi if hi < 99 else '+'}" for lo, hi in bands]
    print(f"\n===== E12: DOES THE PLAN EXPAND OUTWARD? (step={step}, boards={B}, test={len(te_b)}, K={K}, chance 0.25) =====")
    print(f"  decoded-action ACCURACY by distance-to-goal band, per tick:")
    print(f"     tick \\ band   " + "  ".join(f"{b:>5}" for b in bl))
    for t in range(K):
        print(f"     {t+1:>4}        " + "  ".join(f"{acc[t,j]:5.2f}" if np.isfinite(acc[t,j]) else "  .  " for j in range(len(bands))))
    print(f"  GOALWARD fraction by band, per tick:")
    for t in range(K):
        print(f"     {t+1:>4}        " + "  ".join(f"{gw[t,j]:5.2f}" if np.isfinite(gw[t,j]) else "  .  " for j in range(len(bands))))
    print(f"  per-band change tick1->K (accuracy):")
    for j, b in enumerate(bl):
        v = acc[:, j][np.isfinite(acc[:, j])]
        print(f"     {b:>6}: {v[0]:.2f} -> {v[-1]:.2f}  (delta {v[-1]-v[0]:+.2f})" if len(v) else f"     {b:>6}: n/a")
    # frontier: first tick each band's goalward exceeds 0.5
    onset = []
    for j in range(len(bands)):
        col = gw[:, j]; hit = np.where(np.isfinite(col) & (col > 0.5))[0]
        onset.append(int(hit[0]) + 1 if len(hit) else -1)
    print(f"  first tick goalward>0.5 per band: {onset}  ({'STAGGERED near->far = EXPANDS OUTWARD' if (onset[-1] > onset[0] >= 1) else 'simultaneous / no clear outward frontier'})")
    print("PLOT_E12=" + repr(dict(bands=bl, acc=[[round(float(acc[t,j]),3) if np.isfinite(acc[t,j]) else None for j in range(len(bands))] for t in range(K)],
                                   gw=[[round(float(gw[t,j]),3) if np.isfinite(gw[t,j]) else None for j in range(len(bands))] for t in range(K)], onset=onset)))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=192); ap.add_argument("--ticks", type=int, default=8)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
