"""THROUGH-WALLS test: at matched Euclidean (pixel) distance, does a perturbation BEHIND a wall (no/long
navigable path) influence the agent latent as much as an OPEN one? If yes -> routes through walls (spatial/
global, not relational). If blocked cells influence the agent less -> propagation respects 𝒩 (search).

Per board: flip one floor cell at Euclidean distance in [3.5,7] (beyond the ~7x7 conv RF, so the agent's
embed is unaffected -> any influence is via the recurrent attention). Record graph-distance (BFS around
walls; unreachable=inf) and Euclidean. Compare agent-cell influence (final ||Δh|| and onset tick) for
BLOCKED (graph≫euclid or unreachable) vs OPEN (graph≈euclid) cells at matched Euclidean distance.

  python -m experiments.interp.wall --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
import numpy as np
import jax, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, FLOOR, AGENT


def main(cp_dir, n_boards):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    rng = np.random.default_rng(0)

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)
    B, _, H, W = obs0.shape; tiles = decode_tiles(obs0)
    obs_p = obs0.copy(); ag_sq = np.full(B, -1); cell = np.full(B, -1); gd = np.full(B, np.nan); ed = np.full(B, np.nan)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]
        if not len(ag):
            continue
        a = int(ag[0]); ag_sq[b] = a; ar, ac = divmod(a, W)
        dist = bfs_from([a], tiles[b], H, W)                                   # finite if navigable, nan if not
        fl = np.where(tiles[b] == FLOOR)[0]; rr, cc = fl // W, fl % W
        euc = np.hypot(rr - ar, cc - ac)
        band = (euc >= 3.5) & (euc <= 7.0)                                     # beyond conv RF, on-board
        if band.sum() < 1:
            continue
        cand = fl[band]; gdc = dist[cand]; euc_b = euc[band]
        gdc = np.where(np.isfinite(gdc), gdc, 99.0)                            # unreachable -> 99
        # half boards: pick most-blocked (max graph/euclid, incl unreachable); half: most-open
        score = gdc / (euc_b + 1e-9); pick = np.argmax(score) if (b % 2 == 0) else np.argmin(score)
        c = int(cand[pick]); cell[b] = c; gd[b] = float(gdc[pick]); ed[b] = float(euc_b[pick])
        r, cq = divmod(c, W); obs_p[b, :, r, cq] = 0

    v = (ag_sq >= 0) & (cell >= 0) & np.isfinite(ed)
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    h0 = np.asarray(recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs0)), K)[0])
    hp = np.asarray(recompute_d3(cps, get_embed(policy, params, jnp.asarray(obs_p)), K)[0])
    fin = np.full(B, np.nan); com = np.full(B, np.nan)
    for b in range(B):
        if not v[b]:
            continue
        d = np.linalg.norm(hp[:, b, ag_sq[b]] - h0[:, b, ag_sq[b]], axis=-1)
        fin[b] = float(d[-1]); rel = d / (d[-1] + 1e-9)
        inc = np.diff(np.concatenate([[0.0], rel])); com[b] = float((np.arange(1, K + 1) * inc).sum() / (inc.sum() + 1e-9))

    g, e, F, C = gd[v], ed[v], fin[v], com[v]
    blocked = (g >= 99) | (g / (e + 1e-9) > 1.8); openc = (g / (e + 1e-9) < 1.2)
    pcorr = lambda y, x, z: np.corrcoef(y - np.polyval(np.polyfit(z, y, 1), z),
                                        x - np.polyval(np.polyfit(z, x, 1), z))[0, 1]
    print(f"\n== THROUGH-WALLS (step={step}, n={int(v.sum())}, euclid-band 3.5-7, K={K}) ==")
    print(f"  BLOCKED cells (graph≫euclid/unreachable, n={int(blocked.sum())}): final||Δh||={F[blocked].mean():.3f}  onset={C[blocked].mean():.2f}  euclid={e[blocked].mean():.1f}")
    print(f"  OPEN    cells (graph≈euclid,           n={int(openc.sum())}): final||Δh||={F[openc].mean():.3f}  onset={C[openc].mean():.2f}  euclid={e[openc].mean():.1f}")
    if blocked.sum() >= 4 and openc.sum() >= 4:
        ratio = F[blocked].mean() / (F[openc].mean() + 1e-9)
        print(f"  blocked/open influence ratio = {ratio:.2f}   partial corr(final, graph | euclid) = {pcorr(F, g, e):+.2f}")
        print(f"  --> {'ROUTES THROUGH WALLS (blocked≈open)' if ratio > 0.7 else 'RESPECTS WALLS (blocked≪open)'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
