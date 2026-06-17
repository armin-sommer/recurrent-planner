"""E3 -- WHAT PROPAGATION ALGORITHM DID IT LEARN? The attention kernel vs GRAPH distance. (D=3 entmax.)

Each tick updates a cell's value by z=(A v)W_out -- a convex combination of OTHER cells weighted by the
entmax attention A. So "where does a cell's value get updated from" == the support of A by GRAPH distance
(BFS over N), and "what algorithm" == the SHAPE of that kernel and whether its reach grows over ticks:

  mass concentrated at d=1, grows reach with ticks       -> serial one-hop value iteration (Jacobi)
  flat within radius R, ~0 beyond, stable                -> k-hop neighbourhood aggregation
  geometric decay over MANY hops, broad, stable at tick1 -> AMORTIZED RESOLVENT: each cell directly reads
                                                            the whole reachable graph with ~gamma^d weight
                                                            (computes (I-gamma P)^-1 in one broad apply)
  mass on unreachable (behind-wall) cells                -> NOT graph-respecting

We bin A(s,k) by exact BFS graph-distance d(s,k), per tick and per head, and read the mass-by-distance
profile (rows of A sum to 1, so this is the fraction of each cell's update arriving from each hop-shell).

  python -m experiments.interp.e3 --ckpt <cp_dir> --boards 64 --ticks 8
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.plan import bfs_from, WALL
from experiments.interp.slots import decode_tiles


def pairwise_graphdist(tiles_b, H, W):
    """(S,) tiles -> (S,S) graph distance d(s,k): finite=reachable, inf=floor-but-unreachable, nan=wall."""
    S = H * W
    D = np.full((S, S), np.nan)
    for s in range(S):
        if tiles_b[s] != WALL:
            D[s] = bfs_from([s], tiles_b, H, W)     # nan at walls, inf if unreachable, else hops
    return D


def _profile(Aw, Dr, maxd=6):
    """Aw (Sq,S) attention rows, Dr (Sq,S) their graph distances. -> mass per distance shell, summed over rows."""
    bins = {}
    fin = np.isfinite(Dr)
    bins["self"] = Aw[(Dr == 0)].sum()
    for d in range(1, maxd + 1):
        bins[str(d)] = Aw[(Dr == d)].sum()
    bins[f">{maxd}"] = Aw[fin & (Dr > maxd)].sum()
    bins["unreach"] = Aw[np.isinf(Dr)].sum()
    bins["wall"] = Aw[np.isnan(Dr)].sum()
    return bins


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)
    B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    emb = np.asarray(get_embed(policy, params, jnp.asarray(obs0)))
    _, attn = recompute_d3(cps, jnp.asarray(emb), K)                                  # (K,D,B,nh,S,S)
    Atop = np.asarray(jnp.asarray(attn)[:, D - 1].mean(2))                            # head-avg top cell (K,B,S,S)
    Ahead_last = np.asarray(jnp.asarray(attn)[K - 1, D - 1])                          # (B,nh,S,S) last tick per head

    Dmats = [pairwise_graphdist(tiles[b], H, W) for b in range(B)]
    nonwall = [tiles[b] != WALL for b in range(B)]
    maxd = 6
    keys = ["self"] + [str(d) for d in range(1, maxd + 1)] + [f">{maxd}", "unreach", "wall"]

    def agg(Ak):                                                                      # Ak (B,S,S) -> mean mass per shell
        tot = {k: 0.0 for k in keys}; nq = 0
        for b in range(B):
            m = nonwall[b]; rows = Ak[b][m]; Dr = Dmats[b][m]
            p = _profile(rows, Dr, maxd)
            for k in keys:
                tot[k] += p[k]
            nq += int(m.sum())
        return {k: tot[k] / nq for k in keys}

    print(f"\n===== E3: ATTENTION PROPAGATION KERNEL vs GRAPH DISTANCE (step={step}, boards={B}, K={K}, trained {Ktr}) =====")
    print("  fraction of each cell's attention mass arriving from cells d hops away (rows sum to ~1):")
    print("  tick  " + "  ".join(f"{k:>7}" for k in keys))
    prof = {}
    for t in range(K):
        prof[t] = agg(Atop[t])
        print(f"  {t:>4}  " + "  ".join(f"{prof[t][k]:7.3f}" for k in keys))

    p1, pK = prof[0], prof[K - 1]
    cum = lambda p, upto: sum(p[k] for k in (["self"] + [str(d) for d in range(1, upto + 1)]))
    # per-hop attenuation (geometric?) on the converged kernel
    fr = [pK[str(d)] for d in range(1, maxd + 1)]
    ratios = [fr[d] / (fr[d - 1] + 1e-9) for d in range(1, maxd) if fr[d - 1] > 1e-4]
    print("\n  -- converged kernel (last tick) --")
    print(f"     mass within <=1 hop (self+d1) : {cum(pK,1):.3f}")
    print(f"     mass within <=3 hops          : {cum(pK,3):.3f}")
    print(f"     mass at >=4 hops (incl tail)  : {1 - cum(pK,3) - pK['unreach'] - pK['wall']:.3f}")
    print(f"     mass on UNREACHABLE (behind walls): {pK['unreach']:.3f}   (should be ~0 if graph-respecting)")
    print(f"     per-hop attenuation mass(d+1)/mass(d), d=1..: {['%.2f'%r for r in ratios]}  (constant => geometric/discounted-distance kernel)")
    print("\n  -- reach over ticks (serial vs one-shot) --")
    print(f"     mass beyond 3 hops:  tick0 {1-cum(p1,3)-p1['unreach']-p1['wall']:.3f}  ->  tick{K-1} {1-cum(pK,3)-pK['unreach']-pK['wall']:.3f}")
    print(f"     ({'GROWS with ticks -> serial propagation' if (1-cum(pK,3)) > (1-cum(p1,3)) + 0.05 else 'STABLE -> broad one-shot kernel (amortized), not serial one-hop'})")

    print("\n  -- per-head reach (last tick): mass beyond 2 hops, by head --")
    for h in range(Ahead_last.shape[1]):
        ph = agg(Ahead_last[:, h])                                                    # (B,S,S)
        beyond2 = 1 - cum(ph, 2) - ph["unreach"] - ph["wall"]
        print(f"     head {h}: <=1hop {cum(ph,1):.3f}  <=2hop {cum(ph,2):.3f}  >2hop {beyond2:.3f}")

    print("\n  -- VERDICT --")
    far = 1 - cum(pK, 3) - pK["unreach"] - pK["wall"]
    stable = (1 - cum(pK, 3)) <= (1 - cum(p1, 3)) + 0.05
    geometric = len(ratios) >= 3 and (max(ratios) - min(ratios) < 0.35)
    if pK["unreach"] < 0.03 and far > 0.05 and stable:
        print("     graph-respecting (no mass through walls), BROAD reach already at tick 1, stable over ticks,")
        print(f"     {'geometric/discounted-distance decay' if geometric else 'monotone decay'} -> AMORTIZED RESOLVENT-LIKE KERNEL:")
        print("     each cell directly reads the whole REACHABLE graph with hop-decaying weight (one broad apply,")
        print("     then refined), NOT serial one-hop value iteration.")
    elif not stable:
        print("     reach GROWS with ticks -> serial/iterative propagation (closer to textbook value iteration)")
    else:
        print(f"     concentrated within a few hops (>3hop mass {far:.2f}), stable -> k-hop neighbourhood aggregation")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=64); ap.add_argument("--ticks", type=int, default=8)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
