"""E11 -- is there a decodable PLAN as a policy field? (per-node greedy action = the value gradient)

Hypothesis: value sits on every node and the action is its gradient (move toward the higher-value
neighbour). So from EACH node we should be able to decode "the action it wants", and the field of those
actions is the plan (arrows flowing to the goal). This is distinct from interp_planq (which found the
*executed trajectory* a1..a5 is NOT stored): the policy field is the value's gradient, present wherever
value is, not a stored sequence.

We (1) linear-probe the per-node latent h(s) -> greedy move-direction (4-way), held-out boards, per tick;
(2) test PLAN COHERENCE: follow the decoded field from the agent -- does it reach the goal? and what
fraction of nodes point goalward (toward lower BFS-distance)?; (3) tie to value: decode value(s)=gamma^d,
take its gradient (max-value neighbour), and check it agrees with the directly-decoded action -- i.e. the
action IS the value gradient. Export one board's field for a figure.

  python -m results.interp_e11_d3 --ckpt <cp_dir> --boards 192 --ticks 6
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from results.interp_planning_d3 import recompute_d3, get_embed
from results.interp_slots import decode_tiles
from results.interp_plan import bfs_from, greedy_dir, WALL, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right (indices 0..3)


def fit_dir(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Y = np.eye(4)[y]
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return mu, sd, W


def fit_val(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, float(y.mean())


def pred_dir(h, p):
    mu, sd, W = p
    return (((h - mu) / sd) @ W).argmax(-1)


def pred_val(h, p):
    mu, sd, w, b = p
    return ((h - mu) / sd) @ w + b


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; D = net.n_recurrent; gamma = cp_cfg.loss.gamma
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs); RR, CC = np.arange(S) // W, np.arange(S) % W
    top_h = np.asarray(recompute_d3(cps, jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs)))), K)[0])  # (K,B,S,C)

    dT = np.full((B, S), np.nan); gdir = np.full((B, S), -1, int); agent = np.full(B, -1)
    for b in range(B):
        tg = np.where(tiles[b] == TARGET)[0]; ag = np.where(tiles[b] == AGENT)[0]
        if len(tg) and len(ag):
            dT[b] = bfs_from([int(tg[0])], tiles[b], H, W); gdir[b] = greedy_dir(tiles[b], dT[b], H, W); agent[b] = int(ag[0])
    Vstar = gamma ** dT

    rng = np.random.default_rng(0); perm = rng.permutation(B); tr_b, te_b = perm[:int(0.8 * B)], perm[int(0.8 * B):]
    tr = [(b, s) for b in tr_b for s in range(S) if gdir[b, s] >= 0]
    te = [(b, s) for b in te_b for s in range(S) if gdir[b, s] >= 0]
    ytr = np.array([gdir[b, s] for b, s in tr])

    # (1) per-tick direction decodability (held-out)
    diracc = []
    for t in range(K):
        p = fit_dir(np.stack([top_h[t, b, s] for b, s in tr]), ytr)
        pr = pred_dir(np.stack([top_h[t, b, s] for b, s in te]), p)
        diracc.append(float((pr == np.array([gdir[b, s] for b, s in te])).mean()))

    # (2) plan coherence at the settled tick, on held-out boards
    pdir = fit_dir(np.stack([top_h[-1, b, s] for b, s in tr]), ytr)
    vpb = fit_val(np.stack([top_h[-1, b, s] for b, s in tr]), np.array([Vstar[b, s] for b, s in tr]))
    goalward, reach, vg_dir_agree, vg_true_agree, exrec = [], [], [], [], None
    for b in te_b:
        nonwall = np.where(tiles[b] != WALL)[0]
        pd = np.full(S, -1, int); pd[nonwall] = pred_dir(top_h[-1, b, nonwall], pdir)
        vv = np.full(S, -np.inf); vv[nonwall] = pred_val(top_h[-1, b, nonwall], vpb)
        # goalward: decoded action moves to a lower-dT neighbour
        for s in nonwall:
            if gdir[b, s] < 0:
                continue
            r, c = divmod(s, W); dr, dc = DIRS[pd[s]]; nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[b, nr * W + nc]):
                goalward.append(float(dT[b, nr * W + nc] < dT[b, s]))
        # value-gradient direction at each cell, vs decoded action and vs truth
        for s in nonwall:
            if gdir[b, s] < 0:
                continue
            r, c = divmod(s, W); best, bd = -1, -np.inf
            for di, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and tiles[b, nr * W + nc] != WALL and vv[nr * W + nc] > bd:
                    bd = vv[nr * W + nc]; best = di
            if best >= 0:
                vg_dir_agree.append(float(best == pd[s])); vg_true_agree.append(float(best == gdir[b, s]))
        # rollout: follow the decoded field from the agent
        cur = int(agent[b]); seen = set(); ok = False
        for _ in range(3 * (H + W)):
            if tiles[b, cur] == TARGET or (np.isfinite(dT[b, cur]) and dT[b, cur] == 0):
                ok = True; break
            if cur in seen:
                break
            seen.add(cur); r, c = divmod(cur, W); dr, dc = DIRS[pd[cur]]; nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or tiles[b, nr * W + nc] == WALL:
                break
            cur = nr * W + nc
        reach.append(float(ok))
        if exrec is None and agent[b] >= 0 and np.isfinite(dT[b, agent[b]]):
            exrec = (int(b), pd.copy(), np.where(np.isfinite(vv), vv, np.nan).copy())

    f = lambda x: float(np.mean(x)) if len(x) else float("nan")
    print(f"\n===== E11: DECODABLE PLAN AS A POLICY FIELD (step={step}, boards={B}, test={len(te_b)}, K={K}) =====")
    print(f"  (1) decode per-node greedy action h(s)->dir (chance 0.25), by tick: " + "[" + " ".join("%.2f" % a for a in diracc) + "]")
    print(f"      settled dir-decode accuracy = {diracc[-1]:.3f}")
    print(f"  (2) plan coherence (held-out boards):")
    print(f"      fraction of nodes whose decoded action points GOALWARD (lower BFS-dist) = {f(goalward):.3f} (chance ~0.33)")
    print(f"      follow the decoded field from the agent -> reaches the goal on {f(reach):.3f} of boards")
    print(f"  (3) action = value gradient:")
    print(f"      decoded action == value-gradient direction (max-value neighbour) on {f(vg_dir_agree):.3f}")
    print(f"      value-gradient direction == true greedy direction on {f(vg_true_agree):.3f}")
    print("PLOT_E11=" + repr(dict(diracc=[round(a, 3) for a in diracc], goalward=round(f(goalward), 3),
                                   reach=round(f(reach), 3), vg_dir=round(f(vg_dir_agree), 3), vg_true=round(f(vg_true_agree), 3))))
    if exrec is not None:
        b, pd, vv = exrec
        print(f"EX_b={b} agent={int(agent[b])} goal={int(np.where(tiles[b]==TARGET)[0][0])} H={H} W={W}")
        print("EX_tiles=" + repr(list(map(int, tiles[b]))))
        print("EX_pdir=" + repr(list(map(int, pd))))
        print("EX_truedir=" + repr(list(map(int, gdir[b]))))
        print("EX_val=" + repr([round(float(x), 3) if np.isfinite(x) else None for x in vv]))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=192); ap.add_argument("--ticks", type=int, default=6)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
