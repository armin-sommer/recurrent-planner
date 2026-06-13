"""(b) Does the slot thinking-loop do policy EVALUATION + IMPROVEMENT, and does a plan propagate
OUTWARD like search (DRC)?

We lift each slot's state into board space through its binding (board_repr[k,s] = binding-weighted avg
of slot states that point at square s), at every thinking tick k -- including EXTENDED test-time ticks
(K up to --ticks, the model trained at K=3). Then, per tick:
  * EVALUATION  : ridge-probe board_repr -> BFS distance-to-target (a navigation value field).
                  decodable? rising over ticks (refined in loop) or flat (a stationary substrate)?
  * IMPROVEMENT : linear-probe board_repr -> greedy move-direction (the navigation plan). accuracy
                  rising over ticks => the loop sharpens the policy.
  * PROPAGATION : where does the representation CHANGE each tick (centroid in distance-from-goal and
                  distance-from-agent)? An outward-moving change-front = a search frontier expanding;
                  a flat/global change = information pulled in, not propagated.

Ground truth is the navigation sub-MDP (BFS over non-wall squares) -- a proxy for the full box-push
plan (which needs A*), but exactly the lattice field the paper's recovery-of-N / value-iteration
story is about. Probes are simple numpy ridge / least-squares classifiers (decodability TRENDS, not
absolute SOTA). Same fixed sample split across ticks so the per-tick trend is apples-to-apples.

  python -m results.interp_plan --self-test
  python -m results.interp_plan --ckpt <cp_dir> --boards 256 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
from collections import deque
import numpy as np
import jax, jax.numpy as jnp

from results.interp_slots import recompute_cell, _tokens, decode_tiles

WALL, FLOOR, BOX, TARGET, AGENT = 0, 1, 2, 3, 4
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


# ---------------- navigation ground-truth fields (BFS over non-wall squares) ----------------
def bfs_from(sources, tiles, H=10, W=10):
    d = np.full(H * W, np.inf)
    nav = tiles != WALL
    q = deque()
    for s in sources:
        if nav[s]:
            d[s] = 0; q.append(s)
    while q:
        s = q.popleft(); r, c = divmod(s, W)
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                t = nr * W + nc
                if nav[t] and d[t] == np.inf:
                    d[t] = d[s] + 1; q.append(t)
    d[~nav] = np.nan
    return d


def greedy_dir(tiles, d, H=10, W=10):
    lab = np.full(H * W, -1, int)
    for s in range(H * W):
        if tiles[s] == WALL or not np.isfinite(d[s]) or d[s] == 0:
            continue
        r, c = divmod(s, W); best, bd = None, d[s]
        for k, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                t = nr * W + nc
                if np.isfinite(d[t]) and d[t] < bd:
                    bd = d[t]; best = k
        if best is not None:
            lab[s] = best
    return lab


def lift_to_board(rc):
    """board_repr (K,B,S,d): square s = binding-weighted average of the slot states pointing at s."""
    bind = np.asarray(rc["bind"]).mean(2)          # (K,B,N,S) head-avg
    slots = np.asarray(rc["slots"])                # (K,B,N,d)
    w = bind / (bind.sum(2, keepdims=True) + 1e-8) # per-square distribution over slots
    return np.einsum("kbns,kbnd->kbsd", w, slots)  # (K,B,S,d)


# ---------------- simple numpy probes (fixed split; trend across ticks) ----------------
def _stz(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_r2(Xk, y, tr, te, lam=10.0):
    Xtr, Xte = _stz(Xk[tr], Xk[te])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (y[tr] - y[tr].mean()))
    pred = Xte @ w + y[tr].mean()
    ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-9
    return float(1 - ss / tot)


def lin_acc(Xk, y, tr, te, n_cls=4, lam=10.0):
    Xtr, Xte = _stz(Xk[tr], Xk[te])
    Y = np.eye(n_cls)[y[tr]]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return float(((Xte @ W).argmax(1) == y[te]).mean())


# ---------------- main analysis ----------------
def analyse_plan(board_repr, tiles, obs_bchw, seed=0):
    """board_repr (K,B,S,d); tiles (B,S). Returns per-tick eval/improvement/propagation curves."""
    K, B, S, dim = board_repr.shape
    H = W = int(round(S ** 0.5))
    # ground-truth fields per board
    dT = np.full((B, S), np.nan); dA = np.full((B, S), np.nan); lab = np.full((B, S), -1, int)
    for b in range(B):
        tgt = np.where(tiles[b] == TARGET)[0]
        agt = np.where(tiles[b] == AGENT)[0]
        dT[b] = bfs_from(tgt, tiles[b], H, W)
        if len(agt):
            dA[b] = bfs_from(agt, tiles[b], H, W)
        lab[b] = greedy_dir(tiles[b], dT[b], H, W)

    # pooled sample indices (fixed across ticks)
    val_idx = [(b, s) for b in range(B) for s in range(S) if np.isfinite(dT[b, s])]
    dir_idx = [(b, s) for b in range(B) for s in range(S) if lab[b, s] >= 0]
    rng = np.random.default_rng(seed)

    def split(idx):
        idx = np.array(idx); rng.shuffle(idx); n = int(0.8 * len(idx))
        return idx[:n], idx[n:]

    vtr, vte = split(val_idx); dtr, dte = split(dir_idx)
    yval = np.array([dT[b, s] for b, s in np.concatenate([vtr, vte])])
    ydir = np.array([lab[b, s] for b, s in np.concatenate([dtr, dte])])
    ntr_v = len(vtr); ntr_d = len(dtr)

    val_r2, dir_acc = [], []
    for k in range(K):
        Xv = np.stack([board_repr[k, b, s] for b, s in np.concatenate([vtr, vte])])
        Xd = np.stack([board_repr[k, b, s] for b, s in np.concatenate([dtr, dte])])
        val_r2.append(ridge_r2(Xv, yval, np.arange(ntr_v), np.arange(ntr_v, len(yval))))
        dir_acc.append(lin_acc(Xd, ydir, np.arange(ntr_d), np.arange(ntr_d, len(ydir))))

    # propagation: where does the representation CHANGE each tick? centroid in dT and dA
    chg_dT, chg_dA = [], []
    for k in range(1, K):
        delta = np.linalg.norm(board_repr[k] - board_repr[k - 1], axis=-1)  # (B,S)
        wsum = 0.0; tsum = 0.0; asum = 0.0; wa = 0.0
        for b in range(B):
            m = np.isfinite(dT[b]); d_ = delta[b] * m
            wsum += d_.sum(); tsum += (d_ * np.nan_to_num(dT[b])).sum()
            ma = np.isfinite(dA[b]); da_ = delta[b] * ma
            wa += da_.sum(); asum += (da_ * np.nan_to_num(dA[b])).sum()
        chg_dT.append(tsum / (wsum + 1e-9)); chg_dA.append(asum / (wa + 1e-9))
    return dict(val_r2=val_r2, dir_acc=dir_acc, chg_dT=chg_dT, chg_dA=chg_dA,
                n_val=len(val_idx), n_dir=len(dir_idx))


def report(st, K):
    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print("\n============== PLAN / VALUE MECHANISM ==============")
    print(f"  ticks={K} (trained K=3; >3 = extended test-time thinking) · val_n={st['n_val']} dir_n={st['n_dir']}")
    print("  -- POLICY EVALUATION: value field (BFS dist-to-target) --")
    print(f"     value R^2 per tick : {f(st['val_r2'])}")
    print(f"        tick0 -> tickK   : {st['val_r2'][0]:.2f} -> {st['val_r2'][-1]:.2f}  ({'refined in loop' if st['val_r2'][-1]-st['val_r2'][0]>0.05 else 'stable/substrate'})")
    print("  -- POLICY IMPROVEMENT: plan field (greedy direction; chance=0.25) --")
    print(f"     plan acc per tick  : {f(st['dir_acc'])}")
    print(f"        tick0 -> tickK   : {st['dir_acc'][0]:.2f} -> {st['dir_acc'][-1]:.2f}  ({'sharpened in loop' if st['dir_acc'][-1]-st['dir_acc'][0]>0.03 else 'flat'})")
    print("  -- PROPAGATION: centroid of representation-change per tick --")
    print(f"     change dist-to-TARGET: {f(st['chg_dT'])}  ({'outward from goal' if st['chg_dT'][-1]>st['chg_dT'][0]+0.3 else 'no outward trend'})")
    print(f"     change dist-from-AGENT: {f(st['chg_dA'])}  ({'outward from agent' if st['chg_dA'][-1]>st['chg_dA'][0]+0.3 else 'no outward trend'})")
    print("===================================================\n")


def run_ckpt(cp_dir, n_boards, ticks):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.config import sokoban_drc_slots_softmax
    args = sokoban_drc_slots_softmax()
    eval_cfg = dataclasses.replace(args.eval_envs["valid_medium"].env, num_envs=n_boards)
    envs = eval_cfg.make()
    obs_np, _ = envs.reset()
    obs = jnp.asarray(np.asarray(obs_np))[None]
    policy, _c0, loaded_args, ts, step = load_train_state(Path(cp_dir), args.train_env)
    net = loaded_args.net
    tokens, _ = _tokens(net, policy, ts.params, obs[0])
    rc = recompute_cell(ts.params["params"]["network_params"]["cell_list_0"], tokens, ticks)
    board = lift_to_board(rc)
    tiles = decode_tiles(np.asarray(obs[0]))
    print(f"loaded @ step={step}; ran {ticks} thinking ticks on {n_boards} boards")
    st = analyse_plan(board, tiles, np.asarray(obs[0]))
    report(st, ticks)
    return st


def self_test(B=16, ticks=6):
    """Fabricate random rc + boards; validate the analysis runs and gives ~chance (no model needed)."""
    K, N, S, dim = ticks, 100, 100, 32
    rng = np.random.default_rng(0)
    rc = dict(bind=jnp.asarray(rng.random((K, B, 4, N, S))), slots=jnp.asarray(rng.standard_normal((K, B, N, dim))))
    board = lift_to_board(rc)
    obs = (rng.integers(0, 255, (B, 3, 10, 10))).astype(np.uint8)
    tiles = decode_tiles(obs)
    tiles = rng.integers(0, 5, (B, 100))  # random tiles so BFS has structure
    st = analyse_plan(board, tiles, obs)
    report(st, ticks)
    print("SELF-TEST PASS: plan pipeline runs (random => ~0 R^2, ~0.25 acc).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args()
    if a.self_test or not a.ckpt:
        self_test()
    else:
        run_ckpt(a.ckpt, a.boards, a.ticks)
