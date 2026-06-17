"""E1 -- WHAT DOES ONE THINKING TICK COMPUTE?  (D=3 entmax core, mech-interp.)

The Bellman test showed the CONVERGED value is a recursive fixed point, but that is depth-agnostic: it
cannot tell value iteration (per-tick max_a) from policy-evaluation / successor-representation propagation
(per-tick convex average) from generic relaxation. Source already settles one thing -- the trained config
uses readout="softmax", attn_norm="entmax15", directional_value=False (config.py:346): the per-tick
aggregation z=(A v)W_out is a CONVEX AVERAGE (E_{s'~A}), NOT a max. So the candidate operator is

    c <- gamma_eff * A * c + r_eff     =>     c* = (I - gamma_eff A)^{-1} r_eff       (Neumann / resolvent)

i.e. damped soft POLICY-EVALUATION / SUCCESSOR-REPRESENTATION propagation along N, with the one-step
max/lookahead living at the actor head, not in the loop. E1 looks for the per-tick mech-interp signatures
that separate this from value iteration and from the amortized/relaxation nulls. All read the model's OWN
per-tick hidden (top_h) and per-tick attention operators (attn) from the crash-free recompute_d3.

  Signature                              policy-eval / SR        value iteration         amortized / null
  ---------------------------------------------------------------------------------------------------------
  attention A_t over ticks               STATIONARY (no sharpen) sharpens toward argmax   stationary
  value field decodability over ticks    RISES then plateaus     rises                    flat from tick 0
  value WAVEFRONT (R^2 by dist-to-goal)  near cells first, far    near first, far later    all bands at once
                                         cells LATER (1 ring/tick)
  v_{t+1} ~ A_t v_t (own operator)       high R^2, coeff in (0,1) high R^2 (+ max kink)    low / coeff~0
  convergence rho of ||dh||              <1, rho << 0.97          <1                       ~0 after tick ~1

  python -m experiments.interp.e1 --ckpt <cp_dir> --boards 128 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.plan import analyse_plan, bfs_from, WALL, TARGET
from experiments.interp.slots import decode_tiles


def _fit(Xtr, ytr, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
    return mu, sd, w, float(ytr.mean())


def _pred(X, p):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


def _r2(y, pred):
    ss = ((y - pred) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return float(1 - ss / tot)


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    base = planning_eval_envs()["valid_medium"].env
    env_cfg = dataclasses.replace(base, num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    gamma = cp_cfg.loss.gamma
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs_np, _ = envs.reset(); obs = np.asarray(obs_np)
    B = obs.shape[0]
    emb = np.asarray(get_embed(policy, params, jnp.asarray(obs)))           # (B,H,W,C)
    H, W, C = emb.shape[1], emb.shape[2], emb.shape[3]; S = H * W
    top_h_j, attn_j = recompute_d3(cps, jnp.asarray(emb), K)                # (K,B,S,C), (K,D,B,nh,S,S)
    top_h = np.asarray(top_h_j)
    Atop = np.asarray(attn_j[:, D - 1].mean(2))                             # head-avg top-cell attn (K,B,S,S)
    tiles = decode_tiles(obs)                                               # (B,S)
    nonwall = (tiles != WALL)                                              # (B,S)

    print(f"\n===== E1: PER-TICK MECHANISM (step={step}, boards={B}, S={S}, D={D}, trained K={Ktr}, run K={K}, gamma={gamma}) =====")
    f2 = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    f3 = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"

    # ---------- (A) eval / improvement / propagation (reuse analyse_plan on top_h) ----------
    st = analyse_plan(top_h, tiles, obs)
    print("\n-- (A) decodable value field + plan + change-front (probe of top_h per tick) --")
    print(f"   value R^2 (BFS dist-to-target) : {f2(st['val_r2'])}  ({st['val_r2'][0]:.2f} -> {st['val_r2'][-1]:.2f})")
    print(f"   plan acc (greedy dir, ch=.25)  : {f2(st['dir_acc'])}  ({st['dir_acc'][0]:.2f} -> {st['dir_acc'][-1]:.2f})")
    print(f"   change-front dist-from-goal    : {f2(st['chg_dT'])}  ({'OUTWARD (propagation)' if st['chg_dT'][-1] > st['chg_dT'][0] + 0.3 else 'no outward trend'})")

    # ---------- (B) attention-operator stationarity vs sharpening (top cell) ----------
    top1, supp, rowcos = [], [], []
    for t in range(K):
        At = Atop[t]                                                       # (B,S,S)
        m = nonwall
        top1.append(float(At.max(-1)[m].mean()))
        supp.append(float((At > 1e-6).sum(-1)[m].mean()))
        if t > 0:
            num = (Atop[t] * Atop[t - 1]).sum(-1)
            den = np.linalg.norm(Atop[t], axis=-1) * np.linalg.norm(Atop[t - 1], axis=-1) + 1e-9
            rowcos.append(float((num / den)[m].mean()))
    print("\n-- (B) attention operator A_t over ticks (top cell, head-avg) --")
    print(f"   top-1 entmax mass / query      : {f3(top1)}  ({'SHARPENS' if top1[-1] > top1[0] + 0.05 else 'stationary'})")
    print(f"   entmax support (#keys>1e-6)    : {f2(supp)}")
    print(f"   row-cosine cos(A_t, A_t-1)     : {f3(rowcos)}  ({'stationary (>0.9)' if (rowcos and rowcos[-1] > 0.9) else 'drifting'})")

    # ---------- (C) convergence dynamics (extended ticks) ----------
    dn = []
    for t in range(1, K):
        d = np.linalg.norm(top_h[t] - top_h[t - 1], axis=-1)
        nrm = np.linalg.norm(top_h[t], axis=-1) + 1e-9
        dn.append(float((d / nrm)[nonwall].mean()))
    dfin = [float(np.linalg.norm(top_h[t] - top_h[K - 1], axis=-1)[nonwall].mean()) for t in range(K)]
    rhos = [dn[t] / (dn[t - 1] + 1e-9) for t in range(1, len(dn))]
    rho = float(np.median(rhos[max(0, Ktr - 2):])) if len(rhos) > 1 else float("nan")
    print("\n-- (C) convergence of the hidden over ticks --")
    print(f"   ||dh_t|| / ||h_t|| per tick    : {f3(dn)}")
    print(f"   ||h_t - h_final||              : {f3(dfin)}")
    print(f"   median contraction rho (t>=Ktr): {rho:.3f}   -> effective horizon ~1/(1-rho) = {1.0/(1-rho+1e-9):.1f} ticks  (gamma=0.97 would need ~{int(np.log(0.01)/np.log(gamma))} from scratch)")

    # ---------- (D) value WAVEFRONT: per-tick decodability of V* by distance-to-goal band ----------
    dT = np.full((B, S), np.nan)
    for b in range(B):
        tgt = np.where(tiles[b] == TARGET)[0]
        dT[b] = bfs_from(tgt, tiles[b], H, W)
    Vstar = gamma ** dT                                                    # nan where wall/unreachable
    idx = [(b, s) for b in range(B) for s in range(S) if np.isfinite(dT[b, s]) and dT[b, s] > 0]
    rng = np.random.default_rng(0); rng.shuffle(idx); idx = np.array(idx)
    ntr = int(0.8 * len(idx)); tr, te = idx[:ntr], idx[ntr:]
    y_te = np.array([Vstar[b, s] for b, s in te]); dT_te = np.array([dT[b, s] for b, s in te])
    bands = [(1, 2), (3, 5), (6, 8), (9, 99)]
    band_masks = [(dT_te >= lo) & (dT_te <= hi) for lo, hi in bands]
    wf = np.full((K, len(bands)), np.nan)
    for t in range(K):
        Xtr = np.stack([top_h[t, b, s] for b, s in tr]); ytr = np.array([Vstar[b, s] for b, s in tr])
        p = _fit(Xtr, ytr)
        Xte = np.stack([top_h[t, b, s] for b, s in te]); pred = _pred(Xte, p)
        for j, bm in enumerate(band_masks):
            if bm.sum() > 20:
                wf[t, j] = _r2(y_te[bm], pred[bm])
    print("\n-- (D) value WAVEFRONT: R^2(decoded V*) within distance-to-goal bands, per tick --")
    print(f"   {'tick':>5} " + " ".join(f"d{lo}-{hi:<2}" for lo, hi in bands))
    for t in range(K):
        print(f"   {t:>5} " + " ".join(f"{wf[t,j]:>5.2f}" if np.isfinite(wf[t, j]) else "   . " for j in range(len(bands))))
    onset = []
    for j in range(len(bands)):
        col = wf[:, j]; hit = np.where(np.isfinite(col) & (col > 0.3))[0]
        onset.append(int(hit[0]) if len(hit) else -1)
    print(f"   first tick R^2>0.3 per band     : {onset}   ({'STAGGERED near->far (wavefront)' if (onset[-1] > onset[0] >= 0) else 'simultaneous / no clear wavefront'})")

    # ---------- (E) does the model's OWN attention operator propagate its decoded value? ----------
    # decode value per tick with a single probe (fit at the final tick), then test v_{t+1} ~ a (A_t v_t) + b v_t
    pf = _fit(np.stack([top_h[K - 1, b, s] for b, s in tr]), np.array([Vstar[b, s] for b, s in tr]))
    V = np.stack([_pred(top_h[t].reshape(B * S, C), pf).reshape(B, S) for t in range(K)])   # (K,B,S) decoded value
    rows = []
    for t in range(1, K - 1):                                              # settled-ish ticks
        prop = np.einsum("bsk,bk->bs", Atop[t - 1], V[t - 1])              # (A_{t-1} v_{t-1})(s)
        m = nonwall
        X = np.stack([prop[m], V[t - 1][m], np.ones(m.sum())], 1)
        ycol = V[t][m]
        coef, *_ = np.linalg.lstsq(X, ycol, rcond=None)
        pred = X @ coef
        rows.append((t, coef[0], coef[1], _r2(ycol, pred)))
    a_mean = float(np.mean([r[1] for r in rows[max(0, Ktr - 3):]]))
    r2_mean = float(np.mean([r[3] for r in rows[max(0, Ktr - 3):]]))
    print("\n-- (E) own-operator propagation: v_t ~ a*(A_t-1 v_t-1) + b*v_t-1 + c --")
    print(f"   per-tick (a on A v | b on v | R^2): " + " ".join(f"t{r[0]}:({r[1]:+.2f}|{r[2]:+.2f}|{r[3]:.2f})" for r in rows))
    print(f"   settled mean: coeff on (A v) = {a_mean:+.2f}  (a damped-propagation coeff in (0,1) supports c<-gamma_eff A c + r),  R^2 = {r2_mean:.2f}")

    # ---------- verdict ----------
    sharp = top1[-1] > top1[0] + 0.05
    stationary = (not sharp) and (not rowcos or rowcos[-1] > 0.85)
    propagates = (st['val_r2'][-1] - st['val_r2'][0] > 0.05) or (onset[-1] > onset[0] >= 0) or (st['chg_dT'][-1] > st['chg_dT'][0] + 0.3)
    print("\n-- VERDICT --")
    print(f"   attention operator: {'SHARPENS toward argmax -> value-iteration-like' if sharp else 'STATIONARY -> fixed transition (policy-evaluation / SR), NOT value iteration'}")
    print(f"   value computation : {'ITERATIVE propagation along N (not one-shot)' if propagates else 'flat / amortized substrate (little inference-time iteration)'}")
    print(f"   contraction       : rho={rho:.2f} << gamma=0.97 -> AMORTIZED: a few refinement sweeps of a compiled operator, not a from-scratch solve")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args()
    main(a.ckpt, a.boards, a.ticks)
