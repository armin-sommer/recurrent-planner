"""E1 (SLOT CORE) -- WHAT DOES ONE THINKING TICK COMPUTE?  per-tick OPERATOR + AMORTIZED value.

Port of experiments/interp/e1.py from the spatial ATTENTION core to the learnable-SLOT core
(cleanba.slot_lstm). The original E1 looked for the per-tick mech-interp signatures that separate damped
soft POLICY-EVALUATION / SR propagation (c <- gamma_eff A c + r_eff) from value iteration and from an
amortized/relaxation null. The same questions transfer; only the substrate changes.

  attn core (original)                       slot core (this port)
  -----------------------------------------  ---------------------------------------------------------
  N = H*W cells, cell i == board square i    N FREE slots, NOT tied to board position
  per-tick operator A_t = top-cell attn S->S per-tick operator = ROUTE (slot<->slot) N->N
  "cell i" value target = BFS dist at i      slot value target = BFS dist at the slot's BOUND position
                                             (via the LEARNED sigma: pos[b,slot] = decode_sigma(bind))
  v_{t+1} ~ A_t v_t   (own operator)         v_{t+1} ~ ROUTE_t v_t   (own operator)
  graph dist between cells i,j == |i-j| grid graph dist == BFS board dist between pos[i] and pos[j]

To talk about a board square we go THROUGH sigma: a slot's "value" is the navigation value of the board
square it binds; we lift slot hidden into board space (binding-weighted avg) for the analyse_plan curves,
and we band slots by their bound-position distance-to-goal for the wavefront. The AMORTIZED test (the
focus of E1) regresses the per-tick decoded slot-value on (ROUTE @ v_{t-1}) and v_{t-1}: a compiled
operator that has already converged shows a coeff that is non-zero only on the first tick and ||dv||/||v||
collapsing to ~0; an inference-time value-iteration would keep a non-zero propagation coeff every tick.

  python -m experiments.interp.e1_slot --ckpt <cp_dir> --boards 256 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.plan import analyse_plan, bfs_from, greedy_dir, WALL, FLOOR, BOX, TARGET, AGENT
from experiments.interp.slots import decode_tiles


# ---------------- simple ridge probe (same as the original E1) ----------------
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


def _lift_to_board(h, bind):
    """board_repr (K,B,S,d): board square s = binding-weighted average of the slot states pointing at s.
    h: (K,B,N,d) per-tick slot hidden.  bind: (K,B,nh,N,S) slot->board-position attention.
    Mirrors experiments.interp.plan.lift_to_board (the slot core's analogue of cell-i==square-i)."""
    w = bind.mean(2)                                    # (K,B,N,S) head-avg
    w = w / (w.sum(2, keepdims=True) + 1e-8)            # per-square distribution over slots
    return np.einsum("kbns,kbnd->kbsd", w, h)          # (K,B,S,d)


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    base = planning_eval_envs()["valid_medium"].env
    env_cfg = dataclasses.replace(base, num_envs=n_boards, n_levels_to_load=n_boards,
                                  load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    gamma = cp_cfg.loss.gamma

    envs = env_cfg.make(); obs_np, _ = envs.reset(); obs = np.asarray(obs_np)  # (B,3,H,W)
    B = obs.shape[0]

    # --- slot foundation: per-tick top-slot hidden + binding/routing of the deepest cell ---
    h, bind, route = slot_per_tick(policy, params, jnp.asarray(obs), K)
    # h:(K,B,N,d)  bind:(K,B,nh,N,S)  route:(K,B,nh,N,N)
    N, dslot = h.shape[2], h.shape[3]
    Rt = route.mean(2)                                  # (K,B,N,N) head-avg slot<->slot operator
    tiles = decode_tiles(obs)                           # (B,S) board tile ids
    S = tiles.shape[1]; Hg = Wg = int(round(S ** 0.5))

    # --- decode the LEARNED sigma at the final tick: slot -> board position + confidence ---
    pos, mass = decode_sigma(bind, tick=-1)            # pos:(B,N) in 0..S-1, mass:(B,N)
    boundtile = np.take_along_axis(tiles, pos, axis=1)  # (B,N) tile under each slot's bound square
    # a slot is "valid" if it binds a navigable (non-wall) square -- those are the ones with a value.
    slot_nav = (boundtile != WALL)                     # (B,N) analogue of the attn core's nonwall mask

    print(f"\n===== E1 (slot core): PER-TICK OPERATOR + AMORTIZED VALUE "
          f"(step={step}, boards={B}, slots={N}, board_S={S}, D={D}, trained K={Ktr}, run K={K}, gamma={gamma}) =====")
    f2 = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    f3 = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"

    inj = float(np.mean([len(np.unique(pos[b])) for b in range(B)]) / N)
    print(f"   sigma: slots binding a navigable square = {slot_nav.mean():.2f} | distinct positions / N = {inj:.2f}")

    # ---------- (A) eval / improvement / propagation in BOARD space (lift slots via sigma) ----------
    board = _lift_to_board(h, bind)                     # (K,B,S,d)
    st = analyse_plan(board, tiles, obs)
    print("\n-- (A) decodable value field + plan + change-front (slot hidden lifted to board via sigma) --")
    print(f"   value R^2 (BFS dist-to-target) : {f2(st['val_r2'])}  ({st['val_r2'][0]:.2f} -> {st['val_r2'][-1]:.2f})")
    print(f"   plan acc (greedy dir, ch=.25)  : {f2(st['dir_acc'])}  ({st['dir_acc'][0]:.2f} -> {st['dir_acc'][-1]:.2f})")
    print(f"   change-front dist-from-goal    : {f2(st['chg_dT'])}  ({'OUTWARD (propagation)' if st['chg_dT'][-1] > st['chg_dT'][0] + 0.3 else 'no outward trend'})")

    # ---------- (B) ROUTING operator stationarity vs sharpening (slot<->slot) ----------
    # operator A_t == route_t (slot<->slot, head-avg). Top-1 mass / support / row-cosine over ticks.
    top1, supp, rowcos = [], [], []
    for t in range(K):
        At = Rt[t]                                      # (B,N,N)
        m = slot_nav
        top1.append(float(At.max(-1)[m].mean()))
        supp.append(float((At > 1e-6).sum(-1)[m].mean()))
        if t > 0:
            num = (Rt[t] * Rt[t - 1]).sum(-1)
            den = np.linalg.norm(Rt[t], axis=-1) * np.linalg.norm(Rt[t - 1], axis=-1) + 1e-9
            rowcos.append(float((num / den)[m].mean()))
    print("\n-- (B) routing operator R_t over ticks (slot<->slot, head-avg) --")
    print(f"   top-1 routing mass / slot      : {f3(top1)}  ({'SHARPENS' if top1[-1] > top1[0] + 0.05 else 'stationary'})")
    print(f"   routing support (#slots>1e-6)  : {f2(supp)}")
    print(f"   row-cosine cos(R_t, R_t-1)     : {f3(rowcos)}  ({'stationary (>0.9)' if (rowcos and rowcos[-1] > 0.9) else 'drifting'})")

    # ---------- (C) convergence dynamics of the slot hidden over ticks ----------
    dn = []
    for t in range(1, K):
        d = np.linalg.norm(h[t] - h[t - 1], axis=-1)
        nrm = np.linalg.norm(h[t], axis=-1) + 1e-9
        dn.append(float((d / nrm)[slot_nav].mean()))
    dfin = [float(np.linalg.norm(h[t] - h[K - 1], axis=-1)[slot_nav].mean()) for t in range(K)]
    rhos = [dn[t] / (dn[t - 1] + 1e-9) for t in range(1, len(dn))]
    rho = float(np.median(rhos[max(0, Ktr - 2):])) if len(rhos) > 1 else float("nan")
    print("\n-- (C) convergence of the slot hidden over ticks --")
    print(f"   ||dh_t|| / ||h_t|| per tick    : {f3(dn)}")
    print(f"   ||h_t - h_final||              : {f3(dfin)}")
    print(f"   median contraction rho (t>=Ktr): {rho:.3f}   -> effective horizon ~1/(1-rho) = {1.0/(1-rho+1e-9):.1f} ticks  (gamma={gamma} would need ~{int(np.log(0.01)/np.log(gamma))} from scratch)")

    # ---------- value targets per slot (BFS dist-to-goal of the slot's BOUND board position) ----------
    dT = np.full((B, S), np.nan)
    for b in range(B):
        tgt = np.where(tiles[b] == TARGET)[0]
        dT[b] = bfs_from(tgt, tiles[b], Hg, Wg)
    dT_slot = np.take_along_axis(dT, pos, axis=1)       # (B,N) dist-to-goal at each slot's bound square
    Vstar_slot = gamma ** dT_slot                       # (B,N) value target per slot (nan where wall/unreach)

    # ---------- (D) value WAVEFRONT: per-tick decodability of V* by slot-bound-position dist band ----------
    idx = [(b, n) for b in range(B) for n in range(N) if np.isfinite(dT_slot[b, n]) and dT_slot[b, n] > 0]
    rng = np.random.default_rng(0); rng.shuffle(idx); idx = np.array(idx)
    ntr = int(0.8 * len(idx)); tr, te = idx[:ntr], idx[ntr:]
    y_te = np.array([Vstar_slot[b, n] for b, n in te]); dT_te = np.array([dT_slot[b, n] for b, n in te])
    bands = [(1, 2), (3, 5), (6, 8), (9, 99)]
    band_masks = [(dT_te >= lo) & (dT_te <= hi) for lo, hi in bands]
    wf = np.full((K, len(bands)), np.nan)
    for t in range(K):
        Xtr = np.stack([h[t, b, n] for b, n in tr]); ytr = np.array([Vstar_slot[b, n] for b, n in tr])
        p = _fit(Xtr, ytr)
        Xte = np.stack([h[t, b, n] for b, n in te]); pred = _pred(Xte, p)
        for j, bm in enumerate(band_masks):
            if bm.sum() > 20:
                wf[t, j] = _r2(y_te[bm], pred[bm])
    print("\n-- (D) value WAVEFRONT: R^2(decoded V*) within slot-bound-position dist-to-goal bands, per tick --")
    print(f"   {'tick':>5} " + " ".join(f"d{lo}-{hi:<2}" for lo, hi in bands))
    for t in range(K):
        print(f"   {t:>5} " + " ".join(f"{wf[t,j]:>5.2f}" if np.isfinite(wf[t, j]) else "   . " for j in range(len(bands))))
    onset = []
    for j in range(len(bands)):
        col = wf[:, j]; hit = np.where(np.isfinite(col) & (col > 0.3))[0]
        onset.append(int(hit[0]) if len(hit) else -1)
    print(f"   first tick R^2>0.3 per band     : {onset}   ({'STAGGERED near->far (wavefront)' if (onset[-1] > onset[0] >= 0) else 'simultaneous / no clear wavefront'})")

    # ---------- (E) AMORTIZED test: does the model's OWN ROUTE operator propagate its decoded value? ----------
    # decode value per slot per tick with a single probe (fit at the final tick), then test
    # v_t ~ a*(ROUTE_{t-1} v_{t-1}) + b*v_{t-1} + c.  Operator = route (slot<->slot).
    pf = _fit(np.stack([h[K - 1, b, n] for b, n in tr]), np.array([Vstar_slot[b, n] for b, n in tr]))
    V = np.stack([_pred(h[t].reshape(B * N, dslot), pf).reshape(B, N) for t in range(K)])   # (K,B,N) decoded slot value
    rows = []
    for t in range(1, K - 1):                            # settled-ish ticks
        prop = np.einsum("bnm,bm->bn", Rt[t - 1], V[t - 1])   # (ROUTE_{t-1} v_{t-1})(slot n)
        m = slot_nav
        X = np.stack([prop[m], V[t - 1][m], np.ones(m.sum())], 1)
        ycol = V[t][m]
        coef, *_ = np.linalg.lstsq(X, ycol, rcond=None)
        pred = X @ coef
        rows.append((t, coef[0], coef[1], _r2(ycol, pred)))
    a_mean = float(np.mean([r[1] for r in rows[max(0, Ktr - 3):]])) if rows else float("nan")
    r2_mean = float(np.mean([r[3] for r in rows[max(0, Ktr - 3):]])) if rows else float("nan")
    # ||dv||/||v|| of the DECODED value across ticks -> ~0 means the value field has amortized/converged.
    dv = []
    for t in range(1, K):
        d = np.abs(V[t] - V[t - 1])[slot_nav].mean()
        nrm = np.abs(V[t])[slot_nav].mean() + 1e-9
        dv.append(float(d / nrm))
    print("\n-- (E) AMORTIZED own-operator propagation: v_t ~ a*(R_t-1 v_t-1) + b*v_t-1 + c --")
    print(f"   per-tick (a on R v | b on v | R^2): " + " ".join(f"t{r[0]}:({r[1]:+.2f}|{r[2]:+.2f}|{r[3]:.2f})" for r in rows))
    print(f"   settled mean: coeff on (R v) = {a_mean:+.2f}  (a damped-propagation coeff in (0,1) supports c<-gamma_eff R c + r),  R^2 = {r2_mean:.2f}")
    print(f"   ||dv_t|| / ||v_t|| per tick    : {f3(dv)}  ({'-> 0 (value AMORTIZED across ticks)' if (dv and dv[-1] < 0.1) else 'still moving'})")
    a_first = rows[0][1] if rows else float("nan")
    a_late = a_mean
    print(f"   propagation coeff: tick1 a = {a_first:+.2f} vs settled a = {a_late:+.2f}  "
          f"({'NON-ZERO only early -> amortized (compiled operator)' if (np.isfinite(a_first) and abs(a_first) > 0.1 and abs(a_late) < 0.1) else 'persistent propagation -> inference-time iteration'})")

    # ---------- machine-readable summary line (PLOT_*=) ----------
    print("\nPLOT_E1_SLOT="
          f"step={step};slots={N};Ktr={Ktr};K={K};gamma={gamma};"
          f"val_r2_0={st['val_r2'][0]:.4f};val_r2_K={st['val_r2'][-1]:.4f};"
          f"dir_acc_0={st['dir_acc'][0]:.4f};dir_acc_K={st['dir_acc'][-1]:.4f};"
          f"route_top1_0={top1[0]:.4f};route_top1_K={top1[-1]:.4f};"
          f"route_rowcos_K={(rowcos[-1] if rowcos else float('nan')):.4f};"
          f"rho={rho:.4f};"
          f"wavefront_onset={'|'.join(str(o) for o in onset)};"
          f"amort_a_tick1={a_first:.4f};amort_a_settled={a_late:.4f};amort_r2={r2_mean:.4f};"
          f"dv_K={(dv[-1] if dv else float('nan')):.4f}")

    # ---------- verdict ----------
    sharp = top1[-1] > top1[0] + 0.05
    stationary = (not sharp) and (not rowcos or rowcos[-1] > 0.85)
    propagates = (st['val_r2'][-1] - st['val_r2'][0] > 0.05) or (onset[-1] > onset[0] >= 0) or (st['chg_dT'][-1] > st['chg_dT'][0] + 0.3)
    amortized = (dv and dv[-1] < 0.1) and (np.isfinite(a_late) and abs(a_late) < 0.1)
    print("\n-- VERDICT --")
    print(f"   routing operator  : {'SHARPENS toward argmax -> value-iteration-like' if sharp else 'STATIONARY -> fixed transition (policy-evaluation / SR), NOT value iteration'}")
    print(f"   value computation : {'ITERATIVE propagation along the slot graph (not one-shot)' if propagates else 'flat / amortized substrate (little inference-time iteration)'}")
    print(f"   amortization      : {'AMORTIZED -- ||dv||->0 and the own-operator propagation coeff is non-zero only early (a compiled operator, a few refinement sweeps)' if amortized else 'persistent per-tick propagation (inference-time value iteration)'}")
    print(f"   contraction       : rho={rho:.2f} vs gamma={gamma} -> {'<< gamma: a few refinement sweeps, not a from-scratch solve' if rho < gamma else 'comparable to gamma'}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args()
    main(a.ckpt, a.boards, a.ticks)
