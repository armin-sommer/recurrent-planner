"""value_ticks -- DOES THE VALUE CHANGE OVER THINKING TICKS, really?

E1's value R^2 refits a probe each tick, so flat R^2 can't distinguish a FIXED field from a MOVING one.
E13's "state-value per tick" is the model's own critic scalar, but it's a global pooled readout of a
still-settling hidden (and OOD past the trained depth). Here we test it cleanly two ways:

  (A) model's own scalar value V_t = critic(top_h_t), with the spread across boards as the yardstick:
      - |dV| per tick vs std(V) across boards (is the tick-to-tick change big or small?)
      - corr(V_t, V_settled) across boards (=1 => the value RANKING of boards is fixed by tick t)
  (B) the per-cell value FIELD with a FIXED probe (fit ONCE at the trained depth, applied to all ticks):
      - ||field_t - field_settled|| / ||field_settled||  (=0 => field unchanged)
      - corr(field_t, field_settled) per board
      - for contrast, the RAW hidden change ||h_t - h_settled|| / ||h_settled||

If the field is set early (rel-change small, corr->1 by tick ~2) while the raw hidden keeps moving, the
value is AMORTIZED, not iterated -- i.e. the loop is not doing sustained value propagation/DP on V.

  python -m experiments.interp.value_ticks --ckpt <cp_dir> --boards 256 --ticks 8
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, TARGET


def ridge(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return mu, sd, w


def applyp(X, p):
    mu, sd, w = p
    return ((X - mu) / sd) @ w


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    hsf = getattr(net, "head_scale", 1.0); NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wv = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bv = np.asarray(params["params"]["critic_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)
    emb = np.asarray(get_embed(policy, params, jnp.asarray(obs))); C = emb.shape[-1]; embr = emb.reshape(B, S, C)
    th = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])             # (K,B,S,C)
    ref = Ktr - 1                                                          # reference tick = trained depth (index)

    # (A) model's own scalar value per tick
    def Vscalar(tht):
        mlp = np.maximum((tht + embr).reshape(B, S * C) @ Wd + bd, 0.0)
        return (mlp @ Wv + bv).reshape(B) * hsf
    V = np.stack([Vscalar(th[t]) for t in range(K)])                      # (K,B)
    stdV = float(V[ref].std())
    dV = np.array([np.nan if t == 0 else np.abs(V[t] - V[t - 1]).mean() for t in range(K)])
    relV = np.array([np.abs(V[t] - V[ref]).mean() for t in range(K)]) / (stdV + 1e-9)
    corrV = np.array([np.corrcoef(V[t], V[ref])[0, 1] for t in range(K)])

    # (B) fixed-probe value FIELD: target = -BFS dist to target (per reachable cell)
    y = np.full((B, S), np.nan)
    for b in range(B):
        tg = np.where(tiles[b] == TARGET)[0]
        if len(tg):
            dT = bfs_from([int(tg[0])], tiles[b], H, W); m = np.isfinite(dT); y[b, m] = -dT[m]
    mask = np.isfinite(y)
    rng = np.random.RandomState(0); idx = rng.permutation(B); tr = idx[:B // 2]; te = idx[B // 2:]
    def samples(t, boards):
        X = [th[t, b, mask[b]] for b in boards]; Y = [y[b, mask[b]] for b in boards]
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = samples(ref, tr)
    P = ridge(Xtr, Ytr)                                                   # FIXED probe, fit at trained depth

    cosb = np.zeros((K, len(te))); relb = np.zeros((K, len(te))); rawb = np.zeros((K, len(te)))
    for j, b in enumerate(te):
        m = mask[b]; fref = applyp(th[ref, b, m], P); nref = np.linalg.norm(fref) + 1e-9
        href = th[ref, b, m]; hn = np.linalg.norm(href) + 1e-9
        for t in range(K):
            ft = applyp(th[t, b, m], P)
            cosb[t, j] = (fref @ ft) / (nref * (np.linalg.norm(ft) + 1e-9))
            relb[t, j] = np.linalg.norm(ft - fref) / nref
            rawb[t, j] = np.linalg.norm(th[t, b, m] - href) / hn
    cos = cosb.mean(1); rel = relb.mean(1); raw = rawb.mean(1)
    r2 = np.zeros(K)
    for t in range(K):
        Xt, Yt = samples(t, te); pred = applyp(Xt, P)
        r2[t] = 1 - np.sum((Yt - pred) ** 2) / np.sum((Yt - Yt.mean()) ** 2)

    f = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== VALUE OVER TICKS (step={step}, B={B}, K={K}, K_train={Ktr}, ref tick={ref+1}) =====")
    print("-- (A) model's own scalar value V_t = critic(top_h_t) --")
    print(f"   mean V per tick          : {f(V.mean(1))}")
    print(f"   |dV| tick-to-tick        : {f(dV)}   (yardstick: std(V across boards) = {stdV:.3f})")
    print(f"   |V_t - V_settled|/std(V) : {f(relV)}   (0 => same as trained-depth value)")
    print(f"   corr(V_t, V_settled)     : {f(corrV)}   (1 => board value-ranking fixed by tick t)")
    print("-- (B) per-cell value FIELD, FIXED probe fit at trained depth --")
    print(f"   ||field_t-field_ref||/||ref|| : {f(rel)}   (0 => field unchanged over ticks)")
    print(f"   corr(field_t, field_ref)/board: {f(cos)}")
    print(f"   fixed-probe field R^2         : {f(r2)}")
    print(f"   (contrast) raw ||h_t-h_ref||/||ref||: {f(raw)}   (how much the hidden ITSELF moves)")
    settle = "by tick %d" % (int(np.argmax(relV < 0.15) + 1) if (relV < 0.15).any() else K)
    print(f"-- read: scalar value within trained depth settles {settle}; "
          f"field rel-change at tick1={rel[0]:.2f}, tick2={rel[1]:.2f}; raw hidden moves {raw[0]:.2f}->.. --")
    print("PLOT_VT=" + repr(dict(Vmean=[round(float(x),3) for x in V.mean(1)], dV=[None if t==0 else round(float(dV[t]),3) for t in range(K)],
                                 stdV=round(stdV,3), relV=[round(float(x),3) for x in relV], corrV=[round(float(x),3) for x in corrV],
                                 field_rel=[round(float(x),3) for x in rel], field_cos=[round(float(x),3) for x in cos],
                                 field_r2=[round(float(x),3) for x in r2], raw=[round(float(x),3) for x in raw], Ktr=int(Ktr))))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256); ap.add_argument("--ticks", type=int, default=8)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
