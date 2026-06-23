"""Does the binding live in the slot HIDDEN, even though the attention READ is diffuse? (slot core)

slot_binding_structure showed n50's binding *attention* is diffuse (top-mass ~0.12). But a diffuse read
does not prove no binding -- the tile at a position could still be DECODABLE from the slot that represents
it (the attn-core `binding_balanced` measured exactly this decodability, not the attention map).

Test: at the settled tick, for each board position p find its WINNING slot (the slot that reads p most,
argmax over slots of the binding map), and ask whether THAT slot's hidden linearly decodes p's tile --
per-object balanced accuracy (chance 0.50), the same probe as binding_balanced (so numbers are comparable
to the attn core's 0.74/0.73/0.72/0.78). Contrast with:
  - RANDOM slot's hidden (per position): winning >> random  => real PER-SLOT binding (the winning slot
    specifically carries its position); winning ~= random (both high) => DISTRIBUTED code (every slot
    carries the whole board); all ~0.50 => no binding in the hidden.

  python -m experiments.interp.slot_binding_decode --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick
from experiments.interp.slots import decode_tiles

NAMES = {0: "wall", 1: "floor", 2: "box", 3: "target", 4: "agent"}


def balanced_acc(X, y, c, rng):
    """Per-object balanced-accuracy ridge probe (identical to binding_balanced): balance pos/neg of
    class c, 80/20 split, ridge to +-0.5, threshold at 0, return test accuracy. chance 0.50."""
    z = (y == c).astype(int); pos = np.where(z == 1)[0]; neg = np.where(z == 0)[0]
    m = min(len(pos), len(neg))
    if m < 20: return float("nan"), len(pos)
    p = pos.copy(); n = neg.copy(); rng.shuffle(p); rng.shuffle(n)
    sel = np.concatenate([p[:m], n[:m]]); rng.shuffle(sel)
    ntr = int(0.8 * len(sel)); tr, te = sel[:ntr], sel[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    C = X.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + 10.0 * np.eye(C), Xtr.T @ (z[tr] - 0.5))
    return float(((Xte @ w > 0).astype(int) == z[te]).mean()), len(pos)


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; N = net.recurrent.num_slots
    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)                                                  # (B,S)

    h, bind, _route = slot_per_tick(policy, params, jnp.asarray(obs), K)       # h:(K,B,N,d) bind:(K,B,nh,N,S)
    hid = h[-1]                                                                # (B,N,d) settled
    ba = bind[-1].mean(1)                                                      # (B,N,S) head-avg binding
    rng = np.random.default_rng(0)

    win = ba.argmax(1)                                                         # (B,S) winning slot per position
    rnd = rng.integers(0, N, size=(B, S))                                      # random slot per position
    Xwin = np.take_along_axis(hid, win[:, :, None], 1).reshape(B * S, hid.shape[-1])
    Xrnd = np.take_along_axis(hid, rnd[:, :, None], 1).reshape(B * S, hid.shape[-1])
    y = tiles.reshape(B * S).astype(int)

    print(f"\n===== SLOT BINDING DECODABILITY (slot core, step={step}, boards={B}, slots N={N}, chance 0.50) =====")
    print(f"  per-object BALANCED decode accuracy of tile(p) from the slot hidden, settled tick:")
    print(f"   {'tile':>7}  {'WINNING-slot':>12}  {'random-slot':>11}   (winning>>random => per-slot binding)")
    res = {}
    for c in (0, 1, 2, 3, 4):
        aw, npos = balanced_acc(Xwin, y, c, np.random.default_rng(1))
        ar, _ = balanced_acc(Xrnd, y, c, np.random.default_rng(1))
        res[c] = (aw, ar)
        print(f"   {NAMES[c]:>7}  {aw:>12.3f}  {ar:>11.3f}   (npos={npos})")
    gap = np.nanmean([res[c][0] - res[c][1] for c in res])
    win_mean = np.nanmean([res[c][0] for c in res]); rnd_mean = np.nanmean([res[c][1] for c in res])
    print(f"  mean: winning={win_mean:.3f}  random={rnd_mean:.3f}  gap={gap:+.3f}")
    if win_mean > 0.6 and gap > 0.08:
        verd = "PER-SLOT binding present in the hidden (winning slot carries its position, > random)"
    elif win_mean > 0.6 and gap <= 0.08:
        verd = "DISTRIBUTED code: board decodable from any slot (winning ~= random); no per-slot localization"
    else:
        verd = "weak/no binding decodable from the hidden"
    print(f"  --> {verd}")
    print("PLOT_BINDDECODE=" + repr({NAMES[c]: [round(res[c][0], 3), round(res[c][1], 3)] for c in res}
                                    | {"win_mean": round(win_mean, 3), "rnd_mean": round(rnd_mean, 3)}))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
