"""Geometry of the slot binding: is the diffuse read a LOCAL spatial blob or a GLOBAL smear, and is the
navigable-cell -> slot assignment injective? (slot core)

We know: read is diffuse (entropy ~3.8), but the winning slot's HIDDEN decodes its top position (~0.71).
This pins HOW positions map to slots:
  - read spatial spread   : per-slot read-weighted std of board (row,col), / the uniform-over-board std.
                            << 1 => each slot reads a LOCAL BLOB (soft receptive field); ~1 => global smear.
  - radial read profile   : read mass vs board-distance from the slot's argmax position (decay => local).
  - navigable mass         : total read mass a slot puts on non-wall cells (vs the ~0.31 navigable fraction).
  - pos->slot injectivity  : among NAVIGABLE positions, fraction whose winning slot is distinct.
  - competition peakedness : for each position, entropy over slots of who-reads-it (peaked => clean assignment).

  python -m experiments.interp.slot_binding_geometry --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import WALL


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; N = net.recurrent.num_slots
    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)
    _h, bind, _r = slot_per_tick(policy, params, jnp.asarray(obs), K)
    w = bind[-1].mean(1)                                                       # (B,N,S) settled read, head-avg

    rr, cc = np.divmod(np.arange(S), W); rr = rr.astype(float); cc = cc.astype(float)
    # uniform-over-board spatial std (denominator)
    uni_std = np.sqrt(((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2).mean())

    spreads, navmass, comp_ent, radial = [], [], [], []
    inj = []
    radbins = np.zeros(6); radcnt = np.zeros(6)
    for b in range(B):
        wb = w[b]                                                              # (N,S)
        cr = wb @ rr; ccol = wb @ cc                                           # (N,) read centroid
        var = wb @ (rr ** 2 + cc ** 2) - (cr ** 2 + ccol ** 2)
        spreads.append(np.sqrt(np.clip(var, 0, None)).mean() / uni_std)
        nav = (tiles[b] != WALL).astype(float)
        navmass.append((wb @ nav).mean())                                      # read mass on navigable per slot
        # radial profile around each slot's argmax
        am = wb.argmax(1)
        d = np.sqrt((rr[None, :] - rr[am][:, None]) ** 2 + (cc[None, :] - cc[am][:, None]) ** 2)  # (N,S)
        db = np.clip(d.astype(int), 0, 5)
        for k in range(6):
            mask = (db == k); radbins[k] += (wb * mask).sum(); radcnt[k] += mask.sum()
        # competition peakedness: per position, distribution over slots of who reads it (normalize over slots)
        comp = wb / (wb.sum(0, keepdims=True) + 1e-9)                          # (N,S) col-normalized
        pe = -(comp * np.log(np.clip(comp, 1e-9, 1))).sum(0)                   # (S,) entropy over slots
        comp_ent.append(pe.mean())
        # cell->slot coverage over navigable positions: distinct winning slots / #navigable. NOTE this is the
        # TRANSPOSED map (which slot owns each cell), not slot->cell sigma-injectivity; and it rises with N even
        # under random binding (argmax over a bigger slot pool => more distinct winners), so compare to a null.
        navpos = np.where(tiles[b] != WALL)[0]
        if len(navpos):
            win = wb[:, navpos].argmax(0)                                      # winning slot per navigable pos
            inj.append(len(np.unique(win)) / len(navpos))

    m = lambda x: float(np.nanmean(x))
    radial_profile = radbins / np.maximum(radcnt, 1)
    print(f"\n===== SLOT BINDING GEOMETRY (slot core, step={step}, boards={B}, N={N}, S={S}) =====")
    print(f"  read spatial spread / uniform : {m(spreads):.3f}   (<<1 => LOCAL blob; ~1 => GLOBAL smear; uniform_std={uni_std:.2f})")
    print(f"  radial read profile (mean read mass per cell at board-dist 0..5 from argmax):")
    print(f"     dist:  " + " ".join(f"{k:>6}" for k in range(6)))
    print(f"     mass:  " + " ".join(f"{radial_profile[k]:.4f}" for k in range(6)) + "   (decay => spatially local)")
    print(f"  read mass on navigable cells  : {m(navmass):.3f}   (vs navigable fraction ~{(tiles!=WALL).mean():.2f}; > => avoids walls)")
    print(f"  pos->slot injectivity (navig.) : {m(inj):.3f}   (1 => each navigable cell a distinct winning slot)")
    print(f"  competition entropy over slots : {m(comp_ent):.2f} nats (ln(N)={np.log(N):.2f}=uniform; low => one slot wins each cell)")
    print("PLOT_GEOM=" + repr(dict(spread=round(m(spreads), 3), navmass=round(m(navmass), 3),
          inj=round(m(inj), 3), comp_ent=round(m(comp_ent), 2),
          radial=[round(float(x), 4) for x in radial_profile])))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
