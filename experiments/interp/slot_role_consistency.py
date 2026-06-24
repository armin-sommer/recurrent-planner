"""Is the slot binding CONSISTENT across tasks (a fixed slot->role / slot->position map) or RE-INDEXED
per board (content-addressed competition picks a different slot each time)? (slot core)

We have N slots and ~31 navigable squares per board. This asks WHICH squares get a slot and whether a
given slot index means the same thing across boards. For each slot n, across `boards` boards:

  tile-role entropy   : distribution over {wall,floor,box,target,agent} of what slot n's argmax binds.
                        low  => slot n is ROLE-specialized (e.g. it tends to read the agent / a target);
                        high => slot n binds whatever wins the competition that board.
  position spread     : std of slot n's bound (row,col) across boards, / the uniform-over-board std.
                        ~1   => slot n points to a DIFFERENT square each task (board position RE-INDEXED);
                        <<1  => slot n is locked to a fixed board location (position-indexed like the grid).
  role-slot stability : for the AGENT and the TARGET square, the slot index that reads it most each board;
                        entropy over boards + top-1 frequency. low entropy / high top-1 => a DEDICATED
                        agent/target slot (consistent index); ~ln(N) => a different slot binds it each board.

Disambiguation: a slot that always reads the agent will have LOW tile-role entropy but HIGH position
spread (the agent sits at a different square each board) -- that is role-consistent + position-re-indexed.
A slot with HIGH tile-role entropy AND high position spread is fully content-re-indexed.

  python -m experiments.interp.slot_role_consistency --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import WALL, FLOOR, BOX, TARGET, AGENT

TILE = {WALL: "wall", FLOOR: "floor", BOX: "box", TARGET: "target", AGENT: "agent"}


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; N = net.recurrent.num_slots
    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)                                                  # (B,S)

    _h, bind, _route = slot_per_tick(policy, params, jnp.asarray(obs), K)
    pos, mass = decode_sigma(bind, tick=-1)                                    # pos:(B,N) slot->square; mass:(B,N)
    ba = bind[-1].mean(1)                                                      # (B,N,S) head-avg read
    bt = np.take_along_axis(tiles, pos, axis=1)                                # (B,N) tile each slot binds

    rr, cc = np.divmod(np.arange(S), W); rr = rr.astype(float); cc = cc.astype(float)
    uni_std = np.sqrt(((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2).mean())

    # board tile composition baseline (what a random binder would draw)
    base = np.array([ (tiles == k).mean() for k in (WALL, FLOOR, BOX, TARGET, AGENT) ])
    base = base / base.sum()

    # --- per-slot tile-role entropy + position spread ---
    tile_ents, pos_spreads, p_agent, p_target = [], [], [], []
    for n in range(N):
        cnt = np.array([ (bt[:, n] == k).sum() for k in (WALL, FLOOR, BOX, TARGET, AGENT) ], float)
        p = cnt / cnt.sum()
        tile_ents.append(ent(p))
        p_agent.append(p[4]); p_target.append(p[3])
        r = rr[pos[:, n]]; c = cc[pos[:, n]]
        pos_spreads.append(np.sqrt(r.var() + c.var()) / uni_std)
    tile_ents = np.array(tile_ents); pos_spreads = np.array(pos_spreads)
    p_agent = np.array(p_agent); p_target = np.array(p_target)

    # --- role-slot index stability: which slot READS the agent/target square most, each board ---
    def role_slot_stats(tid):
        idxs = []
        for b in range(B):
            sq = np.where(tiles[b] == tid)[0]
            if not len(sq):
                continue
            idxs.append(int(ba[b, :, sq[0]].argmax()))                         # slot that reads that square most
        idxs = np.array(idxs)
        if not len(idxs):
            return float("nan"), float("nan"), 0
        c = np.bincount(idxs, minlength=N).astype(float); pr = c / c.sum()
        return ent(pr), float(c.max() / c.sum()), len(idxs)                    # entropy, top-1 freq, n boards

    ag_ent, ag_top1, ag_nb = role_slot_stats(AGENT)
    tg_ent, tg_top1, tg_nb = role_slot_stats(TARGET)
    lnN = np.log(N)
    base_ent = ent(base)

    m = lambda x: float(np.nanmean(x))
    print(f"\n===== SLOT ROLE / POSITION CONSISTENCY ACROSS TASKS (slot core, step={step}, boards={B}, N={N}) =====")
    print(f"  -- per-slot TILE-ROLE entropy (over {B} boards; 0 => slot always binds one tile type; ln5={np.log(5):.2f}=uniform) --")
    print(f"     mean per-slot tile entropy : {m(tile_ents):.3f}   (board-composition baseline entropy {base_ent:.3f})")
    print(f"     slots with P(bind AGENT)  >0.5 / >0.3 / >0.1 : {(p_agent>.5).sum()} / {(p_agent>.3).sum()} / {(p_agent>.1).sum()}   (max P={p_agent.max():.2f})")
    print(f"     slots with P(bind TARGET) >0.5 / >0.3 / >0.1 : {(p_target>.5).sum()} / {(p_target>.3).sum()} / {(p_target>.1).sum()}   (max P={p_target.max():.2f})")
    print(f"  -- per-slot POSITION spread across boards (/uniform; ~1 => points to a different square each task) --")
    print(f"     mean per-slot position spread / uniform : {m(pos_spreads):.3f}   (min over slots {pos_spreads.min():.3f}; uniform_std={uni_std:.2f})")
    print(f"     slots position-locked (spread<0.3)      : {(pos_spreads<0.3).sum()} / {N}")
    print(f"  -- ROLE-SLOT INDEX stability (which slot reads the role-square most, each board) --")
    print(f"     AGENT  slot-index entropy : {ag_ent:.2f} / ln(N)={lnN:.2f}    top-1 index used on {ag_top1*100:.0f}% of boards   (n={ag_nb})")
    print(f"     TARGET slot-index entropy : {tg_ent:.2f} / ln(N)={lnN:.2f}    top-1 index used on {tg_top1*100:.0f}% of boards   (n={tg_nb})")
    verdict = ("RE-INDEXED per task: high position spread + high role-slot entropy => content-addressed, "
               "no fixed slot->square map" if m(pos_spreads) > 0.7 and ag_ent > 0.6 * lnN else
               "some FIXED structure -- inspect above")
    print(f"  --> {verdict}")
    print("PLOT_CONSIST=" + repr(dict(tile_ent=round(m(tile_ents), 3), base_ent=round(base_ent, 3),
          pos_spread=round(m(pos_spreads), 3), pos_locked=int((pos_spreads < 0.3).sum()),
          agent_max_p=round(float(p_agent.max()), 3), target_max_p=round(float(p_target.max()), 3),
          agent_idx_ent=round(ag_ent, 3), agent_top1=round(ag_top1, 3),
          target_idx_ent=round(tg_ent, 3), target_top1=round(tg_top1, 3), lnN=round(float(lnN), 3))))
    print("=" * 96 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
