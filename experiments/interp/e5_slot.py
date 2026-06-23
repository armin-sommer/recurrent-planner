"""E5 (SLOT CORE) -- does effective REACH grow with ticks? (iterative deepening / do pulled-from slots
get updated?)

This is the slot-core port of experiments/interp/e5.py. The original probe was written for the spatial
ATTENTION core, where latent cell i IS board square i, so "the agent's cell" was just h[:, b, agent_sq].
The SLOT core (cleanba.slot_lstm) has N FREE slots that are NOT tied to board position: the binding
sigma (slot <-> board square) is LEARNED. So we go through sigma: the "agent cell" becomes the AGENT
SLOT -- the slot whose bound board position == the agent square (decoded from the slot-attention map on
the unperturbed board). Graph distance is still BFS board distance, but between the *bound positions* of
slots, not between slot indices.

The question is unchanged. E4-style results show one tick pulls value ~few hops; here we ask whether with
MORE ticks a slot integrates info from FURTHER horizons -- which requires that the slots it routes from are
THEMSELVES updated (carry forward their own propagated value). Test causally: perturb one floor cell at
graph-distance d from the agent square (flip->wall in the obs) and measure the agent-SLOT latent shift
||dh(agent_slot)|| at EACH tick. A single ~few-hop routing kernel cannot reach d=8 in one tick; so if far
perturbations influence the agent slot and that influence ARRIVES LATER for larger d, intermediate slots
must relay it across ticks => compounding => answer is YES.

  arrival-tick rises with d  (horizon grows ~k hops/tick)   -> REACH COMPOUNDS: routed-from slots get updated
  far d (>3) never influences / arrives at tick 1 flat       -> fixed single-tick blur, no deepening

  python -m experiments.interp.e5_slot --ckpt <cp_dir> --boards 256 --dmax 9 --ticks 12

Notes vs the attention port:
  * recompute_d3 -> slot_per_tick (experiments.interp.slot_interp): per-tick top-slot hidden + bind/route.
  * cell-by-position indexing -> sigma mapping: decode_sigma(bind) gives slot -> bound board position; the
    agent slot is the slot whose bound position is the agent square (on the UNPERTURBED board). We track
    that SAME slot index through both h0 and hp (||dh|| of a fixed agent slot), exactly as the attn port
    tracked the fixed agent cell.
  * --ticks sets K (test-time thinking depth); the slot model trained at K=3 (extended thinking >3).
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, AGENT


def agent_slot_for_board(pos, mass, ag_sq, dist_from_agent, W):
    """Pick the slot that represents the agent square for board b.

    pos:  (N,) slot -> bound board-position index (0..S-1)   (from sigma on the unperturbed board)
    mass: (N,) slot binding confidence
    ag_sq: agent board-square index. dist_from_agent: (S,) BFS dist from the agent square.

    Preference: among slots whose bound position == agent square, the one with the highest binding
    confidence (the agent slot). If NO slot binds the agent square exactly, fall back to the slot whose
    bound position is graph-closest to the agent (smallest BFS dist), breaking ties by confidence -- the
    sigma analogue of "the agent's cell" when binding is imperfect.
    """
    exact = np.where(pos == ag_sq)[0]
    if len(exact):
        return int(exact[np.argmax(mass[exact])])
    # fallback: closest bound position to the agent on the navigation graph
    d = dist_from_agent[pos]                                   # (N,) BFS dist of each slot's bound pos
    d = np.where(np.isfinite(d), d, np.inf)
    best = np.flatnonzero(d == d.min())
    return int(best[np.argmax(mass[best])])


def main(cp_dir, n_boards, dmax, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params
    net = cp_cfg.net
    if K is None:
        K = net.repeats_per_step
    N_slots = net.recurrent.num_slots
    D = net.n_recurrent

    envs = env_cfg.make(); obs0 = np.asarray(envs.reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)

    # --- pick a perturbation per board: flip a floor cell at graph-dist td from the agent -> wall ---
    obs_p = obs0.copy()
    ag = np.full(B, -1); d_used = np.full(B, -1)
    dist_all = np.full((B, S), np.nan)
    for b in range(B):
        a = np.where(tiles[b] == AGENT)[0]
        if not len(a):
            continue
        a = int(a[0]); ag[b] = a
        dist = bfs_from([a], tiles[b], H, W); dist_all[b] = dist
        floor = np.where(tiles[b] == FLOOR)[0]; floor = floor[np.isfinite(dist[floor])]
        if not len(floor):
            continue
        td = (b % dmax) + 1
        c = int(floor[np.argmin(np.abs(dist[floor] - td))])
        d_used[b] = int(dist[c]); r, cc = divmod(c, W); obs_p[b, :, r, cc] = 0       # flip floor -> wall

    valid = (ag >= 0) & (d_used >= 1)

    # --- per-tick top-slot hidden: unperturbed (h0) and perturbed (hp) ---
    h0, bind0, _ = slot_per_tick(policy, params, jnp.asarray(obs0), K)               # (K,B,N,d)
    hp, _, _ = slot_per_tick(policy, params, jnp.asarray(obs_p), K)
    h0 = np.asarray(h0); hp = np.asarray(hp)

    # --- sigma on the UNPERTURBED board: slot -> bound board position; identify the agent slot per board ---
    pos, mass = decode_sigma(bind0, tick=-1)                                         # (B,N), (B,N)
    ag_slot = np.full(B, -1)
    sigma_hit = 0                                                                    # how often a slot binds the agent square exactly
    for b in range(B):
        if not valid[b]:
            continue
        if np.any(pos[b] == ag[b]):
            sigma_hit += 1
        ag_slot[b] = agent_slot_for_board(pos[b], mass[b], ag[b], dist_all[b], W)

    # --- influence per tick = ||dh(agent_slot)|| (same fixed slot through h0 and hp) ---
    infl = np.zeros((B, K))
    for b in range(B):
        if valid[b] and ag_slot[b] >= 0:
            infl[b] = np.linalg.norm(hp[:, b, ag_slot[b]] - h0[:, b, ag_slot[b]], axis=-1)   # (K,)

    f = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    nb = int(valid.sum())
    sigma_frac = (sigma_hit / nb) if nb else 0.0
    print(f"\n===== E5: REACH vs TICKS (slot core)  "
          f"(step={step}, boards={nb}, K={K}, D={D}, slots={N_slots}) =====")
    print(f"  perturb a floor cell at graph-dist d from agent SQUARE (flip->wall); agent-SLOT ||dh|| per tick.")
    print(f"  agent slot = slot whose bound board-position == agent square (sigma, unperturbed board).")
    print(f"  sigma binds the agent square exactly on {sigma_frac*100:.0f}% of boards "
          f"(else nearest-bound-slot fallback).")
    print(f"  {'d':>3} {'n':>4} {'arr':>4} {'final':>7}   normalized onset (||dh||_t / ||dh||_final)")
    arr_d, arr_t, mat = [], [], {}
    for d in range(1, dmax + 1):
        m = valid & (d_used == d)
        if m.sum() < 3:
            continue
        a = infl[m].mean(0)                                                          # (K,) abs influence
        mat[d] = a
        rel = a / (a[-1] + 1e-9)
        arrival = next((t + 1 for t in range(K) if rel[t] >= 0.5), K)
        arr_d.append(d); arr_t.append(arrival)
        print(f"  {d:>3} {int(m.sum()):>4} {arrival:>4} {a[-1]:7.3f}   {f(rel)}")

    # horizon(tick): largest d whose signal has arrived (>=50% of its own final) by that tick
    print(f"\n  horizon(tick) = max d with signal >=50% arrived:")
    horizon = []
    for t in range(K):
        hd = 0
        for d in mat:
            if mat[d][t] >= 0.5 * (mat[d][-1] + 1e-9):
                hd = max(hd, d)
        horizon.append(hd)
    print(f"     tick:    " + " ".join(f"{t+1:>3}" for t in range(K)))
    print(f"     horizon: " + " ".join(f"{horizon[t]:>3}" for t in range(K)))

    # absolute influence matrix (rows=d, cols=tick) -- does far-d influence appear only late?
    print(f"\n  absolute ||dh(agent_slot)|| by (d x tick):")
    print(f"     {'d/t':>4} " + " ".join(f"{t+1:>5}" for t in range(K)))
    for d in sorted(mat):
        print(f"     {d:>4} " + " ".join(f"{mat[d][t]:5.2f}" for t in range(K)))

    if len(arr_d) >= 3:
        far = [(d, t) for d, t in zip(arr_d, arr_t) if d >= 2]
        dd = np.array([d for d, _ in far]); tt = np.array([t for _, t in far])
        slope = float(np.polyfit(dd, tt, 1)[0])
        hops_per_tick = (1.0 / slope) if slope > 1e-6 else float("inf")
        print(f"\n  arrival-tick vs d slope = {slope:+.2f}  (~{hops_per_tick:.1f} hops/tick); "
              f"horizon {horizon[0]}->{horizon[-1]} over ticks")
        verdict = ("REACH COMPOUNDS over ticks -> routed-from slots are updated (iterative deepening / multi-step PE)"
                   if (slope > 0.15 or horizon[-1] > horizon[0] + 1) else
                   "reach ~flat -> single wide kernel, little compounding")
        print(f"  --> {verdict}")
    # machine-readable
    print("PLOT_ARRIVAL=" + repr({int(d): int(t) for d, t in zip(arr_d, arr_t)}))
    print("PLOT_HORIZON=" + repr([int(x) for x in horizon]))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--dmax", type=int, default=9)
    ap.add_argument("--ticks", type=int, default=None, help="K thinking ticks (default = model's repeats_per_step)")
    a = ap.parse_args(); main(a.ckpt, a.boards, a.dmax, a.ticks)
