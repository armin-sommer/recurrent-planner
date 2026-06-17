"""E7 -- how does the POLICY-EVALUATION PROPAGATION itself change under interventions? (not the readout)

Policy evaluation is V=(I-gamma A)^{-1} r: value spreads along the transition operator A from the reward
source r. So "how it changes" has two measurable, operator-level signatures (read from the attention A,
NOT the value head):
  (A) REWARD RE-ANCHORS: in E4 each cell puts ~0.07 of its attention mass DIRECTLY on the goal cell (a
      source anchor). Move the goal -> does that anchor RELOCATE to the new goal cell (and leave the old)?
  (B) OPERATOR RE-ROUTES: add an obstacle / move a box -> does A stop routing value INTO the now-blocked
      cell (mass onto it drops), and does the vacated cell re-open (mass onto the old box cell rises)?

We compute the settled top-cell head-averaged attention A on the original and intervened boards and read
these masses off directly. This is the propagation mechanism re-forming, independent of the readout.

  python -m experiments.interp.e7 --ckpt <cp_dir> --boards 160
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_RGB = np.array([254, 126, 125], np.uint8)
WALL_RGB = np.array([0, 0, 0], np.uint8)
BOX_RGB = np.array([142, 121, 56], np.uint8)


def geodesic_mid(agent, dT, H, W):
    s = agent; path = [s]
    while np.isfinite(dT[s]) and dT[s] > 0:
        r, c = divmod(s, W); best, bd = None, dT[s]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                t = nr * W + nc
                if np.isfinite(dT[t]) and dT[t] < bd:
                    bd = dT[t]; best = t
        if best is None:
            break
        s = best; path.append(s)
    return path[len(path) // 2] if len(path) >= 3 else None


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); rc = np.arange(S); RR, CC = rc // W, rc % W

    def far_floor(b, frm, banned):
        fl = np.array([s for s in np.where(tiles[b] == FLOOR)[0] if s not in banned])
        return int(fl[np.argmax(np.hypot(RR[fl] - RR[frm], CC[fl] - CC[frm]))]) if len(fl) else -1

    obs_g = obs0.copy(); obs_o = obs0.copy(); obs_b = obs0.copy()
    G0 = np.full(B, -1); G1 = np.full(B, -1); A0 = np.full(B, -1); OB = np.full(B, -1); BX0 = np.full(B, -1); BX1 = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]; fl0 = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl0)):
            continue
        a = int(ag[0]); A0[b] = a; G0[b] = int(tg[0]); tgts = set(map(int, tg))
        fr, fc = divmod(int(fl0[0]), W); floor_rgb = obs0[b, :, fr, fc]; dT = bfs_from([G0[b]], tiles[b], H, W)
        # goal -> far side
        g1 = far_floor(b, G0[b], {a})
        if g1 >= 0:
            G1[b] = g1; obs_g[b, :, RR[G0[b]], CC[G0[b]]] = floor_rgb; obs_g[b, :, RR[g1], CC[g1]] = TARGET_RGB
        # obstacle on agent->goal geodesic
        if np.isfinite(dT[a]):
            mid = geodesic_mid(a, dT, H, W)
            if mid is not None and tiles[b, mid] == FLOOR:
                OB[b] = mid; obs_o[b, :, RR[mid], CC[mid]] = WALL_RGB
        # box -> adjacent floor
        boxes = [s for s in np.where(tiles[b] == BOX)[0] if np.array_equal(obs0[b, :, RR[s], CC[s]], BOX_RGB)]
        for s in boxes:
            br, bc = divmod(int(s), W); moved = False
            for dr, dc in DIRS:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < H and 0 <= nc < W and tiles[b, nr * W + nc] == FLOOR:
                    obs_b[b, :, br, bc] = floor_rgb; obs_b[b, :, nr, nc] = BOX_RGB; BX0[b] = int(s); BX1[b] = nr * W + nc; moved = True; break
            if moved:
                break
    tg_g = decode_tiles(obs_g); tg_o = decode_tiles(obs_o); tg_b = decode_tiles(obs_b)
    okg = np.array([G1[b] >= 0 and tg_g[b, G1[b]] == TARGET and tg_g[b, G0[b]] == FLOOR for b in range(B)])
    oko = np.array([OB[b] >= 0 and tg_o[b, OB[b]] == WALL for b in range(B)])
    okb = np.array([BX1[b] >= 0 and tg_b[b, BX1[b]] == BOX and tg_b[b, BX0[b]] == FLOOR for b in range(B)])

    def A_settled(o):
        emb = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(o))))
        return np.asarray(recompute_d3(cps, emb, K)[1][-1, D - 1].mean(1))             # (B,S,S) head-avg top-cell, last tick
    A0a = A_settled(obs0); Aga = A_settled(obs_g); Aoa = A_settled(obs_o); Aba = A_settled(obs_b)

    def mass_on(A, b, cell):                                                           # mean mass non-wall queries put on `cell`
        q = tiles[b] != WALL
        return float(A[b][q, cell].mean())

    # (A) reward re-anchoring
    m_oldgoal_o, m_newgoal_o, m_oldgoal_a, m_newgoal_a = [], [], [], []
    for b in range(B):
        if not okg[b]:
            continue
        m_oldgoal_o.append(mass_on(A0a, b, G0[b]))     # mass on the goal cell when it IS the goal (orig)
        m_newgoal_o.append(mass_on(A0a, b, G1[b]))     # mass on the future-goal cell when it's still floor (orig)
        m_oldgoal_a.append(mass_on(Aga, b, G0[b]))     # mass on old-goal cell after it became floor
        m_newgoal_a.append(mass_on(Aga, b, G1[b]))     # mass on new-goal cell after the move
    # (B) operator re-routing
    m_obsX_o, m_obsX_a = [], []
    for b in range(B):
        if not oko[b]:
            continue
        m_obsX_o.append(mass_on(A0a, b, OB[b]))        # mass onto the cell when it's floor
        m_obsX_a.append(mass_on(Aoa, b, OB[b]))        # mass onto it after it became a wall
    m_nb_o, m_nb_a, m_ob_o, m_ob_a = [], [], [], []
    for b in range(B):
        if not okb[b]:
            continue
        m_nb_o.append(mass_on(A0a, b, BX1[b]))         # new box cell: floor (orig) -> box (after): mass should DROP
        m_nb_a.append(mass_on(Aba, b, BX1[b]))
        m_ob_o.append(mass_on(A0a, b, BX0[b]))         # old box cell: box (orig) -> floor (after): mass should RISE
        m_ob_a.append(mass_on(Aba, b, BX0[b]))

    f = lambda x: float(np.mean(x)) if len(x) else float("nan")
    print(f"\n===== E7: HOW THE POLICY-EVALUATION PROPAGATION RE-FORMS (step={step}, boards={B}) =====")
    print(f"  (A) REWARD RE-ANCHORING (mass each cell puts directly ON the goal cell), n={len(m_oldgoal_o)} --")
    print(f"      old-goal cell:  is-goal {f(m_oldgoal_o):.3f}  -> became-floor {f(m_oldgoal_a):.3f}   (anchor leaves)")
    print(f"      new-goal cell:  was-floor {f(m_newgoal_o):.3f} -> became-goal {f(m_newgoal_a):.3f}   (anchor arrives)")
    reloc = f(m_newgoal_a) > 2 * f(m_newgoal_o) and f(m_oldgoal_a) < 0.6 * f(m_oldgoal_o)
    print(f"      => goal anchor {'RELOCATES to the new goal (propagation re-targets the reward)' if reloc else 'does NOT clearly relocate'}")
    print(f"  (B) OPERATOR RE-ROUTING (mass landing on the blocked cell) --")
    print(f"      obstacle cell:  floor {f(m_obsX_o):.3f} -> wall {f(m_obsX_a):.3f}   (n={len(m_obsX_o)}; value stops routing in)")
    print(f"      box new cell:   floor {f(m_nb_o):.3f} -> box {f(m_nb_a):.3f}   (n={len(m_nb_o)}; blocks)")
    print(f"      box old cell:   box {f(m_ob_o):.3f} -> floor {f(m_ob_a):.3f}   (re-opens)")
    rr = f(m_obsX_a) < 0.7 * f(m_obsX_o) and f(m_nb_a) < 0.9 * f(m_nb_o)
    print(f"      => operator {'RE-ROUTES around new blocks (respects the new transition graph)' if rr else 'does NOT clearly re-route'}")
    print("PLOT_E7=" + repr(dict(oldgoal=[round(f(m_oldgoal_o), 4), round(f(m_oldgoal_a), 4)],
                                  newgoal=[round(f(m_newgoal_o), 4), round(f(m_newgoal_a), 4)],
                                  obsX=[round(f(m_obsX_o), 4), round(f(m_obsX_a), 4)],
                                  boxnew=[round(f(m_nb_o), 4), round(f(m_nb_a), 4)],
                                  boxold=[round(f(m_ob_o), 4), round(f(m_ob_a), 4)])))
    print("=" * 88 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=160)
    a = ap.parse_args(); main(a.ckpt, a.boards)
