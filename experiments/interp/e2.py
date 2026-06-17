"""E2 -- REWARD-RELABEL INVARIANCE: successor representation vs policy evaluation. (D=3 entmax core.)

E1 + source showed the per-tick op is a fixed, stationary, convex-average (no-max) propagation operator
-> the family is the resolvent (I - gamma_eff A)^{-1} r_eff. Two members remain:
  * POLICY EVALUATION: reward/goal is BAKED INTO the iterate, c* = (I-gamma P^pi)^{-1} r. Moving the goal
                       re-forms the whole propagated value field -> h2 changes far from the goal too.
  * SUCCESSOR REPRESENTATION: the loop builds reward-AGNOSTIC occupancy features M=(I-gamma P^pi)^{-1} in
                       h2; reward enters only at the critic readout V = M r. Moving the goal leaves the far
                       /agent latent ~unchanged, but the readout value V still shifts.

Test: relocate the TARGET to a far floor cell (changes the REWARD location, keeps transitions P fixed).
Reference: flip that same far cell to a WALL (changes P, keeps the reward) -- the through-walls
perturbation, to calibrate how much a far LOCAL change propagates at all. Measure the recurrent latent
shift ||dh2|| at cells OUTSIDE the ~7x7 conv RF of both edits (so it is propagated, not conv leakage),
plus the agent cell and the readout value V. Every edit is re-decoded to verify it landed.

  reward-move dh2 (far) << wall dh2 (far)  AND  V still shifts  -> SUCCESSOR REPRESENTATION
  reward-move dh2 (far) ~  wall dh2 (far)                       -> POLICY EVALUATION (goal in the iterate)

  python -m experiments.interp.e2 --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, TARGET, AGENT

TARGET_RGB = np.array([254, 126, 125], dtype=np.uint8)
WALL_RGB = np.array([0, 0, 0], dtype=np.uint8)


def readout_value(h_fin, emb, Wd, bd, Wc, bc, hs):
    B, S, C = h_fin.shape
    mlp = np.maximum((h_fin + emb.reshape(B, S, C)).reshape(B, S * C) @ Wd + bd, 0.0)
    return (mlp @ Wc + bc)[..., 0] * hs


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    hs = getattr(net, "head_scale", 1.0)
    NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wc = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bc = np.asarray(params["params"]["critic_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)            # (B,3,H,W) uint8
    B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0)
    rc = np.arange(S); RR, CC = rc // W, rc % W                                        # cell row/col

    obs_g = obs0.copy(); obs_w = obs0.copy()
    t_old = np.full(B, -1); t_new = np.full(B, -1); a_sq = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]
        fl = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl)):
            continue
        a = int(ag[0]); to = int(tg[0]); a_sq[b] = a; t_old[b] = to
        ar, ac = divmod(a, W); tor, toc = divmod(to, W)
        # candidate far floor cells: outside conv RF (euclid>=4) of BOTH agent and old target
        eu_a = np.hypot(RR[fl] - ar, CC[fl] - ac); eu_t = np.hypot(RR[fl] - tor, CC[fl] - toc)
        cand = fl[(eu_a >= 4.0) & (eu_t >= 4.0)]
        if not len(cand):
            continue
        tn = int(cand[np.argmax(np.hypot(RR[cand] - ar, CC[cand] - ac))])               # farthest such cell from agent
        t_new[b] = tn; tnr, tnc = divmod(tn, W)
        fr, fc = divmod(int(fl[0]), W); floor_rgb = obs0[b, :, fr, fc]                   # sample a real floor colour
        obs_g[b, :, tor, toc] = floor_rgb                                               # vacate old target -> floor
        obs_g[b, :, tnr, tnc] = TARGET_RGB                                              # new target at far cell
        obs_w[b, :, tnr, tnc] = WALL_RGB                                                # wall at the same far cell

    # verify every edit decoded as intended
    tg_g = decode_tiles(obs_g); tg_w = decode_tiles(obs_w)
    ok = np.zeros(B, bool)
    for b in range(B):
        if t_new[b] < 0:
            continue
        ok[b] = (tg_g[b, t_new[b]] == TARGET and tg_g[b, t_old[b]] == FLOOR and tg_w[b, t_new[b]] == WALL)
    print(f"\n===== E2: REWARD-RELABEL INVARIANCE (step={step}, boards={B}, clean relabels={int(ok.sum())}, K={K}) =====")

    emb0 = np.asarray(get_embed(policy, params, jnp.asarray(obs0)))
    embg = np.asarray(get_embed(policy, params, jnp.asarray(obs_g)))
    embw = np.asarray(get_embed(policy, params, jnp.asarray(obs_w)))
    h0 = np.asarray(recompute_d3(cps, jnp.asarray(emb0), K)[0])[-1]                      # (B,S,C) final-tick top hidden
    hg = np.asarray(recompute_d3(cps, jnp.asarray(embg), K)[0])[-1]
    hw = np.asarray(recompute_d3(cps, jnp.asarray(embw), K)[0])[-1]
    V0 = readout_value(h0, emb0, Wd, bd, Wc, bc, hs)
    Vg = readout_value(hg, embg, Wd, bd, Wc, bc, hs)
    Vw = readout_value(hw, embw, Wd, bd, Wc, bc, hs)
    stdV = float(np.std(V0))

    g_far, w_far, g_ag, w_ag = [], [], [], []
    for b in range(B):
        if not ok[b]:
            continue
        to, tn, a = t_old[b], t_new[b], a_sq[b]
        nrm = np.linalg.norm(h0[b], axis=-1) + 1e-9
        dg = np.linalg.norm(hg[b] - h0[b], axis=-1) / nrm                                # relative latent shift per cell
        dw = np.linalg.norm(hw[b] - h0[b], axis=-1) / nrm
        eu_to = np.hypot(RR - RR[to], CC - CC[to]); eu_tn = np.hypot(RR - RR[tn], CC - CC[tn])
        farm = (tiles[b] != WALL) & (eu_to >= 4.0) & (eu_tn >= 4.0)                      # outside conv RF of both edits
        if farm.sum() >= 3:
            g_far.append(float(dg[farm].mean())); w_far.append(float(dw[farm].mean()))
        if eu_to[a] >= 4.0 and eu_tn[a] >= 4.0:                                          # agent clean of both edits
            g_ag.append(float(dg[a])); w_ag.append(float(dw[a]))

    gf, wf = np.mean(g_far), np.mean(w_far)
    dVg = float(np.mean(np.abs(Vg[ok] - V0[ok]))) / (stdV + 1e-9)
    dVw = float(np.mean(np.abs(Vw[ok] - V0[ok]))) / (stdV + 1e-9)
    print(f"  far-field relative latent shift ||dh2||/||h2||  (cells > conv-RF from both edits) --")
    print(f"     reward-move (goal relocated): {gf:.4f}   over n={len(g_far)} boards")
    print(f"     wall-flip   (P changed)     : {wf:.4f}")
    print(f"     ratio reward/wall           : {gf/(wf+1e-9):.2f}")
    if g_ag:
        print(f"  agent-cell relative latent shift (agent clean of both edits, n={len(g_ag)}) --")
        print(f"     reward-move {np.mean(g_ag):.4f}   wall-flip {np.mean(w_ag):.4f}   ratio {np.mean(g_ag)/(np.mean(w_ag)+1e-9):.2f}")
    print(f"  readout value shift |dV|/std(V) : reward-move {dVg:.3f}   wall-flip {dVw:.3f}   (std(V)={stdV:.2f})")
    ratio = gf / (wf + 1e-9)
    print("  -- VERDICT --")
    if ratio < 0.5 and dVg > 0.1:
        print(f"     reward-move barely propagates in the latent (ratio {ratio:.2f}) BUT value V still shifts ({dVg:.2f})")
        print(f"     -> reward-AGNOSTIC propagated features + value at readout = SUCCESSOR-REPRESENTATION-like")
    elif ratio > 0.8:
        print(f"     reward-move propagates ~as much as a P-change (ratio {ratio:.2f})")
        print(f"     -> goal is BAKED INTO the propagated field = POLICY-EVALUATION-like")
    else:
        print(f"     intermediate (ratio {ratio:.2f}, dV {dVg:.2f}) -> partially goal-conditioned propagation")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
