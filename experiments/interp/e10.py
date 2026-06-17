"""E10 -- does the value-change from a new wall reach the AGENT's square and change its DECISION?

E9 showed a path-blocking wall lowers the (board) value. But does that change reach the AGENT's own cell
and re-plan its action -- or does it stay out near the wall? We place the wall on the agent->goal path but
OUTSIDE the agent's ~7x7 conv view (so any effect at the agent arrives by recurrent propagation, not local
pixels), require it to actually lengthen the path, and compare to an OFF-path wall at matched distance.

  agent-cell latent shift ||dh(agent)||  -- does the change reach the agent's square (and build over ticks)?
  greedy-action change rate              -- does it re-plan the agent's move?
on-path >> off-path on both => the re-formed value propagates back to the agent and changes its decision.

  python -m experiments.interp.e10 --ckpt <cp_dir> --boards 200 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_RGB = np.array([0, 0, 0], np.uint8)


def geodesic_path(agent, dT, H, W):
    s = agent; path = [s]
    while np.isfinite(dT[s]) and dT[s] > 0:
        r, c = divmod(s, W); best, bd = None, dT[s]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dT[nr * W + nc]) and dT[nr * W + nc] < bd:
                bd = dT[nr * W + nc]; best = nr * W + nc
        if best is None:
            break
        s = best; path.append(s)
    return path


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent
    hs = getattr(net, "head_scale", 1.0); NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wa = np.asarray(params["params"]["actor_params"]["Output"]["kernel"]); ba = np.asarray(params["params"]["actor_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W

    obs_on = obs0.copy(); obs_off = obs0.copy(); A0 = np.full(B, -1); useon = np.zeros(B, bool); useoff = np.zeros(B, bool)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]; fl = np.where(tiles[b] == FLOOR)[0]
        if not (len(ag) and len(tg) and len(fl)):
            continue
        a = int(ag[0]); A0[b] = a; dT = bfs_from([int(tg[0])], tiles[b], H, W)
        if not np.isfinite(dT[a]):
            continue
        path = geodesic_path(a, dT, H, W); pathset = set(path)
        eu_a = lambda c: np.hypot(RR[c] - RR[a], CC[c] - CC[a])
        # on-path wall: a path floor cell OUTSIDE the agent RF that lengthens the geodesic
        for c in path:
            if tiles[b, c] == FLOOR and eu_a(c) >= 4.0:
                t2 = tiles[b].copy(); t2[c] = WALL
                if bfs_from([int(tg[0])], t2, H, W)[a] > dT[a]:
                    obs_on[b, :, RR[c], CC[c]] = WALL_RGB; useon[b] = True; break
        # off-path wall: floor cell off the path at euclid>=4 from agent
        cand = [c for c in fl if c not in pathset and eu_a(c) >= 4.0 and min(np.hypot(RR[c] - RR[p], CC[c] - CC[p]) for p in path) >= 2.0]
        if cand:
            coff = int(cand[len(cand) // 2]); obs_off[b, :, RR[coff], CC[coff]] = WALL_RGB; useoff[b] = True
    ok = useon & useoff
    for nm, o in (("on", obs_on), ("off", obs_off)):
        td = decode_tiles(o)  # (verification folded below)

    def fields(o):
        emb = np.asarray(get_embed(policy, params, jnp.asarray(o)))
        th = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])                      # (K,B,S,C)
        return th, emb
    def action(th_t, emb):
        B_, S_, C_ = th_t.shape
        mlp = np.maximum((th_t + emb.reshape(B_, S_, C_)).reshape(B_, S_ * C_) @ Wd + bd, 0.0)
        return (mlp @ Wa + ba).argmax(-1)                                               # (B,)
    th0, e0 = fields(obs0); thn, en = fields(obs_on); thf, ef = fields(obs_off)

    # agent-cell latent shift per tick
    def agent_shift(th_int):
        out = np.zeros((K, B))
        for b in range(B):
            if A0[b] < 0:
                continue
            out[:, b] = np.linalg.norm(th_int[:, b, A0[b]] - th0[:, b, A0[b]], axis=-1) / (np.linalg.norm(th0[:, b, A0[b]], axis=-1) + 1e-9)
        return out
    dh_on = agent_shift(thn); dh_off = agent_shift(thf)
    # action change at trained depth and full depth
    ak = Ktr - 1
    a0_tr = action(th0[ak], e0); aon_tr = action(thn[ak], en); aoff_tr = action(thf[ak], ef)
    a0_f = action(th0[-1], e0); aon_f = action(thn[-1], en); aoff_f = action(thf[-1], ef)

    m = ok
    f2 = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print(f"\n===== E10: DOES THE WALL'S VALUE-CHANGE REACH THE AGENT + RE-PLAN? (step={step}, boards={B}, n={int(m.sum())}, K={K}) =====")
    print(f"  wall placed on the agent->goal path, OUTSIDE the agent's conv view (propagated, not local).")
    print(f"  agent-cell latent shift ||dh(agent)||/||h|| per tick:")
    print(f"     on-path : {f2(dh_on[:, m].mean(1))}")
    print(f"     off-path: {f2(dh_off[:, m].mean(1))}")
    print(f"     final-tick on/off ratio = {dh_on[-1, m].mean() / (dh_off[-1, m].mean() + 1e-9):.2f}")
    print(f"  greedy-action CHANGE rate (agent re-plans its move):")
    print(f"     trained depth K={Ktr}: on-path {float((a0_tr[m]!=aon_tr[m]).mean()):.3f}   off-path {float((a0_tr[m]!=aoff_tr[m]).mean()):.3f}")
    print(f"     full depth  K={K}: on-path {float((a0_f[m]!=aon_f[m]).mean()):.3f}   off-path {float((a0_f[m]!=aoff_f[m]).mean()):.3f}")
    reach = dh_on[-1, m].mean() > 1.3 * dh_off[-1, m].mean()
    replan = float((a0_f[m] != aon_f[m]).mean()) > 1.5 * float((a0_f[m] != aoff_f[m]).mean())
    print(f"  --> reaches the agent's square: {'YES' if reach else 'not clearly'};  re-plans the action: {'YES' if replan else 'not clearly'}")
    print("PLOT_E10=" + repr(dict(dh_on=[round(float(x),4) for x in dh_on[:, m].mean(1)],
                                   dh_off=[round(float(x),4) for x in dh_off[:, m].mean(1)],
                                   act_on=round(float((a0_f[m]!=aon_f[m]).mean()),3), act_off=round(float((a0_f[m]!=aoff_f[m]).mean()),3))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=200); ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
