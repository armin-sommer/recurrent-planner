"""E10 (SLOT CORE) -- does the value-change from a new wall reach the AGENT's slot + re-plan its move?

Slot port of experiments/interp/e10.py. The attention core assumed latent cell i == board square i, so
"the agent's cell" was just board-position[agent]. The slot core has N FREE slots not tied to the grid:
the binding sigma (slot <-> board position) is LEARNED. So here we go through sigma -- decode_sigma turns
the learned slot-attention binding into pos[b,slot] = the board position that slot reads from, and "the
agent's cell" becomes "the slot whose bound position is the agent square".

E9/the spatial E10 showed a path-blocking wall lowers the (board) value. Here we ask: does that change
reach the AGENT's own slot and re-plan its action -- or stay out near the wall? We place the wall on the
agent->goal path but OUTSIDE the agent's ~7x7 conv view (so any effect at the agent arrives by recurrent
propagation, not local pixels), require it to actually lengthen the geodesic, and compare to an OFF-path
cosmetic wall at matched distance.

  value-change at the AGENT slot (decoded value, + per-tick agent-slot latent shift)
  greedy-action change rate (does the model re-plan its first move at the settled tick?)
on-path >> off-path on both (path-block ~3x the cosmetic wall) => the re-formed value propagates back to
the agent slot and changes its decision.

  python -m experiments.interp.e10_slot --ckpt <cp_dir> --boards 256 --ticks 12
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick, decode_sigma
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_RGB = np.array([0, 0, 0], np.uint8)


def geodesic_path(agent, dT, H, W):
    """Greedy descent of the BFS distance-to-target field: the agent->goal shortest path (list of squares)."""
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
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step
    hs = getattr(net, "head_scale", 1.0); NP = params["params"]["network_params"]
    # Head readout (slot core): BaseLSTM._mlp flattens the (B,N,d) slot hidden -> (B,N*d), then
    # dense_list_0 -> relu -> actor/critic Output. norm is identity in this recipe. skip_final=False
    # so there is NO spatial-embed skip-add (unlike the attn core). So the head is a pure function of
    # the flattened per-tick slot hidden -- we apply it tick by tick below.
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wa = np.asarray(params["params"]["actor_params"]["Output"]["kernel"]); ba = np.asarray(params["params"]["actor_params"]["Output"]["bias"])
    Wv = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bv = np.asarray(params["params"]["critic_params"]["Output"]["bias"])

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W

    # ---- build the on-path (path-blocking) and off-path (cosmetic) wall perturbations ----
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
        # off-path wall: floor cell off the path at euclid>=4 from agent, kept >=2 from the path (cosmetic)
        cand = [c for c in fl if c not in pathset and eu_a(c) >= 4.0 and min(np.hypot(RR[c] - RR[p], CC[c] - CC[p]) for p in path) >= 2.0]
        if cand:
            coff = int(cand[len(cand) // 2]); obs_off[b, :, RR[coff], CC[coff]] = WALL_RGB; useoff[b] = True
    ok = useon & useoff

    # ---- run the slot recurrence for K ticks on each variant; recover sigma from the binding ----
    def fields(o):
        h, bind, _route = slot_per_tick(policy, params, jnp.asarray(o), K)   # h:(K,B,N,d) bind:(K,B,nh,N,S)
        pos, mass = decode_sigma(bind, tick=-1)                              # pos:(B,N) slot->board-pos; mass:(B,N)
        return np.asarray(h), pos, mass

    def head(h_t):
        """Apply the model's own readout to one tick's slot hidden h_t:(B,N,d). Returns logits (B,nact), value (B,)."""
        B_, N_, d_ = h_t.shape
        flat = h_t.reshape(B_, N_ * d_)
        mlp = np.maximum(flat @ Wd + bd, 0.0)
        logits = mlp @ Wa + ba
        value = (mlp @ Wv + bv).reshape(B_) * hs
        return logits, value

    h0, pos0, mass0 = fields(obs0); hn, posn, _ = fields(obs_on); hf, posf, _ = fields(obs_off)
    N = h0.shape[2]

    # ---- agent SLOT per board: the slot whose decoded position is the agent square (sigma mapping) ----
    # "the agent's cell" (attn core, cell==square) -> "the slot bound to the agent square" (slot core).
    # If several slots bind the agent square, take the most confident (max binding mass) one.
    AS = np.full(B, -1)
    for b in range(B):
        if A0[b] < 0:
            continue
        cand = np.where(pos0[b] == A0[b])[0]
        if len(cand):
            AS[b] = int(cand[np.argmax(mass0[b, cand])])
    has_slot = AS >= 0
    m = ok & has_slot

    # ---- (1) per-tick latent shift at the AGENT SLOT (does the change reach it + build over ticks?) ----
    def agent_slot_shift(h_int):
        out = np.zeros((K, B))
        for b in range(B):
            if AS[b] < 0:
                continue
            s = AS[b]
            out[:, b] = np.linalg.norm(h_int[:, b, s] - h0[:, b, s], axis=-1) / (np.linalg.norm(h0[:, b, s], axis=-1) + 1e-9)
        return out
    dh_on = agent_slot_shift(hn); dh_off = agent_slot_shift(hf)

    # ---- (2) decoded VALUE-change at the settled tick (path-block should move value ~3x the cosmetic wall) ----
    # Value is a global readout, but its argument is the slot field; with the agent slot re-bound the
    # value the agent's decision rides on shifts. We report the change in decoded state-value, settled tick.
    ak = Ktr - 1  # trained thinking depth
    _, v0_tr = head(h0[ak]); _, von_tr = head(hn[ak]); _, voff_tr = head(hf[ak])
    _, v0_f = head(h0[-1]); _, von_f = head(hn[-1]); _, voff_f = head(hf[-1])
    dv_on_tr = np.abs(von_tr - v0_tr); dv_off_tr = np.abs(voff_tr - v0_tr)
    dv_on_f = np.abs(von_f - v0_f); dv_off_f = np.abs(voff_f - v0_f)

    # ---- (3) first-action change (apply actor head, settled tick): does the model re-plan its move? ----
    def act(h_t): return head(h_t)[0].argmax(-1)
    a0_tr = act(h0[ak]); aon_tr = act(hn[ak]); aoff_tr = act(hf[ak])
    a0_f = act(h0[-1]); aon_f = act(hn[-1]); aoff_f = act(hf[-1])

    f2 = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    sh_on = dh_on[-1, m].mean(); sh_off = dh_off[-1, m].mean()
    vr_on = dv_on_f[m].mean(); vr_off = dv_off_f[m].mean()
    act_on = float((a0_f[m] != aon_f[m]).mean()); act_off = float((a0_f[m] != aoff_f[m]).mean())

    print(f"\n===== E10 (slot core): DOES THE WALL'S VALUE-CHANGE REACH THE AGENT SLOT + RE-PLAN? (step={step}, boards={B}, n={int(m.sum())}, K={K}, slots={N}) =====")
    print(f"  via sigma: 'agent cell' -> the SLOT bound to the agent square (decode_sigma); wall placed on the")
    print(f"  agent->goal path, OUTSIDE the agent's conv view (so effect at the agent slot is propagated, not local).")
    print(f"  agent slots found (sigma binds the agent square): {int(has_slot.sum())}/{B}")
    print(f"  -- (1) agent-SLOT latent shift ||dh(agent slot)||/||h|| per tick:")
    print(f"       on-path : {f2(dh_on[:, m].mean(1))}")
    print(f"       off-path: {f2(dh_off[:, m].mean(1))}")
    print(f"       final-tick on/off ratio = {sh_on / (sh_off + 1e-9):.2f}")
    print(f"  -- (2) decoded VALUE-change at the agent's decision |dV| (settled tick K={K}):")
    print(f"       on-path {vr_on:.4f}   off-path {vr_off:.4f}   on/off ratio = {vr_on / (vr_off + 1e-9):.2f}")
    print(f"       (trained depth K={Ktr}: on {dv_on_tr[m].mean():.4f}  off {dv_off_tr[m].mean():.4f}  ratio {dv_on_tr[m].mean()/(dv_off_tr[m].mean()+1e-9):.2f})")
    print(f"  -- (3) greedy first-action CHANGE rate (agent re-plans its move, actor head):")
    print(f"       trained depth K={Ktr}: on-path {float((a0_tr[m]!=aon_tr[m]).mean()):.3f}   off-path {float((a0_tr[m]!=aoff_tr[m]).mean()):.3f}")
    print(f"       full depth  K={K}: on-path {act_on:.3f}   off-path {act_off:.3f}")
    reach = sh_on > 1.3 * sh_off
    moves = vr_on > 1.5 * vr_off          # value moves ~more (target ~3x) for the path-block
    replan = act_on > 1.5 * act_off
    print(f"  --> reaches the agent slot: {'YES' if reach else 'not clearly'};  moves the decoded value: {'YES' if moves else 'not clearly'};  re-plans the action: {'YES' if replan else 'not clearly'}")
    print("PLOT_E10_SLOT=" + repr(dict(
        dh_on=[round(float(x), 4) for x in dh_on[:, m].mean(1)],
        dh_off=[round(float(x), 4) for x in dh_off[:, m].mean(1)],
        dv_on=round(float(vr_on), 4), dv_off=round(float(vr_off), 4), dv_ratio=round(float(vr_on / (vr_off + 1e-9)), 2),
        act_on=round(act_on, 3), act_off=round(act_off, 3),
        n=int(m.sum()), slots=int(N))))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256); ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
