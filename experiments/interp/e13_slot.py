"""E13 (SLOT CORE port) -- does the model's OWN decision change & improve over thinking ticks?

This is the slot-core analogue of experiments/interp/e13.py (written for the spatial ATTENTION core).
E13 is the MOST PORTABLE probe: it never needs the binding sigma. It asks whether the model's *own*
actor/critic decision changes and converges over the K thinking ticks, and whether it gets more goalward
(esp. on far boards, where value must propagate further). The only thing that changes vs the attn core is
HOW we get the per-tick board hidden + HOW we run the head:

  * attn core: recompute_d3(...) gives per-tick cell hidden (K,B,S,C); heads = (th + embed_skip) flattened
    -> dense_list_0 -> actor/critic. (cells == board squares; skip_final=True adds the embed residual.)
  * slot core: slot_per_tick(...) gives per-tick TOP-SLOT hidden (K,B,N,d); the slots are FREE (not tied
    to board position) -- but E13 doesn't care, it reads the GLOBAL decision. We feed each tick's slot
    hidden through the model's OWN _mlp (flatten (B,N,d)->(B,N*d) -> dense_list -> norm/relu) and the
    actor/critic heads. skip_final is False for the slot core (SlotLSTMConfig requires it), so there is
    NO embed-residual add -- the head reads the slots directly, which is exactly what the trained model
    does. We apply the real Flax modules (network_params._mlp, actor_params, critic_params) per tick so
    the readout is faithful (handles norm + head_scale exactly as the model does at inference).

Everything else -- the BFS greedy ground-truth, the distance bands, the action->dir recovery, the
change-rate / goalward / margin / |dV| metrics, the printed summary + machine-readable PLOT_ line -- is
kept identical to the original probe.

  python -m experiments.interp.e13_slot --ckpt <cp_dir> --boards 256 --ticks 8
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.slot_interp import slot_per_tick
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, greedy_dir, TARGET, AGENT

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BANDS = [(1, 3), (4, 7), (8, 12), (13, 99)]


def main(cp_dir, n_boards, K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; Ktr = net.repeats_per_step; D = net.n_recurrent

    envs = env_cfg.make(); obs = np.asarray(envs.reset()[0])               # (B,3,H,W)
    B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)

    # ---- per-tick TOP-SLOT hidden via the slot foundation (replaces recompute_d3) ----
    # h: (K,B,N,d). We only need the hidden for E13 (no sigma needed -- the head is a GLOBAL readout).
    h, _bind, _route = slot_per_tick(policy, params, jnp.asarray(obs), K)   # h:(K,B,N,d)

    # ---- the model's OWN readout per tick: feed slot hidden -> _mlp -> actor/critic (faithful) ----
    # Mirrors network.Policy.get_logits_and_value, but skips the recurrence: we already have the per-tick
    # slot hidden, so we run only the post-recurrence head stack on it. _mlp flattens (B,N,d)->(B,N*d),
    # applies dense_list with the config norm + relu; the heads apply the same norm; the critic already
    # multiplies by head_scale internally (Critic.kernel_scale), so we do NOT rescale here.
    def heads(m, hidden_bnd):                                              # hidden_bnd: (B,N,d)
        feat = m.network_params._mlp(hidden_bnd)                           # (B, mlp_hidden)
        logits, _ = m.actor_params(feat)                                   # (B, nact)
        value, _ = m.critic_params(feat)                                   # (B, 1) (already *head_scale)
        return logits, value.squeeze(-1)                                   # (B,nact), (B,)

    # per-board: agent square, optimal first move (BFS greedy), and difficulty = agent->goal distance
    agent = np.full(B, -1); optd = np.full(B, -1, int); dist = np.full(B, np.nan)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]; tg = np.where(tiles[b] == TARGET)[0]
        if len(ag) and len(tg):
            agent[b] = int(ag[0]); dT = bfs_from([int(tg[0])], tiles[b], H, W)
            if np.isfinite(dT[agent[b]]):
                dist[b] = dT[agent[b]]; optd[b] = int(greedy_dir(tiles[b], dT, H, W)[agent[b]])
    valid = (agent >= 0) & np.isfinite(dist) & (optd >= 0) & (dist > 0)

    L = None; V = np.zeros((K, B))
    for t in range(K):
        lt, vt = policy.apply(params, jnp.asarray(h[t]), method=heads)
        lt = np.asarray(lt); vt = np.asarray(vt)
        if L is None:
            nact = lt.shape[1]; L = np.zeros((K, B, nact))
        L[t] = lt; V[t] = vt
    nact = L.shape[-1]
    act = L.argmax(-1)                                                     # (K,B) the model's decision per tick

    # recover action-index -> DIRS mapping from the settled tick (trained policy is mostly goalward)
    conf = np.zeros((nact, 4))
    for b in np.where(valid)[0]:
        conf[act[-1, b], optd[b]] += 1
    amap = conf.argmax(1)                                                  # amap[action_idx] -> DIRS idx
    mapped = amap[act]                                                     # (K,B)

    def bmask(lo, hi): return valid & (dist >= lo) & (dist <= hi)
    f = lambda xs: "[" + " ".join(("%.2f" % x if np.isfinite(x) else " . ") for x in xs) + "]"

    opt_all = np.array([(mapped[t][valid] == optd[valid]).mean() for t in range(K)])
    chg_all = np.array([(act[t][valid] != act[-1][valid]).mean() for t in range(K)])
    margin = np.array([(np.sort(L[t][valid], 1)[:, -1] - np.sort(L[t][valid], 1)[:, -2]).mean() for t in range(K)])
    Vmean = np.array([V[t][valid].mean() for t in range(K)])
    dV = np.array([np.nan if t == 0 else np.abs(V[t][valid] - V[t - 1][valid]).mean() for t in range(K)])
    opt_band = np.array([[ (mapped[t][bmask(lo, hi)] == optd[bmask(lo, hi)]).mean() if bmask(lo, hi).sum() >= 15 else np.nan
                          for (lo, hi) in BANDS] for t in range(K)])
    chg1K_band = [float((act[0][bmask(lo, hi)] != act[-1][bmask(lo, hi)]).mean()) if bmask(lo, hi).sum() >= 15 else np.nan for (lo, hi) in BANDS]
    n_band = [int(bmask(lo, hi).sum()) for (lo, hi) in BANDS]
    bl = [f"d{lo}-{hi if hi < 99 else '+'}" for lo, hi in BANDS]

    NS = h.shape[2]
    print(f"\n===== E13 (slot core): DOES THE MODEL'S OWN DECISION CHANGE & IMPROVE OVER TICKS? (step={step}, boards={B}, valid={int(valid.sum())}, slots={NS}, D={D}, K={K}, K_train={Ktr}) =====")
    print(f"  recovered action->dir map = {list(map(int, amap))}; settled-tick optimality (goalward) = {opt_all[-1]:.3f}  (chance 0.25)")
    print(f"  board counts by agent->goal distance: " + "  ".join(f"{b}:{n}" for b, n in zip(bl, n_band)))
    print(f"  (1) DOES THINKING CHANGE THE DECISION?  action_t != settled-action, per tick:")
    print(f"        {f(chg_all)}   tick1 vs settled = {chg_all[0]:.3f} of boards differ")
    print(f"      action change tick1->settled, by distance band: " + "  ".join(f"{b}={(c if np.isfinite(c) else float('nan')):.3f}" for b, c in zip(bl, chg1K_band)))
    print(f"  (2) DOES THINKING IMPROVE THE DECISION?  optimality (action is goalward) per tick:")
    print(f"        overall: {f(opt_all)}   tick1->K  {opt_all[0]:.3f} -> {opt_all[-1]:.3f}  (delta {opt_all[-1]-opt_all[0]:+.3f})")
    print(f"        by band, tick \\ band   " + "  ".join(f"{b:>6}" for b in bl))
    for t in range(K):
        print(f"            {t+1:>4}           " + "  ".join(f"{opt_band[t,j]:6.2f}" if np.isfinite(opt_band[t, j]) else "   .  " for j in range(len(BANDS))))
    print(f"      per-band optimality tick1->K:")
    for j, b in enumerate(bl):
        col = opt_band[:, j][np.isfinite(opt_band[:, j])]
        print(f"        {b:>6}: {col[0]:.2f} -> {col[-1]:.2f}  (delta {col[-1]-col[0]:+.2f})" if len(col) else f"        {b:>6}: n/a")
    print(f"  (3) THE VALUE IT RIDES ON:  mean state-value per tick {f(Vmean)}")
    print(f"        |dV| per tick (settling) {f(dV)}   ;  decision margin (top1-top2 logit) {f(margin)}")
    verdict_chg = chg_all[0] > 0.05
    verdict_imp = (opt_all[-1] - opt_all[0]) > 0.02 or (np.isfinite(chg1K_band[-1]) and chg1K_band[-1] > chg1K_band[0] + 0.05)
    print(f"  --> thinking CHANGES the decision: {'YES' if verdict_chg else 'little'};  IMPROVES it (more goalward / more on far boards): {'YES' if verdict_imp else 'not clearly'}")
    print("PLOT_E13_SLOT=" + repr(dict(chg_all=[round(float(x), 3) for x in chg_all], opt_all=[round(float(x), 3) for x in opt_all],
                                   opt_band=[[round(float(opt_band[t, j]), 3) if np.isfinite(opt_band[t, j]) else None for j in range(len(BANDS))] for t in range(K)],
                                   chg1K_band=[round(float(c), 3) if np.isfinite(c) else None for c in chg1K_band], bands=bl,
                                   Vmean=[round(float(x), 3) for x in Vmean], dV=[round(float(x), 4) if np.isfinite(x) else None for x in dV],
                                   margin=[round(float(x), 3) for x in margin], amap=list(map(int, amap)), opt_settled=round(float(opt_all[-1]), 3),
                                   slots=int(NS), D=int(D))))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256); ap.add_argument("--ticks", type=int, default=8)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.ticks)
