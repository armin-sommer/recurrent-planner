"""BELLMAN SELF-CONSISTENCY: is the value V a value-iteration fixed point at the agent AND recursively
at successor states? (D=3 entmax core.)

If the weight-tied operator makes every state do the same max-backup, then V(s) ~= max_a[r_a + gamma
V(s'_a)] should hold not just at the agent (depth 0) but at the states it transitions into (depth 1, 2,
...). We measure the OPTIMALITY residual  resid_k = V(s_k) - max_a[r_a + gamma V(s'_{k,a})]  along the
greedy trajectory s_0 -> s_1 -> ... (branching all actions at each depth via deterministic env replay),
plus the cheap on-policy TD residual  V(s_t) - (r_t + gamma V(s_{t+1})).

  low & FLAT across depth  -> V is a VI fixed point everywhere -> the recursion holds (successors also
                             back up) -> effectively multi-step value propagation, unbounded-ish depth.
  grows with depth         -> fixed point near the agent, degrades deeper -> BOUNDED effective depth.

V and the greedy policy are read from the crash-free recompute_d3 + head matmuls (GPU-safe). Successors
come from the real env (deterministic level load -> replay recorded greedy actions, then branch).

  python -m experiments.interp.bellman --ckpt <cp_dir> --boards 128 --depth 4
"""
from __future__ import annotations
import argparse, dataclasses
import numpy as np
import jax, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed


def main(cp_dir, n_boards, Kdepth):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    base = planning_eval_envs()["valid_medium"].env
    env_cfg = dataclasses.replace(base, num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    gamma = cp_cfg.loss.gamma; hs = getattr(net, "head_scale", 1.0)
    n_act = int(env_cfg.make().single_action_space.n)
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wa = np.asarray(params["params"]["actor_params"]["Output"]["kernel"]); ba = np.asarray(params["params"]["actor_params"]["Output"]["bias"])
    Wc = np.asarray(params["params"]["critic_params"]["Output"]["kernel"]); bc = np.asarray(params["params"]["critic_params"]["Output"]["bias"])

    def vl(obs):                                                                  # -> logits (B,n_act), value (B,)
        B = obs.shape[0]; emb = np.asarray(get_embed(policy, params, jnp.asarray(obs)))
        S = emb.shape[1] * emb.shape[2]; C = emb.shape[3]
        th = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])[-1]            # full depth top hidden (B,S,C)
        mlp = np.maximum((th + emb.reshape(B, S, C)).reshape(B, S * C) @ Wd + bd, 0.0)
        return mlp @ Wa + ba, (mlp @ Wc + bc)[..., 0] * hs

    def env_step(e, a):
        e.step_async(np.asarray(a, np.int32)); o, r, term, trunc, _ = e.step_wait()
        return np.asarray(o), np.asarray(r), np.asarray(term) | np.asarray(trunc)

    # ---- greedy rollout, recording states/actions/rewards ----
    env = env_cfg.make(); obs0, _ = env.reset(); obs0 = np.asarray(obs0)
    obs_t = [obs0]; act_t = []; rew_t = []; done_t = []; V_t = []
    o = obs0; alive = np.ones(obs0.shape[0], bool)
    for t in range(Kdepth):
        lg, v = vl(o); V_t.append(v); a = lg.argmax(-1); act_t.append(a)
        o, r, dn = env_step(env, a); rew_t.append(r); done_t.append(dn); obs_t.append(o)
    V_t.append(vl(o)[1])                                                          # V(s_Kdepth)

    # ---- optimality residual at each depth (branch all actions via deterministic replay) ----
    print(f"\n===== BELLMAN SELF-CONSISTENCY (step={step}, boards={obs0.shape[0]}, gamma={gamma}, K={K}) =====")
    Vstd = float(np.std(V_t[0]))
    det_ok = True
    f = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    opt_res = []
    for k in range(Kdepth):
        Q = np.full((n_act, obs0.shape[0]), -1e9)
        for a in range(n_act):
            e = env_cfg.make(); ock, _ = e.reset()
            det_ok &= bool(np.array_equal(np.asarray(ock), obs0))
            for t in range(k):
                env_step(e, act_t[t])                                            # replay recorded greedy to s_k
            o2, r2, dn2 = env_step(e, np.full(obs0.shape[0], a))                 # branch action a
            Q[a] = r2 + gamma * vl(o2)[1] * (1 - dn2)
        resid = V_t[k] - Q.max(0)
        opt_res.append(float(np.mean(np.abs(resid))))
        print(f"  depth {k}: |V - max_a(r+gV(s'))| = {opt_res[-1]:.3f}   "
              f"(V_mean={V_t[k].mean():.2f}, maxQ_mean={Q.max(0).mean():.2f}, signed={float(resid.mean()):+.3f})")
    # ---- on-policy TD residual along the greedy trajectory ----
    td = [float(np.mean(np.abs(V_t[t] - (rew_t[t] + gamma * V_t[t + 1] * (1 - done_t[t]))))) for t in range(Kdepth)]
    print(f"  deterministic replay: {det_ok}   value spread std(V)={Vstd:.2f}")
    print(f"  optimality residual / std(V) by depth: {f([r / (Vstd + 1e-9) for r in opt_res])}")
    print(f"  on-policy TD residual by depth        : {f(td)}")
    grow = opt_res[-1] - opt_res[0]
    print(f"  --> {'FIXED POINT at all depths (recursion holds: VI-like multi-step)' if (opt_res[0] < 0.5 * Vstd and abs(grow) < 0.5 * Vstd) else ('residual GROWS with depth -> bounded effective depth' if grow > 0.5 * Vstd else 'residual already large at depth 0 -> V not a clean backup fixed point')}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=128); ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.depth)
