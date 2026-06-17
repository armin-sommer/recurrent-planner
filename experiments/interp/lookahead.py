"""LOOKAHEAD-CONSISTENCY test: does the policy evaluate actions at the states they transition INTO?

For the agent state s0, step into each action's successor s'_a (real env), read the model's OWN value
V(s'_a), and check whether the chosen action a* = argmax_a [ r_a + gamma*V(s'_a) ] -- i.e. is the policy
consistent with 1-step lookahead over its own value (evaluating actions at their successor states)? And
does that consistency RISE with thinking depth (the loop doing the lookahead)?

Logits AND value are computed from the crash-free recompute_d3 (faithful to the model; self-test=1.8e-7)
plus the MLP + actor/critic head matmuls -- we do NOT call the model's get_action/get_logits_and_value,
which hit the rel_bias-gather tracer error under nn.scan. GPU-safe.

  python -m experiments.interp.lookahead --ckpt <cp_dir> --boards 256
"""
from __future__ import annotations
import argparse, dataclasses
import numpy as np
import jax, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed


def main(cp_dir, n_boards):
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

    def logits_value(obs):                                                       # (B,3,H,W) -> logits (K,B,n_act), value (K,B)
        B = obs.shape[0]
        emb = np.asarray(get_embed(policy, params, jnp.asarray(obs)))            # (B,H,W,C)
        S = emb.shape[1] * emb.shape[2]; C = emb.shape[3]
        top_h = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])            # (K,B,S,C)
        mlp = np.maximum((top_h + emb.reshape(B, S, C)[None]).reshape(K, B, S * C) @ Wd + bd, 0.0)  # (K,B,256), IdentityNorm
        return mlp @ Wa + ba, ((mlp @ Wc + bc)[..., 0] * hs)

    def reset_obs():
        e = env_cfg.make(); o, _ = e.reset(); return e, np.asarray(o)

    env0, obs0 = reset_obs(); B = obs0.shape[0]
    logits0, _ = logits_value(obs0)                                             # (K,B,n_act)
    a_star = {k: logits0[k - 1].argmax(-1) for k in range(1, K + 1)}             # greedy action at each thinking depth

    Q = np.full((n_act, B), -1e9); det_ok = True
    for a in range(n_act):
        e, ochk = reset_obs(); det_ok &= bool(np.array_equal(ochk, obs0))
        e.step_async(np.full(B, a, np.int32)); o2, r, term, trunc, _ = e.step_wait()
        done = np.asarray(term) | np.asarray(trunc)
        _, v2 = logits_value(np.asarray(o2))                                     # V(s'_a) full depth
        Q[a] = np.asarray(r) + gamma * v2[K - 1] * (1 - done)
    a_LA = Q.argmax(0)                                                          # 1-step lookahead over own value

    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    ch = 1.0 / n_act
    print(f"\n===== LOOKAHEAD CONSISTENCY (step={step}, boards={B}, n_act={n_act}, gamma={gamma}, K={K}) =====")
    print(f"  deterministic level load across resets: {det_ok}  (must be True)")
    agree = float((a_star[K] == a_LA).mean())
    print(f"  policy(full depth) == 1-step-lookahead(own value): {agree:.3f}   (chance {ch:.2f})")
    by_depth = [float((a_star[k] == a_LA).mean()) for k in range(1, K + 1)]
    print(f"  agreement w/ lookahead by thinking depth d=1..{K}: {f(by_depth)}  d1->K {by_depth[0]:.2f}->{by_depth[-1]:.2f}")
    flips = float(((a_star[1] != a_star[K]) & (a_star[K] == a_LA)).mean())
    print(f"  boards where thinking changed the action TO the lookahead choice: {flips:.3f}")
    print(f"  --> {'EVALUATES SUCCESSORS (lookahead-consistent)' if agree > ch + 0.15 else 'NOT clearly lookahead-consistent'}"
          f"; thinking {'INCREASES' if by_depth[-1] > by_depth[0] + 0.03 else 'does not increase'} consistency")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
