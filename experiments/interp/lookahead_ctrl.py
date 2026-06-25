"""LOOKAHEAD with CONTROLS: is the policy genuinely greedy over its OWN value? (policy-improvement test)

Reproduces lookahead.py (policy vs a*=argmax_a[r_a + gamma V(s'_a)] over the model's own critic) and adds the
controls the adversarial panel asked for:
  (1) V-SHUFFLE: permute V(s'_a) across the action-successors within each board (deterministic rolls). If the
      action is genuinely greedy over its OWN successor values, agreement must COLLAPSE toward chance.
  (2) DECOMPOSE: agreement using Q=r only (immediate reward) vs Q=gamma V only (value), to see whether the
      lookahead-consistency rides on the reward channel or the value channel.
A real "policy improvement over its own value" needs: real agreement >> V-shuffled (~chance) AND value-only
agreement well above chance.

  python -m experiments.interp.lookahead_ctrl --ckpt <cp_dir> --boards 256
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

    def logits_value(obs):
        B = obs.shape[0]
        emb = np.asarray(get_embed(policy, params, jnp.asarray(obs)))
        S = emb.shape[1] * emb.shape[2]; C = emb.shape[3]
        top_h = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])
        mlp = np.maximum((top_h + emb.reshape(B, S, C)[None]).reshape(K, B, S * C) @ Wd + bd, 0.0)
        return mlp @ Wa + ba, ((mlp @ Wc + bc)[..., 0] * hs)

    def reset_obs():
        e = env_cfg.make(); o, _ = e.reset(); return e, np.asarray(o)

    env0, obs0 = reset_obs(); B = obs0.shape[0]
    logits0, _ = logits_value(obs0)
    a_star = {k: logits0[k - 1].argmax(-1) for k in range(1, K + 1)}

    R = np.zeros((n_act, B)); V2 = np.zeros((n_act, B)); det_ok = True
    for a in range(n_act):
        e, ochk = reset_obs(); det_ok &= bool(np.array_equal(ochk, obs0))
        e.step_async(np.full(B, a, np.int32)); o2, r, term, trunc, _ = e.step_wait()
        done = np.asarray(term) | np.asarray(trunc)
        _, v2 = logits_value(np.asarray(o2))
        R[a] = np.asarray(r); V2[a] = v2[K - 1] * (1 - done)

    aK = a_star[K]
    Q = R + gamma * V2
    agree = float((Q.argmax(0) == aK).mean())
    agree_r = float((R.argmax(0) == aK).mean())                                                    # reward only
    agree_V = float((V2.argmax(0) == aK).mean())                                                   # value only (gamma>0 monotone)
    # V-SHUFFLE control: permute V across actions within each board (deterministic rolls 1..n_act-1)
    shuf = []
    for s in range(1, n_act):
        Qs = R + gamma * np.roll(V2, s, axis=0)
        shuf.append(float((Qs.argmax(0) == aK).mean()))
    agree_shuf = float(np.mean(shuf))
    # V-only shuffle (isolate the value channel)
    shufV = [float((np.roll(V2, s, axis=0).argmax(0) == aK).mean()) for s in range(1, n_act)]

    ch = 1.0 / n_act
    f = lambda xs: "[" + " ".join("%.3f" % x for x in xs) + "]"
    print(f"\n===== LOOKAHEAD + CONTROLS (step={step}, boards={B}, n_act={n_act}, gamma={gamma}, K={K}, chance={ch:.2f}) =====")
    print(f"  deterministic resets: {det_ok}")
    print(f"  (real)  policy == argmax_a[r + gV(s'_a)] over OWN value : {agree:.3f}")
    print(f"  (ctrl)  V shuffled across successors (r kept real)      : {agree_shuf:.3f}   rolls {f(shuf)}")
    print(f"  decompose: reward-only argmax_a r_a == policy           : {agree_r:.3f}")
    print(f"  decompose: value-only  argmax_a V(s'_a) == policy       : {agree_V:.3f}")
    print(f"  value-only SHUFFLED (rolls)                             : {agree_shuf and f(shufV)}")
    drop = agree - agree_shuf
    print(f"  --> greedy-over-OWN-V: {'CONFIRMED' if (agree > ch + 0.1 and drop > 0.1 and agree_V > ch + 0.1) else 'WEAK/NOT clean'} "
          f"(real {agree:.2f} vs V-shuffled {agree_shuf:.2f}, value-only {agree_V:.2f}, chance {ch:.2f})")
    print("PLOT_LACTRL=" + repr(dict(agree=round(agree,3), agree_shuf=round(agree_shuf,3), agree_r=round(agree_r,3),
                                      agree_V=round(agree_V,3), shuf=[round(x,3) for x in shuf], shufV=[round(x,3) for x in shufV],
                                      chance=round(ch,3))))
    print("=" * 84 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=256)
    a = ap.parse_args(); main(a.ckpt, a.boards)
