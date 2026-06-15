"""PLAN-QUALITY vs TICK (trajectory-optimization vs reactive), D=3 entmax core.

Reach is already global in ~2 ticks (perturbation onset is distance-independent), so the question is not
horizon but QUALITY: with the whole problem in view, do the thinking ticks REFINE a multi-step plan?
We decode the model's OWN executed future actions a0,a1,..,a_{H-1} (greedy rollout = its realized plan)
from the policy READOUT at each thinking tick on the INITIAL board, per horizon.

  trajectory optimization : a multi-step plan is decodable (a_t for t>0 >> chance) and its quality
                            RISES over ticks (plan refined in the loop).
  reactive / amortized    : decided by ~tick 2, flat/eroding after; far horizons not above chance.

Readout per tick = relu((top_h[k] + embed).flat @ W_dense + b)  -- exactly what the actor head reads
(IdentityNorm in this lineage). Probes are ridge-LSQ (decodability TRENDS, same split across ticks).

  python -m results.interp_planq_d3 --ckpt <cp_dir> --boards 512 --horizon 6
"""
from __future__ import annotations
import argparse, dataclasses
import numpy as np
import jax, jax.numpy as jnp
from results.interp_planning_d3 import recompute_d3, get_embed
from results.interp_search_d3 import greedy_rollout
from results.interp_plan import lin_acc


def main(cp_dir, n_boards, H):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    n_act = int(env_cfg.make().single_action_space.n)

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)
    B, _, Hh, Wd = obs0.shape; S = Hh * Wd
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    embed = np.asarray(get_embed(policy, params, jnp.asarray(obs0))).reshape(B, S, -1)        # (B,S,C)
    top_h = np.asarray(recompute_d3(cps, jnp.asarray(embed).reshape(B, Hh, Wd, -1), K)[0])     # (K,B,S,C)
    Wdense = np.asarray(params["params"]["network_params"]["dense_list_0"]["kernel"])          # (S*C,256)
    bdense = np.asarray(params["params"]["network_params"]["dense_list_0"]["bias"])
    readout = np.maximum((top_h + embed[None]).reshape(K, B, S * embed.shape[-1]) @ Wdense + bdense, 0.0)  # (K,B,256)

    pos, act = greedy_rollout(policy, params, envs, obs0, H + 2)
    y = np.full((B, H), -1)
    for b in range(B):
        for t in range(min(H, len(act[b]))):
            y[b, t] = act[b][t]

    rng = np.random.default_rng(0)
    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print(f"\n===== PLAN-QUALITY vs TICK (step={step}, boards={B}, K={K}, horizon={H}, n_act={n_act}) =====")
    print(f"  decode executed action a_t from policy readout, per thinking tick:")
    accs = np.zeros((H, K)); chance = np.zeros(H)
    for t in range(H):
        v = np.where(y[:, t] >= 0)[0]
        if len(v) < 20:
            continue
        idx = rng.permutation(len(v)); k = int(0.8 * len(v)); tr, te = idx[:k], idx[k:]
        yt = y[v, t]; chance[t] = float(np.bincount(yt[te], minlength=n_act).max() / len(te))
        accs[t] = [lin_acc(readout[kk][v], yt, tr, te, n_cls=n_act) for kk in range(K)]
        print(f"   a_{t} (chance {chance[t]:.2f}, n={len(v)}): {f(accs[t])}  peak@tick{int(np.argmax(accs[t]))+1}")
    pq = accs.mean(0)
    lift = accs - chance[:, None]
    print(f"  PLAN-QUALITY (mean acc over horizons) per tick: {f(pq)}   tick1->K {pq[0]:.2f}->{pq[-1]:.2f}")
    print(f"  mean lift-over-chance: tick1={lift[:, 0].mean():.2f}  peak={lift.max(0).mean():.2f}  tickK={lift[:, -1].mean():.2f}")
    multi = (lift[1:].max(1) > 0.05).sum()                                          # horizons t>=1 decodable above chance
    rising = pq[min(3, K - 1)] - pq[0]                                              # tick1 -> ~tick4 change
    print(f"  multi-step plan: {int(multi)}/{H-1} future horizons (t>=1) decodable >chance+0.05")
    print(f"  refinement tick1->~tick4: {rising:+.2f}  ({'REFINES (trajectory opt)' if rising > 0.03 else 'flat/early-decided (reactive)'})")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=512); ap.add_argument("--horizon", type=int, default=6)
    a = ap.parse_args(); main(a.ckpt, a.boards, a.horizon)
