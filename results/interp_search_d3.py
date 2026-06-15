"""Behavior-grounded mech-interp for the D=3 entmax core: object binding (Q1) + search signature (Q3).

Q3 reframed: don't assume value iteration. Probe for ANY decision statistic that informs the next real
action, grounded in the model's OWN greedy rollout (its realized box-push plan), and test whether that
statistic FORMS over thinking ticks and PROPAGATES outward along the plan. Specifically, on the INITIAL
board we recompute the K thinking ticks (top-layer h per tick) and, using the agent's realized trajectory
from that board as ground truth, probe per tick:
  * next-real-action a0  : decode the actual first action from the AGENT-cell latent -> does the decision
                           crystallize over ticks? (reactive => already decided at tick0; planning => rises)
  * plan/policy field    : decode the move-direction the agent takes at each visited square (4-class) ->
                           does the policy field sharpen over ticks? (policy-iteration signature)
  * on-path membership   : is square s on the realized path? (balanced) -> is the plan "drawn" over ticks?
  * reach-time field     : steps until the agent arrives at s (forward search-distance) -> ridge R^2, AND
                           PROPAGATION: does the per-tick representation-change move to higher reach-time
                           over ticks (a frontier extending the plan outward)? = the search signature.

Q1 per-object: balanced-accuracy probe h(s) -> {is wall / box / target / agent}, per tick (does the latent
bind the *task objects*, not just floor).

Reuses the validated recompute_d3 (faithful to the model, see interp_planning_d3 --self-test) and the
probe helpers from interp_plan. No solver / no navigation proxy: ground truth is the model's own behavior.

  python -m results.interp_search_d3 --ckpt <cp_dir> --boards 256 --steps 30
"""
from __future__ import annotations

import argparse
import dataclasses
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from results.interp_planning_d3 import recompute_d3, get_embed
from results.interp_slots import decode_tiles
from results.interp_plan import ridge_r2, lin_acc, _stz

WALL, FLOOR, BOX, TARGET, AGENT = 0, 1, 2, 3, 4


def bin_balacc(X, y, tr, te, lam=10.0):
    """Balanced accuracy of a 2-class ridge-LSQ probe (handles rare positives)."""
    Xtr, Xte = _stz(X[tr], X[te])
    Y = np.eye(2)[y[tr].astype(int)]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    pred = (Xte @ W).argmax(1); yt = y[te].astype(int)
    rp = float((pred[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0
    rn = float((pred[yt == 0] == 0).mean()) if (yt == 0).any() else 0.0
    return 0.5 * (rp + rn)


def greedy_rollout(policy, params, envs, obs0, L):
    """Greedy (temp=0, full-depth) rollout from obs0. Returns per-board agent-square sequence + actions,
    frozen at episode end. obs0: (B,3,H,W) numpy."""
    get_action = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    B = obs0.shape[0]
    carry = policy.apply(params, jax.random.PRNGKey(0), obs0.shape, method=policy.initialize_carry)
    obs = jnp.asarray(np.asarray(obs0)); eps = jnp.ones((B,), bool); key = jax.random.PRNGKey(0)
    alive = np.ones(B, bool)
    pos = [[] for _ in range(B)]; act = [[] for _ in range(B)]
    for _ in range(L):
        carry, a, _, key = get_action(params, carry, obs, eps, key, temperature=0.0, n_active=None)
        a_np = np.asarray(a); tiles = decode_tiles(np.asarray(obs))
        for b in range(B):
            if alive[b]:
                ag = np.where(tiles[b] == AGENT)[0]
                pos[b].append(int(ag[0]) if len(ag) else -1); act[b].append(int(a_np[b]))
        envs.step_async(a_np); obs_np, _, term, trunc, _ = envs.step_wait()
        done = np.asarray(term) | np.asarray(trunc)
        alive = alive & ~done
        obs = jnp.asarray(obs_np); eps = jnp.asarray(done)
    return pos, act


def plan_labels(pos, S, W=10):
    """From agent-square sequences -> per-square on_path, reach_t (first-visit step), move_dir (4-class)."""
    B = len(pos)
    on = np.zeros((B, S)); reach = np.full((B, S), -1); mdir = np.full((B, S), -1)
    agent0 = np.array([p[0] if p else -1 for p in pos])
    delta = {-W: 0, W: 1, -1: 2, 1: 3}                                  # up,down,left,right
    for b in range(B):
        seq = pos[b]
        for i, s in enumerate(seq):
            if s < 0:
                continue
            if reach[b, s] < 0:
                reach[b, s] = i; on[b, s] = 1
                if i + 1 < len(seq) and seq[i + 1] >= 0:
                    mdir[b, s] = delta.get(seq[i + 1] - s, -1)
    return on, reach, mdir, agent0


def _split(n, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(n); k = int(0.8 * n)
    return idx[:k], idx[k:]


def main(cp_dir, n_boards, L):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    n_act = int(env_cfg.make().single_action_space.n)

    envs = env_cfg.make(); obs0, _ = envs.reset(); obs0 = np.asarray(obs0)
    B = obs0.shape[0]
    embed = get_embed(policy, params, jnp.asarray(obs0)); _, H, Wd, C = embed.shape; S = H * Wd
    cell_ps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    top_h, _ = recompute_d3(cell_ps, embed, K)                          # (K,B,S,C) per-tick latents
    top_h = np.asarray(top_h)
    tiles = decode_tiles(obs0)                                          # (B,S)

    pos, act = greedy_rollout(policy, params, envs, obs0, L)
    on, reach, mdir, agent0 = plan_labels(pos, S, Wd)
    a0 = np.array([act[b][0] if act[b] else -1 for b in range(B)])
    solved = float(np.mean([reach[b].max() >= 0 and (reach[b] >= 0).sum() < L for b in range(B)]))
    pathlen = float(np.mean([(reach[b] >= 0).sum() for b in range(B)]))

    f = lambda xs: "[" + " ".join("%.2f" % x for x in xs) + "]"
    print(f"\n===== D=3 SEARCH/BINDING MECH-INTERP (step={step}, boards={B}, K={K}, D={D}, rollout L={L}) =====")
    print(f"  rollout: mean realized path length = {pathlen:.1f} squares | n_actions={n_act}")

    # ---- Q1 per-object binding (balanced acc, per tick) ----
    print("  -- Q1 BINDING: balanced acc decode h(s) -> is-OBJECT (0.5=chance) --")
    flat = top_h.reshape(K, B * S, C)
    for name, cls in [("wall", WALL), ("box", BOX), ("target", TARGET), ("agent", AGENT)]:
        y = (tiles.reshape(B * S) == cls).astype(int)
        base = float(y.mean())
        tr, te = _split(B * S)
        accs = [bin_balacc(flat[k], y, tr, te) for k in range(K)]
        print(f"     {name:<7} (base rate {base:.3f}): {f(accs)}  tick0->K {accs[0]:.2f}->{accs[-1]:.2f}")

    # ---- Q3 decision statistics (per tick) ----
    print("  -- Q3 SEARCH: decision statistics on the model's OWN plan (per tick) --")
    # (a) next real action a0, decoded from the AGENT-cell latent
    vb = np.where((agent0 >= 0) & (a0 >= 0))[0]
    Xa = np.stack([top_h[:, b, agent0[b], :] for b in vb], axis=1)       # (K, nb, C)
    ya = a0[vb]; tr, te = _split(len(vb))
    acc_a0 = [lin_acc(Xa[k], ya, tr, te, n_cls=n_act) for k in range(K)]
    print(f"     next-action a0 @agent-cell (ch={1.0/n_act:.2f}): {f(acc_a0)}  tick0->K {acc_a0[0]:.2f}->{acc_a0[-1]:.2f}")
    # (b) move-direction (policy field) over visited squares
    pij = [(b, s) for b in range(B) for s in range(S) if mdir[b, s] >= 0]
    if pij:
        Xd = np.stack([np.stack([top_h[k, b, s] for b, s in pij]) for k in range(K)])
        yd = np.array([mdir[b, s] for b, s in pij]); tr, te = _split(len(pij))
        acc_dir = [lin_acc(Xd[k], yd, tr, te, n_cls=4) for k in range(K)]
        print(f"     plan move-dir field (ch=0.25):  {f(acc_dir)}  tick0->K {acc_dir[0]:.2f}->{acc_dir[-1]:.2f}")
    # (c) on-path membership (balanced)
    nij = [(b, s) for b in range(B) for s in range(S) if tiles[b, s] != WALL]
    Xp = np.stack([np.stack([top_h[k, b, s] for b, s in nij]) for k in range(K)])
    yp = np.array([on[b, s] for b, s in nij]).astype(int); tr, te = _split(len(nij))
    acc_on = [bin_balacc(Xp[k], yp, tr, te) for k in range(K)]
    print(f"     on-path membership (bal,0.5):   {f(acc_on)}  tick0->K {acc_on[0]:.2f}->{acc_on[-1]:.2f}")
    # (d) reach-time field + PROPAGATION (does change move outward along the plan?)
    rij = [(b, s) for b in range(B) for s in range(S) if reach[b, s] >= 0]
    Xr = np.stack([np.stack([top_h[k, b, s] for b, s in rij]) for k in range(K)])
    yr = np.array([reach[b, s] for b, s in rij], float); tr, te = _split(len(rij))
    r2_reach = [ridge_r2(Xr[k], yr, tr, te) for k in range(K)]
    print(f"     reach-time field R^2:           {f(r2_reach)}  tick0->K {r2_reach[0]:.2f}->{r2_reach[-1]:.2f}")
    chg = []
    for k in range(1, K):
        delta = np.linalg.norm(top_h[k] - top_h[k - 1], axis=-1)        # (B,S)
        num = den = 0.0
        for b in range(B):
            m = reach[b] >= 0; d_ = delta[b] * m
            den += d_.sum(); num += (d_ * np.where(m, reach[b], 0)).sum()
        chg.append(num / (den + 1e-9))
    trend = "OUTWARD (frontier extends the plan)" if (len(chg) > 1 and chg[-1] > chg[0] + 0.3) else "no outward trend"
    print(f"     change-front reach-time/tick:   {f(chg)}  -> {trend}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    main(a.ckpt, a.boards, a.steps)
