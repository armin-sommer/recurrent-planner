"""Localized-readout test (dense attention core): is the action effectively read from the AGENT's cell?

Assumption ass:readout claims pi(.|s_t) = rho(h(sigma(s_t))) -- the action depends only on the current
state's cell. Architecturally FALSE for both cores: the head flattens ALL S cells
(mlp = relu((th+embr).reshape(B, S*C) @ Wd + bd); logits = mlp @ Wa + ba). We test whether it holds
EFFECTIVELY on the dense core, two ways (chance action = 1/nact = 0.25):

  DECODE  : predict the model's own settled action a0 from a linear probe of
              (i) the AGENT cell's hidden (C dims), (ii) a RANDOM non-agent cell (C dims), (iii) the full
              field (S*C). agent >> random (same dim) => the action lives in the agent cell.
  CAUSAL  : mean-ablate the readout field and re-read the head (mean = per-cell dataset mean, the standard
            "cell doesn't matter" null):
              keep-agent-only : all OTHER cells set to mean, agent cell kept  -> a0 unchanged? (SUFFICIENCY)
              ablate-agent    : agent cell set to mean, others kept           -> a0 flips?     (NECESSITY)
              keep-random-only: one random non-agent cell kept, rest mean     -> baseline for keep-agent
              all-mean        : whole field = mean                            -> sanity floor
  agent SUFFICIENT (keep-agent-only agreement ~ full, >> keep-random) AND NECESSARY (ablate-agent low)
  => the readout is effectively localized to the agent's cell.

  python -m experiments.interp.localized_readout --ckpt <cp_dir> --boards 512
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import AGENT


def _decode(X, y, nact, rng, tr_frac=0.8):
    """Linear multiclass least-squares probe; return test accuracy (chance 1/nact)."""
    n = len(y); idx = rng.permutation(n); k = int(tr_frac * n); tr, te = idx[:k], idx[k:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    Y = np.eye(nact)[y[tr]]
    W = np.linalg.solve(Xtr.T @ Xtr + 10.0 * np.eye(X.shape[1]), Xtr.T @ Y)
    return float(((Xte @ W).argmax(1) == y[te]).mean())


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    NP = params["params"]["network_params"]
    Wd = np.asarray(NP["dense_list_0"]["kernel"]); bd = np.asarray(NP["dense_list_0"]["bias"])
    Wa = np.asarray(params["params"]["actor_params"]["Output"]["kernel"]); ba = np.asarray(params["params"]["actor_params"]["Output"]["bias"])
    cps = [NP[f"cell_list_{i}"] for i in range(D)]

    obs = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs.shape; S = H * W
    tiles = decode_tiles(obs)
    emb = np.asarray(get_embed(policy, params, jnp.asarray(obs))); C = emb.shape[-1]; embr = emb.reshape(B, S, C)
    th = np.asarray(recompute_d3(cps, jnp.asarray(emb), K)[0])                 # (K,B,S,C)
    field = th[-1] + embr                                                     # (B,S,C) settled readout field
    nact = Wa.shape[1]

    def act(f):                                                               # model action from a field (B,S,C)
        mlp = np.maximum(f.reshape(f.shape[0], S * C) @ Wd + bd, 0.0)
        return (mlp @ Wa + ba).argmax(1)

    a0 = act(field)                                                           # (B,) the model's decision
    agent = np.array([int(np.where(tiles[b] == AGENT)[0][0]) if (tiles[b] == AGENT).any() else -1 for b in range(B)])
    okB = np.where(agent >= 0)[0]
    rng = np.random.default_rng(0)
    randcell = np.array([int(rng.choice(np.delete(np.arange(S), agent[b]))) if agent[b] >= 0 else 0 for b in range(B)])

    # ---------------- DECODE: where does the action live? ----------------
    Xa = field[np.arange(B), agent]                                           # (B,C) agent cell
    Xr = field[np.arange(B), randcell]                                        # (B,C) random non-agent cell
    Xf = field.reshape(B, S * C)                                              # (B,S*C) full field
    dec_agent = _decode(Xa[okB], a0[okB], nact, np.random.default_rng(1))
    dec_rand = _decode(Xr[okB], a0[okB], nact, np.random.default_rng(1))
    dec_full = _decode(Xf[okB], a0[okB], nact, np.random.default_rng(1))

    # ---------------- CAUSAL: mean-ablation of the readout field ----------------
    mu = field.mean(0)                                                        # (S,C) per-cell dataset mean (the null)
    def agree(f): return float((act(f)[okB] == a0[okB]).mean())

    f_ka = np.broadcast_to(mu, (B, S, C)).copy()                              # keep-agent-only
    f_kr = np.broadcast_to(mu, (B, S, C)).copy()                              # keep-random-only
    f_aa = field.copy()                                                       # ablate-agent
    f_ar = field.copy()                                                       # ablate-random (necessity control)
    for b in okB:
        f_ka[b, agent[b]] = field[b, agent[b]]
        f_kr[b, randcell[b]] = field[b, randcell[b]]
        f_aa[b, agent[b]] = mu[agent[b]]
        f_ar[b, randcell[b]] = mu[randcell[b]]
    ag_keepagent = agree(f_ka); ag_keeprand = agree(f_kr)
    ag_ablagent = agree(f_aa); ag_ablrand = agree(f_ar); ag_allmean = agree(np.broadcast_to(mu, (B, S, C)))

    print(f"\n===== LOCALIZED-READOUT TEST (dense attn, step={step}, boards={len(okB)}, K={K}, nact={nact}, chance={1/nact:.2f}) =====")
    print(f"  -- DECODE model action a0 from (same C={C} dims for agent vs random) --")
    print(f"     agent-cell : {dec_agent:.3f}    random-cell : {dec_rand:.3f}    full-field({S*C}d): {dec_full:.3f}   (chance {1/nact:.2f})")
    print(f"  -- CAUSAL mean-ablation: P(action unchanged vs full field) --")
    print(f"     keep-AGENT-only  : {ag_keepagent:.3f}   (sufficiency; vs keep-random {ag_keeprand:.3f}, all-mean {ag_allmean:.3f})")
    print(f"     ablate-AGENT     : {ag_ablagent:.3f}   vs ablate-RANDOM {ag_ablrand:.3f}   (necessity control; agent<<random => agent needed)")
    suff = ag_keepagent > 0.6 and ag_keepagent > ag_keeprand + 0.15
    nec = ag_ablagent < ag_ablrand - 0.15
    verd = ("LOCALIZED: agent cell is sufficient (keep-agent >> keep-random) and necessary (ablate-agent drops)"
            if (suff and nec) else
            "PARTIAL: agent cell carries the action but the readout also uses other cells -- inspect")
    print(f"  --> {verd}")
    print("PLOT_LOCALREAD=" + repr(dict(dec_agent=round(dec_agent, 3), dec_rand=round(dec_rand, 3), dec_full=round(dec_full, 3),
          keep_agent=round(ag_keepagent, 3), keep_rand=round(ag_keeprand, 3), ablate_agent=round(ag_ablagent, 3),
          ablate_rand=round(ag_ablrand, 3), all_mean=round(ag_allmean, 3), chance=round(1 / nact, 3))))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=512)
    a = ap.parse_args(); main(a.ckpt, a.boards)
