"""CPU sanity check for D1 (binding="positional"): the learned-but-task-STABLE slot binding.

Verifies, for the n100 content vs positional cores:
  * params: positional has bind_addr/bind_kpos and NO bind_q/bind_k/bind_norm (content is the reverse);
  * forward logits/value finite; loss-grad finite (trainable);
  * THE KEY PROPERTY: in positional mode the binding map bind_attn is BOARD-INDEPENDENT (identical across
    the batch -> a fixed slot<->position template, the same across tasks), whereas content-mode bind_attn
    varies board to board. (In positional mode it is also tick-independent.)

  python -m experiments.interp.check_posbind
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, numpy as np
import gymnasium as gym
from flax.traverse_util import flatten_dict
from cleanba.config import sokoban_drc_slots_d3_fixed4_n100, sokoban_drc_slots_d3_fixed4_n100_posbind
from experiments.interp.slot_interp import slot_per_tick

B, N_ACT = 8, 4


class FakeEnvs:
    single_observation_space = gym.spaces.Box(0, 255, (3, 10, 10), np.uint8)
    observation_space = gym.spaces.Box(0, 255, (B, 3, 10, 10), np.uint8)
    single_action_space = gym.spaces.Discrete(N_ACT)
    action_space = gym.spaces.MultiDiscrete([N_ACT] * B)


def bind_param_names(params):
    flat = flatten_dict(params)
    return sorted({k[-1] for k in flat if any("bind" in str(p) for p in k)})


for name, fn in [("content-n100", sokoban_drc_slots_d3_fixed4_n100),
                 ("POSBIND-n100", sokoban_drc_slots_d3_fixed4_n100_posbind)]:
    args = fn(); net = args.net; rec = net.recurrent
    print(f"\n=== {name}: binding={rec.binding} num_slots={rec.num_slots} ticks={net.repeats_per_step} D={net.n_recurrent} ===")
    policy, carry, params = net.init_params(FakeEnvs(), jax.random.PRNGKey(0))
    print(f"   binding params: {bind_param_names(params)}")

    eps = jnp.ones((1, B), dtype=bool)
    obs = jax.random.randint(jax.random.PRNGKey(1), (1, B, 3, 10, 10), 0, 256).astype(jnp.uint8)
    _, logits, value, _ = policy.apply(params, carry, obs, eps, method=policy.get_logits_and_value)
    print(f"   forward: logits={logits.shape} finite={bool(jnp.all(jnp.isfinite(logits)))} "
          f"value finite={bool(jnp.all(jnp.isfinite(value)))}")

    def loss(p):
        _, lg, _, _ = policy.apply(p, carry, obs, eps, method=policy.get_logits_and_value)
        return jnp.mean(lg ** 2)
    g = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    allfin = all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    print(f"   grad: all-finite={allfin} norm={float(jnp.sqrt(sum(float(jnp.sum(x**2)) for x in leaves))):.3e}")

    # THE KEY CHECK: is the binding map board-independent? (positional => yes; content => no)
    h, bind, route = slot_per_tick(policy, params, obs[0], net.repeats_per_step)   # bind:(K,B,nh,N,S)
    ba = bind[-1]                                                                  # (B,nh,N,S) settled
    batch_spread = float(ba.std(axis=0).max())          # 0 => identical across boards (stable template)
    tick_spread = float(np.stack([bind[k] for k in range(bind.shape[0])]).std(axis=0).max())
    print(f"   bind_attn shape={ba.shape}  batch-spread(max std over boards)={batch_spread:.2e}  "
          f"tick-spread={tick_spread:.2e}")
    if rec.binding == "positional":
        assert batch_spread < 1e-6, f"positional binding should be board-INDEPENDENT, got spread {batch_spread}"
        wmean = ba.mean(1)[0]                                                      # (N,S) head-avg (board-indep)
        diag_frac = float((wmean.argmax(1) == np.arange(wmean.shape[0])).mean())   # slot i reads position i?
        print(f"   identity warm-start: slot-i-reads-position-i fraction = {diag_frac:.3f} (expect ~1.0 at init)")
        assert diag_frac > 0.9, f"identity warm-start expected slot i -> position i, got {diag_frac}"
        print("   ==> OK: positional binding is a FIXED template warm-started to the cell=square identity.")
    else:
        assert batch_spread > 1e-4, f"content binding should vary by board, got spread {batch_spread}"
        print("   ==> OK: content binding varies board-to-board (re-indexed), as expected.")

print("\nALL OK")
