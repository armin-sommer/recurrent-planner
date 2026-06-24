"""CPU smoke test for the pool-and-inject core: init_params + forward + grad, for BOTH the Sokoban
(3x10x10) and MiniWorld (3x60x80) obs shapes. Run before any pod launch:

    JAX_PLATFORMS=cpu python -m experiments.test_poolinject_smoke
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
import gymnasium as gym
from cleanba.config import sokoban_poolinject_d3_fixed4_4gpu_mb4, miniworld_poolinject_d3_fixed4


def fake_envs(C, H, W, nact, B=8):
    class FakeEnvs:
        single_observation_space = gym.spaces.Box(0, 255, (C, H, W), np.uint8)
        observation_space = gym.spaces.Box(0, 255, (B, C, H, W), np.uint8)
        single_action_space = gym.spaces.Discrete(nact)
        action_space = gym.spaces.MultiDiscrete([nact] * B)
    return FakeEnvs(), B


cases = [
    ("sokoban-poolinject", sokoban_poolinject_d3_fixed4_4gpu_mb4, 3, 10, 10, 4),
    ("miniworld-poolinject", miniworld_poolinject_d3_fixed4, 3, 60, 80, 3),
]
for name, fn, C, H, W, nact in cases:
    args = fn(); net = args.net; rec = net.recurrent
    print(f"\n=== {name}: D={net.n_recurrent} ticks={net.repeats_per_step} cells={rec.num_cells} "
          f"routing={rec.routing_norm} pool={rec.pool} skip_final={net.skip_final} "
          f"embed_layers={len(net.embed)} vtd={args.variable_thinking_depth}")
    envs, B = fake_envs(C, H, W, nact)
    policy, carry, params = net.init_params(envs, jax.random.PRNGKey(0))
    eps = jnp.ones((1, B), dtype=bool)
    obs = jax.random.randint(jax.random.PRNGKey(1), (1, B, C, H, W), 0, 256).astype(jnp.uint8)
    _, logits, value, _ = policy.apply(params, carry, obs, eps, method=policy.get_logits_and_value)
    nparams = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    print(f"   forward: logits={logits.shape} finite={bool(jnp.all(jnp.isfinite(logits)))} "
          f"value={value.shape} finite={bool(jnp.all(jnp.isfinite(value)))}  params={nparams/1e6:.2f}M")

    def loss(p):
        _, lg, _, _ = policy.apply(p, carry, obs, eps, method=policy.get_logits_and_value)
        return jnp.mean(lg ** 2)
    g = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    allfin = all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    gnorm = float(jnp.sqrt(sum(float(jnp.sum(x ** 2)) for x in leaves)))
    assert bool(jnp.all(jnp.isfinite(logits))) and allfin, f"{name}: non-finite forward/grad"
    print(f"   grad: all-finite={allfin} norm={gnorm:.3e} param_leaves={len(leaves)}")
print("\nALL OK")
