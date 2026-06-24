"""CPU sanity check for the positional-binding cell-count sweep (n100 / n50 / n20).

For each posbind config verifies:
  * binding params are bind_logits (the coarse-grid template), grad finite, forward finite;
  * THE KEY PROPERTY: the binding map is BOARD-INDEPENDENT (a fixed template, same across tasks);
  * the warm-started partition TILES the board -- prints the (H,W) grid of which cell owns each square,
    plus region-size stats; n100 must be the exact identity (slot i <- square i = dense attention).

  python -m experiments.interp.check_posbind
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, numpy as np
import gymnasium as gym
from flax.traverse_util import flatten_dict
from cleanba.config import (sokoban_drc_slots_d3_fixed4_n100,        # content baseline (re-indexes)
                            sokoban_drc_slots_d3_fixed4_n100_posbind,
                            sokoban_drc_slots_d3_fixed4_n50_posbind,
                            sokoban_drc_slots_d3_fixed4_n20_posbind)
from experiments.interp.slot_interp import slot_per_tick

B, N_ACT, H, W = 8, 4, 10, 10


class FakeEnvs:
    single_observation_space = gym.spaces.Box(0, 255, (3, H, W), np.uint8)
    observation_space = gym.spaces.Box(0, 255, (B, 3, H, W), np.uint8)
    single_action_space = gym.spaces.Discrete(N_ACT)
    action_space = gym.spaces.MultiDiscrete([N_ACT] * B)


def bind_param_names(params):
    return sorted({k[-1] for k in flatten_dict(params) if any("bind" in str(p) for p in k)})


def run(name, fn, content):
    args = fn(); net = args.net; rec = net.recurrent; N = rec.num_slots
    print(f"\n=== {name}: binding={rec.binding} num_slots={N} ticks={net.repeats_per_step} ===")
    policy, carry, params = net.init_params(FakeEnvs(), jax.random.PRNGKey(0))
    print(f"   binding params: {bind_param_names(params)}")
    eps = jnp.ones((1, B), dtype=bool)
    obs = jax.random.randint(jax.random.PRNGKey(1), (1, B, 3, H, W), 0, 256).astype(jnp.uint8)
    _, logits, value, _ = policy.apply(params, carry, obs, eps, method=policy.get_logits_and_value)

    def loss(p):
        _, lg, _, _ = policy.apply(p, carry, obs, eps, method=policy.get_logits_and_value)
        return jnp.mean(lg ** 2)
    g = jax.tree_util.tree_leaves(jax.grad(loss)(params))
    print(f"   forward finite={bool(jnp.all(jnp.isfinite(logits)) & jnp.all(jnp.isfinite(value)))}  "
          f"grad finite={all(bool(jnp.all(jnp.isfinite(x))) for x in g)}")

    _h, bind, _r = slot_per_tick(policy, params, obs[0], net.repeats_per_step)   # bind:(K,B,nh,N,S)
    ba = bind[-1]                                                                # (B,nh,N,S)
    spread = float(ba.std(axis=0).max())
    if content:
        print(f"   bind_attn batch-spread={spread:.2e}  (content => varies board-to-board, OK)")
        return
    assert spread < 1e-6, f"positional binding must be board-independent, got {spread}"
    wmean = ba.mean(1)[0]                                                        # (N,S) head-avg (board-indep)
    owner = wmean.argmax(0)                                                      # (S,) cell owning each square
    sizes = np.bincount(owner, minlength=N)
    print(f"   board-independent (spread={spread:.1e}); cells used={int((sizes>0).sum())}/{N}; "
          f"region size min/mean/max={sizes[sizes>0].min()}/{sizes.sum()/max(1,(sizes>0).sum()):.1f}/{sizes.max()}")
    if N == H * W:
        diag = float((owner == np.arange(H * W)).mean())
        print(f"   identity check (slot i <- square i): {diag:.3f}  {'OK (= dense attention)' if diag>0.99 else 'FAIL'}")
    print("   partition (cell id owning each board square):")
    grid = owner.reshape(H, W)
    for r in range(H):
        print("     " + " ".join(f"{int(grid[r, c]):3d}" for c in range(W)))


run("content-n100", sokoban_drc_slots_d3_fixed4_n100, content=True)
run("POSBIND-n100", sokoban_drc_slots_d3_fixed4_n100_posbind, content=False)
run("POSBIND-n50", sokoban_drc_slots_d3_fixed4_n50_posbind, content=False)
run("POSBIND-n20", sokoban_drc_slots_d3_fixed4_n20_posbind, content=False)
print("\nALL OK")
