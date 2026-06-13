#!/usr/bin/env python3
"""Verify gradients flow through EVERY active thinking tick under variable-depth training.

Runs the EXACT vardepth net (cleanba.config.sokoban_drc_attn_vardepth: dense softmax
attention, n_recurrent=1, K_max=repeats_per_step=6) in a *training-shaped* call -- a
time-scan over T steps with a scalar thinking-depth, exactly like the learner uses
(impala_loss.py passes minibatch.n_active_t.reshape(-1)[0]).

Three checks:

 [A/B] Depth scaling. ||grad -> recurrent CELL params|| is EXACTLY 0 at depth d=0
       (no tick runs, so the output reaches the loss only via the conv-embed + skip +
       MLP, never through the cell) and grows as d=1..K_max. Each added thinking tick
       injects gradient into the weight-tied core  =>  gradient flows through every
       active tick.

 [C]   Exactness / no leakage. The grad of the GATED model at n_active=d equals the grad
       of a NATIVELY d-tick model (repeats_per_step=d, no gating) to fp tolerance. So the
       K_max-d inactive ticks leak ZERO gradient, and the d active ticks backprop exactly
       like a true d-step recurrence would. (This is the bit-exactness the handoff claims.)

Run on the local CPU diag venv (see memory: local-jax-diag-venv):
    JAX_PLATFORMS=cpu /tmp/attn-diag/bin/python -m tests.check_vardepth_grad
"""
from __future__ import annotations

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np

T, B, BOARD, N_ACT = 3, 4, 6, 4  # tiny: time-scan T steps, batch B, BOARD*BOARD tokens
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "sokoban_drc_attn_vardepth"  # any vardepth config fn


def build(repeats: int):
    """The chosen vardepth net with repeats_per_step overridden. Param tree is INDEPENDENT of
    repeats (weight-tied cell), so params/carry built here are interchangeable across repeats."""
    import gymnasium as gym

    from cleanba import config as cfgmod

    net = getattr(cfgmod, CONFIG)().net
    net = dataclasses.replace(net, repeats_per_step=repeats)

    class FakeEnvs:
        single_observation_space = gym.spaces.Box(0, 255, (3, BOARD, BOARD), np.uint8)
        observation_space = gym.spaces.Box(0, 255, (B, 3, BOARD, BOARD), np.uint8)
        single_action_space = gym.spaces.Discrete(N_ACT)
        action_space = gym.spaces.MultiDiscrete([N_ACT] * B)

    policy, carry, params = net.init_params(FakeEnvs(), jax.random.PRNGKey(0))
    return policy, carry, params


def _cell_leaves(tree):
    out = []
    for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]:
        name = "/".join(str(getattr(x, "key", x)) for x in path)
        if "cell_list" in name:  # the recurrent AttentionCell params live under network_params/cell_list_*
            out.append(v)
    return out


def _gnorm(leaves):
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(v.astype(jnp.float32))) for v in leaves)))


def main():
    from cleanba import config as cfgmod

    net0 = getattr(cfgmod, CONFIG)().net
    K_MAX = net0.repeats_per_step
    policy, carry, params = build(K_MAX)
    print(f"# config={CONFIG}  n_recurrent(D)={net0.n_recurrent}  attn_norm={net0.recurrent.attn_norm}  "
          f"mask={net0.recurrent.use_attention_mask}  readout={net0.recurrent.readout}")

    key = jax.random.PRNGKey(1)
    obs = jax.random.randint(jax.random.fold_in(key, 1), (T, B, 3, BOARD, BOARD), 0, 256).astype(jnp.uint8)
    eps = jnp.zeros((T, B), dtype=bool).at[0].set(True)  # fresh episode at t=0
    target = jax.random.randint(jax.random.fold_in(key, 2), (T, B), 0, N_ACT)

    def loss_for(p, policy_, n_active):
        _, logits, _, _ = policy_.apply(p, carry, obs, eps, n_active, method=policy_.get_logits_and_value)
        logp = jax.nn.log_softmax(logits, axis=-1)  # (T, B, N_ACT)
        return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

    grad_fn = jax.grad(loss_for)

    print(f"=== gradient through thinking ticks  (K_max={K_MAX}, T={T}, B={B}, board={BOARD}x{BOARD}) ===")

    # ---- [A/B] cell-grad vs sampled depth d (gated model, repeats=K_MAX) ------------------------
    print("\n[A/B] ||grad -> recurrent cell params|| vs sampled thinking-depth d  (gated model):")
    norms = []
    for d in range(0, K_MAX + 1):
        g = grad_fn(params, policy, d)
        n = _gnorm(_cell_leaves(g))
        norms.append(n)
        print(f"      d={d}:  ||g_cell|| = {n:.6e}")
    zero_at_0 = norms[0] < 1e-9
    pos_for_active = all(n > 0 for n in norms[1:])
    grows = norms[-1] > norms[1]
    print(f"      -> cell-grad == 0 at d=0: {zero_at_0}   |   >0 for every d>=1: {pos_for_active}"
          f"   |   grows d=1..{K_MAX}: {grows}")

    # ---- [C] exactness: gated(n_active=d)  ==  native d-tick model (repeats=d) -------------------
    print("\n[C] gated(n_active=d) grad   vs   native d-tick (repeats=d) grad   [max |Δ| over ALL params]:")
    exact = True
    for d in range(1, K_MAX + 1):
        g_gated = grad_fn(params, policy, d)
        policy_ref = build(d)[0]
        g_ref = grad_fn(params, policy_ref, None)  # repeats=d, no gating -> a genuine d-tick recurrence
        diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in
                   zip(jax.tree_util.tree_leaves(g_gated), jax.tree_util.tree_leaves(g_ref)))
        scale = max(float(jnp.max(jnp.abs(v))) for v in jax.tree_util.tree_leaves(g_ref))
        rel = diff / (scale + 1e-12)
        ok = rel < 1e-4
        exact = exact and ok
        print(f"      d={d}:  max|Δgrad| = {diff:.3e}   (grad scale {scale:.3e}, rel {rel:.2e})  {'ok' if ok else '<-- MISMATCH'}")

    # ---- verdict --------------------------------------------------------------------------------
    print("\n==================== VERDICT ====================")
    if zero_at_0 and pos_for_active and grows and exact:
        print("  PASS:")
        print("   - cell-grad is exactly 0 at d=0 and grows with d  => gradient flows through every active tick;")
        print("   - gated(n_active=d) grad == native d-tick grad to fp tol  => no leakage from inactive ticks,")
        print("     and the d active ticks backprop exactly like a true d-step recurrence.")
    else:
        print("  CHECK the numbers above:")
        if not zero_at_0:
            print("   - cell-grad NOT ~0 at d=0 (inactive ticks may be leaking gradient).")
        if not (pos_for_active and grows):
            print("   - cell-grad does not grow with depth (ticks past the first may not be backpropping).")
        if not exact:
            print("   - gated grad != native d-tick grad (the gating is not gradient-exact).")
    print("=================================================")


if __name__ == "__main__":
    main()
