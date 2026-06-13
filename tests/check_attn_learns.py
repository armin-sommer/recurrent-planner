#!/usr/bin/env python3
"""Fast architecture sanity test for the attention recurrent core (attn_lstm.py).

The DRC(3,3) ConvLSTM core trains fine on this env; the attention core sits at
near-uniform entropy and ~0 success. This isolates whether the *architecture*
can learn at all -- in seconds, with NO envpool and NO multi-M-step run.

Three checks on the EXACT net from cleanba.config:
  A. Input sensitivity  -- do the logits change when the observation changes?
                           (a constant policy => the core ignores its input.)
  B. Overfit one batch  -- can it memorize a fixed (obs -> random action) map?
                           A healthy net drives cross-entropy from ~ln(4)=1.386
                           toward ~0. If it stays pinned near 1.386, the core
                           cannot condition on state => a real architecture bug.
  C. Gradient flow      -- does every parameter subtree receive nonzero, finite
                           gradient? A dead (zero-grad) subtree is the bug site.

Usage (on the box, venv active; also runs anywhere jax/flax/optax are installed):
    python -m tests.check_attn_learns            # maxplus readout (default config)
    python -m tests.check_attn_learns softmax    # softmax readout variant
    python -m tests.check_attn_learns drc         # ConvLSTM control (should pass)
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

B = 32          # batch of distinct levels to memorize
N_ACT = 4       # PushUp/Down/Left/Right
STEPS = 600
LR = 3e-3


def build_policy(variant: str):
    import gymnasium as gym
    from cleanba.config import (
        sokoban_drc_attn_3_3,
        sokoban_drc_attn_3_3_softmax,
        sokoban_drc_attn_3_3_dir,
        sokoban_drc_attn_3_3_dir_softmax,
        sokoban_drc_3_3,
        sokoban_drc_slots_softmax,
        sokoban_drc_slots_maxplus,
    )

    spec = {
        "maxplus": sokoban_drc_attn_3_3,
        "softmax": sokoban_drc_attn_3_3_softmax,
        "dir": sokoban_drc_attn_3_3_dir,                  # + per-offset value routing (maxplus)
        "dir_softmax": sokoban_drc_attn_3_3_dir_softmax,  # + per-offset value routing (softmax)
        "drc": sokoban_drc_3_3,
        "slots": sokoban_drc_slots_softmax,               # learnable-slot core, softmax routing (headline)
        "slots_maxplus": sokoban_drc_slots_maxplus,       # learnable-slot core, max-plus routing
    }[variant]().net

    class FakeEnvs:
        single_observation_space = gym.spaces.Box(0, 255, (3, 10, 10), np.uint8)
        observation_space = gym.spaces.Box(0, 255, (B, 3, 10, 10), np.uint8)
        single_action_space = gym.spaces.Discrete(N_ACT)
        action_space = gym.spaces.MultiDiscrete([N_ACT] * B)

    policy, carry, params = spec.init_params(FakeEnvs(), jax.random.PRNGKey(0))
    return policy, carry, params, type(spec).__name__


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "maxplus"
    policy, carry, params, net_name = build_policy(variant)
    print(f"=== attn-core sanity: variant={variant!r}  net={net_name} ===")

    eps = jnp.ones((1, B), dtype=bool)  # T=1; fresh episode so carry resets

    def logits_for(p, obs):
        _, logits, _, _ = policy.apply(p, carry, obs, eps, method=policy.get_logits_and_value)
        return logits[0]  # (B, N_ACT)

    # ---- A. input sensitivity --------------------------------------------------
    k = jax.random.PRNGKey(1)
    o1 = jax.random.randint(jax.random.fold_in(k, 1), (1, B, 3, 10, 10), 0, 256).astype(jnp.uint8)
    o2 = jax.random.randint(jax.random.fold_in(k, 2), (1, B, 3, 10, 10), 0, 256).astype(jnp.uint8)
    l1, l2 = logits_for(params, o1), logits_for(params, o2)
    finite = bool(jnp.all(jnp.isfinite(l1)))
    cross_input_spread = float(jnp.mean(jnp.abs(l1 - l2)))
    within_batch_std = float(jnp.std(l1))
    print("\n[A] input sensitivity")
    print(f"    logits finite: {finite}")
    print(f"    mean|logits(obs1)-logits(obs2)| = {cross_input_spread:.4e}  (≈0 => ignores input)")
    print(f"    within-batch logit std         = {within_batch_std:.4e}")

    # ---- B. overfit one fixed batch -------------------------------------------
    obs = jax.random.randint(jax.random.fold_in(k, 3), (1, B, 3, 10, 10), 0, 256).astype(jnp.uint8)
    target = jax.random.randint(jax.random.fold_in(k, 4), (B,), 0, N_ACT)

    def loss_fn(p):
        logp = jax.nn.log_softmax(logits_for(p, obs), axis=-1)
        return -jnp.mean(logp[jnp.arange(B), target])

    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, l

    p = params
    l0 = float(loss_fn(p))
    print("\n[B] overfit one fixed batch (target ~ln(4)=1.386 -> 0 if it can learn)")
    print(f"    step    0: loss = {l0:.4f}")
    last = l0
    for i in range(1, STEPS + 1):
        p, opt_state, last = step(p, opt_state)
        if i % 100 == 0:
            print(f"    step {i:4d}: loss = {float(last):.4f}")
    last = float(last)

    # ---- C. gradient flow per subtree -----------------------------------------
    _, g = jax.value_and_grad(loss_fn)(params)
    flat = jax.tree_util.tree_flatten_with_path(g)[0]
    print("\n[C] gradient norms per parameter (at init)")
    dead = []
    for path, leaf in flat:
        name = "/".join(str(getattr(x, "key", x)) for x in path)
        gn = float(jnp.linalg.norm(leaf.astype(jnp.float32)))
        is_critic = "critic_params" in name  # expected-dead: this test's loss is policy-only
        tag = "  <-- DEAD" if gn < 1e-9 else ("  <-- NaN/Inf" if not np.isfinite(gn) else "")
        if tag and not is_critic:
            dead.append(name)
        if tag and is_critic:
            tag += " (expected: policy-only loss never touches the value head)"
        if tag or gn < 1e-5:
            print(f"    {gn:.3e}  {name}{tag}")
    if not dead:
        print("    (all subtrees have finite, nonzero gradient)")

    # ---- verdict ---------------------------------------------------------------
    print("\n==================== VERDICT ====================")
    print(f"  init loss {l0:.3f} -> final loss {last:.3f}")
    learned = last < 0.3
    sensitive = cross_input_spread > 1e-3
    if learned and sensitive and not dead:
        print("  PASS: the core CAN overfit a batch and is input-sensitive.")
        print("  => architecture is capable of learning. The training failure is")
        print("     a RECIPE/optimization issue (try the drc33_59 recipe), not a code bug.")
    else:
        print("  FAIL: the core could NOT learn a fixed batch.")
        if not sensitive:
            print("   - logits barely change with input -> the core ignores its observation.")
        if dead:
            print(f"   - dead/NaN gradient subtrees: {dead}")
        if not learned:
            print("   - loss did not drop -> gradient cannot shape the policy. Real bug in attn_lstm.py.")
    print("=================================================")


if __name__ == "__main__":
    main()
