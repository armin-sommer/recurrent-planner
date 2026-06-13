#!/usr/bin/env python3
"""Validate the sparse attention normalizers (cleanba/entmax.py) in isolation.

Checks, for sparsemax and 1.5-entmax (and softmax as a control):
  [F] Forward:  output >= 0, sums to 1 over the axis, and the sparse ones place EXACT
                zeros (true hard support), unlike softmax which is strictly positive.
  [G] Gradient: the custom_vjp Jacobian matches central finite differences (float64),
                including the alpha=1.5 internal /2 factor -- this is the decisive check
                that the backward (and any factor-of-2) is correct before it touches a run.
  [M] Masking:  a -1e9 logit (the out-of-grid / masked-edge sentinel) maps to exactly 0.

Run on the local CPU diag venv:
    JAX_PLATFORMS=cpu /tmp/attn-diag/bin/python -m tests.check_entmax
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # tight finite-difference comparison

import jax.numpy as jnp  # noqa: E402

from cleanba.entmax import entmax15, normalize, sparsemax  # noqa: E402


def fd_grad(f, z, eps=1e-6):
    """Central finite-difference gradient of scalar f at z."""
    g = jnp.zeros_like(z).ravel()
    zf = z.ravel()
    for i in range(zf.size):
        e = jnp.zeros_like(zf).at[i].set(eps)
        g = g.at[i].set((f((zf + e).reshape(z.shape)) - f((zf - e).reshape(z.shape))) / (2 * eps))
    return g.reshape(z.shape)


def main():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (3, 7), dtype=jnp.float64) * 2.0  # a few rows of 7 logits
    c = jax.random.normal(jax.random.fold_in(key, 1), (3, 7), dtype=jnp.float64)  # loss weights

    print("=== sparse attention normalizers (sparsemax, entmax15) ===")
    print("\n[F] forward properties (axis=-1):")
    for name, fn in [("softmax", lambda x: normalize("softmax", x, -1)),
                     ("sparsemax", sparsemax), ("entmax15", entmax15)]:
        p = fn(z)
        sums = jnp.sum(p, axis=-1)
        nnz = jnp.sum(p > 0, axis=-1)
        print(f"    {name:9s}: min={float(p.min()):+.4f}  sum(row)~={float(sums.mean()):.6f}  "
              f"nonzeros/row={[int(x) for x in nnz]}  (row len 7)")

    print("\n[G] gradient vs central finite differences (float64), loss = sum(p * c):")
    ok_all = True
    for name, fn in [("sparsemax", sparsemax), ("entmax15", entmax15)]:
        loss = lambda x: jnp.sum(fn(x) * c)  # noqa: E731
        g_auto = jax.grad(loss)(z)
        g_fd = fd_grad(loss, z)
        err = float(jnp.max(jnp.abs(g_auto - g_fd)))
        scale = float(jnp.max(jnp.abs(g_fd))) + 1e-12
        ok = err < 1e-5
        ok_all = ok_all and ok
        print(f"    {name:9s}: max|Δgrad| = {err:.3e}  (grad scale {scale:.3e})  {'ok' if ok else '<-- MISMATCH'}")

    print("\n[M] masking: logit -1e9 -> weight 0 (exact)?")
    zmask = z.at[:, 0].set(-1e9)
    for name, fn in [("sparsemax", sparsemax), ("entmax15", entmax15)]:
        p = fn(zmask)
        col0 = float(jnp.max(jnp.abs(p[:, 0])))
        print(f"    {name:9s}: max weight on masked col = {col0:.3e}  sum(row)~={float(jnp.sum(p,axis=-1).mean()):.6f}")

    print("\n==================== VERDICT ====================")
    print("  PASS: gradients match finite differences." if ok_all else "  FAIL: gradient mismatch (see above).")
    print("=================================================")


if __name__ == "__main__":
    main()
