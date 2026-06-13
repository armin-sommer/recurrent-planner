"""Sparse attention normalizers: sparsemax (alpha=2) and 1.5-entmax (alpha=1.5).

Drop-in replacements for ``jax.nn.softmax`` along one axis that produce EXACTLY sparse
distributions (true zeros): the model can place hard zeros on irrelevant keys, so a dense
attention layer learns its OWN sparse support -- a learned neighbour graph -- with no
auxiliary loss, no scan plumbing, and no coefficient to tune. This is the architectural
sparsity used by the planning core (cleanba/attn_lstm.py, AttentionCellConfig.attn_norm).

Refs: Martins & Astudillo 2016 (sparsemax); Peters, Niculae & Martins 2019 (alpha-entmax).
Both outputs are nonnegative and sum to 1 over ``axis``. We supply the closed-form Jacobian
via ``custom_vjp`` (the standard entmax backward: dX_i = s_i (dY_i - <s, dY> / <s, 1>) with
s_i = p_i^{2-alpha}), so there is no backprop through the internal sort/threshold. The
forward is shift-invariant; we subtract the per-row max for numerical stability only.
"""
from functools import partial

import jax
import jax.numpy as jnp


def _rho(K: int, ndim: int, axis: int, dtype) -> jax.Array:
    """[1,...,K,...,1] ramp broadcastable against an `ndim`-array along `axis`."""
    shape = [1] * ndim
    shape[axis] = K
    return jnp.arange(1, K + 1, dtype=dtype).reshape(shape)


def _sparse_bwd(power_is_zero: bool, axis: int, p: jax.Array, dY: jax.Array):
    """Shared entmax backward. s_i = p_i^{2-alpha}: indicator (sparsemax) or sqrt(p) (entmax1.5)."""
    s = (p > 0).astype(p.dtype) if power_is_zero else jnp.sqrt(p)
    v = dY * s
    q = jnp.sum(v, axis=axis, keepdims=True) / jnp.sum(s, axis=axis, keepdims=True)
    return (v - q * s,)


# ---------------------------------------------------------------- sparsemax (alpha = 2) ----
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sparsemax(z: jax.Array, axis: int = -1) -> jax.Array:
    z = z - jnp.max(z, axis=axis, keepdims=True)
    z_srt = jnp.flip(jnp.sort(z, axis=axis), axis=axis)
    rho = _rho(z.shape[axis], z.ndim, axis, z.dtype)
    csum = jnp.cumsum(z_srt, axis=axis)
    support = (1.0 + rho * z_srt) > csum
    k = jnp.sum(support, axis=axis, keepdims=True)
    tau = (jnp.take_along_axis(csum, k - 1, axis=axis) - 1.0) / k
    return jnp.clip(z - tau, a_min=0.0)


def _sparsemax_fwd(z, axis):
    p = sparsemax(z, axis)
    return p, p


sparsemax.defvjp(_sparsemax_fwd, partial(_sparse_bwd, True))


# -------------------------------------------------------------- 1.5-entmax (alpha = 1.5) ----
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def entmax15(z: jax.Array, axis: int = -1) -> jax.Array:
    z = (z - jnp.max(z, axis=axis, keepdims=True)) / 2.0
    z_srt = jnp.flip(jnp.sort(z, axis=axis), axis=axis)
    rho = _rho(z.shape[axis], z.ndim, axis, z.dtype)
    mean = jnp.cumsum(z_srt, axis=axis) / rho
    mean_sq = jnp.cumsum(z_srt**2, axis=axis) / rho
    ss = rho * (mean_sq - mean**2)
    delta = jnp.clip((1.0 - ss) / rho, a_min=0.0)
    tau = mean - jnp.sqrt(delta)
    k = jnp.sum(tau <= z_srt, axis=axis, keepdims=True)
    tau_star = jnp.take_along_axis(tau, k - 1, axis=axis)
    return jnp.clip(z - tau_star, a_min=0.0) ** 2


def _entmax15_fwd(z, axis):
    p = entmax15(z, axis)
    return p, p


entmax15.defvjp(_entmax15_fwd, partial(_sparse_bwd, False))


def normalize(kind: str, logits: jax.Array, axis: int = -1) -> jax.Array:
    """Dispatch a probability normalizer over `axis`. `kind` in {softmax, sparsemax, entmax15}."""
    if kind == "softmax":
        return jax.nn.softmax(logits, axis=axis)
    if kind == "sparsemax":
        return sparsemax(logits, axis)
    if kind == "entmax15":
        return entmax15(logits, axis)
    raise ValueError(f"unknown attn_norm: {kind!r}")
