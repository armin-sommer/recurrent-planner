"""State-indexed masked-attention recurrent core (a drop-in replacement for the ConvLSTM/DRC core).

This module mirrors ``cleanba.convlstm`` but swaps the convolutional message passing inside the
recurrent cell for **multi-head attention over the S = H*W spatial tokens**, while keeping
everything else identical so that ``cleanba.network`` and most of ``BaseLSTM`` are untouched.

Mapping to the relational-update formalism
    h_t^{k+1}(s') = AGG_{(s,s') in N} F_phi( h_t^k(s) )
- latent cell h_t(s): the C-channel vector at spatial position s in ``LSTMCellState.h`` (NHWC).
- graph N:           the attention support. Controlled by ``use_attention_mask`` (see below).
- edge weight phi:   the attention logit <q_s, k_s'> / sqrt(d) + offset-tied relative bias.
- message F_phi:     the value projection W_v.
- aggregation (+):   masked softmax-weighted sum (readout="softmax"), or a soft Bellman max-plus
                     mellowmax (readout="maxplus", the VIN-aligned operator); then the *same* LSTM
                     gate as ConvLSTM.
- K thinking steps:  inherited inner ``nn.scan`` over ``repeats_per_step`` (BaseLSTM._apply_cells).
- t env time:        inherited outer ``nn.scan`` (BaseLSTM.scan) / single ``BaseLSTM.step``.

Masking (``AttentionCellConfig.use_attention_mask``)
    True  -> hard, data-independent local support: cell s attends only to its grid neighbours
             N(s) (+ self, + global tokens). This makes the relational/locality inductive bias
             *provable*: new_h[s] depends only on carry.c[s], the input at s, the neighbours
             s' in N(s), and the (recomputed) global tokens -- exactly the dependency cone of a
             stacked 3x3 conv. This is the recommended default and the configuration the
             localization-of-decision-statistics argument relies on.
    False -> full (dense) attention: every state may message every other state in one step.
             Strictly more expressive, but it breaks the locality guarantee and is prone to
             over-smoothing under the K x T iterated application. Use as an ablation arm.

Localization safeguards baked in: NHWC carry with a strictly per-token gate (state s is only ever
written back into slot s); the global/register tokens are recomputed each step and kept OUT of the
carry; offset-tied relative bias (O(H*W) params, translation-equivariant, like a conv kernel);
self-edge always kept; zero-init of the relative-bias table and the attention output projection so
the cell starts as a gated identity / soft box-conv.
"""
import dataclasses
import math
from typing import List, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from cleanba.convlstm import BaseLSTM, BaseLSTMConfig, ConvConfig, LSTMCellState, LSTMState


# --------------------------------------------------------------------------------------------------
# Static-shape graph helpers. H and W are Python ints (static under jit / eval_shape), so these are
# constant-folded; they build (S, S) arrays over the S = H*W spatial tokens.
# --------------------------------------------------------------------------------------------------
def _adjacency(H: int, W: int, king: bool) -> jax.Array:
    """[S, S] boolean adjacency of the spatial lattice, self-edge included.

    king=True  -> 8-neighbourhood (Chebyshev distance <= 1), matching a 3x3 conv window.
    king=False -> 4-neighbourhood (von Neumann, Manhattan distance <= 1).
    """
    idx = jnp.arange(H * W)
    r, c = idx // W, idx % W
    dr = jnp.abs(r[:, None] - r[None, :])
    dc = jnp.abs(c[:, None] - c[None, :])
    if king:
        return (dr <= 1) & (dc <= 1)
    return (dr + dc) <= 1


def _rel_offset_index(H: int, W: int) -> jax.Array:
    """[S, S] int index into a flattened (2H-1)*(2W-1) relative-offset table."""
    idx = jnp.arange(H * W)
    r, c = idx // W, idx % W
    return (r[:, None] - r[None, :] + (H - 1)) * (2 * W - 1) + (c[:, None] - c[None, :] + (W - 1))


def _compute_output_dim(dim: int, kernel: int, stride: int, padding) -> int:
    """Mirror of cleanba.convlstm.ConvLSTM.initialize_carry's spatial-dim arithmetic."""
    if isinstance(padding, str):
        if padding.upper() == "SAME":
            return -(-dim // stride)  # ceil(dim / stride)
        elif padding.upper() == "VALID":
            return (dim - kernel + stride) // stride
        raise ValueError(f"Unknown padding: {padding}")
    elif isinstance(padding, int):
        return (dim + 2 * padding - kernel) // stride + 1
    elif isinstance(padding, (tuple, list)):
        p = padding[0] if isinstance(padding[0], int) else sum(padding[0])
        return (dim + 2 * p - kernel) // stride + 1
    raise ValueError(f"Unsupported padding type: {padding}")


# --------------------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AttentionCellConfig:
    features: int = 32                          # n, the per-state latent dimension (C)
    num_heads: int = 4

    # --- attention support / graph N ---
    use_attention_mask: bool = True             # <<< TOGGLE: True = hard local mask, False = dense
    mask_neighborhood: Literal["king", "vonneumann"] = "king"  # only used when masking is ON

    # --- global / register tokens (the attention analogue of pool_and_inject) ---
    n_global: int = 0                           # 0 = pure local (no global/register tokens); >0 adds that many

    # --- aggregation readout: how each cell combines its neighbours' values ---
    readout: Literal["softmax", "maxplus"] = "maxplus"  # soft Bellman max-plus (VIN-like, default) vs convex average
    maxplus_beta_init: float = 1.0              # maxplus only: initial inverse temp (higher = closer to hard max)
    maxplus_beta_max: float = 10.0             # maxplus only: cap on the (learnable) inverse temp, for numerical stability

    # --- direction-specific value routing (VIN/conv-aligned) ---
    # When True, each relative offset gets its OWN value projection W_v^offset (instead of one shared
    # W_v), so a neighbour's value is transported differently depending on the direction it comes from
    # -- exactly the per-offset transition routing a Bellman backup / value-iteration needs (and what a
    # 3x3 conv has but shared-W_v attention lacks). Implemented as an efficient NHWC shift-based local
    # message pass (one shift per offset, O(S*offsets*dh)), so it does NOT materialize the (S,Kn,dh)
    # tensor. Pure-local only (requires n_global=0). The content-dependent attention weight (q.k +
    # rel_bias) and the maxplus/softmax aggregation are unchanged; only the value path becomes anisotropic.
    directional_value: bool = False

    # --- priors / stability ---
    use_rel_bias: bool = True                   # offset-tied relative-position bias (soft-conv prior)
    pre_norm: bool = True                        # RMSNorm the tokens before q/k/v projection

    # --- gate (matches ConvLSTMCellConfig) ---
    forget_bias: float = 0.0
    output_activation: Literal["sigmoid", "tanh"] = "sigmoid"


@dataclasses.dataclass(frozen=True)
class AttentionLSTMConfig(BaseLSTMConfig):
    # NB: subclass BaseLSTMConfig (NOT ConvLSTMConfig) so the ConvLSTM-specific special-casing in
    # cleanba/cleanba_impala.py (the `isinstance(args.net, ConvLSTMConfig)` fence fix-up) is skipped.
    embed: List[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: AttentionCellConfig = AttentionCellConfig()
    use_relu: bool = True

    def make(self) -> "AttentionLSTM":
        return AttentionLSTM(self)


# --------------------------------------------------------------------------------------------------
# Core. Reuses BaseLSTM.step / scan / _apply_cells / apply_cells_once / _mlp verbatim.
# --------------------------------------------------------------------------------------------------
class AttentionLSTM(BaseLSTM):
    cfg: AttentionLSTMConfig

    def setup(self):
        super().setup()  # builds self.dense_list used by BaseLSTM._mlp
        self.conv_list = [
            c.make_conv(kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
            for c in self.cfg.embed
        ]
        self.cell_list = [AttentionCell(self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"observations shape must be [batch, h, w, c] but is {x.shape=}"
        for i, conv in enumerate(self.conv_list):
            x = conv(x)
            if self.cfg.use_relu and i < len(self.conv_list) - 1:
                x = nn.relu(x)
        return x

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> LSTMState:
        # Propagate the input spatial dims through the embed convs so the carry matches the
        # post-embedding feature-map size (mirrors ConvLSTM.initialize_carry, convlstm.py:220).
        n, h, w, c = input_shape
        for conv in self.conv_list:
            ks = conv.kernel_size
            kh, kw = (ks, ks) if isinstance(ks, int) else ks
            st = 1 if conv.strides is None else conv.strides
            sh, sw = (st, st) if isinstance(st, int) else st
            h = _compute_output_dim(h, kh, sh, conv.padding)
            w = _compute_output_dim(w, kw, sw, conv.padding)
        return super().initialize_carry(rng, (n, h, w, c))


class AttentionCell(nn.RNNCellBase):
    cfg: AttentionCellConfig

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        # Contract (identical to ConvLSTMCell.__call__, convlstm.py:318): carry.{c,h}, inputs and
        # prev_layer_hidden are all 4-D NHWC; returns (new_state, new_hidden).
        B, H, W, _ = inputs.shape
        C = self.cfg.features
        nh = self.cfg.num_heads
        assert C % nh == 0, f"features={C} must be divisible by num_heads={nh}"
        dh = C // nh
        S = H * W

        # ----------------------------------------------------------------------------------------
        # Direction-specific value routing (VIN/conv-aligned). Each relative offset gets its OWN
        # value projection W_v^offset, so a neighbour's value is transported differently depending
        # on the direction it arrives from -- the per-offset transition routing a Bellman backup
        # needs (a 3x3 conv has it; shared-W_v attention does not). Efficient NHWC shift impl: one
        # shift per offset, no (S,Kn,dh) tensor. Pure-local only.
        # ----------------------------------------------------------------------------------------
        if self.cfg.directional_value:
            assert self.cfg.n_global == 0, "directional_value requires n_global=0 (pure local)"
            assert self.cfg.use_attention_mask, "directional_value is a local message pass; needs the mask"
            h = carry.h
            if self.cfg.pre_norm:
                h = nn.RMSNorm(name="pre_norm")(h)  # (B,H,W,C)

            if self.cfg.mask_neighborhood == "king":
                offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            else:  # vonneumann
                offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            Onb = len(offsets)

            def shift(x, dr, dc):  # y[b,r,c] = x[b,r+dr,c+dc], 0 outside the grid (no wraparound)
                xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
                return xp[:, 1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W, :]

            q_layer = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="q")
            k_layer = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="k")  # shared k
            q = q_layer(h)  # (B,H,W,nh,dh)
            rel = self.param("rel_bias_dir", nn.initializers.zeros, (nh, Onb))  # per-offset logit bias
            ones = jnp.ones((B, H, W, 1), dtype=h.dtype)

            L_list, V_list, valid_list = [], [], []
            for oi, (dr, dc) in enumerate(offsets):
                h_s = shift(h, dr, dc)                                   # neighbour hidden (B,H,W,C)
                valid = shift(ones, dr, dc)[..., 0]                      # (B,H,W) 1 if neighbour in-grid
                k_o = k_layer(h_s)                                       # (B,H,W,nh,dh)
                v_o = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name=f"v_{oi}")(h_s)  # PER-OFFSET value
                logit = (q * k_o).sum(-1) * (dh ** -0.5) + rel[:, oi][None, None, None, :]  # (B,H,W,nh)
                logit = jnp.where(valid[..., None] > 0, logit, -1e9)
                L_list.append(logit)
                V_list.append(v_o)
                valid_list.append(valid)

            L = jnp.stack(L_list, axis=-1)                  # (B,H,W,nh,Onb)
            V = jnp.stack(V_list, axis=-1)                  # (B,H,W,nh,dh,Onb)
            Kvalid = jnp.stack(valid_list, axis=-1).sum(-1)  # (B,H,W)

            if self.cfg.readout == "softmax":
                w = jax.nn.softmax(L, axis=-1)                      # (B,H,W,nh,Onb)
                attn = (w[..., None, :] * V).sum(-1)                # (B,H,W,nh,dh)
            elif self.cfg.readout == "maxplus":
                bm = self.cfg.maxplus_beta_max
                raw0 = math.log(self.cfg.maxplus_beta_init / (bm - self.cfg.maxplus_beta_init))
                beta = bm * nn.sigmoid(self.param("beta", nn.initializers.constant(raw0), (nh,)))  # (nh,)
                s = beta[None, None, None, :, None, None] * (L[..., None, :] + V)  # (B,H,W,nh,dh,Onb)
                m = jnp.max(s, axis=-1, keepdims=True)                            # stable logsumexp
                lse = m[..., 0] + jnp.log(jnp.sum(jnp.exp(s - m), axis=-1))       # (B,H,W,nh,dh)
                logK = jnp.log(jnp.maximum(Kvalid, 1.0))[..., None, None]         # (B,H,W,1,1)
                attn = (lse - logK) / beta[None, None, None, :, None]            # (B,H,W,nh,dh)
            else:
                raise ValueError(f"{self.cfg.readout=}")

            a = nn.DenseGeneral(
                C, axis=(-2, -1), use_bias=False, name="out",
                kernel_init=nn.initializers.variance_scaling(0.1, "fan_in", "truncated_normal"),
            )(attn)  # (B,H,W,C)

            gate_in = jnp.concatenate([inputs, prev_layer_hidden, a], axis=-1)
            gates = nn.Dense(4 * C, name="gate")(gate_in)
            i, j, f, o = jnp.split(gates, 4, axis=-1)
            i = jnp.tanh(i)
            j = nn.sigmoid(j)
            f = nn.sigmoid(f + self.cfg.forget_bias)
            o = nn.sigmoid(o) if self.cfg.output_activation == "sigmoid" else jnp.tanh(o)
            new_c = carry.c * f + i * j
            new_h = jnp.tanh(new_c) * o
            return LSTMCellState(c=new_c, h=new_h), new_h

        def tok(z):  # (B, H, W, X) -> (B, S, X)
            return z.reshape(B, S, z.shape[-1])

        h_tok = tok(carry.h)
        if self.cfg.pre_norm:
            h_tok = nn.RMSNorm(name="pre_norm")(h_tok)  # per-token norm over the channel axis
        # Local "ih" term: the new observation + the top-down hidden, injected per-token (no mixing).
        in_tok = tok(jnp.concatenate([inputs, prev_layer_hidden], axis=-1))

        # --- global / register tokens == pool_and_inject (recomputed each step, NOT in the carry) ---
        if self.cfg.n_global > 0:
            pooled = jnp.concatenate(
                [h_tok.mean(axis=1, keepdims=True), h_tok.max(axis=1, keepdims=True)], axis=-1
            )  # (B, 1, 2C)
            g = nn.Dense(C, name="global_in")(jnp.repeat(pooled, self.cfg.n_global, axis=1))  # (B, G, C)
            kv = jnp.concatenate([h_tok, g], axis=1)  # (B, S+G, C)
            Kn = S + self.cfg.n_global
        else:
            kv, Kn = h_tok, S

        proj = lambda name, x: nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name=name)(x)
        q = proj("q", h_tok)  # (B, S, nh, dh)
        k = proj("k", kv)     # (B, Kn, nh, dh)
        v = proj("v", kv)     # (B, Kn, nh, dh)

        # --- edge logits: relative-position bias + (optional) hard mask = the graph N ---
        bias = jnp.zeros((1, nh, S, Kn))
        if self.cfg.use_rel_bias:
            tbl = self.param("rel_bias", nn.initializers.zeros, (nh, (2 * H - 1) * (2 * W - 1)))
            rb = tbl[:, _rel_offset_index(H, W)]               # (nh, S, S)  offset-tied, zeros-init
            bias = bias.at[:, :, :, :S].add(rb[None])          # global columns keep a 0 prior
        if self.cfg.use_attention_mask:
            adj = _adjacency(H, W, king=(self.cfg.mask_neighborhood == "king"))  # (S, S), self-edge kept
            if self.cfg.n_global > 0:                          # every token may see the globals
                adj = jnp.concatenate([adj, jnp.ones((S, self.cfg.n_global), dtype=bool)], axis=1)
            bias = jnp.where(adj[None, None], bias, -1e9)

        # --- aggregate neighbour values: softmax convex average, or soft Bellman max-plus ---
        if self.cfg.readout == "softmax":
            attn = nn.dot_product_attention(q, k, v, bias=bias)    # (B, S, nh, dh), softmax over kv axis
        elif self.cfg.readout == "maxplus":
            # out_s(d) = mellowmax_{k in N(s)} [ L_{s,k} + v_{k,d} ]
            #          = (1/beta) * ( logsumexp_k  beta*(L_{s,k} + v_{k,d})  -  log K ),
            # with the edge "reward" L = <q_s,k_k>/sqrt(dh) + bias (rel-pos prior + hard mask). beta -> inf
            # is the hard Bellman max (a shortest-path / value-iteration backup); beta -> 0 is the neighbour
            # mean. Swapping softmax's convex average for this (soft) max removes the over-smoothing that
            # averaging suffers under the K x T iteration -- value propagates one hop per step without
            # geometric attenuation, the property a VIN needs.
            #
            # Implemented as a (B,nh,S,Kn) x (B,nh,Kn,dh) matmul, NOT a (B,nh,S,Kn,dh) tensor: factor the
            # exponent exp(beta*(L+v)) = exp(beta*L)*exp(beta*v) and sum over k via einsum. This keeps the
            # peak memory equal to plain softmax attention (the dropped (S,Kn,dh) intermediate is dh x
            # larger and is retained across the whole T*repeats*layers unroll for backprop -> the OOM).
            # Separable max-shifts (mL per query, mv over values) keep the exponentials in (0, 1].
            L = jnp.einsum("bshd,bkhd->bhsk", q, k) * (dh ** -0.5) + bias        # (B, nh, S, Kn)
            # beta = beta_max * sigmoid(raw), so it stays in (0, beta_max): a learnable inverse temperature
            # bounded for numerical stability (mellowmax at beta~10 is already near hard-max). Init at
            # maxplus_beta_init via the logit of (init / max).
            bm = self.cfg.maxplus_beta_max
            raw0 = math.log(self.cfg.maxplus_beta_init / (bm - self.cfg.maxplus_beta_init))
            beta = bm * nn.sigmoid(self.param("beta", nn.initializers.constant(raw0), (nh,)))[None, :, None, None]
            vh = jnp.transpose(v, (0, 2, 1, 3))                                  # (B, nh, Kn, dh)
            mL = jax.lax.stop_gradient(L.max(axis=3, keepdims=True))             # (B, nh, S, 1) per-query shift
            mv = jax.lax.stop_gradient(vh.max(axis=2, keepdims=True))            # (B, nh, 1, dh) per-channel value shift
            a = jnp.exp(beta * (L - mL))                                         # (B, nh, S, Kn) in (0,1]; masked->0
            w = jnp.exp(beta * (vh - mv))                                        # (B, nh, Kn, dh) in (0,1]
            agg = jnp.einsum("bhsk,bhkd->bhsd", a, w)                            # (B, nh, S, dh) = sum_k a*w
            agg = jnp.maximum(agg, jnp.finfo(agg.dtype).tiny)                    # guard: floor underflow (huge beta*gap)
            if self.cfg.use_attention_mask:  # self + grid neighbours (varies at the boundary) + all globals
                spatial = _adjacency(H, W, king=(self.cfg.mask_neighborhood == "king")).sum(axis=1)  # (S,)
                neighbours = spatial + self.cfg.n_global
            else:
                neighbours = jnp.full((S,), Kn)
            logK = jnp.log(neighbours.astype(agg.dtype))[None, None, :, None]    # (1, 1, S, 1) mellowmax norm
            # mL + mv undo the shifts exactly: out = (1/beta)(logsumexp_k beta(L+v) - log K).
            attn = (mL + mv + (jnp.log(agg) - logK) / beta).transpose(0, 2, 1, 3)  # (B, S, nh, dh)
        else:
            raise ValueError(f"{self.cfg.readout=}")
        # W_o SMALL (not zero) init. Zero-init fully gradient-isolates the entire attention
        # sub-block (q/k/v/rel_bias/beta) and carry.h from the output at step 0 -- so the core can
        # only learn through the conv-embed + gate + skip_final bypass, attention never engages, and
        # the policy stays near-uniform (verified: tests/check_attn_learns.py showed all attn params
        # dead-grad at init). A small nonzero scale keeps init near-identity (stable) while letting
        # gradient reach the attention params from step 0. Tune: lower to 0.01 if unstable, raise
        # toward 1.0 if attention engages too slowly.
        a = nn.DenseGeneral(
            C, axis=(-2, -1), use_bias=False, name="out",
            kernel_init=nn.initializers.variance_scaling(0.1, "fan_in", "truncated_normal"),
        )(attn)

        # --- fused gates (local ih term + attention message) and the IDENTICAL LSTM update ---
        gates = nn.Dense(4 * C, name="gate")(jnp.concatenate([in_tok, a], axis=-1))  # (B, S, 4C)
        i, j, f, o = jnp.split(gates, 4, axis=-1)
        i = jnp.tanh(i)
        j = nn.sigmoid(j)
        f = nn.sigmoid(f + self.cfg.forget_bias)
        if self.cfg.output_activation == "sigmoid":
            o = nn.sigmoid(o)
        elif self.cfg.output_activation == "tanh":
            o = jnp.tanh(o)
        else:
            raise ValueError(f"{self.cfg.output_activation=}")

        new_c = tok(carry.c) * f + i * j
        new_h = nn.tanh(new_c) * o
        new_c = new_c.reshape(B, H, W, C)
        new_h = new_h.reshape(B, H, W, C)
        return LSTMCellState(c=new_c, h=new_h), new_h

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> LSTMCellState:
        shape = (*input_shape[:-1], self.cfg.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))

    def num_feature_axes(self) -> int:
        return 3
