import dataclasses
from typing import List, Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

# These static-shape grid helpers (constant-folded; H, W are Python ints) live in attn_lstm, which is
# their canonical home. Reused here so the two arms share one copy of the lattice indexing math.
from cleanba.attn_lstm import _adjacency, _compute_output_dim, _rel_offset_index
from cleanba.convlstm import BaseLSTM, BaseLSTMConfig, ConvConfig, LSTMCellState, LSTMState


# --------------------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CellwiseLSTMCellConfig:
    features: int = 32  # n, the per-state latent dimension (C)

    # --- message MLP F_phi (applied per source token, shared across positions) ---
    message_hiddens: Tuple[int, ...] = (64,)  # hidden widths; a final Dense(features) is appended

    # --- neighbourhood graph N ---
    use_neighbor_mask: bool = True  # True = local grid neighbourhood; False = dense all-to-all
    mask_neighborhood: Literal["king", "vonneumann"] = "king"  # only used when masking is ON
    aggregation: Literal["mean", "sum", "max"] = "mean"
    use_edge_weights: bool = True  # offset-tied learned edge weights (soft-conv prior); mean/sum only

    # --- message expressivity (pool-free levers to break the value-learning deadlock) ---
    zero_init_message: bool = True   # msg projection zeros-init (gated identity at init). False -> standard init: spatial path ACTIVE from step 0, like DRC's conv.
    per_offset_message: bool = False  # True -> direction-specific final projection (per-offset C-matrix == a real conv kernel) instead of one shared map + scalar edge weight. Requires use_neighbor_mask=True and aggregation="sum".
    attend_inputs: bool = False  # DRC conv_ih analog: fold the live obs + top-down hidden into the message SOURCE so the spatial op processes the CURRENT board, not only the recurrent carry.

    # --- global / register tokens (the analogue of pool_and_inject) ---
    n_global: int = 4  # 0 disables them entirely

    # --- priors / stability ---
    pre_norm: bool = True  # RMSNorm the tokens before the message MLP

    # --- gate (matches ConvLSTMCellConfig / AttentionCellConfig) ---
    forget_bias: float = 0.0
    output_activation: Literal["sigmoid", "tanh"] = "sigmoid"


@dataclasses.dataclass(frozen=True)
class CellwiseLSTMConfig(BaseLSTMConfig):
    # NB: subclass BaseLSTMConfig (NOT ConvLSTMConfig) so the ConvLSTM-specific special-casing in
    # cleanba/cleanba_impala.py (the `isinstance(args.net, ConvLSTMConfig)` fence fix-up) is skipped.
    embed: List[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: CellwiseLSTMCellConfig = CellwiseLSTMCellConfig()
    use_relu: bool = True

    def make(self) -> "CellwiseLSTM":
        return CellwiseLSTM(self)


# --------------------------------------------------------------------------------------------------
# Core. Reuses BaseLSTM.step / scan / _apply_cells / apply_cells_once / _mlp verbatim. The module
# body is identical to AttentionLSTM apart from the cell class it instantiates.
# --------------------------------------------------------------------------------------------------
class CellwiseLSTM(BaseLSTM):
    cfg: CellwiseLSTMConfig

    def setup(self):
        super().setup()  # builds self.dense_list used by BaseLSTM._mlp
        self.conv_list = [
            c.make_conv(kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
            for c in self.cfg.embed
        ]
        self.cell_list = [CellwiseLSTMCell(self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

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


class CellwiseLSTMCell(nn.RNNCellBase):
    cfg: CellwiseLSTMCellConfig

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        # Contract (identical to ConvLSTMCell.__call__, convlstm.py:318): carry.{c,h}, inputs and
        # prev_layer_hidden are all 4-D NHWC; returns (new_state, new_hidden).
        B, H, W, _ = inputs.shape
        C = self.cfg.features
        S = H * W

        def tok(z):  # (B, H, W, X) -> (B, S, X)
            return z.reshape(B, S, z.shape[-1])

        h_tok = tok(carry.h)
        # Local "ih" term: the new observation + the top-down hidden.
        in_tok = tok(jnp.concatenate([inputs, prev_layer_hidden], axis=-1))
        if self.cfg.attend_inputs:  # DRC conv_ih analog: fold the live obs into the message SOURCE so the
            # spatial op (per-offset conv / aggregation) processes the CURRENT board, not only carry.h.
            h_tok = h_tok + nn.Dense(C, name="in_proj")(in_tok)
        if self.cfg.pre_norm:
            h_tok = nn.RMSNorm(name="pre_norm")(h_tok)  # per-token norm over the channel axis

        # --- global / register tokens == pool_and_inject (recomputed each step, NOT in the carry) ---
        if self.cfg.n_global > 0:
            pooled = jnp.concatenate(
                [h_tok.mean(axis=1, keepdims=True), h_tok.max(axis=1, keepdims=True)], axis=-1
            )  # (B, 1, 2C)
            g = nn.Dense(C, name="global_in")(jnp.repeat(pooled, self.cfg.n_global, axis=1))  # (B, G, C)
            src = jnp.concatenate([h_tok, g], axis=1)  # (B, S+G, C)
        else:
            src = h_tok

        # --- message F_phi(h(s)): shared per-source MLP hidden layers, then either ONE shared final
        #     projection (msg_out, aggregated with scalar edge weights) or, if per_offset_message, a
        #     DIRECTION-SPECIFIC per-offset map (== a real conv kernel). zeros-init keeps the cell a
        #     gated identity at start; non-zero init turns the spatial path on from step 0 (like DRC). ---
        msg_init = (nn.initializers.zeros if self.cfg.zero_init_message
                    else nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
        m = src
        for i, hid in enumerate(self.cfg.message_hiddens):
            m = nn.relu(nn.Dense(hid, name=f"msg_hidden_{i}")(m))
        if self.cfg.per_offset_message:
            assert self.cfg.use_neighbor_mask, "per_offset_message is a local op; needs the mask"
            assert self.cfg.aggregation == "sum", "per_offset_message uses sum aggregation (a conv)"
            msg = self._aggregate_per_offset(m, H, W, msg_init)  # (B, S, C)
        else:
            m = nn.Dense(C, name="msg_out", use_bias=False, kernel_init=msg_init)(m)  # (B, S+G, C)
            # --- aggregate messages over each cell's neighbours: the graph N ---
            msg = self._aggregate(m, H, W)  # (B, S, C)

        # --- fused gates (local ih term + aggregated message) and the IDENTICAL LSTM update ---
        gates = nn.Dense(4 * C, name="gate")(jnp.concatenate([in_tok, msg], axis=-1))  # (B, S, 4C)
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

    def _aggregate(self, m: jax.Array, H: int, W: int) -> jax.Array:
        """Aggregate per-source messages ``m`` (B, S+G, C) over each spatial cell's neighbours -> (B, S, C).

        The first S columns of ``m`` are the spatial tokens, the last G are the global tokens (every
        cell sees all globals). The graph operator is input-independent: a fixed adjacency, optionally
        weighted by an offset-tied learned soft-conv kernel.
        """
        S = H * W
        G = self.cfg.n_global

        if self.cfg.use_neighbor_mask:
            adj = _adjacency(H, W, king=(self.cfg.mask_neighborhood == "king"))  # (S, S), self-edge kept
        else:
            adj = jnp.ones((S, S), dtype=bool)  # dense all-to-all message passing (over-smoothing ablation)
        if G > 0:  # every cell may aggregate every global token
            adj = jnp.concatenate([adj, jnp.ones((S, G), dtype=bool)], axis=1)  # (S, S+G)

        if self.cfg.aggregation == "max":
            # Memory-efficient masked/dense max. The naive (B, S, S+G, C) materialization OOMs at the
            # training batch size (B ~ 1e4 -> ~24 GB); reduce directly instead. Masked = elementwise max
            # over the <=9 neighbour OFFSETS via padded shifts (O(B*S*C*offsets)); dense = one global max.
            B = m.shape[0]
            Cc = m.shape[-1]
            neg = jnp.finfo(m.dtype).min
            m_sp = m[:, :S, :].reshape(B, H, W, Cc)  # spatial tokens as a feature map
            if self.cfg.use_neighbor_mask:
                if self.cfg.mask_neighborhood == "king":
                    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
                else:
                    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

                def _shift_neg(x, dr, dc):  # out-of-grid -> -inf so it never wins the max
                    xp = jnp.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=neg)
                    return xp[:, 1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W, :]

                acc = _shift_neg(m_sp, *offsets[0])
                for off in offsets[1:]:
                    acc = jnp.maximum(acc, _shift_neg(m_sp, *off))
                out = acc.reshape(B, S, Cc)
            else:  # dense all-to-all: each cell's neighbourhood is the whole grid -> global max
                out = jnp.broadcast_to(m_sp.reshape(B, S, Cc).max(axis=1, keepdims=True), (B, S, Cc))
            if G > 0:  # every cell also sees all global tokens
                out = jnp.maximum(out, m[:, S:, :].max(axis=1, keepdims=True))
            return out

        # mean / sum: a single (weighted) adjacency matmul, no (S, S+G, C) intermediate.
        weight = adj.astype(m.dtype)  # (S, S+G)
        if self.cfg.use_edge_weights:
            # Offset-tied positive edge weights: O(H*W) params, translation-equivariant (a soft conv
            # kernel). zeros-init -> exp(0)=1 -> a plain mean/sum at the start of training.
            tbl = self.param("edge_logits", nn.initializers.zeros, ((2 * H - 1) * (2 * W - 1),))
            weight = weight.at[:, :S].multiply(jnp.exp(tbl[_rel_offset_index(H, W)]))  # (S, S) spatial block
            if G > 0:
                gtbl = self.param("global_logits", nn.initializers.zeros, (G,))
                weight = weight.at[:, S:].multiply(jnp.exp(gtbl)[None, :])
        if self.cfg.aggregation == "mean":
            weight = weight / weight.sum(axis=1, keepdims=True)  # row-normalize over each cell's neighbours
        elif self.cfg.aggregation != "sum":
            raise ValueError(f"{self.cfg.aggregation=}")
        return jnp.einsum("sk,bkc->bsc", weight, m)  # (B, S, C)

    def _aggregate_per_offset(self, m: jax.Array, H: int, W: int, init) -> jax.Array:
        """Direction-specific local message passing == a 3x3 conv on the hidden message ``m`` (B, S+G, hid).

        Instead of one shared output projection + a scalar offset weight (``_aggregate``), each relative
        offset o in the neighbourhood gets its OWN linear map W_o (hid->C): a neighbour's content is routed
        differently depending on the direction it arrives from -- the anisotropic per-offset routing a 3x3
        conv has but the shared-message+scalar-weight path lacks. Zero-padded at the boundary (conv SAME).
        """
        B = m.shape[0]
        hid = m.shape[-1]
        C = self.cfg.features
        S = H * W
        G = self.cfg.n_global
        m_sp = m[:, :S, :].reshape(B, H, W, hid)  # spatial tokens as a feature map
        if self.cfg.mask_neighborhood == "king":
            offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        else:
            offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        Onb = len(offsets)
        Wv = self.param("v_offset", init, (Onb, hid, C))  # per-offset linear maps (the conv kernel)

        def _shift0(x, dr, dc):  # zero-padded shift (out-of-grid -> 0, i.e. conv SAME)
            xp = jnp.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
            return xp[:, 1 + dr : 1 + dr + H, 1 + dc : 1 + dc + W, :]

        acc = jnp.zeros((B, H, W, C), m.dtype)
        for jj, (dr, dc) in enumerate(offsets):
            acc = acc + jnp.einsum("bhwk,kc->bhwc", _shift0(m_sp, dr, dc), Wv[jj])
        out = acc.reshape(B, S, C)
        if G > 0:  # every cell also receives the pooled global tokens through their own map
            Wg = self.param("v_global", init, (hid, C))
            out = out + jnp.einsum("bgk,kc->bc", m[:, S:, :], Wg)[:, None, :]
        return out

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> LSTMCellState:
        shape = (*input_shape[:-1], self.cfg.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))

    def num_feature_axes(self) -> int:
        return 3
