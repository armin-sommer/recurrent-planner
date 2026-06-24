"""Pool-and-inject relational core: a relational-memory backbone with NO positional binding.

Where ``cleanba.attn_lstm`` pins cell i to board square i (positional input) and ``cleanba.slot_lstm``
gives each cell its own *competitively-bound* slice of the board (slot-attention, softmax over slots),
this core does neither. The encoder is POOLED to one global vector and INJECTED identically into every
cell (DRC-style pool-and-inject, but the relational op is attention, not convolution). The cells start
identical (a learnable per-cell identity mu_i breaks the symmetry) and differentiate purely through the
dense cell<->cell ROUTING attention + the LSTM gate. This is the natural core for first-person / partial
observation, where there is no allocentric grid to pin cells to.

Per thinking tick, each cell receives:
  * INJECT (same for all cells): a projection of the pooled encoder features  -- "every cell same input".
  * TOP-DOWN: prev_layer_hidden, the lower stacked cell's output (so depth D>1 is meaningful; unlike the
    slot cell, which ignores it).
  * ROUTE: a dense cell<->cell self-attention message (softmax / 1.5-entmax / sparsemax over source
    cells), the same routing op as the slot core -- the ``route_attn`` map the E4 probe reads.
Then the proven LSTM gate integrates [inject+topdown, route] with the carry.

Optional ``read_tokens`` adds a Perceiver-style cross-attention read of the encoder tokens (softmax over
TOKENS, not the slot competition) so cells can pull spatial detail -- the ablation knob; default OFF = the
pure pool-and-inject design. ``skip_final`` MUST be False (the carry is (B,N,d), not the spatial embed).
"""
from __future__ import annotations

import dataclasses
from typing import List, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from cleanba.convlstm import BaseLSTM, BaseLSTMConfig, ConvConfig, LSTMCellState
from cleanba.entmax import normalize as normalize_attn


# --------------------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PoolInjectCellConfig:
    features: int = 32                  # d, per-cell latent dim
    num_cells: int = 100                # N relational-memory cells (no spatial position)
    num_heads: int = 4

    # --- routing (cell<->cell) ---
    routing_norm: Literal["softmax", "entmax15", "sparsemax"] = "softmax"  # dense | 1.5-entmax | sparsemax
    out_init_scale: float = 0.1         # routing output-projection scale (small => routing ~off at init)

    # --- pool-and-inject ---
    pool: Literal["mean", "max", "meanmax"] = "meanmax"  # how the encoder tokens are pooled to the global vector
    read_tokens: bool = False           # ABLATION: also Perceiver-cross-attend the encoder tokens (softmax
                                        # over tokens). False = pure pool-and-inject ("every cell same input").

    # --- stability ---
    mu_init_scale: float = 1.0          # stddev of the learnable per-cell identity mu_i (symmetry breaking)
    pre_norm: bool = True               # RMSNorm before q/k/v

    # --- gate ---
    forget_bias: float = 0.0
    output_activation: Literal["sigmoid", "tanh"] = "tanh"


@dataclasses.dataclass(frozen=True)
class PoolInjectLSTMConfig(BaseLSTMConfig):
    # Subclass BaseLSTMConfig (NOT ConvLSTMConfig) so the ConvLSTM fence fix-up in cleanba_impala.py is
    # skipped. skip_final MUST be False (carry is (B,N,d), not the spatial embed map).
    embed: List[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: PoolInjectCellConfig = PoolInjectCellConfig()
    use_relu: bool = True

    def make(self) -> "PoolInjectLSTM":
        return PoolInjectLSTM(self)


# --------------------------------------------------------------------------------------------------
# Core. Reuses BaseLSTM.step / scan / apply_cells_once / _apply_cells / _mlp verbatim (the base zeroes
# the carry at episode start, which is correct here: the per-cell mu re-supplies identity each tick).
# --------------------------------------------------------------------------------------------------
class PoolInjectLSTM(BaseLSTM):
    cfg: PoolInjectLSTMConfig

    def setup(self):
        super().setup()  # builds self.dense_list used by BaseLSTM._mlp
        self.conv_list = [
            c.make_conv(kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
            for c in self.cfg.embed
        ]
        self.cell_list = [PoolInjectCell(self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"observations shape must be [batch, h, w, c] but is {x.shape=}"
        for i, conv in enumerate(self.conv_list):
            x = conv(x)
            if self.cfg.use_relu and i < len(self.conv_list) - 1:
                x = nn.relu(x)
        return x


class PoolInjectCell(nn.RNNCellBase):
    cfg: PoolInjectCellConfig

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        # carry.{c,h}: (B, N, d).  inputs: (B, H, W, C_embed).  prev_layer_hidden: (B, N, d) (== carry.h
        # for a single layer; the lower stacked cell's output for D>1).
        B, H, W, Ce = inputs.shape
        N, d, nh = self.cfg.num_cells, self.cfg.features, self.cfg.num_heads
        assert d % nh == 0, f"features={d} must be divisible by num_heads={nh}"
        dh = d // nh
        S = H * W
        tokens = inputs.reshape(B, S, Ce)

        # Persistent learnable per-cell identity (re-added each tick): the only thing distinguishing the
        # cells at init, so a zeroed carry (episode start) still yields distinct cells.
        cell_id = self.param("cell_mu", nn.initializers.normal(self.cfg.mu_init_scale), (N, d))
        h = carry.h + cell_id[None]  # (B, N, d): identity-augmented recurrent state (routing source)

        # ---------------- POOL + INJECT: one global vector, identical to every cell ----------------
        parts = []
        if self.cfg.pool in ("mean", "meanmax"):
            parts.append(tokens.mean(axis=1))                     # (B, Ce)
        if self.cfg.pool in ("max", "meanmax"):
            parts.append(tokens.max(axis=1))                      # (B, Ce)
        g = jnp.concatenate(parts, axis=-1)                       # (B, Ce or 2Ce)
        inject = nn.Dense(d, name="inject")(g)[:, None, :]        # (B, 1, d) -> broadcast to all N cells
        # Top-down stack input (makes depth meaningful; for D=1 this is a projection of carry.h).
        topdown = nn.Dense(d, use_bias=False, name="topdown")(prev_layer_hidden)  # (B, N, d)
        m_in = inject + topdown                                   # (B, N, d): the per-cell input message

        # Optional Perceiver-style spatial read (ablation; OFF => pure pool-inject).
        if self.cfg.read_tokens:
            hr0 = nn.RMSNorm(name="read_norm")(h) if self.cfg.pre_norm else h
            qr0 = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="read_q")(hr0)      # (B,N,nh,dh)
            kr0 = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="read_k")(tokens)   # (B,S,nh,dh)
            vr0 = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="read_v")(tokens)   # (B,S,nh,dh)
            Lr = jnp.einsum("bnhe,bshe->bhns", qr0, kr0) * (dh ** -0.5)
            wr = jax.nn.softmax(Lr, axis=-1)                       # over TOKENS (standard attention, not competition)
            self.sow("intermediates", "read_attn", wr)
            m_read = jnp.einsum("bhns,bshe->bnhe", wr, vr0)
            m_read = nn.DenseGeneral(d, axis=(-2, -1), use_bias=False, name="read_out")(m_read)
            m_in = m_in + m_read

        # ---------------- ROUTE: dense cell<->cell self-attention (the recovered graph) ----------------
        hr = nn.RMSNorm(name="route_norm")(h) if self.cfg.pre_norm else h
        qr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_q")(hr)  # (B,N,nh,dh)
        kr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_k")(hr)
        vr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_v")(hr)
        L = jnp.einsum("bnhe,bmhe->bhnm", qr, kr) * (dh ** -0.5)                     # (B,nh,N,N)
        w = normalize_attn(self.cfg.routing_norm, L, axis=-1)                        # over source cells m
        self.sow("intermediates", "route_attn", w)                                  # (B,nh,N,N): the routing graph
        m_route = jnp.einsum("bhnm,bmhe->bnhe", w, vr)                              # (B,N,nh,dh)
        m_route = nn.DenseGeneral(
            d, axis=(-2, -1), use_bias=False, name="route_out",
            kernel_init=nn.initializers.variance_scaling(self.cfg.out_init_scale, "fan_in", "truncated_normal"),
        )(m_route)                                                                  # (B,N,d)

        # ---------------- GATE: the proven LSTM update, fed [input message, routing message] ----------
        gates = nn.Dense(4 * d, name="gate")(jnp.concatenate([m_in, m_route], axis=-1))  # (B,N,4d)
        i, j, f, o = jnp.split(gates, 4, axis=-1)
        i = jnp.tanh(i)
        j = nn.sigmoid(j)
        f = nn.sigmoid(f + self.cfg.forget_bias)
        o = nn.sigmoid(o) if self.cfg.output_activation == "sigmoid" else jnp.tanh(o)
        new_c = carry.c * f + i * j
        new_h = jnp.tanh(new_c) * o
        return LSTMCellState(c=new_c, h=new_h), new_h

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> LSTMCellState:
        # Cells decouple from spatial dims: carry is (batch, num_cells, features). Zeroed here; the
        # reset-to-zero at episode start is fine because cell_mu re-supplies identity each tick.
        shape = (input_shape[0], self.cfg.num_cells, self.cfg.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))

    def num_feature_axes(self) -> int:
        return 2
