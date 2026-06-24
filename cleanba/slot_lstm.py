"""Learnable-slot relational recurrent core: a *fully discovered* binding + routing.

This is the "even more amazing" instantiation of the paper's Assumption (recurrent relational
core). Where ``cleanba.attn_lstm`` keeps the latent cells on the H*W spatial grid -- so the binding
sigma:state->cell is *handed to the network* by spatial weight sharing (offset-tied bias, king
mask) -- this core decouples the cells from the grid entirely:

    * The latent cells are N free "slots" h_t(i) in R^d (config ``num_slots``), with NO spatial
      position. At init the only thing distinguishing slot i from slot j is a learnable init vector
      mu_i; the *update* weights are shared across slots (weight-tied, permutation-equivariant), so
      the binding sigma is genuinely *undetermined at init and must be broken by training* -- exactly
      the symmetry-breaking the theory's Assumption describes, which the spatial core cannot test
      because the grid already breaks it.

Mapping to the relational-update formalism  h_t^{k+1}(i) = AGG_j alpha(h(j),h(i)) F(h(j)):
  - latent cell h_t(i):  slot i's d-vector in ``LSTMCellState.h`` of shape (B, N, d).
  - binding sigma:        slot-attention COMPETITION between slots and the H*W input-feature tokens
                          (softmax *over slots* -> each board location is explained by one slot;
                          this competition is the symmetry-breaking that ties slots to states).
  - graph N / routing:    DENSE slot<->slot self-attention (all-to-all over the N slots, NO mask).
                          The theory's central claim is that this *learns* N -- slot i attends slot
                          j iff the states they bind are one transition apart -- with zero spatial
                          prior to hand it the answer (unlike the spatial core, which keeps an
                          offset-tied relative-position bias even when "dense").
  - aggregation (+):      softmax convex average (``routing_readout="softmax"``, default, most
                          stable) or the soft Bellman max-plus mellowmax (``"maxplus"``, VIN-aligned).
  - F (content map):      the value projections W_v.
  - K thinking steps:     the inner nn.scan over ``repeats_per_step`` (BaseLSTM._apply_cells); each
                          tick propagates routing one hop -> K-step lookahead (Proposition).
  - t env time:           the outer nn.scan (BaseLSTM.scan); slots persist as the recurrent carry.

Trained by PURE policy-gradient + TD (no reconstruction / inverse-dynamics aux loss): binding must
emerge from RL signal alone. To keep that claim while preventing the collapse/over-smoothing this
core is prone to (cf. dense_sum diverged; attn_masked hard-collapsed), every anti-collapse device
here is ARCHITECTURAL, not objective-level -- it adds no non-RL gradient:
  1. Slots reset to the learnable per-slot init mu_i at episode start, NEVER to zero (a zeroed,
     identical slot field collapses immediately). mu also gives slots a *stable identity* across
     time, which the flatten->MLP readout and any later probe rely on.
  2. Binding uses the slot-attention competition (softmax over slots) -- the canonical anti-collapse
     normalization (slots must compete, so they cannot all explain everything).
  3. Routing readout is softmax/maxplus, never an unnormalized sum.
  4. The routing output projection is small-init (``out_init_scale``) -> the core starts as a
     binding-only "slot autoencoder" and engages routing gradually.
  5. RMSNorm pre-norm on the slots; the proven LSTM gate integrates messages with the carry.
  6. mu is init with real variance (``mu_init_scale``) so the N slots are distinct from step 0
     (this breaks the init symmetry stochastically, like slot-attention's random init -- it is NOT
     a spatial prior: mu_i are just distinct learnable vectors, not tied to any board position).

Everything else (the conv embed, the gate, BaseLSTM.step/scan/_apply_cells/_mlp, the actor/critic
heads) is reused verbatim. The only contract change is the carry shape (B, N, d) instead of
(B, H, W, C); BaseLSTM._mlp flattens the hidden before the heads, so that is transparent downstream.
``skip_final`` MUST be False (the base would otherwise add the spatial embed map to the slot output).
"""
import dataclasses
import math
from typing import List, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from cleanba.convlstm import BaseLSTM, BaseLSTMConfig, ConvConfig, LSTMCellState
from cleanba.entmax import normalize as normalize_attn


def _identity_logit_init(diag_scale: float):
    """Init a (nh, N, S) positional-binding logit table to the dense-attention IDENTITY: slot i -> board
    position round(i*S/N) gets `diag_scale`, all else 0, so the softmax-over-slots competition starts as
    cell=square (slot i reads ~position i) and is then free to adapt. For N==S this is the exact identity."""
    def init(key, shape, dtype=jnp.float32):
        nh, N, S = shape
        idx = jnp.clip(jnp.round(jnp.arange(N) * (S / N)).astype(jnp.int32), 0, S - 1)
        base = jnp.zeros((N, S), dtype).at[jnp.arange(N), idx].set(diag_scale)
        return jnp.broadcast_to(base[None], (nh, N, S))
    return init


# --------------------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SlotCellConfig:
    features: int = 32                  # d, the per-slot latent dimension
    num_slots: int = 100                # N free latent cells (no spatial position). =H*W for a 1:1
                                        # capacity match to the 10x10 board (cleanest binding).
    num_heads: int = 4

    # --- binding sigma (slot <- board) addressing mode ---
    binding: Literal["content", "positional"] = "content"
                                        # "content" (default slot core): the assignment is queried by the
                                        #   EVOLVING slot hidden -> content-addressed, RE-INDEXES per board
                                        #   (slot i means a different square each task; routing cannot then
                                        #   encode a fixed graph and degenerates to a global broadcast).
                                        # "positional" (D1): the assignment is a learned, board-INDEPENDENT
                                        #   template -- a fixed per-slot address x fixed per-position key,
                                        #   both parameters -- so the slot<->position map is the SAME across
                                        #   tasks (a discovered, stable injection, not the given cell=square
                                        #   identity). WHAT each slot reads is still the board content at its
                                        #   fixed positions. This is the midpoint between dense attention's
                                        #   GIVEN binding and the slot core's per-task LEARNED one, and the
                                        #   test of whether a stable binding restores transition-graph routing.
    bind_init_diag: float = 6.0         # (binding="positional" only) diagonal logit of the warm-started
                                        # identity template: slot i starts reading ~board position i (the
                                        # dense-attention cell=square binding), then adapts. ~6 => ~0.8 of the
                                        # per-position slot-competition on the diagonal at init; higher =>
                                        # sharper/closer to a hard identity (but flatter off-diagonal grads).

    # --- routing (slot<->slot) aggregation ---
    routing_readout: Literal["softmax", "maxplus"] = "softmax"  # softmax = most stable (and the
                                        # strongest attn arm empirically); maxplus = VIN-aligned soft
                                        # Bellman max-plus (the thesis-pure operator, slightly riskier).
    routing_norm: Literal["softmax", "entmax15", "sparsemax"] = "softmax"  # normalizer for the
                                        # softmax routing readout: "softmax" (dense, every slot >0) |
                                        # "entmax15" (alpha=1.5) | "sparsemax" (alpha=2). The sparse
                                        # variants assign EXACT-zero weight to low-scoring slots -> a
                                        # hard learned graph N, mirroring attn_lstm's attn_norm. Only the
                                        # slot<->slot ROUTING uses it; the binding competition stays softmax.
    maxplus_beta_init: float = 1.0
    maxplus_beta_max: float = 10.0

    # --- anti-collapse / stability (all architectural; add NO non-RL training signal) ---
    mu_init_scale: float = 1.0          # stddev of the learnable per-slot init mu_i (>0 => slots
                                        # distinct at init => init symmetry broken stochastically).
    pos_emb: bool = True                # learned positional embedding on the H*W INPUT tokens, so
                                        # binding can localize to board positions. This is a prior on
                                        # the *input* (the conv embed is already spatial); the
                                        # slot<->slot ROUTING graph stays prior-free, which is the
                                        # claim under test.
    pre_norm: bool = True               # RMSNorm slots before q/k/v
    out_init_scale: float = 0.1         # routing output-projection scale (small => routing off at
                                        # init => start as a binding-only slot autoencoder).

    # --- gate (matches ConvLSTM/AttentionCell) ---
    forget_bias: float = 0.0
    output_activation: Literal["sigmoid", "tanh"] = "tanh"


@dataclasses.dataclass(frozen=True)
class SlotLSTMConfig(BaseLSTMConfig):
    # Subclass BaseLSTMConfig (NOT ConvLSTMConfig) so the ConvLSTM-specific fence fix-up in
    # cleanba/cleanba_impala.py (its `isinstance(args.net, ConvLSTMConfig)` branch) is skipped.
    embed: List[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: SlotCellConfig = SlotCellConfig()
    use_relu: bool = True

    def make(self) -> "SlotLSTM":
        return SlotLSTM(self)


# --------------------------------------------------------------------------------------------------
# Core. Reuses BaseLSTM.step / scan / apply_cells_once / _mlp verbatim; overrides _apply_cells only
# to reset slots to the learnable mu (not zero) at episode boundaries.
# --------------------------------------------------------------------------------------------------
class SlotLSTM(BaseLSTM):
    cfg: SlotLSTMConfig

    def setup(self):
        super().setup()  # builds self.dense_list used by BaseLSTM._mlp
        self.conv_list = [
            c.make_conv(kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
            for c in self.cfg.embed
        ]
        self.cell_list = [SlotCell(self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"observations shape must be [batch, h, w, c] but is {x.shape=}"
        for i, conv in enumerate(self.conv_list):
            x = conv(x)
            if self.cfg.use_relu and i < len(self.conv_list) - 1:
                x = nn.relu(x)
        return x

    # NB: no _apply_cells override -- the base zeroes the carry at episode start, which is correct
    # here because the per-slot identity (slot_mu, added each tick inside SlotCell) re-supplies the
    # distinct-slot signal a zeroed carry would otherwise lack. initialize_carry is inherited from
    # BaseLSTM (the SlotCell allocates a (B, N, d) carry).


class SlotCell(nn.RNNCellBase):
    cfg: SlotCellConfig

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        # carry.{c,h}: (B, N, d) slots.  inputs: (B, H, W, C_embed) board features.
        # prev_layer_hidden: (B, N, d); == carry.h for the single-layer config, used only for stacks.
        B, H, W, Ce = inputs.shape
        N, d, nh = self.cfg.num_slots, self.cfg.features, self.cfg.num_heads
        assert d % nh == 0, f"features={d} must be divisible by num_heads={nh}"
        dh = d // nh
        S = H * W

        tokens = inputs.reshape(B, S, Ce)  # the H*W board-feature tokens slots bind to
        if self.cfg.pos_emb:
            tokens = tokens + self.param("slot_pos", nn.initializers.normal(0.02), (S, Ce))[None]

        # Persistent learnable per-slot identity mu_i, added each tick: distinct per slot, so even a
        # zeroed carry (episode start) yields distinct slot queries -> the competition breaks the
        # permutation symmetry and binds slots to states. Realizes the "slots start at mu, not zero"
        # stabilizer as a persistent identity (also keeps slot identity stable across time for the
        # flatten->MLP readout and probing). A CELL param (lazy) -> absent on the param-free
        # initialize_carry path. carry.c (no identity) is what the gate updates; identity re-adds next tick.
        slot_id = self.param("slot_mu", nn.initializers.normal(self.cfg.mu_init_scale), (N, d))
        h = carry.h + slot_id[None]  # (B, N, d): identity-augmented slot state (bind/route source)

        # ----------------------------------------------------------------------------------------
        # BINDING (sigma): slot-attention competition. Slots query the board tokens; the softmax is
        # over SLOTS, so the N slots COMPETE to explain each location -- the anti-collapse
        # normalization that ties (binds) distinct slots to distinct states. Then a per-slot
        # weighted MEAN over its assigned tokens reads that state's local configuration into the slot.
        # ----------------------------------------------------------------------------------------
        if self.cfg.binding == "positional":
            # D1: WHERE each slot reads is a learned, board-INDEPENDENT template, WARM-STARTED to the
            # dense-attention identity (slot i <- board position i) and free to adapt. The logits are a
            # parameter (no board input), so `weights` below is identical on every board -- a stable
            # injection sigma:positions->slots, not re-indexed per task. WHAT each slot reads is still the
            # board content at its positions. (Identity warm-start avoids the degenerate near-uniform init.)
            logits2d = self.param("bind_logits", _identity_logit_init(self.cfg.bind_init_diag), (nh, N, S))
            logits = jnp.broadcast_to(logits2d[None], (B, nh, N, S))                        # (B,nh,N,S) board-indep
            vb = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="bind_v")(tokens)  # (B,S,nh,dh) content
        else:
            hb = nn.RMSNorm(name="bind_norm")(h) if self.cfg.pre_norm else h
            qb = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="bind_q")(hb)      # (B,N,nh,dh)
            kb = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="bind_k")(tokens)  # (B,S,nh,dh)
            vb = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="bind_v")(tokens)  # (B,S,nh,dh)
            logits = jnp.einsum("bnhe,bshe->bhns", qb, kb) * (dh ** -0.5)                   # (B,nh,N,S)
        comp = jax.nn.softmax(logits, axis=2)                                          # over SLOTS (competition)
        weights = comp / (comp.sum(axis=3, keepdims=True) + 1e-8)                       # normalize over tokens
        self.sow("intermediates", "bind_attn", weights)  # (B,nh,N,S): slot->board-position binding map
        m_bind = jnp.einsum("bhns,bshe->bnhe", weights, vb)                             # (B,N,nh,dh)
        m_bind = nn.DenseGeneral(d, axis=(-2, -1), use_bias=False, name="bind_out")(m_bind)  # (B,N,d)

        # ----------------------------------------------------------------------------------------
        # ROUTING (graph N): DENSE slot<->slot self-attention, all-to-all over the N slots, NO mask
        # and NO positional prior. The theory's central claim: this learns to attend slot i -> slot j
        # iff the states they bind are one transition apart, recovering N from content + reward alone.
        # ----------------------------------------------------------------------------------------
        hr = nn.RMSNorm(name="route_norm")(h) if self.cfg.pre_norm else h
        qr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_q")(hr)  # (B,N,nh,dh)
        kr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_k")(hr)  # (B,N,nh,dh)
        vr = nn.DenseGeneral((nh, dh), axis=-1, use_bias=False, name="route_v")(hr)  # (B,N,nh,dh)
        L = jnp.einsum("bnhe,bmhe->bhnm", qr, kr) * (dh ** -0.5)                     # (B,nh,N,N)

        if self.cfg.routing_readout == "softmax":
            w = normalize_attn(self.cfg.routing_norm, L, axis=-1)   # over source slots m; sparse if entmax/sparsemax
            self.sow("intermediates", "route_attn", w)  # (B,nh,N,N): slot<->slot routing graph (recovers N?)
            m_route = jnp.einsum("bhnm,bmhe->bnhe", w, vr)          # (B,N,nh,dh)
        elif self.cfg.routing_readout == "maxplus":
            # mellowmax_m [ L_{n,m} + v_{m,e} ] -- soft Bellman max-plus (value propagates one hop
            # per tick without the geometric attenuation softmax averaging suffers). Separable
            # max-shift impl (no (N,N,dh) tensor), matching cleanba.attn_lstm.
            bm = self.cfg.maxplus_beta_max
            raw0 = math.log(self.cfg.maxplus_beta_init / (bm - self.cfg.maxplus_beta_init))
            beta = bm * nn.sigmoid(self.param("beta", nn.initializers.constant(raw0), (nh,)))[None, :, None, None]
            vh = jnp.transpose(vr, (0, 2, 1, 3))                   # (B,nh,N,dh)
            mL = jax.lax.stop_gradient(L.max(axis=3, keepdims=True))   # (B,nh,N,1)
            mv = jax.lax.stop_gradient(vh.max(axis=2, keepdims=True))  # (B,nh,1,dh)
            a = jnp.exp(beta * (L - mL))                           # (B,nh,N,N) in (0,1]
            wv = jnp.exp(beta * (vh - mv))                         # (B,nh,N,dh) in (0,1]
            agg = jnp.einsum("bhnm,bhme->bhne", a, wv)             # (B,nh,N,dh)
            agg = jnp.maximum(agg, jnp.finfo(agg.dtype).tiny)
            logK = math.log(float(N))                              # dense: every slot is a source
            m_route = (mL + mv + (jnp.log(agg) - logK) / beta).transpose(0, 2, 1, 3)  # (B,N,nh,dh)
        else:
            raise ValueError(f"{self.cfg.routing_readout=}")

        # Small-init output projection: routing starts ~off (binding-only autoencoder), engages slowly.
        m_route = nn.DenseGeneral(
            d, axis=(-2, -1), use_bias=False, name="route_out",
            kernel_init=nn.initializers.variance_scaling(self.cfg.out_init_scale, "fan_in", "truncated_normal"),
        )(m_route)  # (B,N,d)

        # ----------------------------------------------------------------------------------------
        # GATE: the proven LSTM update, fed the two messages (binding read + routing message), the
        # slot analogue of AttentionCell's [in_tok, a].
        # ----------------------------------------------------------------------------------------
        gates = nn.Dense(4 * d, name="gate")(jnp.concatenate([m_bind, m_route], axis=-1))  # (B,N,4d)
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
        # Slots decouple from the spatial dims: carry is (batch, num_slots, features). The zero init
        # here is overwritten by the reset-to-mu on the first (episode_start=True) step.
        shape = (input_shape[0], self.cfg.num_slots, self.cfg.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))

    def num_feature_axes(self) -> int:
        return 2
