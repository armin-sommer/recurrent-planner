"""Slot-core interp foundation: per-tick hidden + the LEARNED binding sigma.

The E1/E4/E5/E10/E13 probes were written for the spatial *attention* core (recompute_d3, cells == board
squares). The *slot* core (cleanba.slot_lstm) is different: N FREE slots, a recurrence of slot-attention
binding (slots <- board tokens) + dense slot<->slot routing, and the binding sigma (slot <-> board state)
is LEARNED, not the grid identity. This module supplies the two things every slot probe needs:

  slot_per_tick(policy, params, obs, K)  -> per-tick top-slot hidden + sown bind_attn / route_attn,
       run as a manual apply_cells_once loop (the model's own cell code, no scan; mirrors the DRC probes).
  decode_sigma(bind_attn)               -> for each slot, the board position it binds (argmax over the
       slot-attention map), i.e. the recovered sigma -- replaces the attn core's "cell i == square i".

With (per-tick hidden indexed by slot) + (sigma: slot -> position) the spatial probes port directly:
work in slot space, then map slots to board positions via sigma to compute BFS graph distance etc.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from experiments.interp.planning import get_embed  # conv embed works for the slot core too (same embed stack)


def slot_per_tick(policy, params, obs_bchw, K):
    """Run the slot recurrence for K thinking ticks from a fresh (episode-start) carry, capturing per
    tick the top-slot hidden and the deepest cell's binding/routing attention.

    Returns:
      h     (K, B, N, d)        top-layer slot hidden after each tick
      bind  (K, B, nh, N, S)    slot-attention binding map (slot -> board position) of the deepest cell
      route (K, B, nh, N, N)    slot<->slot routing map of the deepest cell
    """
    emb = jnp.asarray(get_embed(policy, params, obs_bchw))            # (B,H,W,C)
    carry = policy.apply(params, jax.random.PRNGKey(0), obs_bchw.shape, method=policy.initialize_carry)

    def once(m, carry, e):
        return m.network_params.apply_cells_once(carry, e)

    D = len(carry)                                                    # number of stacked cells
    last = f"cell_list_{D - 1}"
    hs, binds, routes = [], [], []
    for _ in range(K):
        (carry, _), mut = policy.apply(params, carry, emb, method=once, mutable=["intermediates"])
        inter = mut["intermediates"]["network_params"][last]
        hs.append(np.asarray(carry[-1].h))                           # (B,N,d)
        binds.append(np.asarray(inter["bind_attn"][0]))             # (B,nh,N,S)
        routes.append(np.asarray(inter["route_attn"][0]))           # (B,nh,N,N)
    return np.stack(hs), np.stack(binds), np.stack(routes)


def decode_sigma(bind, tick=-1):
    """Recover the binding sigma from the slot-attention map at one tick.

    bind: (K,B,nh,N,S) or (B,nh,N,S). Head-average, then for each slot take argmax over the S board
    positions -> the position that slot reads from. Returns:
      pos   (B, N)  slot -> board-position index (0..S-1)
      mass  (B, N)  the top binding mass (confidence) for that slot
    """
    b = bind[tick] if bind.ndim == 5 else bind                       # (B,nh,N,S)
    w = b.mean(1)                                                     # head-avg (B,N,S)
    pos = w.argmax(-1)                                               # (B,N)
    mass = np.take_along_axis(w, pos[..., None], axis=-1)[..., 0]    # (B,N)
    return pos, mass


def sigma_quality(pos, tiles, S):
    """Sanity stats on a decoded sigma (pos: (B,N), tiles: (B,S) decoded board): mean fraction of slots
    binding DISTINCT positions (injectivity), and how many board positions get no slot (coverage)."""
    B, N = pos.shape
    inj, cov = [], []
    for b in range(B):
        u = np.unique(pos[b])
        inj.append(len(u) / N)                                       # 1.0 == perfectly injective
        cov.append(len(u) / S)
    return float(np.mean(inj)), float(np.mean(cov))
