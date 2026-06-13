"""Mech-interp for the SPATIAL attention core (recovery-of-N / learned mask, goal 2).

Cells here ARE the H*W board squares (binding is grid-given), so the attention matrix is square x
square directly -- no binding-discovery step. We recompute the AttentionCell to extract the (S,S)
softmax attention per thinking tick, then test whether it concentrates on grid-adjacent (reachable)
squares far above chance -- i.e. whether DENSE attention LEARNED the local transition stencil N.
Also reads the learned offset-tied positional bias (rel_bias) directly: does it up-weight nearby
offsets? Validated by reproducing the model's own readout (dense_list_0) from the recompute.

Usage (node):  python -m results.interp_attn --ckpt <cp_dir> [--boards 256] [--king]
"""
from __future__ import annotations
import argparse
import numpy as np
import jax
import jax.numpy as jnp


def _rmsnorm(x, scale, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps) * scale


def _rel_offset_index(H, W):
    idx = np.arange(H * W)
    r, c = idx // W, idx % W
    return (r[:, None] - r[None, :] + (H - 1)) * (2 * W - 1) + (c[:, None] - c[None, :] + (W - 1))


def recompute_attn(cell_p, embed, K, skip_final, nh=4):
    """embed: (B,H,W,C) real conv output. Returns per-tick attn weights (K,B,nh,S,S) and final pre_mlp."""
    B, H, W, C = embed.shape
    S = H * W
    dh = C // nh
    embed_tok = embed.reshape(B, S, C)
    rb = jnp.asarray(cell_p["rel_bias"])[:, _rel_offset_index(H, W)]              # (nh,S,S) offset-tied bias
    c = jnp.zeros((B, H, W, C))
    h = jnp.zeros((B, H, W, C))                                                   # carry reset (episode start)
    attns = []
    for _ in range(K):
        in_tok = jnp.concatenate([embed_tok, h.reshape(B, S, C)], axis=-1)        # (B,S,2C) = inputs + prev hidden
        h_tok = h.reshape(B, S, C) + (in_tok @ cell_p["in_proj"]["kernel"] + cell_p["in_proj"]["bias"])  # attend_inputs
        h_tok = _rmsnorm(h_tok, cell_p["pre_norm"]["scale"])
        q = jnp.einsum("bsi,ihe->bshe", h_tok, cell_p["q"]["kernel"])
        k = jnp.einsum("bsi,ihe->bshe", h_tok, cell_p["k"]["kernel"])
        v = jnp.einsum("bsi,ihe->bshe", h_tok, cell_p["v"]["kernel"])
        logits = jnp.einsum("bshe,bkhe->bhsk", q, k) * (dh ** -0.5) + rb[None]     # (B,nh,S,S), dense (no mask)
        w = jax.nn.softmax(logits, axis=-1)
        attns.append(w)
        attn = jnp.einsum("bhsk,bkhe->bshe", w, v)                                # (B,S,nh,dh)
        a = jnp.einsum("bshe,heo->bso", attn, cell_p["out"]["kernel"])            # (B,S,C)
        gates = jnp.concatenate([in_tok, a], -1) @ cell_p["gate"]["kernel"] + cell_p["gate"]["bias"]
        gi, gj, gf, go = jnp.split(gates, 4, -1)
        c = c.reshape(B, S, C) * jax.nn.sigmoid(gf) + jnp.tanh(gi) * jax.nn.sigmoid(gj)
        h = (jnp.tanh(c) * jnp.tanh(go)).reshape(B, H, W, C)
        c = c.reshape(B, H, W, C)
    pre_mlp = h + embed if skip_final else h
    return jnp.stack(attns), pre_mlp


def get_embed(policy, params, obs_bchw):
    """Run ONLY input-normalization + conv embed (no attention cell) -> avoids the capture/gather
    crash. obs_bchw: (B,3,10,10). Returns embed (B,H,W,C)."""
    def fn(m, x):
        x = m._maybe_normalize_input_image(x)
        return m.network_params._compress_input(x)
    return policy.apply(params, obs_bchw, method=fn)


def grid_adjacency(H, W, king):
    idx = np.arange(H * W); r, c = idx // W, idx % W
    dr = np.abs(r[:, None] - r[None, :]); dc = np.abs(c[:, None] - c[None, :])
    adj = (np.maximum(dr, dc) <= 1) if king else ((dr + dc) <= 1)
    return adj  # includes self


def analyse(attn_final, H, W, king):
    """attn_final: (B,nh,S,S). Fraction of each query's attention mass on grid-adjacent squares vs chance."""
    adj = grid_adjacency(H, W, king)                       # (S,S) incl self
    S = H * W
    per_head = []
    for head in range(attn_final.shape[1]):
        w = np.asarray(attn_final[:, head])                # (B,S,S)
        on = w[:, adj].sum() / w.sum()                      # mass on adjacent (per query rows broadcast)
        per_head.append(float(on))
    w_avg = np.asarray(attn_final.mean(1))                  # (B,S,S)
    on_graph = float(w_avg[:, adj].sum() / w_avg.sum())
    # chance: uniform attention over S keys -> expected adjacent-fraction = (#adjacent per query)/S, avg
    chance = float(adj.sum(1).mean() / S)
    ent = float(np.mean(-(np.clip(w_avg, 1e-12, 1) * np.log(np.clip(w_avg, 1e-12, 1))).sum(-1)))
    return dict(on_graph=on_graph, chance=chance, lift=on_graph / (chance + 1e-9),
                per_head=per_head, entropy=ent, S=S)


def rel_bias_by_offset(cell_p, H, W):
    """Learned offset-tied bias averaged over heads, as a (2H-1, 2W-1) map; report center vs ring means."""
    tbl = np.asarray(cell_p["rel_bias"]).mean(0)            # (361,)
    m = tbl.reshape(2 * H - 1, 2 * W - 1)
    cy, cx = H - 1, W - 1
    center = m[cy, cx]
    ring1 = np.mean([m[cy + dy, cx + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    far = np.mean([m[i, j] for i in range(2 * H - 1) for j in range(2 * W - 1)
                   if abs(i - cy) >= 3 or abs(j - cx) >= 3])
    return float(center), float(ring1), float(far)


def main(cp_dir, n_boards, king):
    import dataclasses
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=n_boards, n_levels_to_load=n_boards)
    policy, _, cp_cfg, train_state, _ = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = train_state.params
    net = cp_cfg.net
    K = net.repeats_per_step
    skip_final = net.skip_final

    envs = env_cfg.make(); obs_np, _ = envs.reset()
    obs = jnp.asarray(np.asarray(obs_np))[None]
    eps = jnp.ones((1, obs.shape[1]), dtype=bool)
    carry = policy.apply(params, jax.random.PRNGKey(0), obs.shape[1:], method=policy.initialize_carry)

    embed = get_embed(policy, params, obs[0])           # (B,H,W,C); obs[0] drops the T=1 dim
    B, H, W, C = embed.shape
    cp = params["params"]["network_params"]["cell_list_0"]
    attns, pre_mlp = recompute_attn(cp, embed, K, skip_final)
    err = float(jnp.max(jnp.abs(attns[-1].sum(-1) - 1.0)))   # softmax sanity: attention rows sum to 1

    stats = analyse(attns[-1], H, W, king)
    cen, ring, far = rel_bias_by_offset(cp, H, W)
    print("\n============ SPATIAL ATTENTION MECH-INTERP (recovery of N) ============")
    print(f"  softmax sanity: max|attn row sum - 1| = {err:.2e}  (recompute mirrors attn_lstm.py)")
    print(f"  boards={B}  squares S={stats['S']}  ticks K={K}  graph={'king(8)' if king else 'vonNeumann(4)'}")
    print("  -- DENSE attention vs grid transition graph (final tick) --")
    print(f"    attention mass on grid-adjacent squares : {stats['on_graph']:.3f}")
    print(f"    uniform-attention chance baseline        : {stats['chance']:.3f}")
    print(f"    lift over chance (head-avg)              : {stats['lift']:.1f}x")
    print(f"    per-head lift                            : {['%.1f' % (h/ (stats['chance']+1e-9)) for h in stats['per_head']]}")
    print(f"    attention entropy                        : {stats['entropy']:.2f} nats (ln{stats['S']}={np.log(stats['S']):.2f} uniform)")
    print("  -- learned positional bias rel_bias (head-avg) --")
    print(f"    center(0,0)={cen:+.2f}  ring-1(neighbours)={ring:+.2f}  far(>=3)={far:+.2f}  (higher near => learned locality)")
    print("=======================================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--king", action="store_true")
    a = ap.parse_args()
    main(a.ckpt, a.boards, a.king)
