"""Mechanistic interpretation of the learnable-slot core (cleanba.slot_lstm).

Tests the paper's thesis on a trained slot agent:
  1. BINDING       -- each free slot locks onto a distinct board square (discovered sigma).
  2. RECOVERY OF N -- slot<->slot routing attention lands on transition-adjacent squares, far above
                      chance, with NO positional prior (the headline claim).
  (per-tick propagation is available via the per-tick tensors returned by recompute_cell.)

The slot cell runs inside an nn.scan, so capture_intermediates can't see its internals (and leaks
tracers through the model's internal vmap-over-time). Instead we:
  * get the real embed tokens by applying the policy's own _maybe_normalize_input_image + the net's
    _compress_input directly (both concrete);
  * RECOMPUTE the K-tick slot cell in plain JAX from the loaded params;
  * VALIDATE that our recomputed FINAL slots equal the model's returned carry[-1].h (get_logits_and_value
    returns (carry, logits, value, metrics)) -- so the extracted attention is faithful, not a reimpl bug.

Usage:
  python -m results.interp_slots --self-test                    # local, random weights: validates recompute
  python -m results.interp_slots --ckpt <cp_dir> --boards 256   # node: trained ckpt, real boards, analyses
"""
from __future__ import annotations

import argparse
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------------------------------
# Faithful recompute of the slot cell (mirrors cleanba/slot_lstm.py SlotCell.__call__ exactly).
# --------------------------------------------------------------------------------------------------
def _rmsnorm(x, scale, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps) * scale


def recompute_cell(cell_p, tokens, K, nh=4):
    """tokens: (B,S,Ce) real embed tokens. cell_p = params[...]['cell_list_0'].
    Returns per-tick slots (K,B,N,d), bind attn (K,B,nh,N,S), route attn (K,B,nh,N,N)."""
    B, S, Ce = tokens.shape
    N, d = cell_p["slot_mu"].shape
    dh = d // nh
    tokens = tokens + cell_p["slot_pos"][None]
    c = jnp.zeros((B, N, d)); h = jnp.zeros((B, N, d))         # post-reset carry (episode start)
    binds, routes, slots, mbinds, mroutes = [], [], [], [], []
    for _ in range(K):
        hsrc = h + cell_p["slot_mu"][None]                     # persistent per-slot identity
        # binding: slot-attention competition (softmax over slots)
        hb = _rmsnorm(hsrc, cell_p["bind_norm"]["scale"])
        qb = jnp.einsum("bni,ihe->bnhe", hb, cell_p["bind_q"]["kernel"])
        kb = jnp.einsum("bsi,ihe->bshe", tokens, cell_p["bind_k"]["kernel"])
        vb = jnp.einsum("bsi,ihe->bshe", tokens, cell_p["bind_v"]["kernel"])
        logits = jnp.einsum("bnhe,bshe->bhns", qb, kb) * (dh ** -0.5)
        comp = jax.nn.softmax(logits, axis=2)
        weights = comp / (comp.sum(3, keepdims=True) + 1e-8)
        m_bind = jnp.einsum("bhns,bshe->bnhe", weights, vb)
        m_bind = jnp.einsum("bnhe,heo->bno", m_bind, cell_p["bind_out"]["kernel"])
        # routing: dense slot<->slot self-attention
        hr = _rmsnorm(hsrc, cell_p["route_norm"]["scale"])
        qr = jnp.einsum("bni,ihe->bnhe", hr, cell_p["route_q"]["kernel"])
        kr = jnp.einsum("bni,ihe->bnhe", hr, cell_p["route_k"]["kernel"])
        vr = jnp.einsum("bni,ihe->bnhe", hr, cell_p["route_v"]["kernel"])
        L = jnp.einsum("bnhe,bmhe->bhnm", qr, kr) * (dh ** -0.5)
        w = jax.nn.softmax(L, axis=-1)
        m_route = jnp.einsum("bhnm,bmhe->bnhe", w, vr)
        m_route = jnp.einsum("bnhe,heo->bno", m_route, cell_p["route_out"]["kernel"])
        # gate
        gates = jnp.concatenate([m_bind, m_route], -1) @ cell_p["gate"]["kernel"] + cell_p["gate"]["bias"]
        gi, gj, gf, go = jnp.split(gates, 4, -1)
        c = c * jax.nn.sigmoid(gf) + jnp.tanh(gi) * jax.nn.sigmoid(gj)
        h = jnp.tanh(c) * jnp.tanh(go)
        binds.append(weights); routes.append(w); slots.append(h)
        mbinds.append(m_bind); mroutes.append(m_route)
    return dict(slots=jnp.stack(slots), bind=jnp.stack(binds), route=jnp.stack(routes),
                mbind=jnp.stack(mbinds), mroute=jnp.stack(mroutes))


def _tokens(net, policy, params, obs_single):
    """obs_single (B,3,10,10) -> real embed tokens (B,S,Ce), via the model's own preprocess+embed."""
    x = policy.apply({}, obs_single, method="_maybe_normalize_input_image")        # (B,H,W,3) concrete
    netp = {"params": params["params"]["network_params"]}
    tok = net.make().apply(netp, x, method="_compress_input")                      # (B,H,W,C)
    B, H, W, C = tok.shape
    return tok.reshape(B, H * W, C), (H, W)


def validate_and_extract(net, policy, params, carry, obs, eps, K):
    """Real forward -> carry[-1].h (final slots). Recompute -> slots[-1]. Return max abs diff + tensors."""
    carry_out, logits, value, _ = policy.apply(params, carry, obs, eps, method=policy.get_logits_and_value)
    tokens, (H, W) = _tokens(net, policy, params, obs[0])
    rc = recompute_cell(params["params"]["network_params"]["cell_list_0"], tokens, K)
    err = float(jnp.max(jnp.abs(rc["slots"][-1] - carry_out[-1].h)))
    return err, rc, (H, W)


# --------------------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------------------
def grid_adjacency(H, W, king=False):
    idx = np.arange(H * W); r, c = idx // W, idx % W
    dr = np.abs(r[:, None] - r[None, :]); dc = np.abs(c[:, None] - c[None, :])
    adj = (np.maximum(dr, dc) <= 1) if king else ((dr + dc) <= 1)
    np.fill_diagonal(adj, False)
    return adj


def analyse(rc, H, W, king=False, rng_seed=0):
    bind = np.asarray(rc["bind"][-1].mean(1))    # (B,N,S) slot -> square (final tick, head-avg)
    route = np.asarray(rc["route"][-1].mean(1))  # (B,N,N) slot -> slot
    B, N, S = bind.shape
    adj = grid_adjacency(H, W, king)
    sq = bind.argmax(-1)                          # (B,N) bound square per slot
    p = np.clip(bind, 1e-12, 1)
    bind_entropy = float((-(p * np.log(p)).sum(-1)).mean())
    bind_peak = float(bind.max(-1).mean())
    distinct = float(np.mean([len(np.unique(sq[b])) for b in range(B)]))
    on, rand = [], []
    rng = np.random.default_rng(rng_seed)
    for b in range(B):
        a = adj[np.ix_(sq[b], sq[b])].copy(); np.fill_diagonal(a, False)
        mass = route[b].copy(); np.fill_diagonal(mass, 0.0)
        denom = mass.sum() + 1e-12
        on.append(mass[a].sum() / denom)
        perm = rng.permutation(N)
        ar = adj[np.ix_(sq[b][perm], sq[b][perm])].copy(); np.fill_diagonal(ar, False)
        rand.append(mass[ar].sum() / denom)
    # ---- diagnostics: is routing sharp? is it used? does any single head route spatially? ----
    rp = np.clip(route, 1e-12, 1); route_entropy = float((-(rp * np.log(rp)).sum(-1)).mean())
    mbind_mag = float(np.abs(np.asarray(rc["mbind"])).mean())
    mroute_mag = float(np.abs(np.asarray(rc["mroute"])).mean())
    bind_h = np.asarray(rc["bind"][-1]); route_h = np.asarray(rc["route"][-1])   # (B,nh,N,S),(B,nh,N,N)
    nh = bind_h.shape[1]; head_lifts = []
    for hd in range(nh):
        sqh = bind_h[:, hd].argmax(-1)
        oh, rh = [], []
        for b in range(B):
            a = adj[np.ix_(sqh[b], sqh[b])].copy(); np.fill_diagonal(a, False)
            mass = route_h[b, hd].copy(); np.fill_diagonal(mass, 0.0); dn = mass.sum() + 1e-12
            oh.append(mass[a].sum() / dn)
            pm = rng.permutation(N); ar = adj[np.ix_(sqh[b][pm], sqh[b][pm])].copy(); np.fill_diagonal(ar, False)
            rh.append(mass[ar].sum() / dn)
        head_lifts.append(float(np.mean(oh) / (np.mean(rh) + 1e-12)))
    return dict(n_boards=B, n_slots=N, bind_peak=bind_peak, bind_entropy=bind_entropy,
                distinct_squares=distinct, route_on_graph=float(np.mean(on)),
                route_on_graph_rand=float(np.mean(rand)),
                lift=float(np.mean(on) / (np.mean(rand) + 1e-12)),
                route_entropy=route_entropy, mbind_mag=mbind_mag, mroute_mag=mroute_mag,
                max_head_lift=float(max(head_lifts)), head_lifts=head_lifts)


_TILE_NAMES = ["wall", "floor", "box", "target", "agent"]


def decode_tiles(obs_bchw):
    """obs (B,3,10,10) gym-sokoban tinyworld RGB -> (B,100) tile categories (0..4)."""
    B = obs_bchw.shape[0]
    px = np.asarray(obs_bchw).transpose(0, 2, 3, 1).reshape(B, 100, 3)
    cat = np.full((B, 100), 1, dtype=int)                    # default floor
    m = lambda c: np.all(px == np.array(c), axis=-1)
    cat[m((0, 0, 0))] = 0                                     # wall
    cat[m((142, 121, 56)) | m((254, 95, 56))] = 2            # box / box-on-target
    cat[m((254, 126, 125))] = 3                              # target
    cat[m((160, 212, 56)) | m((219, 212, 56))] = 4          # agent / agent-on-target
    return cat


def analyse_objects(rc, tiles):
    """Are the slots' bound squares enriched for task objects (box/target/agent)? Do slots specialize?"""
    bind = np.asarray(rc["bind"][-1].mean(1))                # (B,N,S)
    B, N, S = bind.shape
    sq = bind.argmax(-1)                                      # (B,N)
    bound = np.take_along_axis(tiles, sq, axis=1)            # (B,N) tile type of each slot's bound square
    bound_dist = np.array([(bound == t).mean() for t in range(5)])
    board_dist = np.array([(tiles == t).mean() for t in range(5)])
    obj = float(np.isin(bound, [2, 3, 4]).mean())
    obj_chance = float(np.isin(tiles, [2, 3, 4]).mean())
    cons = [np.bincount(bound[:, n], minlength=5).max() / B for n in range(N)]   # per-slot type consistency
    return dict(bound_dist=bound_dist, board_dist=board_dist, obj=obj, obj_chance=obj_chance,
                obj_enrich=obj / (obj_chance + 1e-9), mean_consistency=float(np.mean(cons)))


def report_dig(obj, agree, dyn):
    print("  -- OBJECT BINDING (what do slots bind?) --")
    print("    tile        bound%   board%(chance)")
    for i, nm in enumerate(_TILE_NAMES):
        print(f"    {nm:<10}  {obj['bound_dist'][i]*100:5.1f}    {obj['board_dist'][i]*100:5.1f}")
    print(f"    P(bound is object box/target/agent) : {obj['obj']:.3f}  vs chance {obj['obj_chance']:.3f}  -> {obj['obj_enrich']:.1f}x")
    print(f"    per-slot tile-type consistency      : {obj['mean_consistency']:.2f}  (1.0 = slot always binds same type)")
    print("  -- IS ROUTING USED BEHAVIORALLY? (zero route_out -> action change) --")
    print(f"    action agreement full vs no-routing : {agree:.3f}  (1.0 = routing irrelevant to policy)")
    print(f"    per-tick relative slot change       : {['%.2f' % x for x in dyn]}  (loop activity)")
    print("=========================================================\n")


def report(stats, K, val_err, king):
    nb = stats["n_slots"]
    print("\n==================== SLOT MECH-INTERP ====================")
    print(f"  recompute fidelity: max|mine - model carry| = {val_err:.2e}  ({'OK' if val_err < 1e-2 else 'CHECK!'})")
    print(f"  boards={stats['n_boards']}  slots={nb}  ticks={K}  graph={'king(8)' if king else 'vonNeumann(4)'}")
    print("  -- BINDING (slots -> board squares) --")
    print(f"    mean top-1 binding weight : {stats['bind_peak']:.3f}    (1.0 = slot fixates one square)")
    print(f"    binding entropy           : {stats['bind_entropy']:.2f} nats   (0 sharp; ln{nb}={np.log(nb):.2f} uniform)")
    print(f"    distinct squares / {nb}    : {stats['distinct_squares']:.1f}     (high = injective)")
    print("  -- RECOVERY OF N (routing vs grid transition graph) --")
    print(f"    routing mass on adjacent squares : {stats['route_on_graph']:.3f}")
    print(f"    random slot->square baseline     : {stats['route_on_graph_rand']:.3f}")
    print(f"    lift over chance (head-avg)      : {stats['lift']:.2f}x")
    print(f"    lift over chance (best head)     : {stats['max_head_lift']:.2f}x   heads={['%.2f' % x for x in stats['head_lifts']]}")
    print("  -- DIAGNOSTICS (is routing sharp / used?) --")
    print(f"    routing entropy           : {stats['route_entropy']:.2f} nats  (ln{nb}={np.log(nb):.2f} uniform; low=sharp)")
    print(f"    |m_route| / |m_bind|      : {stats['mroute_mag'] / (stats['mbind_mag'] + 1e-12):.3f}   (routing vs binding drive to the gate)")
    print("=========================================================\n")


# --------------------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------------------
def dig(policy, params, carry, obs, eps, rc):
    """Object-binding + behavioral routing ablation + per-tick loop activity."""
    obj = analyse_objects(rc, decode_tiles(np.asarray(obs[0])))
    from flax.core import unfreeze
    pa = unfreeze(params)
    ro = pa["params"]["network_params"]["cell_list_0"]["route_out"]["kernel"]
    pa["params"]["network_params"]["cell_list_0"]["route_out"]["kernel"] = jnp.zeros_like(ro)
    _, lf, _, _ = policy.apply(params, carry, obs, eps, method=policy.get_logits_and_value)
    _, la, _, _ = policy.apply(pa, carry, obs, eps, method=policy.get_logits_and_value)
    agree = float((np.asarray(lf).argmax(-1) == np.asarray(la).argmax(-1)).mean())
    sl = np.asarray(rc["slots"])
    dyn = [float(np.abs(sl[k] - sl[k - 1]).mean() / (np.abs(sl[k - 1]).mean() + 1e-9)) for k in range(1, len(sl))]
    report_dig(obj, agree, dyn)


def _cfg():
    from cleanba.config import sokoban_drc_slots_softmax
    return sokoban_drc_slots_softmax()


def self_test(B=8):
    import gymnasium as gym
    args = _cfg(); net = args.net; K = net.repeats_per_step

    class FakeEnvs:
        single_observation_space = gym.spaces.Box(0, 255, (3, 10, 10), np.uint8)
        observation_space = gym.spaces.Box(0, 255, (B, 3, 10, 10), np.uint8)
        single_action_space = gym.spaces.Discrete(4)
        action_space = gym.spaces.MultiDiscrete([4] * B)

    policy, carry, params = net.init_params(FakeEnvs(), jax.random.PRNGKey(0))
    obs = jax.random.randint(jax.random.PRNGKey(1), (1, B, 3, 10, 10), 0, 256).astype(jnp.uint8)
    eps = jnp.ones((1, B), dtype=bool)
    err, rc, (H, W) = validate_and_extract(net, policy, params, carry, obs, eps, K)
    stats = analyse(rc, H, W)
    report(stats, K, err, king=False)
    dig(policy, params, carry, obs, eps, rc)
    assert err < 1e-2, f"recompute INFIDELITY {err}: cell reimpl diverges from the model"
    print("SELF-TEST PASS: recompute faithful; analyses run. (numbers are random-weight noise.)")


def run_ckpt(cp_dir, n_boards, king=False):
    from pathlib import Path
    from cleanba.cleanba_impala import load_train_state
    args = _cfg()
    eval_cfg = dataclasses.replace(args.eval_envs["valid_medium"].env, num_envs=n_boards)
    envs = eval_cfg.make()
    obs_np, _ = envs.reset()
    obs = jnp.asarray(np.asarray(obs_np))[None]              # (1,B,3,10,10)
    B = obs.shape[1]; eps = jnp.ones((1, B), dtype=bool)
    # load_train_state -> (Policy, carry, Args, TrainState, step); gives the ckpt's own policy + config
    policy, _carry0, loaded_args, ts, step = load_train_state(Path(cp_dir), args.train_env)
    net = loaded_args.net; K = net.repeats_per_step
    params = ts.params
    carry = policy.apply({}, jax.random.PRNGKey(0), obs.shape[1:], method=policy.initialize_carry)
    err, rc, (H, W) = validate_and_extract(net, policy, params, carry, obs, eps, K)
    print(f"loaded checkpoint @ update_step={step}")
    stats = analyse(rc, H, W, king=king)
    report(stats, K, err, king=king)
    dig(policy, params, carry, obs, eps, rc)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--boards", type=int, default=256)
    ap.add_argument("--king", action="store_true", help="use 8-neighbour graph instead of 4")
    a = ap.parse_args()
    if a.self_test or not a.ckpt:
        self_test()
    else:
        run_ckpt(a.ckpt, a.boards, king=a.king)
