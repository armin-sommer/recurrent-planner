"""E8 -- do WALL cells carry "dead" value content? (does the obstacle-respect ride on v, not on routing A?)

E7b showed the attention operator A does NOT re-route around a new wall (mass onto a cell ~unchanged when
it becomes a wall). Yet through-walls (interp_wall_d3) shows walling a cell DOES change the propagated
latent. Hypothesis: the obstacle-respect is carried by the VALUE projection v (z=(A v)W_out), not the
routing -- a wall cell's v is "dead", so even though A still attends to it, it transmits ~no value.

We replicate the validated recompute and extract the top cell's value projection v (B,S,nh,dh), then:
  (1) NATURAL boards: mean ||v(s)|| by tile type (wall vs floor/box/target/agent) -- are walls dead?
  (2) FLIP a floor cell X -> wall: ||v(X)|| as floor vs as wall -- does it go dead causally?
Self-check: our replicated top hidden matches recompute_d3 to ~0.

  python -m results.interp_e8_d3 --ckpt <cp_dir> --boards 160
"""
from __future__ import annotations
import argparse, dataclasses
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from results.interp_planning_d3 import recompute_d3, get_embed, _rmsnorm, _rel_offset_index
from results.interp_slots import decode_tiles
from results.interp_plan import bfs_from, WALL, FLOOR, BOX, TARGET, AGENT
from cleanba.entmax import entmax15

WALL_RGB = np.array([0, 0, 0], np.uint8)
_TN = {WALL: "wall", FLOOR: "floor", BOX: "box", TARGET: "target", AGENT: "agent"}


def recompute_topv(cell_ps, embed, K, nh=4):
    """Replicate recompute_d3 and also return the top cell's value projection v per tick."""
    B, H, W, C = embed.shape; S = H * W; dh = C // nh; Dn = len(cell_ps)
    embed_tok = embed.reshape(B, S, C)
    rbs = [jnp.asarray(cp["rel_bias"])[:, _rel_offset_index(H, W)] for cp in cell_ps]
    cs = [jnp.zeros((B, S, C)) for _ in cell_ps]; hs = [jnp.zeros((B, S, C)) for _ in cell_ps]
    top_h, top_v = [], []
    for _ in range(K):
        prev = hs[-1]; cur_v = None
        for d, cp in enumerate(cell_ps):
            in_tok = jnp.concatenate([embed_tok, prev], axis=-1)
            h_tok = hs[d] + (in_tok @ cp["in_proj"]["kernel"] + cp["in_proj"]["bias"])
            h_tok = _rmsnorm(h_tok, cp["pre_norm"]["scale"])
            q = jnp.einsum("bsc,cnd->bsnd", h_tok, cp["q"]["kernel"])
            k = jnp.einsum("bsc,cnd->bsnd", h_tok, cp["k"]["kernel"])
            v = jnp.einsum("bsc,cnd->bsnd", h_tok, cp["v"]["kernel"])
            logits = jnp.einsum("bsnd,bknd->bnsk", q, k) * (dh ** -0.5) + rbs[d][None]
            w = entmax15(logits, axis=-1)
            attn = jnp.einsum("bnsk,bknd->bsnd", w, v)
            a = jnp.einsum("bsnd,ndo->bso", attn, cp["out"]["kernel"])
            gates = jnp.concatenate([in_tok, a], -1) @ cp["gate"]["kernel"] + cp["gate"]["bias"]
            gi, gj, gf, go = jnp.split(gates, 4, -1)
            cs[d] = cs[d] * jax.nn.sigmoid(gf) + jnp.tanh(gi) * jax.nn.sigmoid(gj)
            hs[d] = jnp.tanh(cs[d]) * jnp.tanh(go)
            prev = hs[d]
            if d == Dn - 1:
                cur_v = v
        top_h.append(hs[-1]); top_v.append(cur_v)
    return jnp.stack(top_h), jnp.stack(top_v)                                          # (K,B,S,C), (K,B,S,nh,dh)


def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy, _, cp_cfg, ts, step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params = ts.params; net = cp_cfg.net; K = net.repeats_per_step; D = net.n_recurrent
    cps = [params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]

    obs0 = np.asarray(env_cfg.make().reset()[0]); B, _, H, W = obs0.shape; S = H * W
    tiles = decode_tiles(obs0); RR, CC = np.arange(S) // W, np.arange(S) % W
    emb0 = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs0))))
    th, tv = recompute_topv(cps, emb0, K)
    err = float(jnp.max(jnp.abs(th[-1] - recompute_d3(cps, emb0, K)[0][-1])))           # fidelity self-check
    vnorm = np.asarray(jnp.linalg.norm(tv[-1], axis=(-2, -1)))                          # (B,S) ||v(s)|| over heads*dh

    # (1) natural boards: ||v|| by tile type
    print(f"\n===== E8: VALUE-PROJECTION CONTENT (step={step}, boards={B}, recompute self-test max|diff|={err:.1e}) =====")
    print(f"  (1) mean ||v(s)|| by tile type (natural boards):")
    base = {}
    for t, nm in _TN.items():
        m = tiles == t
        if m.any():
            base[nm] = float(vnorm[m].mean())
            print(f"      {nm:<7} {base[nm]:.3f}   (n={int(m.sum())})")
    if "wall" in base and "floor" in base:
        print(f"      wall/floor ratio = {base['wall']/base['floor']:.2f}   ({'walls DEAD (low v)' if base['wall'] < 0.7*base['floor'] else 'walls NOT low-norm' if base['wall'] > 0.9*base['floor'] else 'walls somewhat lower'})")

    # (2) flip a floor cell X -> wall, compare ||v(X)||
    obs_w = obs0.copy(); X = np.full(B, -1)
    for b in range(B):
        ag = np.where(tiles[b] == AGENT)[0]
        if not len(ag):
            continue
        dist = bfs_from([int(ag[0])], tiles[b], H, W); fl = np.where(tiles[b] == FLOOR)[0]
        fl = fl[np.isfinite(dist[fl]) & (dist[fl] >= 2) & (dist[fl] <= 6)]
        if not len(fl):
            continue
        x = int(fl[len(fl) // 2]); X[b] = x; obs_w[b, :, RR[x], CC[x]] = WALL_RGB
    tg_w = decode_tiles(obs_w); okx = np.array([X[b] >= 0 and tg_w[b, X[b]] == WALL for b in range(B)])
    embw = jnp.asarray(np.asarray(get_embed(policy, params, jnp.asarray(obs_w))))
    _, tvw = recompute_topv(cps, embw, K)
    vnorm_w = np.asarray(jnp.linalg.norm(tvw[-1], axis=(-2, -1)))
    vf = [vnorm[b, X[b]] for b in range(B) if okx[b]]; vw = [vnorm_w[b, X[b]] for b in range(B) if okx[b]]
    vf, vw = float(np.mean(vf)), float(np.mean(vw))
    print(f"  (2) flip a floor cell X -> wall (n={int(okx.sum())}):")
    print(f"      ||v(X)|| as floor = {vf:.3f}  ->  as wall = {vw:.3f}   ratio {vw/(vf+1e-9):.2f}")
    print(f"      => {'value content GOES DEAD when the cell becomes a wall (obstacle-respect via v, not A)' if vw < 0.7*vf else 'value content does NOT go dead -> mechanism is elsewhere (gates/direction)'}")
    print("PLOT_E8=" + repr(dict(by_tile={k: round(v, 4) for k, v in base.items()}, flip=[round(vf, 4), round(vw, 4)])))
    print("=" * 88 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--boards", type=int, default=160)
    a = ap.parse_args(); main(a.ckpt, a.boards)
