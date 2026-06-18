"""DRC(3,3) propagation/contraction probe (e1.py + e5.py analogs, model-faithful, no value proxy).
E1: per-tick hidden change ||h_t - h_{t-1}|| over ticks -> does the loop contract toward a fixed point?
E5: ablate ONE cell's input embedding; influence ||dh_t(s)|| on other cells by graph-distance from it,
    per tick -> does the influenced radius GROW with ticks (local wavefront / propagation), or is it
    global from tick 1 (pool-and-inject)? Onset = first tick a band reaches 50% of its settled influence."""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, WALL, FLOOR
BANDS=[(0,1),(2,3),(4,5),(6,7),(8,99)]
def main(nb,T):
    import eval_pretrained as ep
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    cpdir=ep.make_clean_checkpoint(ep.resolve_checkpoint_dir())
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=nb, n_levels_to_load=nb, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(cpdir, env_cfg=env_cfg)
    params=ts.params
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W
    tiles=decode_tiles(obs); emb=np.asarray(get_embed(policy,params,jnp.asarray(obs))); C=emb.shape[-1]
    def once(m,carry,e): return m.network_params.apply_cells_once(carry,e)
    def runticks(embj):
        carry=policy.apply(params, jax.random.PRNGKey(0), obs.shape, method=policy.initialize_carry); Hs=[]
        for t in range(T): carry,_=policy.apply(params, carry, embj, method=once); Hs.append(np.asarray(carry[-1].h).reshape(B,S,C))
        return np.stack(Hs)
    Hb=runticks(jnp.asarray(emb))
    # ---- E1: contraction ----
    delta=[float(np.linalg.norm(Hb[t]-Hb[t-1],axis=-1).mean()) for t in range(1,T)]
    hn=[float(np.linalg.norm(Hb[t],axis=-1).mean()) for t in range(T)]
    ratio=[delta[i]/delta[i-1] if i>0 and delta[i-1]>1e-9 else float('nan') for i in range(len(delta))]
    # ---- E5: ablate one floor cell's input per board; influence by graph-distance ----
    embf=emb.reshape(B,S,C).copy(); rng=np.random.default_rng(0); cper=np.full(B,-1)
    for b in range(B):
        fl=np.where(tiles[b]==FLOOR)[0]
        if len(fl): c=int(rng.choice(fl)); cper[b]=c; embf[b,c,:]=0.0
    Hp=runticks(jnp.asarray(embf.reshape(B,H,W,C)))
    infl=np.linalg.norm(Hp-Hb,axis=-1)                       # (T,B,S)
    dist=np.full((B,S),np.nan)
    for b in range(B):
        if cper[b]>=0: dist[b]=bfs_from([int(cper[b])],tiles[b],H,W)
    Ib=np.full((T,len(BANDS)),np.nan)
    for j,(lo,hi) in enumerate(BANDS):
        mask=np.zeros((B,S),bool)
        for b in range(B):
            if cper[b]>=0: mask[b]=(dist[b]>=lo)&(dist[b]<=hi)
        if mask.sum()>0:
            for t in range(T): Ib[t,j]=infl[t][mask].mean()
    # onset: first tick each band reaches 50% of its settled (tick-T) influence
    onset=[]
    for j in range(len(BANDS)):
        col=Ib[:,j]; 
        if not np.isfinite(col[-1]) or col[-1]<=0: onset.append(-1); continue
        hit=np.where(col>=0.5*col[-1])[0]; onset.append(int(hit[0])+1 if len(hit) else -1)
    f=lambda xs:"["+" ".join(("%.3f"%x if np.isfinite(x) else " . ") for x in xs)+"]"
    bl=[f"d{lo}-{hi if hi<99 else '+'}" for lo,hi in BANDS]
    print(f"\n===== DRC(3,3) CONTRACTION + PROPAGATION (step={step}, boards={B}, ticks={T}) =====")
    print(f"  [E1] ||h_t - h_(t-1)|| per tick (t=2..{T}): {f(delta)}")
    print(f"       ratio delta_t/delta_(t-1)           : {f(ratio)}   ({'CONTRACTS (->fixed point)' if np.nanmean(ratio[-3:])<0.9 else 'does not clearly contract'})")
    print(f"       ||h_t|| per tick                    : {f(hn)}")
    print(f"  [E5] mean influence ||dh_t(s)|| by graph-distance from ablated cell, per tick:")
    print(f"        tick \\ band   "+"  ".join(f"{b:>7}" for b in bl))
    for t in range(T): print(f"        {t+1:>4}        "+"  ".join((f"{Ib[t,j]:7.3f}" if np.isfinite(Ib[t,j]) else "   .   ") for j in range(len(BANDS))))
    print(f"       onset tick (reach 50% of settled) per band: {onset}")
    print(f"       --> {'STAGGERED near->far = LOCAL WAVEFRONT (propagation over ticks)' if (onset[-1]>onset[0]>=1) else 'near-simultaneous = global/fast (pool-and-inject), little outward propagation'}")
    print("PLOT_DRCPROP="+repr(dict(delta=[round(x,4) for x in delta],ratio=[round(x,3) if np.isfinite(x) else None for x in ratio],
          Ib=[[round(float(Ib[t,j]),4) if np.isfinite(Ib[t,j]) else None for j in range(len(BANDS))] for t in range(T)],onset=onset,bands=bl)))
    print("="*92)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--boards",type=int,default=256); ap.add_argument("--ticks",type=int,default=12)
    a=ap.parse_args(); main(a.boards,a.ticks)
