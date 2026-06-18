"""Direct box-push value propagation, DRC core: decode V_box per tick (apply_cells_once loop); ||dV|| per
tick (static?); V_t ~ a*(Nmean v_{t-1})+b*v (graph-neighbour mean = the conv's local support; no single A)."""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import get_embed
from experiments.interp._propcommon import bfs_box, nmean, fit, pred, reg, decode_tiles, WALL, TARGET
def main(nb,T):
    import eval_pretrained as ep
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    cpdir=ep.make_clean_checkpoint(ep.resolve_checkpoint_dir())
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=nb, n_levels_to_load=nb, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(cpdir, env_cfg=env_cfg)
    params=ts.params; gamma=getattr(getattr(cp_cfg,'loss',None),'gamma',0.97)
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W; tiles=decode_tiles(obs)
    emb=np.asarray(get_embed(policy,params,jnp.asarray(obs))); C=emb.shape[-1]
    def once(m,carry,e): return m.network_params.apply_cells_once(carry,e)
    carry=policy.apply(params, jax.random.PRNGKey(0), obs.shape, method=policy.initialize_carry); embj=jnp.asarray(emb); th=[]
    for t in range(T): carry,_=policy.apply(params,carry,embj,method=once); th.append(np.asarray(carry[-1].h).reshape(B,S,C))
    th=np.stack(th)
    dB=np.full((B,S),np.inf)
    for b in range(B):
        if (tiles[b]==TARGET).any(): dB[b]=bfs_box(tiles[b],H,W)
    Vb=np.where(np.isfinite(dB),gamma**np.where(np.isfinite(dB),dB,0),0.0); fin=np.isfinite(dB)
    tr=[(b,s) for b in range(B) for s in range(S) if fin[b,s]]
    pf=fit(np.stack([th[-1,b,s] for b,s in tr]), np.array([Vb[b,s] for b,s in tr]))
    V=np.stack([pred(th[t].reshape(B*S,C),pf).reshape(B,S) for t in range(T)])
    aN=[];rN=[];dv=[]
    for t in range(1,T):
        pN=[];vp=[];vt=[];dd=[]
        for b in range(B):
            nw=np.where(tiles[b]!=WALL)[0]; propN=nmean(V[t-1,b],tiles[b],H,W)
            pN.append(propN[nw]); vp.append(V[t-1,b,nw]); vt.append(V[t,b,nw])
            fb=np.where(fin[b])[0]
            if len(fb): dd.append(np.linalg.norm(V[t,b,fb]-V[t-1,b,fb])/(np.linalg.norm(V[t,b,fb])+1e-9))
        pN=np.concatenate(pN);vp=np.concatenate(vp);vt=np.concatenate(vt)
        a2,_,c2=reg(pN,vp,vt); aN.append(a2);rN.append(c2);dv.append(float(np.mean(dd)))
    f=lambda xs:"["+" ".join("%.2f"%x for x in xs)+"]"
    print(f"\n===== DRC box-push value propagation (step={step}, B={B}, T={T}) =====")
    print(f"  ||dV_box||/||V|| per tick (static if ~0): {f(dv)}")
    print(f"  graph-N V_t~a*(Nmean v)+b*v : a {f(aN)}  R^2 {f(rN)}  settled a={np.mean(aN[-3:]):+.2f}")
    print("PLOT_DRCPROP="+repr(dict(dv=[round(x,3) for x in dv],aN=[round(x,3) for x in aN])))
    print("="*88)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--boards",type=int,default=192); ap.add_argument("--ticks",type=int,default=10)
    a=ap.parse_args(); main(a.boards,a.ticks)
