"""Direct box-push value propagation, ATTENTION core: decode V_box per tick; ||dV|| per tick (static?);
V_t ~ a*(A_{t-1} v_{t-1})+b*v (own learned A) and ~ a*(Nmean v_{t-1})+b*v (graph-neighbour mean)."""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp._propcommon import bfs_box, nmean, fit, pred, reg, decode_tiles, WALL, TARGET
def main(cp,nb,K):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=nb, n_levels_to_load=nb, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(Path(cp), env_cfg=env_cfg)
    params=ts.params; net=cp_cfg.net; D=net.n_recurrent; gamma=cp_cfg.loss.gamma
    cps=[params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W; tiles=decode_tiles(obs)
    emb=np.asarray(get_embed(policy,params,jnp.asarray(obs))); top_h,attn=recompute_d3(cps,jnp.asarray(emb),K)
    top_h=np.asarray(top_h); Atop=np.asarray(attn[:,D-1].mean(2)); C=top_h.shape[-1]
    dB=np.full((B,S),np.inf)
    for b in range(B):
        if (tiles[b]==TARGET).any(): dB[b]=bfs_box(tiles[b],H,W)
    Vb=np.where(np.isfinite(dB),gamma**np.where(np.isfinite(dB),dB,0),0.0); fin=np.isfinite(dB)
    tr=[(b,s) for b in range(B) for s in range(S) if fin[b,s]]
    pf=fit(np.stack([top_h[-1,b,s] for b,s in tr]), np.array([Vb[b,s] for b,s in tr]))
    V=np.stack([pred(top_h[t].reshape(B*S,C),pf).reshape(B,S) for t in range(K)])
    aA=[];aN=[];rA=[];dv=[]
    for t in range(1,K):
        pA=[];pN=[];vp=[];vt=[];dd=[]
        for b in range(B):
            nw=np.where(tiles[b]!=WALL)[0]; propA=Atop[t-1,b]@V[t-1,b]; propN=nmean(V[t-1,b],tiles[b],H,W)
            pA.append(propA[nw]); pN.append(propN[nw]); vp.append(V[t-1,b,nw]); vt.append(V[t,b,nw])
            fb=np.where(fin[b])[0]
            if len(fb): dd.append(np.linalg.norm(V[t,b,fb]-V[t-1,b,fb])/(np.linalg.norm(V[t,b,fb])+1e-9))
        pA=np.concatenate(pA);pN=np.concatenate(pN);vp=np.concatenate(vp);vt=np.concatenate(vt)
        a1,_,c1=reg(pA,vp,vt); a2,_,_=reg(pN,vp,vt); aA.append(a1);rA.append(c1);aN.append(a2);dv.append(float(np.mean(dd)))
    f=lambda xs:"["+" ".join("%.2f"%x for x in xs)+"]"
    print(f"\n===== ATTENTION box-push value propagation (step={step}, B={B}, K={K}) =====")
    print(f"  ||dV_box||/||V|| per tick (static if ~0): {f(dv)}")
    print(f"  own-A   V_t~a*(A v)+b*v : a {f(aA)}  R^2 {f(rA)}  settled a={np.mean(aA[-3:]):+.2f}")
    print(f"  graph-N V_t~a*(Nmean v)+b*v : a {f(aN)}  settled a={np.mean(aN[-3:]):+.2f}")
    print("PLOT_ATTNPROP="+repr(dict(dv=[round(x,3) for x in dv],aA=[round(x,3) for x in aA],aN=[round(x,3) for x in aN])))
    print("="*88)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--boards",type=int,default=192); ap.add_argument("--ticks",type=int,default=8)
    a=ap.parse_args(); main(a.ckpt,a.boards,a.ticks)
