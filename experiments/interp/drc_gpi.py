"""DRC(3,3) cellwise-GPI probe. Does the conv core carry, per cell and over thinking ticks,
   (E) a decodable VALUE field that propagates  [policy evaluation], and
   (I) a decodable greedy POLICY that is greedy w.r.t. that value and improves [policy improvement],
   mutually consistent (policy = value gradient)?  Contrast to dense attention (E12): the LOCAL conv
   core should show an OUTWARD frontier (staggered near->far), i.e. cellwise propagation in the loop.
   Per-tick hidden via apply_cells_once (no scan). Reuses the env-agnostic BFS/greedy/decode helpers.
"""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import bfs_from, greedy_dir, WALL, TARGET
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]; BANDS=[(1,2),(3,5),(6,8),(9,99)]
def fit(X,y,lam=10.0):
    mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd
    if y.ndim==1: return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-y.mean())),float(y.mean()))
    return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@y),None)
def pred(X,p): mu,sd,W,b=p; return (X-mu)/sd@W+(b if b is not None else 0.0)
def main(cp, nb, T):
    import eval_pretrained as ep
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    cpdir=ep.make_clean_checkpoint(ep.resolve_checkpoint_dir())
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=nb, n_levels_to_load=nb, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(cpdir, env_cfg=env_cfg)
    params=ts.params; gamma=getattr(getattr(cp_cfg,'loss',None),'gamma',0.97)
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W
    tiles=decode_tiles(obs); emb=np.asarray(get_embed(policy,params,jnp.asarray(obs))); C=emb.shape[-1]
    def once(m,carry,e): return m.network_params.apply_cells_once(carry,e)
    carry=policy.apply(params, jax.random.PRNGKey(0), obs.shape, method=policy.initialize_carry)
    embj=jnp.asarray(emb); th=[]
    for t in range(T):
        carry,_=policy.apply(params, carry, embj, method=once); th.append(np.asarray(carry[-1].h).reshape(B,S,C))
    th=np.stack(th)
    dT=np.full((B,S),np.nan); gd=np.full((B,S),-1,int)
    for b in range(B):
        tg=np.where(tiles[b]==TARGET)[0]
        if len(tg): dT[b]=bfs_from([int(tg[0])],tiles[b],H,W); gd[b]=greedy_dir(tiles[b],dT[b],H,W)
    Vstar=gamma**dT
    rng=np.random.default_rng(0); perm=rng.permutation(B); trb,teb=perm[:int(0.8*B)],perm[int(0.8*B):]
    tr=[(b,s) for b in trb for s in range(S) if gd[b,s]>=0]; te=[(b,s) for b in teb for s in range(S) if gd[b,s]>=0]
    ytrd=np.array([gd[b,s] for b,s in tr]); yted=np.array([gd[b,s] for b,s in te])
    ytrv=np.array([Vstar[b,s] for b,s in tr]); ytev=np.array([Vstar[b,s] for b,s in te])
    dte=np.array([dT[b,s] for b,s in te]); bm=[(dte>=lo)&(dte<=hi) for lo,hi in BANDS]
    valr2=[];diracc=[];accb=np.full((T,len(BANDS)),np.nan);gag=[];gop=[]
    for t in range(T):
        Xtr=np.stack([th[t,b,s] for b,s in tr]); Xte=np.stack([th[t,b,s] for b,s in te])
        pv=fit(Xtr,ytrv); vp=pred(Xte,pv); valr2.append(float(1-((ytev-vp)**2).sum()/(((ytev-ytev.mean())**2).sum()+1e-9)))
        pd=fit(Xtr,np.eye(4)[ytrd]); dp=pred(Xte,pd).argmax(1); diracc.append(float((dp==yted).mean()))
        for j,m in enumerate(bm):
            if m.sum()>=20: accb[t,j]=(dp[m]==yted[m]).mean()
        ag=[];ao=[]
        for b in teb:
            nw=np.where(tiles[b]!=WALL)[0]; vv=np.full(S,-1e9); vv[nw]=pred(th[t,b,nw],pv); dd=np.full(S,-1,int); dd[nw]=pred(th[t,b,nw],pd).argmax(1)
            for s in nw:
                if gd[b,s]<0: continue
                r,c=divmod(s,W); best,bd=-1,-1e9
                for di,(dr,dc) in enumerate(DIRS):
                    nr,nc=r+dr,c+dc
                    if 0<=nr<H and 0<=nc<W and tiles[b,nr*W+nc]!=WALL and vv[nr*W+nc]>bd: bd=vv[nr*W+nc];best=di
                if best>=0: ag.append(float(best==dd[s])); ao.append(float(best==gd[b,s]))
        gag.append(float(np.mean(ag))); gop.append(float(np.mean(ao)))
    f=lambda xs:"["+" ".join("%.2f"%x for x in xs)+"]"
    bl=[f"d{lo}-{hi if hi<99 else '+'}" for lo,hi in BANDS]
    print(f"\n===== DRC(3,3) CELLWISE GPI (step={step}, boards={B}, S={S}, ticks={T}, gamma={gamma}) =====")
    print(f"  (E) value R^2 (h(s)->gamma^dist), per tick: {f(valr2)}   {valr2[0]:.2f}->{valr2[-1]:.2f}")
    print(f"  (I) greedy-dir acc (chance .25), per tick : {f(diracc)}   {diracc[0]:.2f}->{diracc[-1]:.2f}")
    print(f"      dir acc by distance band (OUTWARD frontier?):")
    print(f"        tick \\ band  "+"  ".join(f"{b:>6}" for b in bl))
    for t in range(T): print(f"        {t+1:>4}       "+"  ".join((f"{accb[t,j]:6.2f}" if np.isfinite(accb[t,j]) else "   .  ") for j in range(len(BANDS))))
    onset=[int(np.where(np.isfinite(accb[:,j])&(accb[:,j]>0.45))[0][0])+1 if np.any(np.isfinite(accb[:,j])&(accb[:,j]>0.45)) else -1 for j in range(len(BANDS))]
    print(f"      first tick acc>0.45 per band: {onset}  ({'STAGGERED near->far = OUTWARD (cellwise eval propagates)' if onset[-1]>onset[0]>=1 else 'simultaneous'})")
    print(f"  (C) decoded policy == value-gradient (cellwise consistency), per tick: {f(gag)}   {gag[-1]:.2f}")
    print(f"      value-gradient == optimal dir, per tick: {f(gop)}   {gop[-1]:.2f}")
    print("PLOT_DRC="+repr(dict(valr2=[round(x,3) for x in valr2],diracc=[round(x,3) for x in diracc],
          accb=[[round(float(accb[t,j]),3) if np.isfinite(accb[t,j]) else None for j in range(len(BANDS))] for t in range(T)],
          onset=onset,gradagree=[round(x,3) for x in gag],gradopt=[round(x,3) for x in gop],bands=bl)))
    print("="*92)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--boards",type=int,default=256); ap.add_argument("--ticks",type=int,default=12)
    a=ap.parse_args(); main(None,a.boards,a.ticks)
