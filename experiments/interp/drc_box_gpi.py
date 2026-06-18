"""DRC(3,3) cellwise-GPI probe, BOX-CENTRIC (the faithful Sokoban value/plan).
Ground truth per cell s: box-push distance to nearest target dB(s) (BFS on the push graph: box b->b+d
needs b+d free AND agent-stand cell b-d free), value V_box=gamma^dB, and optimal first-push direction.
Tests over thinking ticks: (E) is V_box decodable + does it propagate/refine (outward frontier by push-
distance band)? (I) is the push-direction decodable, greedy w.r.t. V_box, improving? (C) consistency.
"""
import argparse, dataclasses
from collections import deque
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import WALL, TARGET
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]; BANDS=[(1,2),(3,5),(6,8),(9,99)]
def bfs_box(t,H,W):
    S=H*W; wall=(t==WALL); dB=np.full(S,np.inf); q=deque()
    for tg in np.where(t==TARGET)[0]: dB[tg]=0; q.append(int(tg))
    while q:
        cur=q.popleft(); r,c=divmod(cur,W)
        for dr,dc in DIRS:
            br,bc=r-dr,c-dc; ar,ac=r-2*dr,c-2*dc            # b=cur-d (pushed into cur); agent stands b-d=cur-2d
            if 0<=br<H and 0<=bc<W and 0<=ar<H and 0<=ac<W:
                b=br*W+bc; a=ar*W+ac
                if not wall[b] and not wall[a] and dB[b]>dB[cur]+1: dB[b]=dB[cur]+1; q.append(b)
    return dB
def box_dir(t,dB,H,W):
    S=H*W; wall=(t==WALL); gd=np.full(S,-1,int)
    for s in range(S):
        if wall[s] or not np.isfinite(dB[s]) or dB[s]==0: continue
        r,c=divmod(s,W); best=-1; bv=dB[s]
        for di,(dr,dc) in enumerate(DIRS):
            sr,sc=r+dr,c+dc; ar,ac=r-dr,c-dc                # box -> s+d ; agent stands s-d
            if 0<=sr<H and 0<=sc<W and 0<=ar<H and 0<=ac<W:
                s2=sr*W+sc; a=ar*W+ac
                if not wall[s2] and not wall[a] and np.isfinite(dB[s2]) and dB[s2]<bv: bv=dB[s2]; best=di
        gd[s]=best
    return gd
def fit(X,y,lam=10.0):
    mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd
    if y.ndim==1: return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-y.mean())),float(y.mean()))
    return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@y),None)
def pred(X,p): mu,sd,W,b=p; return (X-mu)/sd@W+(b if b is not None else 0.0)
def main(nb,T):
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
    for t in range(T): carry,_=policy.apply(params, carry, embj, method=once); th.append(np.asarray(carry[-1].h).reshape(B,S,C))
    th=np.stack(th)
    dB=np.full((B,S),np.inf); gd=np.full((B,S),-1,int)
    for b in range(B):
        if (tiles[b]==TARGET).any(): dB[b]=bfs_box(tiles[b],H,W); gd[b]=box_dir(tiles[b],dB[b],H,W)
    Vb=np.where(np.isfinite(dB),gamma**np.where(np.isfinite(dB),dB,0),0.0)
    finite=np.isfinite(dB)
    rng=np.random.default_rng(0); perm=rng.permutation(B); trb,teb=perm[:int(0.8*B)],perm[int(0.8*B):]
    trv=[(b,s) for b in trb for s in range(S) if finite[b,s]]; tev=[(b,s) for b in teb for s in range(S) if finite[b,s]]
    trd=[(b,s) for b in trb for s in range(S) if gd[b,s]>=0]; ted=[(b,s) for b in teb for s in range(S) if gd[b,s]>=0]
    ytrv=np.array([Vb[b,s] for b,s in trv]); ytev=np.array([Vb[b,s] for b,s in tev])
    ytrd=np.array([gd[b,s] for b,s in trd]); yted=np.array([gd[b,s] for b,s in ted])
    dB_te=np.array([dB[b,s] for b,s in ted]); bm=[(dB_te>=lo)&(dB_te<=hi) for lo,hi in BANDS]
    valr2=[];diracc=[];accb=np.full((T,len(BANDS)),np.nan);gag=[];gop=[]
    for t in range(T):
        Xv=np.stack([th[t,b,s] for b,s in trv]); Xvt=np.stack([th[t,b,s] for b,s in tev]); pv=fit(Xv,ytrv); vp=pred(Xvt,pv)
        valr2.append(float(1-((ytev-vp)**2).sum()/(((ytev-ytev.mean())**2).sum()+1e-9)))
        Xd=np.stack([th[t,b,s] for b,s in trd]); Xdt=np.stack([th[t,b,s] for b,s in ted]); pd=fit(Xd,np.eye(4)[ytrd]); dp=pred(Xdt,pd).argmax(1)
        diracc.append(float((dp==yted).mean()))
        for j,m in enumerate(bm):
            if m.sum()>=20: accb[t,j]=(dp[m]==yted[m]).mean()
        ag=[];ao=[]
        for b in teb:
            ok=np.where(finite[b])[0]
            if len(ok)<4: continue
            vv=np.full(S,-1e9); vv[ok]=pred(th[t,b,ok],pv); dd=np.full(S,-1,int); dd[ok]=pred(th[t,b,ok],pd).argmax(1)
            for s in ok:
                if gd[b,s]<0: continue
                r,c=divmod(s,W); best,bd=-1,-1e9
                for di,(dr,dc) in enumerate(DIRS):
                    sr,sc=r+dr,c+dc; ar,ac=r-dr,c-dc
                    if 0<=sr<H and 0<=sc<W and 0<=ar<H and 0<=ac<W and tiles[b,sr*W+sc]!=WALL and tiles[b,ar*W+ac]!=WALL and vv[sr*W+sc]>bd: bd=vv[sr*W+sc]; best=di
                if best>=0: ag.append(float(best==dd[s])); ao.append(float(best==gd[b,s]))
        gag.append(float(np.mean(ag)) if ag else float('nan')); gop.append(float(np.mean(ao)) if ao else float('nan'))
    f=lambda xs:"["+" ".join(("%.2f"%x if np.isfinite(x) else " . ") for x in xs)+"]"
    bl=[f"d{lo}-{hi if hi<99 else '+'}" for lo,hi in BANDS]
    print(f"\n===== DRC(3,3) BOX-CENTRIC CELLWISE GPI (step={step}, boards={B}, ticks={T}, gamma={gamma}) =====")
    print(f"  mean #pushable cells/board={int(finite.sum()/B)}, mean #push-dir cells/board={int((gd>=0).sum()/B)}")
    print(f"  (E) box-push value R^2 (h->gamma^pushdist), per tick: {f(valr2)}   {valr2[0]:.2f}->{valr2[-1]:.2f}")
    print(f"  (I) box-push dir acc (chance .25), per tick         : {f(diracc)}   {diracc[0]:.2f}->{diracc[-1]:.2f}")
    print(f"      dir acc by push-distance band (OUTWARD frontier?):")
    print(f"        tick \\ band  "+"  ".join(f"{b:>6}" for b in bl))
    for t in range(T): print(f"        {t+1:>4}       "+"  ".join((f"{accb[t,j]:6.2f}" if np.isfinite(accb[t,j]) else "   .  ") for j in range(len(BANDS))))
    onset=[int(np.where(np.isfinite(accb[:,j])&(accb[:,j]>0.40))[0][0])+1 if np.any(np.isfinite(accb[:,j])&(accb[:,j]>0.40)) else -1 for j in range(len(BANDS))]
    print(f"      first tick acc>0.40 per band: {onset}  ({'STAGGERED near->far = OUTWARD' if onset[-1]>onset[0]>=1 else 'simultaneous'})")
    print(f"  (C) decoded push-dir == value-gradient, per tick: {f(gag)}")
    print(f"      value-gradient == optimal push-dir, per tick : {f(gop)}")
    print("PLOT_DRCBOX="+repr(dict(valr2=[round(x,3) for x in valr2],diracc=[round(x,3) for x in diracc],onset=onset,
          accb=[[round(float(accb[t,j]),3) if np.isfinite(accb[t,j]) else None for j in range(len(BANDS))] for t in range(T)],
          gradagree=[round(x,3) if np.isfinite(x) else None for x in gag],gradopt=[round(x,3) if np.isfinite(x) else None for x in gop])))
    print("="*92)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--boards",type=int,default=256); ap.add_argument("--ticks",type=int,default=12)
    a=ap.parse_args(); main(a.boards,a.ticks)
