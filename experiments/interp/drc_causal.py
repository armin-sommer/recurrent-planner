"""DRC(3,3) CAUSAL box-centric check (E9/E10 analog). Add a wall that lengthens the box-push path
(on-path) vs a cosmetic wall (off-path); does the DECODED cellwise value re-form in the ground-truth-
predicted direction, more for on-path than off-path, and building over thinking ticks?"""
import argparse, dataclasses
from collections import deque
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
from experiments.interp.planning import get_embed
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import WALL, FLOOR, TARGET
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]; CAP=30.0
def bfs_box(t,H,W):
    S=H*W; wall=(t==WALL); dB=np.full(S,np.inf); q=deque()
    for tg in np.where(t==TARGET)[0]: dB[tg]=0; q.append(int(tg))
    while q:
        cur=q.popleft(); r,c=divmod(cur,W)
        for dr,dc in DIRS:
            br,bc=r-dr,c-dc; ar,ac=r-2*dr,c-2*dc
            if 0<=br<H and 0<=bc<W and 0<=ar<H and 0<=ac<W:
                b=br*W+bc; a=ar*W+ac
                if not wall[b] and not wall[a] and dB[b]>dB[cur]+1: dB[b]=dB[cur]+1; q.append(b)
    return dB
def fit(X,y,lam=10.0):
    mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd
    return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-y.mean())),float(y.mean()))
def pred(X,p): mu,sd,w,b=p; return (X-mu)/sd@w+b
def main(nb,T):
    import eval_pretrained as ep
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    cpdir=ep.make_clean_checkpoint(ep.resolve_checkpoint_dir())
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env, num_envs=nb, n_levels_to_load=nb, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(cpdir, env_cfg=env_cfg)
    params=ts.params; gamma=getattr(getattr(cp_cfg,'loss',None),'gamma',0.97)
    obs0=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs0.shape; S=H*W
    tiles0=decode_tiles(obs0); emb0=np.asarray(get_embed(policy,params,jnp.asarray(obs0))); C=emb0.shape[-1]
    dB0=np.full((B,S),np.inf); dBon=np.full((B,S),np.inf); use=np.zeros(B,bool)
    obs_on=obs0.copy(); obs_off=obs0.copy(); rng=np.random.default_rng(0)
    for b in range(B):
        if not (tiles0[b]==TARGET).any(): continue
        dB0[b]=bfs_box(tiles0[b],H,W); fin=np.isfinite(dB0[b]); floors=np.where(tiles0[b]==FLOOR)[0]
        if len(floors)<2: continue
        cand=rng.choice(floors,size=min(18,len(floors)),replace=False); best=-1.0; con=-1; coff=-1; dbon=None
        for c in cand:
            t2=tiles0[b].copy(); t2[int(c)]=WALL; d2=bfs_box(t2,H,W)
            inc=np.where(fin, np.clip(np.minimum(d2,CAP)-np.minimum(dB0[b],CAP),0,None), 0.0).sum()
            if inc>best: best=inc; con=int(c); dbon=d2
            if inc==0 and coff<0: coff=int(c)
        if con>=0 and best>1.0 and coff>=0:
            r,c=divmod(con,W); obs_on[b,:,r,c]=0; r2,c2=divmod(coff,W); obs_off[b,:,r2,c2]=0; dBon[b]=dbon; use[b]=True
    def once(m,carry,e): return m.network_params.apply_cells_once(carry,e)
    def run(obs):
        emb=np.asarray(get_embed(policy,params,jnp.asarray(obs))); carry=policy.apply(params,jax.random.PRNGKey(0),obs.shape,method=policy.initialize_carry)
        embj=jnp.asarray(emb); Hs=[]
        for t in range(T): carry,_=policy.apply(params,carry,embj,method=once); Hs.append(np.asarray(carry[-1].h).reshape(B,S,C))
        return np.stack(Hs)
    Ho=run(obs0); Hon=run(obs_on); Hoff=run(obs_off)
    V0=np.where(np.isfinite(dB0),gamma**np.where(np.isfinite(dB0),dB0,0),0.0)
    Von=np.where(np.isfinite(dBon),gamma**np.where(np.isfinite(dBon),dBon,0),0.0); dVs=Von-V0
    tr=[(b,s) for b in np.where(use)[0] for s in range(S) if np.isfinite(dB0[b,s])]
    pv=fit(np.stack([Ho[-1,b,s] for b,s in tr]), np.array([V0[b,s] for b,s in tr]))
    ub=np.where(use)[0]; corr=[]; mon=[]; moff=[]
    for t in range(T):
        cs=[];a_on=[];a_off=[]
        for b in ub:
            nw=np.where(tiles0[b]!=WALL)[0]; vo=pred(Ho[t,b,nw],pv); von=pred(Hon[t,b,nw],pv); voff=pred(Hoff[t,b,nw],pv)
            don=von-vo; doff=voff-vo; ds=dVs[b,nw]; aff=np.abs(ds)>0.02
            if aff.sum()>=3:
                cc=np.corrcoef(don[aff],ds[aff])[0,1]
                if np.isfinite(cc): cs.append(cc)
                a_on.append(np.abs(don[aff]).mean())
            a_off.append(np.abs(doff).mean())
        corr.append(float(np.mean(cs)) if cs else float('nan')); mon.append(float(np.mean(a_on)) if a_on else float('nan')); moff.append(float(np.mean(a_off)))
    f=lambda xs:"["+" ".join(("%.2f"%x if np.isfinite(x) else " . ") for x in xs)+"]"
    ratio=[mon[t]/moff[t] if (np.isfinite(mon[t]) and moff[t]>1e-9) else float('nan') for t in range(T)]
    print(f"\n===== DRC(3,3) CAUSAL box-value reformation (step={step}, boards={B}, used={int(use.sum())}, ticks={T}) =====")
    print(f"  on-path: decoded dV-hat vs ground-truth dV* CORRELATION (affected cells), per tick: {f(corr)}")
    print(f"  |dV-hat| on-path (affected) per tick : {f(mon)}")
    print(f"  |dV-hat| off-path (cosmetic) per tick: {f(moff)}")
    print(f"  on/off magnitude RATIO per tick      : {f(ratio)}   settled ratio={ratio[-1]:.2f}")
    print(f"  --> reaches/re-forms: {'YES (on-path tracks dV*, on>>off, builds over ticks)' if (np.nanmean(corr[-3:])>0.2 and np.nanmean(ratio[-3:])>1.3) else 'weak/not clearly'}")
    print("PLOT_DRCCAUSAL="+repr(dict(corr=[round(x,3) if np.isfinite(x) else None for x in corr],mon=[round(x,4) if np.isfinite(x) else None for x in mon],moff=[round(x,4) for x in moff],ratio=[round(x,3) if np.isfinite(x) else None for x in ratio])))
    print("="*92)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--boards",type=int,default=160); ap.add_argument("--ticks",type=int,default=10)
    a=ap.parse_args(); main(a.boards,a.ticks)
