import numpy as np
from collections import deque
from experiments.interp.slots import decode_tiles
from experiments.interp.plan import WALL, TARGET
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
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
def nmean(Vp,tb,H,W):
    S=H*W; out=Vp.copy()
    for s in range(S):
        if tb[s]==WALL: continue
        r,c=divmod(s,W); acc=[]
        for dr,dc in DIRS:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and tb[nr*W+nc]!=WALL: acc.append(Vp[nr*W+nc])
        if acc: out[s]=np.mean(acc)
    return out
def fit(X,y,lam=10.0):
    mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd
    return (mu,sd,np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-y.mean())),float(y.mean()))
def pred(X,p): mu,sd,w,b=p; return (X-mu)/sd@w+b
def r2(y,yp): return float(1-((y-yp)**2).sum()/(((y-y.mean())**2).sum()+1e-9))
def reg(prop,vprev,vt):
    X=np.stack([prop,vprev,np.ones(len(vt))],1); coef,*_=np.linalg.lstsq(X,vt,rcond=None); return float(coef[0]),float(coef[1]),r2(vt,X@coef)
