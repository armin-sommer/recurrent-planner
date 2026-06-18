"""Per-object BINARY balanced-accuracy binding probe (the standard for rare-class decodability):
for each object, balance pos/neg, fit a linear probe cell->is-object, report test accuracy on the
balanced split (= balanced accuracy, chance 0.50), per tick. This is the writeup's metric."""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
NAMES={0:'wall',1:'floor',2:'box',3:'target',4:'agent'}
def main(cp_dir,n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg=dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                num_envs=n_boards,n_levels_to_load=n_boards,load_sequentially=True,seed=0)
    policy,_,cp_cfg,ts,step=load_train_state(Path(cp_dir),env_cfg=env_cfg)
    params=ts.params; net=cp_cfg.net; K=net.repeats_per_step; D=net.n_recurrent
    cps=[params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W
    tiles=decode_tiles(obs)
    emb=np.asarray(get_embed(policy,params,jnp.asarray(obs)))
    th=np.asarray(recompute_d3(cps,jnp.asarray(emb),K)[0]); Kk,_,_,C=th.shape
    X=th.reshape(Kk,B*S,C); y=tiles.reshape(B*S).astype(int)
    rng=np.random.default_rng(0)
    rows={}
    for c in [0,2,3,4]:
        z=(y==c).astype(int); pos=np.where(z==1)[0]; neg=np.where(z==0)[0]
        m=min(len(pos),len(neg)); p=pos.copy(); n_=neg.copy(); rng.shuffle(p); rng.shuffle(n_)
        sel=np.concatenate([p[:m],n_[:m]]); rng.shuffle(sel)
        ntr=int(0.8*len(sel)); tr,te=sel[:ntr],sel[ntr:]
        accs=[]
        for k in range(Kk):
            Z=X[k]; mu,sd=Z[tr].mean(0),Z[tr].std(0)+1e-6; Ztr=(Z[tr]-mu)/sd; Zte=(Z[te]-mu)/sd
            w=np.linalg.solve(Ztr.T@Ztr+10.0*np.eye(C),Ztr.T@(z[tr]-0.5))
            pred=((Zte@w)>0).astype(int)
            accs.append(float((pred==z[te]).mean()))
        rows[c]=accs; print(f"  {NAMES[c]:>7} (npos={len(pos)}): ["+ " ".join("%.2f"%a for a in accs)+f"]  settled={accs[-1]:.3f}")
    print(f"===== BINARY BALANCED BINDING (step={step}, boards={B}, K={Kk}, chance=0.50) =====")
    print("SETTLED300M " + " ".join(f"{NAMES[c]}={rows[c][-1]:.3f}" for c in [0,2,3,4]))
    print("="*70)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--boards",type=int,default=256)
    a=ap.parse_args(); main(a.ckpt,a.boards)
