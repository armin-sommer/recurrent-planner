"""Per-object one-vs-rest BALANCED binding accuracy at a checkpoint (chance 0.50), per tick.
Matches the writeup's 'per-object balanced accuracy' metric (the main script prints raw multiclass)."""
import argparse, dataclasses
from pathlib import Path
import numpy as np, jax.numpy as jnp
from experiments.interp.planning import recompute_d3, get_embed
from experiments.interp.slots import decode_tiles
NAMES = {0:'wall',1:'floor',2:'box',3:'target',4:'agent'}
def main(cp_dir, n_boards):
    from cleanba.cleanba_impala import load_train_state
    from cleanba.load_and_eval import planning_eval_envs
    env_cfg = dataclasses.replace(planning_eval_envs()["valid_medium"].env,
                                  num_envs=n_boards, n_levels_to_load=n_boards, load_sequentially=True, seed=0)
    policy,_,cp_cfg,ts,step = load_train_state(Path(cp_dir), env_cfg=env_cfg)
    params=ts.params; net=cp_cfg.net; K=net.repeats_per_step; D=net.n_recurrent
    cps=[params["params"]["network_params"][f"cell_list_{i}"] for i in range(D)]
    obs=np.asarray(env_cfg.make().reset()[0]); B,_,H,W=obs.shape; S=H*W
    tiles=decode_tiles(obs)
    emb=np.asarray(get_embed(policy,params,jnp.asarray(obs)))
    th=np.asarray(recompute_d3(cps,jnp.asarray(emb),K)[0])
    Kk,_,_,C=th.shape
    X=th.reshape(Kk,B*S,C); y=tiles.reshape(B*S).astype(int)
    rng=np.random.default_rng(0); idx=rng.permutation(B*S); n=int(0.8*len(idx)); tr,te=idx[:n],idx[n:]
    rows={c:[] for c in [0,2,3,4]}
    for k in range(Kk):
        Z=X[k]; mu,sd=Z[tr].mean(0),Z[tr].std(0)+1e-6; Ztr=(Z[tr]-mu)/sd; Zte=(Z[te]-mu)/sd
        W=np.linalg.solve(Ztr.T@Ztr+10.0*np.eye(C), Ztr.T@np.eye(5)[y[tr]])
        pred=(Zte@W).argmax(1); yte=y[te]
        for c in [0,2,3,4]:
            pos=yte==c; neg=~pos
            rec=(pred[pos]==c).mean() if pos.sum() else float('nan')
            spec=(pred[neg]!=c).mean() if neg.sum() else float('nan')
            rows[c].append(0.5*(rec+spec))
    print(f"===== BALANCED BINDING (step={step}, boards={B}, K={Kk}, chance=0.50) =====")
    for c in [0,2,3,4]:
        v=rows[c]; print(f"  {NAMES[c]:>7}: "+("[%s]"%" ".join("%.2f"%x for x in v))+f"  settled={v[-1]:.3f}")
    print("SETTLED300M " + " ".join(f"{NAMES[c]}={rows[c][-1]:.3f}" for c in [0,2,3,4]))
    print("="*70)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--boards",type=int,default=256)
    a=ap.parse_args(); main(a.ckpt,a.boards)
