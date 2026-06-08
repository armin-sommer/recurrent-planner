#!/usr/bin/env python3
"""
Isolation test: load the FULLY-TRAINED pretrained DRC(3,3) checkpoint
(AlignmentResearch/learned-planner :: drc33/bkynosqi/cp_2002944000) and run
a small, fast Sokoban eval in THIS env build.

INTERPRETATION OF RESULT
------------------------
The "success" metric is the mean over episodes of a per-episode boolean
"did the level terminate (get solved)" flag (cleanba/evaluate.py:88, :143).
A healthy, fully-trained DRC(3,3) solves a LARGE FRACTION of unfiltered test
levels -- success WELL ABOVE 0, and it should climb with extra think steps
(steps_to_think) because DRC benefits from planning.

  * success > 0 (esp. larger at think=4) => env + obs/action pipeline in this
    build is CORRECT. The training-not-learning issue is a dynamics/version
    problem, not an obs/action wiring bug.
  * success ~= 0.0 at EVERY think step => the obs/action wiring in THIS build
    is broken (the trained brain can't act through this env interface).

Run on the POD (Linux + envpool + GPU). It will NOT run on Mac (envpool is
imported lazily inside env_cfg.make(), cleanba/environments.py:46-47).

This script does NOT import the learned_planner package. It pulls the
checkpoint with huggingface_hub.snapshot_download directly, and drives
cleanba.evaluate.EvalConfig.run -- NO WandbWriter, NO load_and_eval.py.
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

import jax

# ---------------------------------------------------------------------------
# 0. Config knobs for the SMALL/FAST eval.
# ---------------------------------------------------------------------------
REPO_ID = "AlignmentResearch/learned-planner"
CKPT_IN_REPO = "drc33/bkynosqi/cp_2002944000"          # final 2B-step DRC(3,3)
ALLOW_PATTERN = CKPT_IN_REPO + "/*"                    # globs cfg.json + model

CACHE_PATH = Path("/opt/sokoban_cache")                # boxoban-levels-master lives here
SPLIT = "test"                                         # unfiltered/test (falls back to train)
DIFFICULTY = "unfiltered"

NUM_ENVS = 64                                          # small but representative slice
N_EPISODE_MULTIPLE = 1                                 # total episodes = NUM_ENVS * this
STEPS_TO_THINK = [0, 4]                                # [0] = act immediately, [4] = DRC plans
EPISODE_STEPS = 120                                    # fixed-length episodes (mirror default_eval_envs)


# ---------------------------------------------------------------------------
# 1. Download ONLY the one checkpoint (no learned_planner package needed).
#    snapshot_download returns the CACHE SNAPSHOT ROOT; we must join the
#    in-repo subpath to reach the dir that holds {cfg.json, model}.
#    (Research finding 3, snapshot_download return-path gotcha.)
# ---------------------------------------------------------------------------
def resolve_checkpoint_dir() -> Path:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(REPO_ID, allow_patterns=[ALLOW_PATTERN])
    ckpt_dir = Path(snap) / CKPT_IN_REPO
    cfg = ckpt_dir / "cfg.json"
    model = ckpt_dir / "model"
    if not cfg.exists() or not model.exists():
        raise FileNotFoundError(
            f"Expected flat files cfg.json + model under {ckpt_dir}.\n"
            f"  cfg.json exists: {cfg.exists()}\n"
            f"  model    exists: {model.exists()}\n"
            f"Listing: {sorted(p.name for p in ckpt_dir.iterdir()) if ckpt_dir.exists() else 'MISSING'}\n"
            "If 'model' is a DIRECTORY this is an orbax checkpoint and load_train_state "
            "(which does open(dir/'model','rb')) will NOT handle it."
        )
    return ckpt_dir


# ---------------------------------------------------------------------------
# 2. DEFENSIVE pre-clean of cfg.json for the multi-extra-key loader bug.
#
#    cleanba_impala.py:898-900 deletes extra keys but re-calls
#    farconf.from_dict INSIDE the delete loop with NO inner try/except, so the
#    2nd+ unknown key (or an unknown key at a NESTED level) raises an UNCAUGHT
#    databind ConversionError and load_train_state dies.
#
#    We pre-parse the SAME nested dict the loader uses (args_dict["cfg"], see
#    cleanba_impala.py:888) in a robust while-loop that strips EVERY batch of
#    extra keys before re-parsing, then rewrite a CLEANED cfg.json into a
#    private copy of the checkpoint dir. The loader then hits ZERO extra keys
#    and the buggy loop never fires.
#
#    NOTE: this only fixes "encountered extra keys" drift. A MISSING-required
#    field or changed-type field raises a DIFFERENT ConversionError that this
#    cannot repair (Research finding 1/4 risks) -- if you see that, a new
#    required Args field lacks a default and must be added.
# ---------------------------------------------------------------------------
def make_clean_checkpoint(ckpt_dir: Path) -> Path:
    import farconf
    import databind.core.converter

    from cleanba.config import Args

    with open(ckpt_dir / "cfg.json", "r") as f:
        args_dict = json.load(f)

    # Mirror the loader: prefer the nested "cfg" sub-dict, else the whole dict.
    loaded_cfg = args_dict.get("cfg", args_dict)

    removed_total: set[str] = set()
    while True:
        try:
            farconf.from_dict(loaded_cfg, Args)
            break
        except databind.core.converter.ConversionError as e:
            m = re.search(r"encountered extra keys: \{(.*?)\}", e.message)
            if m is None:
                # Not an extra-keys problem -- likely a missing/required or
                # changed-type field. We cannot auto-fix that here.
                raise RuntimeError(
                    "cfg.json failed to parse into Args and it is NOT an "
                    "'extra keys' error -- a required Args field probably lacks "
                    "a default on this branch. Original error:\n" + str(e)
                ) from e
            batch = {kk.strip().strip("'\"") for kk in m.group(1).split(",")}
            before = len(loaded_cfg)
            for k in batch:
                loaded_cfg.pop(k, None)
            removed_total |= batch
            if len(loaded_cfg) == before:
                # Could not remove any key it complained about -> nested key.
                raise RuntimeError(
                    f"Extra keys reported but none were at the top level of the "
                    f"cfg sub-dict: {batch}. They are nested (inside net/train_env). "
                    "Edit cfg.json by hand or patch the loader's while-loop."
                ) from e

    if not removed_total:
        # Clean already; just hand back the original dir.
        print("[cfg.json] no extra keys -- using checkpoint dir as-is.")
        return ckpt_dir

    print(f"[cfg.json] stripped {len(removed_total)} extra key(s): {sorted(removed_total)}")
    # Write a cleaned copy alongside, preserving the {"cfg":..., "update_step":...}
    # wrapper if present so update_step is still read (cleanba_impala.py:884).
    if "cfg" in args_dict:
        args_dict["cfg"] = loaded_cfg
    else:
        args_dict = loaded_cfg

    clean_dir = ckpt_dir.parent / (ckpt_dir.name + "_clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    with open(clean_dir / "cfg.json", "w") as f:
        json.dump(args_dict, f)
    # Symlink (or copy) the model blob so the cleaned dir is a full checkpoint.
    model_link = clean_dir / "model"
    if not model_link.exists():
        try:
            model_link.symlink_to(ckpt_dir / "model")
        except OSError:
            import shutil
            shutil.copy2(ckpt_dir / "model", model_link)
    return clean_dir


# ---------------------------------------------------------------------------
# 3. Main: build env_cfg, load, eval, print.
# ---------------------------------------------------------------------------
def main() -> None:
    from cleanba.cleanba_impala import load_train_state
    from cleanba.environments import EnvpoolBoxobanConfig
    from cleanba.evaluate import EvalConfig

    ckpt_dir = resolve_checkpoint_dir()
    print(f"[ckpt] resolved checkpoint dir: {ckpt_dir}")

    # Defensive cfg.json pre-clean for the >1 / nested extra-key loader bug.
    load_dir = make_clean_checkpoint(ckpt_dir)

    # The eval env_cfg. nn_without_noop defaults True (environments.py:111),
    # which gives a 4-action head matching the DRC(3,3) trained Output layer
    # (hidden, 4). Do NOT set nn_without_noop=False / finetune_with_noop_head:
    # that builds a 5-wide head and from_bytes shape-mismatches.
    #   * load_sequentially=True is REQUIRED so n_levels_to_load != -1
    #     (environments.py:127-128).
    #   * n_levels_to_load == NUM_ENVS so the first NUM_ENVS levels are swept.
    env_cfg = EnvpoolBoxobanConfig(
        max_episode_steps=EPISODE_STEPS,
        min_episode_steps=EPISODE_STEPS,
        num_envs=NUM_ENVS,
        seed=0,
        load_sequentially=True,
        n_levels_to_load=NUM_ENVS,
        cache_path=CACHE_PATH,
        split=SPLIT,
        difficulty=DIFFICULTY,
        nn_without_noop=True,
    )

    # Same env_cfg used for BOTH load (head/obs shapes) and eval.
    # load_train_state internally runs env_cfg.make() + args.net.init_params,
    # then flax.serialization.from_bytes (expect the dont_inject_lr ValueError
    # retry at cleanba_impala.py:924-931 for this paper checkpoint -- NORMAL).
    print("[load] load_train_state(...) -- expect a 'dont_inject_lr' retry; that is normal.")
    policy, _carry, cp_cfg, train_state, update_step = load_train_state(load_dir, env_cfg=env_cfg)
    print(f"[load] OK. update_step={update_step}")

    # Sanity-print the action-space convention the checkpoint was trained with.
    try:
        trained_no_noop = getattr(cp_cfg.train_env, "nn_without_noop", "??")
        print(f"[check] checkpoint train_env.nn_without_noop = {trained_no_noop} "
              f"(eval uses nn_without_noop=True -> 4-action head)")
    except Exception:
        pass

    # get_action_fn -- verbatim from load_and_eval.py:128.
    get_action_fn = jax.jit(
        partial(policy.apply, method=policy.get_action),
        static_argnames="temperature",
    )

    evaluator = EvalConfig(
        env=env_cfg,
        n_episode_multiple=N_EPISODE_MULTIPLE,
        steps_to_think=STEPS_TO_THINK,
        temperature=0.0,   # argmax / greedy (network.py:147-148)
    )

    print(f"[eval] running {NUM_ENVS} envs x {N_EPISODE_MULTIPLE} = "
          f"{NUM_ENVS * N_EPISODE_MULTIPLE} episodes, think steps {STEPS_TO_THINK} ...")
    metrics = evaluator.run(policy, get_action_fn, train_state.params, key=jax.random.PRNGKey(1234))

    print("\n==================== RESULTS ====================")
    best = 0.0
    for stt in evaluator.steps_to_think:
        p = f"{stt:02d}"
        succ = metrics[p + "_episode_successes"]
        ret = metrics[p + "_episode_returns"]
        length = metrics[p + "_episode_lengths"]
        best = max(best, succ)
        print(f"  think={stt:2d}   success={succ:.3f}   return={ret:7.3f}   length={length:6.1f}")
    print("=================================================")
    print(f"  episodes evaluated per think step: {NUM_ENVS * N_EPISODE_MULTIPLE}")
    print(f"  levels: {DIFFICULTY}/{SPLIT}")
    print("-------------------------------------------------")
    if best > 0.0:
        print(f"  VERDICT: success={best:.3f} > 0  ==>  env + obs/action pipeline is CORRECT.")
        print("           Training-not-learning is a DYNAMICS / VERSION issue, NOT wiring.")
    else:
        print("  VERDICT: success ~= 0.0 at EVERY think step  ==>  the obs/action wiring")
        print("           in THIS build is BROKEN (a fully-trained DRC(3,3) cannot get 0).")
    print("=================================================")


if __name__ == "__main__":
    main()
