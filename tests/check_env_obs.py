"""Standalone diagnostic for the stuck DRC Sokoban training.

Run ON THE TRAINING BOX (where envpool + boxoban levels live):

    python -m tests.check_env_obs          # uses sokoban_drc33_59 train_env
    python -m tests.check_env_obs --drc     # uses plain sokoban_drc(3,3) train_env

Background: the DRC net normalizes input as `x / 255.0` (normalize_input=False,
see cleanba/network.py). It therefore REQUIRES the observation to be uint8 RGB in
[0, 255] with shape (N, 3, 10, 10). If envpool hands back float [0,1], a wrong
channel count, or NHWC, the net silently sees a near-constant input -> frozen
entropy, ~0 solve rate, high-variance return (the exact symptom we're chasing).

This script makes the real training env, resets/steps it, and asserts the
observation + dependency + data assumptions, printing PASS/FAIL with the
interpretation for each.
"""

import subprocess
import sys

import numpy as np

# Pinned submodule commits the working envpool/gym-sokoban wheels are built from
# (from `git ls-tree HEAD third_party/...`). If the checked-out HEAD differs, the
# installed Sokoban binary may emit a different observation format.
PINNED_ENVPOOL = "ae30e34c8ec64a8d5a5a254f0a528bd75c3cf00f"
PINNED_GYM_SOKOBAN = "3b9fea75ed3b83f188d86929818ca349f3304a17"


def _banner(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def _result(ok, label, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}" + (f"  --  {detail}" if detail else ""))
    return ok


def check_versions():
    _banner("1. DEPENDENCY VERSIONS")
    ok = True

    npv = np.__version__
    ok_np = npv.startswith("1.26")
    _result(ok_np, f"numpy == 1.26.x (found {npv})",
            "" if ok_np else "numpy 2.x changes dtype/casting behaviour; pin numpy==1.26.4")
    ok &= ok_np

    try:
        import envpool  # noqa
        ev = getattr(envpool, "__version__", "unknown")
        print(f"  envpool path: {envpool.__file__}")
        # The AlignmentResearch Sokoban fork registers Sokoban-v0; mainline PyPI does not.
        try:
            import envpool.sokoban  # noqa
            has_sokoban = True
        except Exception:
            has_sokoban = False
        _result(ev == "0.8.4", f"envpool == 0.8.4 (found {ev})",
                "" if ev == "0.8.4" else "expected the AlignmentResearch Sokoban fork 0.8.4")
        _result(has_sokoban, "envpool.sokoban importable (AlignmentResearch fork)",
                "" if has_sokoban else "mainline PyPI envpool has NO Sokoban env -> wrong package installed")
        ok &= has_sokoban
    except Exception as e:
        _result(False, "import envpool", f"{type(e).__name__}: {e}")
        ok = False

    return ok


def check_submodule_commits():
    _banner("1b. SUBMODULE COMMIT PROVENANCE")
    ok = True
    for path, pinned in [("third_party/envpool", PINNED_ENVPOOL),
                         ("third_party/gym-sokoban", PINNED_GYM_SOKOBAN)]:
        try:
            head = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
            match = head == pinned
            _result(match, f"{path} @ {head[:12]} (pinned {pinned[:12]})",
                    "" if match else "checked-out commit differs from pin -> rebuild from pinned commit")
            ok &= match
        except Exception as e:
            _result(False, f"{path} rev-parse", f"{type(e).__name__}: {e} (submodule not initialized?)")
            ok = False
    return ok


def check_levels(cfg):
    _banner("2. BOXOBAN LEVEL DATA")
    import os
    ok = True
    try:
        levels_dir = cfg.levels_dir  # property validates existence + .txt files
        files = [f for f in os.listdir(levels_dir) if f.endswith(".txt")]
        ok_files = len(files) > 0
        _result(ok_files, f"train levels present at {levels_dir}",
                f"{len(files)} .txt files" if ok_files else "no .txt level files found")
        ok &= ok_files
    except Exception as e:
        _result(False, "resolve/list levels_dir", f"{type(e).__name__}: {e}")
        ok = False
    return ok


def check_obs(cfg, num_envs=8):
    _banner("3. OBSERVATION SANITY  (the decisive test)")
    import dataclasses

    made = dataclasses.replace(cfg, num_envs=num_envs).make()
    env = made() if callable(made) else made

    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        obs = np.asarray(obs)

        print(f"  dtype={obs.dtype}  shape={obs.shape}  "
              f"min={obs.min()}  max={obs.max()}  mean={obs.mean():.3f}  "
              f"n_unique={np.unique(obs).size}")

        ok_dtype = obs.dtype == np.uint8
        _result(ok_dtype, "dtype == uint8",
                "" if ok_dtype else f"got {obs.dtype}; net does x/255 so float [0,1] -> near-zero input")

        ok_shape = obs.ndim == 4 and obs.shape[0] == num_envs and obs.shape[1] == 3
        _result(ok_shape, f"shape == ({num_envs}, 3, 10, 10) NCHW",
                "" if ok_shape else f"got {obs.shape}; wrong channel/layout -> net mis-reads board")

        ok_range = obs.max() > 1 and obs.max() <= 255 and obs.min() >= 0
        _result(ok_range, "values span 0..255 (RGB), not 0..1",
                "" if ok_range else f"max={obs.max()}; if max<=1 the obs is already normalized -> x/255 kills it")

        ok_var = np.unique(obs).size > 2
        _result(ok_var, "observation is not near-constant",
                "" if ok_var else "obs has <=2 distinct values -> net cannot condition on state")

        # Step a few random actions: confirm obs changes and reward is delivered.
        n_act = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space[0].n
        rng = np.random.default_rng(0)
        total_r = 0.0
        changed = False
        prev = obs.copy()
        for _ in range(20):
            actions = rng.integers(0, n_act, size=num_envs)
            step_out = env.step(actions)
            nobs = np.asarray(step_out[0])
            rew = np.asarray(step_out[1])
            total_r += float(rew.sum())
            if not np.array_equal(nobs, prev):
                changed = True
            prev = nobs
        _result(changed, "observation changes when actions are taken",
                "" if changed else "obs frozen across steps -> env not actually advancing")
        print(f"  (20 random steps, summed reward across {num_envs} envs = {total_r:.2f})")

        return ok_dtype and ok_shape and ok_range and ok_var and changed
    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    use_drc = "--drc" in sys.argv
    if use_drc:
        from cleanba.config import sokoban_drc
        args = sokoban_drc(3, 3)
        print("Using train_env from: sokoban_drc(3, 3)")
    else:
        from cleanba.config import sokoban_drc33_59
        args = sokoban_drc33_59()
        print("Using train_env from: sokoban_drc33_59  (published recipe)")

    cfg = args.train_env
    print(f"  split={cfg.split!r} difficulty={cfg.difficulty!r} cache_path={cfg.cache_path}")

    results = {
        "versions": check_versions(),
        "submodules": check_submodule_commits(),
        "levels": check_levels(cfg),
        "observation": check_obs(cfg),
    }

    _banner("SUMMARY")
    for k, v in results.items():
        print(f"  {k:12s}: {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        print("\n>>> At least one check FAILED. The first FAIL above is the likely root cause")
        print(">>> of the stuck training. Fix it and re-run before launching a new job.")
        sys.exit(1)
    print("\n>>> All env/deps/data checks PASS. The observation pipeline is healthy;")
    print(">>> if training is still stuck, the cause is elsewhere (look at scale/duration).")


if __name__ == "__main__":
    main()
