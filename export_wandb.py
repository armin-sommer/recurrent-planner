#!/usr/bin/env python
"""Export full wandb run history to CSV for offline inspection.

Usage:
    wandb login                      # once, if not already authenticated locally
    python export_wandb.py <entity>/<project>/<run_id> [more run paths...]

Find <entity>/<project>/<run_id> in the run's browser URL:
    https://wandb.ai/<entity>/<project>/runs/<run_id>
                     ^^^^^^^^ ^^^^^^^^^      ^^^^^^^
Pass the convlstm baseline run (and optionally the attention run to compare).
Writes one CSV per run: wandb_<run_id>.csv
"""
import csv
import sys

import wandb


def resolve_run(api, run_path: str):
    """Accept entity/project/<run_id OR display-name>."""
    try:
        return api.run(run_path)  # works when the last segment is the real run id
    except Exception:
        pass
    entity, project, ident = run_path.split("/")
    runs = api.runs(f"{entity}/{project}")
    for r in runs:
        if r.id == ident or r.name == ident:
            return r
    names = [f"{r.name}  (id={r.id}, state={r.state})" for r in runs]
    raise SystemExit(
        f"Could not find '{ident}' in {entity}/{project}. Available runs:\n  " + "\n  ".join(names)
    )


def export(run_path: str) -> None:
    api = wandb.Api()
    run = resolve_run(api, run_path)
    rows = list(run.scan_history())  # every logged step, not downsampled

    # union of all metric keys, in first-seen order
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys = sorted(keys)

    out = f"wandb_{run.id}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== {run_path}  (name={run.name}, state={run.state}) ===")
    print(f"wrote {out}: {len(rows)} rows, {len(keys)} columns")
    print("columns:", keys)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        export(p)
