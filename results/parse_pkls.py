#!/usr/bin/env python
"""Parse cleanba.load_and_eval metrics pkls into a tidy CSV of the thinking-step sweep.

Each pkl is one (arm x eval_set) at its evaluated checkpoint, with keys like
``NN_episode_successes`` for NN in 00..12 (extra thinking steps), plus zero_boxes,
cycle/noop stats, etc. Run on the node where the pkls live:

    python parse_pkls.py [iso_dir=/tmp/eval_iso] [out_csv=results_thinking.csv]
"""
import csv
import glob
import os
import pickle
import sys

iso = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval_iso"
out = sys.argv[2] if len(sys.argv) > 2 else "results_thinking.csv"

rows = []
for arm_dir in sorted(glob.glob(os.path.join(iso, "*"))):
    arm = os.path.basename(arm_dir)
    for pkl in sorted(glob.glob(os.path.join(arm_dir, "cp_*", "*_metrics_dict.pkl"))):
        eval_set = os.path.basename(pkl).replace("_metrics_dict.pkl", "")
        checkpoint = os.path.basename(os.path.dirname(pkl))
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception as e:
            print("skip", pkl, e)
            continue
        for t in range(13):
            k = f"{t:02d}_episode_successes"
            if k in d:
                rows.append(
                    dict(
                        arm=arm,
                        checkpoint=checkpoint,
                        eval_set=eval_set,
                        steps_to_think=t,
                        success=d[k],
                        zero_boxes=d.get(f"{t:02d}_episode_zero_boxes"),
                        noops_per_eps=d.get(f"{t:02d}_episode_num_noops_per_eps"),
                    )
                )

with open(out, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["arm", "checkpoint", "eval_set", "steps_to_think", "success", "zero_boxes", "noops_per_eps"],
    )
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out} from {len(set(r['arm'] for r in rows))} arms")
