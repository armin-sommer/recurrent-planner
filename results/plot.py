#!/usr/bin/env python
"""Plot the thinking-step sweep from results/data/thinking.csv.

Produces, per eval set (valid_medium = held-out generalization, train_unfiltered = train distribution):
  - figures/thinking_<set>.png : success vs extra thinking steps (0..12), one line per arm
  - figures/local_vs_dense_<set>.png : the matched cellwise mask contrast
and prints a summary table (success @0, @12, thinking lift) for RESULTS.md.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "thinking.csv")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# (arm, eval_set) -> {tick: success}
data = defaultdict(dict)
arms = []
for r in csv.DictReader(open(CSV)):
    key = (r["arm"], r["eval_set"])
    if r["arm"] not in arms:
        arms.append(r["arm"])
    data[key][int(r["steps_to_think"])] = float(r["success"])

eval_sets = sorted({k[1] for k in data})


def curve(arm, es):
    d = data.get((arm, es))
    if not d:
        return None, None
    xs = sorted(d)
    return xs, [d[x] for x in xs]


for es in eval_sets:
    plt.figure(figsize=(8, 6))
    for arm in arms:
        xs, ys = curve(arm, es)
        if xs:
            plt.plot(xs, ys, marker="o", ms=4, label=arm.replace("_200m", ""))
    plt.xlabel("extra thinking steps")
    plt.ylabel("success rate")
    plt.title(f"Thinking-time scaling — {es}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG, f"thinking_{es}.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", f"thinking_{es}.png")

# summary table
print("\narm,eval_set,succ@0,succ@12,lift(@12-@0)")
for arm in arms:
    for es in eval_sets:
        d = data.get((arm, es))
        if d and 0 in d:
            s0 = d.get(0)
            s12 = d.get(12, d[max(d)])
            print(f"{arm},{es},{s0:.3f},{s12:.3f},{s12 - s0:+.3f}")
