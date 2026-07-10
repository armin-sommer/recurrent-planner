"""Fig: solve rate vs thinking steps on valid_medium, ConvLSTM (DRC) vs AttnLSTM (ours).

Data: results/data/solve_vs_ticks.csv (n_active inner-thinking-depth sweep; whole episode run at depth K).
  AttnLSTM  = sokoban_drc_attn_vardepth_entmax_d3, cp_299996160 (300M).
  ConvLSTM  = pretrained DRC(3,3) AlignmentResearch/learned-planner drc33/bkynosqi/cp_2002944000 (2B).
Both measured this session on the same held-out valid_medium eval.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ticks, attn, drc = [], [], []
with open("results/data/solve_vs_ticks.csv") as f:
    for row in csv.DictReader(f):
        ticks.append(int(row["ticks"]))
        attn.append(float(row["attn_valid_medium"]))
        drc.append(float(row["drc_valid_medium"]))

BLUE, VERM = "#0072B2", "#D55E00"          # Okabe-Ito, CVD-safe, well-separated
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(4.4, 3.1))

ax.plot(ticks, drc, "-s", color=VERM, lw=2, ms=6, label="ConvLSTM / DRC(3,3), 2B (prior work)")
ax.plot(ticks, attn, "-o", color=BLUE, lw=2, ms=6, label="AttnLSTM (ours), 300M")

ax.set_xlabel("Thinking steps $K$")
ax.set_ylabel("Solve rate (valid-medium)")
ax.set_ylim(-0.02, 0.82)
ax.set_xlim(-0.2, 8.2)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 8])
ax.grid(True, alpha=0.3, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# direct end-labels at the plateau (identity beyond the legend)
ax.annotate(f"{drc[3]:.2f}", (3, drc[3]), textcoords="offset points", xytext=(4, 5), color=VERM, fontsize=9)
ax.annotate(f"{max(attn):.2f}", (4, max(attn)), textcoords="offset points", xytext=(4, 5), color=BLUE, fontsize=9)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
fig.tight_layout()
fig.savefig("results/figures/solve_vs_ticks.pdf")
fig.savefig("results/figures/solve_vs_ticks.png", dpi=200)
print("saved results/figures/solve_vs_ticks.{pdf,png}")
