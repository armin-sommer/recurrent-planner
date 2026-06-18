# Probe run logs (raw stdout from the training pod)

The actual printed outputs of the mech-interp probes, captured on the RunPod H100 that ran them
(recovered before the node was deleted). These are the **provenance for the numbers in `paper/` and
`writeup/`** — each `PLOT_*=` line is the machine-readable result a figure/table is built from. All runs
are on the 300M attention checkpoint (`step=58593`) unless the name says `drc` (pretrained DRC(3,3), 2B).

Reproducible from the committed code + checkpoints: `python -m experiments.interp.<probe> --ckpt <cp_dir>`.

| log | probe | key result |
|-----|-------|------------|
| `e1_300m.log` | `e1.py` | operator stationary; amortized value |
| `bp_attn.log` | `box_prop_attn.py` | `‖ΔV_box‖→0`; own-A coeff `0.23→0.01` (value set by tick 1) |
| `bp_drc.log` | `box_prop_drc.py` | DRC box value likewise amortized |
| `e12_300m.log`, `e12_300m_b512.log` | `e12.py` | per-cell field flat across ticks (no frontier) |
| `e13_300m.log` | `e13.py` | decision changes 35%, goalward `+0.15` at far distance |
| `binding_300m.log`, `bindbal_300m.log`, `bindbal2_300m.log` | `binding_balanced.py` (+ precursors) | per-object balanced binding |
| `drc_box.log` | `drc_box_gpi.py` | DRC cellwise box value/plan (0.68/0.75/0.55) |
| `drc_causal.log` | `drc_causal.py` | DRC value physics-sensitive (2.5–5×) |
| `drc_gpi.log`, `drc_prop.log`, `drc_dl.log` | `drc_gpi.py` / `drc_propagation.py` / checkpoint download | DRC baselines + setup |
| `build.log`, `progress.log`, `progress_vardepth.log` | env build / training progress | provenance only |
