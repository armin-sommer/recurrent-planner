# Paper — *Planning Emerges from Reinforcement Learning with Relational Latent States*

The integrated theory + empirical paper (ICLR format). Theory: a relational latent-state substrate, trained by
model-free RL, learns to **bind** latent cells to states, **recover the transition graph** `𝒩` by routing, and
run a **policy-evaluation operator** over that graph whose value field it largely *amortizes* — with the
**decision** refined at the head over thinking ticks (generalized policy iteration in latent space). Empirics
verify this on a fully-learned **attention** core and cross-check it on the convolutional **DRC(3,3)** planner.

## Build

```sh
tectonic planning_emerges_relational.tex      # from this directory; PDF is gitignored
```

Self-contained: the ICLR style (`iclr2026_conference.sty/.bst`), `math_commands.tex`, `fancyhdr.sty`,
`natbib.sty`, and `references.bib` are included. Notes:
- `references.bib` is **minimal** (just the cited keys) — swap in your fuller bib if you have one.
- The bundled `iclr2026_conference.sty` is used as-is (no `[preprint]` option); uncomment `\iclrfinalcopy` for
  camera-ready. A NeurIPS-style `ack` environment is defined in the preamble so the acknowledgments compile.

## Provenance of the empirical numbers

The experimental section's figures/table come from the mechanistic-interpretability probes in
[`../experiments/interp/`](../experiments/interp) run on the checkpoints in
[`../checkpoints/`](../checkpoints) (attention core) and the pretrained DRC(3,3)
(`AlignmentResearch/learned-planner :: drc33/bkynosqi/cp_2002944000`). See
[`../experiments/interp/README.md`](../experiments/interp/README.md) for the per-probe index. The standalone
graph-notation version of the empirical results is [`../writeup/planning_emergence.tex`](../writeup/planning_emergence.tex).
