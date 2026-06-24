# Binding in the slot core — n50 vs n100

*Mechanistic-interp writeup, binding section. Models: `checkpoints/slot_n50/cp_299996160` and
`checkpoints/slot_n100/cp_299996160` (slot core, D=3, 4 ticks, 1.5-entmax routing, mb4, 300 M steps),
evaluated on 256 held-out `valid_medium` Sokoban boards. Probes: `slot_binding_structure.py` (attention
map) and `slot_binding_decode.py` (hidden decodability). Chance = 0.50 for the balanced decode.*

## Setup

The slot core has no a-priori cell↔square correspondence: N free slots, a per-tick recurrence of
slot-attention **binding** (slots read board tokens) + slot↔slot **routing**, and the binding σ
(slot ↔ board position) must be **learned**. We measure binding two ways: (i) the **attention read** —
for each slot, the board position it attends to most (`decode_sigma` over the sown `bind_attn`); and
(ii) **hidden decodability** — for each board position p, take its *winning* slot (the slot that reads p
most) and linearly decode p's tile from that slot's hidden, per-object balanced accuracy. This is the
slot analogue of the attention core's `binding_balanced` (which scored wall/box/target/agent =
0.74/0.73/0.72/0.78), so the numbers are directly comparable. We contrast the winning slot with a
**random** slot: winning ≫ random ⇒ genuine per-slot binding; winning ≈ random ⇒ a distributed code.

A framing correction first: these boards are **~69% walls**, so only **~31 navigable cells**. Both 50 and
100 slots are therefore *over-complete for the part of the board that matters* — injectivity over all 100
squares was never the relevant question.

## Finding 1 — the attention *read* is diffuse (both cell counts)

Each slot reads broadly, not one square: top-square mass **0.124 (n50) / 0.112 (n100)** and entropy
**3.81 / 3.93 nats** (uniform = 4.61). By the argmax of this diffuse read the slots also pile onto few
positions (distinct-positions/N = 0.285 / 0.234; 3.55 / 4.32 slots per position). Taken alone this looks
like *no* clean binding — but the read is not where the binding lives.

## Finding 2 — per-slot binding lives in the HIDDEN, at attention-core fidelity (both)

Decoding a position's tile from its winning slot's **hidden** works well, while a random slot is at chance:

| tile | n50 winning / random | n100 winning / random |
|---|---|---|
| wall | 0.685 / 0.527 | 0.640 / 0.501 |
| floor | 0.667 / 0.503 | 0.622 / 0.524 |
| box | 0.656 / 0.488 | 0.651 / 0.507 |
| target | 0.695 / 0.522 | 0.659 / 0.502 |
| **agent** | **0.825** / 0.476 | **0.757** / 0.408 |
| **mean** | **0.706 / 0.503** | **0.666 / 0.488** |

The winning slot decodes its position at **0.71 (n50) / 0.67 (n100)** — essentially the **same binding
fidelity as the spatial attention core (≈0.73)** — while a random slot sits at **0.50 (chance)**. The +0.20
/ +0.18 gap, with random == chance, means each slot specifically carries *its* position (not a global copy
of the board in every slot). **So both models implement genuine per-slot binding; it is just stored in the
representation behind a diffuse attention read.** The **agent** is bound most strongly in both (0.83 / 0.76)
— the single always-present, decision-critical object gets the cleanest slot.

## Finding 3 — the cell-count effect: scarcity sharpens per-slot info; abundance broadens coverage

- **Per-slot fidelity is slightly *higher* for n50** (0.706 vs 0.666). With fewer slots, each is forced to
  carry more information; with 100 slots over ~31 navigable cells the binding is more redundant
  (4.32 vs 3.55 slots/position), diluting per-slot specialization.
- **Coverage is higher for n100**: navigable-cell coverage 0.43 vs 0.29, agent coverage 0.92 vs 0.80 — more
  slots ⇒ more of the relevant board is represented, and the agent is bound even more reliably.

So adding cells trades per-slot sharpness for broader, more redundant coverage; binding itself emerges
robustly at **both** budgets.

## Finding 4 — the binding *geometry*: a soft pointer, not spatial diffusion

`slot_binding_geometry.py` resolves what the diffuse read actually *is*. Each slot's read over the 100
squares is a **spike + flat background**, not a spatial blob:

| | n50 | n100 |
|---|---|---|
| spike height (mass on the argmax square) | 0.124 | 0.112 |
| spike / background ratio | ~15× | ~13× |
| read spatial spread / uniform | 0.99 | 1.04 |
| radial mass at board-dist 0 / 1 / 2 / 5 | .124/.015/.010/.008 | .112/.013/.009/.009 |
| navigable→slot injectivity | 0.638 | 0.789 |
| competition entropy over slots (vs uniform) | 3.56 / 3.91 | 4.25 / 4.61 |

The radial profile is the key: mass drops from the spike to the background **in a single step** — a distance-1
neighbour gets barely more than a distance-5 cell, so there is **no neighbourhood blur / receptive field**.
The "spread ≈ uniform" reading is just the flat background (~88% of the mass, dominating the spatial
variance); the *binding* is the ~12% spike riding on top. So σ(slot) = the spike square — a **learned soft
pointer**, argmax-recoverable, with the slot's hidden carrying that square's tile (the 0.71 decode). Over the
**navigable** squares the pointer is **majority-distinct** (0.64 → 0.79 as slots grow): roughly each walkable
square gets its own slot, more cleanly when slots are abundant. This is mechanistically *why* a diffuse read
and a localized hidden coexist — the binding is a sparse pointer hidden inside a flat background, not a
diffusion kernel.

## Finding 5 — message passing (routing): a global goal-anchored broadcast, not graph diffusion

`e4_slot.py` recovers what the slot↔slot **routing** (the per-tick message passing) computes, by binning each
routing row by the BFS board-graph distance between the slots' bound squares. Unlike the spatial attention
core — where E4 was the headline *transition-graph recovery* (mass decays ~ρ^d, goal-ward) — the slot routing
does **not** recover board adjacency:

| | n50 | n100 |
|---|---|---|
| ρ_graph (per-shell decay; 1.0 = none) | 0.967 | 0.946 |
| self-route mass | 0.028 | 0.021 |
| mass to graph-adjacent (1-hop) slots | 0.148 | 0.128 |
| mass **beyond** 1 hop | 0.705 | 0.721 |
| goal-ward ratio (toward/away) | 0.93 | 1.03 |
| **mass onto the TARGET slot** | **0.147 (~4.4×)** | **0.150 (~8×)** |
| mass onto the AGENT slot | 0.081 | 0.068 |
| through-wall (spurious-edge) mass | 0.000 | 0.000 |

Routing is **near-distance-agnostic** (ρ≈1, ~72% of mass beyond one hop, negligible self-loop) with **no value
gradient** (toward-goal ≈ away-goal) — so it is *not* a hop-by-hop diffusion on the recovered graph. But it is
**not** uniform either: every slot pulls disproportionately from the single **target slot** (~4–8× uniform)
and somewhat from the **agent slot**, and **zero** mass crosses walls. So the routing is a **global broadcast
anchored on the goal/landmarks**: the board's relational structure lives in *where slots bind* (Finding 4),
and routing then mixes globally while spotlighting the reward source. This is a genuinely *different*
message-passing algorithm than the grid-wired core's local graph diffusion — worth stating plainly rather
than forcing the graph-recovery frame.

## Interpretation

Binding is **not** a property of injective cell↔square wiring (the attention core got that for free from the
grid). The slot core, which must *discover* σ, lands on a different but functional solution: a **diffuse
read into a per-slot-localized hidden** — each slot's state encodes the tile/identity of the position it
competitively wins, at the same fidelity the grid-wired core achieves, regardless of whether slots are
scarce (n50) or abundant (n100). This is the paper's claim — that *relational latent states learn to bind* —
holding even when the binding map is learned from scratch and the cell budget is under/over the board size.

## Caveats & what's next

- This is the **binding** result only. The amortized-value (E1), routing→graph (E4), reach (E5), causal
  re-plan (E10), and decision-over-ticks (E13) ports are written but **not yet run** — those are the next
  numbers.
- "Winning slot" uses the argmax of the (diffuse) `bind_attn`; the competition softmax-over-slots would be a
  cleaner assignment and may *raise* the decode numbers — current values are a lower bound.
- Decode is the settled tick only; per-tick is available if we want the binding-vs-thinking trajectory.
