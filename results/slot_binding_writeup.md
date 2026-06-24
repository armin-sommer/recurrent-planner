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

The radial profile is the key: mass drops from the ~15× spike almost to the flat background within one ring
(a faint shoulder remains — dist-1 ≈ 1.9×, dist-2 ≈ 1.25× the floor — a weak local component, *not* a wide
receptive field). The "spread ≈ uniform" reading is just the flat background (~88% of the mass, dominating the
spatial variance; for contrast a true local Gaussian blob simulates at spread ≈0.35); the *binding* is the
~12% spike riding on top. So σ(slot) = the spike square — a **learned soft pointer**, argmax-recoverable, with
the slot's hidden carrying that square's tile (the 0.71 decode). This is mechanistically *why* a diffuse read
and a localized hidden coexist — the binding is a sparse pointer inside a near-flat background, not a diffusion
kernel. (The 0.638/0.789 figures are the *transposed* cell→slot coverage — which slot owns each navigable
square — and rise with N even under random binding; see caveats.)

## Finding 5 — the binding RE-INDEXES per task; emergent agent-slot role

`slot_role_consistency.py` asks whether slot #k *means* the same thing across boards. Position is **content-
addressed / re-indexed**, with an emergent **role** prior for the agent (numbers vs the corrected
navigable-square null):

| | n50 | n100 |
|---|---|---|
| slots position-locked (fixed grid cell) | **0 / 50** | **0 / 100** |
| per-slot position spread / navigable-null | 1.05 | 1.13 |
| agent-slot index entropy (vs ln N) | **1.37 / 3.91** | 2.42 / 4.61 |
| agent: top-1 slot used on … of boards | **54 %** | 32 % |
| target-slot index entropy (all targets) | 3.20 / 3.91 | 3.67 / 4.61 |
| max P(a slot binds the agent) | 0.62 | 0.38 |

**No slot is a fixed grid cell** (0 position-locked; spread ≈ the random-navigable ceiling), so the slot↔square
map is rebuilt per board by content competition — *not* a static grid. But the network has learned a **role
prior**: there is a fairly **dedicated agent slot** (one index reads the agent on 54 % of n50 boards, index
entropy 1.37 ≪ 3.91), with weaker target specialization. n100 smears the agent role over more redundant slots.
This is the slot-attention signature — bind-by-content, re-index per input, with emergent specialization for
the salient recurring entity (the agent).

## Finding 6 — message passing (routing): a near-uniform global broadcast, not graph diffusion

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
| TARGET-slot enrichment (group-corrected) | **0.83×** | **1.04×** |
| AGENT-slot enrichment (group-corrected) | **0.94×** | **1.08×** |
| through-wall (spurious-edge) mass | 0.000 | 0.000 |

Routing is **near-distance-agnostic** (ρ≈1, ~72 % of mass beyond one hop, negligible self-loop) with **no value
gradient** (toward-goal ≈ away-goal) — so it is *not* a hop-by-hop diffusion on the recovered graph. Once the
landmark mass is compared to its **proportional group share** (≈3–4 slots bind each landmark square, so the
raw 0.15 must be divided by that group's fair share), the target/agent **enrichment is ≈1× — no spotlight**.
So the routing is simply a **near-uniform global broadcast** (the earlier "~4–8× goal-anchored" reading was a
single-slot-baseline artifact). The board's relational structure lives entirely in *where slots bind*
(Findings 4–5); routing just mixes globally, and **zero** mass crosses walls. This is a genuinely *different*
message-passing algorithm than the grid-wired core's local graph diffusion.

## Interpretation

Binding is **not** a property of injective cell↔square wiring (the attention core got that for free from the
grid). The slot core, which must *discover* σ, lands on a different but functional solution: a **soft-pointer
read into a per-slot-localized hidden** — the winning slot for a square carries more of that square than a
random slot does (so the slots are not identical board copies), the pointer re-indexes per board, and a global
broadcast mixes the slots. This supports the paper's claim — that *relational latent states learn to bind* —
even when σ is learned from scratch and the cell budget is under/over the board size; it just does **not**
recover the explicit transition graph in its routing the way the grid-wired core does.

## Caveats (post adversarial-verification)

- **Hidden-binding (Finding 2) is a lower bound on a real effect, not grid-fidelity.** The winning≫random gap
  rules out a fully-distributed/identical-copy code, but the winner is *selected* for max attention on its
  square and `m_bind = weights·v` makes it carry the most of that square by construction, so the gap is
  expected for *any* soft pointer. Read it as "slots are not identical copies; the winner carries more of its
  square," not "each slot cleanly carries its square." The 0.706/0.666 averages 5 classes **including floor**;
  the attn-core 0.72–0.78 averages 4 (floor excluded) — per-class the non-agent slot numbers sit ~3–8 pts
  below the attn core, so drop "at attention-core fidelity."
- **Geometry (Finding 4): not a *pure* delta.** There is a faint graded shoulder (dist-1 ≈ 1.9×, dist-2 ≈
  1.25× the floor) — a weak local component on top of the dominant ~15× spike; "drops to the floor in one
  step / no receptive field" was overstated. The floor ≈ 1/S is partly architectural (near-uniform slot
  competition). The 0.638/0.789 "injectivity" is the **transposed cell→slot** coverage (and rises with N even
  under random binding), not slot→cell σ-injectivity.
- **Cell-count (Finding 3): the contrast is not established.** The 0.706 vs 0.666 per-slot gap is 0.04 with no
  error bars across two single runs (plausibly noise); coverage / injectivity / slots-per-position rise with N
  even under random binding (N is in the numerator). Defensible core: **per-slot binding emerges at both
  budgets (~0.67–0.71 ≫ 0.50)**; "scarcity sharpens vs abundance broadens" needs seeds + an N-matched null.
- **Still to run:** the causal/temporal ports — reach-vs-ticks (E5), per-tick operator/amortized-value (E1),
  causal re-plan (E10), decision-over-ticks (E13) — which test whether information still *propagates over
  ticks* despite the non-local routing.
