import dataclasses
from dataclasses import field
from pathlib import Path
from typing import List, Literal, Optional

from cleanba.attn_lstm import AttentionCellConfig, AttentionLSTMConfig
from cleanba.cellwise_lstm import CellwiseLSTMCellConfig, CellwiseLSTMConfig
from cleanba.slot_lstm import SlotCellConfig, SlotLSTMConfig
from cleanba.pool_inject_lstm import PoolInjectCellConfig, PoolInjectLSTMConfig
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig
from cleanba.environments import AtariEnv, BoxWorldConfig, EnvConfig, EnvpoolBoxobanConfig, MiniPacManConfig, MiniWorldConfig, random_seed
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import (
    ImpalaLossConfig,
)
from cleanba.network import AtariCNNSpec, GuezResNetConfig, IdentityNorm, PolicySpec, SokobanResNetConfig


@dataclasses.dataclass
class Args:
    train_env: EnvConfig = dataclasses.field(  # Environment to do training, including seed
        # default_factory=lambda: SokobanConfig(
        #     asynchronous=False, max_episode_steps=40, num_envs=64, tinyworld_obs=True, dim_room=(5, 5), num_boxes=1
        # )
        default_factory=lambda: AtariEnv(env_id="Breakout-v5"),
    )
    eval_envs: dict[str, EvalConfig] = dataclasses.field(  # How to evaluate the algorithm? Including envs and seeds
        default_factory=lambda: dict(eval=EvalConfig(AtariEnv(env_id="Breakout-v5", num_envs=128)))
    )
    eval_at_steps: frozenset[int] = frozenset(
        [195 * i for i in range(1, 10)]
        + [1950 * i for i in range(10)]
        + [19500 * i for i in range(10)]
        + [195000 * i for i in range(10)]
    )

    seed: int = dataclasses.field(default_factory=random_seed)  # A seed to make the experiment deterministic

    save_model: bool = True  # whether to save model into the wandb run folder
    log_frequency: int = 10  # the logging frequency of the model performance (in terms of `updates`)
    sync_frequency: int = (
        400  # how often to copy the first learner's parameters to all of them, with multiple learner devices.
    )

    actor_update_frequency: int = (
        1  # Update the actor every `actor_update_frequency` steps, until `actor_update_cutoff` is reached.
    )
    actor_update_cutoff: int = int(1e9)  # After this number of updates, update the actors every step

    base_run_dir: Path = Path("/tmp/cleanba")

    loss: ImpalaLossConfig = ImpalaLossConfig()

    net: PolicySpec = AtariCNNSpec(channels=(16, 32, 32), mlp_hiddens=(256,))

    # Algorithm specific arguments
    total_timesteps: int = 100_000_000  # total timesteps of the experiments
    # Variable thinking depth. If (lo, hi), the actor samples the recurrent thinking-depth budget
    # uniformly in {lo..hi} per rollout cycle (and the learner replays at that depth), training the
    # recurrence to benefit from test-time thinking. None = fixed depth (= repeats_per_step).
    variable_thinking_depth: Optional[tuple[int, int]] = None
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    final_learning_rate: float = 0.0  # The learning rate at the end of training
    warmup_updates: int = 0  # linear LR warmup over this many outer updates (0 = off). Stabilizes attention.
    local_num_envs: int = 64  # the number of parallel game environments for every actor device
    num_steps: int = 20  # the number of steps to run in each environment per policy rollout
    train_epochs: int = 1  # Repetitions of going through the collected training
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    num_minibatches: int = 4  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    max_grad_norm: float = 0.0625  # the maximum norm for the gradient clipping
    optimizer: str = "rmsprop"
    adam_b1: float = 0.9
    rmsprop_eps: float = 1.5625e-05
    rmsprop_decay: float = 0.99
    optimizer_yang: bool = False
    base_fan_in: int = 3 * 3 * 32

    queue_timeout: float = 300.0  # If any of the actor/learner queues takes at least this many seconds, crash training.

    num_actor_threads: int = 2  # The number of environment threads per actor device
    actor_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that actor workers will use
    learner_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that learner workers will use
    distributed: bool = False  # whether to use `jax.distributed`
    concurrency: bool = True  # whether to run the actor and learner concurrently
    learner_policy_version: int = 0  # learner policy version that is updated every outer iteration of training

    load_path: Optional[Path] = None  # Where to load the initial training state from

    finetune_with_noop_head: bool = False  # Whether to finetune the model with a noop head
    frozen_finetune_steps_ratio: float = 0.5  # fraction of steps to finetune ONLY the head of model with new noop action


def sokoban_resnet() -> Args:
    CACHE_PATH = Path("/opt/sokoban_cache")
    return Args(
        train_env=EnvpoolBoxobanConfig(
            max_episode_steps=120,
            min_episode_steps=120 * 3 // 4,
            num_envs=1,
            cache_path=CACHE_PATH,
            split="train",
            difficulty="unfiltered",
        ),
        eval_envs=dict(
            valid_unfiltered=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="unfiltered",
                ),
                n_episode_multiple=2,
            ),
            test_unfiltered=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="test",
                    difficulty="unfiltered",
                ),
                n_episode_multiple=2,
            ),
            train_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="train",
                    difficulty="medium",
                ),
                n_episode_multiple=2,
            ),
            valid_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="medium",
                ),
                n_episode_multiple=2,
            ),
        ),
        seed=1234,
        save_model=False,
        log_frequency=10,
        sync_frequency=int(4e9),
        net=SokobanResNetConfig(),
        total_timesteps=int(1e9),
    )


def sokoban_drc(n_recurrent: int, num_repeats: int) -> Args:
    CACHE_PATH = Path("/opt/sokoban_cache")
    return Args(
        train_env=EnvpoolBoxobanConfig(
            max_episode_steps=120,
            min_episode_steps=120 * 3 // 4,
            num_envs=1,
            cache_path=CACHE_PATH,
            split="train",
            difficulty="unfiltered",
        ),
        eval_envs=dict(
            test_unfiltered=EvalConfig(
                EnvpoolBoxobanConfig(
                    seed=5454,
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="test",
                    difficulty="unfiltered",
                ),
                n_episode_multiple=2,
            ),
            valid_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    seed=5454,
                    max_episode_steps=240,
                    min_episode_steps=240,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="medium",
                ),
                n_episode_multiple=2,
                steps_to_think=[0, 2, 4, 8],
            ),
        ),
        log_frequency=10,
        net=ConvLSTMConfig(
            embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
            recurrent=ConvLSTMCellConfig(
                ConvConfig(32, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal", fence_pad="same"
            ),
            n_recurrent=n_recurrent,
            mlp_hiddens=(256,),
            repeats_per_step=num_repeats,
        ),
        loss=ImpalaLossConfig(
            vtrace_lambda=0.97,
            weight_l2_coef=1.5625e-07,
            gamma=0.97,
            logit_l2_coef=1.5625e-05,
        ),
        actor_update_cutoff=100000000000000000000,
        sync_frequency=100000000000000000000,
        num_minibatches=8,
        rmsprop_eps=1.5625e-07,
        local_num_envs=256,
        total_timesteps=300001280,  # ~300M, batch-aligned: 58594 updates * (local_num_envs 256 * num_steps 20)
        base_run_dir=Path("/training/cleanba"),
        learning_rate=0.0004,
        optimizer="adam",
        base_fan_in=1,
        anneal_lr=True,
        max_grad_norm=0.015,
        num_actor_threads=1,
    )


# fmt: off
def sokoban_drc_3_3(): return sokoban_drc(3, 3)
def sokoban_drc_1_1(): return sokoban_drc(1, 1)
# fmt: on


def sokoban_drc_attn(
    n_recurrent: int,
    num_repeats: int,
    use_attention_mask: bool = True,
    readout: Literal["softmax", "maxplus"] = "maxplus",
    mask_neighborhood: Literal["king", "vonneumann"] = "king",
    directional_value: bool = True,   # ON by default: per-offset value routing (VIN/conv-aligned)
    relative_key: bool = True,        # ON by default: content x direction in the score
    attend_inputs: bool = True,       # ON by default: live obs in the attention source (DRC conv_ih analog)
    attn_norm: Literal["softmax", "entmax15", "sparsemax"] = "softmax",  # readout="softmax" only: sparse weight normalizer
) -> Args:
    """Same setup as `sokoban_drc`, but with the state-indexed masked-attention core instead of ConvLSTM.

    `use_attention_mask=True` (default) restricts each cell to its grid neighbourhood (provable
    locality); `False` uses full dense attention. `readout` defaults to "maxplus" (the VIN-aligned
    soft Bellman max-plus aggregator); pass `readout="softmax"` for the original convex-average
    attention. `mask_neighborhood` chooses the grid stencil ("king" = 8-neighbourhood, "vonneumann" =
    the 4 one-move-reachable cells). `Args.net` is mutable, so we reuse the DRC env/loss/optimizer
    setup and just swap the backbone.
    """
    # relative_key only takes effect in the directional path; make the incongruence loud, not silent.
    assert directional_value or not relative_key, "relative_key=True requires directional_value=True"
    args = sokoban_drc33_59()  # inherit the PROVEN recipe (grad clip 2.5e-4, vtrace_lambda 0.5,
    # light L2, valid_medium eval), then swap in the attention/cellwise backbone below.
    # (Previously based on the weaker `sokoban_drc` recipe, which confounded the DRC comparison.)
    args.net = AttentionLSTMConfig(
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=AttentionCellConfig(
            features=32,
            num_heads=4,
            use_attention_mask=use_attention_mask,
            mask_neighborhood=mask_neighborhood,
            n_global=0,  # pure local: each cell attends ONLY to its king-neighbours + self (no global/register tokens)
            readout=readout,
            directional_value=directional_value,  # per-offset value projection (VIN/conv-aligned routing)
            relative_key=relative_key,             # per-offset relative key (content x direction in the score)
            attend_inputs=attend_inputs,           # live obs folded into q/k/v source (DRC conv_ih analog)
            attn_norm=attn_norm,                   # softmax | entmax15 | sparsemax (sparse = learned support)
        ),
        n_recurrent=n_recurrent,
        mlp_hiddens=(256,),
        repeats_per_step=num_repeats,
    )
    # Lighten the DURING-training eval (same as the cellwise matrix): 2 ticks + sparse points.
    # Full steps_to_think sweep is recovered post-hoc via cleanba.load_and_eval on the checkpoints.
    args.eval_envs["valid_medium"].steps_to_think = [0, 8]
    _bs = 256 * 20  # local_num_envs * num_steps
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(10e6 / _bs) * i for i in range(1, 21)])
    return args


# fmt: off
def sokoban_drc_attn_3_3(): return sokoban_drc_attn(3, 3)                                         # FULL model: maxplus, king, directional value + relative key (defaults ON)
def sokoban_drc_attn_3_3_plain(): return sokoban_drc_attn(3, 3, directional_value=False, relative_key=False)  # shared-W_v dense-value baseline (ablation)
def sokoban_drc_attn_3_3_dir(): return sokoban_drc_attn(3, 3, directional_value=True)             # + per-offset value routing (VIN-aligned)
def sokoban_drc_attn_3_3_dir_relk(): return sokoban_drc_attn(3, 3, directional_value=True, relative_key=True)  # + content x direction key
def sokoban_drc_attn_3_3_dir_softmax(): return sokoban_drc_attn(3, 3, readout="softmax", directional_value=True)
def sokoban_drc_attn_1_1(): return sokoban_drc_attn(1, 1)
def sokoban_drc_attn_3_3_vn(): return sokoban_drc_attn(3, 3, mask_neighborhood="vonneumann")      # maxplus, von Neumann
def sokoban_drc_attn_3_3_softmax(): return sokoban_drc_attn(3, 3, readout="softmax")              # old convex-average attention
def sokoban_drc_attn_3_3_nomask(): return sokoban_drc_attn(3, 3, use_attention_mask=False, directional_value=False, relative_key=False)  # dense attention (no mask); directional needs the mask so it's off here
# fmt: on


def sokoban_drc_attn_vardepth():
    """Spatial DENSE attention (no mask), softmax readout, ONE weight-tied cell ITERATED, trained with
    VARIABLE thinking depth (d ~ U{0..6} sampled per rollout cycle; learner replays at that depth).

    The variable depth forces the recurrence to be a monotone-improvement operator across depths, so
    it should benefit from test-time thinking (unlike the fixed-depth dense arms, which were flat).
    This is the substrate for the three goals: (1) binding to env states under attention, (2) a LEARNED
    mask to reachable neighbours (does dense attention concentrate on N?), (3) emergent planning.
    repeats_per_step=6 is the max depth K_max (the scan always runs 6 ticks; depths past d are
    identity), so the eval can sweep thinking depth 0..6 in-distribution.
    """
    args = sokoban_drc_attn(1, 6, use_attention_mask=False, readout="softmax",
                            directional_value=False, relative_key=False)
    args.variable_thinking_depth = (0, 6)
    args.total_timesteps = 200_000_000
    return args


def sokoban_drc_attn_vardepth_600m():
    """Follow-up run: like sokoban_drc_attn_vardepth, but depth d ~ U{1..6} (ALWAYS >=1 tick -- the
    recurrent core runs at least once every step; no pure-reactive d=0) and 600M steps. Eval/checkpoint
    points extended to cover the full 600M so checkpoints are saved throughout for interp."""
    args = sokoban_drc_attn(1, 6, use_attention_mask=False, readout="softmax",
                            directional_value=False, relative_key=False)
    args.variable_thinking_depth = (1, 6)
    args.total_timesteps = 600_000_000
    _bs = 256 * 20
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(30e6 / _bs) * i for i in range(1, 21)])
    return args


def sokoban_drc_attn_vardepth_entmax_d3():
    """Planning core (current best guess). Two changes over `sokoban_drc_attn_vardepth`, each fixing
    one cause our 200M post-mortem identified:

      (1) SPARSE attention via 1.5-entmax (attn_norm="entmax15") on DENSE (unmasked) attention -> the
          model learns its OWN hard sparse support (exact-zero weights = a learned neighbour graph),
          fixing the diffuse/over-smoothed routing. No auxiliary loss, no coefficient -- the sparsity
          is architectural (entmax), not a penalty. Replaces the reverted entropy-penalty approach.
      (2) n_recurrent=3 (D=3, matching DRC) -> each thinking tick is a deep-enough operator to be
          worth iterating, fixing the per-tick under-capacity suspected behind the flat thinking-curve.

    Keeps the variable thinking depth d ~ U{1..6} (>=1 tick; gradients flow through all d ticks,
    verified by tests/check_vardepth_grad.py). The planning claim still rests on the thinking-curve
    over d (the N-loop); D=3 should unblock it and entmax should sharpen recovery-of-N to a hard mask.
    ~3x the per-step compute of the D=1 vardepth (3 cells/tick), so expect proportionally lower SPS.
    """
    args = sokoban_drc_attn(3, 6, use_attention_mask=False, readout="softmax",
                            directional_value=False, relative_key=False, attn_norm="entmax15")
    args.variable_thinking_depth = (1, 6)
    args.total_timesteps = 300_000_000
    _bs = 256 * 20  # checkpoints at 2M, 5M, then every 20M to 300M (15 pts) for the thinking-curve sweep
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(20e6 / _bs) * i for i in range(1, 16)])
    return args


def sokoban_drc_slots(
    n_recurrent: int = 1,
    num_repeats: int = 3,
    num_slots: int = 100,
    routing_readout: Literal["softmax", "maxplus"] = "softmax",
    routing_norm: Literal["softmax", "entmax15", "sparsemax"] = "softmax",
    num_heads: int = 4,
    binding: Literal["content", "positional"] = "content",
) -> Args:
    """Same setup as `sokoban_drc`, but with the LEARNABLE-SLOT relational core (cleanba.slot_lstm).

    The N latent cells are *free slots* with no spatial position: the binding sigma (slot<->state) is
    discovered by a slot-attention competition over the H*W board tokens, and the routing graph N is
    learned by DENSE slot<->slot attention with no spatial/positional prior. This is the strongest
    form of the paper's thesis -- the conv/attention cores hand sigma (and, masked, N) to the network
    via the grid; this core must discover BOTH from pure RL signal. `num_slots=100` matches the 10x10
    board for a 1:1 capacity (cleanest binding). `skip_final=False` is REQUIRED: the base would
    otherwise add the spatial embed map to the (B,N,d) slot output (a shape error).
    """
    args = sokoban_drc33_59()  # inherit the PROVEN recipe (grad clip 2.5e-4, vtrace_lambda 0.5,
    # light L2, valid_medium eval), then swap in the slot backbone below.
    args.net = SlotLSTMConfig(
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=SlotCellConfig(
            features=32,
            num_slots=num_slots,
            num_heads=num_heads,
            routing_readout=routing_readout,
            routing_norm=routing_norm,
            binding=binding,
        ),
        n_recurrent=n_recurrent,
        mlp_hiddens=(256,),
        repeats_per_step=num_repeats,
        skip_final=False,  # CRITICAL: (B,N,d) slot carry can't accept the base's spatial embed skip-add
    )
    # Lighten the DURING-training eval (same as the attn/cellwise arms): 2 ticks + sparse points.
    args.eval_envs["valid_medium"].steps_to_think = [0, 8]
    _bs = 256 * 20  # local_num_envs * num_steps
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(10e6 / _bs) * i for i in range(1, 21)])
    return args


# fmt: off
def sokoban_drc_slots_softmax(): return sokoban_drc_slots(routing_readout="softmax")  # headline: pure-RL discovered binding + N, softmax routing
def sokoban_drc_slots_maxplus(): return sokoban_drc_slots(routing_readout="maxplus")  # VIN-aligned soft Bellman max-plus routing variant


def _slots_d3_fixed4(num_slots: int, binding: Literal["content", "positional"] = "content") -> Args:
    """Latent-cell-count sweep on the slot core: D=3, FIXED 4 thinking ticks, dense slot<->slot routing
    with 1.5-entmax (exact-zero weights => a hard learned graph N, matching the attention planning core),
    softmax binding competition. `num_slots` varies the number of free latent cells against the
    100-square board (the binding-capacity question). `binding="positional"` (D1) swaps the content-
    addressed binding for a learned but task-STABLE slot<->position template (see SlotCellConfig.binding).
    No `variable_thinking_depth` => the scan runs exactly 4 ticks every env step; the during-training eval
    still sweeps thinking 0..8 (steps_to_think inherited from sokoban_drc_slots) to test reactive (0) and
    extrapolation (8). Same 300M schedule + checkpoint ladder (2M, 5M, then every 20M to 300M) as the d3
    planning core, so the probes transfer."""
    args = sokoban_drc_slots(n_recurrent=3, num_repeats=4, num_slots=num_slots,
                             routing_readout="softmax", routing_norm="entmax15", binding=binding)
    args.total_timesteps = 300_000_000
    _bs = 256 * 20  # checkpoints at 2M, 5M, then every 20M to 300M (15 pts)
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(20e6 / _bs) * i for i in range(1, 16)])
    return args


# fmt: off
def sokoban_drc_slots_d3_fixed4_n100(): return _slots_d3_fixed4(100)  # 1.0x cells: 100 slots == 100 board states (1:1)
def sokoban_drc_slots_d3_fixed4_n200(): return _slots_d3_fixed4(200)  # 2.0x cells: over-complete (redundant slots)
def sokoban_drc_slots_d3_fixed4_n50():  return _slots_d3_fixed4(50)   # 0.5x cells: under-complete (slots must share)
# D1: learned-but-task-STABLE positional binding (midpoint between dense attention's GIVEN cell=square and
# the slot core's per-task content-addressed sigma). n100 = 1:1 capacity match to dense attention so the only
# difference vs the content-binding n100 is the addressing -> isolates "does a stable binding restore graph routing".
def sokoban_drc_slots_d3_fixed4_n100_posbind(): return _slots_d3_fixed4(100, binding="positional")
# fmt: on


# ==================================================================================================
# POOL-AND-INJECT relational core (cleanba.pool_inject_lstm): the encoder is POOLED to one global vector
# and INJECTED identically into every cell (no positional binding); cells differentiate only via dense
# cell<->cell attention routing + a learnable per-cell identity. The natural core for first-person /
# partial observation. Run on BOTH Sokoban (positional-binding-free control) and MiniWorld (the target).
# ==================================================================================================
def sokoban_drc_poolinject(n_recurrent: int = 3, num_repeats: int = 4, num_cells: int = 100,
                           routing_norm: Literal["softmax", "entmax15", "sparsemax"] = "entmax15",
                           num_heads: int = 4) -> Args:
    """Pool-and-inject core on Sokoban -- the positional-binding-free CONTROL for the first-person runs.
    Same proven recipe as `sokoban_drc_slots`, swapping the slot binding (competitive per-position read)
    for pool-and-inject (one global vector to every cell). The cells get NO spatial layout, only a pooled
    summary, so on Sokoban this is expected to be a weaker planner than the positional dense/slot cores --
    that gap is the point of the comparison."""
    args = sokoban_drc33_59()  # inherit the proven recipe (grad clip, vtrace, light L2, valid_medium eval)
    args.net = PoolInjectLSTMConfig(
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=PoolInjectCellConfig(
            features=32, num_cells=num_cells, num_heads=num_heads, routing_norm=routing_norm,
        ),
        n_recurrent=n_recurrent, mlp_hiddens=(256,), repeats_per_step=num_repeats,
        skip_final=False,  # CRITICAL: (B,N,d) carry can't accept the base's spatial embed skip-add
    )
    args.eval_envs["valid_medium"].steps_to_think = [0, 8]
    args.total_timesteps = 300_000_000
    _bs = 256 * 20
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(20e6 / _bs) * i for i in range(1, 16)])
    return args


def sokoban_poolinject_d3_fixed4_4gpu_mb4(num_cells: int = 100) -> Args:
    """Sokoban pool-inject, D=3, FIXED 4 ticks, 1.5-entmax routing, on 4 GPUs (2 actor [0,1] + 2 learner
    [2,3], mb4) -- the proven slot-run layout. batch 5120 / num_envs 256 / 300M, comparable to the slot runs."""
    args = sokoban_drc_poolinject(n_recurrent=3, num_repeats=4, num_cells=num_cells, routing_norm="entmax15")
    args.local_num_envs = 128
    args.num_actor_threads = 1
    args.actor_device_ids = [0, 1]
    args.learner_device_ids = [2, 3]
    args.num_minibatches = 4
    return args


def miniworld_poolinject_d3_fixed4(env_id: str = "MiniWorld-MazeS3Fast-v0", num_cells: int = 100,
                                   local_num_envs: int = 32,
                                   actor_device_ids=(0,), learner_device_ids=(0,)) -> Args:
    """First-person MiniWorld with the pool-and-inject core, D=3, FIXED 4 ticks, 1.5-entmax routing.

    num_cells is set to the ENV STATE-SPACE size (the navigable positions the agent plans over), NOT the
    visual-input dims: MazeS3 is a 9.5x9.5 maze => ~100 navigable unit-cells (measured), so num_cells=100,
    a 1:1 binding capacity exactly like Sokoban's 100 board squares. (FourRooms would be ~208; scale
    num_cells with the maze if env_id changes.) Egocentric (60x80x3) pixels, Discrete(3) nav. The encoder
    downsamples 60x80 to a ~8x10 token map purely for perception; the pool over those tokens is size-
    agnostic, so the visual resolution does NOT set num_cells. Default layout is single-GPU (actor+learner
    share device 0) since MiniWorld is OpenGL-render-bound, not learner-bound."""
    args = sokoban_drc_poolinject(n_recurrent=3, num_repeats=4, num_cells=num_cells, routing_norm="entmax15")
    # downsampling embed for 60x80 (Sokoban's stride-1 4x4 embed would leave a 60x80=4800-token map)
    args.net = dataclasses.replace(args.net, embed=[
        ConvConfig(32, (8, 8), (4, 4), "SAME", True),   # 60x80 -> 15x20
        ConvConfig(32, (4, 4), (2, 2), "SAME", True),   # 15x20 -> 8x10
        ConvConfig(32, (3, 3), (1, 1), "SAME", True),   # 8x10
    ])
    # headless=False => pyglet X11 backend; the run MUST be launched under `xvfb-run` (NVIDIA EGL works in
    # the main process but FAILS in spawned AsyncVectorEnv workers -- eglChooseConfig NoSuchConfig -- so a
    # shared xvfb virtual display is the working multiprocess path; ~3.7k env-steps/s, validated on the pod).
    args.train_env = MiniWorldConfig(env_id=env_id, max_episode_steps=300, asynchronous=True, headless=False)
    args.eval_envs = dict(
        miniworld=EvalConfig(
            MiniWorldConfig(env_id=env_id, num_envs=16, max_episode_steps=300, asynchronous=True, headless=False),
            n_episode_multiple=1, steps_to_think=[0, 4],
        )
    )
    nl = len(learner_device_ids)
    assert local_num_envs % nl == 0 and (local_num_envs // nl) % 4 == 0, "layout violates divisibility"
    args.local_num_envs = local_num_envs
    args.num_actor_threads = 1
    args.actor_device_ids = list(actor_device_ids)
    args.learner_device_ids = list(learner_device_ids)
    args.num_minibatches = 4
    return args


def _slots_d3_fixed4_4gpu(num_slots: int) -> Args:
    """Same run as `_slots_d3_fixed4` but split across 4 GPUs on one node for ~3-4x wall-clock: 2 actor
    devices [0,1] + 2 learner devices [2,3] (the learner shards via pmap). `local_num_envs` is halved to
    128 so num_envs (256), the batch (128*20*1*2 = 5120) and total steps (300M) are IDENTICAL to the
    1-GPU config -- only wall-clock changes, keeping the runs directly comparable. Decouples the env
    rollout from the gradient step (the 1-GPU config time-shares both on GPU0). Divisibility:
    local_num_envs 128 % len(learner_ids) 2 == 0; int(128/2)*1 % num_minibatches 8 == 0."""
    args = _slots_d3_fixed4(num_slots)
    args.local_num_envs = 128
    args.num_actor_threads = 1
    args.actor_device_ids = [0, 1]
    args.learner_device_ids = [2, 3]
    return args


# fmt: off
def sokoban_drc_slots_d3_fixed4_n100_4gpu(): return _slots_d3_fixed4_4gpu(100)
def sokoban_drc_slots_d3_fixed4_n200_4gpu(): return _slots_d3_fixed4_4gpu(200)
def sokoban_drc_slots_d3_fixed4_n50_4gpu():  return _slots_d3_fixed4_4gpu(50)
# fmt: on


def _slots_d3_fixed4_4gpu_bigl(num_slots: int) -> Args:
    """NEGATIVE RESULT (kept for the record): puts the LEARNER on ALL 4 devices, sharing the 2 idle actor
    GPUs (actor_device_ids=[0,1], learner_device_ids=[0,1,2,3], local_num_envs=128; batch/envs/steps
    unchanged & faithful). Hypothesis was ~2x -- the actor GPUs sit at ~0% util (rollout ~0.18s) while the
    2-GPU learner is pegged at 100%. MEASURED on the n50 pod: 5688 SPS vs the 2+2 baseline of 5688 -- ZERO
    speedup, and the 4-way shard load-balances worse (a learner GPU drops to 0%). The model is small
    enough that each update is dominated by the SEQUENTIAL recurrent unroll (4 ticks x 3 depth = 12 small-
    kernel cell applications) -- launch/sync overhead, not parallelizable learner compute -- so more
    learner sharding can't help. The 2+2 split (`_slots_d3_fixed4_4gpu`) is the ~18h floor on 4 GPUs.
    Divisibility: 128 % 4 == 0; (128/4)*1 % 8 == 0."""
    args = _slots_d3_fixed4_4gpu(num_slots)   # actor[0,1], learner[2,3], lne=128
    args.learner_device_ids = [0, 1, 2, 3]
    return args


# fmt: off
def sokoban_drc_slots_d3_fixed4_n100_4gpu_bigl(): return _slots_d3_fixed4_4gpu_bigl(100)
def sokoban_drc_slots_d3_fixed4_n200_4gpu_bigl(): return _slots_d3_fixed4_4gpu_bigl(200)
def sokoban_drc_slots_d3_fixed4_n50_4gpu_bigl():  return _slots_d3_fixed4_4gpu_bigl(50)
# fmt: on


def _slots_d3_fixed4_4gpu_mb4(num_slots: int) -> Args:
    """4-GPU 2-actor/2-learner layout (`_slots_d3_fixed4_4gpu`) but with num_minibatches 8 -> 4: the
    learner does HALF as many sequential minibatch passes per update (each 2x larger), which should cut
    the dominant per-update overhead ~1.3-1.5x on this small launch-bound model. This is a RECIPE change
    (4 SGD steps/update instead of 8 -> different optimization), not just a layout change. Batch 5120 /
    num_envs 256 / 300M unchanged. Divisibility: (128/2)*1 % 4 == 0."""
    args = _slots_d3_fixed4_4gpu(num_slots)
    args.num_minibatches = 4
    return args


def _slots_d3_fixed4_4gpu_ns10(num_slots: int) -> Args:
    """num_steps 20 -> 10 with local_num_envs 128 -> 256: trades the learner's SEQUENTIAL temporal replay
    depth (num_steps x 4 ticks x 3 depth cell-apps -- the real launch/scan bottleneck) for parallel env
    width. 2+2 layout; batch = 256*10*2 = 5120 and 300M held (num_envs grows 256 -> 512). Likely the
    biggest single lever, BUT num_steps is also the V-trace credit horizon -- halving it risks hurting
    long-horizon Sokoban learning, so watch returns. Divisibility: 256 % 2 == 0; (256/2)*1 % 8 == 0."""
    args = _slots_d3_fixed4_4gpu(num_slots)   # actor[0,1], learner[2,3], lne=128, ns=20
    args.num_steps = 10
    args.local_num_envs = 256
    return args


def _slots_layout(num_slots: int, n_actor: int, n_learner: int, num_minibatches: int = 4) -> Args:
    """General device layout for the slot runs (e.g. for 4090 pods with more actor GPUs). Actors on
    devices [0..n_actor), learners on the next n_learner. local_num_envs = 256/n_actor keeps num_envs=256
    and batch=5120 FAITHFUL; mb4 by default. Use n_actor=4,n_learner=2 (6 GPUs) for n200 to relieve its
    actor-inference bottleneck. n_actor MUST divide 256 (2/4/8); the asserts below enforce the learner-
    shard + minibatch divisibility."""
    assert 256 % n_actor == 0, f"n_actor={n_actor} must divide 256"
    lne = 256 // n_actor
    assert lne % n_learner == 0 and (lne // n_learner) % num_minibatches == 0, "layout violates shard/minibatch divisibility"
    args = _slots_d3_fixed4(num_slots)              # ns=20, 300M schedule
    args.local_num_envs = lne
    args.num_actor_threads = 1
    args.actor_device_ids = list(range(n_actor))
    args.learner_device_ids = list(range(n_actor, n_actor + n_learner))
    args.num_minibatches = num_minibatches
    return args


# fmt: off
def sokoban_drc_slots_d3_n200_4a2l(): return _slots_layout(200, 4, 2)  # 6 GPUs: relieve n200 actor bottleneck
def sokoban_drc_slots_d3_n200_8a2l(): return _slots_layout(200, 8, 2)  # 10 GPUs: n200 max-effort
def sokoban_drc_slots_d3_n100_4a2l(): return _slots_layout(100, 4, 2)  # 6 GPUs (n100 unlikely to need it)
def sokoban_drc_slots_d3_n50_4a2l():  return _slots_layout(50, 4, 2)
def sokoban_drc_slots_d3_n150_4a2l(): return _slots_layout(150, 4, 2)  # 6 GPUs: 1.5x cells (third run, 150 vs 200)
def sokoban_drc_slots_d3_n150_1a2l(): return _slots_layout(150, 1, 2)  # 3 GPUs: n150 is learner-bound (actors ~idle at 4a), 1 actor suffices
# fmt: on


def _slots_d3_2gpu(num_slots: int) -> Args:
    """2-GPU layout to test whether we can DROP GPUs without losing throughput: 1 actor (device 0) + 1
    learner (device 1), local_num_envs=256 -> num_envs 256, batch 5120, 300M unchanged (faithful). Since
    learner-sharding past 1 GPU showed no gain (the `bigl` negative result), 2 decoupled GPUs may match
    the 4-GPU 2+2 throughput at half the cost. Divisibility: 256 % 1 == 0; (256/1)*1 % 8 == 0."""
    args = _slots_d3_fixed4(num_slots)        # local_num_envs=256, num_actor_threads=1
    args.actor_device_ids = [0]
    args.learner_device_ids = [1]
    return args


# fmt: off
def sokoban_drc_slots_d3_fixed4_n100_mb4(): return _slots_d3_fixed4_4gpu_mb4(100)
def sokoban_drc_slots_d3_fixed4_n200_mb4(): return _slots_d3_fixed4_4gpu_mb4(200)
def sokoban_drc_slots_d3_fixed4_n50_mb4():  return _slots_d3_fixed4_4gpu_mb4(50)
def sokoban_drc_slots_d3_fixed4_n150_mb4(): return _slots_d3_fixed4_4gpu_mb4(150)
def sokoban_drc_slots_d3_fixed4_n100_2gpu(): return _slots_d3_2gpu(100)
def sokoban_drc_slots_d3_fixed4_n200_2gpu(): return _slots_d3_2gpu(200)
def sokoban_drc_slots_d3_fixed4_n50_2gpu():  return _slots_d3_2gpu(50)
def sokoban_drc_slots_d3_fixed4_n100_ns10(): return _slots_d3_fixed4_4gpu_ns10(100)
def sokoban_drc_slots_d3_fixed4_n200_ns10(): return _slots_d3_fixed4_4gpu_ns10(200)
def sokoban_drc_slots_d3_fixed4_n50_ns10():  return _slots_d3_fixed4_4gpu_ns10(50)
# fmt: on


def _slots_d3_fixed4_4gpu_mb4(num_slots: int) -> Args:
    """2+2 4-GPU split with num_minibatches=4 (vs 8): half as many sequential learner grad passes per
    update (minibatch_size 640->1280). Since each update is dominated by the sequential recurrent-unroll
    overhead (see `_slots_d3_fixed4_4gpu_bigl`), halving the pass count should cut wall-clock toward ~12h.
    RECIPE CHANGE -- different optimization than the faithful `_slots_d3_fixed4_4gpu` (fewer, larger SGD
    steps per batch), acceptable for the cell-count *capacity* sweep but NOT identical to the 300M planning
    core. local_num_envs=128, batch 5120, num_envs 256, 300M unchanged. Divisibility: (128/2)*1 % 4 == 0."""
    args = _slots_d3_fixed4_4gpu(num_slots)
    args.num_minibatches = 4
    return args


# fmt: off
def sokoban_drc_slots_d3_fixed4_n100_4gpu_mb4(): return _slots_d3_fixed4_4gpu_mb4(100)
def sokoban_drc_slots_d3_fixed4_n200_4gpu_mb4(): return _slots_d3_fixed4_4gpu_mb4(200)
def sokoban_drc_slots_d3_fixed4_n50_4gpu_mb4():  return _slots_d3_fixed4_4gpu_mb4(50)
# fmt: on
# fmt: on


def sokoban_drc_cellwise(n_recurrent: int, num_repeats: int, aggregation: str = "mean") -> Args:
    """Same setup as `sokoban_drc`, but with the cellwise MLP-message-passing core instead of ConvLSTM.

    Each cell computes a shared-MLP message from every (neighbour) cell's hidden state and aggregates
    them over its grid neighbourhood (`aggregation` in {"mean", "sum", "max"}). This is the explicit
    message-passing sibling of `sokoban_drc_attn`: an input-independent, translation-equivariant
    graph operator in place of attention's input-dependent softmax. `Args.net` is mutable, so we
    reuse the DRC env/loss/optimizer setup and just swap the backbone.
    """
    args = sokoban_drc33_59()  # inherit the PROVEN recipe (grad clip 2.5e-4, vtrace_lambda 0.5,
    # light L2, valid_medium eval), then swap in the attention/cellwise backbone below.
    # (Previously based on the weaker `sokoban_drc` recipe, which confounded the DRC comparison.)
    args.net = CellwiseLSTMConfig(
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=CellwiseLSTMCellConfig(
            features=32,
            message_hiddens=(64,),
            use_neighbor_mask=True,
            mask_neighborhood="king",
            aggregation=aggregation,
            n_global=4,
        ),
        n_recurrent=n_recurrent,
        mlp_hiddens=(256,),
        repeats_per_step=num_repeats,
    )
    # Lighten the DURING-training eval (it dominates CPU on the parallel matrix runs): keep just
    # 2 thinking ticks (0 + 8) and sparser eval points. The full steps_to_think sweep with error
    # bars is produced post-hoc by cleanba.load_and_eval on the saved checkpoints.
    args.eval_envs["valid_medium"].steps_to_think = [0, 8]
    _bs = 256 * 20  # local_num_envs * num_steps (inherited from sokoban_resnet59)
    args.eval_at_steps = frozenset([int(2e6 / _bs), int(5e6 / _bs)] + [int(10e6 / _bs) * i for i in range(1, 21)])
    return args


# fmt: off
def sokoban_drc_cellwise_3_3(): return sokoban_drc_cellwise(3, 3)
def sokoban_drc_cellwise_1_1(): return sokoban_drc_cellwise(1, 1)
def sokoban_drc_cellwise_3_3_sum(): return sokoban_drc_cellwise(3, 3, aggregation="sum")
def sokoban_drc_cellwise_3_3_max(): return sokoban_drc_cellwise(3, 3, aggregation="max")
# fmt: on


def sokoban_resnet59():
    CACHE_PATH = Path("/opt/sokoban_cache")
    return Args(
        train_env=EnvpoolBoxobanConfig(
            seed=1234,
            max_episode_steps=120,
            min_episode_steps=30,
            num_envs=1,
            cache_path=CACHE_PATH,
            split="train",
            difficulty="unfiltered",
        ),
        eval_envs=dict(
            valid_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    seed=0,
                    load_sequentially=True,
                    max_episode_steps=120,
                    min_episode_steps=120,
                    num_envs=256,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="medium",
                ),
                n_episode_multiple=4,
                steps_to_think=[0, 2, 4, 8, 12, 16, 24, 32],
            ),
        ),
        log_frequency=10,
        net=GuezResNetConfig(yang_init=False, norm=IdentityNorm(), normalize_input=False),
        loss=ImpalaLossConfig(
            vtrace_lambda=0.5,
            gamma=0.97,
            vf_coef=0.25,
            ent_coef=0.01,
            normalize_advantage=False,
            logit_l2_coef=1.5625e-06,
            weight_l2_coef=1.5625e-08,
            vf_loss_type="square",
            advantage_multiplier="one",
        ),
        num_steps=20,
        eval_at_steps=frozenset([int(195600 / div * i) for div in [1000, 100, 10] for i in range(1, 21)]),
        actor_update_cutoff=int(1e20),
        sync_frequency=int(1e20),
        rmsprop_eps=1.5625e-07,
        rmsprop_decay=0.99,
        adam_b1=0.9,
        optimizer="adam",
        optimizer_yang=False,
        local_num_envs=256,
        num_minibatches=8,
        total_timesteps=2_002_944_000,
        base_run_dir=Path("/training/cleanba"),
        learning_rate=4e-4,
        final_learning_rate=4e-6,
        anneal_lr=True,
        base_fan_in=1,
        max_grad_norm=2.5e-4,
        num_actor_threads=1,
        seed=4242,
    )


def sokoban_drc33_59() -> Args:
    drc_n_n = 3

    out = sokoban_resnet59()
    out.net = ConvLSTMConfig(
        n_recurrent=drc_n_n,
        repeats_per_step=drc_n_n,
        skip_final=True,
        residual=False,
        use_relu=False,
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=ConvLSTMCellConfig(
            ConvConfig(32, (3, 3), (1, 1), "SAME", True),
            pool_and_inject="horizontal",
            pool_projection="per-channel",
            output_activation="tanh",
            fence_pad="valid",
            forget_bias=0.0,
        ),
        head_scale=1.0,
    )
    return out


def boxworld_drc33() -> Args:
    drc_n_n = 3

    out = sokoban_resnet59()

    out.train_env = BoxWorldConfig(
        seed=1234,
        max_episode_steps=120,
        num_envs=1,
        step_cost=0.1,
    )

    out.eval_envs = dict(
        valid=EvalConfig(
            BoxWorldConfig(
                seed=0,
                max_episode_steps=120,
                num_envs=256,
            ),
            n_episode_multiple=4,
            steps_to_think=[0, 2, 4, 8, 12, 16, 24, 32],
        ),
    )

    out.net = ConvLSTMConfig(
        n_recurrent=drc_n_n,
        repeats_per_step=drc_n_n,
        skip_final=True,
        residual=False,
        use_relu=False,
        embed=[ConvConfig(32, (3, 3), (1, 1), "SAME", True)] * 2,
        recurrent=ConvLSTMCellConfig(
            ConvConfig(32, (3, 3), (1, 1), "SAME", True),
            pool_and_inject="horizontal",
            pool_projection="per-channel",
            output_activation="tanh",
            fence_pad="valid",
            forget_bias=0.0,
        ),
        head_scale=1.0,
    )

    out.total_timesteps = 200_000_000
    return out


def minipacman_drc33() -> Args:
    out = boxworld_drc33()
    out.train_env = MiniPacManConfig(seed=1234, max_episode_steps=1000, num_envs=1, nghosts_init=3, npills=3)
    out.eval_envs = dict(valid=EvalConfig(MiniPacManConfig(seed=0, max_episode_steps=500, num_envs=256), n_episode_multiple=4))
    return out
