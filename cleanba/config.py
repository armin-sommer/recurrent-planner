import dataclasses
from dataclasses import field
from pathlib import Path
from typing import List, Literal, Optional

from cleanba.attn_lstm import AttentionCellConfig, AttentionLSTMConfig
from cleanba.cellwise_lstm import CellwiseLSTMCellConfig, CellwiseLSTMConfig
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig
from cleanba.environments import AtariEnv, BoxWorldConfig, EnvConfig, EnvpoolBoxobanConfig, MiniPacManConfig, random_seed
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
    directional_value: bool = False,
    relative_key: bool = False,
) -> Args:
    """Same setup as `sokoban_drc`, but with the state-indexed masked-attention core instead of ConvLSTM.

    `use_attention_mask=True` (default) restricts each cell to its grid neighbourhood (provable
    locality); `False` uses full dense attention. `readout` defaults to "maxplus" (the VIN-aligned
    soft Bellman max-plus aggregator); pass `readout="softmax"` for the original convex-average
    attention. `mask_neighborhood` chooses the grid stencil ("king" = 8-neighbourhood, "vonneumann" =
    the 4 one-move-reachable cells). `Args.net` is mutable, so we reuse the DRC env/loss/optimizer
    setup and just swap the backbone.
    """
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
        ),
        n_recurrent=n_recurrent,
        mlp_hiddens=(256,),
        repeats_per_step=num_repeats,
    )
    return args


# fmt: off
def sokoban_drc_attn_3_3(): return sokoban_drc_attn(3, 3)                                         # maxplus, king (default)
def sokoban_drc_attn_3_3_dir(): return sokoban_drc_attn(3, 3, directional_value=True)             # + per-offset value routing (VIN-aligned)
def sokoban_drc_attn_3_3_dir_relk(): return sokoban_drc_attn(3, 3, directional_value=True, relative_key=True)  # + content x direction key
def sokoban_drc_attn_3_3_dir_softmax(): return sokoban_drc_attn(3, 3, readout="softmax", directional_value=True)
def sokoban_drc_attn_1_1(): return sokoban_drc_attn(1, 1)
def sokoban_drc_attn_3_3_vn(): return sokoban_drc_attn(3, 3, mask_neighborhood="vonneumann")      # maxplus, von Neumann
def sokoban_drc_attn_3_3_softmax(): return sokoban_drc_attn(3, 3, readout="softmax")              # old convex-average attention
def sokoban_drc_attn_3_3_nomask(): return sokoban_drc_attn(3, 3, use_attention_mask=False)        # maxplus, dense (heavy: large Kn)
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
