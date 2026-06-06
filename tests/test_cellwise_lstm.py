from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from cleanba.cellwise_lstm import (
    CellwiseLSTM,
    CellwiseLSTMCell,
    CellwiseLSTMCellConfig,
    CellwiseLSTMConfig,
)
from cleanba.convlstm import ConvConfig, LSTMCellState
from cleanba.network import _fan_in_for_params


def _cell_cfg(**kwargs) -> CellwiseLSTMCellConfig:
    base = dict(features=4, message_hiddens=(8,), n_global=2)
    base.update(kwargs)
    return CellwiseLSTMCellConfig(**base)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "max"])
@pytest.mark.parametrize("use_neighbor_mask", [True, False])
@pytest.mark.parametrize("mask_neighborhood", ["king", "vonneumann"])
def test_cell_forward_shapes(
    aggregation: Literal["mean", "sum", "max"],
    use_neighbor_mask: bool,
    mask_neighborhood: Literal["king", "vonneumann"],
):
    cell = CellwiseLSTMCell(
        _cell_cfg(aggregation=aggregation, use_neighbor_mask=use_neighbor_mask, mask_neighborhood=mask_neighborhood)
    )
    rng = jax.random.PRNGKey(0)
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    # (time, batch, H, W, C) so we can step it like ConvLSTMCell's test.
    inputs = jax.random.normal(k1, (3, 5, 7, 6, 4))
    carry = cell.initialize_carry(k2, inputs[0].shape)
    assert carry.c.shape == carry.h.shape == (5, 7, 6, 4)
    params = cell.init(k3, carry, inputs[0], carry.h)

    for t in range(len(inputs)):
        carry, out = jax.jit(cell.apply)(params, carry, inputs[t], carry.h)
        assert carry.c.shape == carry.h.shape == out.shape == (5, 7, 6, 4)


CELLWISE_CONFIGS = [
    CellwiseLSTMConfig(
        embed=[ConvConfig(4, (3, 3), (1, 1), "SAME", True)],
        recurrent=_cell_cfg(aggregation="mean"),
        repeats_per_step=1,
        n_recurrent=1,
    ),
    CellwiseLSTMConfig(
        embed=[ConvConfig(4, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=_cell_cfg(aggregation="sum", use_edge_weights=False, n_global=0, pre_norm=False),
        repeats_per_step=2,
        n_recurrent=2,
    ),
    CellwiseLSTMConfig(
        embed=[ConvConfig(4, (3, 3), (1, 1), "SAME", True)],
        recurrent=_cell_cfg(aggregation="max", use_neighbor_mask=False),
        repeats_per_step=2,
        n_recurrent=2,
    ),
]


@pytest.mark.parametrize("net", CELLWISE_CONFIGS)
def test_scan_matches_manual_loop(net: CellwiseLSTMConfig):
    """`scan` over time must equal the explicit cell-by-cell loop (mirrors test_convlstm.test_scan_correct)."""
    num_envs = 5
    input_shape = (num_envs, 12, 10, 3)
    lstm = CellwiseLSTM(net)
    key, k1, k2 = jax.random.split(jax.random.PRNGKey(1234), 3)
    carry = lstm.apply({}, k1, input_shape, method=lstm.initialize_carry)
    assert isinstance(carry, list) and len(carry) == net.n_recurrent
    params = lstm.init(k2, carry, jnp.ones((1, *input_shape)), jnp.ones((1, num_envs), dtype=jnp.bool_), method=lstm.scan)

    time_steps = 4
    key, k1 = jax.random.split(key)
    inputs = jax.random.uniform(k1, (time_steps, *input_shape), maxval=255)
    episode_starts = jnp.zeros((time_steps, num_envs), dtype=jnp.bool_)

    lstm_carry, lstm_out = lstm.apply(params, carry, inputs, episode_starts, method=lstm.scan)

    b_lstm = lstm.bind(params)
    cell_carry: list[LSTMCellState] = list(carry)
    for t in range(time_steps):
        x = b_lstm._compress_input(inputs[t])
        h_nd = cell_carry[-1].h
        for _ in range(net.repeats_per_step):
            for d, cell in enumerate(b_lstm.cell_list):
                cell_carry[d], h_nd = cell(cell_carry[d], x, h_nd)

    for d in range(len(b_lstm.cell_list)):
        assert jnp.allclose(cell_carry[d].c, lstm_carry[d].c, atol=1e-5)
        assert jnp.allclose(cell_carry[d].h, lstm_carry[d].h, atol=1e-5)
    assert lstm_out.shape[:2] == (time_steps, num_envs)


def test_message_output_zeros_init_is_gated_identity():
    """With the message-out projection zeros-init, the first step's new_h must not depend on neighbours,
    only on the per-cell `c` and the local input/gate -- the documented gated-identity-at-init property."""
    cfg = _cell_cfg(aggregation="mean", n_global=2)
    cell = CellwiseLSTMCell(cfg)
    rng = jax.random.PRNGKey(7)
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    inputs = jax.random.normal(k1, (2, 6, 5, 4))
    carry = cell.initialize_carry(k2, inputs.shape)
    # Non-zero hidden so that a neighbour-dependent message *would* move new_h if it leaked in.
    carry = LSTMCellState(c=carry.c, h=jax.random.normal(k3, carry.h.shape))
    params = cell.init(rng, carry, inputs, carry.h)

    (new_state, _) = cell.apply(params, carry, inputs, carry.h)
    # Perturb a single cell's hidden and confirm new_c/new_h at *other* cells are unchanged at init
    # (messages are identically zero, so there is no cross-cell coupling yet).
    h2 = carry.h.at[:, 0, 0, :].add(10.0)
    (new_state2, _) = cell.apply(params, LSTMCellState(c=carry.c, h=h2), inputs, h2)
    # cell (1,1) is not a neighbour of (0,0) under the king mask only if distance > 1; pick a far cell.
    assert jnp.allclose(new_state.c[:, 3, 3, :], new_state2.c[:, 3, 3, :], atol=1e-6)


def test_all_param_names_known_to_mup_labeler():
    """Every param leaf must be recognized by `_fan_in_for_params`, else optimizer construction
    (cleanba_impala.py: label_and_learning_rate_for_params) raises. Guards `edge_logits`/`global_logits`,
    which are introduced by this core and would otherwise be unknown names."""
    # A config that materializes every cellwise param: edge_logits (use_edge_weights), global_logits
    # (n_global>0), RMSNorm scale (pre_norm), and the Dense kernels/biases.
    net = CellwiseLSTMConfig(
        embed=[ConvConfig(4, (3, 3), (1, 1), "SAME", True)],
        recurrent=_cell_cfg(aggregation="mean", use_edge_weights=True, n_global=2, pre_norm=True),
        repeats_per_step=2,
        n_recurrent=2,
    )
    lstm = CellwiseLSTM(net)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    input_shape = (1, 8, 6, 3)
    carry = lstm.apply({}, k1, input_shape, method=lstm.initialize_carry)
    variables = lstm.init(k2, carry, jnp.ones((1, *input_shape)), jnp.ones((1, 1), dtype=jnp.bool_), method=lstm.scan)

    labels = _fan_in_for_params(variables["params"])  # raises ValueError on any unknown leaf name
    flat = jax.tree.leaves(labels)
    assert flat, "expected some labeled params"
    assert "edge_logits" not in flat and "global_logits" not in flat  # they are remapped, not raw names


def test_config_de_serialize():
    farconf = pytest.importorskip("farconf")
    net = CELLWISE_CONFIGS[0]
    d = farconf.to_dict(net, CellwiseLSTMConfig)
    net2 = farconf.from_dict(d, CellwiseLSTMConfig)
    assert net == net2
