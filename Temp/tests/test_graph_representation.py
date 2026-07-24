"""Regression coverage for graph state features and action injection."""
from __future__ import annotations

import pytest

from risk.game.actions import FortifyAction, ReinforcementAction
from risk.game.phase import Phase
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.graph_adapter import (
    GraphAdapter,
    armies_column_index,
    proposed_army_delta_column_index,
    unfinished_attack_target_column_index,
)

from .conftest import make_env


def test_graph_exposes_turn_history_and_keeps_base_proposed_delta_zero() -> None:
    env = make_env(n=3, seed=7, agent_kind="human")
    state = env.current_state()
    state.unfinished_attack_targets_this_turn = {3}
    state.conquered_this_turn = True

    graph = GraphAdapter(env.topology, env.settings)(state)

    unfinished_col = unfinished_attack_target_column_index(env.topology)
    delta_col = proposed_army_delta_column_index(env.topology)
    assert graph.x[3, unfinished_col].item() == pytest.approx(1.0)
    assert graph.x[:, delta_col].sum().item() == pytest.approx(0.0)
    assert graph.u[0, -1].item() == pytest.approx(1.0)


def test_action_injection_preserves_armies_and_uses_signed_delta() -> None:
    env = make_env(n=3, seed=7, agent_kind="human")
    state = env.current_state()
    owned = next(i for i, owner in enumerate(state.owners) if owner == state.current_player_index)
    target = env.topology.territory_at(owned)
    state.phase = Phase.REINFORCE_PLACE
    state.reinforcement_budget = 3
    base = GraphAdapter(env.topology, env.settings)(state)
    injected = ActionGraphBuilder(env.topology)(
        base, ReinforcementAction(placements={target: 3}), state
    )

    army_col = armies_column_index(env.topology)
    delta_col = proposed_army_delta_column_index(env.topology)
    assert injected.x[owned, army_col].item() == pytest.approx(base.x[owned, army_col].item())
    assert injected.x[owned, delta_col].item() == pytest.approx(3.0)

    destination_name = env.topology.neighbors(target)[0]
    destination = env.topology.index_of(destination_name)
    state.owners[destination] = state.current_player_index
    fortify = ActionGraphBuilder(env.topology)(
        base,
        FortifyAction(from_territory=target, to_territory=destination_name, count=2),
        state,
    )
    assert fortify.x[owned, delta_col].item() == pytest.approx(-2.0)
    assert fortify.x[destination, delta_col].item() == pytest.approx(2.0)
