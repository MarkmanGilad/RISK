"""Board-state -> PyTorch Geometric graph adapter.

Converts `(State, BoardTopology, GameSettings)` into a `torch_geometric.data.Data`
snapshot for the GCN+DQN trainer. Lives in `risk/learning/`, not `risk/game/`,
so the core rules engine stays torch-free (see `State.to_features`'s docstring).
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from risk.constants import MAX_PLAYERS, card_set_value
from risk.game.board_topology import BoardTopology
from risk.game.phase import Phase
from risk.game.settings import GameSettings
from risk.game.state import State

# Node feature layout: continent one-hot | owner one-hot (padded to
# MAX_PLAYERS) | army count.


def state_to_pyg(state: State, topology: BoardTopology, settings: GameSettings) -> Data:
    """Build one graph snapshot of `state` for a GCN.

    Node order matches `topology.territories` (`topology.index_of(t)` is the
    node index for territory `t`). Player-indexed features are padded to
    `MAX_PLAYERS` so the feature width is constant across 3..6-player games.
    """
    x = _node_features(state, topology)
    edge_index = _edge_index(topology)
    u = _global_features(state, topology, settings)
    return Data(x=x, edge_index=edge_index, u=u)


def _node_features(state: State, topology: BoardTopology) -> torch.Tensor:
    n = len(topology)
    continents = topology.continents
    continent_col = {c: i for i, c in enumerate(continents)}
    armies_col = len(continents) + MAX_PLAYERS

    x = torch.zeros((n, armies_col + 1), dtype=torch.float32)
    for i in range(n):
        territory = topology.territory_at(i)
        x[i, continent_col[topology.continent_of(territory)]] = 1.0
        owner = state.owners[i]
        if owner is not None:
            x[i, len(continents) + owner] = 1.0
        x[i, armies_col] = float(state.armies[i])
    return x


def _edge_index(topology: BoardTopology) -> torch.Tensor:
    src, dst = topology.edge_index()
    return torch.tensor([src, dst], dtype=torch.long)


def _global_features(
    state: State, topology: BoardTopology, settings: GameSettings
) -> torch.Tensor:
    n_players = settings.player_count

    cards_per_player = [0.0] * MAX_PLAYERS
    eliminated = [0.0] * MAX_PLAYERS
    for p in range(n_players):
        cards_per_player[p] = float(len(state.hands[p]))
        eliminated[p] = 1.0 if p in state.eliminated else 0.0

    current_player_onehot = [0.0] * MAX_PLAYERS
    current_player_onehot[state.current_player_index] = 1.0

    phase_onehot = [0.0] * len(Phase)
    phase_onehot[int(state.phase)] = 1.0

    continent_worth = [topology.continent_bonus(c) for c in topology.continents]
    next_trade_value = card_set_value(state.cards_traded_in_count)

    values = (
        [float(n_players), float(next_trade_value)]
        + cards_per_player
        + phase_onehot
        + current_player_onehot
        + [float(b) for b in continent_worth]
        + [float(state.reinforcement_budget)]
        + eliminated
    )
    return torch.tensor([values], dtype=torch.float32)


__all__ = ["state_to_pyg"]
