"""Board-state -> PyTorch Geometric graph adapter.

Converts `(State, BoardTopology, GameSettings)` into a `torch_geometric.data.Data`
snapshot for the GNN+DQN trainer. Lives in `risk/learning/`, not `risk/game/`,
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


EDGE_ATTR_DIM = 2   # [selected_attack, dice_count] — see ActionGraphBuilder


def armies_column_index(topology: BoardTopology) -> int:
    """Column of `x` holding the raw army count (see `_node_features` below).

    A plain function, not a `GraphAdapter` method, so `ActionGraphBuilder`
    (which perturbs that same column to inject `REINFORCE_PLACE`/`OCCUPY`/
    `FORTIFY` actions — `Docs/NetworkArchitectures.md`) can reuse this
    layout without needing a `GraphAdapter` instance or duplicating the
    offset math.
    """
    return len(topology.continents) + MAX_PLAYERS


class GraphAdapter:
    """Builds one `Data` graph snapshot of a `State` for a GNN.

    `topology`/`settings` don't change during a match, so build one
    instance per game and call it every step with just the `State`:

        adapter = GraphAdapter(topology, settings)
        data = adapter(state)

    Node feature layout: continent one-hot | owner one-hot (padded to
    MAX_PLAYERS) | army count. `edge_attr` is always present — all-zero
    here, since the base graph carries no action — so every consumer
    (`Encoder`, `ActionGraphBuilder`) can rely on it existing rather than
    defaulting it themselves. `ActionGraphBuilder` clones and overwrites it
    per candidate instead of building its own from scratch.
    """

    def __init__(self, topology: BoardTopology, settings: GameSettings) -> None:
        self.topology = topology
        self.settings = settings

    def __call__(self, state: State, perspective: int = 0) -> Data:
        """Build the graph snapshot for `state`.

        Node order matches `topology.territories` (`topology.index_of(t)` is
        the node index for territory `t`). Player-indexed features are
        padded to `MAX_PLAYERS` so the feature width is constant across
        3..6-player games. `n_players` for the padding/rotation math below
        is read off `len(state.hands)`, not `self.settings.player_count` —
        a trainer that reuses one `GraphAdapter` (via `GNN_DQN_Agent.attach`)
        across many self-play episodes of *different* sizes would otherwise
        score an older replay-buffer `state` against the wrong (current
        episode's) player count, indexing the wrong number of hands/owner
        slots.

        `perspective` rotates every player-indexed slot (owner one-hot,
        cards-per-player, current-player one-hot, eliminated) so that
        `perspective` itself always lands in relative slot `0`, the next
        player after it in slot `1`, and so on (`(p - perspective) %
        n_players`) — turn order is preserved, only the starting point
        shifts. Default `0` is a no-op (`p % n_players == p`), so existing
        callers that never pass it keep seeing raw, absolute player ids.
        This is for the learning agent: across self-play episodes it gets
        assigned a different physical seat by the trainer, but should always
        *learn* from one consistent "this is
        me, these are my opponents in turn order" frame rather than having
        to re-derive "which absolute id am I this game" from `u`'s
        current-player one-hot every time.
        """
        x = self._node_features(state, perspective)
        edge_index = self._edge_index()
        edge_attr = torch.zeros((edge_index.shape[1], EDGE_ATTR_DIM), dtype=torch.float32)
        u = self._global_features(state, perspective)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)

    def _node_features(self, state: State, perspective: int) -> torch.Tensor:
        topology = self.topology
        n = len(topology)
        n_players = len(state.hands)
        continents = topology.continents
        continent_col = {c: i for i, c in enumerate(continents)}
        armies_col = armies_column_index(topology)

        x = torch.zeros((n, armies_col + 1), dtype=torch.float32)
        for i in range(n):
            territory = topology.territory_at(i)
            x[i, continent_col[topology.continent_of(territory)]] = 1.0
            owner = state.owners[i]
            if owner is not None:
                x[i, len(continents) + (owner - perspective) % n_players] = 1.0
            x[i, armies_col] = float(state.armies[i])
        return x

    def _edge_index(self) -> torch.Tensor:
        src, dst = self.topology.edge_index()
        return torch.tensor([src, dst], dtype=torch.long)

    def _global_features(self, state: State, perspective: int) -> torch.Tensor:
        topology = self.topology
        n_players = len(state.hands)

        cards_per_player = [0.0] * MAX_PLAYERS
        eliminated = [0.0] * MAX_PLAYERS
        for p in range(n_players):
            rel = (p - perspective) % n_players
            cards_per_player[rel] = float(len(state.hands[p]))
            eliminated[rel] = 1.0 if p in state.eliminated else 0.0

        current_player_onehot = [0.0] * MAX_PLAYERS
        current_player_onehot[(state.current_player_index - perspective) % n_players] = 1.0

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


__all__ = ["GraphAdapter", "armies_column_index", "EDGE_ATTR_DIM"]
