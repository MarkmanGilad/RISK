"""Per-action injection for Net A (DQN+inject)/Net C (PPO+inject).

See `Docs/NetworkArchitectures.md`'s "Action injection" section. Given the
base `Data` graph for a state (`GraphAdapter`) and one legal `Action`,
builds a *modified copy* carrying that action's perturbation — the base
graph is never mutated, since every action needs its own copy to diverge
from. `GraphAdapter` already gives every base graph a zero-filled
`edge_attr` (`EDGE_ATTR_DIM` wide), so this clones and overwrites that
instead of building one from scratch.

- `AttackAction`s mark the attacked edge `[selected_attack, dice /
  MAX_ATTACK_DICE]`; every other edge (including the reverse direction)
  stays `[0, 0]`.
- `ReinforcementAction`/`OccupyAction`/`FortifyAction` write directly into
  `x`'s existing army-count column at the affected territory row(s) — the
  simpler of the two options `Docs/NetworkArchitectures.md` leaves open (vs.
  a parallel `proposed_delta` column); revisit if training shows the
  network can't disentangle a proposed change from the real count.
- `StopAttackAction`/skip-`FortifyAction` get no perturbation at all — the
  result *is* the unmodified base graph (plus the same zero `edge_attr`
  every other action gets, so it still batches with them).
- `TradeInAction` isn't injected at all — per `Docs/Action.md` its head
  reads card-hand embeddings, never the graph, so this raises instead of
  silently returning something meaningless.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data

from risk.constants import MAX_ATTACK_DICE
from risk.game.actions import Action
from risk.game.board_topology import BoardTopology
from risk.game.phase import Phase
from risk.game.state import State
from risk.learning.graph_adapter import EDGE_ATTR_DIM, proposed_army_delta_column_index


class ActionGraphBuilder:
    """Injects one legal `Action` into a base state graph.

        builder = ActionGraphBuilder(topology)
        action_graph = builder(base, action, state)

    `topology`'s edge order is fixed per game (`BoardTopology` sorts once at
    load time), so the `(from, to) -> edge row` mapping is precomputed once
    here instead of searched per action.
    """

    def __init__(self, topology: BoardTopology) -> None:
        self.topology = topology
        self._proposed_army_delta_col = proposed_army_delta_column_index(topology)
        src, dst = topology.edge_index()
        self._edge_row = {pair: i for i, pair in enumerate(zip(src, dst))}

    def __call__(self, base: Data, action: Action, state: Optional[State] = None) -> Data:
        stage, t1, t2, n = action.dqn_index(self.topology, state)
        if stage == Phase.TRADE_IN:
            return self._copy_base(base)

        x = base.x.clone()
        edge_attr = base.edge_attr.clone()

        if stage == Phase.ATTACK and t1 != Action.NONE_INDEX:
            edge_attr[self._edge_row[(t1, t2)]] = torch.tensor([1.0, n / MAX_ATTACK_DICE])
        elif stage == Phase.REINFORCE_PLACE:
            x[t1, self._proposed_army_delta_col] += n
        elif stage == Phase.OCCUPY:
            x[t1, self._proposed_army_delta_col] -= n
            x[t2, self._proposed_army_delta_col] += n
        elif stage == Phase.FORTIFY and t1 != Action.NONE_INDEX:
            x[t1, self._proposed_army_delta_col] -= n
            x[t2, self._proposed_army_delta_col] += n

        return Data(x=x, edge_index=base.edge_index, edge_attr=edge_attr, u=base.u)

    @staticmethod
    def _copy_base(base: Data) -> Data:
        return Data(x=base.x, edge_index=base.edge_index, edge_attr=base.edge_attr, u=base.u)


__all__ = ["ActionGraphBuilder", "EDGE_ATTR_DIM"]
