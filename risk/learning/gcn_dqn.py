"""Net A — DQN + injection (`Docs/NetworkArchitectures.md`).

`GCN_DQN` wires the shared building blocks together into the full forward
pass from that doc's Net A diagram:

    base Data(state)
       |  + one legal action (per action)
       v
    ActionGraphBuilder  -->  N modified Data graphs
       v
    Batch.from_data_list
       v
    Encoder (batched)      -->  H per graph
       v
    pool(H, u)              -->  g  [N, g_dim]
       v
    heads[stage](g)          -->  Q(s, a)  [N]

`TRADE_IN` actions skip `ActionGraphBuilder` entirely (it refuses them —
see `Docs/ActionGraphBuilder.md`) and instead score off the *unmodified*
base graph's pooled context plus `Heads.trade_in`'s card-slot embeddings.
`forward` always splits the legal actions by stage (`TRADE_IN` vs. the
rest), scores each group through its own path, and merges the
per-action scalars back into the caller's original order.

`Encoder`/`pool`/`Heads` stay in their own modules (`encoder.py`,
`pooling.py`, `heads.py`) rather than being merged into this class — they're
shared, unmodified, by all four planned nets (DQN/PPO x inject/lookup), not
specific to this one.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from risk.game.actions import Action, ActionStage
from risk.game.board_topology import BoardTopology
from risk.game.state import State
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.encoder import Encoder
from risk.learning.heads import Heads
from risk.learning.pooling import pool


class GCN_DQN(nn.Module):
    """`(base state graph, legal actions) -> Q(s, a)` for every legal action.

        net = GCN_DQN(topology, in_dim=13, hidden_dim=64, edge_dim=2, u_dim=33)
        q = net(base, legal_actions, state)   # [len(legal_actions)], same order
        best = legal_actions[q.argmax()]
    """

    def __init__(
        self,
        topology: BoardTopology,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        u_dim: int,
        n_layers: int = 4,
        card_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.action_builder = ActionGraphBuilder(topology)
        self.encoder = Encoder(in_dim, hidden_dim, edge_dim, n_layers)
        self.heads = Heads(g_dim=2 * hidden_dim + u_dim, card_embed_dim=card_embed_dim)

    def forward(
        self, base: Data, legal_actions: Sequence[Action], state: Optional[State] = None
    ) -> torch.Tensor:
        q = [torch.empty(())] * len(legal_actions)

        graph_idx = [i for i, a in enumerate(legal_actions) if a.stage != ActionStage.TRADE_IN]
        trade_idx = [i for i, a in enumerate(legal_actions) if a.stage == ActionStage.TRADE_IN]

        if graph_idx:
            graph_actions = [legal_actions[i] for i in graph_idx]
            graphs = [self.action_builder(base, a, state) for a in graph_actions]
            batch = Batch.from_data_list(graphs)
            h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
            g = pool(h, batch.batch, batch.u)

            stages = {a.stage for a in graph_actions}
            for stage in stages:
                rows = [j for j, a in enumerate(graph_actions) if a.stage == stage]
                scores = self.heads(stage, g[rows])
                for k, j in enumerate(rows):
                    q[graph_idx[j]] = scores[k]

        if trade_idx:
            trade_actions = [legal_actions[i] for i in trade_idx]
            zero_edge_attr = torch.zeros((base.edge_index.shape[1], self.edge_dim))
            node_batch = torch.zeros(base.x.shape[0], dtype=torch.long)
            h_base = self.encoder(base.x, base.edge_index, zero_edge_attr)
            g_base = pool(h_base, node_batch, base.u).expand(len(trade_actions), -1)
            card_idx = torch.tensor([a.card_indices for a in trade_actions], dtype=torch.long)
            scores = self.heads.trade_in(g_base, card_idx)
            for k, i in enumerate(trade_idx):
                q[i] = scores[k]

        return torch.stack(q)


__all__ = ["GCN_DQN"]
