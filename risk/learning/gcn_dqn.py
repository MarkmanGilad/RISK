"""Net A's network — DQN + injection (`Docs/NetworkArchitectures.md`).

`GCN_DQN` is the graph network: `(state graph(s), phase label per graph,
card_indices) -> Q(s, a)`, all 5 stages in one call, one uniform loop. It
owns no game logic beyond the one leaf `Phase` enum (`risk/game/phase.py`,
no dependencies of its own — not `Action`/`ActionGraphBuilder`/
`Environment`) needed to pick which head scores each row. Building the
per-candidate graphs, batching them, and merging scores back into
`legal_actions()`'s order is still the *agent's* job (the not-yet-built
`GCN_DQN_Agent`), the same split `Temp/Examples/DQN_Agent.py` uses around
its bare `DQN` network — this class just goes one step further than a
bare encoder by also routing each row to its head internally.

    net = GCN_DQN(in_dim=13, hidden_dim=64, edge_dim=2, u_dim=34)
    q = net(state, phase, card_indices)   # state: always a Batch, even for N=1

`Encoder`/`pool` stay in their own modules (`encoder.py`, `pooling.py`)
rather than being merged into this class — they're shared, unmodified, by
all four planned nets (DQN/PPO x inject/lookup), not specific to this one.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch

from risk.game.phase import Phase
from risk.learning.encoder import Encoder
from risk.learning.heads import ScoringHead, TradeInHead
from risk.learning.pooling import pool


class GCN_DQN(nn.Module):
    """`(state graph(s), phase per graph, card_indices) -> Q(s, a)`."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        u_dim: int,
        n_layers: int = 4,
        card_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        g_dim = 2 * hidden_dim + u_dim
        self.encoder = Encoder(in_dim, hidden_dim, edge_dim, n_layers)
        self.reinforce_place_head = ScoringHead(g_dim)
        self.attack_head = ScoringHead(g_dim)
        self.occupy_head = ScoringHead(g_dim)
        self.fortify_head = ScoringHead(g_dim)
        self.trade_in_head = TradeInHead(g_dim, card_embed_dim)
        # Plain dict, not a ModuleDict — the heads are already registered
        # as submodules via the named attributes above; this is just an
        # internal alias map for `forward`'s row-routing, not a second
        # registration. All 5 heads share one call signature
        # `(g, card_indices) -> Q` now (`ScoringHead` ignores the second
        # arg), so `TRADE_IN` needs no special-casing here either.
        self._heads_by_phase = {
            int(Phase.TRADE_IN): self.trade_in_head,
            int(Phase.REINFORCE_PLACE): self.reinforce_place_head,
            int(Phase.ATTACK): self.attack_head,
            int(Phase.OCCUPY): self.occupy_head,
            int(Phase.FORTIFY): self.fortify_head,
        }

    def forward(self, state: Batch, phase: torch.Tensor, card_indices: torch.Tensor) -> torch.Tensor:
        """`state`: one graph per row being scored, always a `Batch` —
        even scoring a single candidate is `Batch.from_data_list([one_graph])`,
        same convention as carrying a batch dimension of 1 in any other
        PyTorch net; the caller's job, not something `forward` detects and
        special-cases. *Injected* (`ActionGraphBuilder`) for every stage
        except `TRADE_IN`, where it's just the unmodified base state graph
        (`GraphAdapter`'s bare output — already carries a zero-filled
        `edge_attr` of its own, so there's nothing for `forward` to default
        here) repeated once per `TRADE_IN` candidate in that decision, since
        none of them perturb the graph. `phase`: `[N]` long, one `Phase`
        value per row — same convention as `ReplayBuffer`'s `stage`/
        `next_stage`. `card_indices`: `[N, 3]` long, each row's
        `dqn_index()` `(t1, t2, n)` for `TRADE_IN` rows — not optional, just
        zeros (or anything; ignored) where a row isn't `TRADE_IN`, so every
        row can be routed through its head the same way, with no branching
        on which kind of head it is. Returns `[N]`.
        """
        h = self.encoder(state.x, state.edge_index, state.edge_attr)
        g = pool(h, state.batch, state.u)

        q = torch.empty(g.shape[0], dtype=g.dtype, device=g.device)
        # Risk exposes 5 DQN phases/heads: trade-in, reinforce-place,
        # attack, occupy, and fortify.
        for stage in range(5):
            mask = phase == stage
            if mask.any():
                q[mask] = self._heads_by_phase[stage](g[mask], card_indices[mask])
        return q


__all__ = ["GCN_DQN"]
