"""PQN's injected-action policy/value network — `Docs/PQN.md`.

`PQN` is a copy of `Dueling_DQN` (`dueling_dqn.py`) with one change: instead
of combining `V(s)` and `A(s, a)` into a fused `Q(s, a)` and returning that,
`forward` returns the two raw streams uncombined. Per `Docs/PQN.md` §24.A,
combining them into `Q`, and turning `A` into a policy, is `pqn_agent.py`'s
job, not the net's — the same "picking which head to call... is the agent's
job, not the net's" split `heads.py` already documents for phase-head
routing, just applied one step further. Everything else — encoder, pooling,
per-phase head routing, `value_mask`/`group_index` call convention — stays
identical to `Dueling_DQN`.

    net = PQN(in_dim=13, hidden_dim=64, edge_dim=2, u_dim=34)
    value, advantage = net(state, phase, card_indices, value_mask=value_mask, group_index=group_ix)
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from risk.game.phase import Phase
from risk.learning.encoder import Encoder
from risk.learning.heads import ScoringHead, TradeInHead
from risk.learning.pooling import pool


class PQN(nn.Module):
    """`(state graph(s), phase per graph, card_indices, group_index) -> (V(s), A(s, a))`."""

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
        self.value_head = nn.Sequential(
            nn.Linear(g_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Same plain-dict alias map as `Dueling_DQN` — see its docstring for
        # why this isn't a `ModuleDict`. These heads output `A(s, a)`.
        self._heads_by_phase = {
            int(Phase.TRADE_IN): self.trade_in_head,
            int(Phase.REINFORCE_PLACE): self.reinforce_place_head,
            int(Phase.ATTACK): self.attack_head,
            int(Phase.OCCUPY): self.occupy_head,
            int(Phase.FORTIFY): self.fortify_head,
        }

    def forward(
        self,
        state: Batch,
        phase: torch.Tensor,
        card_indices: torch.Tensor,
        value_mask: torch.Tensor,
        group_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same row/phase/card_indices/value_mask/group_index convention as
        `Dueling_DQN.forward` (see its docstring for the full contract).

        Returns `(value, advantage)` uncombined, not a fused `Q`:

        - `value`: `[n_groups]`, one `V(s)` per decision group, from the
          clean (non-injected) row.
        - `advantage`: `[N_action_rows]`, one `A(s, a_i)` per action row.

        `Q(s, a_i) = V(s) + A(s, a_i) - mean_j(A(s, a_j))` and
        `pi(a_i | s) = softmax(A(s, legal_actions))[i]` are both
        `pqn_agent.py`'s responsibility (`Docs/PQN.md` §24.A-B), not this
        method's — it only produces the two raw streams the heads compute.
        """
        h = self.encoder(state.x, state.edge_index, state.edge_attr)
        g = pool(h, state.batch, state.u)

        if group_index is None:
            group_index = torch.zeros(g.shape[0], dtype=torch.long, device=g.device)
        else:
            group_index = group_index.to(device=g.device, dtype=torch.long)

        value_mask = value_mask.to(device=g.device, dtype=torch.bool)
        if value_mask.shape != group_index.shape:
            raise ValueError(
                f"value_mask shape {tuple(value_mask.shape)} must match "
                f"group_index shape {tuple(group_index.shape)}"
            )
        if not value_mask.any():
            raise ValueError("PQN.forward requires at least one clean value row")

        n_groups = int(group_index.max().item()) + 1
        action_mask = ~value_mask
        action_phase = phase.to(device=g.device)[action_mask]
        action_card_indices = card_indices.to(device=g.device)[action_mask]
        action_g = g[action_mask]

        advantage = torch.empty(action_g.shape[0], dtype=g.dtype, device=g.device)
        for stage in range(5):
            mask = action_phase == stage
            if mask.any():
                advantage[mask] = self._heads_by_phase[stage](action_g[mask], action_card_indices[mask])

        value = self.value_head(g[value_mask]).squeeze(-1)
        value_mean = scatter(
            value, group_index[value_mask], dim=0, dim_size=n_groups, reduce="mean"
        )

        return value_mean, advantage


__all__ = ["PQN"]
