"""ADQN's injected-action dueling network — ``Docs/ADQN.md``.

ADQN currently owns this network as an intentional copy of the raw dueling
``(V, A)`` architecture.  Keeping a separate class makes ADQN, PQN, and
Dueling DQN algorithm siblings; a shared raw-dueling base can be considered
later without coupling their implementations now.
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


class ADQN(nn.Module):
    """``(state graphs, phase, cards, groups) -> raw (V, A)``."""

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
        """Return one raw value per group and one raw advantage per action."""
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
            raise ValueError("ADQN.forward requires at least one clean value row")

        n_groups = int(group_index.max().item()) + 1
        action_mask = ~value_mask
        action_phase = phase.to(device=g.device)[action_mask]
        action_card_indices = card_indices.to(device=g.device)[action_mask]
        action_g = g[action_mask]

        advantage = torch.empty(action_g.shape[0], dtype=g.dtype, device=g.device)
        for stage in range(5):
            mask = action_phase == stage
            if mask.any():
                advantage[mask] = self._heads_by_phase[stage](
                    action_g[mask], action_card_indices[mask]
                )

        value = self.value_head(g[value_mask]).squeeze(-1)
        value_mean = scatter(
            value, group_index[value_mask], dim=0, dim_size=n_groups, reduce="mean"
        )
        return value_mean, advantage


__all__ = ["ADQN"]
