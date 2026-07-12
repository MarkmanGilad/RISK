"""Injected-action policy/value network used by :class:`PPO_Agent`."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from risk.game.phase import Phase
from risk.learning.encoder import Encoder
from risk.learning.heads import ScoringHead, TradeInHead
from risk.learning.pooling import pool


class PPO_Net(nn.Module):
    """Return one policy logit per action row and one clean-row value per group."""

    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int, u_dim: int,
                 n_layers: int = 4, card_embed_dim: int = 8) -> None:
        super().__init__()
        g_dim = 2 * hidden_dim + u_dim
        self.encoder = Encoder(in_dim, hidden_dim, edge_dim, n_layers)
        self.reinforce_place_head = ScoringHead(g_dim)
        self.attack_head = ScoringHead(g_dim)
        self.occupy_head = ScoringHead(g_dim)
        self.fortify_head = ScoringHead(g_dim)
        self.trade_in_head = TradeInHead(g_dim, card_embed_dim)
        self.value_head = nn.Sequential(nn.Linear(g_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self._heads_by_phase = {
            int(Phase.TRADE_IN): self.trade_in_head,
            int(Phase.REINFORCE_PLACE): self.reinforce_place_head,
            int(Phase.ATTACK): self.attack_head,
            int(Phase.OCCUPY): self.occupy_head,
            int(Phase.FORTIFY): self.fortify_head,
        }

    def forward(self, state: Batch, phase: torch.Tensor, card_indices: torch.Tensor,
                group_index: torch.Tensor, value_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(state.x, state.edge_index, state.edge_attr)
        g = pool(h, state.batch, state.u)
        group_index = group_index.to(g.device, dtype=torch.long)
        value_mask = value_mask.to(g.device, dtype=torch.bool)
        if group_index.shape != value_mask.shape or not value_mask.any():
            raise ValueError("PPO_Net requires matching group_index/value_mask and a clean value row")
        action_mask = ~value_mask
        action_phase = phase.to(g.device)[action_mask]
        action_cards = card_indices.to(g.device)[action_mask]
        action_g = g[action_mask]
        logits = torch.empty(action_g.shape[0], dtype=g.dtype, device=g.device)
        for stage, head in self._heads_by_phase.items():
            mask = action_phase == stage
            if mask.any():
                logits[mask] = head(action_g[mask], action_cards[mask])
        groups = group_index[value_mask]
        n_groups = int(group_index.max().item()) + 1
        values = scatter(self.value_head(g[value_mask]).squeeze(-1), groups, dim=0, dim_size=n_groups, reduce="mean")
        return logits, values


__all__ = ["PPO_Net"]
