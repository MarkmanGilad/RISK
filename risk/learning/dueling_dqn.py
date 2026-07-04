"""Dueling variant of Net A's network — `Docs/DuelingDQN.md`.

`Dueling_DQN` is a copy of `GNN_DQN` (`gnn_dqn.py`) with one architectural
change: instead of scoring `Q(s, a)` directly, each per-phase head now
outputs an advantage `A(s, a)`, and a new `value_head` estimates `V(s)`.
They combine as `Q(s, a) = V(s) + A(s, a) - mean(A(s, legal_actions))`, the
mean taken over the legal actions of the same decision (`group_index`).
Per the minimal-diff policy in `Docs/DuelingDQN.md`, everything else —
encoder, pooling, per-phase head routing, call convention — stays identical
to `GNN_DQN`.

    net = Dueling_DQN(in_dim=13, hidden_dim=64, edge_dim=2, u_dim=34)
    q = net(state, phase, card_indices, group_index=group_ix, value_mask=value_mask)
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


class Dueling_DQN(nn.Module):
    """`(state graph(s), phase per graph, card_indices, group_index) -> Q(s, a)`."""

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
        # Same plain-dict alias map as `GNN_DQN` — see its docstring for why
        # this isn't a `ModuleDict`. These heads now output `A(s, a)`, not
        # final `Q(s, a)`; `forward` combines them with `value_head` below.
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
        group_index: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same row/phase/card_indices convention as `GNN_DQN.forward`.

        `group_index`: `[N]` long tensor saying which decision each row
        belongs to — the dueling mean is taken over each group's rows.
        Omit it (or pass all-zeros) when every row in the batch belongs to
        one decision, e.g. `score_actions`' "one state, all legal actions"
        shape. Pass real group ids (`0..B-1`) for a flattened replay
        minibatch where each group is one sampled transition's legal-action
        set (`Docs/DuelingDQN.md`'s "Important training detail").

        `value_mask`: `[N]` bool tensor marking clean, non-injected state rows
        used for `V(s)`. All unmasked rows are action-injected rows routed to
        the phase advantage heads. Returns one Q value per action row, not for
        the clean value rows.
        """
        h = self.encoder(state.x, state.edge_index, state.edge_attr)
        g = pool(h, state.batch, state.u)

        if group_index is None:
            group_index = torch.zeros(g.shape[0], dtype=torch.long, device=g.device)
        else:
            group_index = group_index.to(device=g.device, dtype=torch.long)

        # Backward-compatible fallback for ad-hoc calls without clean value
        # rows: use the old v1 approximation (average value over action rows).
        if value_mask is None:
            advantage = torch.empty(g.shape[0], dtype=g.dtype, device=g.device)
            for stage in range(5):
                mask = phase == stage
                if mask.any():
                    advantage[mask] = self._heads_by_phase[stage](g[mask], card_indices[mask])

            value = self.value_head(g).squeeze(-1)
            n_groups = int(group_index.max().item()) + 1
            value_mean = scatter(value, group_index, dim=0, dim_size=n_groups, reduce="mean")
            adv_mean = scatter(advantage, group_index, dim=0, dim_size=n_groups, reduce="mean")
            return value_mean[group_index] + advantage - adv_mean[group_index]

        value_mask = value_mask.to(device=g.device, dtype=torch.bool)
        if value_mask.shape != group_index.shape:
            raise ValueError(
                f"value_mask shape {tuple(value_mask.shape)} must match "
                f"group_index shape {tuple(group_index.shape)}"
            )
        if not value_mask.any():
            raise ValueError("Dueling_DQN.forward requires at least one clean value row")

        n_groups = int(group_index.max().item()) + 1
        action_mask = ~value_mask
        action_group_index = group_index[action_mask]
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
        adv_mean = scatter(
            advantage, action_group_index, dim=0, dim_size=n_groups, reduce="mean"
        )

        return value_mean[action_group_index] + advantage - adv_mean[action_group_index]


__all__ = ["Dueling_DQN"]
