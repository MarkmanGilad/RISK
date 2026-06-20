"""Shared GNN encoder for all four nets (`Docs/NetworkArchitectures.md`).

One module, reused unmodified by every net regardless of algorithm
(DQN/PPO) or action representation (inject/lookup) — what differs between
them is only what gets passed into `forward`, not the encoder itself:

- **Inject** (Nets A, C): `edge_attr` carries `ActionGraphBuilder`'s
  per-action marker, `x` is the (possibly perturbed) node features.
  Called once per *action*.
- **Lookup** (Nets B, D): `edge_attr` is zero/unused, `x` is the
  unmodified base graph's. Called once per *decision*.

`TransformerConv` (graph attention, not a spectral GCN) still runs
attention either way — a zero `edge_attr` just means the edge term
contributes nothing, degrading to plain neighbor-attention rather than a
no-op.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv


class Encoder(nn.Module):
    """`(x, edge_index, edge_attr) -> H`, one row of `hidden_dim` per node."""

    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int, n_layers: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv in self.convs:
            h = h + F.relu(conv(h, edge_index, edge_attr))   # residual
        return h


__all__ = ["Encoder"]
