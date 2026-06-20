"""Whole-graph pooling, shared by every net (`Docs/NetworkArchitectures.md`).

Turns `Encoder`'s per-node `H` into one vector per graph — used wherever an
action or a state needs a single scalar/vector readout (a head's input for
Nets A-C, the critic's input for Nets C/D).
"""
from __future__ import annotations

import torch
from torch_geometric.nn import global_max_pool, global_mean_pool


def pool(h: torch.Tensor, batch: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """`(H, batch index, globals) -> g`, one row per graph.

    `u` is concatenated in after pooling, not fed through the GNN — it's
    already global (phase, budget, eliminations, ...), nothing for message
    passing to refine, and pure node pooling alone is blind to it.
    """
    return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), u], dim=-1)


__all__ = ["pool"]
