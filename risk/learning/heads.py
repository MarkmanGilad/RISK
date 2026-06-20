"""Per-stage scoring heads, shared by every net (`Docs/NetworkArchitectures.md`).

Five small heads keyed by `Phase` — the stages are different decisions
with different `n`-semantics and value scales, so one head per stage
avoids forcing a shared tail to disambiguate that implicitly. What differs
per net is only the head's *input* (a pooled graph for an injected action,
for Nets A/C, a lookup vector for B/D) and the output's *meaning* (`Q` for
A/B, `logit` for C/D) — each head class just maps its input to a scalar.

Each head is its own named `nn.Module` instance (e.g. `net.attack_head`),
not a `ModuleDict` with internal dispatch — picking which head to call for
which legal actions is the *agent's* job, not the net's (`Docs/
NetworkArchitectures.md`'s Net A section). `REINFORCE_PLACE`/`ATTACK`/
`OCCUPY`/`FORTIFY` share identical architecture, so they're four separate
`ScoringHead` instances (independent weights, same class) rather than four
near-duplicate class bodies.

`TRADE_IN` is the odd stage out: `ActionGraphBuilder` refuses to inject it
(its legal actions are card-hand-slot indices, not territories — there's
nothing on the graph for it to perturb), so it never produces a per-action
`g` the way the other four stages do. `TradeInHead` instead reads the
*decision's* pooled base-graph context plus a small learned embedding for
the three card-slot indices — a different enough input shape to warrant
its own class.
"""
from __future__ import annotations

import torch
from torch import nn

from risk.constants import MAX_CARDS_IN_HAND


def _mlp(in_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )


class ScoringHead(nn.Module):
    """`(g, card_indices) -> Q(s, a)` (or `logit(s, a)` for the PPO nets).

    `g`: `[N, g_dim]` -> `[N]`. Takes `card_indices` too, unused, purely so
    every head — this one and `TradeInHead` — shares one call signature:
    `GNN_DQN.forward` then calls whichever head a row's `phase` picks out
    the same way regardless of which one it is, no per-head branching.
    """

    def __init__(self, g_dim: int) -> None:
        super().__init__()
        self.net = _mlp(g_dim)

    def forward(self, g: torch.Tensor, card_indices: torch.Tensor) -> torch.Tensor:
        return self.net(g).squeeze(-1)


class TradeInHead(nn.Module):
    """`(g_base, card_indices) -> Q(s, a)`, one scalar per `TRADE_IN` action.

        head = TradeInHead(g_dim)
        q = head(g_base, card_indices)   # g_base: [N, g_dim], card_indices: [N, 3] long -> [N]

    `g_base` is the decision's pooled *base*-graph context, one row per
    action (the same row repeated — `TRADE_IN` actions share one state,
    just different card slots). `card_indices` is each action's
    `dqn_index()` `(t1, t2, n)` — real card-hand-slot indices for
    `TradeInAction`, or the `(-1, -1, 0)` sentinel for `SkipTradeAction`.
    The embedding table has one extra row reserved for that sentinel (`-1`
    -> a fixed *learned* "none" vector, never a real lookup — `Docs/
    Action.md`'s "Network wiring" section). Detected per *row*, not per
    element: `n == 0` is `t1`'s "stop" sentinel for `SkipTradeAction` but a
    legitimate real card-slot index for `TradeInAction`, so only `t1 < 0`
    (unambiguous either way) decides whether the whole row is a skip.
    """

    def __init__(self, g_dim: int, card_embed_dim: int = 8) -> None:
        super().__init__()
        self._none_slot = MAX_CARDS_IN_HAND
        self.card_embedding = nn.Embedding(MAX_CARDS_IN_HAND + 1, card_embed_dim)
        self.net = _mlp(g_dim + 3 * card_embed_dim)

    def forward(self, g_base: torch.Tensor, card_indices: torch.Tensor) -> torch.Tensor:
        is_skip = card_indices[:, 0] < 0
        idx = card_indices.masked_fill(is_skip.unsqueeze(1), self._none_slot)
        cards = self.card_embedding(idx).flatten(1)
        return self.net(torch.cat([g_base, cards], dim=-1)).squeeze(-1)


__all__ = ["ScoringHead", "TradeInHead"]
