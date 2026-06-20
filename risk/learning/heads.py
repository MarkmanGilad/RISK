"""Per-stage scoring heads, shared by every net (`Docs/NetworkArchitectures.md`).

Five small heads keyed by `ActionStage` — the stages are different
decisions with different `n`-semantics and value scales, so one head per
stage avoids forcing a shared tail to disambiguate that implicitly. What
differs per net is only the head's *input* (a pooled graph for an injected
action, for Nets A/C, a lookup vector for B/D) and the output's *meaning*
(`Q` for A/B, `logit` for C/D) — `Heads` itself just maps `g -> scalar`.

`TRADE_IN` is the odd stage out: `ActionGraphBuilder` refuses to inject it
(its legal actions are card-hand-slot indices, not territories — there's
nothing on the graph for it to perturb), so it never produces a per-action
`g` the way the other four stages do. Its head instead reads
the *decision's* pooled base-graph context plus a small learned embedding
for the three card-slot indices.
"""
from __future__ import annotations

import torch
from torch import nn

from risk.constants import MAX_CARDS_IN_HAND
from risk.game.actions import ActionStage

_GRAPH_STAGES = (
    ActionStage.REINFORCE_PLACE,
    ActionStage.ATTACK,
    ActionStage.OCCUPY,
    ActionStage.FORTIFY,
)


class Heads(nn.Module):
    """`(stage, g) -> Q(s, a)` (or `logit(s, a)` for the PPO nets), one scalar per action.

        heads = Heads(g_dim=2 * hidden_dim + u_dim)
        q = heads(ActionStage.ATTACK, g)                      # g: [N, g_dim] -> [N]
        q_trade = heads.trade_in(g_base, card_indices)         # card_indices: [N, 3] long -> [N]
    """

    def __init__(self, g_dim: int, card_embed_dim: int = 8) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(MAX_CARDS_IN_HAND, card_embed_dim)
        self.heads = nn.ModuleDict(
            {stage.name: self._make_head(g_dim) for stage in _GRAPH_STAGES}
        )
        self.heads[ActionStage.TRADE_IN.name] = self._make_head(g_dim + 3 * card_embed_dim)

    def _make_head(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, stage: ActionStage, g: torch.Tensor) -> torch.Tensor:
        """Score legal actions of one of the four graph-based stages. `g`: `[N, g_dim]` -> `[N]`."""
        return self.heads[stage.name](g).squeeze(-1)

    def trade_in(self, g_base: torch.Tensor, card_indices: torch.Tensor) -> torch.Tensor:
        """Score `TRADE_IN` legal actions.

        `g_base` is the decision's pooled *base*-graph context, one row per
        action (the same row repeated — `TRADE_IN` actions share one state,
        just different card slots). `card_indices`: `[N, 3]` long.
        """
        cards = self.card_embedding(card_indices).flatten(1)
        return self.heads[ActionStage.TRADE_IN.name](torch.cat([g_base, cards], dim=-1)).squeeze(-1)


__all__ = ["Heads"]
