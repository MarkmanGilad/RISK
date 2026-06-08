"""Track the 'last AI action' highlight overlay.

Pure state, no pygame. `highlights()` returns the dict the renderer expects, or
None when nothing should be highlighted.
"""
from __future__ import annotations

from typing import Optional

from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    TradeInAction,
)
from risk.game.board_topology import BoardTopology


def action_territories(action: Action, pre_pending) -> list[str]:
    """Return the territory ids a marker should highlight for `action`.

    `pre_pending` is the `pending_attack` captured *before* the action ran (an
    OccupyAction needs it to know which two territories were involved).
    """
    if isinstance(action, ReinforcementAction):
        return [t for t, n in action.placements.items() if n > 0]
    if isinstance(action, AttackAction):
        return [action.from_territory, action.to_territory]
    if isinstance(action, OccupyAction):
        if pre_pending is None:
            return []
        topo = BoardTopology()
        return [
            topo.territory_at(pre_pending.from_index),
            topo.territory_at(pre_pending.to_index),
        ]
    if isinstance(action, FortifyAction):
        if action.is_skip:
            return []
        return [action.from_territory, action.to_territory]
    return []


def describe_action(action: Action, player_name: str, pre_pending=None) -> str:
    """Human-readable one-line summary of an executed action."""
    if isinstance(action, ReinforcementAction):
        return f"{player_name} reinforced {action.total} armies"
    if isinstance(action, TradeInAction):
        return f"{player_name} traded a card set"
    if isinstance(action, AttackAction):
        return (
            f"{player_name} attacked {action.from_territory} -> "
            f"{action.to_territory} ({action.dice}d)"
        )
    if isinstance(action, OccupyAction):
        dest = ""
        if pre_pending is not None:
            dest = " into " + BoardTopology().territory_at(pre_pending.to_index)
        return f"{player_name} moved {action.count} armies{dest}"
    if isinstance(action, FortifyAction):
        if action.is_skip:
            return f"{player_name} skipped fortify"
        return (
            f"{player_name} fortified {action.from_territory} -> "
            f"{action.to_territory} ({action.count})"
        )
    return f"{player_name} ended attacking"


def _fmt_dice(rolls) -> str:
    return ", ".join(str(r) for r in rolls) if rolls else "-"


def action_report(action: Action, player_name: str, info=None, pre_pending=None) -> tuple[str, ...]:
    """Multi-line summary of an executed action for the side panel.

    For attacks this includes the dice that were rolled and the casualties on
    each side. Everything else falls back to the one-line `describe_action`.
    """
    info = info or {}
    if isinstance(action, AttackAction):
        a_loss = info.get("attacker_losses", 0)
        d_loss = info.get("defender_losses", 0)
        lines = [
            f"{player_name} attacked",
            f"{action.from_territory} -> {action.to_territory}",
            f"atk dice: {_fmt_dice(info.get('attacker_rolls'))}",
            f"def dice: {_fmt_dice(info.get('defender_rolls'))}",
            f"attacker lost {a_loss}, defender lost {d_loss}",
        ]
        if info.get("conquered"):
            lines.append("territory conquered!")
        return tuple(lines)
    return (describe_action(action, player_name, pre_pending),)


class ActionMarker:
    def __init__(self, color: tuple[int, int, int], duration_ms: int) -> None:
        self.color = color
        self.duration_ms = duration_ms
        self._territories: list[str] = []
        self._expires_ms = 0

    def set_action(self, action: Action, pre_pending, now_ms: int) -> None:
        terrs = action_territories(action, pre_pending)
        if terrs and self.duration_ms > 0:
            self._territories = terrs
            self._expires_ms = now_ms + self.duration_ms

    def expire(self, now_ms: int) -> None:
        if self._territories and now_ms >= self._expires_ms:
            self._territories = []

    def highlights(self) -> Optional[dict[str, tuple[int, int, int]]]:
        if not self._territories:
            return None
        return {t: self.color for t in self._territories}


__all__ = ["ActionMarker", "action_territories", "describe_action", "action_report"]
