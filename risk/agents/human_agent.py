"""Human agent. Bridges pygame input events to actions.

The UI (Phase 6) calls `submit(action)` from event handlers; the game
loop calls `act(state, legal_actions)` once per tick. Returns `None`
until `submit` was called with a legal action.
"""
from __future__ import annotations

from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.state import State


class HumanAgent(BaseAgent):
    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self._pending: Optional[Action] = None

    def submit(self, action: Action) -> None:
        """Called by the UI layer once a complete decision has been assembled."""
        self._pending = action

    def clear(self) -> None:
        self._pending = None

    def act(self, state: State, legal_actions: Sequence[Action]) -> Optional[Action]:
        if self._pending is None:
            return None
        # Only return the pending action if it is legal; otherwise drop and wait.
        # Equality between actions is by `to_dict` to tolerate equivalent constructions.
        legal_dicts = [a.to_dict() for a in legal_actions]
        if self._pending.to_dict() in legal_dicts:
            chosen = self._pending
            self._pending = None
            return chosen
        # Illegal -> silently discard (UI should have prevented it) and wait.
        self._pending = None
        return None


__all__ = ["HumanAgent"]
