"""Agent interface for the Risk game loop.

Agents are *pure decision functions*: given a read-only state and the
list of legal actions, they return one Action — or `None` if they have
not yet finished deciding (used by `HumanAgent` to wait for input).

The Environment owns all legality checks; an agent must only return
something that appears in `legal_actions`. The game loop validates by
re-running `env.step(action)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from risk.game.actions import Action
from risk.game.state import State


class BaseAgent(ABC):
    """Contract for any agent (human, random, future RL)."""

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    @abstractmethod
    def act(self, state: State, legal_actions: Sequence[Action]) -> Optional[Action]:
        """Return an action, or `None` if more input is needed (humans)."""
        raise NotImplementedError

    def on_turn_start(self, state: State) -> None:
        """Optional hook called at the start of each owned turn."""
        return None

    def on_turn_end(self, state: State) -> None:
        """Optional hook called at the end of each owned turn."""
        return None


__all__ = ["BaseAgent"]
