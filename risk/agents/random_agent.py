"""A non-RL random agent. Used to validate AI seats end-to-end."""
from __future__ import annotations

import random
from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.state import State


class RandomAgent(BaseAgent):
    def __init__(self, player_id: int, seed: Optional[int] = None) -> None:
        super().__init__(player_id)
        self._rng = random.Random(seed)

    def act(self, state: State, legal_actions: Sequence[Action]) -> Optional[Action]:
        if not legal_actions:
            return None
        return self._rng.choice(list(legal_actions))


__all__ = ["RandomAgent"]
