"""A non-RL random agent. Used to validate AI seats end-to-end.

The agent holds the `Environment` and queries `legal_actions()` itself, so
the loop can treat it like any other agent: `agent(events, state)`. It
ignores `events` and returns a move immediately (never `None` while it has
legal actions).
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.environment import Environment
from risk.game.state import State


class RandomAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        env: Optional[Environment] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id)
        self.env = env
        self._rng = random.Random(seed)

    def act(self, events: Sequence[object], state: State) -> Optional[Action]:
        if self.env is None:
            return None
        legal = self.env.legal_actions()
        if not legal:
            return None
        return self._rng.choice(list(legal))


__all__ = ["RandomAgent"]
