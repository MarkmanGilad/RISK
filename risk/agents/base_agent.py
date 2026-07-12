"""Agent interface for the Risk game loop.

Every seat — human, random, future RL — is a uniform callable:

    action = agent(events, state)   # -> Action | None

The loop hands each agent the frame's input events and the current
read-only state. AI agents ignore `events`, read `legal_actions` from the
environment they hold, and return a move immediately. A `HumanAgent`
consumes `events`, building a move across many frames, and returns `None`
until its decision is complete.

The Environment owns all legality checks; the loop validates by running
`env.step(action)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from risk.agents.human_input import HudActionPanelModel
from risk.game.actions import Action
from risk.game.state import State


class BaseAgent(ABC):
    """Contract for any agent (human, random, future RL)."""

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def __call__(
        self, events: Sequence[object], state: State
    ) -> Optional[Action]:
        """Uniform entry point used by the loop."""
        return self.act(events, state)

    @abstractmethod
    def act(
        self, events: Sequence[object], state: State
    ) -> Optional[Action]:
        """Return an action, or `None` if more input is needed (humans)."""
        raise NotImplementedError

    def widgets(self, state: State) -> HudActionPanelModel:
        """HUD action-panel model for this agent. Empty for non-humans."""
        return HudActionPanelModel()

    def on_turn_start(self, state: State) -> None:
        """Optional hook called at the start of each owned turn."""
        return None

    def on_turn_end(self, state: State) -> None:
        """Optional hook called at the end of each owned turn."""
        return None

    def on_episode_start(self, episode: int) -> None:
        """Optional hook called once per training episode, before the
        episode plays. No-op by default; DQN-style agents override it to
        update per-episode exploration state (`Docs/Trainer.md`)."""
        return None


__all__ = ["BaseAgent"]
