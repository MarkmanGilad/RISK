"""Headless game loop orchestrator.

`Game` wires Environment + Agents. It does NOT own rules and does NOT
own rendering. The pygame app entry (`risk/app/main.py`) wraps a `Game`
with a renderer and an event loop.

Designed to be drivable from tests:
    game = Game(env, agents)
    while not game.is_terminal():
        game.tick()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.environment import Environment, StepResult


@dataclass
class TickResult:
    """Outcome of one `tick()`. `step` is None if no agent action this tick."""

    step: Optional[StepResult]
    waiting_on_player: int


class Game:
    def __init__(self, env: Environment, agents: Sequence[BaseAgent]) -> None:
        if len(agents) != env.settings.player_count:
            raise ValueError(
                f"Expected {env.settings.player_count} agents, got {len(agents)}"
            )
        # Defensive: agents must cover every player id.
        agent_ids = sorted(a.player_id for a in agents)
        expected_ids = sorted(p.id for p in env.settings.players)
        if agent_ids != expected_ids:
            raise ValueError(
                f"Agent player_ids {agent_ids} do not match game ids {expected_ids}"
            )
        self.env = env
        self.agents: list[BaseAgent] = sorted(agents, key=lambda a: a.player_id)
        # Agents query `env.legal_actions()` themselves; make sure each one
        # points at this environment (tests may construct agents bare).
        for agent in self.agents:
            if getattr(agent, "env", None) is None:
                agent.env = env
        self.history: list[Action] = []
        self.last_info: dict = {}

    # --- queries ----------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.env.is_terminal()

    def winner(self) -> Optional[int]:
        return self.env.winner()

    def current_agent(self) -> BaseAgent:
        pid = self.env.current_state().current_player_index
        return self.agents[pid]

    # --- loop -------------------------------------------------------------

    def tick(self) -> TickResult:
        """Advance one decision step (or wait if a HumanAgent is undecided)."""
        if self.is_terminal():
            return TickResult(step=None, waiting_on_player=-1)
        state = self.env.current_state()
        pid = state.current_player_index
        agent = self.agents[pid]
        chosen = agent((), state)
        if chosen is None:
            return TickResult(step=None, waiting_on_player=pid)
        result = self.env.step(chosen)
        self.history.append(chosen)
        self.last_info = result.info
        return TickResult(step=result, waiting_on_player=pid)

    def play_until_terminal(self, max_steps: int = 10000) -> Optional[int]:
        """Run the loop until termination or `max_steps`. Returns winner id."""
        for _ in range(max_steps):
            if self.is_terminal():
                break
            r = self.tick()
            if r.step is None:
                # Waiting on a human-style agent and no event source -> bail.
                break
        return self.winner()


__all__ = ["Game", "TickResult"]
