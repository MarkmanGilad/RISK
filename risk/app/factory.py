"""Build stage: turn a `GameSettings` into a runnable game.

This module is **pygame-free** on purpose. It is the single seam shared by the
interactive app and headless training: both call `GameFactory.build(settings)` and get
back a `GameContext` they can run.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk.agents.base_agent import BaseAgent
from risk.agents.heuristic_agent import EmpireAgent, KillbotAgent, RaiderAgent, SentinelAgent
from risk.agents.random_agent import RandomAgent
from risk.app.game import Game
from risk.game.environment import Environment
from risk.game.settings import GameSettings


@dataclass
class GameContext:
    """The runnable game produced from a `GameSettings`."""

    settings: GameSettings
    env: Environment
    agents: list[BaseAgent]
    game: Game


class GameFactory:
    """Wires a `GameSettings` into a runnable `GameContext`."""

    @staticmethod
    def build_agents(settings: GameSettings, env: Environment) -> list[BaseAgent]:
        """Instantiate one agent per seat from the roster.

        Every agent holds the `env` so it can query `legal_actions()` itself.
        Human agents also receive the `view` later (injected by the loop) for
        pixel hit-testing; headless paths leave it unset.
        """
        out: list[BaseAgent] = []
        for p in settings.players:
            seed = (settings.seed or 0) + p.id + 1
            if p.agent_kind == "human":
                from risk.agents.human_agent import HumanAgent

                out.append(HumanAgent(player_id=p.id, env=env, settings=settings))
            elif p.agent_kind in {"ai", "random"}:
                out.append(
                    RandomAgent(
                        player_id=p.id, env=env, seed=seed
                    )
                )
            elif p.agent_kind == "raider":
                out.append(RaiderAgent(player_id=p.id, env=env, seed=seed))
            elif p.agent_kind == "sentinel":
                out.append(SentinelAgent(player_id=p.id, env=env, seed=seed))
            elif p.agent_kind == "empire":
                out.append(EmpireAgent(player_id=p.id, env=env, seed=seed))
            elif p.agent_kind == "killbot":
                out.append(KillbotAgent(player_id=p.id, env=env, seed=seed))
            else:
                raise ValueError(f"Unknown agent_kind {p.agent_kind!r}")
        return out

    @classmethod
    def build(cls, settings: GameSettings) -> GameContext:
        """Wire Environment + agents + Game. No pygame, no rendering."""
        env = Environment()
        env.reset(settings)
        agents = cls.build_agents(settings, env)
        game = Game(env, agents)
        return GameContext(settings=settings, env=env, agents=agents, game=game)


__all__ = ["GameContext", "GameFactory"]
