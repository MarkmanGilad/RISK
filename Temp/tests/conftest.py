"""Shared `GameSettings` / `Environment` builders for the test suite.

Several test modules each defined their own near-identical `_settings()` /
`_fresh_env()` helpers (same shape: build N players, wrap in GameSettings,
optionally reset an Environment). Centralized here so there's one place to
look when the `Player`/`GameSettings` constructor signature changes.
"""
from __future__ import annotations

from typing import Optional

from risk.game.environment import Environment
from risk.game.player import Player
from risk.game.settings import GameSettings


def make_settings(
    n: int = 3,
    seed: Optional[int] = 0,
    agent_kind: str = "ai",
    human_ids: Optional[set[int]] = None,
) -> GameSettings:
    """Build a GameSettings with `n` players.

    Every seat defaults to `agent_kind`; ids in `human_ids` (if given) are
    "human" instead, for tests that need a mix of human/AI seats.
    """
    human_ids = human_ids or set()
    return GameSettings(
        players=tuple(
            Player(
                id=i,
                name=f"P{i}",
                color=(i * 30, i * 40, i * 50),  # stays in 0..255 for i in 0..5
                agent_kind="human" if i in human_ids else agent_kind,
            )
            for i in range(n)
        ),
        seed=seed,
    )


def make_env(
    n: int = 3,
    seed: Optional[int] = 0,
    agent_kind: str = "ai",
    human_ids: Optional[set[int]] = None,
) -> Environment:
    """Build and `reset()` an Environment from `make_settings(...)`."""
    env = Environment()
    env.reset(make_settings(n=n, seed=seed, agent_kind=agent_kind, human_ids=human_ids))
    return env
