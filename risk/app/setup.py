"""Pre-game stage: choose who plays and produce a `GameSettings`.

The only output of this stage is **data** (`GameSettings`). It may open the
pygame setup screen, but it owns no game rules and no game loop.
"""
from __future__ import annotations

from typing import Optional

import pygame

from risk.game.settings import GameSettings
from risk.ui.input.init_screen import InitScreenState
from risk.ui.render.init_screen_view import run_init_screen


def default_settings(n: int = 3, seed: Optional[int] = 0) -> GameSettings:
    """A default all-AI roster (used by smoke tests and `--skip-menu`)."""
    state = InitScreenState()
    state.set_player_count(n)
    for i in range(n):
        state.set_agent_kind(i, "ai")
    return state.build_settings(seed=seed)


def run_setup(
    screen: pygame.Surface,
    *,
    players: int,
    seed: int,
    skip_menu: bool,
) -> Optional[GameSettings]:
    """Return the chosen settings, or None if the user closed the menu.

    When `skip_menu` is set, bypass the interactive screen and start a default
    all-AI game.
    """
    if skip_menu:
        return default_settings(n=players, seed=seed)
    return run_init_screen(screen)


__all__ = ["default_settings", "run_setup"]
