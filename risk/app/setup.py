"""Pre-game stage: choose who plays and produce a `GameSettings`.

The only output of this stage is **data** (`GameSettings`). It may open the
pygame setup screen, but it owns no game rules and no game loop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from risk.game.settings import GameSettings
from risk.ui.input.init_screen import InitScreenState

if TYPE_CHECKING:
    import pygame


class SetupStage:
    """Pre-game stage: produces a `GameSettings` from either a default
    roster or the interactive setup screen."""

    @staticmethod
    def default_settings(n: int = 3, seed: Optional[int] = 0) -> GameSettings:
        """A default all-random roster (used by smoke tests and `--skip-menu`)."""
        state = InitScreenState()
        state.set_player_count(n)
        for i in range(n):
            state.set_agent_kind(i, "random")
        return state.build_settings(seed=seed)

    @classmethod
    def run_setup(
        cls,
        screen: "pygame.Surface",
        *,
        players: int,
        seed: int,
        skip_menu: bool,
    ) -> Optional[GameSettings]:
        """Return the chosen settings, or None if the user closed the menu.

        When `skip_menu` is set, bypass the interactive screen and start a
        default all-random game.
        """
        if skip_menu:
            return cls.default_settings(n=players, seed=seed)
        from risk.ui.render.init_screen_view import run_init_screen

        return run_init_screen(screen)


__all__ = ["SetupStage"]
