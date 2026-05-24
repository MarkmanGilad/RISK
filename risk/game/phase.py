"""Turn-phase enum used across the game.

Kept in its own module so `State`, `Environment`, and action classes can
all agree on the current step without a circular import. The integer
values are stable so a `State` can be serialized to JSON and replayed.
"""
from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    SETUP = 0
    REINFORCE = 1
    ATTACK = 2
    FORTIFY = 3
    GAME_OVER = 4
    OCCUPY = 5


__all__ = ["Phase"]
