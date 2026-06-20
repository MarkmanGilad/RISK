"""Turn-phase enum used across the game.

Kept in its own module so `State`, `Environment`, and action classes can
all agree on the current step without a circular import. The integer
values are stable so a `State` can be serialized to JSON and replayed —
note this set of values changed when `REINFORCE` split into `TRADE_IN` +
`REINFORCE_PLACE` (see `Docs/RL-Prep-Changes.md`), a one-time break with no
persisted save files affected.

One phase per `ActionStage` (`risk/game/actions.py`), plus `SETUP`/
`GAME_OVER` which aren't agent decisions: a player is always in exactly one
phase, and an explicit action ends it and advances to the next (trading
ends via `SkipTradeAction` or running out of trades, placing ends once the
budget is spent, attacking ends via `StopAttackAction`, occupying ends
automatically after one `OccupyAction`, fortifying ends via a skip
`FortifyAction`).
"""
from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    SETUP = 0
    TRADE_IN = 1
    REINFORCE_PLACE = 2
    ATTACK = 3
    OCCUPY = 4
    FORTIFY = 5
    GAME_OVER = 6


__all__ = ["Phase"]
