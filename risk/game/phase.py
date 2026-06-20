"""Turn-phase enum used across the game.

Kept in its own module so `State`, `Environment`, and action classes can
all agree on the current step without a circular import. The integer
values are stable so a `State` can be serialized to JSON and replayed —
this set of values has changed twice now (first when `REINFORCE` split
into `TRADE_IN` + `REINFORCE_PLACE`, now to fold in what used to be a
separate `ActionStage` enum — see `Docs/RL-Prep-Changes.md`), both one-time
breaks with no persisted save files affected.

`Phase` doubles as the DQN action-representation "stage" now — `Action.phase`
(`risk/game/actions.py`) is read directly for that, no separate `ActionStage`
type. A player is always in exactly one phase, and an explicit action ends
it and advances to the next (trading ends via `SkipTradeAction` or running
out of trades, placing ends once the budget is spent, attacking ends via
`StopAttackAction`, occupying ends automatically after one `OccupyAction`,
fortifying ends via a skip `FortifyAction`). `SETUP`/`GAME_OVER` aren't
agent decisions, so no `Action` ever has those as its `phase`.

Numbered so `TRADE_IN..GAME_OVER` (0-5) cover everything an `Action`/
`ReplayBuffer` transition ever sees, with `done == (phase == GAME_OVER)` a
clean invariant — `SETUP` sits last (6) since it's never actually observed
during play, only via `State.initial()` directly, before the first turn
begins.
"""
from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    TRADE_IN = 0
    REINFORCE_PLACE = 1
    ATTACK = 2
    OCCUPY = 3
    FORTIFY = 4
    GAME_OVER = 5
    SETUP = 6


__all__ = ["Phase"]
