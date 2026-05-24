"""Shared, pygame-free game constants for the Risk engine.

Everything here is a plain value or a pure function. No state, no I/O,
no pygame import. `Environment`, `State`, action classes, agents, and UI
code may all read from this module without creating import cycles.
"""
from __future__ import annotations

from typing import Final


# --- Players ---------------------------------------------------------------

MIN_PLAYERS: Final[int] = 3
MAX_PLAYERS: Final[int] = 6

# Classic Risk starting armies per player count.
# Indexed by player count for direct lookup.
STARTING_ARMIES: Final[dict[int, int]] = {
    3: 35,
    4: 30,
    5: 25,
    6: 20,
}


# --- Reinforcement ---------------------------------------------------------

# At the start of a turn, a player gets max(MIN_REINFORCEMENT,
# owned_territories // TERRITORIES_PER_REINFORCEMENT) plus continent bonuses.
MIN_REINFORCEMENT: Final[int] = 3
TERRITORIES_PER_REINFORCEMENT: Final[int] = 3


# --- Dice / combat ---------------------------------------------------------

MAX_ATTACK_DICE: Final[int] = 3
MAX_DEFEND_DICE: Final[int] = 2
DIE_SIDES: Final[int] = 6


# --- Cards -----------------------------------------------------------------

# Symbols on the territory cards plus the special wild card.
CARD_SYMBOLS: Final[tuple[str, ...]] = ("infantry", "cavalry", "artillery")
WILD_SYMBOL: Final[str] = "wild"

# Maximum cards a player may hold before being forced to trade in.
MAX_CARDS_IN_HAND: Final[int] = 5

# Classic Risk card-set trade-in progression. After the 6th trade-in
# (value 15) each further trade-in is worth the previous value + 5.
CARD_SET_VALUES: Final[tuple[int, ...]] = (4, 6, 8, 10, 12, 15)
CARD_SET_INCREMENT_AFTER_FIXED: Final[int] = 5


def card_set_value(trade_in_index: int) -> int:
    """Reinforcement bonus awarded for the n-th card set traded in (0-based).

    The first six values come from :data:`CARD_SET_VALUES`. From the seventh
    trade-in (index 6) onward each set is worth 5 more than the previous.
    """
    if trade_in_index < 0:
        raise ValueError(f"trade_in_index must be >= 0, got {trade_in_index}")
    if trade_in_index < len(CARD_SET_VALUES):
        return CARD_SET_VALUES[trade_in_index]
    extra = trade_in_index - (len(CARD_SET_VALUES) - 1)
    return CARD_SET_VALUES[-1] + extra * CARD_SET_INCREMENT_AFTER_FIXED


def starting_armies_for(player_count: int) -> int:
    """Number of armies each player starts the game with."""
    if player_count not in STARTING_ARMIES:
        raise ValueError(
            f"player_count must be {MIN_PLAYERS}..{MAX_PLAYERS}, got {player_count}"
        )
    return STARTING_ARMIES[player_count]


__all__ = [
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "STARTING_ARMIES",
    "MIN_REINFORCEMENT",
    "TERRITORIES_PER_REINFORCEMENT",
    "MAX_ATTACK_DICE",
    "MAX_DEFEND_DICE",
    "DIE_SIDES",
    "CARD_SYMBOLS",
    "WILD_SYMBOL",
    "MAX_CARDS_IN_HAND",
    "CARD_SET_VALUES",
    "CARD_SET_INCREMENT_AFTER_FIXED",
    "card_set_value",
    "starting_armies_for",
]
