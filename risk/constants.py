"""All game-rule constants and magic numbers for the Risk engine.

Everything here is a plain value or a pure function — no state, no I/O, no
pygame import. `Environment`, `State`, action classes, and agents all read
from this module. Import with `from risk.constants import *`.

UI-only constants (colors, layout, pacing) live in `risk.ui_constants`.
"""
from __future__ import annotations

from typing import Final

# --- Players -----------------------------------------------------------

MIN_PLAYERS: Final[int] = 3
MAX_PLAYERS: Final[int] = 6

# Classic Risk starting armies per player count.
STARTING_ARMIES: Final[dict[int, int]] = {
    3: 35,
    4: 30,
    5: 25,
    6: 20,
}


# --- Reinforcement -------------------------------------------------------

# At the start of a turn, a player gets max(MIN_REINFORCEMENT,
# owned_territories // TERRITORIES_PER_REINFORCEMENT) plus continent bonuses.
MIN_REINFORCEMENT: Final[int] = 3
TERRITORIES_PER_REINFORCEMENT: Final[int] = 3


# --- Dice / combat ---------------------------------------------------------

MAX_ATTACK_DICE: Final[int] = 3
MAX_DEFEND_DICE: Final[int] = 2
DIE_SIDES: Final[int] = 6

# Exact one-roll attacker win probabilities for Risk dice. Keys are
# (attacker_dice, defender_dice); values are the probability that the
# attacker inflicts more losses than the defender on that roll.
ATTACKER_ROLL_EDGE: Final[dict[tuple[int, int], float]] = {
    (1, 1): 15 / 36,
    (1, 2): 55 / 216,
    (2, 1): 125 / 216,
    (2, 2): 295 / 1296,
    (3, 1): 855 / 1296,
    (3, 2): 2890 / 7776,
}

# Full outcome distribution for a single roll: (attacker_dice, defender_dice)
# -> tuple of (attacker_losses, defender_losses, probability).
ROLL_OUTCOMES: Final[dict[tuple[int, int], tuple[tuple[int, int, float], ...]]] = {
    (1, 1): ((1, 0, 21 / 36), (0, 1, 15 / 36)),
    (1, 2): ((1, 0, 161 / 216), (0, 1, 55 / 216)),
    (2, 1): ((1, 0, 91 / 216), (0, 1, 125 / 216)),
    (2, 2): ((2, 0, 581 / 1296), (1, 1, 420 / 1296), (0, 2, 295 / 1296)),
    (3, 1): ((1, 0, 441 / 1296), (0, 1, 855 / 1296)),
    (3, 2): ((2, 0, 2275 / 7776), (1, 1, 2611 / 7776), (0, 2, 2890 / 7776)),
}


# --- Cards -----------------------------------------------------------------

# Symbols on the territory cards plus the special wild card.
CARD_SYMBOLS: Final[tuple[str, ...]] = ("infantry", "cavalry", "artillery")
WILD_SYMBOL: Final[str] = "wild"

# Maximum cards a player may hold before being forced to trade in.
MAX_CARDS_IN_HAND: Final[int] = 5

# Highest a hand can momentarily reach mid-attack, before the forced
# trade-in (Environment._apply_attack) brings it back below
# MAX_CARDS_IN_HAND. Only an eliminated defender's transferred cards force
# an immediate mid-attack trade-in — an ordinary conquest-card draw instead
# defers to the attacker's next TRADE_IN phase, so by the time an
# elimination happens either side may already be sitting at the full
# MAX_CARDS_IN_HAND: the attacker's own hand (<= MAX_CARDS_IN_HAND, holding
# a deferred conquest card from earlier this turn or this same conquest),
# plus the eliminated defender's own hand (<= MAX_CARDS_IN_HAND, deferred
# the same way since their own last turn). Used to size TradeInHead's
# card-slot embedding table so a real hand-slot index is never out of range.
MAX_TRANSIENT_HAND_SIZE: Final[int] = 2 * MAX_CARDS_IN_HAND

# Classic Risk card-set trade-in progression. After the 6th trade-in
# (value 15) each further trade-in is worth the previous value + 5,
# capped so very long games do not spiral into huge reinforcement swings.
CARD_SET_VALUES: Final[tuple[int, ...]] = (4, 6, 8, 10, 12, 15)
CARD_SET_INCREMENT_AFTER_FIXED: Final[int] = 5
CARD_SET_MAX_VALUE: Final[int] = 50

# Extra armies placed immediately on a territory when a traded-in card
# depicts a territory the trading player currently occupies.
CARD_TERRITORY_BONUS_ARMIES: Final[int] = 2


def card_set_value(trade_in_index: int) -> int:
    """Reinforcement bonus awarded for the n-th card set traded in (0-based).

    The first six values come from :data:`CARD_SET_VALUES`. From the
    seventh trade-in (index 6) onward each set is worth 5 more than the
    previous, capped at :data:`CARD_SET_MAX_VALUE`.
    """
    if trade_in_index < 0:
        raise ValueError(f"trade_in_index must be >= 0, got {trade_in_index}")
    if trade_in_index < len(CARD_SET_VALUES):
        return CARD_SET_VALUES[trade_in_index]
    extra = trade_in_index - (len(CARD_SET_VALUES) - 1)
    value = CARD_SET_VALUES[-1] + extra * CARD_SET_INCREMENT_AFTER_FIXED
    return min(value, CARD_SET_MAX_VALUE)


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
    "ATTACKER_ROLL_EDGE",
    "ROLL_OUTCOMES",
    "CARD_SYMBOLS",
    "WILD_SYMBOL",
    "MAX_CARDS_IN_HAND",
    "MAX_TRANSIENT_HAND_SIZE",
    "CARD_SET_VALUES",
    "CARD_SET_INCREMENT_AFTER_FIXED",
    "CARD_SET_MAX_VALUE",
    "CARD_TERRITORY_BONUS_ARMIES",
    "card_set_value",
    "starting_armies_for",
]
