"""Tests for `risk.game.constants` and `risk.game.phase` (Phase 3.1)."""
from __future__ import annotations

import sys

import pytest

from risk.game import constants as C
from risk.game.phase import Phase


# --- Player limits ---------------------------------------------------------


def test_player_limits_are_3_to_6() -> None:
    assert C.MIN_PLAYERS == 3
    assert C.MAX_PLAYERS == 6


def test_starting_armies_classic_table() -> None:
    assert C.STARTING_ARMIES == {3: 35, 4: 30, 5: 25, 6: 20}
    for n in range(C.MIN_PLAYERS, C.MAX_PLAYERS + 1):
        assert C.starting_armies_for(n) == C.STARTING_ARMIES[n]


@pytest.mark.parametrize("bad", [0, 1, 2, 7, 10, -1])
def test_starting_armies_for_rejects_invalid_counts(bad: int) -> None:
    with pytest.raises(ValueError):
        C.starting_armies_for(bad)


# --- Reinforcement / dice --------------------------------------------------


def test_reinforcement_rules() -> None:
    assert C.MIN_REINFORCEMENT == 3
    assert C.TERRITORIES_PER_REINFORCEMENT == 3


def test_dice_rules() -> None:
    assert C.MAX_ATTACK_DICE == 3
    assert C.MAX_DEFEND_DICE == 2
    assert C.DIE_SIDES == 6


# --- Phase enum ------------------------------------------------------------


def test_phase_has_five_members() -> None:
    assert {p.name for p in Phase} == {
        "SETUP",
        "REINFORCE",
        "ATTACK",
        "FORTIFY",
        "GAME_OVER",
        "OCCUPY",
    }


def test_phase_values_are_stable_for_serialization() -> None:
    # IntEnum with explicit values -> round-trip through int() and back.
    for p in Phase:
        assert isinstance(p.value, int)
        assert Phase(p.value) is p


def test_phase_is_ordered() -> None:
    # The natural play-order. GAME_OVER is the terminal sink.
    assert Phase.SETUP < Phase.REINFORCE < Phase.ATTACK < Phase.FORTIFY < Phase.GAME_OVER


# --- Cards -----------------------------------------------------------------


def test_card_symbols_are_three_plus_wild() -> None:
    assert C.CARD_SYMBOLS == ("infantry", "cavalry", "artillery")
    assert C.WILD_SYMBOL == "wild"
    assert C.WILD_SYMBOL not in C.CARD_SYMBOLS


def test_max_cards_in_hand_is_five() -> None:
    assert C.MAX_CARDS_IN_HAND == 5


def test_card_set_progression_is_monotonic() -> None:
    assert C.CARD_SET_VALUES == (4, 6, 8, 10, 12, 15)
    for a, b in zip(C.CARD_SET_VALUES, C.CARD_SET_VALUES[1:]):
        assert b > a


def test_card_set_value_fixed_range() -> None:
    for i, expected in enumerate(C.CARD_SET_VALUES):
        assert C.card_set_value(i) == expected


def test_card_set_value_plus_5_after_15() -> None:
    # 6th trade-in (index 6) is 15 + 5 = 20, then 25, 30, ...
    assert C.card_set_value(6) == 20
    assert C.card_set_value(7) == 25
    assert C.card_set_value(10) == 40


def test_card_set_value_rejects_negative() -> None:
    with pytest.raises(ValueError):
        C.card_set_value(-1)


# --- Module hygiene --------------------------------------------------------


def test_no_pygame_import_after_loading_constants_and_phase() -> None:
    # Importing these modules must not drag pygame into the interpreter.
    # If pygame was already imported by something else, skip the assertion.
    if "pygame" in sys.modules:
        pytest.skip("pygame already imported by an earlier test or plugin")
    import importlib

    importlib.import_module("risk.game.constants")
    importlib.import_module("risk.game.phase")
    assert "pygame" not in sys.modules
