"""Tests for `Player`, `Card`, `GameSettings` (Phase 3.2)."""
from __future__ import annotations

import pytest

from risk.game.board_topology import BoardTopology
from risk.game.card import (
    Card,
    find_valid_set,
    is_valid_set,
    validate_against_topology,
)
from risk.game.player import Player
from risk.game.settings import GameSettings


# --- Player ----------------------------------------------------------------


def _player(pid: int = 0, **overrides) -> Player:
    kwargs = dict(id=pid, name=f"P{pid}", color=(10, 20, 30), agent_kind="human")
    kwargs.update(overrides)
    return Player(**kwargs)


def test_player_construction_ok() -> None:
    p = _player()
    assert p.id == 0
    assert p.name == "P0"
    assert p.color == (10, 20, 30)
    assert p.agent_kind == "human"


def test_player_is_frozen() -> None:
    p = _player()
    with pytest.raises(Exception):
        p.id = 1  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", "ROBOT", "HUMAN", "scripted", "random"])
def test_player_rejects_unknown_agent_kind(bad: str) -> None:
    with pytest.raises(ValueError):
        _player(agent_kind=bad)


def test_player_rejects_negative_id() -> None:
    with pytest.raises(ValueError):
        _player(pid=-1)


def test_player_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        _player(name="")


@pytest.mark.parametrize(
    "bad_color",
    [(10, 20), (10, 20, 30, 40), (-1, 0, 0), (0, 256, 0), (1.0, 2, 3), "red"],
)
def test_player_rejects_bad_color(bad_color) -> None:
    with pytest.raises(ValueError):
        _player(color=bad_color)


# --- Card ------------------------------------------------------------------


def test_card_construction_ok() -> None:
    c = Card(territory_id="Alaska", symbol="infantry")
    assert c.symbol == "infantry"
    assert not c.is_wild


def test_wild_card_construction_ok() -> None:
    c = Card(territory_id=None, symbol="wild")
    assert c.is_wild


def test_wild_card_rejects_territory_id() -> None:
    with pytest.raises(ValueError):
        Card(territory_id="Alaska", symbol="wild")


def test_non_wild_card_requires_territory_id() -> None:
    with pytest.raises(ValueError):
        Card(territory_id=None, symbol="infantry")


def test_card_rejects_unknown_symbol() -> None:
    with pytest.raises(ValueError):
        Card(territory_id="Alaska", symbol="tank")


def test_card_validate_against_topology() -> None:
    topo = BoardTopology()
    ok = Card(territory_id="Alaska", symbol="infantry")
    bad = Card(territory_id="Atlantis", symbol="infantry")
    wild = Card(territory_id=None, symbol="wild")
    validate_against_topology(ok, topo)        # no raise
    validate_against_topology(wild, topo)      # no raise
    with pytest.raises(ValueError):
        validate_against_topology(bad, topo)


# --- Card trade-in sets ----------------------------------------------------


def _c(sym: str, terr: str | None = None) -> Card:
    if sym == "wild":
        return Card(territory_id=None, symbol="wild")
    return Card(territory_id=terr or f"T_{sym}", symbol=sym)


def test_three_of_a_kind_is_valid() -> None:
    assert is_valid_set([_c("infantry", "A"), _c("infantry", "B"), _c("infantry", "C")])
    assert is_valid_set([_c("cavalry", "A"), _c("cavalry", "B"), _c("cavalry", "C")])


def test_one_of_each_is_valid() -> None:
    assert is_valid_set([_c("infantry", "A"), _c("cavalry", "B"), _c("artillery", "C")])


def test_wilds_substitute() -> None:
    # Two infantry + 1 wild -> three of a kind
    assert is_valid_set([_c("infantry", "A"), _c("infantry", "B"), _c("wild")])
    # 1 infantry + 1 cavalry + 1 wild -> one of each (wild as artillery)
    assert is_valid_set([_c("infantry", "A"), _c("cavalry", "B"), _c("wild")])
    # 1 regular + 2 wilds -> always valid (wilds can be anything)
    assert is_valid_set([_c("artillery", "A"), _c("wild"), _c("wild")])
    # 3 wilds -> valid
    assert is_valid_set([_c("wild"), _c("wild"), _c("wild")])


def test_invalid_sets_rejected() -> None:
    # Two of one + one of another with no wild -> invalid
    assert not is_valid_set(
        [_c("infantry", "A"), _c("infantry", "B"), _c("cavalry", "C")]
    )
    # Wrong card count
    assert not is_valid_set([_c("infantry", "A"), _c("infantry", "B")])
    assert not is_valid_set(
        [_c("infantry", "A"), _c("infantry", "B"), _c("infantry", "C"), _c("wild")]
    )


def test_find_valid_set_picks_a_subset() -> None:
    hand = [
        _c("infantry", "A"),
        _c("cavalry", "B"),
        _c("infantry", "C"),
        _c("artillery", "D"),
    ]
    trio = find_valid_set(hand)
    assert trio is not None
    assert is_valid_set(trio)


def test_find_valid_set_returns_none_when_no_set() -> None:
    hand = [_c("infantry", "A"), _c("infantry", "B"), _c("cavalry", "C")]
    assert find_valid_set(hand) is None


# --- GameSettings ----------------------------------------------------------


def _roster(n: int) -> tuple[Player, ...]:
    return tuple(
        Player(id=i, name=f"P{i}", color=(10 * i, 20, 30), agent_kind="human")
        for i in range(n)
    )


def test_game_settings_construction_ok() -> None:
    s = GameSettings(players=_roster(3))
    assert s.player_count == 3
    assert s.seed is None
    assert s.rule_toggles == {}


def test_game_settings_accepts_seed() -> None:
    s = GameSettings(players=_roster(4), seed=42)
    assert s.seed == 42


@pytest.mark.parametrize("n", [0, 1, 2, 7, 10])
def test_game_settings_rejects_bad_player_count(n: int) -> None:
    with pytest.raises(ValueError):
        GameSettings(players=_roster(n))


def test_game_settings_rejects_duplicate_ids() -> None:
    p0 = Player(id=0, name="A", color=(1, 1, 1), agent_kind="human")
    p1 = Player(id=0, name="B", color=(2, 2, 2), agent_kind="ai")
    p2 = Player(id=1, name="C", color=(3, 3, 3), agent_kind="human")
    with pytest.raises(ValueError):
        GameSettings(players=(p0, p1, p2))


def test_game_settings_rejects_bad_seed_type() -> None:
    with pytest.raises(ValueError):
        GameSettings(players=_roster(3), seed="42")  # type: ignore[arg-type]


def test_game_settings_is_frozen() -> None:
    s = GameSettings(players=_roster(3))
    with pytest.raises(Exception):
        s.seed = 1  # type: ignore[misc]
