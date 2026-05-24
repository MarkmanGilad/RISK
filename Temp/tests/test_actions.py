"""Tests for `Action` types (Phase 3.4)."""
from __future__ import annotations

import json

import pytest

from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    ReinforcementAction,
    StopAttackAction,
    action_from_dict,
)
from risk.game.board_topology import BoardTopology
from risk.game.phase import Phase


@pytest.fixture(scope="module")
def topo() -> BoardTopology:
    return BoardTopology()


# --- phase attribute ------------------------------------------------------


def test_action_phase_attributes() -> None:
    assert ReinforcementAction.phase is Phase.REINFORCE
    assert AttackAction.phase is Phase.ATTACK
    assert StopAttackAction.phase is Phase.ATTACK
    assert FortifyAction.phase is Phase.FORTIFY


# --- ReinforcementAction --------------------------------------------------


def test_reinforcement_ok() -> None:
    a = ReinforcementAction(placements={"Alaska": 3, "Alberta": 2})
    assert a.total == 5


def test_reinforcement_rejects_negative() -> None:
    with pytest.raises(ValueError):
        ReinforcementAction(placements={"Alaska": -1})


def test_reinforcement_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ReinforcementAction(placements={})


def test_reinforcement_rejects_all_zero() -> None:
    with pytest.raises(ValueError):
        ReinforcementAction(placements={"Alaska": 0, "Alberta": 0})


def test_reinforcement_rejects_non_int_count() -> None:
    with pytest.raises(ValueError):
        ReinforcementAction(placements={"Alaska": 1.5})  # type: ignore[dict-item]


def test_reinforcement_validate_budget_match(topo: BoardTopology) -> None:
    a = ReinforcementAction(placements={"Alaska": 3, "Alberta": 2})
    a.validate_against(topo, budget=5)  # ok
    with pytest.raises(ValueError):
        a.validate_against(topo, budget=4)


def test_reinforcement_validate_unknown_territory(topo: BoardTopology) -> None:
    a = ReinforcementAction(placements={"Atlantis": 1})
    with pytest.raises(ValueError):
        a.validate_against(topo)


# --- AttackAction ---------------------------------------------------------


def test_attack_ok() -> None:
    a = AttackAction(from_territory="Alaska", to_territory="Alberta", dice=3)
    assert a.dice == 3


@pytest.mark.parametrize("d", [0, -1, 4, 5, 100])
def test_attack_rejects_bad_dice(d: int) -> None:
    with pytest.raises(ValueError):
        AttackAction(from_territory="Alaska", to_territory="Alberta", dice=d)


def test_attack_rejects_same_territory() -> None:
    with pytest.raises(ValueError):
        AttackAction(from_territory="Alaska", to_territory="Alaska", dice=1)


def test_attack_validate_unknown_territory(topo: BoardTopology) -> None:
    a = AttackAction(from_territory="Atlantis", to_territory="Alberta", dice=1)
    with pytest.raises(ValueError):
        a.validate_against(topo)


# --- StopAttackAction -----------------------------------------------------


def test_stop_attack_ok() -> None:
    StopAttackAction()


# --- FortifyAction --------------------------------------------------------


def test_fortify_ok() -> None:
    f = FortifyAction(from_territory="Alaska", to_territory="Alberta", count=4)
    assert not f.is_skip


def test_fortify_skip() -> None:
    f = FortifyAction(from_territory=None, to_territory=None, count=0)
    assert f.is_skip


def test_fortify_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        FortifyAction(from_territory="A", to_territory="B", count=-1)


def test_fortify_requires_territories_when_count_positive() -> None:
    with pytest.raises(ValueError):
        FortifyAction(from_territory=None, to_territory="Alberta", count=2)
    with pytest.raises(ValueError):
        FortifyAction(from_territory="Alaska", to_territory=None, count=2)


def test_fortify_rejects_same_territory() -> None:
    with pytest.raises(ValueError):
        FortifyAction(from_territory="Alaska", to_territory="Alaska", count=2)


def test_fortify_validate_unknown_territory(topo: BoardTopology) -> None:
    f = FortifyAction(from_territory="Atlantis", to_territory="Alberta", count=2)
    with pytest.raises(ValueError):
        f.validate_against(topo)


# --- round-trip -----------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        ReinforcementAction(placements={"Alaska": 3, "Alberta": 2}),
        AttackAction(from_territory="Alaska", to_territory="Alberta", dice=2),
        StopAttackAction(),
        FortifyAction(from_territory="Alaska", to_territory="Alberta", count=3),
        FortifyAction(from_territory=None, to_territory=None, count=0),
    ],
)
def test_action_round_trip(action: Action) -> None:
    text = json.dumps(action.to_dict())
    rebuilt = action_from_dict(json.loads(text))
    assert type(rebuilt) is type(action)
    assert rebuilt.to_dict() == action.to_dict()


def test_action_from_dict_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        action_from_dict({"type": "warp_drive"})
