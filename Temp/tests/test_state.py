"""Tests for `State` (Phase 3.3)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from risk.game.board_topology import BoardTopology
from risk.game.card import Card
from risk.game.phase import Phase
from risk.game.state import PendingAttack, State


@pytest.fixture(scope="module")
def topo() -> BoardTopology:
    return BoardTopology()


# --- construction / sizing -------------------------------------------------


def test_initial_state_aligned_with_topology(topo: BoardTopology) -> None:
    s = State.initial(topo, player_count=4)
    assert len(s.owners) == len(topo) == 42
    assert len(s.armies) == 42
    assert all(o is None for o in s.owners)
    assert all(a == 0 for a in s.armies)
    assert s.phase is Phase.SETUP
    assert s.current_player_index == 0
    assert s.reinforcement_budget == 0
    assert s.eliminated == set()
    assert s.cards_traded_in_count == 0
    assert s.pending_attack is None
    assert s.unfinished_attack_targets_this_turn == set()
    assert len(s.hands) == 4
    assert all(h == [] for h in s.hands)


def test_state_indices_align_with_topology(topo: BoardTopology) -> None:
    s = State.initial(topo, player_count=3)
    alaska = topo.index_of("Alaska")
    s.owners[alaska] = 0
    s.armies[alaska] = 5
    assert s.owners[topo.index_of("Alaska")] == 0
    assert s.armies[topo.index_of("Alaska")] == 5


# --- copy isolation --------------------------------------------------------


def _populate(topo: BoardTopology) -> State:
    s = State.initial(topo, player_count=3)
    for i in range(len(topo)):
        s.owners[i] = i % 3
        s.armies[i] = i + 1
    s.hands[0].append(Card(territory_id="Alaska", symbol="infantry"))
    s.hands[1].append(Card(territory_id=None, symbol="wild"))
    s.eliminated.add(2)
    s.current_player_index = 1
    s.phase = Phase.ATTACK
    s.reinforcement_budget = 7
    s.cards_traded_in_count = 3
    s.pending_attack = PendingAttack(
        from_index=topo.index_of("Alaska"),
        to_index=topo.index_of("NorthWestTerritory"),
        attacker_dice=3,
    )
    s.conquered_this_turn = True
    s.unfinished_attack_targets_this_turn = {2, 7}
    return s


def test_copy_is_independent(topo: BoardTopology) -> None:
    s = _populate(topo)
    c = s.copy()
    assert s == c

    # Mutate the copy in every container; original must not change.
    c.owners[0] = 99
    c.armies[0] = 999
    c.hands[0].append(Card(territory_id="Brazil", symbol="cavalry"))
    c.eliminated.add(0)
    c.current_player_index = 2
    c.phase = Phase.FORTIFY
    c.reinforcement_budget = 0
    c.cards_traded_in_count = 99
    assert c.pending_attack is not None
    c.pending_attack.attacker_dice = 1
    c.unfinished_attack_targets_this_turn.add(9)

    assert s.owners[0] != 99
    assert s.armies[0] != 999
    assert len(s.hands[0]) == 1
    assert 0 not in s.eliminated
    assert s.current_player_index == 1
    assert s.phase is Phase.ATTACK
    assert s.reinforcement_budget == 7
    assert s.cards_traded_in_count == 3
    assert s.pending_attack.attacker_dice == 3
    assert s.unfinished_attack_targets_this_turn == {2, 7}


def test_snapshot_is_alias_for_copy(topo: BoardTopology) -> None:
    s = _populate(topo)
    snap = s.snapshot()
    assert snap == s
    assert snap is not s


# --- serialization round-trip ---------------------------------------------


def test_to_dict_json_serializable(topo: BoardTopology) -> None:
    s = _populate(topo)
    d = s.to_dict()
    text = json.dumps(d)  # must not raise
    assert isinstance(text, str) and text


def test_to_dict_round_trip_initial(topo: BoardTopology) -> None:
    s = State.initial(topo, player_count=5)
    s2 = State.from_dict(s.to_dict())
    assert s2 == s


def test_to_dict_round_trip_populated(topo: BoardTopology) -> None:
    s = _populate(topo)
    text = json.dumps(s.to_dict())
    s2 = State.from_dict(json.loads(text))
    assert s2 == s
    # Phase deserialized to the enum, not just an int.
    assert isinstance(s2.phase, Phase)
    # PendingAttack survives.
    assert s2.pending_attack is not None
    assert s2.pending_attack.attacker_dice == 3


# --- learning hook ---------------------------------------------------------


def test_to_features_is_a_stub(topo: BoardTopology) -> None:
    s = State.initial(topo, player_count=3)
    with pytest.raises(NotImplementedError):
        s.to_features(topo)


# --- import hygiene --------------------------------------------------------


def test_state_module_does_not_import_pygame() -> None:
    code = (
        "import sys; "
        "import risk.game.state; "
        "assert 'pygame' not in sys.modules, sorted(k for k in sys.modules if 'pygame' in k)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
