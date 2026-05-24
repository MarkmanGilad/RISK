"""Tests for `risk.game.board_topology.BoardTopology`.

Run from the repo root:
    python -m pytest tests/test_board_topology.py
"""
from __future__ import annotations

import pytest

from risk.game.board_topology import BoardTopology


EXPECTED_TERRITORY_COUNT = 42
EXPECTED_CONTINENTS = {
    "NorthAmerica": (9, 5),
    "SouthAmerica": (4, 2),
    "Europe": (7, 5),
    "Africa": (6, 3),
    "Asia": (12, 7),
    "Australia": (4, 2),
}


@pytest.fixture(scope="module")
def topo() -> BoardTopology:
    return BoardTopology()


def test_loads_42_unique_territories(topo: BoardTopology) -> None:
    assert len(topo) == EXPECTED_TERRITORY_COUNT
    assert len(set(topo.territories)) == EXPECTED_TERRITORY_COUNT


def test_territory_order_is_deterministic(topo: BoardTopology) -> None:
    again = BoardTopology()
    assert topo.territories == again.territories


def test_index_round_trip(topo: BoardTopology) -> None:
    for i, name in enumerate(topo.territories):
        assert topo.index_of(name) == i
        assert topo.territory_at(i) == name


def test_known_adjacencies(topo: BoardTopology) -> None:
    # Classic land neighbors
    assert "Alberta" in topo.neighbors("Alaska")
    assert "WesternUnitedStates" in topo.neighbors("CentralAmerica")
    # Sea routes (the historically tricky ones)
    assert topo.are_adjacent("Alaska", "Kamchatka")
    assert topo.are_adjacent("Greenland", "Iceland")
    assert topo.are_adjacent("Brazil", "NorthAfrica")
    assert topo.are_adjacent("WesternEurope", "NorthAfrica")
    assert topo.are_adjacent("SouthernEurope", "Egypt")
    assert topo.are_adjacent("EastAfrica", "MiddleEast")
    assert topo.are_adjacent("Siam", "Indonesia")


def test_adjacency_is_symmetric(topo: BoardTopology) -> None:
    for t in topo.territories:
        for n in topo.neighbors(t):
            assert t in topo.neighbors(n), f"{t} -> {n} not symmetric"


def test_no_self_loops(topo: BoardTopology) -> None:
    for t in topo.territories:
        assert t not in topo.neighbors(t)


def test_every_territory_in_exactly_one_continent(topo: BoardTopology) -> None:
    seen: dict[str, str] = {}
    for cid in topo.continents:
        for t in topo.territories_in(cid):
            assert t not in seen, f"{t} duplicated in {seen[t]} and {cid}"
            seen[t] = cid
    assert set(seen.keys()) == set(topo.territories)


def test_continent_sizes_and_bonuses(topo: BoardTopology) -> None:
    assert set(topo.continents) == set(EXPECTED_CONTINENTS.keys())
    for cid, (size, bonus) in EXPECTED_CONTINENTS.items():
        assert len(topo.territories_in(cid)) == size
        assert topo.continent_bonus(cid) == bonus


def test_edge_index_shape_and_symmetry(topo: BoardTopology) -> None:
    src, dst = topo.edge_index()
    assert len(src) == len(dst)
    # Every directed edge has its reverse partner.
    edges = set(zip(src, dst))
    for a, b in edges:
        assert (b, a) in edges
    # Total directed edge count equals sum of degrees.
    expected = sum(len(topo.neighbors(t)) for t in topo.territories)
    assert len(src) == expected


def test_invalid_data_raises() -> None:
    bad = {
        "territory_names": {"A": "A", "B": "B"},
        "adjacency": {"A": ["B"], "B": []},  # not symmetric
        "continents": {"X": {"territories": ["A", "B"], "bonus": 1}},
    }
    with pytest.raises(ValueError):
        BoardTopology(bad)
