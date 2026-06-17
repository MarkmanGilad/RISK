"""Static board model for Risk.

`BoardTopology` is the read-only graph view of the board: which territories
exist, which are neighbors, and which continent each belongs to. It does
not know anything about ownership, armies, or turn state — those live in
`State`.

The class is built from the board data file (default:
``Assets/RiskMap/risk_map_data.json``). Territory ids are the JSON keys
(e.g. ``"Alaska"``), and a stable integer index is assigned by sorted-name
order so node-aligned arrays (used later by `State` and by a future GNN
adapter) line up deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "Assets"
    / "RiskMap"
    / "risk_map_data.json"
)


class BoardTopology:
    """Read-only static board model.

    Public API:
        territories        - tuple of territory ids in deterministic order
        index_of(t)        - integer index of a territory id
        territory_at(i)    - territory id at a given index
        neighbors(t)       - tuple of neighboring territory ids
        are_adjacent(a, b) - bool
        continents         - tuple of continent ids
        continent_of(t)    - continent id that owns territory t
        territories_in(c)  - tuple of territory ids in continent c
        continent_bonus(c) - integer reinforcement bonus
        continent_member_indices(c)        - integer indices of c's territories
        continent_owner_counts(owners,c,p) - (owned_by_p, total) for c
        owns_continent(owners, c, p)       - True if p owns every territory in c
        edge_index()       - (src, dst) parallel index tuples for GNN use
    """

    def __init__(self, data: Mapping[str, object] | None = None) -> None:
        if data is None:
            data = json.loads(_DEFAULT_DATA_PATH.read_text(encoding="utf-8"))

        territory_names = data.get("territory_names")
        adjacency = data.get("adjacency")
        continents = data.get("continents")
        if not isinstance(territory_names, Mapping):
            raise ValueError("board data missing 'territory_names' mapping")
        if not isinstance(adjacency, Mapping):
            raise ValueError("board data missing 'adjacency' mapping")
        if not isinstance(continents, Mapping):
            raise ValueError("board data missing 'continents' mapping")

        self._territories: tuple[str, ...] = tuple(sorted(territory_names.keys()))
        self._index: dict[str, int] = {t: i for i, t in enumerate(self._territories)}

        self._validate_adjacency(adjacency)
        self._neighbors: dict[str, tuple[str, ...]] = {
            t: tuple(sorted(adjacency[t])) for t in self._territories
        }

        self._continents: tuple[str, ...] = tuple(sorted(continents.keys()))
        self._continent_of: dict[str, str] = {}
        self._territories_in: dict[str, tuple[str, ...]] = {}
        self._continent_bonus: dict[str, int] = {}
        self._continent_indices: dict[str, tuple[int, ...]] = {}
        self._validate_continents(continents)
        for cid in self._continents:
            members = tuple(sorted(continents[cid]["territories"]))
            self._territories_in[cid] = members
            self._continent_bonus[cid] = int(continents[cid]["bonus"])
            for t in members:
                self._continent_of[t] = cid
        for cid in self._continents:
            self._continent_indices[cid] = tuple(
                self._index[t] for t in self._territories_in[cid]
            )

        src: list[int] = []
        dst: list[int] = []
        for t, neigh in self._neighbors.items():
            i = self._index[t]
            for n in neigh:
                src.append(i)
                dst.append(self._index[n])
        self._edge_src: tuple[int, ...] = tuple(src)
        self._edge_dst: tuple[int, ...] = tuple(dst)

    @classmethod
    def from_file(cls, path: str | Path) -> "BoardTopology":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    @property
    def territories(self) -> tuple[str, ...]:
        return self._territories

    @property
    def continents(self) -> tuple[str, ...]:
        return self._continents

    def __len__(self) -> int:
        return len(self._territories)

    def index_of(self, territory: str) -> int:
        return self._index[territory]

    def territory_at(self, index: int) -> str:
        return self._territories[index]

    def neighbors(self, territory: str) -> tuple[str, ...]:
        return self._neighbors[territory]

    def are_adjacent(self, a: str, b: str) -> bool:
        return b in self._neighbors[a]

    def continent_of(self, territory: str) -> str:
        return self._continent_of[territory]

    def territories_in(self, continent: str) -> tuple[str, ...]:
        return self._territories_in[continent]

    def continent_bonus(self, continent: str) -> int:
        return self._continent_bonus[continent]

    def continent_member_indices(self, continent: str) -> tuple[int, ...]:
        """Integer indices of every territory in `continent`."""
        return self._continent_indices[continent]

    def continent_owner_counts(
        self, owners: Sequence[object], continent: str, player_id: object
    ) -> tuple[int, int]:
        """Return `(owned_by_player, total_members)` for `continent`.

        `owners` is a state's owner array (`owners[i]` is the player id that
        owns `territory_at(i)`, or `None`).
        """
        members = self._continent_indices[continent]
        owned = sum(1 for i in members if owners[i] == player_id)
        return owned, len(members)

    def owns_continent(
        self, owners: Sequence[object], continent: str, player_id: object
    ) -> bool:
        """True if `player_id` owns every territory in `continent`."""
        owned, total = self.continent_owner_counts(owners, continent, player_id)
        return owned == total

    def edge_index(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Two parallel tuples (src, dst) of node indices, ready for GNN use."""
        return self._edge_src, self._edge_dst

    def _validate_adjacency(self, adjacency: Mapping[str, Iterable[str]]) -> None:
        keys = set(adjacency.keys())
        expected = set(self._territories)
        if keys != expected:
            raise ValueError(
                f"adjacency keys mismatch: extra={keys - expected}, "
                f"missing={expected - keys}"
            )
        for src, neighbors in adjacency.items():
            seen: set[str] = set()
            for dst in neighbors:
                if dst == src:
                    raise ValueError(f"{src} lists itself as a neighbor")
                if dst not in expected:
                    raise ValueError(f"{src} -> unknown territory {dst}")
                if dst in seen:
                    raise ValueError(f"{src} has duplicate neighbor {dst}")
                seen.add(dst)
                if src not in adjacency[dst]:
                    raise ValueError(f"adjacency not symmetric: {src} -> {dst}")

    def _validate_continents(self, continents: Mapping[str, Mapping[str, object]]) -> None:
        seen: set[str] = set()
        for cid, meta in continents.items():
            territories = meta.get("territories")
            bonus = meta.get("bonus")
            if not isinstance(territories, Sequence) or isinstance(territories, str):
                raise ValueError(f"continent {cid} missing 'territories' list")
            if not isinstance(bonus, int):
                raise ValueError(f"continent {cid} missing integer 'bonus'")
            for t in territories:
                if t not in self._index:
                    raise ValueError(f"continent {cid} references unknown territory {t}")
                if t in seen:
                    raise ValueError(f"territory {t} belongs to more than one continent")
                seen.add(t)
        missing = set(self._territories) - seen
        if missing:
            raise ValueError(f"territories not assigned to any continent: {missing}")
