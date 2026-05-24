# BoardTopology

The static, read-only graph model of the Risk board. It answers the
question *"what does the board look like?"* — which territories exist,
which are neighbors, and which continent each belongs to. It knows
**nothing** about ownership, armies, turns, or phases: that all lives in
`State`.

- File: [risk/game/board_topology.py](../risk/game/board_topology.py)
- Data: [Assets/RiskMap/risk_map_data.json](../Assets/RiskMap/risk_map_data.json)
- Tests: [Temp/tests/test_board_topology.py](../Temp/tests/test_board_topology.py)

## Why It Exists

- The board never changes during a game, so it is loaded once and shared.
- Keeping it separate from `State` keeps `State` small and cheap to copy
  or serialize — important for RL rollouts and for sending observations
  to a GNN.
- It is the single source of truth for **legal-move adjacency**, so the
  environment never has to re-derive neighbors from polygons or strings.

## Data Source

`BoardTopology` loads from `Assets/RiskMap/risk_map_data.json`. Three
keys are required:

- `territory_names` — the 42 canonical territory ids (keys) and their
  display names (values).
- `continents` — for each continent: the list of member territories and
  the reinforcement `bonus`.
- `adjacency` — for each territory, the list of neighboring territory
  ids. Includes all classic Risk sea routes (Alaska↔Kamchatka,
  Greenland↔Iceland, Brazil↔NorthAfrica, etc.).

The adjacency block was authored once by the helper script
[Temp/scripts/add_adjacency.py](../Temp/scripts/add_adjacency.py) and lives in the
JSON alongside the existing renderer data.

## Stable Territory Order

Territory ids are sorted alphabetically and assigned integer indices
`0 .. 41`. This order is **deterministic across runs** and is the order
used by:

- `State`'s per-territory arrays (ownership, army counts) — Phase 3.
- The `(src, dst)` parallel index tuples returned by `edge_index()` —
  ready for a future GNN `edge_index` tensor.

## Public API

```python
from risk.game.board_topology import BoardTopology

topo = BoardTopology()                    # loads default JSON
topo = BoardTopology.from_file(path)      # custom path
topo = BoardTopology(data_dict)           # already-loaded dict

len(topo)                                 # 42
topo.territories                          # ("Afghanistan", "Alaska", ...)
topo.index_of("Alaska")                   # int
topo.territory_at(0)                      # str

topo.neighbors("Alaska")                  # ("Alberta", "Kamchatka", "NorthWestTerritory")
topo.are_adjacent("Alaska", "Kamchatka")  # True

topo.continents                           # ("Africa", "Asia", ...)
topo.continent_of("Egypt")                # "Africa"
topo.territories_in("Australia")          # ("EasternAustralia", "Indonesia", ...)
topo.continent_bonus("Asia")              # 7

src, dst = topo.edge_index()              # two parallel tuples of node indices
```

All return values are immutable (`tuple` / primitive); the instance has
no mutating methods.

## Validation On Construction

The constructor refuses to build an inconsistent board. It raises
`ValueError` if any of these fail:

- adjacency keys don't match the territory set exactly,
- a neighbor list references an unknown territory,
- a territory lists itself as a neighbor,
- a neighbor appears twice in the same list,
- adjacency is not symmetric (`A -> B` without `B -> A`),
- a continent references an unknown territory,
- a territory belongs to more than one continent or to none,
- a continent is missing its territory list or integer bonus.

This means downstream code (legal-action generation, reinforcement
bonus, GNN adapter) can trust the topology without re-validating it.

## Board Facts

| Continent     | Territories | Bonus |
|---------------|-------------|-------|
| North America | 9           | 5     |
| South America | 4           | 2     |
| Europe        | 7           | 5     |
| Africa        | 6           | 3     |
| Asia          | 12          | 7     |
| Australia     | 4           | 2     |
| **Total**     | **42**      |       |

Total directed edges in `edge_index()`: equal to the sum of all
territory degrees (each undirected border contributes two directed
edges, which is the shape a GNN expects).

## Future GNN Hook

`edge_index()` is the seam for the future graph adapter
(`risk/learning/graph_adapter.py`, planned in the v1 plan as a
post-game-engine step). The adapter will combine:

- `BoardTopology.edge_index()` → graph structure,
- `State` per-territory data (owner one-hot, army count, etc.) → node
  features,

and produce framework-specific tensors (PyG / DGL). No PyTorch imports
live in `BoardTopology` itself.

## What It Is *Not*

- Not a renderer. Polygons and visual data are still owned by
  `RiskMapRenderer` in `risk/graphics/`.
- Not stateful. It never changes after construction.
- Not a rules engine. It answers "are A and B adjacent?" but never
  "can player P attack from A to B?" — that belongs to `Environment`.
