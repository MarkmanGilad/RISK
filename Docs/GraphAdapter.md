# Graph adapter

Reference for [`risk/learning/graph_adapter.py`](../risk/learning/graph_adapter.py),
which converts a game snapshot into a `torch_geometric.data.Data` object for
the GCN+DQN trainer.

`topology`/`settings` don't change during a match, so the normal usage is
to build one adapter per game and call it every step with just the state:

```python
from risk.learning.graph_adapter import GraphAdapter

adapter = GraphAdapter(env.topology, ctx.settings)
data = adapter(env.current_state())
# Data(x=[42, 13], edge_index=[2, 166], u=[1, 34])
```

It lives in `risk/learning/`, not `risk/game/`, so the core rules engine
(`risk/game/`) stays free of a torch/PyG import — `State.to_features()`'s
docstring reserves exactly this seam. `GraphAdapter` is the real
implementation that stub was waiting for.

---

## Why a graph at all

`BoardTopology` is already a graph (42 territories, adjacency = borders);
`State` is the dynamic data sitting on top of it (`owners[i]`, `armies[i]`,
...). A GCN needs that combination expressed as PyTorch tensors —
node features, edge connectivity, and whatever isn't naturally per-node
("global" state like whose turn it is). `GraphAdapter` does that mapping
once per call, deterministically, so the same `State` always produces the
same tensors (no shuffling, no hidden randomness).

Node order is always `topology.territories`' sorted order
(`topology.index_of(t)` is node `t`'s row in every tensor below) — stable
across calls and across games, since `BoardTopology` sorts territories once
at load time.

---

## `x` — node features, shape `[42, 13]`

One row per territory:

| Slice | Width | Meaning |
|---|---|---|
| `x[:, 0:6]` | 6 | Continent one-hot (`topology.continents`, sorted: Africa, Asia, Australia, Europe, NorthAmerica, SouthAmerica) |
| `x[:, 6:12]` | 6 | Owner one-hot, **padded to `MAX_PLAYERS` (6)** regardless of how many players are actually in the game |
| `x[:, 12]` | 1 | Army count on that territory, raw integer (no normalization) |

**Why pad the owner one-hot to 6 instead of `n_players`?** So the same GCN
architecture works unmodified for a 3-player game and a 6-player game — the
input width never changes. Unused player slots (e.g. slots 4 and 5 in a
4-player game) are always zero.

**Why raw army counts, not log-scaled?** Simplicity — if training stability
becomes an issue later, normalize inside the network (e.g. a
`BatchNorm`/`LayerNorm` right after the input layer) rather than baking a
transform into the adapter.

## `edge_index` — borders, shape `[2, 166]`

Taken directly from `topology.edge_index()` with no transformation — that
method already returns `(src, dst)` parallel tuples covering both
directions of every adjacent pair (the board JSON's adjacency is validated
symmetric at load time), which is exactly the format PyG expects.

There's no `edge_attr`. An edge's mere presence in `edge_index` already
means "these two territories border each other" — a `0/1 has-border` flag
on top would be redundant (every edge present means `1`; absence means
`0`). If multiple edge *types* are ever needed (e.g. flagging the handful of
sea routes like Alaska↔Kamchatka differently from land borders), that's a
one-line addition: an `edge_attr: [num_edges, 1]` tensor.

## `u` — global attributes, shape `[1, 34]`

Not a built-in PyG concept — just a plain tensor attribute, named `u` after
the usual graph-network convention (Battaglia et al.). Shaping it `[1, F]`
per graph means `Batch.from_data_list([...])` concatenates it correctly
into `[batch_size, F]` for free (verified — see below).

| Slice | Width | Meaning | Source |
|---|---|---|---|
| `u[0]` | 1 | Number of players in this game | `settings.player_count` |
| `u[1]` | 1 | Value of the **next** card trade-in (not the last one cashed) | `card_set_value(state.cards_traded_in_count)` |
| `u[2:8]` | 6 | Cards currently in each player's hand, padded | `len(state.hands[p])` |
| `u[8:15]` | 7 | Current phase, one-hot | `state.phase` against `Phase` (`SETUP, TRADE_IN, REINFORCE_PLACE, ATTACK, OCCUPY, FORTIFY, GAME_OVER`) |
| `u[15:21]` | 6 | Whose turn it is, one-hot, padded | `state.current_player_index` |
| `u[21:27]` | 6 | Reinforcement bonus for each continent | `topology.continent_bonus(c)`, same order as the node continent one-hot |
| `u[27]` | 1 | Current player's reinforcement budget | `state.reinforcement_budget` (already `0` outside `TRADE_IN`/`REINFORCE_PLACE` — it's being built up by trades during `TRADE_IN` and spent during `REINFORCE_PLACE`) |
| `u[28:34]` | 6 | Which players are eliminated, padded | `p in state.eliminated` |

The last two (`reinforcement_budget`, `eliminated`) weren't in the original
ask but were added deliberately: budget directly bounds the legal action
space during `TRADE_IN`/`REINFORCE_PLACE`, and an explicit elimination flag
lets the model recognize "this player is permanently out" instead of
inferring it from zero territories (which looks identical to "just got
wiped out this turn").

---

## Design decisions made along the way

- **Padding over variable width** — every player-indexed feature
  (owner one-hot, cards-per-player, current-player one-hot, eliminated) is
  padded to `MAX_PLAYERS = 6`, not sized to `n_players`. One network
  architecture for every game size, at the cost of a few always-zero
  columns in smaller games.
- **No normalization in the adapter** — army counts and continent bonuses
  are passed through raw. Keeps this method a pure, debuggable mapping;
  any scaling belongs in the model.
- **Reused `topology.edge_index()` instead of rebuilding adjacency** — it
  already existed for exactly this purpose (`BoardTopology`'s docstring
  calls it out as "ready for GNN use").
- **No `edge_attr`** — would have encoded information edge *presence*
  already carries.

---

## Verified

```python
from risk.app.factory import GameFactory
from risk.app.setup import SetupStage
from risk.learning.graph_adapter import GraphAdapter

ctx = GameFactory.build(SetupStage.default_settings(n=4, seed=0))
adapter = GraphAdapter(ctx.env.topology, ctx.settings)
data = adapter(ctx.env.current_state())
data.validate(raise_on_error=True)   # passes: well-formed Data object
```

- `Data(x=[42, 13], edge_index=[2, 166], u=[1, 34])` — shapes match spec
  (`u`'s width grew 33→34 when `Phase.REINFORCE` split into `Phase.TRADE_IN`
  + `Phase.REINFORCE_PLACE` — `phase_onehot`'s width is `len(Phase)`,
  already dynamic, so this needed no code change, just a wider one-hot).
- `Data.validate()` passes (edge indices in range, tensor shapes consistent).
- `torch_geometric.data.Batch.from_data_list([...])` on 3 separate game
  snapshots produced `x: [126, 13]`, `u: [3, 34]`, `edge_index: [2, 498]` —
  confirms `u`'s `[1, F]` shape batches the way a `DataLoader` will use it.
- Full test suite (`Temp/tests`): 215 passed, 1 skipped.

## Open extension points

- Per-territory **"is this a contested border"** flag (enemy-adjacent) —
  cheap to add as a 14th node feature if the model needs it explicitly
  rather than deriving it from neighbor owner one-hots itself.
- **Edge types** (sea vs. land routes) via `edge_attr`, if the model should
  treat them differently.
- A **legal-action mask** alongside the graph — not part of the state
  representation itself, but something the DQN side will need regardless
  (probably built from `Environment.legal_actions()` rather than living in
  this adapter).
