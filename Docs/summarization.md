# Summarization

Short summary of the graph state representation, action injection, and current graph network architecture. Detailed references: [GraphAdapter.md](GraphAdapter.md), [ActionGraphBuilder.md](ActionGraphBuilder.md), and [NetworkArchitectures.md](NetworkArchitectures.md).

## A. State graph structure

The state graph is built by `risk/learning/graph_adapter.py` as a PyTorch Geometric `Data` object. It represents the fixed Risk board topology plus the dynamic game state.

Typical shape:

```python
Data(x=[42, 13], edge_index=[2, 166], edge_attr=[166, 2], u=[1, 34])
```

### Nodes

Each node is one territory. Node order is stable and follows `BoardTopology.territories` / `topology.index_of(territory)`.

### Node attributes: `x`, shape `[42, 13]`

Each territory row contains:

| Slice | Width | Meaning |
|---|---:|---|
| `x[:, 0:6]` | 6 | Continent one-hot: Africa, Asia, Australia, Europe, NorthAmerica, SouthAmerica |
| `x[:, 6:12]` | 6 | Owner one-hot, padded to `MAX_PLAYERS = 6` |
| `x[:, 12]` | 1 | Raw army count on the territory |

Player-indexed features can be perspective-rotated so the learning agent sees itself as relative player slot `0`.

### Edges

`edge_index`, shape `[2, 166]`, contains directed border edges. Both directions of each border are present, so an undirected border appears as `A -> B` and `B -> A`.

### Edge attributes: `edge_attr`, shape `[166, 2]`

The base state graph has all-zero edge attributes:

```python
[selected_attack, dice_count]
```

These columns are reserved for action injection. In the plain state graph, no action has been selected yet.

### Global attributes: `u`, shape `[1, 34]`

The global vector contains game-wide information that is not naturally per-territory:

| Slice | Width | Meaning |
|---|---:|---|
| `u[0]` | 1 | Number of players |
| `u[1]` | 1 | Next card trade-in value |
| `u[2:8]` | 6 | Cards per player, padded |
| `u[8:15]` | 7 | Current phase one-hot |
| `u[15:21]` | 6 | Current player one-hot, padded |
| `u[21:27]` | 6 | Continent reinforcement bonuses |
| `u[27]` | 1 | Current reinforcement budget |
| `u[28:34]` | 6 | Eliminated-player flags, padded |

Exact card identities are still part of the game `State`, but not part of the graph tensor. They live in `state.hands`, where each player has a list of `Card(territory_id, symbol)` objects. The graph only receives the number of cards each player holds through `u[2:8]`; it does not receive card symbols, territory ids, or wild-card identity.

## B. Action injection: input graph to the net

The current implemented network is Net A: DQN plus action injection. The net scores `Q(s, a)`, so each legal action is injected into a copy of the base state graph before the GNN runs.

Pipeline:

```text
State
  -> GraphAdapter
  -> base Data graph
  -> ActionGraphBuilder(base graph, legal action, state)
  -> one modified graph per legal action
  -> Batch.from_data_list(action_graphs)
  -> GNN_DQN.forward(batch, phase, card_indices)
  -> Q value per legal action
```

`ActionGraphBuilder` never mutates the base graph. It clones the graph and changes only `x` or `edge_attr` depending on the action type.

### Attack action

For `AttackAction(from=A, to=B, dice=d)`, only the directed attack edge is marked:

```python
edge_attr[row_of(A -> B)] = [1, d / MAX_ATTACK_DICE]
```

The reverse edge and all other edges stay `[0, 0]`. Node attributes are unchanged.

### Reinforce, occupy, and fortify actions

These actions are injected by changing the army-count column in `x`:

| Stage | Injection |
|---|---|
| `REINFORCE_PLACE` | `x[target, army_col] += count` |
| `OCCUPY` | `x[from, army_col] -= count`, `x[to, army_col] += count` |
| `FORTIFY` | `x[from, army_col] -= count`, `x[to, army_col] += count` |

### Skip or no-op actions

`StopAttackAction`, skip-fortify, and trade-in graph copies are unmodified base graphs. Trade-in actions are scored through card-slot embeddings in the trade-in head, not by graph perturbation.

For a `TradeInAction`, the action stores `card_indices=(i, j, k)`, which are positions in the current player's `state.hands[current_player]`. The environment has already checked that those three hand slots form a valid set. The network receives those slot indices, not the actual `Card` objects and not their symbols or territory ids.

> **Future development — inject `TRADE_IN` like the other stages.**
>
> Trade-in is the one action that perturbs nothing today, so the net never
> sees *which* set is being cashed. A candidate `TradeInAction` could be
> injected instead:
>
> - **Non-wild cards:** set a per-node flag (a new `x` column) on each
>   card's territory. This grounds the +2 territory bonus
>   ([Action.md](Action.md)) in board space, next to the node's existing
>   owner one-hot, so the GNN can weigh the bonus placements in
>   topological context.
> - **Wild cards:** have no territory, so they can't sit on a node. Carry
>   them as a per-action `wilds_in_set` count written into `u` at build
>   time. Note this is a *new capability* — `ActionGraphBuilder` today
>   only perturbs `x`/`edge_attr` and never `u`; varying `u` per candidate
>   action would be the new step (the `num_legal_actions` scalar in
>   [NetworkArchitectures.md](NetworkArchitectures.md) also lives in `u`
>   but is written *per-state* by `GraphAdapter`, not per-action).
>
> This complements, not replaces, the card-hand signal — node flags
> capture the immediate territory-bonus differentiator; the residual-hand
> ("which cards to keep") consideration still wants hand context. See
> [ActionGraphBuilder.md](ActionGraphBuilder.md) for the matching note.


### Extra inputs beside the graph

`GNN_DQN.forward` receives:

| Input | Shape | Meaning |
|---|---:|---|
| `state` | PyG `Batch` | One graph per action candidate |
| `phase` | `[N]` | Phase/head id for each candidate graph |
| `card_indices` | `[N, 3]` | Trade-in hand-slot indices `(i, j, k)`; ignored by non-trade heads |

## C. Net structure

The implemented network is `risk/learning/gnn_dqn.py::GNN_DQN`. It is the DQN + injection architecture: one batched forward pass over the injected action graphs returns one scalar `Q(s, a)` per legal action.

### High-level flow

```text
Batched action graphs
  -> Encoder(x, edge_index, edge_attr)
  -> node embeddings H
  -> pool(H, batch, u)
  -> graph embeddings g
  -> per-phase scoring head
  -> Q(s, a)
```

### Encoder / graph net

The shared encoder is `risk/learning/encoder.py::Encoder`:

```text
x -> Linear(in_dim, hidden_dim)
  -> TransformerConv + residual + ReLU
  -> TransformerConv + residual + ReLU
  -> TransformerConv + residual + ReLU
  -> TransformerConv + residual + ReLU
  -> H [num_nodes, hidden_dim]
```

Default values used by the current DQN setup are:

| Parameter | Meaning |
|---|---|
| `in_dim = 13` | Node feature width |
| `hidden_dim = 64` | Node embedding width in typical usage |
| `edge_dim = 2` | Edge attribute width |
| `n_layers = 4` | Number of `TransformerConv` layers |

The encoder sees the injected action because `edge_attr` or `x` has already been modified before batching.

### Pooling

`risk/learning/pooling.py::pool` converts node embeddings into one graph vector:

```python
g = concat(global_mean_pool(H), global_max_pool(H), u)
```

So the pooled graph dimension is:

```text
g_dim = 2 * hidden_dim + u_dim
```

With `hidden_dim = 64` and `u_dim = 34`, this gives `g_dim = 162`.

### Heads

`GNN_DQN` has five named heads, one per decision phase used by the DQN action representation:

| Phase | Head attribute | Head class | Input |
|---|---|---|---|
| `TRADE_IN` | `trade_in_head` | `TradeInHead` | pooled graph `g` + 3 card-slot embeddings |
| `REINFORCE_PLACE` | `reinforce_place_head` | `ScoringHead` | pooled injected graph `g` |
| `ATTACK` | `attack_head` | `ScoringHead` | pooled injected graph `g` |
| `OCCUPY` | `occupy_head` | `ScoringHead` | pooled injected graph `g` |
| `FORTIFY` | `fortify_head` | `ScoringHead` | pooled injected graph `g` |

`ScoringHead` architecture:

```text
g
  -> Linear(g_dim, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Linear(128, 1)
  -> scalar Q
```

`TradeInHead` uses the same MLP tail, but its input is larger:

```text
concat(g, embedding(card_1), embedding(card_2), embedding(card_3))
  -> Linear(g_dim + 3 * card_embed_dim, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Linear(128, 1)
  -> scalar Q
```

The trade-in embedding table has `MAX_TRANSIENT_HAND_SIZE + 1` rows. The extra row is a learned sentinel for `SkipTradeAction`.

Important limitation: these embeddings represent hand positions, not card contents. So two different legal trade-in sets using the same slot pattern would look the same to the head if the surrounding state graph and globals are the same. The exact card contents remain available to the rules engine through `state.hands`, and legality is enforced before the action reaches the network.

### Head routing

`GNN_DQN.forward` encodes and pools the whole batch once, then routes each row to the correct head by its `phase` value. The result is a tensor of shape `[N]`, where `N` is the number of action candidates in the batch.

The agent then maps these Q-values back to `Environment.legal_actions()` order and chooses the action with the highest score.
