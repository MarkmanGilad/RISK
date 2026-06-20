# Action graph builder

Reference for [`risk/learning/action_graph_builder.py`](../risk/learning/action_graph_builder.py),
which implements the "Action injection" piece of
[NetworkArchitectures.md](NetworkArchitectures.md) — used by Net A
(DQN+inject) and Net C (PPO+inject) only; Nets B/D (lookup) don't need it.

Given the base `Data` graph for a state ([GraphAdapter](GraphAdapter.md))
and one legal `Action` ([Action.md](Action.md)), `ActionGraphBuilder`
returns a **modified copy** carrying that action's perturbation:

```python
from risk.learning.graph_adapter import GraphAdapter
from risk.learning.action_graph_builder import ActionGraphBuilder

adapter = GraphAdapter(env.topology, ctx.settings)
builder = ActionGraphBuilder(env.topology)

base = adapter(env.current_state())
legal_actions = env.legal_actions()
graphs = [builder(base, action, env.current_state()) for action in legal_actions]
# Batch.from_data_list(graphs) -> one Encoder call for all N legal actions
```

The base graph is never mutated — every action needs its own copy to
diverge from. `state` is only required for `OccupyAction` (its `from`/`to`
live on `state.pending_attack`, not the action itself — same reason
`Action.dqn_index()` needs it).

## How each stage is injected

| Stage | What changes |
|---|---|
| `ATTACK` (`AttackAction`) | `edge_attr[row_of(from -> to)] = [1, dice / MAX_ATTACK_DICE]`. Every other edge, including the reverse direction, stays `[0, 0]`. `x` untouched. |
| `ATTACK` (`StopAttackAction`) | No perturbation at all — the result *is* the base graph (plus the same zero `edge_attr` every other action gets, so it still batches with them). |
| `REINFORCE_PLACE` | `x[t1, armies_col] += n` — written directly into the existing army-count column. |
| `OCCUPY` | `x[from, armies_col] -= n`, `x[to, armies_col] += n` (`from`/`to` from `state.pending_attack`). |
| `FORTIFY` (real move) | `x[from, armies_col] -= n`, `x[to, armies_col] += n`. |
| `FORTIFY` (skip, `count == 0`) | No perturbation — same as `StopAttackAction`. |
| `TRADE_IN` | **Raises `ValueError`.** Its head reads card-hand embeddings, never the graph (`Action.md`), so silently returning an unperturbed graph would be misleading — callers should route `TRADE_IN` actions around this class entirely. |

`edge_attr` is added to **every** action's graph, even ones that don't
touch an edge (`REINFORCE_PLACE`/`OCCUPY`/`FORTIFY`, and the unmodified
sentinels) — PyG's `Batch.from_data_list` concatenates `x`/`edge_attr`
across graphs, so every graph in a batch needs the same fields/widths
regardless of which stage it came from. This is the same "zero `edge_attr`
degrades gracefully" reasoning `NetworkArchitectures.md` already applies to
the shared `Encoder`.

`(from, to) -> edge row` is precomputed once in `__init__` from
`topology.edge_index()` (`BoardTopology` sorts territories once at load
time, so the mapping is stable for the whole game) — not searched per
action.

## Design decision: army column, not a parallel `proposed_delta` column

`NetworkArchitectures.md` left this open: write the perturbation into the
existing army-count column directly, or add a parallel `proposed_delta`
column so the signal stays legible to attention instead of pre-merged into
the real count. Implemented the simpler option (direct write) first — no
change to `x`'s width, nothing for `GraphAdapter`/`Encoder` to know about.
Revisit with a parallel column if training shows the network can't
disentangle "proposed" from "actual" army counts.

## Verified

Exercised against real `Environment` rollouts (`SelfPlay`-style loop,
`RandomAgent` roster) across `REINFORCE_PLACE`, `ATTACK` (including
`StopAttackAction`), `OCCUPY`, and `FORTIFY` (including skip):

- Every produced `Data` has `x` the same shape as the base graph and
  `edge_attr` shaped `[num_edges, 2]`.
- `StopAttackAction`/skip-`FortifyAction` results are bit-identical to
  the base graph's `x`, with all-zero `edge_attr`.
- `AttackAction`s mark exactly the attacked edge's row with
  `[1, dice/MAX_ATTACK_DICE]`, leave `x` untouched.
- `OccupyAction`/`ReinforcementAction`/`FortifyAction` shift exactly the
  affected territory row(s)' army column by the expected amount.
- `TradeInAction` raises `ValueError` rather than returning a misleading
  unperturbed graph.
- A full `ATTACK`-phase legal-action set (68 actions) built and
  `Batch.from_data_list`-ed together without shape errors:
  `DataBatch(x=[2856, 13], edge_index=[2, 11288], edge_attr=[11288, 2], u=[68, 33], ...)`.
