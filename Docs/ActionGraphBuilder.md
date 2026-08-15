# Action graph builder

## Current injection rule

Candidate reinforce, occupy, and fortify graphs preserve base `armies`
exactly. They write signed proposed movement into `proposed_army_delta`:
reinforce is `+n` at its target; occupy and fortify are `-n` at source and
`+n` at destination. Attack remains an edge injection. This keeps Dueling's
value stream state-only while exposing the proposal to its advantage stream.

Reference for [`risk/learning/action_graph_builder.py`](../risk/learning/action_graph_builder.py),
which implements the "Action injection" piece of
[NetworkArchitectures.md](NetworkArchitectures.md). Every implemented learner
uses this shared injected-action representation; the network-specific heads
decide how each candidate graph is scored.

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
`Action.dqn_index()` needs it). `TRADE_IN` is handled internally by
returning an unmodified copy of the base graph, so callers can hand every
legal action to this builder and let it decide whether injection is needed.

## How each stage is injected

| Stage | What changes |
|---|---|
| `ATTACK` (`AttackAction`) | `edge_attr[row_of(from -> to)] = [1, dice / MAX_ATTACK_DICE]`. Every other edge, including the reverse direction, stays `[0, 0]`. `x` untouched. |
| `ATTACK` (`StopAttackAction`) | No perturbation at all — the result *is* the base graph (plus the same zero `edge_attr` every other action gets, so it still batches with them). |
| `REINFORCE_PLACE` | `x[t1, proposed_army_delta_col] += n`; the real army-count column is unchanged. |
| `OCCUPY` | `x[from, proposed_army_delta_col] -= n`, `x[to, proposed_army_delta_col] += n` (`from`/`to` from `state.pending_attack`); real army counts are unchanged. |
| `FORTIFY` (real move) | `x[from, proposed_army_delta_col] -= n`, `x[to, proposed_army_delta_col] += n`; real army counts are unchanged. |
| `FORTIFY` (skip, `count == 0`) | No perturbation — same as `StopAttackAction`. |
| `TRADE_IN` | Returns an unmodified copy of the base graph. Its current head receives only the selected **hand-slot positions**, not the identities of the cards in those slots. See the deferred correction plan below. |

`edge_attr` is already present on every base graph (`GraphAdapter`,
`Docs/GraphAdapter.md`, all-zero) — this class clones and overwrites it
rather than building one from scratch, so it ends up on **every** action's
graph, even ones that don't touch an edge (`REINFORCE_PLACE`/`OCCUPY`/
`FORTIFY`, and the unmodified `TRADE_IN` copy). PyG's `Batch.from_data_list`
concatenates `x`/`edge_attr` across graphs, so every graph in a batch
needs the same fields/widths regardless of which stage it came from. This
is the same "zero `edge_attr` degrades gracefully" reasoning
`NetworkArchitectures.md` already applies to the shared `Encoder`.

`(from, to) -> edge row` is precomputed once in `__init__` from
`topology.edge_index()` (`BoardTopology` sorts territories once at load
time, so the mapping is stable for the whole game) — not searched per
action.

## Deferred plan: correct the `TRADE_IN` representation gap

**Status: recorded for the next model family; do not change the current
models or checkpoints for the poster/evaluation run.** This is a learner
observation and action-representation limitation, not an error in the game
rules.

### Current limitation

Every `TradeInAction` currently receives the same unmodified `x`, `edge_attr`,
and `u`. `GraphAdapter` does not encode individual cards, their symbols,
territory associations, or wild status. The current `TradeInHead` then embeds
only hand-slot numbers such as `(0, 1, 2)`. A slot number is not a card
identity: slot 0 means whichever card happens to be first in the hand.

#### What the trade-in head sees today

Every trade-in candidate receives the normal board graph — territory owners,
army counts, continents, borders, current phase, and current player — plus the
same state-level global attributes. The only card-related global attributes
are:

| Input | Available information |
|---|---|
| `u[1]` | The value of the next card set, for example 4 armies. It is the same for every legal trade-in candidate in this state. |
| `u[2:8]` | The number of cards held by each player. |
| `u[27]` | The reinforcement budget accumulated from any earlier trade-ins this turn. |
| `card_indices = (i, j, k)` | The selected hand positions supplied separately to `TradeInHead`, such as `(0, 1, 2)`. These are positions only, not card identities. |

It does **not** see which card occupies a slot; any card's symbol, territory,
or wild status; whether a selected card territory is owned; the candidate's
specific +2 territory bonuses; or the cards that would remain after trading.

As a result, otherwise identical board states with different cards in the
same selected slots are indistinguishable to the scorer. This hides two
important decision signals: the immediate +2 bonus for a selected owned card
territory, and the composition of the cards retained after the trade.

### Next-version implementation plan

1. Add an all-zero `trade_in_selected` node feature to the base graph. For a
   `TradeInAction`, mark the territory node for every selected non-wild card;
   leave it zero for `SkipTradeAction`. This gives the GNN the selected
   territory in its existing board context without changing the real state.

2. Replace the positional slot embedding in `TradeInHead` with a real,
   order-invariant hand encoder. Each card representation must include the
   card symbol, the corresponding territory-node embedding or a learned wild
   token, and a selected-versus-retained flag. Pool those card representations
   and concatenate them with the candidate graph embedding before scoring.

3. Keep wild cards in the card encoder rather than adding a varying action
   value to `u`: a wild has no territory node, but its wild token and selected
   flag still make it visible to the trade-in head.

4. Update DQN, Dueling DQN, and PPO together, including inference rows,
   replay/rollout reconstruction, and target-network handling. Dueling's
   clean value row remains state-only; only candidate advantage rows carry
   selected-card information.

5. Add focused tests for selected owned versus unowned territories, wild-card
   sets, skip trade, base-graph immutability, mixed-action batching,
   hand-order invariance, and inference/replay consistency for all learners.

6. Version the model inputs and retrain all three learner families. The new
   node feature and trade-in-head parameters make current checkpoints
   incompatible. Preserve the present checkpoints and evaluation results as
   the current-representation baseline.

## Design decision: separate `proposed_army_delta` column

The injection proposal must not overwrite the state. `GraphAdapter` supplies
a zero-filled `proposed_army_delta` column alongside the real army count, and
this builder changes only that column for reinforce, occupy, and fortify.
The separation leaves clean base graphs genuinely state-only for a value
stream while making a candidate's proposed movement explicit to an advantage
or action-scoring stream. The named column helper lives in `GraphAdapter`, so
the two components share one feature layout rather than duplicating offsets.

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
- `OccupyAction`/`ReinforcementAction`/`FortifyAction` preserve every real
  army-count value and write the expected signed amounts only to
  `proposed_army_delta` at the affected territory row(s).
- `TradeInAction` returns an unmodified copy, as does `SkipTradeAction`;
  the current scorer varies only by hand-slot positions, not card identity or
  a graph perturbation. The limitation and deferred correction are documented
  above.
- A full `ATTACK`-phase legal-action set (68 actions) built and
  `Batch.from_data_list`-ed together without shape errors:
  `DataBatch(x=[2856, 15], edge_index=[2, 11288], edge_attr=[11288, 2], u=[68, 35], ...)`.
