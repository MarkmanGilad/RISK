# General update plan

This is the single planning document for the next large DQN update. **Approved
to implement** as one combined change set (2026-07-24) — nothing here is
coded yet, but the design below is no longer provisional. `Docs/Reward.md`
documents current reward behavior; `Docs/ActionGraphBuilder.md` documents
current action injection.

## One combined experiment

The next run will make the reward and action-injection changes in this document
as one coordinated update. It addresses several known weaknesses at once:

1. local rewards can remain strongly positive in a lost game;
2. abandoning a damaged target is not represented by the reward; and
3. army-moving action candidates overwrite the real army count, so the network
   cannot directly distinguish current state from proposed action effect.

This saves several multi-day runs. The tradeoff is that the result cannot
identify which individual change caused an improvement or regression. Use the
component metrics, roster difficulty, and action behavior to diagnose the
combined result.

## Reward update

### Outcome-balanced territory reward

Set:

```python
REWARD_TERRITORY_HOLD = 0.0       # was 0.05
REWARD_TERRITORY_DELTA = 20.0     # was 1.0; first candidate value
```

On a 42-territory board, one lost territory is currently only
`1.0 x (-1/42) = -0.024`; the candidate becomes `-0.476`. Removing the
always-positive hold term and strengthening territory delta should make
territory trading and long losing games less profitable. Keep all other
existing reward constants, including conquest, card, army-trade, continent,
elimination, and the already-negative low-ratio attack reward.

### Markov-safe unfinished-target penalty

Track each distinct enemy territory attacked but not conquered during the
current player's turn. A later conquest removes that target. Do not penalize
individual failed dice rolls: several rolls may be needed and army trade still
measures progress.

At `StopAttackAction`:

| Situation at stop | Reward |
| --- | ---: |
| Unfinished target(s), no conquest this turn | existing `-2.0` once |
| Unfinished target(s), at least one conquest this turn | new `-0.5 x target count` |
| No unfinished target, no conquest, real attack still legal | existing `-2.0` |
| Otherwise | `0.0` |

Only one row applies; never stack the `-2.0` and per-target penalties.
Because this depends on turn history, it must be visible to the network:

- node feature `unfinished_attack_target` (`0.0`/`1.0` per territory);
- global feature `conquered_this_turn` (`0.0`/`1.0`).

Without these two features, visually identical observations can get different
stop rewards, which violates the Markov assumption from the DQN's perspective.

### Reward code changes

1. **`risk/learning/train_constants.py`** — set the two territory constants
   above and add `REWARD_ATTACK_UNFINISHED_TARGET = -0.5`.
2. **`risk/game/state.py`** — add
   `unfinished_attack_targets_this_turn: set[int]`; initialize, copy, and
   serialize it. No legacy state/checkpoint compatibility is required.
3. **`risk/game/environment.py`** — clear the set in `_begin_turn_for`; add a
   target after its non-conquering attack and remove it after conquest.
4. **`risk/learning/reward.py`** — apply the stop table. Refactor the current
   `_attack(...) -> (total_attack, eliminate)` component interface so it also
   returns a named `unfinished_attack` component. Include that amount in the
   total `attack` reward and log it separately as
   `reward_component_unfinished_attack`.
5. **`risk/learning/graph_adapter.py`** — expose both history features above;
   update declared feature dimensions and dependent network construction.
6. **Tests** — cover state round-trip/reset, target add/deduplicate/remove,
   every stop-reward row, no double penalty, graph features, and dimensions.

## Action-injection update

### Problem

For a reinforcement, occupy, or fortify candidate, the current action graph
rewrites the normal `armies` node feature. For example, state `5` plus a
candidate reinforce of `3` appears to the network only as `8`. It cannot
directly see that `5` was real state and `+3` was the action being evaluated.

Attack already uses the clearer design: it leaves the board unchanged and
marks the selected edge plus dice count in `edge_attr`.

This is the "revisit" case [`Docs/ActionGraphBuilder.md`](ActionGraphBuilder.md#design-decision-army-column-not-a-parallel-proposed_delta-column)
already anticipated: the direct-write design was chosen first specifically
because it required no width change, on the explicit condition that a
parallel column would be added "if training shows the network can't
disentangle proposed from actual army counts." No dedicated evidence of that
failure mode has been collected — this change is approved anyway as part of
the combined experiment (2026-07-24), accepting that a regression in the
combined run will not by itself show whether this piece was the cause. See
"One combined experiment" above for that tradeoff.

### Proposed representation

Keep the real `armies` node feature unchanged. Add one action-only node
feature named `proposed_army_delta` for every candidate graph:

| Candidate action | `armies` | `proposed_army_delta` |
| --- | --- | --- |
| Reinforce 3 at target | unchanged | target `+3` |
| Occupy 4 | unchanged | source `-4`, target `+4` |
| Fortify 6 | unchanged | source `-6`, target `+6` |
| Attack, stop, skip | unchanged | all `0` |

Phase is already a global feature, so the network knows whether the candidate
delta is reinforcement, occupy, or fortify. This is an action encoding only:
the actual post-step environment state still correctly shows the new army
counts and needs no historical "reinforced" marker.

This representation is especially relevant to Dueling DQN. Its decomposition
is `Q(s, a) = V(s) + A(s, a) - mean(A)`: `V(s)` should describe the real board
state, while `A(s, a)` should compare candidate actions from that same state.
The current direct overwrite makes a reinforce candidate from five armies by
three look like an eight-army state, blending state and action. Keeping
`armies = 5` and writing `proposed_army_delta = +3` makes the candidate action
explicit. Plain DQN also scores action candidates and can benefit, but this is
more directly aligned with Dueling's advantage stream.

The final graph layout is:

```text
node:   continent one-hot | owner one-hot | armies
        | unfinished_attack_target | proposed_army_delta

global: existing global features | conquered_this_turn
```

`unfinished_attack_target` is state information and remains the same for all
candidate graphs of a decision. `proposed_army_delta` is action information:
it is zero in the base graph and changes only in the candidate copy.

### Injection code changes

1. **`risk/learning/graph_adapter.py`** — add the state-owned
   `unfinished_attack_target` node column, the zero-filled
   `proposed_army_delta` node column, and the `conquered_this_turn` global
   value. Add named helper functions for the new node-column offsets, as for
   `armies_column_index(...)`; do not hard-code column numbers in the builder
   or tests.
2. **`risk/learning/action_graph_builder.py`** — stop altering the real
   army-count column for reinforce, occupy, and fortify. Instead write their
   signed action amount into `proposed_army_delta`.
3. **Network construction** — update all node-input dimensions. Old model
   checkpoints are intentionally incompatible with the new graph width; do
   not add an adapter or compatibility path.
4. **Tests** — verify base armies remain unchanged; exact deltas appear on
   target/source nodes; history flags are correct; attack edge injection
   remains unchanged; batches retain consistent shapes; and network forward
   passes accept the new node/global widths.

## Combined implementation and run

The recommended base for this update is Dueling DQN: use
`Dueling_DQN_100`. The action-injection change is designed to make the
candidate action clearer to Dueling's `A(s, a)` stream, while the new history
features also make `A(s, StopAttack)` distinguish a successful stop from an
abandoned-target stop.

The comparison caveat remains: `Dueling_DQN_040` used the old 200-episode
epsilon decay. A fresh Dueling control with the current 100-episode decay is
needed for a clean causal comparison. If avoiding that extra multi-day run is
more important, run `Dueling_DQN_100` directly and compare cautiously with
both Dueling 040 and DQN 060; that measures whether the combined update is
promising, not the isolated cause of any difference.

1. Finish DQN 060. If a clean Dueling causal comparison is required, also run
   a fresh Dueling control with the current epsilon schedule; otherwise use
   Dueling 040 and DQN 060 as cautious, non-matched reference curves.
2. Create branch `codex/history-aware-injection` from the current baseline before
   any implementation work. Keep the current branch unchanged so the
   control's code remains immediately available if this combined update
   regresses.
3. Implement all reward and injection changes above in one change set and
   commit them on `codex/history-aware-injection` before starting training. A
   branch does not protect uncommitted work.
4. Change the training launch path to construct `Dueling_DQN_Agent` (or select
   the equivalent Dueling learner-factory option). Confirm the logger/run name
   is `Dueling_DQN_100`; do not accidentally launch the plain-DQN default in
   `trainer.py`.
5. Run the focused tests, then the full test suite.
6. Start a **fresh** combined run at `Dueling_DQN_100` with a new checkpoint
   directory, randomly initialized model, and empty replay buffer. Use
   `resume=False`. Do not load the control checkpoint or any other old
   checkpoint: its network input width, serialized state, action injection,
   and stored rewards are intentionally incompatible with this update.
7. Run with:

   ```python
   REWARD_TERRITORY_HOLD = 0.0
   REWARD_TERRITORY_DELTA = 20.0
   REWARD_ATTACK_UNFINISHED_TARGET = -0.5
   ```

8. Keep evaluation identical to whichever reference(s) step 1 produced — the
   fresh matched Dueling control alone, or both `Dueling_040` and `DQN_060` if
   using the cautious dual-reference path: fixed seeds, opponent roster,
   deterministic learner (`epsilon=0`), episode limit, and number of
   evaluation games. This is measurement-only and must not become another
   training difference.
9. Review at episodes 250-300 and continue to 1K only if competitive with the
   fresh matched control, or, on the cautious path, with both `Dueling_040`
   and `DQN_060`.
10. Compare training win rate, deterministic evaluation, loss-game return,
   loss/win reward ratio, territory-delta and unfinished-target components,
   Q/target scale, gradient clipping, and attack/reinforcement behavior. Also
   compare `player_count`, roster composition, and the conditional winner rate
   for each heuristic opponent, so a harder random roster is not mistaken for
   model regression.

11. After implementation, update `Reward.md`, `GraphAdapter.md`,
    `ActionGraphBuilder.md`, `DuelingDQN.md`, `NetworkArchitectures.md`, and
    `Trainer.md` to
    describe the new current behavior, then add the usual `ChangeLog.md` entry.

Success means losses retain much less positive reward than the current 44% of
average winning return while win rate does not fall. If results regress, use
the reward components and injected-action behavior to decide which part to
isolate next.
