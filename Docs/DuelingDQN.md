# Dueling_DQN build plan

Goal: add a second learner called **`Dueling_DQN`** while keeping the current
`GNN_DQN_Agent` / `GNN_DQN` path intact and usable. This is a conservative
side-by-side build: do not refactor the old agent, do not replace the old
network, and do not make the trainer silently switch behavior. The new agent
gets its own files and its own checkpoints so run_022/run_023 and future
classic-DQN runs remain reproducible.

**Status: v1 implemented.** `risk/learning/dueling_dqn.py`,
`risk/learning/dueling_dqn_agent.py`, and `Temp/tests/test_dueling_dqn.py`
exist and pass. Shared trainer behavior is documented in `Docs/Trainer.md`.
No full comparison run has been launched yet — that's a separate,
explicit step (see "Rollout plan"), not something this doc's existence
implies happened. The rest of this doc is kept as the design record; where
implementation details became concrete they're noted inline rather than
rewritten out.

## Design decision

Use **Dueling Double DQN**, not the larger PQN idea yet. The only algorithmic
change from the current agent should be the network decomposition:

```text
current:       Q(s, a) directly
dueling:       Q(s, a) = V(s) + A(s, a) - mean(A(s, legal_actions))
```

Keep everything else as close as possible to the working agent:

- same injected action graph representation,
- same replay buffer,
- same epsilon-greedy action selection,
- same Double-DQN target selection/evaluation split,
- same reward function,
- same optimizer/loss/gradient clipping defaults,
- same checkpoint shape concept: `net`, `target_net`, `optimizer`,
  `train_steps`, `epsilon`, replay buffer.

The reason is simple: the current agent already trains well. This build is an
A/B comparison of the dueling architecture, not a rewrite of the whole RL
stack.

**Minimal-diff policy.** The copied files (`dueling_dqn.py`,
`dueling_dqn_agent.py`) should stay line-for-line identical to
`gnn_dqn.py`/`gnn_dqn_agent.py` except where the dueling formula itself
forces a change (value head, advantage-mean grouping, `group_index`
plumbing through `score_actions`/`_q_value`/`_max_next_ddqn_q`, class/module
renames and imports). Resist the urge to also clean up, rename, vectorize,
or otherwise "improve" any code path while it's already open in the editor —
if an improvement looks worthwhile, it applies equally to the old agent, and
the old agent is intentionally frozen because it already trains well and
must stay reproducible for run_022/run_023 and later. Log any such idea
separately (a follow-up note or a new doc) rather than folding it into this
build; don't apply it to only one of the two agents. When in doubt, prefer
the smaller diff over the "better" version of a copied block.

## File strategy: copy, then change

Per project direction, do **not** modify the old agent or network in place.
Also avoid importing or subclassing the old `GNN_DQN` / `GNN_DQN_Agent`
implementations in the new implementation. Copy those two files, rename them,
then change the copies.

Planned new files:

| New file | Start from | Purpose |
|---|---|---|
| `risk/learning/dueling_dqn.py` | copy `risk/learning/gnn_dqn.py` | Dueling network: shared encoder, value head, advantage heads, legal-action mean normalization |
| `risk/learning/dueling_dqn_agent.py` | copy `risk/learning/gnn_dqn_agent.py` | Game-facing agent wrapper around `Dueling_DQN` |
| `Temp/tests/test_dueling_dqn.py` | new/copy patterns from `test_reward.py` and agent tests | Focused network/agent behavior tests |
| `Docs/DuelingDQN.md` | this file | Build plan and implementation notes |

Not planned for v1:

| File | Reason |
|---|---|
| `risk/learning/dueling_trainer.py` | The shared `Trainer` requires an explicit agent and should train any agent with the same public methods. Duplicating the loop would create drift risk. See `Docs/Trainer.md`. |
| `risk/learning/dueling_evaluator.py` | The existing `Evaluator` only needs the agent interface (`attach`, action selection, `save_params`) and can evaluate the dueling agent unchanged. |

Import rule for v1: **copy the old agent/network, reuse stable shared
helpers.** The new dueling files should not import/subclass/reuse the old
`GNN_DQN_Agent` or old `GNN_DQN` classes as their implementation, but they
should continue importing shared infrastructure that is not architecture-
specific: `GraphAdapter`, `ActionGraphBuilder`, `ActionEncoder`,
`ReplayBuffer`, `Encoder`, `pool`, `ScoringHead`, `TradeInHead`, and the game
action/state classes. Copying those neutral helpers would add duplication
without making the dueling comparison cleaner.

If the project goal becomes *complete physical duplication* later, the shared
helpers can also be copied under dueling-specific names. For the first pass,
that is unnecessary complexity and would make future bug fixes harder.

## Network plan: `Dueling_DQN`

Start from the current `GNN_DQN` shape:

```python
q = net(batch, phase, card_indices)
```

The dueling network needs one extra concept: which scored action rows belong
to the same decision, because the dueling formula subtracts the mean advantage
over the legal actions of that decision:

```text
Q(s, a_i) = V(s) + A(s, a_i) - mean_i(A(s, a_i))
```

For normal action selection, every row in the batch belongs to one decision,
so the mean is over all rows. For replay minibatches, each row is usually a
different `(state, action)` pair, so computing the dueling mean requires all
legal actions for each sampled state, not only the taken action. This is the
main implementation choice.

### Recommended network API

Keep the scoring API close to the existing one, but add group/value-row
metadata:

```python
q = net(batch, phase, card_indices, group_index=group_index, value_mask=value_mask)
```

Semantics:

- `batch`: one clean graph plus one graph per candidate action for each
  decision group.
- `phase`: one phase per row; the clean row carries the same phase as that
  decision but is ignored by the advantage heads.
- `card_indices`: same shape as today; the clean row carries zeros and is
  ignored by the advantage heads.
- `group_index`: `[N]` long tensor saying which decision each candidate row
  belongs to.
- `value_mask`: `[N]` bool tensor; `True` for the clean row used by the value
  stream, `False` for action-injected rows used by the advantage stream.

This supports both call shapes:

- one state, all legal actions: one clean row plus all candidate rows, all
  `group_index == 0`,
- many states, many legal actions flattened together: one clean row per
  sampled state/decision, group ids `0..B-1`.

### Value stream

Add one value head:

```python
self.value_head = nn.Sequential(
    nn.Linear(g_dim, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 1),
)
```

Question: what graph embedding should feed `V(s)`?

**Implemented answer:** include one clean base graph per decision and route
that row to the value head. The clean row is the first row for the decision:

```text
(s clean), (s + action_1), (s + action_2), ..., (s + action_k)
```

`value_mask` marks the clean row(s). `Dueling_DQN.forward(...)` computes
`V(s)` only from those clean, non-injected rows, while the unmasked rows are
the action-injected candidates routed to the advantage heads. This is more
bookkeeping than averaging value predictions over injected action rows, but it
keeps `V(s)` truly state-only and avoids action perturbations leaking into the
value stream.

### Advantage stream

Reuse the copied per-phase heads as **advantage heads**. They should output
`A(s, a)`, not final `Q(s, a)`. The routing loop remains almost identical to
`GNN_DQN.forward`:

```python
advantage[mask] = self._heads_by_phase[stage](g[mask], card_indices[mask])
```

Then combine:

```python
value = value_head(g[value_mask]).squeeze(-1)
value = mean_by_group(value, group_index[value_mask])
adv_mean = mean_by_group(advantage, group_index[action_mask])
q = value[group_index[action_mask]] + advantage - adv_mean[group_index[action_mask]]
```

Use `torch_geometric.utils.scatter(..., reduce="mean")` for the group means,
matching the existing agent's use of `scatter` for grouped max reductions.

## Agent plan: `Dueling_DQN_Agent`

Copy `GNN_DQN_Agent` and change only what is needed:

1. Rename the class to `Dueling_DQN_Agent`.
2. Import/use `Dueling_DQN` from the new `dueling_dqn.py` file.
3. Keep `resolve_device`, `attach`, `save_params`, `save_checkpoint`,
   `load_checkpoint`, `remember`, `learn`, and `learn_steps` behavior the
   same unless the dueling network API requires a small call-site adjustment.
4. Keep epsilon-greedy action selection unchanged.
5. Keep Double-DQN training unchanged conceptually:
   - online net selects best next action,
   - target net evaluates that selected next action,
   - target is `reward + gamma * max_next_q` when not done.

### Important training detail

For dueling DQN, scoring a single taken action without the other legal actions
does not give the correct `mean(A)` term unless the mean is approximated. The
cleaner plan is:

- For `score_actions(...)`: unchanged — it already scores all legal actions
  for one state, so the dueling mean is exact.
- For `_max_next_ddqn_q(...)`: mostly unchanged — it already builds all legal
  next actions for each next state, so pass a `group_index` and the dueling
  mean is exact.
- For `_q_value(...)`: change from "one row per sampled transition" to
  "all legal actions for each sampled transition's state," then select the Q
  row corresponding to the replayed action. This makes the dueling mean exact
  for the current-state Q as well.

That last item is the main cost of dueling: current-Q training becomes more
expensive because each sampled state needs its legal-action set, not just the
taken action. Keep the first implementation simple and correct; optimize only
if training becomes too slow.

**Implemented as:** `_q_value` rebuilds `legal = self.env.legal_actions(s)`
per transition, prepends one clean base graph row for that transition's
`V(s)`, then appends all legal action-injected rows for `A(s, a)`. It finds
the replayed action's position with `legal.index(a)` and records that row's
index in the returned **action-only** Q tensor before moving to the next
transition. `legal.index(a)` works because every `Action` subclass is a
`@dataclass(frozen=True, slots=True)` (`risk/game/actions.py`), which gets
value-based `__eq__` for free — the freshly rebuilt legal-action list
compares equal to the stored action by field values, not identity, so no
extra action-id/index scheme was needed. Verified directly (a smoke run
through `train_step` matched a transition's stored action back to its row
and produced a finite loss with gradients flowing only through the online
net).

Reuse, don't rebuild, the group bookkeeping that already exists:
`_max_next_ddqn_q` already builds one `groups: list[int]` entry per
candidate row (which next-state each row belongs to) while flattening all
legal next actions into one batch. That same list is exactly the
`group_index` the dueling `net(...)` call needs for its value/advantage
mean — convert it once (`torch.tensor(groups, ...)`) and pass it straight
through, rather than deriving a second grouping scheme.

Per the minimal-diff policy above, this is *additive only*: keep
`_max_next_ddqn_q`'s existing per-group best-online/target-value selection
(the Python dict loop) exactly as it is in the old agent. Do not rewrite
that loop into a `torch_geometric.utils.scatter(..., reduce="max")` form to
match `_max_next_q`'s style, even though that pattern already exists
elsewhere in the same file — that would be an unrelated cleanup of code the
dueling build doesn't need to touch, and the same argument for "fixing" it
would then apply to the old agent too.

## Trainer notes for dueling

Shared trainer behavior lives in `Docs/Trainer.md`. Dueling does not need its
own trainer or evaluator: `Dueling_DQN_Agent` implements the same public agent
interface, so the existing rollout, reward accumulation, eval cadence, W&B
logging, and checkpoint cadence remain unchanged.

The dueling-specific run rules are:

- build `Dueling_DQN_Agent` in the caller and pass it into `Trainer`;
- use a fresh run id and `resume=False` unless intentionally resuming a known
  dueling checkpoint;
- use the default clearly separated checkpoint path and W&B run name,
  `Dueling_DQN_<id>`;
- rely on W&B `run_name`, `agent_class`, `model_class`, plus
  `notes="Dueling_DQN_Agent"` to make the architecture visible in run
  metadata;
- keep `Trainer.train(...)` unchanged. Dueling-specific behavior belongs in
  `Dueling_DQN_Agent` and `Dueling_DQN`.

## Tests

**Implemented** in `Temp/tests/test_dueling_dqn.py` (all 6 pass, alongside
the full existing suite):

1. `Dueling_DQN.forward` returns one Q per action row.
2. If all advantages in a group are equal, `Q` equals `V` for every action in
   that group.
3. Two groups in one flattened batch normalize advantages independently.
4. `Dueling_DQN_Agent.score_actions(...)` returns one score per legal action
   and keeps tensors on the selected device.
5. `_max_next_ddqn_q(...)` still returns zero for done transitions.
6. `save_checkpoint` / `load_checkpoint` round-trips the dueling net and
   target net.

`Temp/tests/test_training_logger.py`'s existing config test also gained one
assertion (`config["agent_class"] == "GNN_DQN_Agent"`) to cover the new
`_build_config` field.

Do not add a new broad integration test until these focused tests pass. The
existing `Docs/Testing.md` convention says reward/network-specific tests belong
under `Temp/tests/`; use `Temp/tests/test_dueling_dqn.py` rather than extending
unrelated test files.

## Review log (2026-07-04)

Reviewed only the new dueling implementation (`dueling_dqn.py`,
`dueling_dqn_agent.py`, `test_dueling_dqn.py`) against this plan; the classic
`GNN_DQN`/`GNN_DQN_Agent` files were left untouched.

Finding: `Dueling_DQN.forward(...)` correctly accepted an optional
`group_index`, and `Dueling_DQN_Agent._score(...)` already moved that tensor to
`self.device`. But the network's public `forward(...)` contract itself did not
enforce device/dtype if called directly (for example from a test, future helper,
or ad-hoc analysis script) with a CPU `group_index` while the graph batch lived
on CUDA. That could produce a device mismatch inside the grouped scatter even
though the agent call path was safe.

Fix applied in `dueling_dqn.py` only:

```python
if group_index is None:
  group_index = torch.zeros(g.shape[0], dtype=torch.long, device=g.device)
else:
  group_index = group_index.to(device=g.device, dtype=torch.long)
```

`Temp/tests/test_dueling_dqn.py`'s grouped-batch test now deliberately passes a
CPU `group_index` while the batch/phase/card tensors are moved to
`agent.device`, so the network-level device normalization is covered. Focused
smoke validation on CUDA passed after the fix:

```text
score_actions (42,) cuda:0 True
train_step_loss 0.1716330647468567 finite True
```

No other dueling-specific bugs were found in this review. Main implementation
remarks to keep in mind for the first real run: `_q_value(...)` is intentionally
more expensive than classic DQN because it enumerates every legal action for
each sampled current state; that is the cost of exact dueling mean
normalization. If training slows noticeably, optimize this path later, but do
not change the old agent while doing so.

Follow-up design change from the user: `V(s)` should come from a clean,
non-injected state row, not from averaging value predictions over the
action-injected rows. Implemented in the new dueling files only:

- `Dueling_DQN.forward(...)` now accepts `value_mask` and returns Q-values for
  action rows only.
- `Dueling_DQN_Agent.score_actions(...)`, `_q_value(...)`, `_max_next_q(...)`,
  and `_max_next_ddqn_q(...)` now prepend one clean base graph row per decision
  group, mark it with `value_mask=True`, and append legal action rows with
  `value_mask=False`.
- The advantage mean is still computed over legal action rows only; the value
  stream reads only the clean row for each group.

Smoke validation after this change on CUDA:

```text
score_actions (42,) cuda:0 True
train_step_loss 0.24711650609970093 finite True
```

## Rollout plan

1. ✅ Create copied files and rename classes/modules only. Run import/smoke tests.
2. ✅ Implement `Dueling_DQN.forward(...)` with value + advantage + grouped mean.
3. ✅ Wire `Dueling_DQN_Agent.score_actions(...)` and validate inference on one
   real state.
4. ✅ Update `_max_next_ddqn_q(...)` and `_q_value(...)` to pass grouped legal
   action batches.
5. ✅ Add checkpoint round-trip tests.
6. ✅ Add `agent_class`/architecture identity to W&B config and pick a clearly
   separated dueling checkpoint path or run naming convention
  (`checkpoint_dir` override on `Trainer.__init__`; see `Docs/Trainer.md`).
7. ✅ Train through the existing `Trainer(agent=dueling_agent, resume=False)`
   for one minibatch/short smoke run (one full episode, `use_wandb=False`,
   custom `checkpoint_dir` — confirmed the episode loop, `remember`/`learn`,
   and checkpoint path all work with the dueling agent unmodified).
8. Run a new full run under a dueling-only run id/checkpoint namespace picked
  at that time (see "Trainer notes for dueling" above). Do not resume a
   classic `GNN_DQN_Agent` checkpoint into a dueling agent. **Not done yet** —
   the smoke run above was one episode, not a real training run; picking a
   run id and launching one is a deliberate separate step.

## Success criteria

The build is complete when:

- the old `GNN_DQN_Agent` still imports, trains, evaluates, and resumes as
  before,
- `Dueling_DQN_Agent` can select actions in the same environments,
- one replay minibatch update changes online dueling parameters while the
  target net stays frozen between syncs,
- dueling checkpoints save/load independently and are not mixed with classic
  checkpoints,
- W&B/local logs identify the agent class clearly,
- first comparison run reports `win`, `win_rate_last_50`, reward components,
  and eval metrics in the same shape as the classic runs.

## Non-goals for v1

- Do not implement PQN policy loss.
- Do not change the reward function.
- Do not change action encoding.
- Do not refactor shared helpers.
- Do not replace the existing trainer in place.
- Do not migrate old checkpoints.
