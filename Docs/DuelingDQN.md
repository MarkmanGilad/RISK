# Dueling_DQN build plan

Goal: add a second learner called **`Dueling_DQN`** while keeping the current
`GNN_DQN_Agent` / `GNN_DQN` path intact and usable. This is a conservative
side-by-side build: do not refactor the old agent, do not replace the old
network, and do not make the trainer silently switch behavior. The new agent
gets its own files and its own checkpoints so run_022/run_023 and future
classic-DQN runs remain reproducible.

This doc is a plan only. No code is implemented here.

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

## File strategy: copy, then change

Per project direction, do **not** modify the old agent or network in place.
Also avoid importing the old `gnn_dqn.py` / `gnn_dqn_agent.py` classes into the
new implementation. Copy the necessary files, rename them, then change the
copies.

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
| `risk/learning/dueling_trainer.py` | The existing `Trainer` already accepts an injected agent and should train any agent with the same public methods. Duplicating the loop would create drift risk. |
| `risk/learning/dueling_evaluator.py` | The existing `Evaluator` only needs the agent interface (`attach`, action selection, `save_params`) and can evaluate the dueling agent unchanged. |

Do not copy neutral shared helpers unless the implementation really needs to
change them. The new files can continue using stable shared infrastructure
such as `GraphAdapter`, `ActionGraphBuilder`, `ActionEncoder`, `ReplayBuffer`,
`Encoder`, `pool`, and the action classes. The important separation is: the
new dueling agent must not subclass/import/reuse the old `GNN_DQN_Agent` or
old `GNN_DQN` network as its implementation.

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

Keep the scoring API close to the existing one, but add an optional group
index:

```python
q = net(batch, phase, card_indices, group_index=None)
```

Semantics:

- `batch`: one graph per candidate action, same as today.
- `phase`: one phase per candidate action, same as today.
- `card_indices`: same as today.
- `group_index`: `[N]` long tensor saying which decision each candidate row
  belongs to. If omitted, treat all rows as one group.

This supports both call shapes:

- one state, all legal actions: `group_index=None` or all zeros,
- many states, many legal actions flattened together: group ids `0..B-1`.

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

Simplest first pass: use the same pooled embedding `g` for each injected
candidate row, then average the produced value per group. This is not perfect
because injected action features slightly perturb `g`, but it avoids adding a
clean-state row and keeps batching simple.

Better second pass, only if needed: include one clean base graph per decision
and route it to the value head. That is architecturally cleaner but requires a
more invasive batching path. Avoid it in v1 unless the simple version behaves
poorly.

### Advantage stream

Reuse the copied per-phase heads as **advantage heads**. They should output
`A(s, a)`, not final `Q(s, a)`. The routing loop remains almost identical to
`GNN_DQN.forward`:

```python
advantage[mask] = self._heads_by_phase[stage](g[mask], card_indices[mask])
```

Then combine:

```python
value = value_head(g).squeeze(-1)
value = mean_by_group(value, group_index)
adv_mean = mean_by_group(advantage, group_index)
q = value + advantage - adv_mean
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

Reuse, don't rebuild, the group bookkeeping that already exists: both
`_max_next_q` and `_max_next_ddqn_q` already build one `groups: list[int]`
entry per candidate row (which next-state each row belongs to) while
flattening all legal next actions into one batch. `_max_next_q` converts
that into a `group_index` tensor and reduces with
`torch_geometric.utils.scatter(..., reduce="max")`; `_max_next_ddqn_q`
currently keeps `groups` as a plain Python list and reduces per-group best
online/target values with a Python dict loop instead of a tensor scatter.
The same `groups` list is exactly the `group_index` the dueling value/
advantage mean needs — convert it once (`torch.tensor(groups, ...)`) and
pass it straight to `net(...)`, rather than deriving a second grouping
scheme for the dueling forward pass.

## Trainer/evaluator plan

Use the existing `Trainer` and `Evaluator`; separate **identity and storage**,
not the training loop. The current `Trainer` already accepts an optional
injected agent, then calls generic methods on that object (`attach`,
`remember`, `learn`, `save_checkpoint`, `load_checkpoint`). If
`Dueling_DQN_Agent` implements the same public interface, rollout, reward
accumulation, eval cadence, W&B episode metrics, and checkpoint cadence can
remain one shared implementation.

Example usage shape:

```python
ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
agent = Dueling_DQN_Agent(player_id=0, env=ctx.env, train_mode=True, epsilon=EPSILON_START)

trainer = Trainer(
    run_id=DUELING_RUN_ID,
    agent=agent,
    resume=False,
    notes="Dueling_DQN_Agent",
)
trainer.train(n_episodes=TRAIN_EPISODES)
```

The one thing that **does** need separation is logging/checkpoint identity:

- W&B config should include `agent_class = type(agent).__name__` in addition
  to the already logged `model_class = type(agent.net).__name__`.
- Dueling runs should use a fresh run id and `resume=False` unless explicitly
  resuming a known dueling checkpoint. `RUN_ID` for classic runs is a plain
  module constant hand-edited in `trainer.py`'s `main()` (see current value
  there before picking a dueling id) — don't hardcode a specific number in
  this doc, since it goes stale as soon as another classic run is started.
- Checkpoint paths should make the architecture obvious rather than reusing
  `Checkpoints/run_NNN` for both. Options:
  - minimal: keep `Checkpoints/run_<id>` but rely on W&B `agent_class`,
    `model_class`, and notes to tell classic and dueling runs apart;
  - better: let `Trainer` accept an optional checkpoint path/name so dueling
    writes under its own namespace, e.g. `Checkpoints/dueling/run_<id>`,
    structurally ruling out any collision with a classic run id regardless
    of numbering.
- Best-policy checkpoints should remain under that run's own `best/` folder,
  so classic and dueling best models never share one directory.

Recommendation for v1: use the shared `Trainer`/`Evaluator`, add only the
small logger/config identity improvement (`agent_class`), and use a distinct
run id or checkpoint folder for dueling. Create a separate trainer only if a
future architecture needs a genuinely different loop (different replay
sampling, different eval rules, PQN policy loss, extra value/advantage metrics,
or different checkpoint cadence).

## Tests

Before training, add narrow tests:

1. `Dueling_DQN.forward` returns one Q per action row.
2. If all advantages in a group are equal, `Q` equals `V` for every action in
   that group.
3. Two groups in one flattened batch normalize advantages independently.
4. `Dueling_DQN_Agent.score_actions(...)` returns one score per legal action
   and keeps tensors on the selected device.
5. `_max_next_ddqn_q(...)` still returns zero for done transitions.
6. `save_checkpoint` / `load_checkpoint` round-trips the dueling net and
   target net.

Do not add a new broad integration test until these focused tests pass. The
existing `Docs/Testing.md` convention says reward/network-specific tests belong
under `Temp/tests/`; use `Temp/tests/test_dueling_dqn.py` rather than extending
unrelated test files.

## Rollout plan

1. Create copied files and rename classes/modules only. Run import/smoke tests.
2. Implement `Dueling_DQN.forward(...)` with value + advantage + grouped mean.
3. Wire `Dueling_DQN_Agent.score_actions(...)` and validate inference on one
   real state.
4. Update `_max_next_ddqn_q(...)` and `_q_value(...)` to pass grouped legal
   action batches.
5. Add checkpoint round-trip tests.
6. Add `agent_class`/architecture identity to W&B config and pick a clearly
  separated dueling checkpoint path or run naming convention.
7. Train through the existing `Trainer(agent=dueling_agent, resume=False)` for
  one minibatch/short smoke run.
8. Run a new full run under a dueling-only run id/checkpoint namespace picked
   at that time (see "Trainer/evaluator plan" above). Do not resume a
   classic `GNN_DQN_Agent` checkpoint into a dueling agent.

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
