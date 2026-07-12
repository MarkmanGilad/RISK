# Network architectures — action-injection roadmap

This doc is the current architecture reference for the learning networks built
on top of [`GraphAdapter`](GraphAdapter.md),
[`ActionGraphBuilder`](ActionGraphBuilder.md), and [`Action.md`](Action.md).

The decision is now settled: **every learner injects the candidate action into
the graph before the GNN encoder runs.** The comparison we care about is the
learning algorithm, not a second action-representation family.

## Settled action representation

Earlier planning considered a cheaper **lookup** alternative where the GNN
encoded the plain board once and the action entered later in the scoring head.
That would have reduced the number of GNN rows per decision, but it also meant
message passing never saw "this board plus this specific action" as one object.
For Risk, that was the wrong tradeoff to keep pursuing: attacks, fortifies,
occupy decisions, and reinforcements are local graph changes whose value often
depends on how the proposed action alters neighboring territory relations.

Action injection has already proven viable in the implemented DQN path and the
Dueling DQN path. Its per-decision cost has not been the blocker in the W&B
runs; system metrics point more toward Python/game/action-graph overhead than a
pure GNN-compute limit. So the roadmap stays on one representation:

```text
State graph + candidate action -> injected graph row -> shared encoder -> scalar(s)
```

Everything below assumes that representation.

## Learners we are building

| Learner | Status | Main output | Training style | Reference |
|---|---|---|---|---|
| **DQN** | implemented | `Q(s, a)` | off-policy replay + Double-DQN target | `risk/learning/gnn_dqn.py`, `risk/learning/gnn_dqn_agent.py` |
| **Dueling DQN** | implemented | `V(s)` + `A(s, a)` -> `Q(s, a)` | off-policy replay + Double-DQN target | `Docs/DuelingDQN.md` |
| **PPO** | planned | policy logits + `V(s)` | on-policy rollout + clipped surrogate | `Docs/PPO.md` |
| **PQN** | planned | dueling Q-values plus policy from advantages | replay-based value + policy improvement | `Docs/PQN.md` |

The order is deliberate. DQN is the working baseline. Dueling DQN isolates the
effect of adding a value stream. PPO changes the optimization method while
keeping the same injected-action encoder shape. PQN is the later unified idea:
keep replay efficiency, use the dueling value/advantage structure, and add an
explicit policy-improvement objective.

## Shared foundation

All four learners use the same graph/action foundation unless a design doc says
otherwise.

### Base state graph

`GraphAdapter` converts a `State` into a PyG `Data` object with territory node
features, directed board edges, edge attributes, and global features:

```text
Data(x=[42, 13], edge_index=[2, 166], edge_attr=[166, 2], u=[1, 34])
```

The base graph is action-independent. It represents the board from the
learner's perspective, so the model sees a stable "me versus others" frame even
when `Trainer` reassigns the learner to different seats across episodes.

### Candidate action graph

`ActionGraphBuilder` takes the base graph, one legal action, and the state, then
returns an injected graph row for that candidate. It never mutates the base
graph.

For attacks, the selected directed edge receives action-specific edge features:

```python
edge_attr[index_of(from_territory -> to_territory)] = [1, dice / MAX_ATTACK_DICE]
```

For reinforcement, occupy, and fortify choices, the relevant territory row(s)
receive the proposed army-count perturbation. Sentinel/no-op decisions such as
stop attack or skip fortify use an unmodified base copy.

The action graph is the unit scored by the network. One legal decision becomes
one batch of graph rows, one row per legal action, plus a clean base row when an
algorithm needs a state-value stream.

### Shared encoder

`risk/learning/encoder.py` provides the common GNN encoder. It projects node
features into `hidden_dim`, runs residual `TransformerConv` layers, and returns
one embedding per territory node:

```python
h = self.input_proj(x)
for conv in self.convs:
    h = h + F.relu(conv(h, edge_index, edge_attr))
return h
```

`TransformerConv` is graph attention, not a spectral GCN, so the neutral name
`Encoder` is intentional. The same encoder class is reused by DQN, Dueling DQN,
PPO, and PQN.

### Pooling

`risk/learning/pooling.py` turns node embeddings into one graph embedding per
row:

```python
g = torch.cat([global_mean_pool(h), global_max_pool(h), u], dim=-1)
```

Global features `u` are concatenated after node pooling because they are already
graph-level inputs; message passing has nothing to refine there.

### Phase heads

`risk/learning/heads.py` contains the per-phase scalar heads:

- `TradeInHead` for `TRADE_IN`, including card-slot embeddings.
- `ScoringHead` for `REINFORCE_PLACE`, `ATTACK`, `OCCUPY`, and `FORTIFY`.

The call shape is uniform:

```python
head(g_rows, card_indices_rows) -> scalar_rows
```

The same head classes can output different meanings depending on the learner:

- DQN: direct Q-values.
- Dueling DQN: action advantages.
- PPO: policy logits.
- PQN: action advantages used for both Q-values and policy logits.

The input representation stays the same; only the training objective and final
interpretation change.

## Shared agent/trainer contract

`Trainer` does not build hidden default agents. The run entry point constructs
the agent explicitly and passes it into `Trainer`:

```python
trainer = Trainer(RUN_ID, agent=agent)
trainer.train(n_episodes=TRAIN_EPISODES)
```

Every learner should provide the interface documented in
[`Trainer.md`](Trainer.md): `attach`, callable action selection, `remember`,
`learn`, checkpoint methods, `set_train_mode`, `epsilon`, `train_mode`, `net`,
`target_net`, and a short `label` used for run/checkpoint naming.

The trainer loop remains shared unless an algorithm truly needs a different
rollout contract. DQN and Dueling DQN already fit it. PPO is planned to keep the
same public calls while making rollout collection/update boundaries agent-owned.
PQN should also keep replay/update logic inside the agent.

## DQN

**Implemented.** `GNN_DQN` scores one scalar per injected action row:

```text
base state
  + legal action a_i
  -> ActionGraphBuilder
  -> Batch.from_data_list([...])
  -> GNN_DQN.forward(batch, phase, card_indices)
  -> Q(s, a_i)
```

`GNN_DQN_Agent` owns legal-action enumeration, graph construction, batching,
epsilon-greedy action selection, replay storage, and learning. Training uses
off-policy replay and a Double-DQN target: the online network selects the next
action and the target network evaluates it.

The default run label is `DQN`, so run/checkpoint names look like `DQN_030`.

## Dueling DQN

**Implemented.** Dueling DQN keeps the same injected action rows but adds one
clean state row for the value stream:

```text
(s, clean), (s, a_1), (s, a_2), ..., (s, a_N)
```

The clean row is routed to a value head:

$$
V(s)
$$

Each injected action row is routed to the phase head as an advantage:

$$
A(s, a_i)
$$

The network combines them per decision group:

$$
Q(s, a_i) = V(s) + A(s, a_i) - \operatorname{mean}_j A(s, a_j)
$$

The clean row is explicit through `value_mask`; `group_index` identifies which
rows belong to the same decision. This is important for replay minibatches,
where each sampled transition may require reconstructing all legal actions for
that sampled state so the advantage mean is correct.

The default run label is `Dueling_DQN`, so run/checkpoint names look like
`Dueling_DQN_040`.

## PPO

**Planned.** PPO reuses the Dueling batch shape because it needs both action
logits and a state value:

```text
(s, clean) -> value head -> V(s)
(s, a_i)  -> phase head -> logit(s, a_i)
```

The policy is a categorical distribution over legal-action logits:

$$
\pi(a_i \mid s) = \operatorname{softmax}(\operatorname{logit}(s, a_i))
$$

The main change is not the graph encoder; it is the learning rhythm. PPO is
on-policy: the agent collects an ordered rollout, stores collection-time
`old_log_prob` and `old_value`, computes GAE, then runs clipped-surrogate
updates over that rollout. `Docs/PPO.md` records the detailed plan, including
detached rollout storage and planned `PPO_*` constants.

The default run label should be `PPO`, so run/checkpoint names look like
`PPO_<id>` once implemented.

## PQN

**Planned.** PQN is the new unified value-policy learner. It starts from the
Dueling architecture and interprets the advantage stream as both value-learning
structure and policy logits:

```text
(s, clean) -> V(s)
(s, a_i)  -> A(s, a_i)

Q(s, a_i) = V(s) + A(s, a_i) - mean(A)
pi(a_i|s) = softmax(A(s, a_i))
```

The Bellman part stays close to Dueling Double DQN. The policy part adds a
replay-based policy-improvement objective, not a PPO-style on-policy gradient.
That distinction matters: PQN keeps replay efficiency and does not store old
policy probabilities in the replay buffer. See [`PQN.md`](PQN.md) for the full
motivation and loss design.

The default run label should be `PQN`, so run/checkpoint names look like
`PQN_<id>` once implemented.

## Comparison plan

The comparison is now among algorithms that share the same injected-action
foundation.

| Learner | Representation | Main question |
|---|---|---|
| DQN | injected action -> direct Q | Baseline: how strong is the current replay/value learner? |
| Dueling DQN | clean state + injected actions -> V/A/Q | Does separating state value from action advantage improve ranking and stability? |
| PPO | clean state + injected actions -> logits/V | Does on-policy policy optimization outperform replay-based value learning? |
| PQN | clean state + injected actions -> V/A/Q/policy | Can one replay-based dueling scorer learn both values and a useful policy? |

Recommended experiment order:

1. Keep DQN as the baseline with known-good checkpoints and W&B history.
2. Finish the Dueling DQN comparison run and evaluate against DQN.
3. Build PPO only after Dueling has a meaningful comparison window.
4. Build PQN after PPO has results, so PQN is compared against both pure value
   learning and pure on-policy policy optimization.
5. Compare agents by eval score, eval win rate, `win_rate_last_50`, wall-clock
   throughput, and system metrics. Do not judge speed from parallel runs on the
   same GPU; W&B system GPU memory/utilization is device-level and will combine
   active processes.

## Non-goals

- Do not add another action representation in v1 of these learners.
- Do not create separate trainers for architecture changes alone.
- Do not refactor the working DQN/Dueling code while building PPO or PQN unless
  the same fix is required for correctness across all learners.
- Do not tune every algorithm equally before the first comparison. Get a clean
  baseline run first, then spend tuning effort where the results justify it.