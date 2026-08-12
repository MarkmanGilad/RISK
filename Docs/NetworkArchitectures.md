# Network architectures

This is the current reference for the three supported learning agents: DQN,
Dueling DQN, and PPO. All use the same injected-action graph representation.

## Shared representation

`GraphAdapter` converts a game state into a graph. `ActionGraphBuilder` makes a
copy for each legal candidate and writes that candidate's local effect into the
graph before the shared `Encoder` runs. `pool(...)` combines node embeddings
with global features into one embedding per graph row.

```text
state + legal action -> injected graph row -> Encoder -> phase head
```

`TradeInHead` handles card choices; `ScoringHead` handles reinforcement,
attack, occupy, and fortify actions.

## DQN

`GNN_DQN` scores each injected row directly as `Q(s, a)`. Its agent uses
epsilon-greedy action selection, replay, Smooth-L1 loss, gradient clipping,
and Double-DQN targets: the online net selects the next action and the target
net evaluates it.

## Dueling DQN

`Dueling_DQN` receives one clean state row plus its action-injected rows.
The clean row produces `V(s)` and the phase heads produce `A(s, a)`:

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, legal actions))
```

`value_mask` identifies clean rows and `group_index` keeps each decision's
rows together. Replay updates reconstruct every legal action for each sampled
state so the advantage mean remains exact. The agent otherwise uses the same
replay and Double-DQN workflow as DQN.

## PPO

`PPO_Net` uses the same clean/action-row layout. It outputs one policy logit
per action row and one `V(s)` per clean row. `PPO_Agent` collects 1,024 ordered
learner transitions, keeps detached collection-time log probabilities and
values, builds 16-step bootstrapped targets, then runs clipped-surrogate
updates over shuffled minibatches. The rollout is discarded after the update.
See [PPO.md](PPO.md) for target and loss details.

## Supported learner factory labels

`build_learner_agent(...)` accepts only `"DQN"`, `"Dueling_DQN"`, and
`"PPO"`. Their run labels and checkpoint namespaces use the same names.
