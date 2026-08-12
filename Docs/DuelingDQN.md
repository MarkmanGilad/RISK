# Dueling DQN

`Dueling_DQN` and `Dueling_DQN_Agent` are the project's replay-based dueling
learner. They are separate from the baseline DQN files so historical DQN
checkpoints remain compatible with their original architecture.

The network scores one clean state graph and every legal action-injected graph
in each decision group:

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, legal actions))
```

The clean state row is routed to `value_head`; action rows are routed to the
phase-specific advantage heads. `value_mask` marks clean rows and
`group_index` maps each row to its decision. For replay, the agent rebuilds
the complete legal-action set for every sampled state before selecting the
stored action's Q value, so the mean advantage is exact.

Action selection remains epsilon-greedy. Learning uses replay, Smooth-L1
Bellman loss, gradient clipping, and Double-DQN next-state targets. The online
network selects the next action and the target network evaluates it. Full
checkpoints store the online/target models, optimizer, replay state, epsilon,
and counters; policy-only files store the online model.

The supported factory label is `"Dueling_DQN"`. See [Trainer.md](Trainer.md)
for training orchestration and [Testing.md](Testing.md) for coverage.
