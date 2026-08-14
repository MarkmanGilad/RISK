# Documentation index

This table indexes the active project documentation. It is a map of the
current codebase, not the retired experiments stored under `Temp/`.

## Start here: using the app

| Document | What it covers |
|---|---|
| [README.md](../README.md) | Project overview, game rules, running the app, choosing players, loading learned seats, training, and testing. |

## Game and environment

| Document | What it covers |
|---|---|
| [Action.md](Action.md) | Action dataclasses, serialization, phase assignment, legality boundary, and RL action encoding. |
| [BoardTopology.md](BoardTopology.md) | Static territories, borders, continents, and board graph data. |
| [Environment.md](Environment.md) | Live rules-engine behavior, including card-trade flow and fortify candidates. |
| [Reward.md](Reward.md) | Current learner reward pipeline, components, and logging. |

## Agents and user interface

| Document | What it covers |
|---|---|
| [HeuristicAgents.md](HeuristicAgents.md) | Random and heuristic opponents used for play and RL training. |

## Graph representation and GATN

| Document | What it covers |
|---|---|
| [GraphAdapter.md](GraphAdapter.md) | Conversion of a Risk state into node, edge, and global graph features. |
| [ActionGraphBuilder.md](ActionGraphBuilder.md) | How each legal candidate action is injected into a graph before scoring. |
| [GraphAttentionNetwork.md](GraphAttentionNetwork.md) | The sparse `TransformerConv` encoder calculation, dimensions, parameters, and visuals. |

## Reinforcement learning and evaluation

| Document | What it covers |
|---|---|
| [NetworkArchitectures.md](NetworkArchitectures.md) | Shared injected-graph representation and the active DQN, Dueling DQN, and PPO learners. |
| [DuelingDQN.md](DuelingDQN.md) | Dueling DQN architecture, value/advantage calculation, replay learning, and checkpoints. |
| [PPO.md](PPO.md) | PPO rollout collection, 16-step targets, clipped optimization, metrics, and checkpoints. |
| [DPQN.md](DPQN.md) | Proposed Deep Policy Q-learning hybrid: shared DQN/policy encoder, replay TD learning, and a recent on-policy policy loss. |
| [Trainer.md](Trainer.md) | Self-play orchestration, learner factory, training loop, metrics, checkpoints, and W&B logging. |
| [Eval.md](Eval.md) | Deterministic in-training evaluation, metrics, scoring, and best-policy retention. |
| [ChooseAgent.md](ChooseAgent.md) | Evaluating saved checkpoints and running policy-versus-policy matches. |

## Project support

| Document | What it covers |
|---|---|
| [Testing.md](Testing.md) | Test command, test-file map, fixtures, and testing conventions. |
| [Poster.md](Poster.md) | Poster-ready explanation and visuals for the project. |
| [ChangeLog.md](ChangeLog.md) | Dated record of code and documentation changes; newest entries are first. |

## Outside the active documentation

- `Temp/retired_algorithms/` preserves removed ADQN/PQN code, tests, and
  experiment records.
- `Temp/retired_documents/` preserves retired plans and conflicting document
  copies.
