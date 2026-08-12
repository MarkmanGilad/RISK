# Risk RL — Graph Attention Network Training Environment

[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Risk-blueviolet)](Docs/NetworkArchitectures.md)
[![Graph Attention Network](https://img.shields.io/badge/Graph%20Attention-TransformerConv-0A7E8C)](Docs/GraphAttentionNetwork.md)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C)](https://pytorch-geometric.readthedocs.io/)
[![Algorithms](https://img.shields.io/badge/Algorithms-DQN%20%7C%20Dueling%20DQN%20%7C%20PPO-2E8B57)](Docs/NetworkArchitectures.md)

Risk RL is a reinforcement-learning training environment for classic Risk,
not only a playable board game. It trains graph-attention neural networks on
the 42-territory Risk board: at every decision, the environment enumerates
only legal moves, injects each candidate move into a graph of territories and
borders, and scores the resulting state-action graph with a shared sparse
Graph Attention Network (GATN). Self-play against heuristic opponents supplies
rollouts, rewards, evaluation, checkpoints, and comparable training runs for
DQN, Dueling DQN, and PPO.

The repository also includes a pygame interface, a pygame-free rules engine,
and human/heuristic/learned seats that all use the same legal-action interface.

## Contents

1. [Quick start](#quick-start)
2. [Play](#play)
3. [Game rules](#game-rules)
4. [Architecture](#architecture)
5. [Risk as a graph-attention RL problem](#risk-as-a-graph-attention-rl-problem)
6. [Learning](#learning)
7. [Tests and documentation](#tests-and-documentation)

## Quick start

```powershell
pip install -r requirements.txt
python -m risk.app.main
```

Run the full test suite with the project virtual environment:

```powershell
& "C:\venvs\ai-rl\Scripts\python.exe" -m pytest Temp/tests -q
```

## Play

Risk is played on a 42-territory world map. The goal is to eliminate every
opponent and control the board. Territories change hands through dice combat:
the attacker commits one to three dice, the defender rolls up to two, and the
highest dice are compared pair by pair. The losing side removes one army for
each comparison. Attacks continue until the attacker stops or conquers the
defending territory.

<img src="Assets/RiskMap/image.png" alt="Playable Risk board" width="760">

The rules engine exposes only actions that are legal in the current state.
This makes the same game flow usable from the pygame interface, heuristic
opponents, headless simulations, and RL training.

The setup screen supports 3–6 players. Each seat can be Human, Random,
Raider, Sentinel, Empire, Killbot, or a saved learned policy. Learned seats
support DQN, Dueling DQN, and PPO policy files.

```text
python -m risk.app.main [--width W] [--height H] [--seed N] [--players N]
                         [--max-ticks N] [--skip-menu] [--auto-restart]
                         [--mode play|train|train-no-render]
                         [--ai-delay-ms N] [--marker-ms N]
```

- `--mode play` is the normal paced graphical game.
- `--mode train` runs graphical AI games without pacing.
- `--mode train-no-render` runs an all-AI game headlessly.
- `--seed N` reproduces the initial deal, card draws, and dice rolls.
- `--skip-menu` starts an all-random game immediately.

### Choose players

Run `python -m risk.app.main` without `--skip-menu` to open the setup screen.

<img src="Assets/RiskMap/start%20UI.png" alt="Risk setup screen for choosing players" width="620">

1. Use the Players `−` and `+` controls to choose 3–6 seats.
2. For each seat, edit its name and cycle its color; names and colors must be
   unique before the game can start.
3. Click the seat-type cell to cycle through Human, Random, Raider, Sentinel,
   Empire, Killbot, and Learned Agent.
4. For a Learned Agent, use **File...** or **Folder...** to select a DQN,
   Dueling DQN, or PPO policy/checkpoint. **Best DQN** cycles the included DQN
   presets.
5. Select **Start Game**.

The setup keeps learned-seat choices outside `GameSettings`: it builds the
ordinary game first, then replaces only the selected `ai` placeholder seats
with fresh, deterministic policy instances. These policies run in evaluation
mode and never train, write checkpoints, or modify their saved weights during
an interactive match.

| Seat type | Behavior |
|---|---|
| Human | Chooses actions through the map and control panel. |
| Random | Samples a legal action uniformly. |
| Raider | Prioritizes aggressive expansion. |
| Sentinel | Prioritizes border defense and safer attacks. |
| Empire | Focuses on capturing and defending continents. |
| Killbot | Uses continent strategy and weak-player elimination. |
| Learned Agent | Loads a saved DQN, Dueling DQN, or PPO policy. |

## Game rules

The map has six continents. Owning every territory in a continent provides an
extra reinforcement bonus. A turn progresses through these phases:

1. **Trade in** cards for bonus armies when required.
2. **Reinforce** owned territories with new armies.
3. **Attack** neighbouring enemy territories zero or more times.
4. **Occupy** a territory immediately after conquering it.
5. **Fortify** once by moving armies between connected owned territories, or
   skip it to end the turn.

Five or more cards require a trade at the start of a turn. An ordinary
conquest that raises a hand from four to five cards does not interrupt the
current turn; card transfers after eliminating an opponent can require trades
before occupation resumes. See [Environment.md](Docs/Environment.md) for the
implemented rule flow.

For each reachable fortify source/destination pair, the learner sees a compact
set of move amounts: one army, half the available armies, or the maximum.
Human and other direct callers may still submit any legal amount.

## Architecture

```text
pygame UI / headless drivers
             |
         risk/app
             |
 risk/game <-> risk/agents <-> risk/learning
```

- `risk/game/` is the pygame-free source of truth for state, legal actions,
  rule application, rewards, and game completion.
- `risk/agents/` implements human, random, and heuristic decision makers.
- `risk/app/` builds matches and runs the interactive or headless loop.
- `risk/ui/` contains rendering and input handling.
- `risk/learning/` contains graph conversion, action injection, training,
  evaluation, checkpoints, and saved-policy loading.

Every agent uses the same callable shape:

```python
agent(events, state) -> Action | None
```

This lets the same environment run an interactive match, a heuristic-only
simulation, or an RL training episode.

## Risk as a graph-attention RL problem

This project is not only a playable Risk game. It is an environment for
training and comparing reinforcement-learning agents on Risk's large,
state-dependent action space.

The board is represented as a graph: the 42 territories are nodes and the 83
board borders are stored as 166 directed edges. Node features describe each
territory's continent, owner, armies, and current tactical state; edge features
describe borders and an injected attack; global features describe turn, phase,
cards, reinforcement budget, and player state.

<img src="Assets/RiskMap/map_graph_nodes_edges.png" alt="Risk board represented as territory nodes and border edges" width="760">

Risk has a huge nominal action space, but most moves are illegal at a given
moment. Instead of using one fixed output for every imaginable move, the
environment enumerates only legal actions. For each candidate, the project
injects that move into a copy of the board graph: attacks mark the selected
border, while reinforcement, occupy, and fortify moves change the affected
territories' proposed-army-delta feature.

<img src="Assets/RiskMap/partial_graph_attributes.png" alt="A candidate attack injected into the Risk graph" width="1000">

One shared graph-attention neural network then scores the resulting
state-action graph. Four residual `TransformerConv` layers pass messages only
along Risk borders, allowing a territory to weigh its neighbouring territories
differently. Mean and max pooling summarize the board, global state is added,
and the current phase selects the appropriate scoring head.

```text
legal action -> action-injected board graph -> graph-attention encoder -> score
```

The same graph representation is used for all supported learners. DQN returns
`Q(s, a)`; Dueling DQN separates state value from action advantage; PPO returns
legal-action policy logits plus a state-value estimate. This keeps the board,
legal-action generator, opponents, and encoder fixed while comparing learning
objectives.

For the complete input dimensions and attention calculation, see
[GraphAttentionNetwork.md](Docs/GraphAttentionNetwork.md).

## Learning

All learning agents score legal actions from action-injected board graphs.
`GraphAdapter` converts the board state to a graph; `ActionGraphBuilder`
creates one candidate graph per legal action.

| Learner | Training method |
|---|---|
| DQN | Replay-based Double DQN |
| Dueling DQN | Replay-based Double DQN with separate value and advantage streams |
| PPO | On-policy clipped policy optimization with a value head |

### Run training

Open `risk/learning/trainer.py` and edit the two settings in `main()`:

```python
RUN_ID = 311
agent = build_learner_agent("PPO", ctx)  # "DQN", "Dueling_DQN", or "PPO"
```

Use a new `RUN_ID` for a fresh run. The launcher currently starts with W&B
logging enabled and does not resume a prior checkpoint (`resume=False`). Then
run:

```powershell
python -m risk.learning.trainer
```

`build_learner_agent(...)` accepts only `"DQN"`, `"Dueling_DQN"`, and
`"PPO"`. Training checkpoints are stored under
`Checkpoints/<learner>_<run_id>/`; evaluation keeps the best policy-only files
in that run's `best/` directory. The trainer samples 3–6-player self-play
games against the configured heuristic opponents and periodically evaluates
the learner on fixed seeded games.

## Tests and documentation

`Temp/tests/` mirrors the major subsystems. Shared fixtures live in
`Temp/tests/conftest.py`; test coverage is mapped in
[Testing.md](Docs/Testing.md).

The active reference documentation is in `Docs/`. Historical experiments and
removed planning material are kept outside the active docs in
`Temp/retired_algorithms/` and `Temp/retired_documents/`.

### Documentation index

The full active-document index is [Docs/Content.md](Docs/Content.md). The same
reference map is included here for readers browsing the repository front page.

#### Game and environment

| Document | What it covers |
|---|---|
| [Action.md](Docs/Action.md) | Game actions, validation boundaries, serialization, and RL action encoding. |
| [BoardTopology.md](Docs/BoardTopology.md) | Territories, borders, continents, and static board data. |
| [Environment.md](Docs/Environment.md) | Rules-engine behavior, card trades, and fortify candidates. |
| [Reward.md](Docs/Reward.md) | Learner reward calculation and component logging. |

#### Agents and user interface

| Document | What it covers |
|---|---|
| [HeuristicAgents.md](Docs/HeuristicAgents.md) | Random and heuristic opponents used for play and training. |

#### Graph representation and GATN

| Document | What it covers |
|---|---|
| [GraphAdapter.md](Docs/GraphAdapter.md) | State-to-graph node, edge, and global features. |
| [ActionGraphBuilder.md](Docs/ActionGraphBuilder.md) | Legal-action injection into candidate graphs. |
| [GraphAttentionNetwork.md](Docs/GraphAttentionNetwork.md) | Sparse `TransformerConv` encoder calculation, parameters, and visuals. |

#### Reinforcement learning and evaluation

| Document | What it covers |
|---|---|
| [NetworkArchitectures.md](Docs/NetworkArchitectures.md) | Shared representation and active learner architectures. |
| [DuelingDQN.md](Docs/DuelingDQN.md) | Dueling DQN value/advantage architecture and replay updates. |
| [PPO.md](Docs/PPO.md) | PPO rollout, targets, clipped optimization, and checkpoints. |
| [Trainer.md](Docs/Trainer.md) | Self-play training orchestration, metrics, and checkpoints. |
| [Eval.md](Docs/Eval.md) | Deterministic in-training evaluation and best-policy retention. |
| [ChooseAgent.md](Docs/ChooseAgent.md) | Offline checkpoint evaluation and policy-versus-policy matches. |

#### Project support

| Document | What it covers |
|---|---|
| [Testing.md](Docs/Testing.md) | Test commands, test map, fixtures, and conventions. |
| [Poster.md](Docs/Poster.md) | Poster-ready project narrative and visuals. |
| [ChangeLog.md](Docs/ChangeLog.md) | Dated history of code and documentation changes. |
