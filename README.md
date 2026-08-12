# Risk

A from-scratch Python implementation of classic Risk. The project has a
pygame-free rules engine, a pygame interface, heuristic opponents, and three
supported graph-based reinforcement-learning agents: DQN, Dueling DQN, and
PPO.

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

1. Use the Players `−` and `+` controls to choose 3–6 seats.
2. For each seat, edit its name and cycle its color; names and colors must be
   unique before the game can start.
3. Click the seat-type cell to cycle through Human, Random, Raider, Sentinel,
   Empire, Killbot, and Learned Agent.
4. For a Learned Agent, use **File...** or **Folder...** to select a DQN,
   Dueling DQN, or PPO policy/checkpoint. **Best DQN** cycles the included DQN
   presets.
5. Select **Start Game**.

## Game rules

The game uses the classic 42-territory, 6-continent map. A turn consists of:

1. Trade cards when required, then reinforce owned territories.
2. Make zero or more attacks.
3. Move armies into conquered territories when required.
4. Fortify once or skip fortification.

Five or more cards require a trade at the start of a turn. An ordinary
conquest that raises a hand from four to five cards does not interrupt the
current turn; card transfers after eliminating an opponent can require trades
before occupation resumes. See [Environment.md](Docs/Environment.md) for the
implemented rule flow.

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

Useful references:

- [NetworkArchitectures.md](Docs/NetworkArchitectures.md)
- [Trainer.md](Docs/Trainer.md)
- [DuelingDQN.md](Docs/DuelingDQN.md)
- [PPO.md](Docs/PPO.md)
- [Eval.md](Docs/Eval.md)

## Tests and documentation

`Temp/tests/` mirrors the major subsystems. Shared fixtures live in
`Temp/tests/conftest.py`; test coverage is mapped in
[Testing.md](Docs/Testing.md).

The active reference documentation is in `Docs/`. Historical experiments and
removed planning material are kept outside the active docs in
`Temp/retired_algorithms/` and `Temp/retired_documents/`.
