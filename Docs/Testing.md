# Testing

Use the project environment, not the incomplete user-local virtual environment:

```powershell
& "C:\venvs\ai-rl\Scripts\python.exe" -m pytest Temp/tests -q
```

For focused coverage, replace `Temp/tests` with the relevant test file. Tests
that require pygame use `SDL_VIDEODRIVER=dummy`.

## Test map

| File | Coverage |
|---|---|
| `test_board_topology.py`, `test_state.py`, `test_constants_and_phase.py` | board, state serialization, and shared constants |
| `test_actions.py`, `test_environment.py`, `test_reward.py` | game rules, legal actions, and reward accounting |
| `test_agents.py`, `test_human_input.py`, `test_game_loop.py`, `test_ui.py` | agents, interactive controls, application loop, and rendering |
| `test_graph_representation.py` | graph adaptation and action injection |
| `test_dueling_dqn.py`, `test_ppo.py` | the supported Dueling DQN and PPO learners |
| `test_trainer.py`, `test_training_logger.py`, `test_evaluator.py` | learner factory, self-play orchestration, logging, checkpoints, and evaluation |
| `test_choose_agent.py`, `test_learned_agent_play.py` | saved-policy evaluation and interactive policy loading |
| `test_self_play.py` | headless multi-agent game completion |

## Conventions

Use the shared `make_settings(...)` and `make_env(...)` fixtures in
`Temp/tests/conftest.py`. Existing subsystem test files are preferred over
new ones. DQN encoder tests should use approximate comparisons for separate
forward/training runs: scatter-based graph aggregation is not bit-identical
across executions.

The active learner set is DQN, Dueling DQN, and PPO. Retired experimental
tests are preserved outside `Temp/tests` under `Temp/retired_algorithms/` and
are intentionally excluded from the suite.
