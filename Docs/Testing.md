# Testing

How `Temp/tests/` is organized, and the conventions worth mirroring when
adding to it. Deliberately **not** a list of individual tests — test names
and bodies change far more often than file-level organization does, so a
line-by-line inventory would go stale fast and end up misleading (worse
than no doc at all). For "does test X still exist / what does it actually
assert," read the file or run:

```bash
python -m pytest Temp/tests -q                  # run everything
python -m pytest Temp/tests --collect-only -q   # list every test name, always accurate, no maintenance
```

Pygame-dependent tests need `SDL_VIDEODRIVER=dummy` set (some files set it
themselves via `os.environ.setdefault(...)` at import time; safe to also
set it in the shell before running the suite).

---

## Which file covers what

| File | Subsystem |
|---|---|
| `test_board_topology.py` | `BoardTopology` — territories, adjacency, continents |
| `test_state.py` | `State` — construction, `to_dict`/`from_dict` round-trip |
| `test_player_card_settings.py` | `Player`, `Card`/`CardRules`, `GameSettings` |
| `test_constants_and_phase.py` | `risk/constants.py`, `Phase` enum (membership, ordering) |
| `test_actions.py` | Every `Action` subclass — construction/validation, `phase` attribute, `to_dict`/`ActionCodec` round-trip |
| `test_environment.py` | `Environment` — the rules engine: reinforcement math, trade-ins, combat resolution, conquest/occupy, fortify, turn advancement, illegal-action rejection |
| `test_agents.py` | `HumanAgent`/`RandomAgent`/heuristic agents (`AttackAgent`, `BSRAgent`, `CompositeAgent`, `RaiderAgent`, `SentinelAgent`, `EmpireAgent`, ...) |
| `test_human_input.py` | `HumanInputController` — the interactive UI decision-builder (clicks, HUD buttons/fields, widget view-models) |
| `test_game_loop.py` | `Game` (the tick-based app loop) and the `risk.app.main` entry point |
| `test_ui.py` | Hit-testing (`TerritoryHitTester`) and the init/setup screen |
| `test_self_play.py` | `SelfPlay.play_headless` — multi-seed fuzz test, full AI-only games played to an actual winner; rendered last-move attribution for turn-advance actions |
| `test_reward.py` | `RewardCalculator` — terminal semantics, phase shaping helpers, reinforcement readiness/total/continent/interior/split formulas and component aggregation |
| `test_graph_representation.py` | `GraphAdapter`/`ActionGraphBuilder` — Markov turn-history features and signed proposed-army-delta injection while preserving real armies |
| `test_evaluator.py` | `Evaluator` (`Docs/Eval.md`) — `evaluate(...)`'s returned metric keys/determinism and `epsilon`/`train_mode` restore, `maybe_save_best(...)`'s top-N retention and manifest sorting |
| `test_training_logger.py` | `TrainingLogger` — config building, checkpoint path/cadence orchestration (`save_checkpoint`/`try_resume`), no-op behavior with W&B disabled. Agent-internals round-tripping stays in `test_agents.py` (below) |
| `test_trainer.py` | `Trainer` (`Docs/Trainer.md`) — the `reached_max_steps` contract passed to `agent.learn(...)` (only the final learn call of a truncated episode gets `True`), `PQN`/`PQN_e`/`PQN_e0` factory selection, short end-to-end smoke runs through `Trainer.train()` for `GNN_DQN_Agent`/`Dueling_DQN_Agent` with a monkeypatched small `MAX_STEPS_PER_EPISODE`, and the logged per-episode metric keys |
| `test_pqn.py` | `PQN`/`PQN_Agent` (`Docs/PQN.md` §24) — raw `(V, A)` scoring, grouped Q calculation, policy-loss TD-weight detachment and graph-connected entropy, sampled-policy and epsilon-greedy-Q action selection, the `PQN_e0` Bellman-only loss control, Dueling-identical epsilon decay, replay/learn threshold behavior, `pqn_*`/epsilon metrics, legacy and current checkpoint loading, and checkpoint round-trip |
| `test_adqn.py` | standalone `ADQN`/`ADQN_Agent` (`Docs/ADQN.md`) — sibling separation from PQN, raw `(V, A)` scoring, centered advantages, scaled/bounded/detached TD weights, signed auxiliary-loss direction, adaptive loss cap and cancellation behavior, DDQN/Dueling parity, epsilon behavior, gradient diagnostics, metric cadence, and checkpoint round-trip |

`risk/learning/` (the RL layer — `GraphAdapter`, `ActionGraphBuilder`,
`Encoder`, `Heads`, `GNN_DQN`, `ReplayBuffer`, `ActionEncoder`) has limited
checked-in tests so far, with `RewardCalculator` (`test_reward.py`),
`TrainingLogger` (`test_training_logger.py`), `Evaluator`
(`test_evaluator.py`), `Trainer` (`test_trainer.py`), `PQN_Agent`
(`test_pqn.py`), and `ADQN_Agent` (`test_adqn.py`) the exceptions — each fully covered per their respective
design docs. `test_agents.py` also
covers `GNN_DQN_Agent.save_checkpoint`/`load_checkpoint` (full training-state
round trip) alongside its existing `save_params`/`load_params` (policy-only)
test — see "Ad-hoc verification" below for how the rest has been validated
instead, and consider promoting some of those scripts into real tests once
these modules stop changing shape every session.

Note on GNN determinism: `GNN_DQN`/`Dueling_DQN`'s encoder uses
`torch_geometric`'s scatter-based `TransformerConv` aggregation, which is
**not** bit-deterministic across separate forward passes even with fixed
`torch`/env/agent seeds (confirmed empirically — running the identical code
twice produces ~1e-5 relative differences in individual weight values after
one gradient step). Tests that need to compare training outcomes across two
runs should compare the returned loss values (stable to displayed precision
in practice) with `pytest.approx`, not raw `net.state_dict()` weights with
exact equality — see `test_*_reached_max_steps_flag_is_inert_with_full_replay`
in `test_agents.py`/`test_dueling_dqn.py` for the pattern.

## Shared fixtures — `conftest.py`

`make_settings(n=3, seed=0, agent_kind="ai", human_ids=None)` and
`make_env(...)` (same args, also calls `reset()`) are the standard way to
get a `GameSettings`/`Environment` without hand-rolling `Player` tuples.
Use `human_ids={0}` etc. to mark specific seats human for UI tests. Most
test files alias these locally as `_settings()`/`_fresh_env()` with their
own defaults.

`test_ppo.py` covers the standalone PPO implementation: policy/value output
shape, GAE cutoff boundaries, collection-time action metadata, rollout gating,
cached action-index validation, grouped minibatches, checkpoint restoration,
rollout-progress metrics, and the non-negative k3 KL estimate used for PPO
early stopping. It also verifies diagnostic optimizer/sample counters and
their checkpoint round trip, plus the separation between optimized Huber
critic loss and raw MSE/RMSE diagnostics. `test_trainer.py` covers common compute metrics
and aggregation of every update in an episode, including `_max` fields.

## Conventions worth mirroring

- **New sentinel/no-op action** (like `SkipTradeAction`, `StopAttackAction`,
  skip-`FortifyAction`): give it a construction test, a `phase` attribute
  assertion, and a round-trip case added to `test_actions.py`'s
  `test_action_round_trip` parametrize list.
- **New phase transition**: mirror `test_environment.py`'s
  `test_fortify_skip_advances_turn` shape — drive the env to the phase via
  the relevant skip/stop action, assert the resulting `phase` and
  `current_player_index`.
- **A phase that requires "warming up" past an earlier phase** (e.g.
  `REINFORCE_PLACE` requires leaving `TRADE_IN` first): add a small helper
  near the top of the file (see `test_environment.py`'s
  `_skip_to_reinforce_place`, `test_human_input.py`'s `_skip_trade`) rather
  than repeating the warm-up steps inline in every test.
- **New UI control** (HUD button/field): add the dispatch case to
  `HumanInputController`, then a test asserting `on_hud_button`/
  `on_hud_field` produces the right pending `Action` — see
  `test_skip_trade_button_submits_skip` for the shape.
- **Full-game regression**: extend `test_self_play.py`'s `SEEDS` range
  rather than adding a new fuzz test file, unless the new roster/scenario
  is genuinely different from "mixed heuristic/random agents to a winner."

## Ad-hoc verification (not checked into the suite)

A lot of validation for `risk/learning/` so far has been one-off
`python -c "..."` scripts run during development rather than checked-in
tests — building a real game rollout, exercising the module under test
against it, and asserting specific claims (shapes, value cross-checks
against the source `State`/`Action` objects, "this row's stage matches
`done`," etc.). This is deliberate while these modules' shapes are still
changing every session (per `Docs/ChangeLog.md`) — a
checked-in test for an API that gets redesigned a few messages later is
wasted upkeep. Once a module's shape stabilizes (the way `Environment`'s
has), promote its verification scripts into real `Temp/tests/` files
following the conventions above.
