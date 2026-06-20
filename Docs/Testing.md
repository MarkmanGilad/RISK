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
| `test_self_play.py` | `SelfPlay.play_headless` — multi-seed fuzz test, full AI-only games played to an actual winner |

`risk/learning/` (the RL layer — `GraphAdapter`, `ActionGraphBuilder`,
`Encoder`, `Heads`, `GCN_DQN`, `ReplayBuffer`, `ActionEncoder`) has **no
checked-in tests yet** — see "Ad-hoc verification" below for how it's been
validated instead, and consider promoting some of those scripts into real
tests once the training loop exists and these modules stop changing shape
every session.

## Shared fixtures — `conftest.py`

`make_settings(n=3, seed=0, agent_kind="ai", human_ids=None)` and
`make_env(...)` (same args, also calls `reset()`) are the standard way to
get a `GameSettings`/`Environment` without hand-rolling `Player` tuples.
Use `human_ids={0}` etc. to mark specific seats human for UI tests. Most
test files alias these locally as `_settings()`/`_fresh_env()` with their
own defaults.

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
changing every session (per `Docs/RL-Prep-Changes.md`'s history) — a
checked-in test for an API that gets redesigned a few messages later is
wasted upkeep. Once a module's shape stabilizes (the way `Environment`'s
has), promote its verification scripts into real `Temp/tests/` files
following the conventions above.
