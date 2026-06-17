# RL-prep pass: bug fix + class-based cleanup

Context: preparing `risk/learning/` for a GCN+DQN training loop. The game
engine/UI was already solid and well-tested (204 tests, all passing before
and after this pass); the changes below are a bug fix plus a consistency
pass converting free-function modules into classes, per request.

## Bug fix (the important one)

**`risk/learning/self_play.py` — `result.state` aliased the same mutable
object for the whole game.**

`Environment` keeps one `State` object per game and mutates it in place on
every `step()` (see `risk/game/environment.py`); it never creates a new
`State` until `reset()`. `StepResult.state` is just a reference to that same
object. `play_headless`/`play_rendered` already snapshotted the **before**
state for the trainer's `on_step` hook, but handed back the raw, mutable
**after** state (`result.state`) unsnapshotted.

For a DQN replay buffer, this is silent corruption: every transition you
store keeps a reference to the *same* `State` object, which keeps getting
mutated as the game continues. By the time the episode ends, every stored
`next_state` in your buffer would point at the same final, fully-mutated
state — not the state that actually followed each action. The bug wouldn't
raise an error or crash; it would just quietly poison the training data.

Verified with a real fix + a regression check: ran 300 self-play steps and
asserted every `result.state` handed to `on_step` is a distinct object
(`risk/learning/self_play.py`, `SelfPlay.play_headless`/`play_rendered` now
do `result.state.snapshot()` before invoking `on_step`).

## Other small fixes while in there

- `risk/game/actions.py`: removed `_REGISTRY`, a module-level dict that was
  built but never read (dead code) and an unused `field` import.

## Class-based refactor

You asked for the free-function modules across `risk/` to become
class-based, ahead of building the trainer on top. Converted, with all
call sites and tests updated (full suite still green):

| Before (free functions) | After |
|---|---:|
| `risk.learning.self_play.play_headless` / `play_rendered` | `SelfPlay.play_headless` / `SelfPlay.play_rendered` (classmethods — subclass `SelfPlay` for the trainer rather than reimplementing the loop) |
| `risk.app.factory.build_game` / `build_agents` | `GameFactory.build` / `GameFactory.build_agents` |
| `risk.app.setup.default_settings` / `run_setup` | `SetupStage.default_settings` / `SetupStage.run_setup` |
| `risk.game.card.is_valid_set` / `find_valid_set` / `validate_against_topology` | `CardRules.is_valid_set` / `CardRules.find_valid_set` / `CardRules.validate_against_topology` |
| `risk.game.actions.action_from_dict` | `ActionCodec.from_dict` |
| `risk.game.actions._check_territory` (module function) | `Action._check_territory` (staticmethod on the base class) |
| `risk.app.marker.action_territories` / `describe_action` / `action_report` | folded into the existing `ActionMarker` class as staticmethods/classmethod |
| `risk.ui.input.hit_test._point_in_polygon` | folded into `TerritoryHitTester` as a staticmethod |
| `risk.game.constants.card_set_value` / `starting_armies_for` | `RuleConstants.card_set_value` / `RuleConstants.starting_armies_for` |

No back-compat aliases were kept for the old names — call sites and tests
were updated directly, to avoid leaving two ways to call the same logic
("simple and light" cuts against redundant API surface).

### Two deliberate exceptions (left as functions)

- **`risk/app/main.py`** (`run`, `main`, `_run_headless`): this is the
  argparse-based CLI entry point. Wrapping an `argparse` script in a class
  adds ceremony without behavior change and isn't idiomatic for a CLI
  entry — left as-is, just updated internally to call `GameFactory.build`
  / `SetupStage.default_settings` / `SetupStage.run_setup`.
- **`risk/ui/render/init_screen_view.py`** (`run_init_screen`,
  `_next_unused_color`): a pygame event loop with only smoke-test coverage
  (no test drives the actual click/keyboard interactions). Refactoring its
  control flow into a class is real risk for no benefit toward the RL work;
  left untouched.

If you'd rather these two were converted as well for full consistency, say
so and I'll do it as a separate pass (smoke-tested via `--max-ticks`, since
neither has interaction-level test coverage to lean on).

### What stayed as plain module constants (not classes)

`MIN_PLAYERS`, `MAX_ATTACK_DICE`, `CARD_SYMBOLS`, etc. in
`risk/game/constants.py` stayed as plain `Final` module constants — they're
data, not behavior, and are read directly by `settings.py`, `environment.py`,
and `actions.py`. Only the two *computed* helpers (`card_set_value`,
`starting_armies_for`) moved into `RuleConstants`, since those are logic.

## Verified

- `python -m pytest Temp/tests -q` → 204 passed, 1 skipped (unchanged from
  before the refactor).
- `python -m risk.learning.self_play` → runs a full game end-to-end with the
  new `SelfPlay` API.
- `python -m risk.app.main --max-ticks 20 --skip-menu` → interactive app
  entry still boots and ticks headlessly.
- Manual check: 300-step self-play rollout, asserted every `on_step`
  `result.state` is a distinct object (the aliasing bug, confirmed fixed).

## Follow-up pass: two-file constants split

Per a later request: consolidated *all* named constants/magic numbers into
two plain modules — no classes, `from ... import *` friendly:

- **`risk/constants.py`** — game rules. Replaces the old
  `risk/game/constants.py` (deleted): `MIN_PLAYERS`, `MAX_ATTACK_DICE`,
  `CARD_SYMBOLS`, `card_set_value()`, `starting_armies_for()`, etc. — now
  plain module functions again (the `RuleConstants` class from the previous
  pass was undone per "don't make it class"). Also absorbed
  `ATTACKER_ROLL_EDGE` / `ROLL_OUTCOMES` (the exact Risk dice-probability
  tables), which used to live inside `risk/agents/heuristic_agent.py` —
  they're rule data, not agent-tuning data, so they belong here.
  `heuristic_agent.py` now imports them instead of redefining them, and its
  two spots that hardcoded the dice caps as literal `3`/`2` now reference
  `MAX_ATTACK_DICE`/`MAX_DEFEND_DICE` instead (was a latent duplication —
  harmless today since the rule constants haven't changed, but would have
  silently desynced if someone ever house-ruled the dice counts).
- **`risk/ui_constants.py`** — UI-only: colors, layout, fonts, pacing.
  Pulled together the *already-named* constants that were scattered across
  `risk/app/view.py`, `risk/app/loop.py`, `risk/app/main.py`,
  `risk/ui/render/panels.py`, `risk/ui/render/init_screen_view.py`,
  `risk/ui/render/risk_map.py`, and `risk/ui/input/init_screen.py`. Notably,
  `risk_map.py`'s `DEFAULT_PLAYER_COLORS` dict and `init_screen.py`'s
  `DEFAULT_COLORS` tuple were the *same six RGB values* duplicated in two
  places — now both derive from one `DEFAULT_PLAYER_COLORS` tuple in
  `ui_constants.py`. Also dropped `risk/app/loop.py`'s `END_LINGER_MS` and
  `WIN_BG`, which were defined but never read anywhere (dead code).

  Every consuming module re-imports under its original local name (e.g.
  `from risk.ui_constants import SETUP_BG as BG`), so none of the actual
  drawing code changed — only where the literal lives. Existing call sites,
  `__all__` exports, and tests were left untouched on purpose.

**Scope boundary (unchanged from before):** literal pixel offsets and font
sizes buried *inside* drawing methods (e.g. `risk_map.py`'s army-circle
radius, label fonts, continent-badge padding) were left in place. Those
aren't named constants today, and inventing names for every draw-call
literal in an already-tested-only-by-smoke-test rendering module is a much
larger, riskier sweep than what was asked for. Say the word if you want
that pass too.

Re-verified: `python -m pytest Temp/tests -q` → 204 passed, 1 skipped;
`python -m risk.app.main --max-ticks 20` and
`python -m risk.learning.self_play` both still run end-to-end.

## Follow-up pass: duplicate-code sweep

Found and merged four real duplications:

1. **"Does player X fully own continent Y" was reimplemented four times**
   with slightly different shapes — `Environment._compute_reinforcement`,
   `heuristic_agent.py`'s `ContinentAgent._territory_score`,
   `_continent_attack_value`, `_continent_defense_value`, and
   `self_play._print_player_summary`. Each rebuilt the continent's member
   indices and re-walked `state.owners` by hand. Added two methods to
   `BoardTopology` (`risk/game/board_topology.py`) — `continent_owner_counts(owners, continent, player_id)` returning `(owned, total)`,
   and `owns_continent(...)` returning a bool — backed by a
   `continent_member_indices` lookup precomputed once in `__init__` instead
   of rebuilt on every call. All five call sites now go through these two
   methods, so there's one definition of "owns a continent" instead of four
   that could quietly drift apart.
2. **`risk/app/marker.py` constructed a brand-new `BoardTopology()` — which
   re-parses the board JSON and rebuilds the whole adjacency graph — on
   every `OccupyAction`/Fortify-into description**, in two separate spots.
   Replaced with a lazily-built, module-level cached instance (the board
   never changes within a process). Matters more than it looks: this runs
   once per step in `SelfPlay.play_rendered`, so it was redundant
   JSON-parsing work in the hot path of every rendered training rollout.
3. **Near-identical `_settings()`/`_fresh_env()` test helpers** were
   hand-duplicated (with slightly different colors/seeds/agent_kind
   defaults) across `test_environment.py`, `test_agents.py`,
   `test_game_loop.py`, and `test_human_input.py`. Added
   `Temp/tests/conftest.py` with `make_settings()`/`make_env()`; each test
   file's `_settings`/`_fresh_env`/`_env` is now a one-line wrapper around
   the shared builder, preserving each file's original call signature and
   defaults exactly (no test logic changed, just where the `GameSettings`
   construction code lives).
4. **`heuristic_agent.py` hardcoded the dice caps as literal `3`/`2`** in
   `battle_win_probability`, `_attack_edge`, and `_is_full_force_attack`,
   duplicating `MAX_ATTACK_DICE`/`MAX_DEFEND_DICE` from `risk/constants.py`
   as magic numbers instead of referencing them. Replaced with the named
   constants — latent risk, not a live bug, since the rule constants
   haven't changed, but would have silently desynced if the dice counts
   were ever house-ruled.

Re-verified: `python -m pytest Temp/tests -q` → 204 passed, 1 skipped;
`python -m risk.app.main --max-ticks 20` and
`python -m risk.learning.self_play` both still run end-to-end.

**Not touched:** `Temp/tests/test_player_card_settings.py`'s `_player()`/
`_roster()` helpers build bare `Player` objects (not `GameSettings`), used
only within that one file's `Player`/`Card` unit tests — different enough
purpose from the `conftest.py` builders that folding them in would blur
what each helper is for, for no real line-count win.
