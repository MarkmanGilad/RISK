# RL-prep pass: bug fix + class-based cleanup

Context: preparing `risk/learning/` for a GNN+DQN training loop. The game
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

## Follow-up pass: rendered self-play last-move attribution

`SelfPlay.play_rendered` now captures the acting player id and any pending
attack before `env.step(action)`. This matches `AppLoop._apply` and matters
for turn-ending actions like skip-fortify: `Environment.step` mutates the
live `State` in place and immediately advances to the next player, so using
`state.current_player_index` after the step could label the previous
player's action as the new current player's "Last move" in the training HUD.

Added a regression in `test_self_play.py` that forces a skip-fortify turn
advance and asserts the rendered description still names the player who
actually acted.

## Follow-up pass: multi-step reinforcement (DQN action-space prep)

Context: designing how to feed actions into a DQN ([Docs/Action.md](Action.md)'s
"Representing actions for DQN" section) surfaced that `ReinforcementAction`
was the one combinatorial action in the game — it required placing the
*entire* turn's budget in a single action (`total == budget`, exactly),
across any split of owned territories. That doesn't fit the
per-candidate-`Q(s,a)` scoring design: there's no small, bounded way to
enumerate "every possible split."

**Rule change — reinforcement is now multi-step.** A `ReinforcementAction`
may place *up to* the remaining budget (`total <= budget`, not `==`); the
environment decrements the budget instead of zeroing it, and only advances
`REINFORCE → ATTACK` once the budget reaches exactly `0`. Placing the whole
budget in one action (the only thing the human UI ever does) still ends
the phase in a single step, unchanged — multi-step is an option the
*engine* now permits, not something every caller has to use.

- `risk/game/environment.py` — `_apply_reinforce`: `>` instead of `!=` for
  the over-budget check; budget decrements; phase transition is now
  conditional on `budget == 0`.
- `risk/game/actions.py` — `ReinforcementAction.validate_against`: same
  `>` relaxation, so the action's own validation agrees with what
  `Environment` enforces instead of being stricter than the real rule.
- `risk/agents/human_agent.py` — `_is_legal_reinforce` had its *own* exact-
  match check (a generic submission safety-net used by `act()`/`submit()`,
  separate from the click-driven UI). Relaxed the same way, or it would
  have rejected otherwise-legal partial reinforcements submitted outside
  the UI's own flow.
- **`HumanInputController._submit_reinforce` (the actual mouse-driven UI)
  was deliberately left untouched** — it still requires
  `placed == budget` before enabling "Place Armies," so human play is
  unaffected. Added an end-to-end test driving a human agent through the
  real click → submit → `act()` → `env.step()` path confirming it still
  reaches `ATTACK` with `budget == 0` in one step.

**Bounding the AI candidate set.** Naively letting `legal_actions()` offer
every integer amount (1..budget per owned territory) would scale with army
count instead of board size — a budget of 100 would mean ~100× more
reinforcement candidates than a budget of 3, for the same ~20 owned
territories. Instead, `Environment._legal_reinforce` now yields a bucketed
amount per territory — **`1`, `budget // 2`, `budget`** (deduplicated, so a
budget of 1 yields a single candidate, not three identical ones). Candidate
count for `REINFORCE_PLACE` stays `O(owned territories)` regardless of army
count; any other split is still reachable over a few steps now that
reinforcement is multi-step.

Updated [`Action.md`](Action.md)'s `ReinforcementAction` section and its
DQN tuple table/notes to match (dropped the now-stale "amount is constant
this step" caveat — `n` is a real per-step choice now).

**Verified:** added regression tests in `test_environment.py` (partial
placement stays in `REINFORCE` and decrements budget; full-budget-in-one-
step still ends the phase; over-budget placement still raises; the bucketed
`legal_actions()` output matches `{1, budget//2, budget}` and stays bounded
by `territories × buckets`) and the human end-to-end test in
`test_human_input.py` described above. `python -m pytest Temp/tests -q` →
208 passed (4 new), 1 skipped. `python -m risk.learning.self_play` and
`python -m risk.app.main` both still run end-to-end (self-play step counts
shift under the same seed vs. before, expected: `RandomAgent` now samples
from a richer bucketed reinforcement candidate set, changing its RNG
consumption pattern — not a bug).

## Follow-up pass: action representation for DQN (implementation)

[`Action.md`](Action.md)'s "Representing actions for DQN" section described
the `(stage, t1, t2, n)` tuple design; this pass implements it.

- **`risk/game/actions.py`** — added an `ActionStage` `IntEnum`
  (`TRADE_IN=0, REINFORCE_PLACE=1, ATTACK=2, OCCUPY=3, FORTIFY=4`) and
  `Action.NONE_INDEX = -1` (the sentinel for an unused `t1`/`t2` slot).
  Every concrete `Action` subclass got a `stage: ClassVar[ActionStage]` and
  a `dqn_index(topology, state=None) -> tuple[int, int, int, int]` method —
  plain Python, no torch import added to the game core (mirrors how
  `to_dict()` already works: each class knows how to serialize itself).
  Two subtleties handled there rather than glossed over:
  - `OccupyAction.dqn_index` needs `state` (not just `topology`) — its
    `from`/`to` territories live on `state.pending_attack`, not on the
    action itself. Raises clearly if `state` is omitted or has no pending
    attack.
  - `ReinforcementAction.dqn_index` raises if `placements` spans more than
    one territory — that shape has no representation in this tuple by
    design (`legal_actions()` never produces it; multi-territory splits
    are a candidate-set design choice, not a tuple-encoding gap).
- **`risk/learning/action_encoder.py`** (new) — `ActionEncoder`, the class
  that batches a list of `Action`s into tensors: `encode_one` (single
  tuple), `encode_many` (`[N, 4]` long tensor), and `encode_legal(env)` —
  the convenience path that calls `env.legal_actions()` and encodes the
  result in one call, passing both `topology` and `state` through
  automatically so `OccupyAction` candidates encode correctly without the
  caller having to remember why.

**Verified against real game states**, not just isolated unit tests — drove
an `Environment` through all 5 stages and checked every encoded row against
a manually-computed expected tuple:
- `REINFORCE`/`TRADE_IN` mixed candidates (both action types appear
  together during `REINFORCE` when the hand holds cards).
- `ATTACK`, including the `StopAttackAction` sentinel row `(ATTACK, -1, -1, 0)`.
- `OCCUPY`, confirming `pending_attack.from_index`/`to_index` come through
  correctly.
- `FORTIFY`, both the skip sentinel and a real move.
- Both intentional error paths (`OccupyAction` without `state`,
  multi-territory `ReinforcementAction`) raise as designed.

`python -m pytest Temp/tests -q` → 208 passed, 1 skipped (unchanged — this
pass added no test-suite regressions; the verification above was done as
standalone scripts against live `Environment` instances rather than
checked-in tests, since it's exercising a not-yet-consumed encoding layer
rather than existing game behavior).

Updated [`Action.md`](Action.md) and [`README.md`](../README.md) (the
`risk/learning/` folder tour and the Roadmap) to reference the real
`ActionStage`/`Action.dqn_index`/`ActionEncoder` names instead of reading
as an open proposal.

## Follow-up pass: `Phase`/`ActionStage` made 1:1 (`TRADE_IN` / `REINFORCE_PLACE`)

Context: while designing the RL replay buffer/agent on top of `Action.md`'s
`(stage, t1, t2, n)` encoding, it became clear `ActionStage` (which head
scores a candidate) wasn't actually a deterministic property of `Phase`
(the state's turn-segment) — every phase except `REINFORCE` was already
1:1, but `Environment.legal_actions()` could return `TRADE_IN` and
`REINFORCE_PLACE` candidates *together* in one call whenever the hand
wasn't full, and this was directly observed in a verification rollout
(mixed decisions, 18-58 candidates, at multiple steps). The fix: a player
is always in exactly one phase, and an explicit action ends it and
advances to the next — the same discipline `StopAttackAction` and skip-
`FortifyAction` already enforce for `ATTACK`/`FORTIFY`. This costs no
expressiveness: a hand can only shrink during one continuous `REINFORCE`
segment (new cards only arrive after a conquest, for a *future* turn), so
forcing "resolve all desired trades first, then place" can't prevent any
sequence of moves a player could already make.

**`risk/game/phase.py`** — `Phase.REINFORCE` split into `Phase.TRADE_IN` +
`Phase.REINFORCE_PLACE`. Every value except `SETUP` changed (a one-time
break to the "stable for serialization" guarantee; no persisted save files
exist in the repo). New order mirrors `ActionStage` with `SETUP`/
`GAME_OVER` bookends: `SETUP, TRADE_IN, REINFORCE_PLACE, ATTACK, OCCUPY,
FORTIFY, GAME_OVER`.

**`risk/game/actions.py`** — `TradeInAction.phase` → `TRADE_IN`,
`ReinforcementAction.phase` → `REINFORCE_PLACE`. New `SkipTradeAction`
(modeled exactly on `StopAttackAction`): no fields, `phase=TRADE_IN`,
`dqn_index()` returns the same `(-1, -1, 0)` sentinel pattern as the other
skip/stop actions. Registered in `ActionCodec.from_dict` (`"skip_trade"`).

**`risk/game/environment.py`**:
- `legal_actions()`'s old combined `REINFORCE` branch split in two:
  `TRADE_IN` returns every valid `TradeInAction` plus a `SkipTradeAction`
  (omitted only when the hand is full and trading is mandatory);
  `REINFORCE_PLACE` returns `_legal_reinforce(s)` alone.
- New `_apply_skip_trade`: `phase = REINFORCE_PLACE`.
- `_apply_trade_in`'s phase guard: `REINFORCE` → `TRADE_IN` (still doesn't
  auto-advance — multiple trades stay legal within `TRADE_IN`).
- **`_apply_reinforce` gained a phase guard it didn't have before**
  (`REINFORCE_PLACE` required). This goes a little beyond a pure rename:
  under the old combined phase there was no separate stage to bypass, but
  once `TRADE_IN` became a real phase you're meant to explicitly leave,
  *not* guarding `_apply_reinforce` would let a `ReinforcementAction`
  silently skip `TRADE_IN` entirely — exactly the loophole this pass exists
  to close.
- `_enter_reinforce_for` renamed to `_begin_turn_for` (now sets
  `Phase.TRADE_IN`; every turn starts there, even with nothing to trade).

**Agents/UI** — `risk/agents/heuristic_agent.py` (trade-or-skip on
`TRADE_IN`, separately from `_reinforce()` on `REINFORCE_PLACE`),
`risk/agents/human_agent.py`, `risk/ui/render/panels.py`, and
`risk/agents/human_input.py` (split the single REINFORCE HUD panel into a
`TRADE_IN` panel with a new "Skip Trading" button and a `REINFORCE_PLACE`
panel with the existing placement UI) all updated to the new phase names
and the new button.

**`risk/learning/graph_adapter.py`** needed no code change — `u`'s phase
one-hot is already `[0.0] * len(Phase)`, so it grew 6→7 wide (`u`: 33→34)
for free. `risk/learning/gnn_dqn.py`'s `TRADE_IN` path (predates this
pass) does `a.card_indices for a in trade_actions`, which breaks once a
`SkipTradeAction` (no `card_indices`) appears in that list — flagged, not
fixed, since that file is already slated for the "pure net + agent"
redesign from a separate, not-yet-built piece of work; it'll naturally use
`action.dqn_index(...)`'s sentinel instead of raw `.card_indices` when that
redesign happens, fixing this for free.

Updated [`Action.md`](Action.md) (new `SkipTradeAction` rows, the
now-accurate "`Phase`/`ActionStage` are 1:1" framing, removed the
no-longer-true "mixed-stage batches" notes), [`NetworkArchitectures.md`
](NetworkArchitectures.md), and [`GraphAdapter.md`](GraphAdapter.md) (`u`
width/slice table) to match.

**Verified:**
- `python -m pytest Temp/tests -q` → 215 passed, 1 skipped (up from 208 —
  7 new tests: `SkipTradeAction` construction/round-trip, a
  `TRADE_IN -> REINFORCE_PLACE` transition test mirroring
  `test_fortify_skip_advances_turn`, a `_apply_reinforce` out-of-phase
  rejection test, a `TRADE_IN`-only `legal_actions()` test split out
  alongside the existing `REINFORCE_PLACE`-only one, and the new
  "Skip Trading" HUD button test).
- A real 500-step `RandomAgent` self-play rollout: every single
  `legal_actions()` call returned candidates of exactly one `ActionStage`
  (asserted explicitly, not just exercised) — confirms the mixing is
  actually gone, not just in the unit tests. All 5 stages were exercised.
- `GraphAdapter` re-verified end-to-end: single snapshot and a 3-game
  `Batch.from_data_list` both produce the new `u` width (`[1, 34]` /
  `[3, 34]`), `Data.validate()` passes.

## Follow-up pass: merge `ActionStage` into `Phase`, and `Action.stage` into `Action.phase`

Context: building the replay buffer's `stage`/`next_stage` tensors (previous
entry) surfaced that `Phase` and `ActionStage` had become pure duplicates —
once `REINFORCE` split into `TRADE_IN`/`REINFORCE_PLACE`, `ActionStage`
stopped encoding anything `Phase` doesn't already carry; it was just
"`Phase` minus `SETUP`/`GAME_OVER`," kept as a separate `IntEnum` numerically
offset by 1 from `Phase` (`Phase.ATTACK=3` vs. old `ActionStage.ATTACK=2`)
— exactly the kind of silent-bug-prone duplication this file's other
entries have been trimming. Checking further found every `Action` subclass
already carried **both** `phase: ClassVar[Phase]` (the original attribute,
documented in `Action.md`'s opening paragraph) **and**
`stage: ClassVar[ActionStage]` (added for DQN encoding) set to the same
conceptual value — identical once the types merge.

**`risk/game/phase.py`** — renumbered a second time:
`TRADE_IN=0, REINFORCE_PLACE=1, ATTACK=2, OCCUPY=3, FORTIFY=4, GAME_OVER=5,
SETUP=6` (previously `SETUP=0, TRADE_IN=1, ..., GAME_OVER=6`). Chosen so
`done == (phase == GAME_OVER)` is a clean invariant; `SETUP` sits last since
it's never actually observed during play — only via `State.initial()`
directly (confirmed via `Grep` before making the change: `state.py`'s
`State.initial()` and `test_state.py` are the only two hits).

**`risk/game/actions.py`** — deleted `ActionStage`, `_PHASE_TO_STAGE`, and
`stage_for_phase()` entirely. Every `Action` subclass's `stage:
ClassVar[ActionStage] = ActionStage.X` line removed, keeping only
`phase: ClassVar[Phase]`; every `dqn_index()` method's `int(self.stage)` →
`int(self.phase)`. Confirmed via `Grep` before changing: no runtime code
read `action.phase`/`action.stage` outside tests — `Environment`/agents
dispatch on `isinstance`, never on these — so repurposing `.phase` as the
sole attribute was safe.

**`risk/learning/{heads,action_graph_builder,gnn_dqn}.py`** — mechanical:
`ActionStage` imports/references → `Phase`, `a.stage` → `a.phase`.

**`risk/learning/replay_buffer.py`** — deleted `_stage_or_sentinel()` and
the `stage_for_phase` import; `push()` now stores `int(action.phase)`/
`int(next_state.phase)` directly, no sentinel needed at all — `GAME_OVER`'s
own value (`5`) already distinguishes it from every real decision phase,
and those rows are `done` anyway.

Updated [`Action.md`](Action.md) (the `(phase, t1, t2, n)` tuple — renamed
from `(stage, ...)` — and the stage table's prose, table *numbers*
unchanged since the new `Phase` 0-4 values match the old `ActionStage` 0-4
exactly by construction), [`NetworkArchitectures.md`](NetworkArchitectures.md),
[`GraphAdapter.md`](GraphAdapter.md) (phase one-hot column order), and
[`README.md`](../README.md) (its `Phase` description line was still
showing the *original pre-split* values — `SETUP, REINFORCE, ATTACK,
FORTIFY, GAME_OVER, OCCUPY` — missed during the `TRADE_IN`/
`REINFORCE_PLACE` split two entries up; fixed now too).

**Verified:**
- `python -m pytest Temp/tests -q` → 225 passed, 1 skipped (one test
  rewritten: `test_phase_is_ordered`'s old single `<` chain across all 7
  values no longer holds — `SETUP` deliberately sorts last now, not
  chronologically first — split into "the 6 in-game-flow values are
  ordered" plus a separate assertion that `SETUP` sorts after `GAME_OVER`).
- A real 5000-step mixed heuristic/`RandomAgent` rollout to an actual
  `GAME_OVER` (752 steps): every `legal_actions()` call asserted
  single-stage, `stage`/`next_stage` cross-checked row-for-row against
  `action.phase`/`next_state.phase`, and the one `GAME_OVER` row's
  `next_stage` (`5`) lines up with `done=True` — confirms the sentinel
  removal is correct, not just type-checked.

## Follow-up pass: `GNN_DQN` split into a pure net + (not yet built) agent

Context: `GNN_DQN` originally owned `ActionGraphBuilder`, built/batched
candidate graphs, looped over legal actions, and dispatched to a stage-
keyed `Heads` `ModuleDict` — all inside `nn.Module.forward`. That's game
logic and Python-level branching/looping living inside a PyTorch network,
which doesn't belong there (mirrors `Temp/Examples/DQN_Agent.py`'s own
split: a bare `DQN` network wrapped by a `DQN_Agent` that owns epsilon-
greedy, batching, and argmax — the network itself never sees an "action").

**`risk/learning/heads.py`** — replaced the single `Heads` class (a
`ModuleDict` keyed by stage name, with a `forward(stage, g)` dispatch
method) with two classes, each a `g_dim`-only constructor:
- `ScoringHead` — the generic 3-layer MLP, instantiated 4 times under
  named attributes (`reinforce_place_head`, `attack_head`, `occupy_head`,
  `fortify_head`) rather than 4 near-duplicate class bodies.
- `TradeInHead` — kept separate (different input shape: pooled base
  context + card embeddings, not a per-candidate `g`). Its embedding table
  gained one extra row for the `SkipTradeAction` sentinel — `dqn_index()`'s
  `(-1, -1, 0)` was previously fed straight into `nn.Embedding`, which
  raised (`-1` is an invalid index) the moment a real game state offered a
  `SkipTradeAction` alongside `TradeInAction`s (always, except when trading
  is mandatory). Detected per *row* (`t1 < 0`), not per element — `n == 0`
  is `SkipTradeAction`'s placeholder but a real card-slot index for
  `TradeInAction`, so per-element masking would have silently mis-embedded
  that slot instead of raising. This was flagged as a known gap two
  entries up ("flagged, not fixed") and surfaced for real the moment this
  pass's own verification script exercised a `TRADE_IN` decision.

**`risk/learning/gnn_dqn.py`** — `GNN_DQN.forward` settled on
`(state, phase, card_indices) -> Q`, all 5 stages in one call:
- `state` — always a `Batch`, one graph per row, even for `N=1`
  (`Batch.from_data_list([one_graph])`) — same convention as carrying a
  batch dimension of 1 in any other PyTorch net; `forward` doesn't
  detect/special-case a bare single `Data` (an earlier version did, via a
  `getattr(state, "batch", None)` fallback — removed by request: "let the
  agent deal with it," not the net). *Injected* (`ActionGraphBuilder`) for
  the 4 graph-based stages; for `TRADE_IN` it's just `GraphAdapter`'s bare
  base graph, since none of its candidates perturb the graph — `forward`
  still defaults a missing `edge_attr` to zero itself, so the caller
  doesn't have to build one just for `TRADE_IN`'s graph shape.
- `phase` — `[N]` long, one `Phase` value per row, same convention as
  `ReplayBuffer`'s `stage`/`next_stage`.
- `card_indices` — `[N, 3]` long, each row's `dqn_index()` `(t1, t2, n)`;
  needed whenever any row is `TRADE_IN`. No explicit guard for a missing
  `card_indices` — trusts the caller, same as the rest of this codebase's
  internal call sites; omitting it with a `TRADE_IN` row present fails
  naturally (`TypeError` indexing `None`) rather than via a custom check.

Internally: `Encoder` + `pool` once for the whole batch regardless of
stage mix, then a plain `dict` lookup (built once in `__init__`, aliasing
the same submodule instances already registered as named attributes — not
a second registration) over the fixed 5 decision phase/head slots, scoring
only rows whose mask is present.

This went through two intermediate states first: initially `forward` did
only `(x, edge_index, edge_attr, batch, u) -> g` with zero `Phase`
awareness, heads called externally; then `(state_action, phase) -> Q` for
the 4 graph-based stages only, with `TRADE_IN` rejected and still scored
via a separate `trade_in_head(...)` call. Folding `TRADE_IN` in too (so
the agent never special-cases it) needed only a `card_indices` parameter
and the `edge_attr` default — `state_action` was renamed to `state` in
the same pass, since for `TRADE_IN` rows it's never actually an injected
"state+action" graph, just the bare state. `state`'s type also briefly
accepted `Union[Batch, Data]` with a `getattr(state, "batch", None)`
fallback for a bare single graph — removed in favor of always requiring a
`Batch` (even `Batch.from_data_list([one_graph])` for `N=1`), matching how
every other PyTorch net carries a batch dimension unconditionally rather
than detecting and special-casing the unbatched case itself. The only
game-adjacent import
stays the one leaf `Phase` enum (no dependencies of its own) — not
`Action`/`ActionGraphBuilder`/`Environment` — same category of import
`heads.py` already makes for `MAX_CARDS_IN_HAND`. The *agent*'s job
shrinks further: build the graphs/`phase`/`card_indices` tensors, one
`forward` call, merge results back, `argmax` — no head-picking by hand
for any stage anymore.

The `edge_attr` default itself (zero-filled when missing, needed because
`GraphAdapter`'s bare base graph had no `edge_attr` of its own) was also
removed, by the same "let the builder deal with it, not the net" request
— **`risk/learning/graph_adapter.py`** now gives every base graph a
zero-filled `edge_attr` (`EDGE_ATTR_DIM` wide) up front, instead of
`forward` synthesizing one lazily every call. `EDGE_ATTR_DIM` moved from
`action_graph_builder.py` to `graph_adapter.py` (the "lower" module in the
dependency graph — `action_graph_builder.py` already imports
`armies_column_index` from it, so the reverse import would've been
circular) and `ActionGraphBuilder` now clones `base.edge_attr` instead of
building a fresh zero tensor itself. Net effect: `edge_attr` exists on
*every* graph from the moment `GraphAdapter` builds it, so neither
`ActionGraphBuilder` nor `GNN_DQN.forward` ever has to construct or
default one — each just inherits/overwrites what's already there.

One more simplification on top, by request: **`card_indices` made
required, not `Optional`** — every row carries `[N, 3]` `card_indices`
unconditionally now, zeros (or anything; ignored) for rows that aren't
`TRADE_IN`, rather than `None` whenever no `TRADE_IN` row happens to be
present. That, plus giving **`risk/learning/heads.py`'s `ScoringHead`** a
`(g, card_indices)` signature (ignoring the second argument — kept purely
so it matches `TradeInHead`'s), let `GNN_DQN.forward`'s two-path dispatch
(a `TRADE_IN`-specific branch, then a loop over the other 4) collapse into
**one loop, no branching**: `TRADE_IN` joins `_heads_by_phase` like every
other stage, and `self._heads_by_phase[stage](g[mask], card_indices[mask])`
is the entire dispatch, regardless of which of the 5 heads `stage` picks
out. The motivating idea: once every head shares one call shape, "which
head" becomes a plain dict lookup keyed by the `phase` tensor, with
nothing left for `forward` to special-case.

Updated [`NetworkArchitectures.md`](NetworkArchitectures.md)'s "Per-stage
heads" and "Net A" sections to match (split into "the net" vs. "the
not-yet-built agent" explicitly, rather than describing one class that
does both).

**Verified:** no checked-in tests exist for `risk/learning/` yet
(`Docs/Testing.md`), so this was exercised via ad-hoc rollout scripts
(`Docs/Testing.md`'s documented practice for these still-shifting
modules):
- `GraphAdapter`'s bare output: `edge_attr` present and all-zero
  (`Data.validate()` passes), confirming the new field exists from the
  moment a base graph is built, not just after injection.
- A real 400-step game, manually playing the agent's role (build graphs
  via `ActionGraphBuilder` for graph-based stages, the bare base graph for
  `TRADE_IN`, batch, call `net(state, phase, card_indices=...)`, `argmax`)
  — every stage scored correctly including the `TRADE_IN` sentinel row
  through `TradeInHead`'s dedicated "none" embedding, gradients confirmed
  flowing every step.
- A fabricated **fully mixed batch** — `TRADE_IN` rows (including a
  `SkipTradeAction` sentinel) *and* graph-based rows from a different
  decision, scored together in one `forward` call, the shape a sampled
  replay-buffer minibatch would actually be — confirms the per-label
  routing handles a genuine mix, not just one stage at a time.
- An `N=1` call via `Batch.from_data_list([one_graph])` (the one-decision
  "play" shape) — confirms scoring a single candidate needs no special
  casing now that `state` is unconditionally a `Batch`.
- The fully mixed batch above and the 400-step rollout both pass
  `card_indices` unconditionally — real `(t1, t2, n)` values for `TRADE_IN`
  rows, all-zero `(0, 0, 0)` placeholders for every graph-based row —
  confirming the uniform `(g, card_indices)` head call shape scores
  correctly either way, with the same loop, no branch on which kind of row
  it is.

Full suite re-run after each step: 225 passed, 1 skipped, unaffected (no
existing test touches these modules).

## Follow-up pass: fixed five-stage `GNN_DQN` routing loop

Per a later cleanup request, `risk/learning/gnn_dqn.py` now routes rows by
looping over `range(5)` instead of `phase.unique().tolist()`. The `5` is
the fixed count of DQN decision phases/heads (`TRADE_IN`,
`REINFORCE_PLACE`, `ATTACK`, `OCCUPY`, `FORTIFY`), and each iteration skips
the head call when that phase is absent from the batch.

This keeps the same masking behavior (`phase == stage`) and output shape,
but makes the routing order explicit and independent of which phase labels
happen to appear in a particular minibatch. `NetworkArchitectures.md` was
updated to describe the fixed five-slot loop.

## Follow-up pass: `GNN_DQN_Agent` (inference path only)

Added `risk/learning/gnn_dqn_agent.py` implementing the not-yet-built
agent wrapper described in `Docs/NetworkArchitectures.md`.

Implemented now:

- `act(events, state)` for `BaseAgent` compatibility.
- Legal action enumeration from `env.legal_actions(state)`.
- Per-action graph row build:
  - `ActionGraphBuilder` for graph-based stages.
  - unmodified base graph rows for `TRADE_IN` (no injection).
- `(phase, t1, t2, n)` batching via `ActionEncoder`.
- One batched `GNN_DQN` forward call + argmax selection.
- Optional epsilon exploration (`train_mode` + `epsilon`).
- Owned `ReplayBuffer` and a `remember(...)` helper for transition storage.
- Agent always constructs its own `GNN_DQN` from the current graph dimensions.
- `load_params(path)` restores a saved net `state_dict` via `torch.load(...)`.

Deferred intentionally:

- `train_step(...)` currently raises `NotImplementedError`; update logic,
  losses, and target-net synchronization are for a later pass.

Regression coverage added in `Temp/tests/test_agents.py` for:

- argmax action selection against a stub net.
- `TRADE_IN` handling without action-graph injection errors.

## Follow-up pass: learner-elimination early stop in self-play

Added an opt-in early-stop control for training episodes where "my learner
lost" should end the rollout immediately, without waiting for full game
termination:

- `SelfPlay.play_headless(..., stop_when_player_eliminated=pid)`
- `SelfPlay.play_rendered(..., stop_when_player_eliminated=pid)`

When this argument is set, the loop exits as soon as `pid` appears in
`state.eliminated` (checked both at loop start and right after each step).
Return type is unchanged (`env.winner()`), so early-stopped episodes usually
return `None` while `env.is_terminal()` stays `False`.

Regression coverage added in `Temp/tests/test_self_play.py`.

## Follow-up pass: self-play scratch-pad learner seat

Updated `risk/learning/self_play.py`'s `main()` scratch-pad roster to use
five players, with seat 4 now assigned an untrained `GNN_DQN_Agent`.
The scratch-pad step-2 comment now matches that flow directly ("add your
agent to the list") so you can run the learner seat as-is and swap in a
different learner class by editing a single line.

## Follow-up pass: runtime device selection/check for learner

`GNN_DQN_Agent` now resolves device in the learning layer (not global
constants): explicit override if provided, otherwise `cuda` when available,
else `cpu`.

Added `GNN_DQN_Agent.device_report(...)` to verify runtime placement for
model and tensors (`batch`, `phase`, `card_indices`), and `self_play.py`
prints that report in `main()` when constructing the learner seat.

`score_actions(...)` now asserts model/tensors are on the selected device,
failing fast on CPU/GPU mismatch instead of silently mixing devices.

## Follow-up pass: `GNN_DQN_Agent.train_step` (Net A training)

Implemented the previously-deferred DQN update, closing out Net A's
"Confirm it plays a legal game, then train" step in
`NetworkArchitectures.md`'s experiment plan.

- `GNN_DQN_Agent.__init__` gained `gamma`, `lr`, `target_update_every`
  kwargs and now owns an `Adam` optimizer over `self.net.parameters()`.
- `train_step(batch_size)` samples a minibatch from `self.replay_buffer`,
  computes `Q(s, a)` for the taken actions via the online net (`_q_value`),
  computes the Double-DQN next-state value via `_max_next_ddqn_q` (online
  net selects the best legal next action, target net evaluates it), and
  takes one Huber-loss gradient step with gradient clipping. Target net is
  hard-synced to the online net every `target_update_every` calls (chosen
  over a soft/Polyak update, per request).
- `_max_next_q`/`_max_next_ddqn_q` candidate count varies per transition (`next_state`'s
  own `legal_actions()` count), so every transition's candidates are
  built/batched/scored across the whole minibatch — avoids one forward pass
  per transition. The original `_max_next_q` keeps the target-net max path
  for easy rollback; `_max_next_ddqn_q` scores candidates with both online
  and target nets, then takes the target-net value at the action selected
  by the online net. Rows for `done` transitions are skipped entirely
  (their target contribution is `0` by construction).
- `_q_value` builds one graph row per `(state, action)` pair directly
  (`ActionGraphBuilder`/`ActionEncoder.encode_one`), rather than going
  through `ActionEncoder.encode_many`/`score_actions`'s "one state, many
  candidate actions" shape — each replay transition has its own `state`,
  not one shared state for the whole minibatch.

**Verified** against two real self-play rollouts (untrained learner seat,
`epsilon`-greedy exploration during collection): `train_step` runs
end-to-end over five calls with a populated replay buffer, loss is a finite
scalar each call; a second rollout explicitly checked that the online net's
parameters change after `train_step` and that the target net's parameters
exactly match the online net's immediately after a sync at
`target_update_every`, confirming the hard-sync timing is correct rather
than just not-erroring.

## Follow-up pass: perspective-relative encoding + `Trainer` (multi-episode self-play)

Context: the next step toward actually training Net A was a real
multi-episode self-play loop — every episode reassigns the learner to a
random seat among 3-6 `RandomAgent` opponents. That surfaced a real bug:
`GraphAdapter`'s owner one-hot/`u` features (`Docs/GraphAdapter.md`) are
keyed by **absolute** player id, so the same physical board position looks
like a different input depending on which seat the learner happened to be
assigned that episode — the net would have to re-learn "which absolute id
is mine" every time its seat changed, instead of learning the game itself.

**`GraphAdapter.__call__`/`_node_features`/`_global_features`** gained a
`perspective: int = 0` parameter that rotates every player-indexed slot
(owner one-hot, cards-per-player, current-player one-hot, eliminated) so
`perspective` always lands in relative slot `0`, preserving turn order
(`(p - perspective) % n_players`). Default `0` is a no-op, so this is fully
backward compatible. Verified against the project's own example ordering
(4-player game, training agent physically seated at `2`): absolute seats
`(0,1,2,3)` → relative slots `(2,3,0,1)`, i.e. training agent first, then
its turn order continuing around the table — exactly the "training, then
the next 3 seats in order" reordering asked for.

**Found and fixed a related latent bug while wiring this up:**
`n_players` inside `GraphAdapter` was read from `self.settings.player_count`
— harmless under the original "one adapter per game" assumption, but wrong
the moment one `GraphAdapter` gets reused across episodes of different
sizes (exactly what the new trainer does): scoring an older replay-buffer
`state` against the *current* episode's `n_players` indexed the wrong
number of hand/owner slots (`IndexError`, caught immediately by the
verification rollout below). Fixed by reading `n_players` from
`len(state.hands)` instead — a property of the state itself, not whichever
episode the adapter instance currently happens to be bound to.

**`GNN_DQN_Agent.remember`**, not `ReplayBuffer`, owns capturing
`perspective` — needed because, unlike `stage` (derivable from
`action.phase`), the learner's seat at push time isn't derivable from
`state`/`action` at all, and isn't guaranteed to match the learner's
*current* seat by the time a transition gets sampled for training (the
trainer reassigns seats every episode), so it has to be captured eagerly,
per transition. It isn't a `ReplayBuffer` column at all, though:
`remember()` now does its own `state.snapshot()` (moved out of
`ReplayBuffer.push`, which dropped its internal copy and just trusts its
caller — `remember()` is the only real one) and tags the fresh copy with
`state_snapshot.perspective = self.player_id` before handing it to
`push()` (`State` is a plain, unfrozen dataclass, so this is just a normal
attribute set, invisible to `to_dict()`/`__eq__`, not a declared field).
The training loop reads it back as `state.perspective`/`next_state.perspective`
per row rather than `sample()` returning a parallel `perspectives` array —
`ReplayBuffer` itself ends up knowing nothing about "perspective" at all,
staying a generic `(state, action, reward, next_state, done, stage,
next_stage)` store, with the seat info co-located with the one `State` it
actually describes instead of riding alongside it in the container.
Confirmed this survives `ReplayBuffer.save`/`load` (`torch.save` pickles
the attribute along with everything else).

**`GNN_DQN_Agent`**:
- `attach(player_id, env)` — rebinds `player_id`/`env`/`adapter`/`builder`/
  `action_encoder` for a new episode/seat, while `net`/`target_net`/
  `optimizer`/`replay_buffer` persist — lets one learner train across many
  episodes instead of being rebuilt (and losing its weights) every time.
- Every adapter call (`score_actions`, `device_report`, `_q_value`,
  `_max_next_q`/`_max_next_ddqn_q`, the constructor's dimension probe) now passes
  `perspective` — `self.player_id` for live inference, the buffer's
  per-transition `perspective` tensor for training (`_q_value`/
  `_max_next_q`/`_max_next_ddqn_q` build each row's graph from *that transition's* seat, not
  whatever seat the agent currently happens to be attached to).
- `remember(...)` captures `perspective=self.player_id` at push time.

**`risk/learning/trainer.py`** (new) — `Trainer`, built on `SelfPlay`
(composition, not subclassing — `SelfPlay`'s own `on_step`-callback shape
already covers what a trainer needs, no override points to subclass).
Each `run_episode()`:
- random `n_players` (3-6) and a random seat, all-`RandomAgent` roster via
  `GameFactory.build`/`SetupStage.default_settings`, learner `attach`ed
  onto the chosen seat — no `HumanAgent` seats, ever.
- plays to game-over or the learner's own elimination
  (`stop_when_player_eliminated`) — once the learner is out, there's
  nothing left for it to learn from that episode.
- a sparse terminal reward on the episode's last transition only: `+1`
  win, `-1` eliminated, `0` on a max-steps timeout (`done` stays `False`
  for a timeout specifically, since that's a truncation, not a real
  terminal — the TD target should still bootstrap from the next state).
- trains every `train_every` episodes once the buffer holds at least one
  batch, and checkpoints to `checkpoint_dir` (default `Checkpoints/`)
  every `checkpoint_every` episodes starting after `checkpoint_after` —
  versioned filenames (`gnn_dqn_ep{N:06d}.pt`) keep every saved snapshot
  rather than overwriting one "latest" file, so a later tournament
  (`NetworkArchitectures.md`'s "Experiment plan" step 4) has multiple
  checkpoints along the training run to pick from, not just the last one.

**Verified:** a 6-episode smoke run (`batch_size=8`, `train_every=1`,
`checkpoint_after=2`, `checkpoint_every=2`) — every episode picked a
different `n_players`/seat combination, the replay buffer grew every
episode, `train_step` ran without error each time, and checkpoint files
landed on disk at episodes 2/4/6 as configured. Full suite re-run:
`python -m pytest Temp/tests -q` → 234 passed, 1 skipped, unaffected (no
existing test touches `risk/learning/`).

## Follow-up pass: explicit trainer loop + agent-owned ingest/schedule

Per request, `risk/learning/trainer.py` was restructured to read like the
example PPO/DQN trainers: one explicit episode loop with clear orchestration
steps (build episode context, rollout, ingest, train-if-due, checkpoint).

- `Trainer.run(...)` now delegates to `Trainer.train(...)` for a more obvious
  entry point name.
- `Trainer.run_episode(...)` is now loop-first and short; internals split into
  `_build_episode_context(...)` and `_play_episode(...)` so the control flow
  is easy to scan.
- trainer-side `_remember_episode(...)` was removed.

That removed logic moved into `risk/learning/gnn_dqn_agent.py`:

- `ingest_episode(...)` now applies the existing sparse terminal reward policy
  on the final transition and pushes transitions to replay.
- `can_train(...)`, `learn_steps(...)`, and `learn_if_ready(...)` now own the
  "is replay ready" and "train every N episodes" cadence decisions.
- `train_step(...)` is unchanged mathematically; this pass only moved
  responsibility boundaries to keep the trainer loop clearer.

No wandb/logging behavior was added in this pass, intentionally.

## Follow-up pass: `run_episode()` returns explicit episode stats

Small trainer ergonomics pass, keeping the same training behavior:

- `risk/learning/trainer.py` now defines `EpisodeStats` (`TypedDict`) and
  `Trainer.run_episode()` returns one compact summary dict instead of only a
  winner id.
- Summary fields are loop-facing only (no logging integration):
  `episode`, `n_players`, `learner_seat`, `transitions`, `winner`,
  `eliminated`, `terminated`, `losses`, `trained`, `replay_size`.
- `Trainer.train(n_episodes)` now returns a `list[EpisodeStats]` (one row per
  episode). `Trainer.run(n_episodes)` is still supported and delegates to
  `train(...)`.

This keeps the loop readable (PPO/DQN-trainer style) while making each
episode's outcome/training step visible to callers without introducing
wandb/logging yet.

## Follow-up pass: simplify trainer API to one public method

Per request, `risk/learning/trainer.py` now exposes a single public training
entry point: `Trainer.train(...)`.

- Removed `Trainer.run(...)` and `Trainer.run_episode(...)`.
- Inlined the episode flow directly inside `train(...)` so all orchestration
  is visible in one function.
- Kept the same behavior for rollout, ingest, train scheduling, checkpointing,
  and returned `EpisodeStats` rows.

This reduces surface area and keeps the training loop easy to read without
adding logging/wandb concerns.

## Follow-up pass: non-deterministic episode seeds

Per request, trainer episode construction now defaults to non-deterministic
randomness:

- `risk/learning/trainer.py` now uses `random.SystemRandom()` when `seed`
  is not provided.
- `_build_episode_context(...)` now passes `seed=None` into
  `SetupStage.default_settings(...)` so each episode's game setup is not
  pinned to a fixed integer seed.
- `main()` constructs the learner agent explicitly and passes it into
  `Trainer(...)`; episode setup randomness remains non-deterministic inside
  the trainer.

Reproducibility for learned weights/checkpoints is handled by the agent and
logger checkpoint path; episode setup intentionally remains non-deterministic
unless a future trainer option reintroduces fixed episode seeds.

## Follow-up pass: inline episode rollout inside train

Per request, `risk/learning/trainer.py` no longer has a separate
`_play_episode(...)` helper.

- Moved the `SelfPlay.play_headless(...)` call and transition collector
  closure directly into `Trainer.train(...)`.
- Kept behavior unchanged (same max steps, stop-on-elimination, and
  transition capture).

This keeps the full training flow visible in one method.

## Follow-up pass: remove trainer stats aggregation

Per request, `risk/learning/trainer.py` no longer builds or returns a per-
episode stats structure.

- Removed `EpisodeStats` from the trainer module.
- `Trainer.train(...)` now returns `None` and runs the loop only.
- Removed row assembly and "last stats" print in `main()`.
- Kept periodic progress output and checkpointing behavior unchanged.

This leaves room to add a dedicated logging/stats class later (e.g. wandb)
without mixing that concern into the core training loop.

## Follow-up pass: fix missing stop-on-elimination in inlined loop

The "inline episode rollout inside train" pass above stated stop-on-elimination
was unchanged, but the actual manual loop in `Trainer.train(...)` never
checked it — it only broke on `result.done` (a real game-over). Once the
learner's seat was eliminated, the inner per-opponent `while` loop kept
running indefinitely (its only exit condition was `result.done`), so episodes
ran until the *entire game* ended rather than stopping at the learner's own
elimination, regularly chewing through all of `MAX_STEPS_PER_EPISODE`.

Fixed in `risk/learning/trainer.py`'s `train(...)`:
- outer loop now breaks immediately if `seat in env.current_state().eliminated`.
- inner opponent-turn `while` loop now also exits on `seat in
  result.state.eliminated`, not just `result.done`.
- the stored transition's `done` is `result.done or seat in
  result.state.eliminated`, so elimination is correctly treated as terminal.

Separately, some episodes still legitimately run very long (one player
taking thousands of consecutive steps without yielding a turn) because
`RandomAgent` picks uniformly among legal actions, including individual
attack actions, and rarely happens to pick `StopAttackAction` — this is
opponent-policy behavior, not a trainer bug.

While in there, also removed the `if current_player_index == seat: ... else:
...` branch from the per-step loop. A small `while` loop now runs before the
main loop to play any opening moves from seats ordered before the learner's,
so by the time the main `for` loop starts it's always the learner's turn —
the loop body only has to handle one case.

## Follow-up pass: `learn_if_ready` -> `learn`, drop `train_every` cadence

Per request, `GNN_DQN_Agent.learn_if_ready(...)` is renamed to
`GNN_DQN_Agent.learn(...)` and no longer takes `episode_index`/`train_every`.
The only remaining gate is `can_train(batch_size)`: skip if the replay buffer
doesn't yet hold `batch_size` transitions. `train_every` had no reason to
exist: if you don't want to train, you don't run the trainer.

- `risk/learning/gnn_dqn_agent.py`: `learn_if_ready(...)` -> `learn(*,
  batch_size, n_steps)`, body is just the `can_train` check + `learn_steps`.
- `risk/learning/train_constants.py`: removed `TRAIN_EVERY`.

Follow-up: `Trainer.train(...)`'s call to `self.agent.learn(...)` moved from
once per episode to inside the per-turn loop, right after `remember(...)`, so
the agent learns from every agent turn instead of once per episode — standard
DQN behavior. This is much more expensive: each `learn()` call runs a full
GNN forward+backward over `batch_size` transitions, measured at ~295ms for
`batch_size=128` on this machine's GPU vs ~82ms for `batch_size=32`. With
episodes often running 1000+ agent turns, this is the difference between
~5 minutes and ~80 seconds of pure gradient compute per episode. The current
`BATCH_SIZE` in `train_constants.py` is `64`, a middle ground between noisy
updates and per-turn training cost.

Follow-up: after an unstable run plateaued, the training defaults were tuned
for more conservative DQN learning: epsilon now decays over 200 episodes,
Net A uses the Double-DQN target helper (`_max_next_ddqn_q`), `train_step`
uses Huber loss (`smooth_l1_loss`), and gradients are clipped before the
optimizer step.

## Follow-up pass: `RewardCalculator` — dense per-step reward shaping

Implements the design in [`Reward.md`](Reward.md): reward computation moved
out of `Environment` entirely and into a new dedicated class,
`risk/learning/reward.py`'s `RewardCalculator`. `Environment.step(...)` calls
`RewardCalculator.compute(...)` and returns the float as-is — no reward
arithmetic (not even the old `+1.0`/`-1.0` literals) lives in
`environment.py` anymore.

**Step 1 — terminal-only baseline.** `RewardCalculator.compute(...)`
returned just `REWARD_TERMINAL_WIN`/`REWARD_TERMINAL_LOSS` (sparse, matching
the old behavior exactly). No `REWARD_TERMINAL_TIMEOUT`: a
`MAX_STEPS_PER_EPISODE` cutoff (`trainer.py`'s episode loop) truncates the
episode without the underlying state ever becoming terminal, so it must
never be paired with `done=True` — that would falsely zero out the TD
bootstrap target for a state that isn't actually terminal. `done=True` with
`winner=None` only ever means "`reward_player` was eliminated while other
players keep going," which is correctly scored as a loss, not a neutral
timeout.

**Step 2 — full per-phase dense shaping**, every term from `Reward.md`'s
"Where each phase even has something to reward" section, gated so every
phase helper only scores the *current actor's own* decision
(`before.current_player_index == reward_player`) — `Environment.step(...)`
is called with `reward_player=seat` for opponents' actions too (the
trainer attributes every step in the chain to the learner's seat), so
without this gate an opponent's attack would get scored as if the learner
made it.

Two structural problems surfaced while wiring this in, both required
touching files beyond `reward.py`/`environment.py`:

- **`trainer.py` was discarding the agent's own per-step reward.** Its
  inner `while` loop (opponents' turns) reassigned `result` on every
  opponent step; by the time `remember(...)` ran, `result.reward` was
  whichever opponent action happened last, not the learner's own action's
  reward. Harmless under the old sparse-only reward (every non-terminal
  step was `0.0` anyway) but would have silently dropped all dense shaping.
  Fixed by accumulating (`reward_total +=`) across the whole chain instead
  of overwriting.
- **End-of-turn opponent-impact terms (`REWARD_TERRITORY_DELTA`,
  `REWARD_ARMY_DELTA_RELATIVE_SCALE`, `REWARD_CONTINENT_DELTA_RELATIVE`)
  can't be computed inside a single `Environment.step()` call** — they
  need the state right when the learner's `FortifyAction` is taken vs. the
  state once every opponent has played their full turn, and no one
  `step()` call ever sees both ends of that span (each call only processes
  one action). `trainer.py` now detects a turn-ending `FortifyAction` and
  calls `RewardCalculator.end_of_turn(...)` directly with the pre-action
  snapshot and the post-opponent-round `next_state`, adding the result into
  `reward_total`. `trainer.py` only decides *when* to call it; all the math
  still lives in `RewardCalculator`. Reachable via a new
  `Environment.reward` read-only property (`self._reward`) since `trainer.py`
  doesn't otherwise have a handle on the environment's calculator instance.
- **`REWARD_ATTACK_STOP_WITHOUT_CARD` needed turn-level memory** ("was a
  card drawn yet this turn") that only existed as a private
  `Environment._conquered_this_turn` attribute, invisible to
  `RewardCalculator` (which only ever sees `State` snapshots). Promoted to
  a real `State.conquered_this_turn` field (set in `_apply_attack`, reset
  in `_begin_turn_for`/on entering `ATTACK`), included in `State.copy()`/
  `to_dict()`/`from_dict()`. `environment.py` no longer has a
  `_conquered_this_turn` instance attribute at all.

Also fixed an inconsistency caught mid-implementation: `compute(...)`
originally short-circuited to terminal-only on `done=True`, skipping all
shaping on the final transition — contradicting `Reward.md`'s own open
question 2 ("shaping should be visible on every transition, including
`done` ones"). Removed the early return; shaping and terminal now always
sum together.

`train_constants.py` gained `REWARD_SHAPING_STEP_CAP` plus every constant
from `Reward.md`'s "Proposed v1 values" section (~27 constants total across
`TRADE_IN`/`REINFORCE_PLACE`/`ATTACK`/`OCCUPY`/`FORTIFY`/end-of-turn).
`REWARD_TERMINAL_TIMEOUT` was added then removed once the done/timeout
distinction above was worked out — never reachable through any real code
path, so kept out rather than left as dead/misleading config.

**Verified:** added 10 new `test_reward.py` cases, one per phase helper plus
`end_of_turn` (hand-built `before`/`after` `State` triples, no `Environment`
needed, per `Reward.md`'s testing plan) — `python -m pytest Temp/tests -q` →
248 passed (10 new), 1 skipped. Beyond the unit suite, ran `Trainer.train(...)`
against the real `Environment` for 1-3 episodes (thousands of real steps,
`end_of_turn` firing on every `FortifyAction`) with no exceptions, confirming
the full shaping pipeline executes correctly against live gameplay, not just
hand-built fixtures.

## Follow-up pass: `TrainingLogger` — checkpointing + optional W&B logging

Implements [`Training-Logging-Plan.md`](Training-Logging-Plan.md). New
`risk/learning/training_logger.py`'s `TrainingLogger` owns *when*/*where* a
checkpoint happens and W&B init/log/finish; it never reaches into
`GNN_DQN_Agent`'s internals directly — same split as `Environment` only
ever calling `RewardCalculator`. `wandb` is optional: wrapped in a
try/except import, every W&B-facing method becomes a no-op if it's missing
or `use_wandb=False`. Added `wandb>=0.16` to `requirements.txt` as an
optional dependency (commented as such).

**`GNN_DQN_Agent` gained `save_checkpoint(dir_path)`/`load_checkpoint(dir_path)`**
(`gnn_dqn_agent.py`) — a *full* training checkpoint (net, target_net,
optimizer state, `_train_steps`, `epsilon`, replay buffer — two files under
`dir_path`: `model.pt` and `replay.pt`, kept separate since the buffer can
be far larger than the model state). Deliberately separate from the
existing `save_params`/`load_params`, which stay as the lightweight
policy-only path (net weights only, for play/inference without dragging in
optimizer/replay state) — `Trainer._checkpoint()` now writes both at the
same cadence.

**`ReplayBuffer`** already supported loading via its constructor's `path`
argument (`ReplayBuffer(path=...)`) — no new method needed, just used as-is
by `load_checkpoint`. While touching it, fixed a `torch.load` `FutureWarning`
by passing `weights_only=False` explicitly (a transition tuple holds
`State`/`Action` domain objects, not just tensors, so the restrictive
unpickler can't load it anyway — this is trusted local checkpoint data, not
an untrusted source).

**`Trainer` integration** (`trainer.py`):
- `__init__` gained `logger`/`use_wandb`/`resume`/`notes` params, builds a
  `TrainingLogger` by default, calls `start_run(...)` then `try_resume(...)`
  — restores `self.episode` if a checkpoint was found.
- `train(...)` now accumulates `episode_reward`/`losses` across the episode
  and calls a new `_log_episode(...)` helper at the end of each episode,
  which builds the metrics dict (`episode_steps`, `agent_turns`,
  `episode_reward`, `win`, `eliminated`, `done`, `epsilon`, `replay_size`,
  `learn_loss_mean`) and hands it to `logger.log_episode(...)`.
- `main()` calls `trainer.logger.finish()` after `train(...)` returns
  (not inside `train()` itself, so repeated `train()` calls on one
  `Trainer` don't end the W&B run prematurely).

**Also removed `GNN_DQN_Agent.ingest_episode`** — dead code (no callers
anywhere in the codebase) left over from before `trainer.py`'s loop called
`remember(...)` directly per step. It also hardcoded the old `+1/-1/0`
sparse reward policy that `RewardCalculator` has since replaced, so leaving
it in place was actively misleading, not just unused.

**Verified:**
- Direct round-trip checks (`GNN_DQN_Agent.save_checkpoint`/`load_checkpoint`,
  `TrainingLogger.save_checkpoint`/`try_resume`) confirm `_train_steps`,
  `epsilon`, optimizer state, and replay buffer contents all survive a
  save/load cycle exactly.
- Added `test_gnn_dqn_agent_save_and_load_checkpoint_round_trip` to
  `test_agents.py` (mirrors the existing `save_params`/`load_params` test's
  shape) and a new `test_training_logger.py` (config building, checkpoint
  cadence/latest-episode selection, no-op-when-disabled, resume-disabled).
  `python -m pytest Temp/tests -q` → 255 passed (7 new), 1 skipped.
- Ran a full explicit-agent `Trainer(...).train(n_episodes=1)` with
  `use_wandb=False` end
  to end (thousands of real steps) with no exceptions, confirming the
  logging/metrics wiring doesn't break the existing training loop.

## Follow-up pass: continent-advantage attack reward

Implements `Docs/Reward.md`'s continent-advantage shaping term. The goal is
to help the agent convert a local attack advantage into an actual continent
plan, instead of only learning which adjacent territory is attackable.

`risk/learning/reward.py` now caches the board-static maximum continent
bonus in `RewardCalculator.__init__` and adds a private
`_continent_advantage(...)` helper. The helper scores the attack target's
continent from the pre-attack state using three gated signals:

- territory edge above the alive-player baseline,
- troop majority edge inside that continent,
- normalized continent bonus value.

Each edge is rescaled back to `[0, 1]` after gating, then multiplied by the
continent value. `_attack(...)` adds the resulting
`REWARD_ATTACK_CONTINENT_ADVANTAGE * advantage / (missing + 1)` term
alongside the existing `REWARD_ATTACK_CONTINENT_DOMINATION` term; the
one-time `REWARD_ATTACK_CONTINENT_CAPTURED` reward is unchanged.

`risk/learning/train_constants.py` gained
`REWARD_ATTACK_CONTINENT_ADVANTAGE = 1.20`, exported through `__all__`, so
the new scale lives with the rest of the reward tuning knobs. `Docs/Reward.md`
was updated in the constant checklist, proposed values, and implementation
notes.

**Verified:** direct `RISK` virtualenv checks pass:
`python -m pytest Temp\tests\test_reward.py -q` -> 16 passed, and
`python -m compileall risk\learning\reward.py risk\learning\train_constants.py`
also passed. Full `Temp/tests` currently reaches 256 passed / 1 skipped, then
fails 3 existing `test_training_logger.py` cases because the tests call a
missing public `TrainingLogger.save_checkpoint(...)` method; those failures
are outside this reward change.

## Follow-up pass: trainer evaluation wiring

`Trainer` now constructs an `Evaluator` alongside `TrainingLogger` and, every
`EVAL_EVERY_EPISODES`, runs deterministic eval before episode metrics are
logged. Eval metrics are merged into the same W&B row as the training episode,
including `eval_saved_best`, then normal resume checkpointing still runs as
before. Best-model policy saves are owned by `Evaluator`; resume checkpoints
remain owned by `TrainingLogger`.

**Verified:**
- New `Temp/tests/test_evaluator.py` (6 cases): `evaluate(...)`'s returned
  metric keys/`episode`/`eval_games` count, `epsilon`/`train_mode` restored
  after eval, determinism across repeated `evaluate(...)` calls on the same
  agent/episode, the score formula's weights, and `maybe_save_best(...)`'s
  top-N retention plus its rejection of a score below the worst kept.
- `python -m pytest Temp/tests -q` → 262 passed, 1 skipped; the only
  failures are the pre-existing 3 `test_training_logger.py` cases unrelated
  to this change (confirmed via `git stash` against plain `main`).
- Ran a real explicit-agent `Trainer(..., use_wandb=False, resume=False).train(
  n_episodes=1)` with `EVAL_EVERY_EPISODES`/`MAX_STEPS_PER_EPISODE`
  temporarily lowered, end to end with no exceptions — confirmed it writes
  `Checkpoints/run_999/best/best_ep000001_score....pt` and `manifest.json`.
