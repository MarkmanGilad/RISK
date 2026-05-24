## Plan: Playable Risk Architecture

Build the first playable pygame-based Risk game on top of the existing map renderer by separating the project into domain logic, rendering/UI, and agent input layers. Reuse the current Graphics map pipeline, add explicit game-state and action models, enrich map data with adjacency and card metadata, and keep the runtime loop small so the same Environment and Agent contracts can later be reused for AI training without rewriting the core game.

**Planned classes and files**
- `BoardTopology`
	The static board-data wrapper. Owns territory ids, adjacency, continent membership, and other graph-like queries needed by the environment and future GNN input. Should expose stable territory ordering and an `edge_index`-style adjacency view so a later graph adapter can build GNN inputs without re-deriving the graph.
- `Phase`
	Enum-like type for turn phases (e.g., `SETUP`, `REINFORCE`, `ATTACK`, `FORTIFY`, `GAME_OVER`). Centralizes phase identity so state, environment, agents, and UI agree on the current step.
- `Player`
	Light value object for a seat: player id, display name, color, agent kind (human or AI), and elimination flag. Keeps per-player metadata out of `State` internals.
- `Card`
	Value object for a Risk territory card. Carries the territory id and the card symbol used for trade-in sets.
- `GameSettings`
	Configuration passed from setup screens to the environment: player count, per-seat `Player` info, rule toggles, and optional RNG seed for reproducibility.
- `Environment`
	The core rule engine. Owns turn flow, setup, legal actions, combat resolution, continent bonuses, cards, elimination, and winner detection. Accepts a `GameSettings` and an optional seed so games can be replayed deterministically for tests and future RL training.
- `State`
	The full game state or observation container. Holds dynamic game data in a pygame-free form: ownership, armies, current player, current phase, reinforcement budget, pending attack context, cards in hand, and eliminated players. Designed to be cheap to copy and serialize, and ready for later conversion into graph or tensor features.
- `Action`
	The base action model for moves sent to the environment. Keeps the game loop and agents working with a consistent action API.
- `ReinforcementAction`
	Start-of-turn troop placement action. Describes which territories receive reinforcements and how many units are placed on each.
- `AttackAction`
	Combat action. Describes the attacking territory, defending territory, and the number of dice used.
- `StopAttackAction`
	Phase-transition action. Ends the attack phase and moves the turn to fortify or end-of-turn handling.
- `FortifyAction`
	End-of-turn troop movement action. Describes moving soldiers from one owned territory to another connected owned territory.
- `RiskMapRenderer`
	The pygame board renderer. Draws the map, territory ownership, labels, continent badges, and army markers from game data.
- `BaseAgent`
	Shared agent interface. Defines how human, scripted, random, or future AI agents receive state and legal actions and return an action or `None`.
- `HumanAgent`
	Reads pygame input and converts clicks or UI choices into legal game actions. Returns `None` while the human player is still deciding.
- `RandomAgent`
	Optional non-RL fallback agent. Useful for testing seat-by-seat AI wiring and the full game loop before any learned model exists.
- `Game`
	The thin orchestrator loop. Initializes systems, asks the active agent for an action, sends that action to the environment, and triggers rendering. Holds no game rules.
- `InitScreen`
	The setup-screen controller for choosing player count, names, colors, seat types, and starting the match. Produces a `GameSettings` for the environment.
- `constants.py`
	The central game constants file. Stores player limits, dice rules, phase names, bonuses, card-set values, and other shared configuration.

## Phase 1 - Structure And Documentation (Done)

**A. What we will build**
- Lock the project structure before gameplay code starts.
- Keep the current renderer demo files as graphics references only.
- Keep the project documentation aligned with the planned architecture.

**B. Classes and files that will be built or changed**
- `Docs/plan.md`
- `Docs/Graphic.md`
- `Graphics/demo_loop.py` as reference-only
- `Graphics/demo_state.py` as reference-only

**C. Tests that we can do**
- Verify the plan document exists and reflects the intended architecture.
- Verify the graphics document explains the renderer and does not claim to be the full game architecture.
- Verify no implementation code is changed in this phase.

## Phase 1.5 - Repo Layout (Done)

**A. What we will build**
- Reorganize the existing workspace so every planned class has a known home before any gameplay code is written.
- Separate production code from demos so the renderer can be reused without dragging demo state/loop along.
- Pick the package import path now so Phase 2+ files land in the right place from day one.

**B. Files and folders that will be created, moved, or renamed**
- Create empty packages (each with `__init__.py`):
	- `risk/game/` for `BoardTopology`, `State`, `Phase`, actions, `Environment`, `constants.py`.
	- `risk/agents/` for `BaseAgent`, `HumanAgent`, `RandomAgent`.
	- `risk/ui/` for `InitScreen` and pygame input/hit-test helpers.
	- `risk/app/` for the thin `Game` loop and `main.py`.
	- `risk/graphics/` to hold the production renderer.
	- `risk/learning/` reserved for the future `graph_adapter.py` (GNN tensors).
- Move and rename:
	- `Graphics/risk_map.py` -> `risk/graphics/risk_map.py` (keep `RiskMapRenderer` name).
	- `Graphics/demo_loop.py` -> `demos/graphics_loop.py`.
	- `Graphics/demo_state.py` -> `demos/graphics_state.py`.
	- `demo_pygame_risk_map.py` -> `demos/pygame_risk_map.py`.
- Update imports in the moved demo files and in any remaining entry points so the existing demo still runs from the new location.
- Delete the old `Graphics/` folder once it is empty.
- `Assets/`, `Docs/`, and `requirements.txt` stay where they are.

**C. Tests that we can do**
- Verify the moved demo (`python -m demos.pygame_risk_map` or equivalent) still launches and renders the board.
- Verify `from risk.graphics.risk_map import RiskMapRenderer` works from a fresh Python session.
- Verify the new empty packages are importable (`import risk.game`, `import risk.agents`, `import risk.ui`, `import risk.app`, `import risk.learning`).
- Verify no production module under `risk/` imports anything from `demos/`.

## Phase 2 - Static Board Model (Done)

**A. What we will build**
- Add explicit static board data for territory ids, adjacency, continent membership, and card types.
- Build the first core gameplay dependency: the board-topology layer.
- Make the board data deterministic and ready for both rule checks and later graph conversion.

**B. Classes and files that will be built or changed**
- `BoardTopology`
- `Assets/RiskMap/risk_map_data.json` or a neighboring board-data file
- optional helper module for board-data loading

**C. Tests that we can do**
- Verify every territory appears exactly once.
- Verify adjacency queries return the expected neighbors for known sample territories (including sea routes like Alaska <-> Kamchatka).
- Verify adjacency is symmetric.
- Verify every territory belongs to exactly one continent.
- Verify continent definitions and bonuses match the territory set.
- Verify the topology exposes a deterministic territory ordering and an edge list suitable for later GNN `edge_index` construction.

## Phase 3 - Dynamic State And Actions

This phase is split into four small steps. Each step has its own test
suite, and after every step the previous tests (Phase 2 BoardTopology
tests included) must still pass. That way every increment is a safe
checkpoint and we always know exactly which step broke something.

Common rule for every step:
- Run the **full** test suite (`python -m pytest Temp/tests/ -v`) and
  re-launch the demo (`python -m demos.pygame_risk_map --save smoke.png`
  headless is enough) to confirm nothing earlier was broken.

### Phase 3.1 - Constants And Enums (Done)

**A. What we will build**
- The shared constants module and the `Phase` enum that the rest of the
  phase depends on. Nothing else changes.

**B. Classes and files that will be built or changed**
- `risk/game/constants.py` (player limits, dice rules, starting-army
  counts per player count, continent bonus reference, card-set
  progression like 4, 6, 8, 10, 12, 15, +5).
- `risk/game/phase.py` with the `Phase` enum (`SETUP`, `REINFORCE`,
  `ATTACK`, `FORTIFY`, `GAME_OVER`).

**C. Tests that we can do** (`Temp/tests/test_constants_and_phase.py`)
- Verify player-count limits are 3..6 and starting armies follow the
  classic table (3p:35, 4p:30, 5p:25, 6p:20).
- Verify `Phase` has the five expected members and is ordered so the
  enum value can be safely compared / serialized.
- Verify the card-set progression is monotonically increasing and that
  the "+5 after the 15" rule is encoded as a function or constant.
- Verify `risk.game.constants` and `risk.game.phase` are importable and
  contain no pygame imports.
- Confirm the Phase 2 tests still pass.

### Phase 3.2 - Player, Card, GameSettings (Done)

**A. What we will build**
- The small value objects that describe *who is playing* and *how the
  match is configured*. Still no game state.

**B. Classes and files that will be built or changed**
- `risk/game/player.py` with `Player` (`id`, `name`, `color`,
  `agent_kind`, `eliminated`).
- `risk/game/card.py` with `Card` (`territory_id`, `symbol`) and a
  helper that recognizes valid trade-in sets.
- `risk/game/settings.py` with `GameSettings` (`players: list[Player]`,
  optional `seed: int | None`, rule toggles).

**C. Tests that we can do** (`Temp/tests/test_player_card_settings.py`)
- Verify `Player` enforces a unique id and rejects invalid agent kinds.
- Verify `Card` rejects unknown territory ids when given a
  `BoardTopology`.
- Verify trade-in detection: three of a kind, one of each symbol,
  reject mixed invalid sets.
- Verify `GameSettings` rejects fewer than 3 or more than 6 players and
  duplicate player ids, and accepts an optional integer seed.
- Confirm Phase 2 and Phase 3.1 tests still pass.

### Phase 3.3 - State (Done)

**A. What we will build**
- The dynamic state container, built on top of `BoardTopology` and the
  metadata types from the previous steps. Pygame-free, copy-friendly,
  ready for later graph conversion.

**B. Classes and files that will be built or changed**
- `risk/game/state.py` with `State` holding:
	- per-territory `owners` (player id) and `armies`, stored as
	  arrays aligned with `BoardTopology` territory order.
	- `current_player_index`, `phase`, `reinforcement_budget`,
	  `pending_attack` (or `None`), per-player `hands: list[list[Card]]`,
	  `eliminated: set[int]`, `cards_traded_in_count` (drives the
	  progression).
- A `State.snapshot()` / `State.copy()` method (deep enough for safe
  rollouts) and a `State.to_dict()` for serialization.
- A documented `State.to_features(topology)` stub that raises
  `NotImplementedError` — reserved for the future GNN adapter.

**C. Tests that we can do** (`Temp/tests/test_state.py`)
- Verify `State` constructs with arrays sized to
  `len(topology) == 42` and indices aligned with
  `topology.territories`.
- Verify `copy()` produces an independent state (mutating the copy does
  not affect the original).
- Verify `to_dict()` is JSON-serializable and round-trips back into an
  equal `State`.
- Verify the `to_features` stub is present and raises
  `NotImplementedError`.
- Verify `State` does not import pygame.
- Confirm Phase 2, 3.1, and 3.2 tests still pass.

### Phase 3.4 - Actions (Done)

**A. What we will build**
- The typed action objects used by every agent and by `Environment`.
  They only describe *what* the player wants to do — they do not
  evaluate legality (that is Phase 4).

**B. Classes and files that will be built or changed**
- `risk/game/actions.py` with `Action` (base), `ReinforcementAction`,
  `AttackAction`, `StopAttackAction`, `FortifyAction`. Each has a
  `phase` class attribute pointing at the `Phase` it belongs to.
- A small `action_from_dict` / `to_dict` pair so actions can be logged
  or replayed.

**C. Tests that we can do** (`Temp/tests/test_actions.py`)
- Verify each action class validates its required fields on
  construction (e.g. `AttackAction` requires `from_territory`,
  `to_territory`, `dice` in 1..3).
- Verify constructing with unknown territory ids raises (when given a
  `BoardTopology`).
- Verify `ReinforcementAction` rejects negative placements and zero-sum
  placements that exceed the reinforcement budget *shape*
  (sum-of-armies validation only — legality vs. ownership is Phase 4).
- Verify each action's `phase` attribute matches the right `Phase`
  member.
- Verify `to_dict` -> `action_from_dict` round-trips for every action
  type.
- Confirm Phase 2, 3.1, 3.2, and 3.3 tests still pass.

After all four steps pass, Phase 3 as a whole is considered done.



## Phase 4 - Environment Rules (Done)

**A. What we will build**
- Implement the `Environment` as the only rule engine.
- Add setup flow, reinforcement calculation, attack resolution, fortify rules, cards, elimination, winner detection, and legal action generation.
- Accept a `GameSettings` and optional RNG seed so games are deterministic and replayable for tests and future training.
- Keep the public environment API narrow and training-friendly: `reset`, `current_state`, `legal_actions`, `step(action)`, `is_terminal`, `winner`.

**B. Classes and files that will be built or changed**
- `Environment`
- `environment.py`
- `State` and action classes as needed for rule support
- board-data access through `BoardTopology`

**C. Tests that we can do**
- Verify random setup with a fixed seed produces a reproducible starting state.
- Verify reinforcement counts, including continent bonuses and minimums.
- Verify legal attacks use adjacency and ownership correctly.
- Verify dice bounds and battle resolution probabilities over many seeded rolls.
- Verify conquest, elimination, fortify legality, and winner detection.
- Verify card draw on successful conquest and trade-in rules, including mandatory trade-ins above the hand limit.
- Verify that applying any illegal action raises and does not mutate state.

## Phase 5 - Agents (Done)

**A. What we will build**
- Add the agent interface after the environment exists.
- Implement a human-controlled agent for pygame input.
- Add an optional simple non-RL agent so Human/AI seats can already work end-to-end.

**B. Classes and files that will be built or changed**
- `BaseAgent`
- `HumanAgent`
- `RandomAgent`
- `base_agent.py`
- `human_agent.py`
- `random_agent.py`

**C. Tests that we can do**
- Verify `BaseAgent` exposes the expected input/output contract.
- Verify `HumanAgent` returns `None` while no complete decision was made.
- Verify `HumanAgent` only returns legal actions.
- Verify `RandomAgent` always chooses from the legal-action set.

## Phase 6 - Graphics And Interaction (Done)

**A. What we will build**
- Reuse the existing renderer as the board view for the real game.
- Add UI rendering for side panels, phase prompts, dice summaries, player status, and setup screens.
- Add territory hit-testing so clicks can be mapped to territory ids.

**B. Classes and files that will be built or changed**
- `RiskMapRenderer`
- renderer-adjacent hit-testing or interaction helper
- UI render modules
- `InitScreen`
- setup-screen modules
- `Graphics/risk_map.py`

**C. Tests that we can do**
- Verify the board still renders correctly with owners and armies.
- Verify click hit-testing resolves sample screen positions to the correct territory ids.
- Verify illegal clicks do not mutate game state.
- Verify setup screens support 3 to 6 players, names, colors, seat types, and Start Game.
- Keep the renderer preview command working as a regression check.

## Phase 7 - Game Loop And App Entry (Done)

**A. What we will build**
- Add the thin `Game` loop that orchestrates environment, agents, and rendering.
- Add the minimal application entry point that wires setup screens, environment creation, and agents together.
- Keep the loop reusable later for AI training flows.

**B. Classes and files that will be built or changed**
- `Game`
- main game-loop module
- application entry-point module
- setup-screen integration points

**C. Tests that we can do**
- Verify the app can initialize all core objects without crashing.
- Verify the loop can advance turns until quit or terminal state.
- Verify the game loop does not own rule logic directly.
- Add a smoke test or boot test for the main entry path if possible.

## Phase 8 - Final Validation (Done)

**A. What we will build**
- Run the full validation pass in dependency order and then as a playable app.
- Confirm the architecture remains ready for future AI-agent and GNN integration.

**B. Classes and files that will be built or changed**
- test files for board topology, state/actions, and environment
- documentation updates if validation reveals mismatches

**C. Tests that we can do**
- Run board-topology tests first.
- Run state and action tests second.
- Run environment rule tests third.
- Manually verify setup flow, reinforcement, attack, stop-attack, fortify, AI seats, and clean shutdown.
- Verify the state and board topology can later be converted into graph inputs without depending on pygame.

**Reference files (existing)**
- `Graphics/risk_map.py` — reuse `RiskMapRenderer`, territory polygons, and render pipeline as the basis for the board renderer and click hit-testing.
- `Graphics/demo_loop.py` and `Graphics/demo_state.py` — graphics/demo references only. Do not grow into the real game loop or state model.
- `Assets/RiskMap/risk_map_data.json` — extend with adjacency and territory/card metadata needed for legal-action generation.
- `Docs/Graphic.md` — graphics-layer reference.
- `Docs/plan.md` — this plan.

Per-phase "Classes and files that will be built or changed" lists are the source of truth for new files; this section only tracks existing reference files.

**Decisions**
- Included in v1: pygame UI, full game rules, card trade-ins, continent bonuses, elimination, winner detection, and setup screens.
- Included in v1: random territory distribution during setup rather than manual territory draft.
- Included in v1: 3-6 players only.
- Included in v1: init screen fields for player count, player names, player colors, Human/AI seat type, and Start Game.
- Included in v1: architecture ready for AI agents, with no RL training code yet.
- Recommended: implement a simple non-RL `RandomAgent` stub now so AI seats can be exercised end-to-end without special cases.
- Excluded from v1: PyTorch tensor conversion implementation, RL environment wrappers, training loop, and model code.
- Excluded from v1: manual territory draft/claim phase, unless requirements change later.

**Further Considerations**
1. Recommended package layout for new code: `risk/game/` (env, state, actions, constants, board topology), `risk/agents/` (base, human, random), `risk/ui/` (init screens, panels, hit-test helpers), `risk/app/` (main entry, game loop). Keep the existing `Graphics/` package unchanged until or unless names are normalized repo-wide.
2. Adjacency data should be authored explicitly in JSON instead of inferred from polygons. This is the main technical prerequisite for a clean and testable environment and for later GNN graph construction.
3. Card progression and starting-army counts should land in `constants.py` and be covered by tests before the environment depends on them, so balance changes do not silently break rules.
4. Future GNN integration should live in a separate adapter (for example `risk/learning/graph_adapter.py`) that consumes `BoardTopology + State` and returns tensors. Keep the core game free of PyTorch imports.
5. Non-goals for v1: PyTorch/RL code, learned agents, manual territory draft, network play, save/load to disk, animations beyond simple rendering.

## Phase 9 - Init Screen UI (Done)

**A. What we will build**
- Add the visible pygame setup screen that sits on top of the headless
  `InitScreenState` from Phase 6, so the user can configure player
  count, names, colors, and seat type (Human / AI) before the match
  starts.
- Make the init screen the default entry point of `main.py`. Smoke
  tests bypass it with an explicit `--skip-menu` flag (or by passing
  `--max-ticks`, which implies non-interactive).
- Keep all setup *state* in the pygame-free `InitScreenState` so the
  view is a thin renderer / event router and stays testable.

**B. Classes and files that will be built or changed**
- `risk/ui/init_screen_view.py` with `run_init_screen(screen) -> GameSettings | None`.
- `risk/app/main.py`: call `run_init_screen` before building
  `Environment` / agents; honor `--skip-menu`.
- `.vscode/launch.json` already exposes the regular Run config plus a
  headless smoke config that uses `--max-ticks` (and therefore skips
  the menu automatically).

**C. Tests that we can do**
- Verify `InitScreenState` still passes its Phase 6 tests (player
  count clamp, name / color / kind mutators, `can_start` validation,
  `build_settings`).
- Verify `python -m risk.app.main --skip-menu --max-ticks N` still
  completes a headless smoke run (no menu blocking).
- Manual: launch the game, change player count, edit names, cycle
  colors, toggle Human/AI, and start. Confirm an invalid config
  (duplicate colors / blank name) disables the Start button and shows
  the reason.

## Phase 10 - Wire Human Agent Input (Done)

**A. What we will build**
- Make Human seats actually playable from the pygame window. Today the
  loop hit-tests clicks but never calls `HumanAgent.submit`, so the
  game stalls forever on the first human turn.
- Use the **side panel as the decision surface** (modelled on
  `Docs/pygame_gameplay_preview.png`). The map is for *selection only*:
  the player clicks territories on the map to populate fields in the
  HUD, and commits the action via HUD buttons. Every action follows
  the same pattern: pick targets on the map -> fill a small form in
  the side panel -> press the action button.
- Add a small **input controller** that owns the per-turn UI state
  (pending placements, selected territories, chosen dice / count) and
  turns map clicks + HUD button clicks into a single completed
  `Action` fed to `HumanAgent.submit`. The controller stays
  pygame-free at its core so it can be unit-tested without a display.

**Common HUD layout while it is the human's turn**

Below the existing per-player table, the HUD grows a "Your Turn
(PHASE)" block whose contents change per phase:

```
Your Turn (PHASE)            <- phase header
-----------------------------
[ phase-specific form ]      <- selections + numeric fields
-----------------------------
[ Primary button ]           <- commits the action
[ Secondary button ]         <- skip / end phase (when allowed)
```

Each filled field shows a "(Clear)" link so the player can wipe a
single selection without losing the others.

**Per-action picking flow** (the rules the controller enforces):

- **Reinforce -> `ReinforcementAction`**
  - Side panel shows: live counter `placed / budget`, a `Placements`
    list with `(-) (+) (Clear)` per row, a primary `[ Place Armies ]`
    button (disabled until `placed == budget`), and `[ Clear All ]`.
  - Map: clicking an *own* territory adds it to the list (or
    increments by 1 if already there). Clicks on enemy / unowned
    territories are ignored.
  - `[ Place Armies ]` submits one
    `ReinforcementAction(placements={...})`.

- **Attack -> `AttackAction`**
  - Side panel shows: `Attack from: <name> (N armies) (Clear)`,
    `Attack: <name> (M armies) (Clear)`, a `Dice [v K]` dropdown whose
    range is auto-clamped to `1..min(3, armies_from - 1)`, primary
    `[ Attack! ]`, secondary `[ End Attack Phase ]`.
  - Map: first click on an *own* territory with >= 2 armies fills
    `Attack from`. Next click on an *adjacent enemy* fills `Attack`.
    Illegal clicks are ignored. After a successful conquest the "to"
    field auto-clears (it is now owned) while "from" stays selected so
    the player can keep clicking `[ Attack! ]`.

- **Stop attack -> `StopAttackAction`**
  - `[ End Attack Phase ]` in the same panel submits it. No map
    interaction needed.

- **Fortify -> `FortifyAction`**
  - Side panel shows: `Move from: <name> (N armies) (Clear)`,
    `Move to: <name> (M armies) (Clear)`, a `Count [v K]` dropdown in
    `1..armies_from - 1`, a `Path: A -> B` connectivity confirmation,
    primary `[ Move Armies ]`, secondary `[ Skip Fortify ]`.
  - Map: click an own territory with >= 2 armies for `Move from`;
    then click an own territory **connected through owned territories**
    (BFS check shared with `Environment._connected_through_owned`) for
    `Move to`.
  - `[ Move Armies ]` submits `FortifyAction(from, to, count)`;
    `[ Skip Fortify ]` submits `FortifyAction(None, None, 0)`.

- **Forced card trade-ins** (hand > 5 entering REINFORCE)
  - Already auto-resolved by `Environment._enter_reinforce_for`. The
    HUD only needs a one-line notice in the *Your Turn* header, e.g.
    `"Auto traded cards: +10 armies added to budget."`. No human
    input required in v1.

**Keyboard shortcuts** are *optional accelerators* on top of the HUD
widgets, never the primary UX:
- `Enter` triggers the panel's primary button.
- `Esc` is `Clear All` for the current phase.
- `S` triggers `End Attack Phase` / `Skip Fortify` when visible.

**B. Classes and files that will be built or changed**
- `risk/ui/human_input.py` with `HumanInputController`:
  - Ephemeral state: `pending_placements: dict[str, int]`,
    `selected_from: int | None`, `selected_to: int | None`,
    `attack_dice: int`, `fortify_count: int`.
  - Public methods (all pygame-free):
    - `on_territory_click(territory_index, button)` — map click on a
      decoded territory index. Routing depends on `state.phase`.
    - `on_hud_button(button_id)` — `"place_armies"`,
      `"clear_all"`, `"attack"`, `"end_attack"`, `"move_armies"`,
      `"skip_fortify"`.
    - `on_hud_field(field_id, delta_or_value)` — increment / decrement
      / clear individual rows (placements list, dice dropdown, count
      dropdown).
    - `on_turn_change(state)` — clears ephemeral state when the active
      seat or phase changes (called by the loop).
    - `widgets(state) -> HudActionPanelModel` — pure data describing
      the form to render (rows, dropdown ranges, button enabled
      flags, prompt string). The renderer turns this into pygame
      rects; the controller never imports pygame.
  - Internally validates every assembled action against
    `env.legal_actions()` before calling `HumanAgent.submit`, so the
    agent never silently drops a click.
- `risk/ui/panels.py`:
  - Extend `HudPanel` with `render_action_panel(target, rect, model)`
    that draws the "Your Turn (PHASE)" form from the controller's
    `HudActionPanelModel`. Returns a list of
    `(rect, button_id | field_id)` so the loop can hit-test HUD
    clicks the same way it hit-tests the map.
- `risk/app/main.py`:
  - Instantiate `HumanInputController(env, agents)` once.
  - On `MOUSEBUTTONDOWN`: first hit-test the HUD action panel; if a
    widget was hit, dispatch to `on_hud_button` / `on_hud_field`.
    Otherwise resolve the territory via `TerritoryHitTester`, look up
    `BoardTopology.index_of(name)`, and call
    `on_territory_click(index, button)`. All event forwarding is
    gated on the current seat being human.
  - Optionally forward `KEYDOWN` for the Enter / Esc / `S`
    accelerators listed above.
  - Call `controller.on_turn_change(state)` whenever the
    `(current_player_index, phase)` tuple changes between ticks.
  - Pass `controller.widgets(state)` into the new
    `HudPanel.render_action_panel`.

**C. Tests that we can do** (`Temp/tests/test_human_input.py`)
- Construct a deterministic `Environment` (fixed seed) so a known
  human seat is on turn in REINFORCE with a known budget. Drive the
  controller with `on_territory_click(...)` map calls and
  `on_hud_button("place_armies")` HUD calls and confirm
  `HumanAgent._pending` ends up with the expected
  `ReinforcementAction`.
- `on_hud_field("placements", "Ural", -1)` decrements that row and
  never drops below 0; `on_hud_button("clear_all")` empties the form.
- Attack flow: click own -> click adjacent enemy ->
  `on_hud_field("dice", 2)` -> `on_hud_button("attack")` -> confirm
  `AttackAction(from, to, dice=2)` is submitted and is in
  `env.legal_actions()`. After conquest, `selected_to` is cleared but
  `selected_from` is retained.
- `on_hud_button("end_attack")` in ATTACK phase submits
  `StopAttackAction()`.
- Fortify flow: select connected pair, set count via
  `on_hud_field("count", k)`, `on_hud_button("move_armies")` submits
  a legal `FortifyAction`. `on_hud_button("skip_fortify")` submits
  `FortifyAction(None, None, 0)`.
- Illegal map clicks (enemy in reinforce, non-adjacent in attack,
  disconnected in fortify) are ignored: controller state unchanged,
  `HumanAgent.submit` never called.
- HUD primary buttons stay disabled when the form is incomplete
  (e.g. `[ Place Armies ]` disabled until `placed == budget`); the
  controller refuses the call if it arrives anyway.
- `widgets(state)` returns the expected `HudActionPanelModel` for each
  phase (rows, dropdown ranges, prompt text, button-enabled flags) —
  purely data, no pygame.
- `on_turn_change` clears all ephemeral state when phase or seat
  changes.
- Confirm all earlier phase tests still pass.

After Phase 10 lands, the loop in `main.py` should be able to run a
mixed Human + AI game end-to-end from the menu through to a winner
using only mouse clicks on the map and the side-panel buttons.
