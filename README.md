# Risk

A from-scratch, pygame implementation of the classic *Risk* board game, built
with a pygame-free rules engine at its core so the same game can be driven by
a human clicking a map, by scripted heuristic bots, or — the end goal — by a
reinforcement-learning agent (GCN + DQN) trained through self-play.

```
python -m risk.app.main              # play it
python -m risk.learning.self_play    # watch/train AI-only self-play
python -m pytest Temp/tests -q       # run the test suite
```

---

## Table of contents

1. [Rules of the game](#rules-of-the-game)
2. [How to play](#how-to-play)
   - [Install & run](#install--run)
   - [Command-line options](#command-line-options)
   - [The setup screen](#the-setup-screen)
   - [Controls during a match](#controls-during-a-match)
3. [Project architecture](#project-architecture)
4. [Folder-by-folder tour](#folder-by-folder-tour)
   - [`risk/constants.py` / `risk/ui_constants.py`](#riskconstantspy--riskui_constantspy)
   - [`risk/game/` — the rules engine](#riskgame--the-rules-engine)
   - [`risk/agents/` — who decides each move](#riskagents--who-decides-each-move)
   - [`risk/app/` — wiring the interactive app](#riskapp--wiring-the-interactive-app)
   - [`risk/ui/` — input & rendering](#riskui--input--rendering)
   - [`risk/learning/` — self-play & training](#risklearning--self-play--training)
   - [`Temp/tests/` — the test suite](#temptests--the-test-suite)
   - [`Assets/`, `Docs/`](#assets-docs)
5. [Testing](#testing)
6. [Roadmap](#roadmap)

---

## Rules of the game

This is classic *Risk*, 3–6 players, played on a 42-territory, 6-continent
world map:

| Continent | Territories | Reinforcement bonus |
|---|---|---|
| Africa | 6 | +3 |
| Asia | 12 | +7 |
| Australia | 4 | +2 |
| Europe | 7 | +5 |
| North America | 9 | +5 |
| South America | 4 | +2 |

**Setup.** Territories are dealt out randomly and evenly among players (one
army each), then each player's remaining starting armies (35 / 30 / 25 / 20
for 3 / 4 / 5 / 6 players) are dropped one at a time onto territories they
already own.

**Turn structure.** Each turn is three phases, always in this order:

1. **Reinforce** — you receive `max(3, owned_territories ÷ 3)` armies, plus
   the bonus for every continent you fully control, and place them on your
   own territories. If you're holding 5+ cards you must trade in a set
   before you can place anything.
2. **Attack** (optional, any number of times) — pick an owned territory with
   ≥2 armies, an adjacent enemy territory, and a number of dice (1–3, capped
   at `armies − 1`). Both sides roll; the defender rolls up to 2 dice (capped
   at their armies). Highest dice are compared pair-wise; ties favor the
   defender. Loser(s) of each pairing lose one army. If the defender's
   armies hit zero, the territory is conquered — you choose how many armies
   (between the dice you attacked with and `armies_from − 1`) to move in
   before continuing. Conquering at least one territory this turn draws you
   one card; eliminating a player transfers all their cards to you.
3. **Fortify** (optional, once) — move any number of armies (leaving at
   least 1 behind) from one owned territory to another owned territory,
   as long as there's a path between them through territories you own.

**Cards.** Three symbols (infantry / cavalry / artillery) plus 2 wild cards.
A tradeable set is three-of-a-kind or one-of-each, with wilds substituting
for anything. Trade-in values follow the classic progression
**4, 6, 8, 10, 12, 15**, then **+5** per set after that (20, 25, 30, ...).
On top of that, any non-wild card in the set whose territory you currently
occupy grants **+2 armies placed on that territory immediately**.

**Winning.** A player who loses their last territory is eliminated. The game
ends when only one player remains.

---

## How to play

### Install & run

```bash
pip install -r requirements.txt   # pygame, svg.path  (Python 3.12)
python -m risk.app.main           # opens the setup screen
```

### Command-line options

```
python -m risk.app.main [--width W] [--height H] [--seed N] [--players N]
                         [--max-ticks N] [--skip-menu] [--auto-restart]
                         [--mode play|train|train-no-render]
                         [--ai-delay-ms N] [--marker-ms N]
```

| Flag | Meaning |
|---|---|
| `--mode play` (default) | AI moves are paced (`--ai-delay-ms`, default 600ms) and briefly highlighted on the board, so a human can follow along. |
| `--mode train` | Same window, but AI moves happen as fast as possible — no pacing, no marker. |
| `--mode train-no-render` | No pygame window at all — pure simulation, fastest option for an all-AI game. |
| `--skip-menu` | Skip the setup screen and start immediately with an all-random roster. |
| `--auto-restart` | When a game ends, immediately start a new one with the next seed instead of showing the win screen. |
| `--seed N` | RNG seed — same seed reproduces the exact same territory deal, army placement, dice rolls, and card draws. |
| `--max-ticks N` | Stop after N ticks regardless of outcome (used by smoke tests). |

### The setup screen

(Skipped entirely by `--skip-menu`.) For each of 3–6 seats, pick a name, a
unique color, and a seat type:

- **Click `−` / `+`** next to "Players" to change the player count (3–6).
- **Click a name cell** to edit it; type, then `Enter` to commit or `Esc` to
  cancel (clicking elsewhere also commits).
- **Click a color swatch** to cycle to the next color not already taken by
  another seat.
- **Click the seat-type cell** to cycle: `Human → Random → Raider → Sentinel
  → Empire → Human ...` (see [agent kinds](#riskagents--who-decides-each-move)
  below).
- **Click "Start Game"** — disabled (greyed out, with a reason shown) while
  any name is blank or two seats share a color.

### Controls during a match

The map is for *selecting* territories; the side panel is the *decision
surface* — every action is: click the map to fill in a field, then press a
button in the panel.

- **Reinforce** — left-click an owned territory to add +1 army to it (right-
  click to remove one); once the placed total matches your budget, **Place
  Armies** commits it. **Clear All** resets your pending placements.
- **Attack** — left-click one of your territories with ≥2 armies (becomes
  *Attack from*), then left-click an adjacent enemy territory (becomes
  *Attack*). Adjust **Dice** (1 up to `min(3, armies − 1)`), then **Attack!**
  to roll. Click **End Attack Phase** when you're done attacking this turn.
- **Occupy** (automatic, right after a conquest) — choose how many armies to
  move into the territory you just took (**Count**, bounded by the rules
  above), then **Move Armies**. The conquered territory becomes your new
  *Attack from*, so you can keep pushing forward in the same turn.
- **Fortify** — left-click an owned territory with ≥2 armies (*Move from*),
  then another owned territory reachable through territory you own (*Move
  to*). Adjust **Count**, then **Move Armies** — or **Skip Fortify** to end
  your turn without moving anything.
- **Cards** — the **Cards (N)** button (bottom of the panel, available
  during Reinforce) shows your hand; click up to 3 cards to select them, and
  if they form a valid set, **Trade Set (+value)** becomes active. Trading
  is forced open and mandatory once you're holding 5 cards.

There are no keyboard shortcuts — everything is mouse-driven, by design (one
click on the map, one click on a button).

---

## Project architecture

```
                ┌──────────────────────────────────────────┐
                │                 app layer                  │
                │  main.py (entry) · loop.py (interactive)   │
                │  game.py (headless Game.tick())            │
                └──────────────────┬───────────────────────-┘
                                    │ uses
        ┌───────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────▼─────┐               ┌──────▼──────┐              ┌──────▼──────┐
   │  agents   │               │  game core   │              │  ui layer   │
   │ (decide)  │◄──────────────┤ (rules+state)├─────────────►│(input+draw) │
   └───────────┘               └──────────────┘              └─────────────┘
                                       ▲
                                       │ reused, no pygame
                                ┌──────┴──────┐
                                │  learning    │
                                │ (self-play)  │
                                └─────────────┘
```

- **`risk/game/`** — the rules engine. Pure Python, **no pygame import
  anywhere in this package** — fully testable and reusable headlessly.
- **`risk/agents/`** — every seat (human, random, heuristic, future RL) is a
  uniform callable: `agent(events, state) -> Action | None`.
- **`risk/app/`** — wires the rules engine + agents + a pygame window into a
  playable app. Split into small single-purpose collaborators (see below)
  rather than one big loop.
- **`risk/ui/`** — pygame rendering and click hit-testing, kept separate
  from both the rules and the agent decision logic.
- **`risk/learning/`** — `SelfPlay`, the AI-only driver used to generate
  training rollouts; the future GCN+DQN trainer builds on top of this
  instead of reimplementing the play loop.

A `State` is deliberately cheap to copy (`state.snapshot()`) and array-
indexed by a stable, sorted territory order (`BoardTopology`), so it can
later be turned into graph/tensor features for a GNN without redesigning
anything.

---

## Folder-by-folder tour

### `risk/constants.py` / `risk/ui_constants.py`

Two plain (no-class) modules holding every named constant in the project —
`from risk.constants import *` for anything rule-related (player limits,
starting armies, dice caps, card values, the exact Risk dice-probability
tables), `from risk.ui_constants import *` for anything visual (colors,
panel widths, AI-pacing durations). Splitting these out means a balance
change or a color tweak never requires hunting through gameplay/rendering
code for a hardcoded number.

### `risk/game/` — the rules engine

The single source of truth for what's legal and what happens. Nothing here
imports pygame.

- **`environment.py` — `Environment`.** Owns the one `State` for the whole
  match and *all* rule logic: `reset(settings)`, `current_state()`,
  `legal_actions()`, `step(action) -> StepResult`, `is_terminal()`,
  `winner()`. Internally dispatches to `_apply_reinforce` /
  `_apply_trade_in` / `_apply_attack` / `_apply_occupy` / `_apply_fortify`,
  handles dice resolution, continent-bonus calculation, card draw/trade-in,
  and elimination. This narrow API (`reset` / `legal_actions` / `step`) is
  the same one a future RL trainer drives.
- **`state.py` — `State`, `PendingAttack`.** The mutable per-match data:
  `owners[i]` / `armies[i]` (arrays aligned to `BoardTopology`'s territory
  order), `current_player_index`, `phase`, `reinforcement_budget`,
  `hands`, `eliminated`, `pending_attack`. `snapshot()`/`copy()` produce an
  independent deep-enough copy (safe to keep in a replay buffer);
  `to_dict()`/`from_dict()` round-trip through JSON for logging/replay.
- **`actions.py` — `Action` and subclasses.** Immutable, self-validating
  move objects: `ReinforcementAction`, `AttackAction`, `StopAttackAction`,
  `TradeInAction`, `OccupyAction`, `FortifyAction`. `ActionCodec.from_dict`
  rebuilds one from its `to_dict()` form (for replay/logging). Construction
  only checks shape (e.g. dice in 1..3) — actual legality (ownership,
  adjacency, budget) is `Environment`'s job.
- **`board_topology.py` — `BoardTopology`.** The static board graph, loaded
  from `Assets/RiskMap/risk_map_data.json`: territory list (sorted, stable
  index order), `neighbors()`/`are_adjacent()`, `continent_of()` /
  `territories_in()` / `continent_bonus()`, `owns_continent(owners,
  continent, player_id)`, and `edge_index()` — a ready-made `(src, dst)`
  adjacency view for a future GNN's graph input.
- **`card.py` — `Card`, `CardRules`.** `Card` is the territory-card value
  object (symbol + optional territory id, wilds have neither).
  `CardRules.is_valid_set()` / `find_valid_set()` / `validate_against_topology()`
  implement the trade-in rules.
- **`phase.py` — `Phase`.** The turn-phase `IntEnum`: `TRADE_IN,
  REINFORCE_PLACE, ATTACK, OCCUPY, FORTIFY, GAME_OVER, SETUP`. Also doubles
  as the DQN action-representation "stage" directly (`Docs/Action.md`) —
  no separate enum for that.
- **`player.py` — `Player`.** Immutable seat description: id, name, color,
  `agent_kind` (`human` / `random` / `raider` / `sentinel` / `empire`).
- **`settings.py` — `GameSettings`.** Immutable match config: the player
  roster, an optional RNG `seed` (the *only* source of randomness in a
  match — same seed replays identically), and rule toggles.

### `risk/agents/` — who decides each move

Every seat — human or AI — is the same shape: `agent(events, state) ->
Action | None`. AI agents ignore `events` and answer immediately; a human
agent consumes mouse `events` across many frames and returns `None` until a
full decision is assembled.

- **`base_agent.py` — `BaseAgent`.** The abstract contract every agent
  implements (`act`, `widgets`, `on_turn_start`/`on_turn_end`).
- **`random_agent.py` — `RandomAgent`.** Picks uniformly among
  `env.legal_actions()`. Used to validate AI seats end-to-end and as a
  weak training opponent.
- **`heuristic_agent.py` — `AttackAgent`, `BSRAgent`, `ContinentAgent`,
  `ShapeAgent`, `CompositeAgent`, and three ready-made personalities built
  on `CompositeAgent`: **`RaiderAgent`** (aggressive — attacks on thinner
  margins, expands fast), **`SentinelAgent`** (defensive — reinforces
  threatened borders, attacks only on strong odds), **`EmpireAgent`**
  (continent-focused — chases and defends continent bonuses). All of them
  rank legal moves using the exact Risk dice-probability math
  (`battle_win_probability`, memoized) plus tunable weights (attack odds,
  border-security ratio, continent value, compactness) — useful both as
  game opponents and as a non-RL baseline/curriculum for training.
- **`human_agent.py` — `HumanAgent`.** Owns a `HumanInputController` and a
  reference to the view; decodes mouse-click pygame events into controller
  calls and returns the assembled `Action` once the player completes one
  (`None` otherwise, meaning "still deciding").
- **`human_input.py` — `HumanInputController`, `HudActionPanelModel`,
  `HudButton`.** The pygame-free decision-builder behind `HumanAgent`: turns
  `on_territory_click(...)` / `on_hud_button(...)` / `on_hud_field(...)`
  calls into a complete, pre-validated `Action`, and exposes `widgets(state)`
  — a pure-data description of what the side panel should currently show
  (used by both the renderer and the controller's own tests).

### `risk/app/` — wiring the interactive app

Each stage of "turn settings into a playable window" is a separate,
narrowly-scoped piece:

| File | Responsibility |
|---|---|
| `setup.py` — `SetupStage` | **Pre-game.** Produces a `GameSettings`, either from the interactive setup screen or `SetupStage.default_settings(...)` for smoke tests / `--skip-menu`. |
| `factory.py` — `GameFactory`, `GameContext` | **Build.** `GameFactory.build(settings)` wires an `Environment` + one agent per seat + a `Game` — pygame-free, the exact seam headless training reuses. |
| `game.py` — `Game`, `TickResult` | The headless orchestrator: `tick()` asks the current agent for a move and applies it via `env.step(...)`; `play_until_terminal()` runs a full game with no rendering. Owns no rules. |
| `loop.py` — `AppLoop` | The interactive frame loop: poll events → ask the current agent → step the env → render. The environment decides whose turn is next, so swapping a seat between human and AI never touches this file. |
| `view.py` — `GameView` | Renders one frame (board + HUD) and resolves map clicks to territories. |
| `pacer.py` — `AITickPacer` | Paces AI moves in `play` mode so a human can actually see each one (no-op in `train` modes). |
| `marker.py` — `ActionMarker` | Tracks the "last AI move" highlight overlay, and the human-readable one-line/multi-line descriptions of an action shown in the side panel. |
| `main.py` | The CLI entry point: parse args → `SetupStage` → `GameFactory.build` → `AppLoop.run()` (or the headless `train-no-render` one-liner). |

### `risk/ui/` — input & rendering

Split from the agent/app logic so rendering changes never risk touching
rules:

- **`ui/input/hit_test.py` — `TerritoryHitTester`.** Maps a screen
  coordinate to a territory id via point-in-polygon, accounting for the
  board image being scaled/letterboxed into its on-screen rect.
- **`ui/input/init_screen.py` — `InitScreenState`.** The pygame-free setup-
  screen state machine: player count, names, colors, seat kinds,
  `can_start()` validation, `build_settings()`.
- **`ui/render/risk_map.py` — `RiskMapRenderer`.** Draws the base map image,
  per-territory ownership tint, army-count labels, continent bonus badges,
  and the AI action-marker overlay.
- **`ui/render/panels.py` — `HudPanel`.** Renders the side panel: the
  player/phase/budget table, the action-panel widgets described by
  `HudActionPanelModel` (including the card screen), and the persistent
  "Last move:" strip — and returns the clickable regions back to the loop.
- **`ui/render/init_screen_view.py`.** The pygame view + event loop for the
  setup screen described in [How to play](#the-setup-screen).

### `risk/learning/` — self-play & training

- **`self_play.py` — `SelfPlay`.** AI-only game driver sharing the same
  `agent(events, state)` contract as the interactive app:
  `SelfPlay.play_headless(ctx, on_step=...)` runs as fast as the CPU allows
  with no window (bulk training rollouts); `SelfPlay.play_rendered(ctx,
  on_step=...)` opens a window and draws every move with no AI pacing, so
  you can *watch* training without slowing it down. Both reject human seats
  outright (a human would stall the loop) and hand every transition to an
  optional `on_step(state_before, action, result)` hook as independent,
  safe-to-keep snapshots — `Environment` mutates one `State` object for
  the whole game, so this is the difference between a usable replay buffer
  and one that silently aliases the same final state. A future GCN+DQN
  trainer should subclass `SelfPlay` rather than reimplementing the loop.
  `main()` is a deliberately editable training scratch pad — build a
  roster, swap in your learning agent for one seat, run a driver — runnable
  via `python -m risk.learning.self_play`.
- **`graph_adapter.py` — `GraphAdapter`.** Converts a game snapshot into a
  `torch_geometric.data.Data` graph: one node per territory (continent
  one-hot + owner one-hot + army count), `edge_index` straight from
  `BoardTopology.edge_index()`, and a global `u` vector (whose turn, phase,
  cards, continent bonuses, reinforcement budget, eliminated players, ...).
  `GraphAdapter(topology, settings)` builds one adapter per game; call it
  every step with just the state — `adapter(state) -> Data`. See
  [Docs/GraphAdapter.md](Docs/GraphAdapter.md) for the full field-by-field
  layout and the design decisions behind it.
- **`action_encoder.py` — `ActionEncoder`.** `ActionEncoder(env)` then
  `encoder()` converts the env's current `legal_actions()` into a `[N, 4]`
  long tensor of `(stage, t1, t2, n)` rows for scoring `Q(s, a)` per
  legal action — the encoding each `Action` subclass exposes itself via
  `Action.dqn_index()` in `risk/game/actions.py`. See
  [Docs/Action.md](Docs/Action.md)'s "Representing actions for DQN"
  section for why this shape (score legal actions, not a fixed
  flat action-ID table) and the full per-stage tuple layout.

### `Temp/tests/` — the test suite

225 tests (plus one platform-conditional skip) covering board topology, state/actions round-tripping,
environment rules (reinforcement math, combat resolution, conquest,
elimination, winner detection, illegal-action rejection), agents, the human
input controller, hit-testing, app-level smoke tests, and a 10-seed
`SelfPlay.play_headless` fuzz test that plays full AI-only games to an
actual winner. `conftest.py`
holds the shared `make_settings()` / `make_env()` builders used across
files instead of each test file hand-rolling its own.

### `Assets/`, `Docs/`

- **`Assets/RiskMap/`** — the board image, label sprite sheet, and
  `risk_map_data.json` (territory names, adjacency, continents/bonuses,
  label/army-marker coordinates) that `BoardTopology` and
  `RiskMapRenderer` both load from.
- **`Docs/`** — reference docs for specific subsystems
  ([`BoardTopology.md`](Docs/BoardTopology.md), [`Action.md`](Docs/Action.md),
  [`GraphAdapter.md`](Docs/GraphAdapter.md)), how the test suite is organized
  ([`Testing.md`](Docs/Testing.md)), plus a running changelog of larger
  changes ([`RL-Prep-Changes.md`](Docs/RL-Prep-Changes.md)).

---

## Testing

```bash
python -m pytest Temp/tests -q
```

Pygame-dependent tests run headlessly via `SDL_VIDEODRIVER=dummy` (set
automatically at the top of the files that need it).

---

## Roadmap

The architecture exists specifically to make this last step a clean
addition rather than a rewrite. Done so far:

- ✅ A graph adapter (`risk.learning.graph_adapter.GraphAdapter`) turning a
  game snapshot into a `torch_geometric.data.Data` object for a GCN.
- ✅ An action representation (`risk.learning.action_encoder.ActionEncoder`)
  scoring legal actions rather than indexing a fixed action-ID table —
  the piece that makes a huge, variable-shaped action space tractable for
  DQN.
- ✅ Multi-step reinforcement — the engine now allows placing part of a
  turn's budget and continuing later, which was the one combinatorial
  action standing in the way of a clean per-action `Q(s, a)` design
  (see [Docs/RL-Prep-Changes.md](Docs/RL-Prep-Changes.md)).

Still open:

- The actual GCN + per-stage scoring heads — `Q(s, a) = head[stage](graph_embedding(s), t1, t2, n)`,
  per [Docs/Action.md](Docs/Action.md)'s "Network wiring" section.
- A DQN training loop built on `risk.learning.SelfPlay`, training against
  the bundled heuristic agents (`RandomAgent` → `RaiderAgent` /
  `SentinelAgent` / `EmpireAgent`) as an opponent curriculum.
- Swapping the trained agent into `risk/app/factory.py`'s roster to play
  against it interactively, with no changes to the app layer.
