# RISK — Project Structure

A summary of how the project is organized today, what each module does, how the
pieces connect at runtime, and where the code is tangled. This is a **reference
for the refactor discussion** — it describes the current state, not a target.

---

## 1. High-level picture

The codebase is split into a **pygame-free game core** and a **pygame UI/app
layer** on top of it.

```
                ┌────────────────────────────────────────────┐
                │                  app layer                   │
                │   main.py  (pygame window + event loop)      │
                │   game.py  (Game: headless tick orchestrator)│
                └───────────────┬──────────────────────────────┘
                                │ uses
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
   ┌────▼─────┐          ┌──────▼──────┐             ┌──────▼──────┐
   │  agents  │          │  game core   │            │  ui + graphics │
   │ (decide) │          │ (rules+state)│            │ (input+render) │
   └──────────┘          └──────────────┘            └────────────────┘
```

- **Game core** (`risk/game/`) — pure rules engine, no pygame. Testable in isolation.
- **Agents** (`risk/agents/`) — uniform callables `agent(events, state) -> Action | None`
  (human, random, future RL). The human seat owns its input controller.
- **App** (`risk/app/`) — `AppLoop` (interactive agent loop) + `Game` (headless loop) + thin `main.py`.
- **UI / Graphics** (`risk/ui/`) — input hit-testing and rendering.
- **Learning** (`risk/learning/`) — AI-only self-play drivers (`play_headless` /
  `play_rendered`) sharing the `agent(events, state)` contract, for training.

---

## 2. Directory map

```
risk/
├── app/
│   ├── game.py        # Game: wires Environment + Agents, exposes tick()  (headless)
│   ├── setup.py       # PRE-GAME: produce GameSettings (menu or defaults)
│   ├── factory.py     # BUILD: build_game(settings) -> GameContext (pygame-free)
│   ├── view.py        # GameView: render one frame + resolve map clicks
│   ├── pacer.py       # AITickPacer: gate AI ticks for visibility (no pygame)
│   ├── marker.py      # ActionMarker + describe_action/action_report: overlay + last-action text (no pygame)
│   ├── loop.py        # AppLoop: agent-driven frame loop (events -> agent -> env -> view)
│   └── main.py        # thin entry point: arg parse -> setup/build/run
├── game/
│   ├── environment.py # rule engine: legality, phases, combat, cards      (~420 lines)
│   ├── state.py       # immutable per-turn State snapshot                  (~150 lines)
│   ├── actions.py     # immutable Action objects                          (~100 lines)
│   ├── board_topology.py # static board graph (territories/adjacency)     (~150 lines)
│   ├── card.py        # territory cards + set validation
│   ├── constants.py   # rule constants (no state)
│   ├── phase.py       # Phase enum
│   ├── player.py      # Player (immutable seat description)
│   └── settings.py    # GameSettings (immutable match config)
├── agents/
│   ├── base_agent.py  # BaseAgent ABC: __call__/act(events, state), widgets()
│   ├── human_agent.py # HumanAgent: owns HumanInputController, decodes events via view
│   ├── human_input.py # HumanInputController + HUD action-panel view-model
│   └── random_agent.py# RandomAgent: holds env, returns a uniform random legal action
├── ui/
│   ├── input/         # pygame-FREE input layer
│   │   ├── human_input.py # re-export shim -> risk.agents.human_input
│   │   ├── hit_test.py    # map click -> territory (point in polygon)
│   │   └── init_screen.py # setup-screen state machine (headless)
│   └── render/        # pygame views
│       ├── risk_map.py        # RiskMapRenderer: board, owners, armies, markers
│       ├── panels.py          # HUD rendering + action panel widgets
│       └── init_screen_view.py # pygame setup screen
└── learning/          # AI-only self-play drivers for training
    └── self_play.py   # play_headless / play_rendered + main() training scratch pad
```

> Note: `risk/graphics/` was merged into `risk/ui/render/`. The old monolithic
> `main.py` loop was split into `setup.py` / `factory.py` / `view.py` /
> `pacer.py` / `marker.py` / `loop.py`. The loop is now **agent-driven**: every
> seat is a uniform `agent(events, state)` callable, so the obsolete
> `events.py` router was removed and `HumanInputController` moved into
> `risk/agents/`. See `Docs/RefactorPlan.md` and `Docs/RefactorPlan_AgentLoop.md`.

---

## 3. Game core (`risk/game/`)

The rules live here and **never import pygame**. This is the part you can reuse
directly for agent training.

### `environment.py` — the rule engine
`Environment` owns the full ruleset and the single source of truth (`_state`).

Public interface (the RL-style API):
- `reset(settings) -> State` — deal territories, set starting armies, enter `REINFORCE`.
- `current_state() -> State`
- `legal_actions() -> list[Action]` — representative legal actions for the current phase.
- `step(action) -> StepResult` — apply an action, mutate state, return new state + info.
- `is_terminal()`, `winner()`.

Internally it dispatches by phase: `_apply_reinforce`, `_apply_trade_in`,
`_apply_attack`, `_apply_occupy`, `_apply_fortify`, plus combat dice resolution,
card draw/trade, continent bonuses, and elimination checks. During REINFORCE,
`legal_actions()` also enumerates legal `TradeInAction`s, so agents (RL/random)
can cash card sets, not just humans.

### `state.py` — the State snapshot
Per-turn data, designed to be copied cheaply for rollouts:
- `owners[i]`, `armies[i]` — arrays indexed by **sorted territory order**.
- `current_player_index`, `phase`, `reinforcement_budget`.
- `hands[player]`, `eliminated`, `pending_attack` (mid-conquest → `OCCUPY`).
- `initial(...)`, `copy()`/`snapshot()`, `to_dict()`/`from_dict()`.

### `actions.py` — Action objects
Immutable, serializable: `ReinforcementAction`, `AttackAction`, `StopAttackAction`,
`TradeInAction`, `OccupyAction`, `FortifyAction`. Each validates its own shape and
can be serialized for replay/logging. `TradeInAction(card_indices)` cashes a
three-card set from the current player's hand during REINFORCE.

### `board_topology.py` — static board graph
`BoardTopology` loads `Assets/RiskMap/risk_map_data.json` (42 territories, 6
continents). Provides `neighbors()`, `are_adjacent()`, `continent_of()`,
`continent_bonus()`, index↔name mapping, and `edge_index()` for future GNNs.

### Smaller modules
- `phase.py` — `Phase` enum: `SETUP, REINFORCE, ATTACK, OCCUPY, FORTIFY, GAME_OVER`.
- `card.py` — `Card` + `is_valid_set()` / `find_valid_set()`.
- `constants.py` — player limits, starting armies, dice, card set values.
- `player.py` — `Player` (id, name, color, agent_kind).
- `settings.py` — `GameSettings` (player roster, seed, rule toggles).

---

## 4. Agents (`risk/agents/`)

A clean, composable interface — ready for an RL agent to slot in.

```python
class BaseAgent(ABC):
    def __init__(self, player_id: int): ...
    @abstractmethod
    def act(self, events: Sequence[object], state: State) -> Optional[Action]:
        """Return an action, or None if more input is needed (humans)."""
```

Every seat is a uniform callable: `agent(events, state) -> Action | None`.

- `RandomAgent` — holds the `env`, ignores `events`, returns a uniformly random
  legal action immediately (never `None` while legal actions exist).
- `HumanAgent` — owns a `HumanInputController` and a reference to the `view`;
  `act()` decodes mouse `events` into the controller and returns the assembled
  action once complete, otherwise `None` ("still waiting for the human").

---

## 5. App layer (`risk/app/`)

### `game.py` — `Game` (the clean part)
`Game` wires `Environment` + agents and exposes a one-step `tick()`:

```python
def tick(self) -> TickResult:
    if self.is_terminal():
        return TickResult(step=None, waiting_on_player=-1)
    state = self.env.current_state()
    pid = state.current_player_index
    agent = self.agents[pid]
    chosen = agent((), state)              # uniform callable; AI reads env itself
    if chosen is None:                     # human not ready yet
        return TickResult(step=None, waiting_on_player=pid)
    result = self.env.step(chosen)
    self.history.append(chosen)
    return TickResult(step=result, waiting_on_player=pid)
```

It owns no rules and no rendering — the simple, trainable loop. The interactive
`AppLoop` runs the same ask -> step with pygame events and a view;
`play_until_terminal()` runs a full headless game.

### The three stages: `setup.py` → `factory.py` → `loop.py`
The old monolithic `main.run()` was split into three stages connected by data:

- **`setup.py`** (PRE-GAME) — `run_setup(...)` / `default_settings(...)` produce a
  `GameSettings`. Owns the menu, but its only output is data.
- **`factory.py`** (BUILD) — `build_game(settings) -> GameContext` wires
  `Environment` + agents + `Game`. **Pygame-free**; this is the exact seam
  headless training reuses.
- **`loop.py`** (`AppLoop`) — the interactive frame loop. Each frame it polls
  events, asks the current `agent(events, state)` for a move, applies it via
  `env.step(...)`, and renders. The environment decides whose turn is next, so
  swapping a human seat for an AI seat changes nothing in the loop.

`AppLoop` delegates each former concern to a focused unit:

| Unit | File | Responsibility | pygame? |
|------|------|----------------|---------|
| `GameView` | `view.py` | render one frame (board + HUD), resolve map clicks, expose HUD click regions | yes |
| `AITickPacer` | `pacer.py` | gate AI ticks for visibility | no |
| `ActionMarker` | `marker.py` | last-action board highlight + `describe_action` / `action_report` text | no |

Mouse events flow straight into the current agent; only `QUIT`/`ESC` stay in the
loop. The former `PygameEventRouter` is gone — the `HumanAgent` decodes its own
clicks using the view.

### `main.py` — thin entry point
`run(...)` now just: init pygame → `run_setup` (data) → `build_game` (wiring) →
`AppLoop(...).run()`. The `train-no-render` mode is a one-liner over
`build_game(...) + play_until_terminal()` — no duplicated loop.

---

## 6. UI & graphics

- `agents/human_input.py` — `HumanInputController`: pygame-free, turns clicks/buttons
  into `Action`s. Holds ephemeral UI state (`pending_placements`, `selected_from/to`,
  `attack_dice`, `occupy_count`, `show_cards`/`selected_cards`) and exposes
  `widgets(state)` as a pure HUD model. The model carries an optional **card
  screen** (toggled by the current human's "Cards" button) that lists the hand and,
  during REINFORCE, lets the player select a valid set and trade it in
  (`TradeInAction`). Owned by the `HumanAgent`; `ui/input/human_input.py` is a
  re-export shim.
- `ui/input/hit_test.py` — `TerritoryHitTester`: screen coords → territory via
  point-in-polygon, accounting for blit rect + image scaling.
- `ui/render/panels.py` — `HudPanel`: renders the player table / phase / budget and
  the action panel widgets (including the card screen); returns clickable regions
  back to the loop. The side panel is `HUD_WIDTH` (360px) wide. The
  "Last move:" area shows `action_report` lines (e.g. an attack's dice rolls and
  per-side casualties) and persists until the next action replaces it.
- `ui/input/init_screen.py` / `ui/render/init_screen_view.py` — setup-screen state
  machine and its pygame view (`run_init_screen(screen) -> GameSettings | None`).
- `ui/render/risk_map.py` — `RiskMapRenderer`: draws base map, ownership tint,
  army labels, and AI action markers; handles scaling.

---

## 7. Runtime flow

```
main.run()
  ├─ run_setup(...) ─► GameSettings        # PRE-GAME (setup.py)
  ├─ build_game(settings) ─► GameContext   # BUILD (factory.py, pygame-free)
  └─ AppLoop.run():                        # GAME (loop.py)
       while running:
         ├─ events = pygame.event.get()           # QUIT/ESC handled here
         ├─ agent = agents[state.current_player_index]
         ├─ action = agent(events, state)         # AI: immediate · Human: when ready
         ├─ if action: env.step(action)           # env picks the next seat
         └─ GameView.render(state, agent.widgets(state), marker, last_action)
```

The headless training path skips setup and the view, reusing the same
ask -> step over `Game`:

```
build_game(default_settings(...)).game.play_until_terminal()   # no pygame, no UI
```

`risk/learning/self_play.py` packages this for training with two drivers that
share the `agent(events, state)` contract:

- `play_headless(ctx, *, max_steps, on_step=None)` — bare ask -> step loop, no
  window; rejects human seats; optional `on_step(state, action, result)` hook
  for collecting transitions. Returns the winner.
- `play_rendered(ctx, *, fps=0, on_step=None, ...)` — opens a pygame window and
  renders every frame (no AI pacing) so you can watch AI/AI training; `fps<=0`
  runs at maximum speed. ESC/QUIT stops.

`main()` is an editable, code-only training scratch pad (build context, swap
agents per seat, run a driver). The file self-bootstraps `sys.path`, so it runs
via `python -m risk.learning.self_play`, the VS Code Run button, or F5.

---

## 8. Strengths to preserve

- The **core is already decoupled**: `Environment` + `State` + `Action` are
  pygame-free and seedable (reproducible).
- `Game.tick()` is a clean, RL-friendly step function.
- The agent interface is minimal and extensible.
- State is array-indexed by sorted territory → friendly to tensor/GNN encodings.

## 9. Pain points (addressed by the agent-loop refactor)

1. **God function** — `main.run()` was split into `setup` / `factory` / `loop`
   plus the `GameView` / `AITickPacer` / `ActionMarker` collaborators.
2. **Two loops** — interactive and headless now share one `agent(events, state)`
   -> `env.step` contract; `train-no-render` reuses `build_game` + `Game`.
3. **Async-ish human input** — formalized: the `HumanAgent` consumes `events`
   each frame and returns `None` until its decision is complete.
4. **Event routing** — the separate router is gone; each agent decodes its own
   input, so a seat can be swapped human↔AI with no loop changes.
