# RISK — Refactor Plan

Goal: make the structure easy to understand, keep a **simple game loop usable
for agent training**, and remove the duplicated/tangled code in `main.py`. This
plan also resolves the `graphics/` vs `ui/` overlap.

The guiding principle for the whole refactor:

> **Separate code by its dependency on pygame, not by "visual vs control".**
> Everything that can run headless (rules, input logic, agents) stays
> pygame-free so it is reusable for training. Everything that touches pygame
> (rendering, the window loop) is isolated at the edge.

---

## The three tiers (why `game/` vs `app/`, and what's wrong today)

The most important boundary in the project is **rules vs running a session**.
But "App" currently lumps together two things that live at different levels, and
that is a big reason `main.py` feels tangled. The real layering is three tiers:

| Tier | What it is | Where today | pygame? | Reused for training? |
|------|------------|-------------|---------|----------------------|
| **1. Rules** | what is legal, what happens next (`Environment`, `State`, `Action`, board, cards) | `risk/game/` | no | ✅ yes |
| **2. Orchestration** | run agents against the env (`Game.tick`, `play_until_terminal`) | `risk/app/game.py` | no | ✅ yes |
| **3. Presentation** | window, input, rendering, pacing | `risk/app/main.py` | yes | ❌ no |

The dependency arrow only ever points **down**:

```
presentation (main.py)  ──uses──►  orchestration (Game)  ──uses──►  rules (Environment)
        ▲ pygame                        ▲ pygame-free              ▲ pygame-free
        └──────── these two are what training reuses; presentation is thrown away ┘
```

Why this matters for the refactor:
- **Tiers 1+2 are the trainable core.** Training wants millions of fast headless
  games — exactly `build_game(...)` + `Game.play_until_terminal()`. The pygame
  tier must never be imported on that path.
- **`app/` mixes tier 2 and tier 3.** `Game` (tier 2) is pure orchestration and
  is arguably closer to "core" than to "app", yet it sits beside `main.py`
  (tier 3). Part B's `build_game` factory makes this explicit: it produces the
  headless tiers 1+2 (`GameContext`), and only `AppLoop` adds tier 3 on top.
- **Different reasons to change.** Rules are stable; presentation (pygame today,
  maybe a web UI or notebook viz later) changes independently. Keeping the arrow
  one-directional means a UI tweak can't break combat resolution.

This is the lens for every part below: each extraction either keeps code in the
pygame-free tiers (so training can reuse it) or pushes it cleanly into tier 3.

---

## Part A — Merge `graphics/` into `ui/`

### Why
- `graphics/` holds a single file (`risk_map.py`). It doesn't earn a package.
- "Rendering lives in graphics" isn't actually true today: `panels.py` and
  `init_screen_view.py` also render with pygame. Rendering is already split.
- `RiskMapRenderer` is just another **view**, like `HudPanel`. It belongs next
  to the other views.

### Target layout
Split `ui/` by the pygame boundary instead:

```
risk/ui/
├── input/                # pygame-FREE: testable, training-safe
│   ├── human_input.py    # HumanInputController
│   ├── hit_test.py       # TerritoryHitTester
│   └── init_screen.py    # InitScreenState
└── render/               # pygame views
    ├── risk_map.py       # RiskMapRenderer   (moved from graphics/)
    ├── panels.py         # HudPanel
    └── init_screen_view.py
```

`risk/graphics/` is deleted after the move.

### Steps
1. Move `risk/graphics/risk_map.py` → `risk/ui/render/risk_map.py`.
2. Move `panels.py`, `init_screen_view.py` → `risk/ui/render/`.
3. Move `human_input.py`, `hit_test.py`, `init_screen.py` → `risk/ui/input/`.
4. Add `__init__.py` to `ui/input/` and `ui/render/` (optionally re-export the
   main classes from `risk/ui/__init__.py` so import sites stay short).
5. Update imports:
   - `from risk.graphics.risk_map import RiskMapRenderer`
     → `from risk.ui.render.risk_map import RiskMapRenderer`
   - `from risk.ui.human_input import HumanInputController`
     → `from risk.ui.input.human_input import HumanInputController`
   - …and the rest.
6. Delete the empty `risk/graphics/` package.
7. Run the test suite (`Temp/tests/`) and a `--skip-menu --max-ticks` smoke run.

> Optional lighter version: if subfolders feel heavy, just move `risk_map.py`
> into `risk/ui/` flat and delete `graphics/`. The input/render split is the
> recommended version because it makes the "what's safe for training" boundary
> explicit.

---

## Part B — Separate pre-game setup from the running game

The biggest conceptual win: today `run()` mixes **two completely different
stages** in one function — choosing *who plays* (the menu / defaults) and
*playing the game* (the loop). They should be distinct stages connected by a
single data contract: `GameSettings`.

```
  ┌─────────────┐   GameSettings   ┌────────────┐   GameContext   ┌──────────┐
  │  PRE-GAME   │ ───────────────► │   BUILD    │ ──────────────► │   GAME   │
  │  (setup)    │   (data only)    │ (wiring)   │  (env+agents)   │  (run)   │
  └─────────────┘                  └────────────┘                 └──────────┘
   menu / defaults                pygame-free factory          loop OR training
```

### B1. Three stages, one contract
- **Pre-game (setup)** — produce a `GameSettings`. Source is either the
  interactive menu (`run_init_screen`) or defaults (`--skip-menu`, smoke tests).
  Owns the menu's pygame view, but its *only output is data*. No rules, no loop.
- **Build (factory)** — turn `GameSettings` into a runnable game. Pure wiring,
  **pygame-free**. This is the exact seam training reuses.
- **Game (run)** — consume the built game: the interactive `AppLoop`, or
  headless `game.play_until_terminal()` for training.

### B2. Proposed files (in `risk/app/`)
| File | Contents | pygame? |
|------|----------|---------|
| `setup.py` | `run_setup(screen, args) -> Optional[GameSettings]`, `default_settings(n, seed)` | yes (menu only) |
| `factory.py` | `build_game(settings) -> GameContext`, `_build_agents(settings)` | no |
| `loop.py` | `AppLoop` (the interactive loop, see Part C) | yes |
| `main.py` | arg parse + orchestrate the three stages | thin |

`GameContext` is a small dataclass — the build output, shared by play & training:
```python
@dataclass
class GameContext:
    settings: GameSettings
    env: Environment
    agents: list[BaseAgent]
    game: Game
```

### B3. Target shape of `main.run()`
```python
def run(...):
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        settings = run_setup(screen, args)      # PRE-GAME -> data or None
        if settings is None:
            return 0                             # user closed the menu
        ctx = build_game(settings)               # BUILD (pygame-free)
        AppLoop(ctx, screen, ...).run()          # GAME
    finally:
        pygame.quit()
```
Training never touches setup or the loop — it calls the same factory:
```python
ctx = build_game(default_settings(n=3, seed=0))
winner = ctx.game.play_until_terminal()
```

### B4. Why this helps
- Pre-game and game no longer share a function or mutable local state.
- `build_game` is the single seam between **play** and **training** — change the
  roster source (menu, CLI, config file, later a web UI) without touching the game.
- Each stage is independently testable: setup returns data, build is pure, the
  loop takes a ready-made `GameContext`.
- It removes the `train-no-render` fork in `main.py` (see Part C1): headless mode
  is just `build_game(...)` + `play_until_terminal()`.

---

## Part C — Untangle `main.py`

Today `run()` is one ~100-line `while` loop doing: event handling, AI pacing,
marker bookkeeping, game stepping, and rendering — plus a duplicated headless
path (`_run_headless`). We break it into small pieces with single
responsibilities.

### C1. One loop, not two
`Game.tick()` / `Game.play_until_terminal()` already *is* the simple trainable
loop. The `train-no-render` path in `main.py` duplicates that. Remove the fork:

- Headless training uses `build_game(...)` + `Game` directly (no `main.py`).
- `main.py` keeps only the **interactive** pygame loop.

### C2. Extract small collaborators (all in `risk/app/`)
Pull the tangled concerns out of `run()` into focused, individually testable
units:

| New unit | Responsibility | pygame? |
|----------|----------------|---------|
| `PygameEventRouter` | translate pygame events → controller calls (HUD regions, territory clicks, quit/escape) | yes |
| `AITickPacer` | decide *whether* an AI tick may run this frame; owns `ai_delay` timing | no |
| `ActionMarker` | track the "last AI action" highlight + expiry | no |
| `GameView` (or `Presenter`) | one `render(state)` call: board + HUD + action panel, returns HUD regions | yes |
| `AppLoop` | the thin `while running:` driver that wires the above + `Game` | yes |

`AppLoop` takes the `GameContext` from Part B and runs the interactive loop.

### C3. Target shape of the loop
```python
while running:
    now = clock_ms()
    events.process()              # PygameEventRouter -> controller -> HumanAgent
    controller.on_turn_change(env.current_state())

    if not game.is_terminal() and pacer.may_tick(now, current_agent_is_ai):
        result = game.tick()
        if result.step is not None and is_ai:
            marker.set(_action_territories(...), now)
            pacer.note_tick(now)

    marker.expire(now)
    view.render(env.current_state(), marker.highlights(), terminal_message)
    clock.tick(60)
```
Each line is now one responsibility, and the timing math, marker state, and
rendering are no longer inlined.

### C4. Fix the small smells
- Call `controller.on_turn_change()` **once** per frame (currently twice).
- Move `_owners_dict` / `_armies_dict` next to the renderer (they're view
  concerns, not app concerns).
- Keep `_action_territories` with `ActionMarker`.

---

## Part D — Human input timing (optional, later)

Right now a human action sits in `HumanAgent` until the next `tick()` reads it,
so decision timing is coupled to frame rate. This works, but if it becomes
awkward we can have `HumanInputController` push the action and let `AppLoop`
tick immediately on submit. **Defer** — not needed for the structure cleanup or
for training.

---

## Suggested order of work

1. **Part A** (graphics → ui) — mechanical, low risk, immediately clarifies the
   layout. Land it first.
2. **Part B** — split pre-game / build / game: add `setup.py` + `factory.py`
   (`build_game` + `GameContext`) and reduce `run()` to the three-stage shape.
   This also deletes the `train-no-render` fork (C1).
3. **Part C2/C3** — extract `GameView`, then `PygameEventRouter`, then
   `AITickPacer` + `ActionMarker`, then collapse the loop into `AppLoop`.
4. **Part C4** — tidy the small smells as you go.
5. **Part D** — only if needed.

After each step: run `Temp/tests/` and a `--skip-menu --max-ticks 200` smoke
run to confirm nothing regressed.

---

## What does NOT change
- The game core (`risk/game/`) — already clean and pygame-free.
- The agent interface (`risk/agents/`).
- `Game` in `risk/app/game.py` — it stays the canonical training loop; the
  refactor makes `main.py` *use* it instead of duplicating it.
