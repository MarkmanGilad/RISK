# RISK — Refactor Plan: Agent-Driven Game Loop

> Status: **proposal only — no code changes yet.**
> Goal: rebuild the main loop around the Tic-Tac-Toe pattern you showed, where
> **every player is a callable agent** that receives input and returns an action,
> the action is sent to the environment, and then play moves to the next agent.
> Swapping a seat from human to AI becomes *just changing which class you
> instantiate* — the loop never changes.

## Decisions (locked in)

1. **`AI_Agent` holds the `env`** and calls `env.legal_actions()` itself, keeping
   the uniform `agent(events, state)` signature.
2. **Fold the "ask-agent → step" orchestration into the loop.** `Game` is no
   longer used for the interactive path; headless training calls the same loop
   with an empty `events` source and no view.
3. **Move `HumanInputController` wholesale into `HumanAgent`.** The human's
   in-progress decision state machine becomes part of the agent.
4. **"Show the last executed action" is a loop+view concern, not an agent concern**
   (see §6.1). The loop keeps the single last applied action; the view renders it.
   For a human move the panel already reflects their clicks; for an AI move the
   loop hands the action to the panel. This does not change decision 3 — the
   human agent still owns only *its own* in-progress decision.

---

## 1. The pattern you want (from Tic-Tac-Toe)

```python
action = player(events, env.state)   # agent decides (or returns None)
if action:
    env.move(action)                 # environment executes
    player = switch_players(player)  # next seat (human or AI)
graphics(env.state)                  # draw every frame
```

Three properties to carry over to Risk:

1. **One uniform call.** `player(events, state)` — the loop does not care whether
   `player` is a `Human_Agent`, `Random_Agent`, or `AI_Agent`.
2. **The agent returns the action.** Humans translate clicks into an action;
   AIs compute one. Same return type.
3. **`None` means "not ready yet."** The loop just keeps spinning and rendering
   until an action comes back, then advances the game.

---

## 2. How Risk differs from Tic-Tac-Toe (and what that means)

These differences are why the current code grew a separate
`HumanInputController`. The plan keeps your pattern but accounts for them.

| Tic-Tac-Toe | Risk |
|-------------|------|
| One action = one click | One action can take **several clicks** (pick from-territory → to-territory → dice → confirm) |
| Players strictly alternate | A player takes **many actions in a row** across phases (REINFORCE → ATTACK → … → FORTIFY) before the turn passes |
| `switch_players()` toggles | The **Environment already advances** `current_player_index`; "switch" = just read whose turn it is now |
| Agent renders its own piece | Risk has a board + HUD action panel that must reflect a half-built decision |

Key consequence: **"switch player" is env-driven.** After `env.step(action)`,
the environment sets the next phase/player. The loop doesn't toggle anything — it
re-reads `state.current_player_index` and calls *that* seat's agent. The same
human keeps being asked while it's still their multi-phase turn. This is exactly
your "you can stay the human if he has another action" — Risk gets it for free
from the environment.

---

## 3. Target agent contract

Unify everything behind a single callable. `act` receives **both** the events and
the state; each agent uses what it needs.

```python
class BaseAgent:
    def __call__(self, events, state) -> Optional[Action]:
        return self.act(events, state)

    def act(self, events, state) -> Optional[Action]:
        ...
```

- **`RandomAgent` / `AI_Agent`** — ignore `events`, look at `state` (+ legal
  actions, which they can get from the env they hold), return an action
  immediately. Never return `None` on their turn.
- **`HumanAgent`** — ignore `state` for *deciding*, consume `events` (mouse
  clicks) to build a decision, return `None` until the decision is complete,
  then return the finished `Action`.

This is the heart of your request: **the seat's class determines behavior; the
loop is identical.** Swap `HumanAgent(...)` for `AIAgent(...)` and the loop runs
itself.

> Note: this changes the signature from today's `act(state, legal_actions)` to
> `act(events, state)`. Agents that need legal actions hold a reference to the
> `env` (as they do in your TTT code) and call `env.legal_actions()` themselves.

---

## 4. Target main loop

```python
def run(self):
    running = True
    while running:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                running = False

        state = self.env.current_state()
        agent = self.agents[state.current_player_index]   # whose turn (env-driven)

        action = agent(events, state)                     # uniform call
        if action is not None:
            self.env.step(action)                         # environment executes
            self.last_action = action                     # for the HUD readout (§6.1)
            # No manual switch: env advanced phase/player. Next frame asks
            # whoever is current now (same human across phases, or next seat).

        self.view.render(
            self.env.current_state(),
            widgets=self._agent_widgets(agent),   # human builder panel (or empty)
            last_action=self.last_action,         # the last action the env executed
        )
        self.clock.tick(60)
```

Compare to TTT: `agent(events, state)` ≙ `player(events, state)`,
`env.step` ≙ `env.move`, and "switch" is implicit because the Risk environment
owns turn/phase advancement.

Optional extras that stay (kept out of the core 3 lines):
- **AI pacing** — wrap the AI call so it only fires every N ms in `play` mode
  (today's `AITickPacer`).
- **Action marker** — highlight the AI's last move (today's `ActionMarker`).
- **End-of-game linger** — `pygame.time.wait(...)` then stop.

These become small wrappers around `action = agent(...)`, not loop logic.

---

## 5. `HumanAgent` redesign (absorb the input controller)

Today the multi-click logic lives in `HumanInputController`
([risk/ui/input/human_input.py](risk/ui/input/human_input.py)) and `HumanAgent`
only holds a `_pending` action that the controller fills in. The new design
**moves that state machine into `HumanAgent`** so the human truly "handles the
input/output and returns the action."

```python
class HumanAgent(BaseAgent):
    def __init__(self, player_id, env, view):
        super().__init__(player_id)
        self.env = env
        self.view = view          # for hit-testing clicks -> territory, + HUD
        self._builder = HumanDecision(env)   # the per-decision state machine

    def act(self, events, state):
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN:
                self._builder.on_click(e.pos, e.button, state)
        action = self._builder.take_completed_action()  # None until ready
        if action is not None:
            self._builder.reset()
        return action

    def widgets(self, state):
        return self._builder.widgets(state)   # HUD action panel model
```

- `HumanDecision` is essentially today's `HumanInputController` renamed and owned
  by the agent: it tracks `selected_from`, `attack_dice`, `pending_placements`,
  etc., and exposes `take_completed_action()` returning `None` until the player
  confirms.
- **Returning `None` until all clicks are done is the chosen solution** — it maps
  cleanly onto your TTT loop, which already tolerates `None`.

### Alternative solution (if you dislike `None`-until-done)
If you'd rather not keep partial state inside the agent across frames, two options:

- **(a) Action builder object.** The loop holds a `decision` object; the human
  agent fills it incrementally and the loop checks `decision.is_complete()`.
  Functionally identical to `None`-until-done, just makes the partial state
  explicit and inspectable.
- **(b) Generator/coroutine agent.** `act` is a generator that `yield`s `None`
  while collecting clicks and `return`s the final action. Elegant, but more
  machinery than the `None` approach needs. Not recommended for now.

Recommendation: **`None`-until-done** (matches TTT, least machinery).

---

## 6. Rendering & the HUD action panel

In TTT the agent renders itself. Risk is similar but the board is shared, so:

- The **loop renders the board every frame** (ownership, armies) — unchanged.
- The **HUD action panel** (buttons/sliders for the human's half-built move) is
  owned by whichever agent is acting. The loop asks the current agent for its
  `widgets(state)`; AI agents return an empty/info-only panel, the human returns
  its live builder model.
- Click hit-testing (`pixel -> territory`) stays in the view/hit-tester; the
  human agent calls `view.territory_at(pos)` from inside `act`.

So "output" (what the human sees while choosing) is driven by the human agent's
builder state, exactly as you asked.

### 6.1 Showing the last executed action

When a human is at the table you want the HUD to also show **what just
happened** — specifically, the **single last action the environment executed**.
This is **not** something `HumanAgent` can provide for an AI move — an agent only
knows its own decision. The owner of "the last executed action" is the **loop**,
because it observes every `env.step`.

Design (kept deliberately simple — one action, not a log):
- The loop keeps a single **`last_action`** value and overwrites it right after
  each successful `env.step(action)`.
- `view.render(state, widgets, last_action)` passes it to the panel:
  - **Human just moved** — the panel already reflects what they clicked; nothing
    extra needed (the builder showed the move as it was assembled).
  - **AI just moved** — the loop/env hands the chosen `Action` to the panel so it
    can display it ("P2 attacked Ural → Siberia (3 dice)"). The existing board
    **`ActionMarker`** highlight is the visual twin and also stays loop-owned.
- The display text is derived from the `Action` object itself (it already carries
  territories/dice), so **no agent change is needed**.

This keeps the clean split:
- **`HumanAgent`** → the in-progress human decision (builder widgets).
- **Loop** → the single last executed action.
- **View/HUD** → renders the human's builder panel, and for an AI move shows the
  last action the loop handed it.

So your requirement is satisfied without weakening decision 3: the human's own
move shows through their builder; the AI's move is pushed to the panel by the
loop.

---

## 7. Impact on existing files

| File | Change |
|------|--------|
| `risk/agents/base_agent.py` | new signature `act(events, state)`; add `__call__` |
| `risk/agents/random_agent.py` | ignore `events`; pull `legal_actions` from `env` |
| `risk/agents/human_agent.py` | absorb the input state machine; own `widgets()` |
| `risk/ui/input/human_input.py` | becomes `HumanDecision` owned by `HumanAgent` (move, don't duplicate) |
| `risk/app/loop.py` | shrink to the §4 loop; hold the single `last_action`; drop the separate controller wiring |
| `risk/app/events.py` | mostly gone — events flow straight into the agent; only QUIT/ESC stays in the loop |
| `risk/app/view.py` | `render(state, widgets, last_action)` — widgets from the current agent, last action from the loop |
| `risk/app/factory.py` | agents now need `env` (+ `view` for humans) at construction |
| `Game` (`risk/app/game.py`) | **removed from the interactive path** (decision 2); ask-agent → step now lives in the loop. Keep only if a separate headless helper is still wanted |

Training is unaffected in spirit: an all-AI roster with no `view` runs the same
loop (or `Game.play_until_terminal()`), since AI agents never touch `events`.

---

## 8. What stays the same
- The **rules engine** (`risk/game/`) — untouched.
- The **env-driven turn/phase model** — it's what gives you "stay on the same
  human for multiple actions" for free.
- **Determinism / training reuse** — AI agents still decide from `state` +
  `legal_actions`; passing an empty `events` list makes the loop fully headless.

---

## 9. Open questions before coding

All three original questions are now answered in **Decisions** at the top:
1. `AI_Agent` holds `env`. ✅
2. Fold ask-agent → step into the loop; training runs the same loop with no view. ✅
3. Move `HumanInputController` wholesale into `HumanAgent`. ✅

Remaining smaller choices to confirm while coding:
- **Action text source**: derive the readout from the `Action` object (no agent
  change), vs. an optional `describe()` on agents. (Recommend: derive from
  `Action`.)
- **Headless training**: keep a thin `play_until_terminal()`-style helper that
  drives the loop with empty events, or call the loop directly with a max-steps
  cap? (Recommend: a thin helper so training code stays one line.)