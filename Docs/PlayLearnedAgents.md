# Interactive learned-agent play plan

## Goal

Let a user start a normal interactive game from **Risk - New Game** with any
mixture of human, heuristic, and saved learned-policy seats. Learned policies
act visibly and deterministically in play mode; they never train, write a
checkpoint, or change their saved weights.

The initial learned-policy choices are intentionally limited to:

| Display name | Learner kind |
|---|---|
| DQN | `DQN` |
| Dueling DQN | `Dueling_DQN` |
| PPO | `PPO` |

Do not expose PQN, ADQN, VQN, or later agents until their inclusion is an
explicit extension of this plan.

## Scope boundary

Change only interactive setup, rendering, and app wiring. Do not change
`Environment`, `Game`, actions, rules, rewards, turn processing,
`GameSettings`, `Player`, `GameFactory`, or any file under `risk/learning/`.
The existing interactive loop already accepts any `BaseAgent`, so no game-rule
change is required.

Learned-seat choices are UI-only data. A learned seat is emitted into ordinary
`GameSettings` as the existing `ai` placeholder, so existing settings and
factory validation remain unchanged. After the unchanged factory builds the
context, the interactive app replaces the placeholder in `ctx.agents` before
`AppLoop.__init__` runs. The placeholder therefore never acts.

The in-game UI keeps a separate `{player_id: display_label}` mapping. It must
prefer that mapping to `AGENT_KIND_LABELS` when drawing the current-player and
player-table labels, so learned seats display their selected model rather than
generic `AI`. This is rendering-only state and never changes `Player`.

## Two model-selection paths

Every **AI Agent** seat offers both choices:

1. **Choose model manually...** opens a native Windows picker for either a
   policy-only `.pt` file or an `epNNNNNN` checkpoint directory. It initially
   opens at `Checkpoints/` when present but permits another local location.
   The selected name becomes the default display label. The user chooses DQN,
   Dueling DQN, or PPO unless the UI can identify it reliably.
2. **Best DQN** currently cycles through the five best DQN 103 checkpoints
   from the recorded checkpoint evaluation. Each click selects the next model,
   shows its label beside the controls, and supplies both the model path and
   learner kind. Dueling DQN and PPO have no preset until their own evaluation
   is completed; use manual selection for them.

Never infer an algorithm only from a filename. A missing, malformed, or
incompatible selection leaves Start Game disabled with an inline explanation;
never replace it silently with a random agent.

## Predefined best-model registry

Add the new repository convention `Params/play_agents.json`. It is a curated
registry for predefined evaluated models, not a recent-files cache. Initially
it contains only the five highest overall win-rate DQN 103 checkpoints from
`Checkpoints/DQN_103/evaluations/checkpoint_eval_ep006200_to_006700.json`.

```json
{
  "version": 1,
  "models": [
    {
      "id": "dqn_103_eval_1_ep6700",
      "label": "DQN 103 eval #1 (ep 6700, 85.2%)",
      "agent_kind": "DQN",
      "checkpoint": "Checkpoints/DQN_103/ep006700"
    }
  ]
}
```

- `id` is unique and stable for the setup selection.
- `label` is user-facing and fits the setup row and sidebar.
- `agent_kind` is exactly `DQN`, `Dueling_DQN`, or `PPO`.
- `checkpoint` names a policy-only `.pt` file or an `epNNNNNN` directory.

`Params/` is not ignored, while checkpoints are local and ignored. Consequently
committed preset paths may not work on another machine; disable an unavailable
preset with an explanation while preserving manual selection. Do not commit
model weights in this registry.

## Architecture and implementation sequence

1. Add a small adapter under `risk/app/` to parse and validate the predefined
   registry and normalize manual selections. It reports invalid JSON, duplicate
   IDs, unsupported kinds, missing paths, and load failures clearly.

   Reuse `risk.learning.choose_agent._read_policy_state(...)` and
   `_new_learned_agent(...)` as-is. They already implement the required
   policy-only behavior: read raw network state, construct DQN/Dueling DQN/PPO,
   attach the agent to its actual seat/environment, set `epsilon = 0.0`, and
   call `set_train_mode(False)`. Import them; do not copy, edit, or modify any
   training file. Because they are private helpers, add a direct coupling test
   so a future incompatible refactor fails loudly.

2. Keep `risk/ui/input/init_screen.py` headless and syntactic-only. It must
   not import the adapter or `risk/learning/`, avoiding the cycle
   `init_screen -> adapter -> choose_agent -> init_screen`. Extend its UI-only
   setup data with each learned seat's source, path/preset ID, learner kind,
   and display label. Its validation checks only structure: non-empty manual
   path, allowed kind, and a known preset ID. It continues to emit `ai` for a
   learned seat in `GameSettings`. Add a UI-only visible-type-cycle helper:
   `Human -> Random -> Raider -> Sentinel -> Empire -> Killbot -> Learned
   Agent -> Human`. Selecting **Learned Agent** stores `ai` underneath and
   marks the seat as learned in the UI-only selection data; switching away
   clears that learned selection and restores the next visible ordinary kind.
   Reducing the player count must discard UI-only learned selections and labels
   for removed seats, so validation and the final setup result contain only
   active seats.
   Do not use the current `next_agent_kind()` behavior unchanged for this
   cycle: its `AGENT_KIND_ORDER` intentionally excludes `ai`.

3. Add an `InteractiveSetupResult` UI/app value containing ordinary
   `GameSettings`, learned-seat selections, and the display-label mapping.
   `run_init_screen(...)` returns it on Start Game, and `SetupStage.run_setup(...)`
   passes it through unchanged. `SetupStage.default_settings(...)` keeps
   returning a bare `GameSettings`, exactly as today: it is called directly as
   `GameFactory.build(SetupStage.default_settings(...))` from
   `risk/learning/self_play.py`, `risk/learning/trainer.py`, and several tests,
   none of which know about learned seats, and both files are out of scope to
   edit. For the skip-menu, auto-restart, and max-ticks paths, `main.py`'s
   `run()` wraps that bare `GameSettings` into a local no-learned-selections
   `InteractiveSetupResult` itself, so the rest of `run()` has one shape to
   work with without changing `default_settings()`'s public contract.

4. Keep checkpoint validation in `risk/ui/render/init_screen_view.py`, which
   owns the live Start Game button, its enabled state, and inline error text.
   It performs a function-local lazy import of the adapter only when a learned
   seat needs validation. Never import it at module scope: that would load
   `torch` and the learning stack for ordinary human/heuristic games.

   For each changed learned selection or roster, validate once—not every render
   frame—by reading the policy state and building a short-lived context through
   the unchanged `GameFactory`, then constructing the policy with
   `_new_learned_agent(...)`. Cache the validation result until that seat's
   selection, learner kind, or player roster changes. This catches both bad
   files and incompatible model shapes before enabling Start Game.

5. Extend the New Game screen's seat-type cycle with UI-only **AI Agent**
   by repurposing the currently dormant `ai` value, which is allowed but not
   currently visible in the cycle. For a learned seat show:

   - **Choose model manually...** plus a DQN/Dueling DQN/PPO kind control;
   - **Use a predefined best model** plus the curated preset list; and
   - the selected model label or its validation message.

   Retain the existing Human, Random, Raider, Sentinel, Empire, and Killbot
   choices unchanged. A learned seat with no selected model remains visibly
   **AI Agent** (not generic `AI`) and cannot start; leaving the learned choice
   clears its model controls and returns the seat to the ordinary cycle.

6. In `main.py`, use the `InteractiveSetupResult` (from `SetupStage.run_setup(...)`
   or wrapped locally around `SetupStage.default_settings(...)`, per step 3),
   build the actual context
   through the unchanged `GameFactory`, then make new learned-agent instances
   from the saved policy state for every selected seat. Do not reuse temporary
   validation instances. Replace only `ctx.agents[seat]` before constructing
   `AppLoop`; that is the interactive path's load-bearing list. Updating the
   separate `ctx.game.agents` list is optional defensive consistency, not a
   functional requirement. Pass the display-label mapping to `AppLoop`, then
   through `GameView` to `HudPanel`. Make this mapping an optional keyword
   argument with an empty default at every layer. `risk.learning.self_play.py`
   constructs `GameView` directly and is out of scope, so its existing call
   must keep rendering the normal `AGENT_KIND_LABELS` labels unchanged.

7. Preserve exact policy-only loading semantics. For a policy-only `.pt`,
   `_read_policy_state(...)` reads the raw network state dictionary. For an
   `epNNNNNN` directory it opens the shared `model.pt` and extracts only its
   `"net"` payload. Never restore replay, optimizer, target network, rollout,
   epsilon decay, or trainer state. Never assign `train_mode` directly; reuse
   the helper's `set_train_mode(False)` call so the network enters eval mode.

8. Support several learned seats in one match, including different algorithms
   or different models. Each seat receives a distinct agent instance, even when
   two seats choose the same file; no environment binding or mutable agent
   state is shared.

## Tests and documentation

Before adding tests, follow `Docs/Testing.md`. Add
`Temp/tests/test_learned_agent_play.py` for the adapter, preset registry,
policy-only artifacts, direct private-helper coupling, pre-start validation
cache, actual-seat attachment, deterministic inference, and a mixed
human/heuristic/learned `AppLoop` smoke path. Keep setup-screen state and UI
control tests in `Temp/tests/test_ui.py`.

Assert that `GameSettings`, `GameFactory`, environment behavior, game rules,
and training entry points are unchanged. Include a compatibility check that
`GameView`/`HudPanel` with no label mapping retain their current labels. Update
`Docs/Testing.md` to document
the final ownership. Update README setup-screen instructions at implementation
time, including its pre-existing omission of Killbot from the seat-type cycle,
and record every change in `Docs/ChangeLog.md`.

## Acceptance criteria

- Any number of seats can use manual DQN, Dueling DQN, or PPO policies, or
  predefined best models, alongside humans and heuristics.
- Start Game is enabled only after each learned selection fully loads in a
  temporary validation context.
- The live player table and current-player label identify each selected model.
- The real match uses fresh deterministic inference agents and writes no
  learning state or checkpoints.
- Existing human-only, heuristic-only, skip-menu, auto-restart, and smoke-test
  behavior remains unchanged.
- The implementation changes only interactive setup, rendering, and app
  wiring; game/domain and training source files remain untouched.
