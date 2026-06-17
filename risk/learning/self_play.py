"""Self-play driver for training — no human seats.

`SelfPlay` runs a full game of AI/learning agents against each other, using
the exact same `agent(events, state) -> action` contract as the interactive
app:

- `SelfPlay.play_headless(...)` — no pygame at all. As fast as the CPU
  allows. Use this for bulk training rollouts.
- `SelfPlay.play_rendered(...)` — opens a pygame window and draws every
  move, but never paces the AI, so it runs fast while still letting you
  *watch* what the agents are doing.

Both accept an optional `on_step` callback that receives every transition
`(state_before, action, result)`, which is where a trainer collects its
experience. Both `state_before` and `result.state` are independent
snapshots — safe to keep in a replay buffer. (`Environment` mutates a single
`State` object in place for the whole game, so without snapshotting, every
transition stored by a trainer would end up aliasing the same, fully-mutated
final state once the episode ended.)

A GCN+DQN trainer built on top of this should subclass `SelfPlay` and
override `on_step`/`make_collector` rather than reimplementing the loop.

Typical use::

    from risk.app.factory import GameFactory
    from risk.app.setup import SetupStage
    from risk.learning.self_play import SelfPlay

    ctx = GameFactory.build(SetupStage.default_settings(n=3, seed=0))
    # swap in your learning agent for seat 0:
    # ctx.agents[0] = MyAgent(player_id=0, env=ctx.env)

    winner = SelfPlay.play_headless(ctx, max_steps=20_000, on_step=my_collector)
    # or, to watch it:
    winner = SelfPlay.play_rendered(ctx, on_step=my_collector)
"""
from __future__ import annotations

from typing import Callable, Optional

# Support being launched as a plain script (VS Code green Run button / F5).
# When run via `python -m risk.learning.self_play` this is a no-op; when run as
# `python risk/learning/self_play.py`, the repo root is not on sys.path, so the
# absolute `from risk...` imports below would fail without this.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from risk.app.factory import GameContext
from risk.game.actions import Action
from risk.game.environment import StepResult
from risk.game.state import State

# A trainer hook: (state_before, action, result) -> None.
StepHook = Callable[[State, Action, StepResult], None]


class SelfPlay:
    """Runs AI-only games for self-play / training rollouts."""

    @staticmethod
    def _reject_humans(ctx: GameContext) -> None:
        """Self-play is AI-only: a HumanAgent would stall (it never gets events)."""
        for agent in ctx.agents:
            if type(agent).__name__ == "HumanAgent":
                raise ValueError(
                    "play_headless / play_rendered are for AI-only self-play; "
                    f"seat {agent.player_id} is a HumanAgent. Use an all-AI "
                    "roster (e.g. SetupStage.default_settings(...)) or a "
                    "learning agent instead."
                )

    @classmethod
    def play_headless(
        cls,
        ctx: GameContext,
        *,
        max_steps: int = 100_000,
        on_step: Optional[StepHook] = None,
    ) -> Optional[int]:
        """Play one full game with no rendering. Returns the winner id (or None).

        This is the bare ask -> step loop, the same one the GUI runs, minus
        the window. `on_step(state_before, action, result)` is called after
        every applied action for experience collection.
        """
        cls._reject_humans(ctx)
        env, agents = ctx.env, ctx.agents
        steps = 0

        for _ in range(max_steps):
            if env.is_terminal():
                break
            state = env.current_state()
            player = agents[state.current_player_index]
            action = player((), state)      # AI agents ignore the events arg
            if action is None:
                break                        # nothing to do (shouldn't happen for AI)
            before = state.snapshot() if on_step else state
            result = env.step(action)
            if on_step is not None:
                # `result.state` is the same mutable object `env` keeps
                # reusing for the rest of the game — snapshot it before
                # handing it to the trainer, or every stored transition would
                # alias the final state once the episode ends.
                on_step(before, action, StepResult(state=result.state.snapshot(), info=result.info))
            steps += 1
            print(f"steps = {steps}", end="\r", flush=True)

        if steps:
            print()
        return env.winner()

    @classmethod
    def play_rendered(
        cls,
        ctx: GameContext,
        *,
        width: int = 1280,
        height: int = 800,
        max_steps: int = 100_000,
        fps: int = 0,
        caption: str = "Risk (training)",
        on_step: Optional[StepHook] = None,
    ) -> Optional[int]:
        """Play one full game in a window, drawing every move but never pacing.

        Same loop as `play_headless`, plus a `GameView` render each frame so
        you can watch training. `fps <= 0` runs as fast as possible; set e.g.
        `fps=30` to slow it to a watchable speed without adding per-move
        delays. Closing the window or pressing ESC stops early.
        """
        cls._reject_humans(ctx)

        # Local imports: pygame is only needed for the rendered path.
        import pygame

        from risk.app.view import GameView

        env, agents = ctx.env, ctx.agents

        pygame.init()
        try:
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(caption)
            view = GameView(screen, ctx.settings, width, height)
            clock = pygame.time.Clock()
            last_action_text = ""

            steps = 0
            running = True
            while running and steps < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                if not env.is_terminal():
                    state = env.current_state()
                    player = agents[state.current_player_index]
                    action = player((), state)
                    if action is not None:
                        before = state.snapshot() if on_step else state
                        result = env.step(action)
                        last_action_text = cls._describe(action, ctx, state)
                        if on_step is not None:
                            on_step(
                                before,
                                action,
                                StepResult(state=result.state.snapshot(), info=result.info),
                            )
                        steps += 1
                        print(f"steps = {steps}", end="\r", flush=True)

                view.render(
                    env.current_state(),
                    agents[env.current_state().current_player_index].widgets(
                        env.current_state()
                    ),
                    None,                   # no action marker in training view
                    cls._end_message(env),
                    last_action_text,
                )
                pygame.display.flip()
                if fps > 0:
                    clock.tick(fps)

                if env.is_terminal():
                    break
            if steps:
                print()
            return env.winner()
        finally:
            pygame.quit()

    @staticmethod
    def _describe(action: Action, ctx: GameContext, state: State) -> str:
        from risk.app.marker import ActionMarker

        pid = state.current_player_index
        return ActionMarker.describe_action(
            action, ctx.settings.players[pid].name, state.pending_attack
        )

    @staticmethod
    def _end_message(env) -> str:
        if not env.is_terminal():
            return ""
        w = env.winner()
        return f"Game over. Winner: P{w}" if w is not None else "Game over."


def _print_player_summary(ctx: GameContext, steps: int) -> None:
    topology = ctx.env.topology
    state = ctx.env.current_state()
    print(f"steps = {steps}")
    for player in ctx.settings.players:
        owned_indices = [i for i, owner in enumerate(state.owners) if owner == player.id]
        territories = len(owned_indices)
        armies = sum(state.armies[i] for i in owned_indices)
        continents = [
            continent
            for continent in topology.continents
            if topology.owns_continent(state.owners, continent, player.id)
        ]
        continent_text = ", ".join(continents) if continents else "none"
        print(
            f"P{player.id} {player.name}: continents={continent_text} "
            f"territories={territories} soldiers={armies}"
        )


def _print_roster(ctx: GameContext) -> None:
    print("roster:")
    for player, agent in zip(ctx.settings.players, ctx.agents):
        print(f"  P{player.id} {player.name}: {type(agent).__name__}")


def _winner_text(ctx: GameContext, winner: Optional[int]) -> str:
    if winner is None:
        return "None"
    player = ctx.settings.players[winner]
    agent = ctx.agents[winner]
    return f"P{winner} {player.name} ({type(agent).__name__})"


def main() -> None:
    """Training entry point. Edit this freely — it is your scratch pad.

    Build the context here (player count, seats, seeds, your learning agent),
    then call `SelfPlay.play_headless` (fast, invisible) or
    `SelfPlay.play_rendered` (watchable).
    Run it with:  python -m risk.learning.self_play
    """
    from risk.agents.heuristic_agent import EmpireAgent, RaiderAgent, SentinelAgent
    from risk.agents.random_agent import RandomAgent
    from risk.app.factory import GameFactory
    from risk.app.setup import SetupStage

    # 1) Roster: Raider vs Sentinel vs Empire vs Random. Change `seed` to rematch.
    ctx = GameFactory.build(SetupStage.default_settings(n=4))
    ctx.agents[0] = RaiderAgent(player_id=0, env=ctx.env)
    ctx.agents[1] = SentinelAgent(player_id=1, env=ctx.env)
    ctx.agents[2] = EmpireAgent(player_id=2, env=ctx.env)
    ctx.agents[3] = SentinelAgent(player_id=3, env=ctx.env)
    _print_roster(ctx)

    # 2) To test your learner, replace one seat:
    #    from risk.learning.my_agent import MyAgent
    #    ctx.agents[0] = MyAgent(player_id=0, env=ctx.env)

    # 3) (Optional) collect transitions for training.
    transitions: list = []

    def collect(state, action, result):
        transitions.append((state, action, result.info))

    # 4) Run one episode. Use SelfPlay.play_rendered(ctx, fps=30) to watch it.
    winner = SelfPlay.play_headless(ctx, max_steps=150_000, on_step=collect)
    # winner = SelfPlay.play_rendered(ctx, fps=30, on_step=collect)

    
    terminated = ctx.env.is_terminal()

    print(f"winner = {_winner_text(ctx, winner)}")
    print(f"terminated = {terminated}")
    _print_player_summary(ctx, len(transitions))


__all__ = ["SelfPlay", "StepHook", "main"]


if __name__ == "__main__":
    main()
