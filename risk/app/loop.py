"""Interactive pygame loop: every seat is a uniform agent.

The loop is deliberately tiny and mirrors a classic board-game loop:

    events = pygame.event.get()
    agent  = agents[state.current_player_index]
    action = agent(events, state)        # AI: immediate, Human: when ready
    if action: env.step(action)          # env decides whose turn is next
    view.render(state, ...)

Swapping a human seat for an AI seat is purely a matter of which agent
object sits in `agents` — the loop does not change. Headless training runs
the same ask -> step over `Game` (see `risk/app/game.py`) with no view.
"""
from __future__ import annotations

from typing import Optional

import pygame

from risk.agents.human_agent import HumanAgent
from risk.app.factory import GameContext
from risk.app.marker import ActionMarker, action_report, describe_action
from risk.app.pacer import AITickPacer
from risk.app.view import GameView

# Marker color for the last AI action overlay.
AI_MARKER_RGB = (255, 230, 80)

END_LINGER_MS = 2000


class AppLoop:
    def __init__(
        self,
        ctx: GameContext,
        screen: pygame.Surface,
        *,
        width: int,
        height: int,
        ai_delay_ms: int,
        marker_ms: int,
    ) -> None:
        self.env = ctx.env
        self.settings = ctx.settings
        self.agents = ctx.agents
        self.view = GameView(screen, ctx.settings, width, height)
        self.pacer = AITickPacer(ai_delay_ms)
        self.marker = ActionMarker(AI_MARKER_RGB, marker_ms)
        self.last_action_text = ""
        self.last_action_lines: tuple[str, ...] = ()

        # Human agents need the view for hit-testing + HUD click regions.
        for agent in self.agents:
            if isinstance(agent, HumanAgent):
                agent.set_view(self.view)

    # --- main loop -------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> int:
        clock = pygame.time.Clock()
        ticks = 0
        while True:
            now = pygame.time.get_ticks()
            events = pygame.event.get()
            if self._wants_quit(events):
                break

            # --- the whole game, right here -----------------------------
            state = self.env.current_state()
            player = self.agents[state.current_player_index]
            if self._ready(player, now):
                action = player(events, state)      # AI: now · Human: when ready
                if action is not None:
                    self._apply(action, state, player, now)
            # ------------------------------------------------------------

            self.marker.expire(now)
            self._render()
            pygame.display.flip()
            clock.tick(60)

            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            if self.env.is_terminal():
                if max_ticks is None:
                    pygame.time.wait(END_LINGER_MS)  # let the user see the end
                break
        return 0

    @staticmethod
    def _wants_quit(events) -> bool:
        for event in events:
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def _ready(self, player, now_ms: int) -> bool:
        """AI moves are paced so a human can watch them; humans never wait."""
        if isinstance(player, HumanAgent):
            return True
        return self.pacer.may_tick(now_ms, True)

    def _apply(self, action, state, player, now_ms: int) -> None:
        """Send the chosen action to the env and record it for the HUD/marker."""
        pid = state.current_player_index
        pre_pending = state.pending_attack
        result = self.env.step(action)
        name = self.settings.players[pid].name
        self.last_action_text = describe_action(action, name, pre_pending)
        self.last_action_lines = action_report(
            action, name, result.info, pre_pending
        )
        if not isinstance(player, HumanAgent):
            self.pacer.note_tick(now_ms)
            self.marker.set_action(action, state.pending_attack, now_ms)

    def _render(self) -> None:
        state = self.env.current_state()
        agent = self.agents[state.current_player_index]
        message = ""
        if self.env.is_terminal():
            winner = self.env.winner()
            message = f"Game over. Winner: P{winner}" if winner is not None else "Game over."
        self.view.render(
            state,
            agent.widgets(state),
            self.marker.highlights(),
            message,
            self.last_action_text,
            self.last_action_lines,
        )


__all__ = ["AppLoop", "AI_MARKER_RGB"]
