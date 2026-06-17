"""Presentation: render one frame (board + HUD) and resolve map clicks.

Owns the pygame renderer, HUD, layout rects, and hit-tester. This is the only
tier-3 (pygame) piece of the loop besides event polling.
"""
from __future__ import annotations

from typing import Optional

import pygame

from risk.game.board_topology import BoardTopology
from risk.game.settings import GameSettings
from risk.game.state import State
from risk.ui.input.hit_test import TerritoryHitTester
from risk.ui.input.human_input import HudActionPanelModel
from risk.ui.render.panels import HudPanel
from risk.ui.render.risk_map import RiskMapRenderer
from risk.ui_constants import BOARD_BG_COLOR as BG_COLOR
from risk.ui_constants import HUD_WIDTH


class GameView:
    def __init__(
        self,
        screen: pygame.Surface,
        settings: GameSettings,
        width: int,
        height: int,
    ) -> None:
        self.screen = screen
        self.settings = settings
        self.renderer = RiskMapRenderer()
        self.hud = HudPanel()
        self.topology = BoardTopology()

        board_w = width - HUD_WIDTH
        self.board_rect = pygame.Rect(0, 0, board_w, height)
        self.hud_rect = pygame.Rect(board_w, 0, HUD_WIDTH, height)

        img_w, img_h = self.renderer.base_map.get_size()
        self.hit_tester = TerritoryHitTester(
            self.renderer.territory_polygons,
            blit_rect=(self.board_rect.x, self.board_rect.y, self.board_rect.w, self.board_rect.h),
            image_size=(img_w, img_h),
        )

        # The HUD action panel lives in the middle of the side panel; the
        # static info (player table) is in the top, and a bottom strip is
        # reserved for the persistent "Last move:" report.
        self.last_move_rect = pygame.Rect(
            self.hud_rect.x + 4,
            self.hud_rect.bottom - 150,
            self.hud_rect.w - 8,
            150,
        )
        self.action_panel_rect = pygame.Rect(
            self.hud_rect.x + 4,
            self.hud_rect.y + 240,
            self.hud_rect.w - 8,
            self.last_move_rect.y - (self.hud_rect.y + 240) - 8,
        )

        # Click regions from the most recent render; read by HumanAgent to
        # match HUD clicks against the panel it last drew.
        self.hud_regions: list[tuple[pygame.Rect, str]] = []

    def territory_at(self, pos: tuple[int, int]) -> Optional[str]:
        return self.hit_tester.territory_at(*pos)

    def render(
        self,
        state: State,
        widgets: HudActionPanelModel,
        highlights: Optional[dict[str, tuple[int, int, int]]],
        message: str,
        last_action_text: str = "",
        last_action_lines: tuple[str, ...] = (),
    ) -> list[tuple[pygame.Rect, str]]:
        """Draw the frame; return the HUD click regions for event routing."""
        self.screen.fill(BG_COLOR)
        board_surface = self.renderer.render(
            owners=self._owners_dict(state),
            armies=self._armies_dict(state),
            highlights=highlights,
        )
        scaled = pygame.transform.smoothscale(
            board_surface, (self.board_rect.w, self.board_rect.h)
        )
        self.screen.blit(scaled, self.board_rect.topleft)

        self.hud.render(
            self.screen,
            self.hud_rect,
            state,
            self.settings,
            message=message,
            last_action_text=last_action_text,
            last_action_lines=last_action_lines,
        )
        self.hud.render_last_move(
            self.screen,
            self.last_move_rect,
            last_action_text=last_action_text,
            last_action_lines=last_action_lines,
        )
        self.hud_regions = self.hud.render_action_panel(
            self.screen, self.action_panel_rect, widgets
        )
        return self.hud_regions

    def _owners_dict(self, state: State) -> dict[str, tuple[int, int, int]]:
        return {
            self.topology.territory_at(i): self.settings.players[o].color
            for i, o in enumerate(state.owners)
            if o is not None
        }

    def _armies_dict(self, state: State) -> dict[str, int]:
        return {self.topology.territory_at(i): a for i, a in enumerate(state.armies)}


__all__ = ["GameView", "HUD_WIDTH"]
