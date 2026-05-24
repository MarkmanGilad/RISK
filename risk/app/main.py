"""Pygame application entry point.

Run:
    python -m risk.app.main

Shows the init / setup screen first, where the user picks player count,
names, colors, and seat types (Human / AI). Once the user clicks **Start
Game**, the chosen `GameSettings` are used to build the environment and
agents.

Smoke tests pass `--skip-menu` (or `--max-ticks`) to bypass the menu and
start with a default all-AI roster.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Support being launched as a plain script (VS Code green Run button).
# When run via `python -m risk.app.main`, this is a no-op; when run as
# `python risk/app/main.py`, the file's parent dir is on sys.path but the
# repo root is not, so absolute `from risk.app...` imports would fail.
if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import pygame

from risk.agents.human_agent import HumanAgent
from risk.agents.random_agent import RandomAgent
from risk.app.game import Game
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.player import Player
from risk.game.settings import GameSettings
from risk.graphics.risk_map import RiskMapRenderer
from risk.ui.hit_test import TerritoryHitTester
from risk.ui.human_input import HumanInputController
from risk.ui.init_screen import DEFAULT_COLORS, InitScreenState
from risk.ui.init_screen_view import run_init_screen
from risk.ui.panels import HudPanel


HUD_WIDTH = 280


def _default_settings(n: int = 3, seed: Optional[int] = 0) -> GameSettings:
    state = InitScreenState()
    state.set_player_count(n)
    for i in range(n):
        state.set_agent_kind(i, "ai")
    return state.build_settings(seed=seed)


def _build_agents(settings: GameSettings):
    out = []
    for p in settings.players:
        if p.agent_kind == "human":
            out.append(HumanAgent(player_id=p.id))
        else:
            out.append(RandomAgent(player_id=p.id, seed=(settings.seed or 0) + p.id + 1))
    return out


def _owners_dict(state, settings) -> dict[str, tuple[int, int, int]]:
    from risk.game.board_topology import BoardTopology

    topo = BoardTopology()
    return {
        topo.territory_at(i): settings.players[o].color
        for i, o in enumerate(state.owners)
        if o is not None
    }


def _armies_dict(state) -> dict[str, int]:
    from risk.game.board_topology import BoardTopology

    topo = BoardTopology()
    return {topo.territory_at(i): a for i, a in enumerate(state.armies)}


def run(width: int = 1280, height: int = 800, seed: int = 0, players: int = 3,
        max_ticks: Optional[int] = None, skip_menu: bool = False) -> int:
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Risk")

        # `--max-ticks` is used by smoke tests and implies no menu.
        if skip_menu or max_ticks is not None:
            settings = _default_settings(n=players, seed=seed)
        else:
            chosen = run_init_screen(screen)
            if chosen is None:
                return 0  # user closed the window
            settings = chosen

        env = Environment()
        env.reset(settings)
        agents = _build_agents(settings)
        game = Game(env, agents)
        controller = HumanInputController(env, agents, settings)

        renderer = RiskMapRenderer()
        hud = HudPanel()

        board_w = width - HUD_WIDTH
        board_rect = pygame.Rect(0, 0, board_w, height)
        hud_rect = pygame.Rect(board_w, 0, HUD_WIDTH, height)

        img_w, img_h = renderer.base_map.get_size()
        hit_tester = TerritoryHitTester(
            renderer.territory_polygons,
            blit_rect=(board_rect.x, board_rect.y, board_rect.w, board_rect.h),
            image_size=(img_w, img_h),
        )

        # The HUD action panel lives in the bottom half of the side panel;
        # the static info (player table) is in the top half.
        action_panel_rect = pygame.Rect(
            hud_rect.x + 4,
            hud_rect.y + 240,
            hud_rect.w - 8,
            hud_rect.h - 260,
        )
        hud_regions: list[tuple[pygame.Rect, str]] = []

        from risk.game.board_topology import BoardTopology
        topo_for_clicks = BoardTopology()

        def _dispatch_hud(region_id: str) -> None:
            if region_id.startswith("button:"):
                controller.on_hud_button(region_id[len("button:") :])
                return
            if region_id.startswith("field:"):
                rest = region_id[len("field:") :]
                if ":" in rest:
                    fid, payload = rest.split(":", 1)
                    controller.on_hud_field(fid, payload)
                else:
                    controller.on_hud_field(rest)

        clock = pygame.time.Clock()
        ticks = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handled = False
                    for r, rid in hud_regions:
                        if r.collidepoint(event.pos):
                            _dispatch_hud(rid)
                            handled = True
                            break
                    if not handled:
                        terr_name = hit_tester.territory_at(*event.pos)
                        if terr_name is not None:
                            try:
                                idx = topo_for_clicks.index_of(terr_name)
                            except (KeyError, ValueError):
                                idx = None
                            if idx is not None:
                                controller.on_territory_click(idx, event.button)

            # Sync controller with current (player, phase) before tick.
            controller.on_turn_change(env.current_state())

            # Advance game one tick per frame.
            if not game.is_terminal():
                game.tick()

            controller.on_turn_change(env.current_state())

            # Render.
            screen.fill((10, 10, 14))
            board_surface = renderer.render(
                owners=_owners_dict(env.current_state(), settings),
                armies=_armies_dict(env.current_state()),
            )
            scaled = pygame.transform.smoothscale(board_surface, (board_rect.w, board_rect.h))
            screen.blit(scaled, board_rect.topleft)

            msg = ""
            if game.is_terminal():
                w = game.winner()
                msg = f"Game over. Winner: P{w}" if w is not None else "Game over."
            hud.render(screen, hud_rect, env.current_state(), settings, message=msg)
            hud_regions = hud.render_action_panel(
                screen, action_panel_rect, controller.widgets(env.current_state())
            )

            pygame.display.flip()
            clock.tick(60)

            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            if game.is_terminal() and max_ticks is None:
                # Linger a bit so user sees end-state.
                pygame.time.wait(2000)
                running = False

        return 0
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument(
        "--max-ticks", type=int, default=None,
        help="Cap the loop (used by smoke tests).",
    )
    parser.add_argument(
        "--skip-menu", action="store_true",
        help="Skip the init screen and start a default all-AI game.",
    )
    args = parser.parse_args(argv)
    return run(
        args.width, args.height, args.seed, args.players, args.max_ticks,
        skip_menu=args.skip_menu,
    )


if __name__ == "__main__":
    sys.exit(main())
