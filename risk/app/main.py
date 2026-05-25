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
from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
)
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

# Default AI action pacing in `play` mode (ms between AI ticks and marker fade).
DEFAULT_PLAY_AI_DELAY_MS = 600
DEFAULT_PLAY_MARKER_MS = 900

# Marker color for the last AI action overlay.
AI_MARKER_RGB = (255, 230, 80)


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


def _action_territories(action: Action, pre_pending) -> list[str]:
    """Return the territory ids the marker should highlight for `action`."""
    from risk.game.board_topology import BoardTopology

    if isinstance(action, ReinforcementAction):
        return [t for t, n in action.placements.items() if n > 0]
    if isinstance(action, AttackAction):
        return [action.from_territory, action.to_territory]
    if isinstance(action, OccupyAction):
        if pre_pending is None:
            return []
        topo = BoardTopology()
        return [
            topo.territory_at(pre_pending.from_index),
            topo.territory_at(pre_pending.to_index),
        ]
    if isinstance(action, FortifyAction):
        if action.is_skip:
            return []
        return [action.from_territory, action.to_territory]
    return []


def _run_headless(env: Environment, agents, max_ticks: Optional[int]) -> int:
    """Train mode without any rendering — pure simulation."""
    game = Game(env, agents)
    cap = max_ticks if max_ticks is not None else 1_000_000
    ticks = 0
    while not game.is_terminal() and ticks < cap:
        r = game.tick()
        if r.step is None:
            # Should not happen with all-AI rosters; bail to avoid spinning.
            break
        ticks += 1
    w = game.winner()
    print(f"[train-no-render] ticks={ticks} winner={'P' + str(w) if w is not None else 'None'}")
    return 0


def run(width: int = 1280, height: int = 800, seed: int = 0, players: int = 3,
        max_ticks: Optional[int] = None, skip_menu: bool = False,
        mode: str = "play",
        ai_delay_ms: Optional[int] = None,
        marker_ms: Optional[int] = None) -> int:
    """Run the game.

    Modes:
        play           — human-friendly: AI ticks are paced so each move is
                         visible, with a yellow marker on the affected
                         territories.
        train          — render normally but never wait between AI ticks.
        train-no-render — pure simulation, no pygame window.
    """
    if mode == "train-no-render":
        # All-AI default settings unless a menu choice is forced.
        settings = _default_settings(n=players, seed=seed)
        env = Environment()
        env.reset(settings)
        agents = _build_agents(settings)
        return _run_headless(env, agents, max_ticks)

    if mode not in ("play", "train"):
        raise ValueError(f"Unknown mode {mode!r}")

    ai_delay = ai_delay_ms if ai_delay_ms is not None else (
        DEFAULT_PLAY_AI_DELAY_MS if mode == "play" else 0
    )
    marker_dur = marker_ms if marker_ms is not None else (
        DEFAULT_PLAY_MARKER_MS if mode == "play" else 0
    )

    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Risk ({mode})")

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
        last_ai_tick_ms = -ai_delay  # allow immediate first AI tick
        marker_territories: list[str] = []
        marker_expires_ms = 0
        while running:
            now_ms = pygame.time.get_ticks()
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

            # Advance game one tick per frame, with AI pacing in `play` mode.
            if not game.is_terminal():
                pre_state = env.current_state()
                pid = pre_state.current_player_index
                acting_agent = agents[pid]
                is_ai = not isinstance(acting_agent, HumanAgent)
                pre_pending = pre_state.pending_attack

                may_tick = True
                if is_ai and ai_delay > 0:
                    may_tick = (now_ms - last_ai_tick_ms) >= ai_delay

                if may_tick:
                    r = game.tick()
                    if r.step is not None:
                        if is_ai:
                            last_ai_tick_ms = now_ms
                            last_action = game.history[-1]
                            terrs = _action_territories(last_action, pre_pending)
                            if terrs and marker_dur > 0:
                                marker_territories = terrs
                                marker_expires_ms = now_ms + marker_dur

            controller.on_turn_change(env.current_state())

            # Expire stale markers.
            if marker_territories and now_ms >= marker_expires_ms:
                marker_territories = []

            # Render.
            screen.fill((10, 10, 14))
            highlights = (
                {t: AI_MARKER_RGB for t in marker_territories}
                if marker_territories else None
            )
            board_surface = renderer.render(
                owners=_owners_dict(env.current_state(), settings),
                armies=_armies_dict(env.current_state()),
                highlights=highlights,
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
    parser.add_argument(
        "--mode",
        choices=("play", "train", "train-no-render"),
        default="play",
        help=(
            "play: AI moves paced + marked for visibility (default). "
            "train: render but no AI pacing. "
            "train-no-render: pure simulation, no window."
        ),
    )
    parser.add_argument(
        "--ai-delay-ms", type=int, default=None,
        help="Override pacing between AI ticks in `play` mode.",
    )
    parser.add_argument(
        "--marker-ms", type=int, default=None,
        help="Override AI action marker duration in `play` mode.",
    )
    args = parser.parse_args(argv)
    return run(
        args.width, args.height, args.seed, args.players, args.max_ticks,
        skip_menu=args.skip_menu,
        mode=args.mode,
        ai_delay_ms=args.ai_delay_ms,
        marker_ms=args.marker_ms,
    )


if __name__ == "__main__":
    sys.exit(main())
