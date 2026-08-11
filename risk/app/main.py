"""Pygame application entry point.

Run:
    python -m risk.app.main

Shows the init / setup screen first, where the user picks player count,
names, colors, and seat types. Once the user clicks **Start
Game**, the chosen `GameSettings` are used to build the environment and
agents.

Smoke tests pass `--skip-menu` (or `--max-ticks`) to bypass the menu and
start with a default all-random roster.
"""
from __future__ import annotations

import argparse
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

from risk.app.factory import GameFactory
from risk.app.loop import AppLoop
from risk.app.setup import SetupStage
from risk.ui_constants import DEFAULT_PLAY_AI_DELAY_MS, DEFAULT_PLAY_MARKER_MS, HUD_WIDTH


def _run_headless(max_ticks: Optional[int], players: int, seed: int, auto_restart: bool) -> int:
    """Train mode without any rendering — pure simulation via the shared factory."""
    current_seed = seed
    while True:
        ctx = GameFactory.build(SetupStage.default_settings(n=players, seed=current_seed))
        winner = ctx.game.play_until_terminal(max_steps=max_ticks or 1_000_000)
        print(
            f"[train-no-render] winner={'P' + str(winner) if winner is not None else 'None'}"
        )
        if not auto_restart:
            return 0
        current_seed += 1


def run(width: Optional[int] = None, height: Optional[int] = None, seed: int = 0, players: int = 3,
        max_ticks: Optional[int] = None, skip_menu: bool = False,
        mode: str = "play",
    auto_restart: bool = False,
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
        return _run_headless(max_ticks, players, seed, auto_restart)

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
        if width is None or height is None:
            info = pygame.display.Info()
            max_w = int(info.current_w * 0.90)
            max_h = int(info.current_h * 0.90)
            # Preserve the board (map) area aspect ratio (1240×1000 from the
            # original 1600×1000 design), then add the HUD width back to get
            # the total window width.
            _BOARD_ASPECT = (1600 - HUD_WIDTH) / 1000  # 1.24
            height = min(max_h, 1000)
            board_w = int(height * _BOARD_ASPECT)
            width = board_w + HUD_WIDTH
            if width > max_w:
                width = max_w
                board_w = width - HUD_WIDTH
                height = int(board_w / _BOARD_ASPECT)
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Risk ({mode})")

        current_seed = seed
        while True:
            # PRE-GAME: produce settings (menu, or defaults for smoke tests).
            # `--max-ticks` is used by smoke tests and implies no menu.
            if skip_menu or max_ticks is not None or auto_restart:
                setup_result = {"settings": SetupStage.default_settings(n=players, seed=current_seed),
                                "selections": {}, "labels": {}}
            else:
                setup_result = SetupStage.run_setup(
                    screen,
                    players=players,
                    seed=current_seed,
                    skip_menu=False,
                )
            if setup_result is None:
                return 0  # user closed the window
            if not isinstance(setup_result, dict):
                setup_result = {"settings": setup_result, "selections": {}, "labels": {}}
            settings = setup_result["settings"]

            # BUILD: pygame-free wiring shared with headless training.
            ctx = GameFactory.build(settings)
            selections = setup_result["selections"]
            if selections:
                from risk.app.learned_agent_play import build_agents

                for seat, agent in build_agents(ctx, selections).items():
                    ctx.agents[seat] = agent

            # GAME: run the interactive loop.
            loop = AppLoop(
                ctx,
                screen,
                width=width,
                height=height,
                ai_delay_ms=ai_delay,
                marker_ms=marker_dur,
                player_labels=setup_result["labels"],
            )
            loop.run(max_ticks=max_ticks, show_win_screen=not auto_restart)

            # Auto-restart is for trainer-style runs: immediately start a new
            # game with a fresh seed and no menu/win-screen pause.
            if auto_restart:
                current_seed += 1
                continue

            # If running with max_ticks (smoke tests / train mode), exit after
            # one game rather than looping back to the menu.
            if max_ticks is not None or skip_menu:
                return 0
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=None,
                        help="Window width in pixels (default: 90%% of screen).")
    parser.add_argument("--height", type=int, default=None,
                        help="Window height in pixels (default: 90%% of screen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument(
        "--max-ticks", type=int, default=None,
        help="Cap the loop (used by smoke tests).",
    )
    parser.add_argument(
        "--skip-menu", action="store_true",
        help="Skip the init screen and start a default all-random game.",
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
        "--auto-restart",
        action="store_true",
        help="Immediately start a new game after the current one ends.",
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
        auto_restart=args.auto_restart,
        ai_delay_ms=args.ai_delay_ms,
        marker_ms=args.marker_ms,
    )


if __name__ == "__main__":
    sys.exit(main())
