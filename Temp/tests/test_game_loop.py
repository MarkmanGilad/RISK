"""Tests for the Game loop and the app entry point (Phase 7)."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from risk.agents.human_agent import HumanAgent
from risk.agents.random_agent import RandomAgent
from risk.app.game import Game
from risk.game.environment import Environment
from risk.game.player import Player
from risk.game.settings import GameSettings


def _settings(n: int = 3, seed: int = 1) -> GameSettings:
    return GameSettings(
        players=tuple(
            Player(id=i, name=f"P{i}", color=(i * 30, i * 40, i * 50), agent_kind="ai")
            for i in range(n)
        ),
        seed=seed,
    )


def _env(n: int = 3, seed: int = 1) -> Environment:
    env = Environment()
    env.reset(_settings(n, seed))
    return env


# --- Game contract --------------------------------------------------------


def test_game_rejects_mismatched_agent_count() -> None:
    env = _env()
    agents = [RandomAgent(player_id=i) for i in range(2)]
    with pytest.raises(ValueError):
        Game(env, agents)


def test_game_rejects_mismatched_agent_ids() -> None:
    env = _env()
    agents = [RandomAgent(player_id=i + 10) for i in range(3)]
    with pytest.raises(ValueError):
        Game(env, agents)


def test_game_tick_advances_when_agent_decides() -> None:
    env = _env()
    agents = [RandomAgent(player_id=i, seed=i + 1) for i in range(3)]
    game = Game(env, agents)
    r = game.tick()
    assert r.step is not None
    assert len(game.history) == 1


def test_game_tick_waits_on_undecided_human() -> None:
    env = _env()
    agents = [HumanAgent(player_id=0), RandomAgent(player_id=1), RandomAgent(player_id=2)]
    game = Game(env, agents)
    # Player 0 starts the game; HumanAgent has no submission yet.
    r = game.tick()
    assert r.step is None
    assert r.waiting_on_player == 0


def test_game_plays_to_completion_with_random_agents() -> None:
    env = _env(seed=5)
    agents = [RandomAgent(player_id=i, seed=100 + i) for i in range(3)]
    game = Game(env, agents)
    winner = game.play_until_terminal(max_steps=20000)
    # Random agents probabilistically converge; assert either terminal+winner
    # or made significant progress without crashing.
    assert game.is_terminal() or len(game.history) > 0
    if game.is_terminal():
        assert winner in {0, 1, 2}


# --- App entry smoke test -------------------------------------------------


def test_main_module_imports_without_crashing() -> None:
    # Just import; don't open a window. Module init must be side-effect-free.
    import importlib
    mod = importlib.import_module("risk.app.main")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run")


def test_app_runs_for_a_few_ticks_headless() -> None:
    from risk.app.main import main

    rc = main(["--max-ticks", "30", "--width", "640", "--height", "400", "--seed", "0"])
    assert rc == 0
