"""Fuzz test for `SelfPlay.play_headless`: several seeded full AI-only games.

Unlike the single-seed smoke checks in `test_game_loop.py`/`test_agents.py`
(capped well below a real game's length), these run a mixed
heuristic/random roster all the way to an actual winner across several
seeds — a fast (~1000-1500 steps/game) regression net for bugs that only
surface deep into a real game (e.g. the `Phase` enum changes in `Docs/
RL-Prep-Changes.md`).
"""
from __future__ import annotations

import pytest

from risk.agents.heuristic_agent import EmpireAgent, RaiderAgent, SentinelAgent
from risk.agents.random_agent import RandomAgent
from risk.app.factory import GameFactory
from risk.app.setup import SetupStage
from risk.learning.self_play import SelfPlay

SEEDS = range(10)


@pytest.mark.parametrize("seed", SEEDS)
def test_self_play_reaches_a_winner(seed: int) -> None:
    ctx = GameFactory.build(SetupStage.default_settings(n=4, seed=seed))
    ctx.agents[0] = RaiderAgent(player_id=0, env=ctx.env)
    ctx.agents[1] = SentinelAgent(player_id=1, env=ctx.env)
    ctx.agents[2] = EmpireAgent(player_id=2, env=ctx.env)
    ctx.agents[3] = RandomAgent(player_id=3, env=ctx.env, seed=seed)

    winner = SelfPlay.play_headless(ctx, max_steps=5000)

    assert ctx.env.is_terminal()
    assert winner in {0, 1, 2, 3}
