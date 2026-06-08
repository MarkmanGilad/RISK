"""Tests for the agent interface and the bundled agents (Phase 5)."""
from __future__ import annotations

import pytest

from risk.agents.base_agent import BaseAgent
from risk.agents.human_agent import HumanAgent
from risk.agents.random_agent import RandomAgent
from risk.game.actions import (
    AttackAction,
    FortifyAction,
    ReinforcementAction,
    StopAttackAction,
)
from risk.game.environment import Environment
from risk.game.player import Player
from risk.game.settings import GameSettings


def _settings(n: int = 3, seed: int = 42) -> GameSettings:
    return GameSettings(
        players=tuple(
            Player(id=i, name=f"P{i}", color=(10, 20, 30), agent_kind="ai")
            for i in range(n)
        ),
        seed=seed,
    )


def _fresh_env(seed: int = 42) -> Environment:
    env = Environment()
    env.reset(_settings(seed=seed))
    return env


# --- BaseAgent contract ---------------------------------------------------


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent(player_id=0)  # type: ignore[abstract]


def test_base_agent_has_act_signature() -> None:
    assert hasattr(BaseAgent, "act")
    assert hasattr(BaseAgent, "on_turn_start")
    assert hasattr(BaseAgent, "on_turn_end")


# --- HumanAgent -----------------------------------------------------------


def test_human_agent_returns_none_without_submission() -> None:
    env = _fresh_env()
    h = HumanAgent(player_id=0, env=env, settings=_settings())
    assert h.act([], env.current_state()) is None


def test_human_agent_returns_submitted_legal_action() -> None:
    env = _fresh_env()
    h = HumanAgent(player_id=0, env=env, settings=_settings())
    chosen = env.legal_actions()[0]
    h.submit(chosen)
    assert h.act([], env.current_state()) is chosen
    # After consumption, must return None again.
    assert h.act([], env.current_state()) is None


def test_human_agent_drops_illegal_submission() -> None:
    env = _fresh_env()
    h = HumanAgent(player_id=0, env=env, settings=_settings())
    # Submit a clearly out-of-phase action: a StopAttackAction during REINFORCE.
    h.submit(StopAttackAction())
    assert h.act([], env.current_state()) is None


# --- RandomAgent ----------------------------------------------------------


def test_random_agent_chooses_legal_actions() -> None:
    env = _fresh_env(seed=7)
    ra = RandomAgent(player_id=0, env=env, seed=7)
    for _ in range(10):
        legal = env.legal_actions()
        chosen = ra.act([], env.current_state())
        assert chosen is not None
        assert chosen.to_dict() in [a.to_dict() for a in legal]
        env.step(chosen)
        if env.is_terminal():
            break


def test_random_agent_deterministic_with_seed() -> None:
    env_a = _fresh_env(seed=11)
    env_b = _fresh_env(seed=11)
    a1 = RandomAgent(player_id=0, env=env_a, seed=99)
    a2 = RandomAgent(player_id=0, env=env_b, seed=99)
    for _ in range(5):
        c1 = a1.act([], env_a.current_state())
        c2 = a2.act([], env_b.current_state())
        assert c1.to_dict() == c2.to_dict()
        env_a.step(c1)
        env_b.step(c2)


# --- End-to-end smoke: random agents play to completion or step cap -------


def test_random_agents_play_many_steps_without_crashing() -> None:
    env = _fresh_env(seed=3)
    agents = [RandomAgent(player_id=i, env=env, seed=100 + i) for i in range(3)]
    for _ in range(500):
        if env.is_terminal():
            break
        s = env.current_state()
        pid = s.current_player_index
        a = agents[pid]
        chosen = a.act([], s)
        assert chosen is not None
        env.step(chosen)
