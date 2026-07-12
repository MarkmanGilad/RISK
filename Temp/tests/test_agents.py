"""Tests for the agent interface and the bundled agents (Phase 5)."""
from __future__ import annotations

import random
from pathlib import Path

import torch
import pytest

from risk.agents.base_agent import BaseAgent
from risk.agents.heuristic_agent import (
    AttackAgent,
    BSRAgent,
    CompositeAgent,
    ContinentAgent,
    EmpireAgent,
    RaiderAgent,
    SentinelAgent,
    ShapeAgent,
    attacker_roll_edge,
    battle_win_probability,
)
from risk.agents.human_agent import HumanAgent
from risk.agents.random_agent import RandomAgent
from risk.game.card import Card
from risk.game.actions import (
    AttackAction,
    FortifyAction,
    ReinforcementAction,
    StopAttackAction,
)
from .conftest import make_env, make_settings
from risk.game.environment import Environment
from risk.game.player import Player
from risk.game.settings import GameSettings
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    TRAIN_STEPS_PER_CALL,
)


def _settings(n: int = 3, seed: int = 42) -> GameSettings:
    return make_settings(n=n, seed=seed, agent_kind="ai")


def _fresh_env(seed: int = 42) -> Environment:
    return make_env(seed=seed, agent_kind="ai")


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
    # Submit a clearly out-of-phase action: a StopAttackAction during TRADE_IN.
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


# --- Heuristic agents -----------------------------------------------------


def test_attacker_roll_edge_uses_exact_risk_probabilities() -> None:
    assert attacker_roll_edge(3, 2) == pytest.approx(2890 / 7776)
    assert attacker_roll_edge(1, 1) == pytest.approx(15 / 36)
    assert battle_win_probability(2, 1) == pytest.approx(15 / 36)


def test_heuristic_agents_choose_valid_actions() -> None:
    env = _fresh_env(seed=17)
    agents = [
        AttackAgent(player_id=0, env=env, seed=10),
        BSRAgent(player_id=1, env=env, seed=11),
        CompositeAgent(player_id=2, env=env, seed=12),
    ]

    for _ in range(250):
        if env.is_terminal():
            break
        state = env.current_state()
        action = agents[state.current_player_index].act([], state)
        assert action is not None
        env.step(action)


def test_all_tiered_heuristic_agents_construct() -> None:
    env = _fresh_env(seed=23)
    agents = [
        AttackAgent(player_id=0, env=env),
        BSRAgent(player_id=0, env=env),
        ContinentAgent(player_id=0, env=env),
        ShapeAgent(player_id=0, env=env),
        CompositeAgent(player_id=0, env=env),
        RaiderAgent(player_id=0, env=env),
        SentinelAgent(player_id=0, env=env),
        EmpireAgent(player_id=0, env=env),
    ]

    for agent in agents:
        action = agent.act([], env.current_state())
        assert action is not None


def test_gnn_dqn_agent_picks_max_scored_legal_action() -> None:
    env = _fresh_env(seed=9)
    agent = GNN_DQN_Agent(
        player_id=0,
        env=env,
        train_mode=False,
    )
    state = env.current_state()
    legal = env.legal_actions(state)

    def fake_score(_state, actions):
        return torch.arange(len(actions), dtype=torch.float32)

    agent.score_actions = fake_score

    chosen = agent.act([], state)

    assert chosen is not None
    assert chosen.to_dict() == legal[-1].to_dict()


def test_gnn_dqn_agent_handles_trade_in_rows_without_injection() -> None:
    env = _fresh_env(seed=13)
    s = env.current_state()
    s.hands[0] = [
        Card(territory_id="Alaska", symbol="infantry"),
        Card(territory_id="Alberta", symbol="cavalry"),
        Card(territory_id="Ontario", symbol="artillery"),
    ]
    env._begin_turn_for(s, 0)
    assert s.phase.name == "TRADE_IN"

    agent = GNN_DQN_Agent(
        player_id=0,
        env=env,
        train_mode=False,
    )

    def fake_score(_state, actions):
        return torch.arange(len(actions), dtype=torch.float32)

    agent.score_actions = fake_score

    chosen = agent.act([], s)

    assert chosen is not None
    assert chosen.phase is s.phase


def test_gnn_dqn_agent_save_and_load_params_round_trip(tmp_path: Path) -> None:
    env = _fresh_env(seed=5)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=False)
    saved_path = tmp_path / "gnn_dqn_state.pt"

    saved_state = {
        key: value.clone()
        for key, value in agent.net.state_dict().items()
    }
    agent.save_params(saved_path)

    with torch.no_grad():
        first_key = next(iter(agent.net.state_dict()))
        agent.net.state_dict()[first_key].add_(1.0)

    agent.load_params(saved_path)

    for key, value in agent.net.state_dict().items():
        assert torch.equal(value, saved_state[key])


def test_gnn_dqn_agent_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    env = _fresh_env(seed=5)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=True, epsilon=0.3)
    state = env.current_state()
    action = env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)
    agent.train_step(batch_size=2)

    ckpt_dir = tmp_path / "full_ckpt"
    agent.save_checkpoint(ckpt_dir)

    agent2 = GNN_DQN_Agent(player_id=0, env=env, train_mode=True)
    agent2.load_checkpoint(ckpt_dir)

    assert agent2._train_steps == agent._train_steps
    assert agent2.epsilon == agent.epsilon
    assert len(agent2.replay_buffer) == len(agent.replay_buffer)
    for key, value in agent.net.state_dict().items():
        assert torch.equal(value, agent2.net.state_dict()[key])
    for key, value in agent.optimizer.state_dict()["state"].items():
        assert key in agent2.optimizer.state_dict()["state"]


def test_gnn_dqn_agent_on_episode_start_decays_epsilon() -> None:
    env = _fresh_env(seed=5)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=True)

    agent.on_episode_start(0)
    assert agent.epsilon == EPSILON_START

    agent.on_episode_start(1)
    assert agent.epsilon == EPSILON_START

    agent.on_episode_start(EPSILON_DECAY_EPISODES)
    expected = EPSILON_START + (EPSILON_END - EPSILON_START) * (
        (EPSILON_DECAY_EPISODES - 1) / EPSILON_DECAY_EPISODES
    )
    assert agent.epsilon == pytest.approx(expected)

    agent.on_episode_start(EPSILON_DECAY_EPISODES + 1)
    assert agent.epsilon == pytest.approx(EPSILON_END)

    agent.on_episode_start(EPSILON_DECAY_EPISODES * 10)
    assert agent.epsilon == pytest.approx(EPSILON_END)


def test_gnn_dqn_agent_learn_accepts_reached_max_steps() -> None:
    agent = GNN_DQN_Agent(player_id=0, env=_fresh_env(seed=5), train_mode=True)

    assert agent.learn(reached_max_steps=True) == []


def test_gnn_dqn_agent_can_train_threshold_matches_batch_size() -> None:
    """`can_train()`/`learn()` now read `BATCH_SIZE`/`TRAIN_STEPS_PER_CALL`
    directly instead of receiving them as call-time arguments
    (`Docs/ChangeLog.md`'s 2026-07-05 entry) — this pins down that the
    threshold behavior is unchanged by that refactor."""
    env = _fresh_env(seed=5)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=True)
    state = env.current_state()
    action = env.legal_actions(state)[0]

    for _ in range(BATCH_SIZE - 1):
        agent.remember(state, action, 0.0, state.snapshot(), False)
    assert agent.can_train() is False
    assert agent.learn() == []

    agent.remember(state, action, 0.0, state.snapshot(), False)
    assert agent.can_train() is True
    losses = agent.learn()
    assert len(losses) == TRAIN_STEPS_PER_CALL


def test_gnn_dqn_agent_reached_max_steps_flag_is_inert_with_full_replay() -> None:
    """`reached_max_steps` is reserved for on-policy agents (`Docs/PPO.md`)
    — for `GNN_DQN_Agent` it must not change the update at all. Compares
    the returned losses (not raw net weights: the GNN encoder's scatter
    aggregation is not bit-deterministic across forward passes even with
    fixed seeds, confirmed empirically against `HEAD`, so a tolerant
    comparison on the observable output is the meaningful check here)."""

    def _make_and_fill(seed: int) -> GNN_DQN_Agent:
        torch.manual_seed(seed)
        env = _fresh_env(seed=5)
        agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=True, seed=seed)
        state = env.current_state()
        action = env.legal_actions(state)[0]
        for i in range(BATCH_SIZE):
            agent.remember(state, action, float(i), state.snapshot(), False)
        return agent

    agent_true = _make_and_fill(seed=42)
    agent_false = _make_and_fill(seed=42)

    random.seed(123)
    losses_true = agent_true.learn(reached_max_steps=True)
    random.seed(123)
    losses_false = agent_false.learn(reached_max_steps=False)

    assert losses_true == pytest.approx(losses_false, rel=1e-2)


def test_base_agent_on_episode_start_is_a_harmless_no_op() -> None:
    env = _fresh_env(seed=5)
    agent = RandomAgent(player_id=0, env=env, seed=1)

    agent.on_episode_start(0)
    agent.on_episode_start(1000)


def test_gnn_dqn_agent_auto_device_selection() -> None:
    env = _fresh_env(seed=6)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert agent.device.type == expected


def test_gnn_dqn_agent_device_report_matches_runtime_tensors() -> None:
    env = _fresh_env(seed=8)
    agent = GNN_DQN_Agent(player_id=0, env=env, train_mode=False)
    report = agent.device_report(env.current_state())

    selected = torch.device(report["selected_device"])
    net_device = torch.device(report["net_device"])
    batch_device = torch.device(report["batch_device"])
    phase_device = torch.device(report["phase_device"])
    card_indices_device = torch.device(report["card_indices_device"])

    assert selected.type == agent.device.type
    assert net_device.type == agent.device.type
    assert batch_device.type == agent.device.type
    assert phase_device.type == agent.device.type
    assert card_indices_device.type == agent.device.type
