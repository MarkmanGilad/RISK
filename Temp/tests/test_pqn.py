"""Tests for `PQN`/`PQN_Agent` (`Docs/PQN.md` §24).

Mirrors `test_dueling_dqn.py`'s structure and coverage, adapted for PQN's
raw `(V, A)` return (instead of a fused `Q`) and its extra replay-based
policy-improvement loss term. Agent plumbing shared with `Dueling_DQN_Agent`
(device selection, `remember`, checkpoint round-trip shape) isn't re-derived
here beyond what differs: no `epsilon`, and `train_step` writes `pqn_*`
metrics instead of `dqn_*`.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from risk.learning.pqn_agent import PQN_Agent
from risk.learning.train_constants import BATCH_SIZE, TRAIN_STEPS_PER_CALL

from .conftest import make_env


def _agent(seed: int = 1, train_mode: bool = True) -> PQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return PQN_Agent(player_id=0, env=env, train_mode=train_mode)


def test_pqn_score_actions_returns_value_and_one_advantage_per_action_row() -> None:
    agent = _agent(seed=2)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    value, advantage = agent.score_actions(state, legal)

    assert value.shape == (1,)
    assert advantage.shape == (len(legal),)


def test_combine_q_matches_dueling_formula_on_one_group() -> None:
    agent = _agent(seed=3)
    value = torch.tensor([2.0])
    advantage = torch.tensor([1.0, -1.0, 3.0])
    group_index = torch.zeros(3, dtype=torch.long)

    q = agent._combine_q(value, advantage, group_index)

    expected = value[0] + advantage - advantage.mean()
    assert torch.allclose(q, expected, atol=1e-6)


def test_combine_q_normalizes_two_groups_independently() -> None:
    agent = _agent(seed=4)
    value = torch.tensor([1.0, 10.0])
    advantage = torch.tensor([1.0, -1.0, 5.0, -5.0])
    group_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    q = agent._combine_q(value, advantage, group_index)

    assert torch.allclose(q[:2], torch.tensor([2.0, 0.0]), atol=1e-6)
    assert torch.allclose(q[2:], torch.tensor([15.0, 5.0]), atol=1e-6)


def test_act_samples_from_advantage_in_train_mode() -> None:
    agent = _agent(seed=5, train_mode=True)
    state = agent.env.current_state()

    action = agent.act(events=[], state=state)

    assert action in agent.env.legal_actions(state)


def test_act_is_argmax_advantage_in_eval_mode() -> None:
    agent = _agent(seed=6, train_mode=False)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    _, advantage = agent.score_actions(state, legal)
    expected = legal[int(torch.argmax(advantage).item())]

    action = agent.act(events=[], state=state)

    assert action == expected


def test_pqn_agent_learn_accepts_reached_max_steps() -> None:
    agent = _agent(seed=8)

    assert agent.learn(reached_max_steps=True) == []


def test_pqn_agent_can_train_threshold_matches_batch_size() -> None:
    agent = _agent(seed=9)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]

    for _ in range(BATCH_SIZE - 1):
        agent.remember(state, action, 0.0, state.snapshot(), False)
    assert agent.can_train() is False
    assert agent.learn() == []

    agent.remember(state, action, 0.0, state.snapshot(), False)
    assert agent.can_train() is True
    losses = agent.learn()
    assert len(losses) == TRAIN_STEPS_PER_CALL


def test_pqn_agent_reached_max_steps_flag_is_inert_with_full_replay() -> None:
    """Same rationale as `Dueling_DQN_Agent`'s equivalent test: `reached_max_steps`
    is reserved for on-policy agents and must not change a PQN update at all."""

    def _make_and_fill(seed: int) -> PQN_Agent:
        torch.manual_seed(seed)
        env = make_env(seed=5, agent_kind="ai")
        agent = PQN_Agent(player_id=0, env=env, train_mode=True, seed=seed)
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


def test_pqn_agent_progress_metrics_reports_replay_state_without_epsilon() -> None:
    agent = _agent(seed=10)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)

    metrics = agent.progress_metrics()

    assert "epsilon" not in metrics
    assert metrics["pqn_replay_buffer_size"] == len(agent.replay_buffer) == 1
    assert metrics["pqn_train_steps_since_target_sync"] == 0.0


def test_pqn_agent_train_step_populates_last_update_metrics() -> None:
    agent = _agent(seed=11)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)

    assert agent.last_update_metrics == {}
    agent.train_step(batch_size=2)

    metrics = agent.last_update_metrics
    expected_keys = {
        "pqn_q_loss", "pqn_policy_loss", "pqn_total_loss",
        "pqn_td_error_mean", "pqn_td_error_abs_mean",
        "pqn_td_advantage_mean", "pqn_td_advantage_abs_mean",
        "pqn_q_value_mean", "pqn_value_mean", "pqn_target_q_mean",
        "pqn_grad_norm", "pqn_grad_norm_clipped",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["pqn_grad_norm"] >= 0.0
    assert metrics["pqn_td_error_abs_mean"] >= 0.0
    assert metrics["pqn_grad_norm_clipped"] in (0.0, 1.0)


def test_policy_loss_detaches_td_advantage() -> None:
    """The policy loss must not backpropagate through its TD weight."""
    agent = _agent(seed=12)
    td_advantage = torch.tensor([1.0, -2.0], requires_grad=True)
    log_pi_taken = torch.tensor([-0.3, -1.2], requires_grad=True)

    agent._policy_loss(td_advantage, log_pi_taken).backward()

    assert td_advantage.grad is None
    assert log_pi_taken.grad is not None
    assert torch.any(log_pi_taken.grad != 0)


def test_save_and_load_checkpoint_round_trips_pqn_net_and_target(tmp_path: Path) -> None:
    agent = _agent(seed=7)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)
    agent.train_step(batch_size=2)

    ckpt_dir = tmp_path / "pqn_ckpt"
    agent.save_checkpoint(ckpt_dir)

    agent2 = PQN_Agent(player_id=0, env=agent.env, train_mode=True)
    agent2.load_checkpoint(ckpt_dir)

    assert agent2._train_steps == agent._train_steps
    assert len(agent2.replay_buffer) == len(agent.replay_buffer)
    for key, value in agent.net.state_dict().items():
        assert torch.equal(value, agent2.net.state_dict()[key])
    for key, value in agent.target_net.state_dict().items():
        assert torch.equal(value, agent2.target_net.state_dict()[key])
