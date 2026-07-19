"""Focused coverage for ``ADQN_Agent`` and ``Docs/ADQN.md`` Sections A-G."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from risk.learning.adqn_agent import ADQN_Agent
from risk.learning.adqn import ADQN
from risk.learning.dueling_dqn_agent import Dueling_DQN_Agent
from risk.learning.pqn_agent import PQN_Agent
from risk.learning.train_constants import EPSILON_DECAY_EPISODES, EPSILON_END, EPSILON_START

from .conftest import make_env


def _agent(seed: int = 1, **kwargs) -> ADQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return ADQN_Agent(player_id=0, env=env, train_mode=True, seed=seed, **kwargs)


def _remember_two(agent: ADQN_Agent) -> None:
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)


def test_adqn_owns_sibling_net_and_agent_with_dueling_behavior() -> None:
    env = make_env(seed=3, agent_kind="ai")
    torch.manual_seed(4)
    dueling = Dueling_DQN_Agent(player_id=0, env=env, train_mode=False, seed=7)
    torch.manual_seed(4)
    adqn = ADQN_Agent(
        player_id=0,
        env=env,
        train_mode=False,
        seed=7,
        advantage_loss_coef=0.0,
    )
    adqn.net.load_state_dict(dueling.net.state_dict())
    adqn.target_net.load_state_dict(dueling.target_net.state_dict())

    assert isinstance(adqn.net, ADQN)
    assert not isinstance(adqn, PQN_Agent)
    state = env.current_state()
    assert adqn.act([], state) == dueling.act([], state)
    adqn.on_episode_start(1)
    assert adqn.epsilon == dueling.epsilon == EPSILON_START
    adqn.on_episode_start(EPSILON_DECAY_EPISODES + 1)
    dueling.on_episode_start(EPSILON_DECAY_EPISODES + 1)
    assert adqn.epsilon == dueling.epsilon == pytest.approx(EPSILON_END)


def test_centered_advantages_have_zero_mean_per_decision() -> None:
    agent = _agent()
    value = torch.tensor([2.0, -1.0])
    advantage = torch.tensor([1.0, 4.0, -2.0, 3.0, 7.0])
    groups = torch.tensor([0, 0, 0, 1, 1])

    centered = agent._combine_q(value, advantage, groups) - value[groups]

    assert centered[groups == 0].mean().item() == pytest.approx(0.0)
    assert centered[groups == 1].mean().item() == pytest.approx(0.0)


def test_advantage_weight_is_scaled_bounded_detached_and_has_correct_direction() -> None:
    agent = _agent(advantage_weight_scale=5.0)
    td_advantage = torch.tensor([100.0, -100.0], requires_grad=True)
    centered = torch.tensor([2.0, -2.0], requires_grad=True)
    q_loss = torch.tensor(10.0)

    weight, _, loss, _, _, _ = agent._advantage_loss_terms(
        td_advantage, centered, q_loss
    )
    loss.backward()

    assert weight.requires_grad is False
    assert torch.all(weight <= 5.0) and torch.all(weight >= -5.0)
    assert weight.tolist() == pytest.approx([5.0, -5.0])
    assert torch.all(weight.abs() >= 5.0 * agent.advantage_weight_saturation)
    assert td_advantage.grad is None
    assert centered.grad[0] < 0  # minimizing raises a positive-TD action
    assert centered.grad[1] > 0  # minimizing lowers a negative-TD action


def test_effective_coefficient_uses_base_when_below_cap() -> None:
    agent = _agent(advantage_loss_coef=0.1, max_advantage_loss_fraction=0.25)
    td_advantage = torch.tensor([0.1])
    centered = torch.tensor([0.1], requires_grad=True)

    terms = agent._advantage_loss_terms(td_advantage, centered, torch.tensor(2.0))

    assert float(terms[4]) == pytest.approx(0.1)


def test_cap_uses_absolute_activity_despite_signed_cancellation() -> None:
    agent = _agent(advantage_loss_coef=1.0, max_advantage_loss_fraction=0.25)
    td_advantage = torch.tensor([100.0, -100.0])
    centered = torch.tensor([100.0, 100.0], requires_grad=True)
    q_loss = torch.tensor(4.0)

    _, _, signed_loss, abs_mean, coefficient, weighted = (
        agent._advantage_loss_terms(td_advantage, centered, q_loss)
    )
    weighted.backward()

    assert float(signed_loss.detach()) == pytest.approx(0.0)
    assert float(abs_mean.detach()) == pytest.approx(500.0)
    assert float(coefficient.detach()) == pytest.approx(0.002)
    assert centered.grad is not None and torch.any(centered.grad != 0)
    assert abs(float(weighted.detach())) <= 0.25 * float(q_loss)


def test_loss_balance_is_finite_when_q_and_advantage_activity_are_zero() -> None:
    agent = _agent()
    terms = agent._advantage_loss_terms(
        torch.zeros(2), torch.zeros(2, requires_grad=True), torch.tensor(0.0)
    )

    for tensor in terms:
        assert torch.isfinite(tensor).all()
    assert float(terms[4]) == 0.0


def test_ddqn_online_selection_and_target_evaluation_match_dueling() -> None:
    env = make_env(seed=13, agent_kind="ai")
    torch.manual_seed(5)
    dueling = Dueling_DQN_Agent(player_id=0, env=env, train_mode=True)
    torch.manual_seed(5)
    adqn = ADQN_Agent(player_id=0, env=env, train_mode=True)
    adqn.net.load_state_dict(dueling.net.state_dict())
    adqn.target_net.load_state_dict(dueling.target_net.state_dict())

    state = env.current_state().snapshot()
    state.perspective = 0
    legal = env.legal_actions(state)
    stage = torch.tensor([legal[0].dqn_index(env.topology, state)[0]])
    done = torch.tensor([False])

    expected = dueling._max_next_ddqn_q([state], done, stage)
    _, actual = adqn._next_state_terms([state], done, stage)

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_encoder_gradient_diagnostic_is_finite_and_does_not_write_grads() -> None:
    agent = _agent()
    parameters = tuple(agent.net.encoder.parameters())
    shared = sum(parameter.square().sum() for parameter in parameters)
    q_loss = shared
    advantage_loss = -shared

    q_norm, advantage_norm, cosine = agent._encoder_gradient_diagnostic(
        q_loss, advantage_loss
    )

    assert q_norm > 0 and advantage_norm > 0
    assert cosine == pytest.approx(-1.0)
    assert all(parameter.grad is None for parameter in parameters)

    _, _, zero_cosine = agent._encoder_gradient_diagnostic(
        shared * 0.0, advantage_loss
    )
    assert zero_cosine == 0.0


def test_train_step_logs_required_metrics_and_respects_diagnostic_cadence() -> None:
    agent = _agent(grad_diagnostic_every=2)
    _remember_two(agent)

    first_loss = agent.train_step(batch_size=2)
    first = agent.last_update_metrics
    assert math.isfinite(first_loss)
    assert "adqn_q_encoder_grad_norm" not in first
    assert first["adqn_advantage_activity_to_q_loss_ratio"] <= (
        agent.max_advantage_loss_fraction + 1e-6
    )
    for key in (
        "adqn_q_loss",
        "adqn_advantage_loss",
        "adqn_advantage_loss_abs_mean",
        "adqn_weighted_advantage_loss",
        "adqn_total_loss",
        "adqn_v_online_mean",
        "adqn_a_centered_taken_mean",
        "adqn_advantage_weight_scale",
        "adqn_advantage_weight_positive_fraction",
        "adqn_advantage_weight_negative_fraction",
        "adqn_advantage_weight_saturated_fraction",
        "adqn_advantage_weight_td_error_correlation",
    ):
        assert key in first and math.isfinite(first[key])

    agent.train_step(batch_size=2)
    second = agent.last_update_metrics
    assert math.isfinite(second["adqn_q_encoder_grad_norm"])
    assert math.isfinite(second["adqn_advantage_encoder_grad_norm"])
    assert -1.0 <= second["adqn_encoder_gradient_cosine_similarity"] <= 1.0


def test_zero_advantage_coefficient_makes_total_equal_bellman_loss() -> None:
    agent = _agent(advantage_loss_coef=0.0)
    _remember_two(agent)

    loss = agent.train_step(batch_size=2)

    assert loss == pytest.approx(agent.last_update_metrics["adqn_q_loss"])
    assert agent.last_update_metrics["adqn_weighted_advantage_loss"] == 0.0


def test_checkpoint_round_trip_preserves_adqn_state_and_settings(tmp_path: Path) -> None:
    agent = _agent(
        advantage_loss_coef=0.2,
        max_advantage_loss_fraction=0.3,
        advantage_weight_scale=4.0,
        advantage_weight_saturation=0.8,
        grad_diagnostic_every=7,
        loss_balance_epsilon=1e-6,
    )
    _remember_two(agent)
    agent.train_step(batch_size=2)
    checkpoint = tmp_path / "adqn"
    agent.save_checkpoint(checkpoint)

    restored = _agent(seed=9)
    restored.load_checkpoint(checkpoint)

    assert restored.label == "ADQN"
    assert restored.advantage_loss_coef == 0.2
    assert restored.max_advantage_loss_fraction == 0.3
    assert restored.advantage_weight_scale == 4.0
    assert restored.advantage_weight_saturation == 0.8
    assert restored.grad_diagnostic_every == 7
    assert restored.loss_balance_epsilon == 1e-6
    assert restored.train_steps == agent.train_steps
    assert len(restored.replay_buffer) == len(agent.replay_buffer)
    for name, value in agent.net.state_dict().items():
        assert torch.equal(value, restored.net.state_dict()[name])
    for name, value in agent.target_net.state_dict().items():
        assert torch.equal(value, restored.target_net.state_dict()[name])
    assert restored.optimizer.state_dict()["param_groups"] == (
        agent.optimizer.state_dict()["param_groups"]
    )
    assert restored.optimizer.state_dict()["state"].keys() == (
        agent.optimizer.state_dict()["state"].keys()
    )


def test_correlation_returns_zero_for_zero_variance() -> None:
    agent = _agent()
    assert agent._correlation(torch.ones(3), torch.arange(3.0)) == 0.0
    correlation = agent._correlation(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]))
    assert -1.0 <= correlation <= 1.0


def test_legacy_checkpoint_restores_original_advantage_weight_scale(
    tmp_path: Path,
) -> None:
    agent = _agent()
    checkpoint = tmp_path / "legacy-adqn"
    agent.save_checkpoint(checkpoint)
    model_path = checkpoint / "model.pt"
    payload = torch.load(model_path, weights_only=False)
    payload.pop("advantage_weight_scale")
    torch.save(payload, model_path)

    restored = _agent(advantage_weight_scale=5.0)
    restored.load_checkpoint(checkpoint)

    assert restored.advantage_weight_scale == 1.0
