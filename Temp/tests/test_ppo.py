"""Focused coverage for the injected-action PPO implementation."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from risk.learning.ppo_agent import PPO_Agent
from risk.learning.rollout_buffer import RolloutBuffer

from .conftest import make_env


def _agent(*, seed: int = 1, rollout_length: int = 4, train_mode: bool = True) -> PPO_Agent:
    return PPO_Agent(
        player_id=0, env=make_env(seed=seed, agent_kind="ai"),
        device=torch.device("cpu"), seed=seed, train_mode=train_mode,
        rollout_length=rollout_length,
    )


def _remember_same_state(agent: PPO_Agent, reward: float = 0.0, done: bool = False) -> None:
    state = agent.env.current_state()
    action = agent((), state)
    assert action is not None
    agent.remember(state, action, reward, state.snapshot(), done)


def test_ppo_net_returns_one_logit_per_action_and_one_value() -> None:
    agent = _agent(seed=2)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    logits, values = agent._forward_actions(state, legal, agent.player_id)

    assert logits.shape == (len(legal),)
    assert values.shape == (1,)


def test_gae_resets_at_cutoff_but_keeps_cutoff_bootstrap() -> None:
    agent = _agent(seed=3)
    buffer = RolloutBuffer()
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    for reward in (1.0, 2.0, 3.0):
        buffer.push(state, action, 0, 0.0, 0.0, reward, state.snapshot(), False)
    transitions = list(buffer.all())
    transitions[1] = transitions[1]._replace(gae_boundary=True)
    agent.gamma = 1.0

    advantages, returns = agent._gae(transitions, torch.tensor([10.0, 20.0, 30.0]))

    # The transition at the artificial episode cutoff includes V(s') = 20,
    # but excludes the next reset game's advantage (33) from its GAE carry.
    assert advantages.tolist() == pytest.approx([31.9, 22.0, 33.0])
    assert torch.equal(advantages, returns)


def test_act_then_remember_stores_plain_collection_metadata() -> None:
    agent = _agent(seed=4, rollout_length=2)
    _remember_same_state(agent)

    transition = agent.rollout_buffer.all()[0]

    assert isinstance(transition.old_log_prob, float)
    assert isinstance(transition.old_value, float)
    assert transition.action_index >= 0
    assert agent.env.legal_actions(transition.state)[transition.action_index] == transition.action


def test_progress_metrics_report_rollout_fill_and_update_count() -> None:
    agent = _agent(seed=41, rollout_length=4)
    _remember_same_state(agent)

    assert agent.progress_metrics() == {
        "ppo_rollout_fill": 1.0,
        "ppo_rollout_fill_fraction": 0.25,
        "ppo_rollout_updates": 0.0,
        "ppo_samples_processed_estimated": 0.0,
    }


def test_rollout_gate_defers_update_until_full() -> None:
    agent = _agent(seed=5, rollout_length=2)
    _remember_same_state(agent)
    before = {name: value.detach().clone() for name, value in agent.net.state_dict().items()}

    assert agent.learn() == []
    assert all(torch.equal(value, agent.net.state_dict()[name]) for name, value in before.items())


def test_k3_kl_estimate_is_non_negative() -> None:
    agent = _agent(seed=50)
    log_ratio = torch.tensor([-0.5, 0.25, 1.0])

    estimate = agent._k3_approx_kl(log_ratio.exp(), log_ratio)

    assert estimate.item() >= 0.0
    assert estimate.item() == pytest.approx(0.2862, abs=1e-4)


def test_kl_limit_stops_remaining_ppo_epochs() -> None:
    agent = _agent(seed=51, rollout_length=2)
    # This deliberately impossible threshold lets the first minibatch run,
    # then verifies that the next epoch is stopped before its optimizer step.
    agent.target_kl = -1.0
    _remember_same_state(agent, reward=1.0)
    _remember_same_state(agent, reward=1.0)

    losses = agent.learn()

    assert len(losses) == 1
    assert agent.last_update_metrics["ppo_early_stopped"] == 1.0
    assert agent.last_update_metrics["ppo_epochs_completed"] == 1.0
    assert agent.last_update_metrics["ppo_early_stop_kl"] > agent.target_kl
    assert agent.last_update_metrics["ppo_optimizer_steps_per_update"] == 1.0
    assert agent.last_update_metrics["ppo_samples_processed_per_update"] == 2.0
    assert agent.optimizer_steps == 1
    assert agent.samples_processed == 2
    assert 0.0 <= agent.last_update_metrics["ppo_normalized_entropy"] <= 1.0
    assert agent.last_update_metrics["ppo_value_rmse"] >= 0.0
    assert agent.last_update_metrics["ppo_value_loss"] == pytest.approx(
        agent.last_update_metrics["ppo_value_mse"]
    )
    assert agent.last_update_metrics["ppo_value_huber_loss"] <= agent.last_update_metrics["ppo_value_mse"]
    assert agent.last_update_metrics["ppo_weighted_value_loss"] == pytest.approx(
        0.5 * agent.last_update_metrics["ppo_value_huber_loss"]
    )
    assert agent.last_update_metrics["ppo_value_rmse"] ** 2 == pytest.approx(
        agent.last_update_metrics["ppo_value_mse"]
    )
    assert agent.last_update_metrics["ppo_grad_norm"] >= 0.0
    assert agent.last_update_metrics["ppo_policy_encoder_grad_norm"] >= 0.0
    assert agent.last_update_metrics["ppo_value_encoder_grad_norm"] >= 0.0


def test_cached_entry_rejects_changed_legal_action_index() -> None:
    agent = _agent(seed=6)
    state = agent.env.current_state().snapshot()
    legal = agent.env.legal_actions(state)
    agent.rollout_buffer.push(state, legal[0], len(legal), 0.0, 0.0, 0.0, state.snapshot(), False)

    with pytest.raises(RuntimeError, match="legal-action index"):
        agent._cache_transition_entry(agent.rollout_buffer.all()[0])


def test_grouped_forward_preserves_each_decision_action_count() -> None:
    agent = _agent(seed=7)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)
    assert len(legal) >= 5
    entries = []
    for actions in (legal[:2], legal[2:5]):
        rows, phase, cards, _, value_mask = agent._decision_rows(state, actions, agent.player_id)
        entries.append((rows, phase, cards, value_mask))

    logits, values, action_groups = agent._forward_grouped(entries)

    assert logits.shape == (5,)
    assert values.shape == (2,)
    assert torch.equal(action_groups.cpu(), torch.tensor([0, 0, 1, 1, 1]))


def test_checkpoint_round_trip_drops_partial_rollout(tmp_path: Path) -> None:
    agent = _agent(seed=8, rollout_length=2)
    _remember_same_state(agent)
    _remember_same_state(agent)
    agent.target_kl = -1.0
    agent.learn()
    _remember_same_state(agent)
    path = tmp_path / "ppo"
    agent.save_checkpoint(path)

    restored = _agent(seed=9, rollout_length=2)
    _remember_same_state(restored)
    restored.load_checkpoint(path)

    assert restored.train_steps == agent.train_steps
    assert restored.optimizer_steps == agent.optimizer_steps
    assert restored.samples_processed == agent.samples_processed
    assert restored.progress_metrics()["ppo_samples_processed_estimated"] == 0.0
    assert len(restored.rollout_buffer) == 0
    for name, value in agent.net.state_dict().items():
        assert torch.equal(value, restored.net.state_dict()[name])
