"""Focused coverage for the injected-action PPO implementation."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import risk.learning.ppo_agent as ppo_agent_module
from risk.learning.ppo_agent import PPO_Agent
from risk.learning.rollout_buffer import RolloutBuffer
from risk.learning.train_constants import PPO_VALUE_LOSS_COEF

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


def test_n_step_targets_use_stored_values_and_boundary_bootstraps(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=3)
    buffer = RolloutBuffer()
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    for value, reward, done in zip((10.0, 20.0, 30.0, 40.0, 50.0), (1.0, 2.0, 3.0, 4.0, 5.0), (False, False, True, False, False)):
        buffer.push(state, action, 0, 0.0, value, reward, state.snapshot(), done)
    transitions = list(buffer.all())
    transitions[1] = transitions[1]._replace(gae_boundary=True)
    agent.gamma = 1.0
    monkeypatch.setattr(ppo_agent_module, "PPO_N_STEP", 3)
    monkeypatch.setattr(agent, "_boundary_values", lambda _: torch.tensor([0.0, 20.0, 0.0, 0.0, 50.0]))

    advantages, returns, horizons, bootstraps = agent._n_step_targets(transitions)

    assert returns.tolist() == pytest.approx([23.0, 22.0, 3.0, 59.0, 55.0])
    assert advantages.tolist() == pytest.approx([13.0, 2.0, -27.0, 19.0, 5.0])
    assert horizons.tolist() == pytest.approx([2.0, 1.0, 1.0, 2.0, 1.0])
    assert bootstraps.tolist() == [True, True, False, True, True]


def test_n_step_target_reuses_the_stored_successor_value(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=30)
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    buffer = RolloutBuffer()
    for value, reward in zip((10.0, 20.0, 30.0, 40.0, 50.0), (1.0, 2.0, 3.0, 4.0, 5.0)):
        buffer.push(state, action, 0, 0.0, value, reward, state.snapshot(), False)
    agent.gamma = 1.0
    monkeypatch.setattr(ppo_agent_module, "PPO_N_STEP", 3)
    monkeypatch.setattr(agent, "_boundary_values", lambda _: torch.tensor([0.0, 0.0, 0.0, 0.0, 99.0]))

    _, targets, horizons, bootstraps = agent._n_step_targets(buffer.all())

    assert targets.tolist() == pytest.approx([46.0, 59.0, 111.0, 108.0, 104.0])
    assert horizons.tolist() == pytest.approx([3.0, 3.0, 3.0, 2.0, 1.0])
    assert bootstraps.tolist() == [True, True, True, True, True]


def test_n_step_uses_stored_value_when_terminal_is_exactly_at_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=301)
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    buffer = RolloutBuffer()
    for value, reward, done in zip((10.0, 20.0, 30.0, 40.0), (1.0, 2.0, 3.0, 4.0), (False, False, False, True)):
        buffer.push(state, action, 0, 0.0, value, reward, state.snapshot(), done)
    agent.gamma = 1.0
    monkeypatch.setattr(ppo_agent_module, "PPO_N_STEP", 3)
    monkeypatch.setattr(agent, "_boundary_values", lambda _: torch.zeros(4))

    _, targets, horizons, bootstraps = agent._n_step_targets(buffer.all())

    assert targets[0].item() == pytest.approx(46.0)
    assert horizons[0].item() == 3.0
    assert bootstraps[0].item() is True
    assert targets[1].item() == pytest.approx(9.0)
    assert bootstraps[1].item() is False


def test_boundary_values_build_clean_rows_without_legal_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=31)
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    buffer = RolloutBuffer()
    for _ in range(3):
        buffer.push(state, action, 0, 0.0, 0.0, 0.0, state.snapshot(), False)
    transitions = list(buffer.all())
    transitions[1] = transitions[1]._replace(gae_boundary=True)
    entries_seen = []

    def fake_forward(entries):
        entries_seen.extend(entries)
        count = len(entries)
        return torch.empty(0), torch.arange(1, count + 1, dtype=torch.float32), torch.empty(0, dtype=torch.long)

    monkeypatch.setattr(agent.env, "legal_actions", lambda _: pytest.fail("boundary bootstrap must not enumerate actions"))
    monkeypatch.setattr(agent, "_forward_grouped", fake_forward)

    values = agent._boundary_values(transitions)

    assert values.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert len(entries_seen) == 2
    for rows, phase, cards, value_mask in entries_seen:
        assert len(rows) == 1
        assert phase.shape == (1,)
        assert cards.shape == (1, 3)
        assert torch.equal(value_mask, torch.tensor([True]))


def test_grouped_clean_value_rows_return_values_without_policy_logits() -> None:
    agent = _agent(seed=32)
    state = agent.env.current_state()
    entries = [agent._clean_value_entry(state, agent.player_id), agent._clean_value_entry(state, agent.player_id)]

    logits, values, action_groups = agent._forward_grouped(entries)

    assert logits.shape == (0,)
    assert values.shape == (2,)
    assert action_groups.shape == (0,)


def test_terminal_rollout_does_not_forward_bootstrap_values(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=33)
    state = agent.env.current_state().snapshot()
    action = agent.env.legal_actions(state)[0]
    buffer = RolloutBuffer()
    buffer.push(state, action, 0, 0.0, 0.0, 7.0, state.snapshot(), True)
    transitions = buffer.all()
    monkeypatch.setattr(agent, "_forward_grouped", lambda _: pytest.fail("terminal rollout must not bootstrap"))

    advantages, returns, horizons, bootstraps = agent._n_step_targets(transitions)

    assert returns.tolist() == pytest.approx([7.0])
    assert advantages.tolist() == pytest.approx([7.0])
    assert horizons.tolist() == pytest.approx([1.0])
    assert bootstraps.tolist() == [False]


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
    assert agent.last_update_metrics["ppo_post_epoch_kl"] > agent.target_kl
    assert agent.last_update_metrics["ppo_kl_stop_epoch"] == 1.0
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
        PPO_VALUE_LOSS_COEF * agent.last_update_metrics["ppo_value_huber_loss"]
    )
    assert agent.last_update_metrics["ppo_value_rmse"] ** 2 == pytest.approx(
        agent.last_update_metrics["ppo_value_mse"]
    )
    assert agent.last_update_metrics["ppo_grad_norm"] >= 0.0
    assert agent.last_update_metrics["ppo_policy_encoder_grad_norm"] >= 0.0
    assert agent.last_update_metrics["ppo_value_encoder_grad_norm"] >= 0.0
    assert agent.last_update_metrics["ppo_value_to_policy_encoder_grad_ratio"] >= 0.0
    assert agent.last_update_metrics["ppo_target_horizon_mean"] == 1.5
    assert agent.last_update_metrics["ppo_target_bootstrap_fraction"] == 1.0
    assert torch.isfinite(torch.tensor(
        agent.last_update_metrics["ppo_value_to_policy_encoder_grad_ratio"]
    ))


def test_post_epoch_kl_is_sample_weighted_across_cached_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=53)
    log_ratios = torch.tensor([0.0, 1.0, 2.0])
    calls: list[list[int]] = []

    def fake_evaluate(_, __, indices):
        calls.append(list(indices))
        selected = log_ratios[indices]
        count = len(indices)
        return selected, torch.zeros(count), torch.zeros(count), torch.zeros(count)

    monkeypatch.setattr(ppo_agent_module, "PPO_MINIBATCH_SIZE", 2)
    monkeypatch.setattr(agent, "_evaluate_indices", fake_evaluate)

    kl = agent._post_epoch_kl([None] * 3, [None] * 3, torch.zeros(3))

    expected = float((log_ratios.exp() - 1 - log_ratios).sum() / 3)
    assert kl == pytest.approx(expected)
    assert calls == [[0, 1], [2]]


def test_high_post_epoch_kl_completes_epoch_one_before_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=54)
    calls: list[list[int]] = []

    def fake_minibatch(_, __, indices, ___, ____, _____):
        calls.append(list(indices))
        return {"loss": 0.0, "approximate_kl": 0.03}

    monkeypatch.setattr(ppo_agent_module, "PPO_MINIBATCH_SIZE", 2)
    monkeypatch.setattr(agent, "_run_minibatch", fake_minibatch)
    monkeypatch.setattr(agent, "_post_epoch_kl", lambda *_: 0.03)
    update = {"cached_entries": [], "old_log_probs": torch.zeros(4), "advantages": torch.zeros(4), "returns": torch.zeros(4)}

    _, completed, early_stopped, stop_kl, post_epoch_kls, stop_epoch = agent._run_update_epochs([None] * 4, update)

    assert len(calls) == 2
    assert completed == 1
    assert early_stopped is True
    assert stop_kl == 0.03
    assert post_epoch_kls == [0.03]
    assert stop_epoch == 1


def test_low_post_epoch_kl_permits_all_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(seed=55)
    calls: list[list[int]] = []

    def fake_minibatch(_, __, indices, ___, ____, _____):
        calls.append(list(indices))
        return {"loss": 0.0}

    monkeypatch.setattr(ppo_agent_module, "PPO_MINIBATCH_SIZE", 2)
    monkeypatch.setattr(agent, "_run_minibatch", fake_minibatch)
    monkeypatch.setattr(agent, "_post_epoch_kl", lambda *_: 0.01)
    update = {"cached_entries": [], "old_log_probs": torch.zeros(4), "advantages": torch.zeros(4), "returns": torch.zeros(4)}

    _, completed, early_stopped, stop_kl, post_epoch_kls, stop_epoch = agent._run_update_epochs([None] * 4, update)

    assert len(calls) == 2 * ppo_agent_module.PPO_EPOCHS
    assert completed == ppo_agent_module.PPO_EPOCHS
    assert early_stopped is False
    assert stop_kl == 0.0
    assert post_epoch_kls == [0.01] * ppo_agent_module.PPO_EPOCHS
    assert stop_epoch == ppo_agent_module.PPO_EPOCHS


def test_value_to_policy_encoder_gradient_ratio_has_a_finite_floor() -> None:
    agent = _agent(seed=52)

    ratio = agent._value_to_policy_encoder_grad_ratio(3.0, 0.0)

    assert ratio == pytest.approx(3e12)
    assert torch.isfinite(torch.tensor(ratio))


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
