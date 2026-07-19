"""Tests for `PQN`/`PQN_Agent` (`Docs/PQN.md` §24).

Mirrors `test_dueling_dqn.py`'s structure and coverage, adapted for PQN's
raw `(V, A)` return (instead of a fused `Q`) and its extra replay-based
policy-improvement loss term. Agent plumbing shared with `Dueling_DQN_Agent`
(device selection, `remember`, checkpoint round-trip shape) isn't re-derived
here beyond what differs: PQN action/loss variants, and `train_step` writes
`pqn_*` metrics instead of `dqn_*`.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from risk.learning.dueling_dqn_agent import Dueling_DQN_Agent
from risk.learning.pqn_agent import PQN_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    PQN_POLICY_LOSS_COEF,
    TRAIN_STEPS_PER_CALL,
)

from .conftest import make_env


def _agent(seed: int = 1, train_mode: bool = True, **kwargs) -> PQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return PQN_Agent(player_id=0, env=env, train_mode=train_mode, **kwargs)


def test_pqn_score_actions_returns_value_and_one_advantage_per_action_row() -> None:
    agent = _agent(seed=2)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    value, advantage = agent.score_actions(state, legal)

    assert value.shape == (1,)
    assert advantage.shape == (len(legal),)


def test_pqn_advantage_index_aligns_with_legal_actions_not_the_clean_row() -> None:
    """`score_actions` builds rows as `[clean S row, legal[0], legal[1], ...]`
    (`pqn_agent.py`) and `PQN.forward` drops the clean row via boolean
    masking, not an index offset (`pqn.py`'s `action_mask = ~value_mask`) —
    this pins down that `advantage[i]` really is `legal_actions[i]` and not
    shifted by the S row, which `act()`'s `legal_actions[index]` depends on."""
    agent = _agent(seed=12, train_mode=False)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)
    assert len(legal) >= 2

    _, advantage_full = agent.score_actions(state, legal)
    _, advantage_first_alone = agent.score_actions(state, [legal[0]])
    _, advantage_second_alone = agent.score_actions(state, [legal[1]])

    assert torch.allclose(advantage_first_alone, advantage_full[:1], atol=1e-3)
    assert torch.allclose(advantage_second_alone, advantage_full[1:2], atol=1e-3)


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


def test_pqn_agent_progress_metrics_reports_replay_state_and_epsilon() -> None:
    agent = _agent(seed=10)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)

    metrics = agent.progress_metrics()

    assert metrics["epsilon"] == EPSILON_START
    assert metrics["pqn_replay_buffer_size"] == len(agent.replay_buffer) == 1
    assert metrics["pqn_train_steps_since_target_sync"] == 0.0


def test_epsilon_greedy_q_uses_q_argmax_when_not_exploring() -> None:
    agent = _agent(seed=10, action_selection="epsilon_greedy_q", epsilon=0.0)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    value, advantage = agent.score_actions(state, legal)
    q_values = agent._combine_q(
        value,
        advantage,
        torch.zeros(len(legal), dtype=torch.long, device=agent.device),
    )

    assert int(torch.argmax(q_values).item()) == int(torch.argmax(advantage).item())
    assert agent.act(events=[], state=state) == legal[int(torch.argmax(q_values).item())]


def test_epsilon_greedy_q_skips_scoring_for_random_action(monkeypatch) -> None:
    agent = _agent(seed=10, action_selection="epsilon_greedy_q", epsilon=1.0)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    def _unexpected_score(*_args, **_kwargs):
        raise AssertionError("epsilon-random action should not score legal actions")

    monkeypatch.setattr(agent, "score_actions", _unexpected_score)

    assert agent.act(events=[], state=state) in legal


def test_epsilon_greedy_q_uses_dueling_epsilon_decay_endpoints() -> None:
    agent = _agent(seed=10, action_selection="epsilon_greedy_q")

    agent.on_episode_start(1)
    assert agent.epsilon == EPSILON_START
    agent.on_episode_start(EPSILON_DECAY_EPISODES + 1)
    assert agent.epsilon == pytest.approx(EPSILON_END)


def test_epsilon_greedy_q_is_greedy_during_evaluation() -> None:
    agent = _agent(
        seed=10,
        train_mode=False,
        action_selection="epsilon_greedy_q",
        epsilon=1.0,
    )
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)
    value, advantage = agent.score_actions(state, legal)
    q_values = agent._combine_q(
        value,
        advantage,
        torch.zeros(len(legal), dtype=torch.long, device=agent.device),
    )

    assert agent.act(events=[], state=state) == legal[int(torch.argmax(q_values).item())]


def test_current_state_policy_entropy_is_available_to_a_future_loss() -> None:
    """Entropy shares the policy graph; logging must not be its only use."""
    agent = _agent(seed=11)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    states, actions, _, _, _, stage, _ = agent.replay_buffer.sample(batch_size=1)

    _, _, _, policy_entropy = agent._current_state_terms(states, actions, stage)

    assert policy_entropy.shape == (1,)
    assert policy_entropy.requires_grad
    assert policy_entropy.item() >= 0.0


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
        "pqn_q_loss", "pqn_policy_loss", "pqn_policy_entropy", "pqn_total_loss",
        "pqn_td_error_mean", "pqn_td_error_abs_mean",
        "pqn_td_advantage_mean", "pqn_td_advantage_abs_mean",
        "pqn_q_value_mean", "pqn_value_mean", "pqn_target_q_mean",
        "pqn_grad_norm", "pqn_grad_norm_clipped",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["pqn_grad_norm"] >= 0.0
    assert metrics["pqn_policy_entropy"] >= 0.0
    assert metrics["pqn_td_error_abs_mean"] >= 0.0
    assert metrics["pqn_grad_norm_clipped"] in (0.0, 1.0)


def test_zero_policy_coefficient_uses_only_q_loss() -> None:
    agent = _agent(seed=11, action_selection="epsilon_greedy_q", policy_loss_coef=0.0)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)

    loss = agent.train_step(batch_size=2)

    assert agent.label == "PQN_e0"
    assert loss == pytest.approx(agent.last_update_metrics["pqn_q_loss"])
    assert agent.last_update_metrics["pqn_total_loss"] == pytest.approx(
        agent.last_update_metrics["pqn_q_loss"]
    )


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

    assert agent2.action_selection == "policy_sample"
    assert agent2.label == "PQN"
    assert agent2.epsilon == agent.epsilon
    assert agent2._train_steps == agent._train_steps
    assert len(agent2.replay_buffer) == len(agent.replay_buffer)
    for key, value in agent.net.state_dict().items():
        assert torch.equal(value, agent2.net.state_dict()[key])
    for key, value in agent.target_net.state_dict().items():
        assert torch.equal(value, agent2.target_net.state_dict()[key])


def test_checkpoint_restores_epsilon_greedy_q_mode(tmp_path: Path) -> None:
    agent = _agent(seed=7, action_selection="epsilon_greedy_q")
    agent.on_episode_start(50)
    ckpt_dir = tmp_path / "pqn_e_ckpt"
    agent.save_checkpoint(ckpt_dir)

    restored = _agent(seed=8)
    restored.load_checkpoint(ckpt_dir)

    assert restored.action_selection == "epsilon_greedy_q"
    assert restored.label == "PQN_e"
    assert restored.epsilon == agent.epsilon


def test_checkpoint_restores_pqn_e0_policy_coefficient(tmp_path: Path) -> None:
    agent = _agent(seed=7, action_selection="epsilon_greedy_q", policy_loss_coef=0.0)
    ckpt_dir = tmp_path / "pqn_e0_ckpt"
    agent.save_checkpoint(ckpt_dir)

    restored = _agent(seed=8)
    restored.load_checkpoint(ckpt_dir)

    assert restored.action_selection == "epsilon_greedy_q"
    assert restored.policy_loss_coef == 0.0
    assert restored.label == "PQN_e0"


def test_load_old_checkpoint_defaults_to_policy_sample_mode(tmp_path: Path) -> None:
    agent = _agent(seed=7)
    ckpt_dir = tmp_path / "legacy_pqn_ckpt"
    agent.save_checkpoint(ckpt_dir)
    model_path = ckpt_dir / "model.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    payload.pop("action_selection")
    payload.pop("epsilon")
    payload.pop("policy_loss_coef")
    torch.save(payload, model_path)

    restored = _agent(seed=8, action_selection="epsilon_greedy_q")
    restored.load_checkpoint(ckpt_dir)

    assert restored.action_selection == "policy_sample"
    assert restored.label == "PQN"
    assert restored.epsilon == EPSILON_START
    assert restored.policy_loss_coef == PQN_POLICY_LOSS_COEF


def test_pqn_e0_reproduces_dueling_dqn_given_identical_weights() -> None:
    """`PQN_e0` (`action_selection="epsilon_greedy_q"`, `policy_loss_coef=0.0`)
    exists as a Dueling-DQN control condition for the PQN comparison
    (`Docs/PQN.md` §24.D) — this pins down that it actually behaves like one:
    same architecture (state_dict copies straight across), same Q values and
    argmax action given identical weights, same epsilon-greedy action rule
    under a shared RNG stream, and the same Bellman loss/gradient update given
    an identical replay minibatch (the extra, always-computed `policy_loss`
    contributes exactly zero since its coefficient is `0.0`)."""
    env = make_env(seed=3, agent_kind="ai")
    torch.manual_seed(0)
    dueling = Dueling_DQN_Agent(player_id=0, env=env, train_mode=True, seed=7)
    torch.manual_seed(0)
    pqn_e0 = PQN_Agent(player_id=0, env=env, train_mode=True, seed=7,
                        action_selection="epsilon_greedy_q", policy_loss_coef=0.0)
    assert pqn_e0.label == "PQN_e0"

    assert dueling.net.state_dict().keys() == pqn_e0.net.state_dict().keys()
    pqn_e0.net.load_state_dict(dueling.net.state_dict())
    pqn_e0.target_net.load_state_dict(dueling.target_net.state_dict())

    state = env.current_state()
    legal = env.legal_actions(state)
    dueling_q = dueling.score_actions(state, legal)
    value, advantage = pqn_e0.score_actions(state, legal)
    group_index = torch.zeros(len(legal), dtype=torch.long, device=pqn_e0.device)
    pqn_q = pqn_e0._combine_q(value, advantage, group_index)
    assert torch.allclose(dueling_q, pqn_q, atol=1e-3)
    assert int(torch.argmax(dueling_q).item()) == int(torch.argmax(pqn_q).item())

    dueling.epsilon = pqn_e0.epsilon = 0.5
    dueling._rng = random.Random(123)
    pqn_e0._rng = random.Random(123)
    for _ in range(20):
        assert dueling.act(events=[], state=state) == pqn_e0.act(events=[], state=state)

    dueling.epsilon = pqn_e0.epsilon = 0.0
    action = legal[0]
    next_state = state.snapshot()
    for i in range(BATCH_SIZE):
        r = float(i % 5) - 2.0
        done = (i % 7 == 0)
        dueling.remember(state, action, r, next_state, done)
        pqn_e0.remember(state, action, r, next_state, done)

    random.seed(999)
    dueling_loss = dueling.train_step(BATCH_SIZE)
    random.seed(999)
    pqn_loss = pqn_e0.train_step(BATCH_SIZE)
    assert pqn_loss == pytest.approx(dueling_loss, rel=1e-6)
    for key in dueling.net.state_dict():
        assert torch.allclose(
            dueling.net.state_dict()[key], pqn_e0.net.state_dict()[key], atol=1e-3
        )
