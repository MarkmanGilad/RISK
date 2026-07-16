"""Tests for `Dueling_DQN`/`Dueling_DQN_Agent` (`Docs/DuelingDQN.md`).

Narrow, focused tests per the build plan's "Tests" section — network-level
grouping/value behavior plus the agent call sites that changed to pass a
`group_index` (`score_actions`, `_max_next_ddqn_q`) or restructure around one
(`_q_value`, exercised indirectly via `train_step` in the checkpoint test).
Agent plumbing shared with `GNN_DQN_Agent` (device selection, `remember`,
`act`) is already covered by `test_agents.py` and isn't re-tested here.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from risk.learning.dueling_dqn_agent import Dueling_DQN_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    TRAIN_STEPS_PER_CALL,
)

from .conftest import make_env


def _agent(seed: int = 1, train_mode: bool = True) -> Dueling_DQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return Dueling_DQN_Agent(player_id=0, env=env, train_mode=train_mode)


def test_dueling_dqn_forward_returns_one_q_per_action_row() -> None:
    agent = _agent(seed=2)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    q = agent.score_actions(state, legal)

    assert q.shape == (len(legal),)


def test_equal_advantage_group_collapses_q_to_value() -> None:
    agent = _agent(seed=3)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)
    assert len({a.phase for a in legal}) == 1  # one decision -> one dueling group

    # Zero the head scoring this phase's actions so every row's advantage is
    # 0 -> mean(A) is 0 -> Q must equal the clean-row V(s) for every row.
    head = agent.net._heads_by_phase[int(legal[0].phase)]
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()

    q = agent.score_actions(state, legal)

    assert torch.allclose(q, q[0].expand_as(q), atol=1e-6)


def test_two_groups_in_one_batch_normalize_independently() -> None:
    agent = _agent(seed=4)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)
    assert len(legal) >= 6, "need enough legal actions to split into two groups"

    group_a_actions = legal[:3]
    group_b_actions = legal[3:6]

    def _rows_phase_cards_value_mask(actions):
        base = agent.adapter(state, perspective=agent.player_id)
        rows = [base]
        rows.extend(agent.builder(base, a, state) for a in actions)
        encoded = agent.action_encoder.encode_many(actions, state)
        phase = torch.cat([encoded[:1, 0], encoded[:, 0]])
        cards = torch.cat([torch.zeros((1, 3), dtype=torch.long), encoded[:, 1:4]])
        value_mask = torch.tensor([True] + [False] * len(actions), dtype=torch.bool)
        return rows, phase, cards, value_mask

    rows_a, phase_a, cards_a, value_mask_a = _rows_phase_cards_value_mask(group_a_actions)
    rows_b, phase_b, cards_b, value_mask_b = _rows_phase_cards_value_mask(group_b_actions)

    def _score_alone(rows, phase, cards, value_mask):
        batch = Batch.from_data_list(rows).to(agent.device)
        with torch.no_grad():
            return agent.net(
                batch,
                phase.to(agent.device),
                cards.to(agent.device),
                group_index=torch.zeros(len(rows), dtype=torch.long),
                value_mask=value_mask,
            )

    q_a_alone = _score_alone(rows_a, phase_a, cards_a, value_mask_a)
    q_b_alone = _score_alone(rows_b, phase_b, cards_b, value_mask_b)

    combined_rows = rows_a + rows_b
    combined_phase = torch.cat([phase_a, phase_b])
    combined_cards = torch.cat([cards_a, cards_b])
    combined_group_index = torch.tensor([0] * len(rows_a) + [1] * len(rows_b), dtype=torch.long)
    combined_value_mask = torch.cat([value_mask_a, value_mask_b])

    combined_batch = Batch.from_data_list(combined_rows).to(agent.device)
    with torch.no_grad():
        q_combined = agent.net(
            combined_batch,
            combined_phase.to(agent.device),
            combined_cards.to(agent.device),
            group_index=combined_group_index,
            value_mask=combined_value_mask,
        )

    assert torch.allclose(q_combined[:3], q_a_alone, atol=1e-6)
    assert torch.allclose(q_combined[3:], q_b_alone, atol=1e-6)


def test_score_actions_keeps_tensors_on_selected_device() -> None:
    agent = _agent(seed=5, train_mode=False)
    state = agent.env.current_state()
    legal = agent.env.legal_actions(state)

    q = agent.score_actions(state, legal)

    assert q.shape == (len(legal),)
    assert q.device.type == agent.device.type


def test_max_next_ddqn_q_returns_zero_for_done_transitions() -> None:
    agent = _agent(seed=6)
    state = agent.env.current_state()

    max_q = agent._max_next_ddqn_q(
        next_states=[state],
        done=torch.tensor([True]),
        next_stage=torch.tensor([int(state.phase)], dtype=torch.long),
    )

    assert torch.equal(max_q, torch.zeros(1, device=agent.device))


def test_dueling_dqn_agent_on_episode_start_decays_epsilon() -> None:
    agent = _agent(seed=8)

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


def test_dueling_dqn_agent_learn_accepts_reached_max_steps() -> None:
    agent = _agent(seed=8)

    assert agent.learn(reached_max_steps=True) == []


def test_dueling_dqn_agent_can_train_threshold_matches_batch_size() -> None:
    """`can_train()`/`learn()` now read `BATCH_SIZE`/`TRAIN_STEPS_PER_CALL`
    directly instead of receiving them as call-time arguments
    (`Docs/ChangeLog.md`'s 2026-07-05 entry) — this pins down that the
    threshold behavior is unchanged by that refactor."""
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


def test_dueling_dqn_agent_reached_max_steps_flag_is_inert_with_full_replay() -> None:
    """`reached_max_steps` is reserved for on-policy agents (`Docs/PPO.md`)
    — for `Dueling_DQN_Agent` it must not change the update at all. Compares
    the returned losses (not raw net weights: the GNN encoder's scatter
    aggregation is not bit-deterministic across forward passes even with
    fixed seeds, confirmed empirically against `HEAD`, so a tolerant
    comparison on the observable output is the meaningful check here)."""

    def _make_and_fill(seed: int) -> Dueling_DQN_Agent:
        torch.manual_seed(seed)
        env = make_env(seed=5, agent_kind="ai")
        agent = Dueling_DQN_Agent(player_id=0, env=env, train_mode=True, seed=seed)
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


def test_dueling_dqn_agent_progress_metrics_reports_epsilon_and_replay_state() -> None:
    agent = _agent(seed=10)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)

    metrics = agent.progress_metrics()

    assert metrics["epsilon"] == agent.epsilon
    assert metrics["dqn_replay_buffer_size"] == len(agent.replay_buffer) == 1
    assert metrics["dqn_train_steps_since_target_sync"] == 0.0


def test_dueling_dqn_agent_train_step_populates_last_update_metrics() -> None:
    agent = _agent(seed=11)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)

    assert agent.last_update_metrics == {}
    agent.train_step(batch_size=2)

    metrics = agent.last_update_metrics
    expected_keys = {
        "dqn_td_error_mean", "dqn_td_error_abs_mean", "dqn_td_error_std",
        "dqn_td_error_abs_max", "dqn_q_value_mean", "dqn_q_value_std",
        "dqn_target_q_mean", "dqn_target_q_std", "dqn_grad_norm",
        "dqn_grad_norm_clipped",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["dqn_grad_norm"] >= 0.0
    assert metrics["dqn_td_error_abs_mean"] >= 0.0
    assert metrics["dqn_grad_norm_clipped"] in (0.0, 1.0)


def test_save_and_load_checkpoint_round_trips_dueling_net_and_target(tmp_path: Path) -> None:
    agent = _agent(seed=7)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.remember(state, action, -1.0, state.snapshot(), True)
    agent.train_step(batch_size=2)

    ckpt_dir = tmp_path / "dueling_ckpt"
    agent.save_checkpoint(ckpt_dir)

    agent2 = Dueling_DQN_Agent(player_id=0, env=agent.env, train_mode=True)
    agent2.load_checkpoint(ckpt_dir)

    assert agent2._train_steps == agent._train_steps
    assert agent2.epsilon == agent.epsilon
    assert len(agent2.replay_buffer) == len(agent.replay_buffer)
    for key, value in agent.net.state_dict().items():
        assert torch.equal(value, agent2.net.state_dict()[key])
    for key, value in agent.target_net.state_dict().items():
        assert torch.equal(value, agent2.target_net.state_dict()[key])
