"""Tests for `TrainingLogger` (`risk/learning/training_logger.py`).

Agent-internals round-tripping (net/optimizer/replay buffer state) is
already covered by `test_agents.py`'s `GNN_DQN_Agent.save_checkpoint`/
`load_checkpoint` test — these tests only cover what `TrainingLogger`
itself is responsible for: config building, checkpoint path/cadence
orchestration, and no-op behavior when W&B is disabled.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from risk.learning import train_constants
from risk.learning import training_logger
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.training_logger import TrainingLogger

from .conftest import make_env


def _agent(seed: int = 1) -> GNN_DQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return GNN_DQN_Agent(player_id=0, env=env, train_mode=True)


def test_checkpoint_cadence_starts_at_two_hundred_then_runs_every_fifty_episodes() -> None:
    assert train_constants.CHECKPOINT_AFTER == 200
    assert train_constants.CHECKPOINT_EVERY == 50


def test_use_wandb_false_disables_wandb_and_is_a_noop(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False)
    agent = _agent()

    assert logger._wandb_enabled is False
    # None of these should raise even though no wandb run was started.
    logger.start_run(agent=agent, trainer=None)
    logger.log_episode(episode=1, metrics={"episode_reward": 1.0})
    logger.finish()


def test_start_run_resumes_the_explicit_wandb_run_id(tmp_path: Path, monkeypatch) -> None:
    init_calls: list[dict] = []
    monkeypatch.setattr(
        training_logger,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs)),
    )
    logger = TrainingLogger(
        run_id=101,
        checkpoint_dir=tmp_path,
        run_name="Dueling_DQN_101",
        wandb_run_id="rc1itpev",
    )

    logger.start_run(agent=_agent(), trainer=None)

    assert init_calls[0]["id"] == "rc1itpev"
    assert init_calls[0]["resume"] == "must"


def test_log_episode_can_use_episode_as_the_wandb_step(tmp_path: Path, monkeypatch) -> None:
    log_calls: list[tuple[dict, int]] = []
    monkeypatch.setattr(
        training_logger,
        "wandb",
        SimpleNamespace(log=lambda payload, *, step: log_calls.append((payload, step))),
    )
    logger = TrainingLogger(
        run_id=101,
        checkpoint_dir=tmp_path,
        wandb_step_from_episode=True,
    )

    logger.log_episode(episode=601, metrics={"win": 1})

    assert log_calls == [({"win": 1, "episode": 601}, 601)]


def test_status_line_identifies_the_learner_seat(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=50, checkpoint_dir=tmp_path, use_wandb=False)
    env = make_env(seed=3, agent_kind="ai")

    line = logger.format_status_line(
        topology=env.topology,
        episode=45,
        n_episodes=10_000,
        step_count=1819,
        agent_turns=477,
        seat=2,
        current_state=env.current_state(),
    )

    assert "run 050" in line
    assert "learner p2" in line


def test_build_config_includes_constants_and_model_identity(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=7, checkpoint_dir=tmp_path, use_wandb=False, run_name="DQN_007")
    agent = _agent()

    config = logger._build_config(agent)

    for name in train_constants.__all__:
        assert config[name] == getattr(train_constants, name)
    assert config["run_id"] == 7
    assert config["run_name"] == "DQN_007"
    assert config["agent_class"] == "GNN_DQN_Agent"
    assert config["model_class"] == "GNN_DQN"
    assert config["model_str"] == str(agent.net)
    assert config["target_model_str"] == str(agent.target_net)
    assert config["param_count"] == sum(p.numel() for p in agent.net.parameters())
    assert config["device"] == str(agent.device)


def test_build_config_records_optional_action_selection(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=7, checkpoint_dir=tmp_path, use_wandb=False)
    agent = _agent()
    agent.action_selection = "epsilon_greedy_q"

    assert logger._build_config(agent)["action_selection"] == "epsilon_greedy_q"


def test_save_checkpoint_and_try_resume_round_trip(tmp_path: Path) -> None:
    agent = _agent(seed=2)
    state = agent.env.current_state()
    action = agent.env.legal_actions(state)[0]
    agent.remember(state, action, 1.0, state.snapshot(), False)
    agent.train_step(batch_size=1)

    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False, resume=True)
    saved_path = logger.save_checkpoint(episode=42, agent=agent)
    assert saved_path == tmp_path / "ep000042"

    fresh_agent = _agent(seed=2)
    resumed = logger.try_resume(agent=fresh_agent)

    assert resumed == {"episode": 42}
    assert fresh_agent._train_steps == agent._train_steps
    assert len(fresh_agent.replay_buffer) == len(agent.replay_buffer)


def test_try_resume_returns_none_when_no_checkpoints_exist(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path / "does_not_exist", use_wandb=False, resume=True)
    agent = _agent()

    assert logger.try_resume(agent=agent) is None


def test_try_resume_returns_none_when_resume_disabled(tmp_path: Path) -> None:
    agent = _agent(seed=3)
    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False, resume=True)
    logger.save_checkpoint(episode=5, agent=agent)

    disabled_logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False, resume=False)
    assert disabled_logger.try_resume(agent=_agent()) is None


def test_save_checkpoint_picks_the_latest_episode(tmp_path: Path) -> None:
    agent = _agent(seed=4)
    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False, resume=True)
    logger.save_checkpoint(episode=10, agent=agent)
    logger.save_checkpoint(episode=20, agent=agent)
    logger.save_checkpoint(episode=15, agent=agent)

    resumed = logger.try_resume(agent=_agent(seed=4))

    assert resumed == {"episode": 20}
