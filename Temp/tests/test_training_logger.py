"""Tests for `TrainingLogger` (`risk/learning/training_logger.py`).

Agent-internals round-tripping (net/optimizer/replay buffer state) is
already covered by `test_agents.py`'s `GNN_DQN_Agent.save_checkpoint`/
`load_checkpoint` test — these tests only cover what `TrainingLogger`
itself is responsible for: config building, checkpoint path/cadence
orchestration, and no-op behavior when W&B is disabled.
"""
from __future__ import annotations

from pathlib import Path

from risk.learning import train_constants
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.training_logger import TrainingLogger

from .conftest import make_env


def _agent(seed: int = 1) -> GNN_DQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return GNN_DQN_Agent(player_id=0, env=env, train_mode=True)


def test_use_wandb_false_disables_wandb_and_is_a_noop(tmp_path: Path) -> None:
    logger = TrainingLogger(run_id=1, checkpoint_dir=tmp_path, use_wandb=False)
    agent = _agent()

    assert logger._wandb_enabled is False
    # None of these should raise even though no wandb run was started.
    logger.start_run(agent=agent, trainer=None)
    logger.log_episode(episode=1, metrics={"episode_reward": 1.0})
    logger.finish()


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
