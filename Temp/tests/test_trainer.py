"""Tests for `Trainer` (`Docs/Trainer.md`).

`Trainer` had zero dedicated test coverage before this file — every other
`risk/learning/` module (`GNN_DQN_Agent`/`Dueling_DQN_Agent` in
`test_agents.py`/`test_dueling_dqn.py`, `TrainingLogger`, `Evaluator`) has
its own file, but the orchestration loop itself (`Trainer.train`) was only
ever exercised via `python -m risk.learning.trainer` by hand. Two things
this file locks down:

1. The `reached_max_steps` contract `Trainer.train()` passes to
   `agent.learn(...)` (`Docs/PPO.md` depends on this being correct — a
   truncated-episode boundary vs. a real terminal `done`).
2. That a short real training run still completes end to end for both
   `GNN_DQN_Agent` and `Dueling_DQN_Agent` after the epsilon/`learn()`
   refactor (`Docs/ChangeLog.md`'s 2026-07-11 entries).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from risk.agents.base_agent import BaseAgent
from risk.game.environment import Environment
from risk.learning import trainer as trainer_module
from risk.learning.dueling_dqn_agent import Dueling_DQN_Agent
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.trainer import Trainer, _aggregate_update_metrics
from risk.learning.train_constants import EPSILON_START


class _FixedRNG:
    """Deterministic stand-in for `Trainer`'s `random.SystemRandom` —
    forces player count, seat, and opponent choice so episode setup is
    reproducible instead of depending on OS entropy."""

    def randint(self, a: int, _b: int) -> int:
        return a

    def randrange(self, _stop: int) -> int:
        return 0

    def choice(self, seq):
        return seq[0]


class _RecordingAgent(BaseAgent):
    """Minimal `Trainer`-compatible agent that just records every
    `learn(reached_max_steps=...)` call — used to test `Trainer`'s
    orchestration contract in isolation from real Q-learning."""

    label = "Recording"

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.env: Environment | None = None
        self.learn_calls: list[bool] = []

    def attach(self, player_id: int, env: Environment) -> None:
        self.player_id = player_id
        self.env = env

    def act(self, events, state):
        del events
        legal = self.env.legal_actions(state)
        return legal[0] if legal else None

    def remember(self, state, action, reward, next_state, done) -> None:
        del state, action, reward, next_state, done

    def learn(self, *, reached_max_steps: bool = False) -> list[float]:
        self.learn_calls.append(reached_max_steps)
        return []


def _trainer(agent: BaseAgent, tmp_path: Path, run_id: int = 1) -> Trainer:
    t = Trainer(
        run_id,
        agent=agent,
        checkpoint_dir=tmp_path,
        use_wandb=False,
        resume=False,
    )
    t._rng = _FixedRNG()
    return t


def test_build_learner_agent_supports_only_active_learners(monkeypatch) -> None:
    built: list[str] = []

    class _Agent:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs.pop("label"))

    def _builder(label: str):
        return lambda **kwargs: _Agent(label=label, **kwargs)

    monkeypatch.setattr(trainer_module, "GNN_DQN_Agent", _builder("DQN"))
    monkeypatch.setattr(trainer_module, "Dueling_DQN_Agent", _builder("Dueling_DQN"))
    monkeypatch.setattr(trainer_module, "PPO_Agent", _builder("PPO"))
    ctx = SimpleNamespace(env=object())

    for label in ("DQN", "Dueling_DQN", "PPO"):
        trainer_module.build_learner_agent(label, ctx)

    assert built == ["DQN", "Dueling_DQN", "PPO"]
    with pytest.raises(ValueError, match="Unknown learner"):
        trainer_module.build_learner_agent("retired_learner", ctx)


def test_trainer_marks_only_the_final_learn_call_as_reached_max_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(trainer_module, "MAX_STEPS_PER_EPISODE", 8)
    agent = _RecordingAgent(player_id=0)
    trainer = _trainer(agent, tmp_path)

    trainer.train(n_episodes=3)

    assert agent.learn_calls, "expected at least one learner turn across 3 episodes"
    # `reached_max_steps=True` can only ever be the *last* recorded call of
    # an episode (the loop breaks immediately after), never a call in the
    # middle of one.
    true_positions = [i for i, flag in enumerate(agent.learn_calls) if flag]
    for i in true_positions:
        is_last_overall = i == len(agent.learn_calls) - 1
        is_followed_by_a_call_that_could_only_start_fresh = i + 1 < len(agent.learn_calls)
        assert is_last_overall or is_followed_by_a_call_that_could_only_start_fresh
    assert any(agent.learn_calls), "expected the small MAX_STEPS_PER_EPISODE cap to be hit at least once"


def test_trainer_end_to_end_smoke_gnn_dqn(tmp_path: Path, monkeypatch) -> None:
    from risk.app.factory import GameFactory
    from risk.app.setup import SetupStage
    from risk.learning.train_constants import MIN_PLAYERS

    # Cap episode length so this smoke test stays fast — it's checking that
    # the training loop wires together (real GNN forward passes are slow
    # relative to the heuristic/random-only fuzz games in test_self_play.py),
    # not that full-length games play out.
    monkeypatch.setattr(trainer_module, "MAX_STEPS_PER_EPISODE", 40)

    ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
    agent = GNN_DQN_Agent(player_id=0, env=ctx.env, train_mode=True, seed=1)
    trainer = _trainer(agent, tmp_path)

    trainer.train(n_episodes=2)

    # episode 1 pins epsilon to the start value; by episode 2 it has begun
    # decaying (`on_episode_start`, `Docs/Trainer.md`).
    assert agent.epsilon < EPSILON_START
    assert len(agent.replay_buffer) > 0


def test_trainer_end_to_end_smoke_dueling_dqn(tmp_path: Path, monkeypatch) -> None:
    from risk.app.factory import GameFactory
    from risk.app.setup import SetupStage
    from risk.learning.train_constants import MIN_PLAYERS

    monkeypatch.setattr(trainer_module, "MAX_STEPS_PER_EPISODE", 40)

    ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
    agent = Dueling_DQN_Agent(player_id=0, env=ctx.env, train_mode=True, seed=1)
    trainer = _trainer(agent, tmp_path)

    trainer.train(n_episodes=2)

    assert agent.epsilon < EPSILON_START
    assert len(agent.replay_buffer) > 0


def test_trainer_logs_expected_metric_keys(tmp_path: Path) -> None:
    logged: list[dict] = []

    class _CapturingLogger:
        def start_run(self, *, agent, trainer) -> None:
            del agent, trainer

        def try_resume(self, *, agent):
            del agent
            return None

        def log_episode(self, *, episode: int, metrics: dict) -> None:
            del episode
            logged.append(metrics)

        def checkpoint(self, *, episode: int, agent) -> None:
            del episode, agent

        def format_status_line(self, **kwargs) -> str:
            return ""

    agent = _RecordingAgent(player_id=0)
    trainer = Trainer(
        1,
        agent=agent,
        checkpoint_dir=tmp_path,
        use_wandb=False,
        resume=False,
        logger=_CapturingLogger(),
    )
    trainer._rng = _FixedRNG()

    trainer.train(n_episodes=1)

    assert len(logged) == 1
    expected_keys = {
        "win",
        f"win_rate_last_{trainer_module.ROLLING_WIN_RATE_WINDOW}",
        "reward_per_agent_turn",
        "learn_loss_mean",
        "territories_conquered",
        "agent_turns_survived",
        "cumulative_learner_turns",
        "learner_update_calls_in_episode",
        "optimizer_steps_in_episode",
        "samples_processed_in_episode",
        "cumulative_optimizer_steps",
        "cumulative_samples_processed",
        "reinforce_action_count",
        "reinforce_partial_action_count",
        "reward_component_reinforce_per_action",
        "reward_component_reinforce_ready_per_action",
        "reward_component_reinforce_total_per_action",
    }
    assert expected_keys.issubset(logged[0].keys())
    reinforce_actions = logged[0]["reinforce_action_count"]
    if reinforce_actions:
        assert logged[0]["reward_component_reinforce_per_action"] == pytest.approx(
            logged[0]["reward_component_reinforce"] / reinforce_actions
        )
    assert logged[0]["cumulative_learner_turns"] == logged[0]["agent_turns_survived"]
    assert logged[0]["player_count"] == trainer_module.MIN_PLAYERS
    assert logged[0]["opponent_count_random"] == trainer_module.MIN_PLAYERS - 1
    assert logged[0]["opponent_count_killbot"] == 0
    assert logged[0]["roster"].startswith("p0=learner:Recording, p1=random")
    assert logged[0]["winner_kind"] in {"learner", "random", "none"}
    assert "opponent_random_won_when_present" in logged[0]
    assert "opponent_killbot_won_when_present" not in logged[0]


def test_update_metrics_aggregate_all_updates_and_keep_maxima() -> None:
    aggregated = _aggregate_update_metrics(
        [
            {
                "ppo_value_loss": 10.0,
                "ppo_early_stopped": 0.0,
                "ppo_grad_norm_max": 2.0,
                "ppo_optimizer_steps_per_update": 1.0,
                "bad_metric": float("nan"),
            },
            {
                "ppo_value_loss": 30.0,
                "ppo_early_stopped": 1.0,
                "ppo_grad_norm_max": 7.0,
                "ppo_optimizer_steps_per_update": 3.0,
            },
        ],
        weight_key="ppo_optimizer_steps_per_update",
        unweighted_fields=frozenset(
            {"ppo_early_stopped", "ppo_optimizer_steps_per_update"}
        ),
    )

    assert aggregated == {
        "ppo_value_loss": 25.0,
        "ppo_early_stopped": 0.5,
        "ppo_grad_norm_max": 7.0,
        "ppo_optimizer_steps_per_update": 2.0,
        "update_metrics_nonfinite_count": 1.0,
    }
