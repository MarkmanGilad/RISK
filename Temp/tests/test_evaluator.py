"""Tests for `Evaluator` — see `Docs/Eval.md`."""
from __future__ import annotations

from pathlib import Path

from risk.learning.evaluator import Evaluator
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent

from .conftest import make_env


def _agent(seed: int = 42) -> GNN_DQN_Agent:
    env = make_env(seed=seed, agent_kind="ai")
    return GNN_DQN_Agent(player_id=0, env=env, train_mode=False)


def test_evaluate_returns_expected_metric_keys(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=3, best_dir=tmp_path / "best")
    agent = _agent()

    result = evaluator.evaluate(agent, episode=10)

    expected_keys = {
        "episode",
        "eval_games",
        "eval_win_rate",
        "eval_avg_territories_conquered",
        "eval_avg_reward_per_agent_turn",
        "eval_avg_agent_turns_survived",
        "eval_score",
    }
    assert expected_keys.issubset(result.keys())
    assert result["episode"] == 10
    assert result["eval_games"] == 6  # 2 suites x 3 seeds


def test_evaluate_restores_epsilon_and_train_mode(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=3, best_dir=tmp_path / "best")
    agent = _agent()
    agent.epsilon = 0.7
    agent.set_train_mode(True)

    evaluator.evaluate(agent, episode=1)

    assert agent.epsilon == 0.7
    assert agent.train_mode is True


def test_evaluate_uses_fixed_rotating_learner_seats(tmp_path: Path, monkeypatch) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=3, best_dir=tmp_path / "best")
    calls: list[tuple[tuple[str, ...], int, int]] = []
    monkeypatch.setattr(
        evaluator,
        "_play_one",
        lambda _agent, opponents, seed, learner_seat: calls.append(
            (opponents, seed, learner_seat)
        )
        or {
            "win": 0,
            "territories_conquered": 0,
            "reward_per_agent_turn": 0.0,
            "agent_turns_survived": 0,
        },
    )

    evaluator.evaluate(_agent(), episode=1)

    assert [seat for _, _, seat in calls] == [0, 1, 2, 0, 2, 4]


def test_evaluate_is_deterministic_across_calls(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=3, best_dir=tmp_path / "best")
    agent = _agent()

    first = evaluator.evaluate(agent, episode=1)
    second = evaluator.evaluate(agent, episode=1)

    assert first == second


def test_score_formula_matches_documented_weights(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=3, best_dir=tmp_path / "best")
    metrics = {
        "eval_win_rate": 0.5,
        "eval_avg_territories_conquered": 2.0,
        "eval_avg_reward_per_agent_turn": 1.5,
    }

    score = evaluator._score(metrics)

    assert score == 100.0 * 0.5 + 2.0 * 2.0 + 5.0 * 1.5


def test_maybe_save_best_keeps_top_n_and_sorts_manifest(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=2, best_dir=tmp_path / "best")
    agent = _agent()

    scores = [10.0, 30.0, 20.0, 5.0]
    saved_flags = [
        evaluator.maybe_save_best(agent, {"episode": i * 100, "eval_score": s})
        for i, s in enumerate(scores)
    ]

    # 5.0 arrives last, after 2 better scores already fill keep_best=2 -> not saved.
    assert saved_flags == [True, True, True, False]

    manifest = evaluator._load_manifest(evaluator.best_dir / "manifest.json")
    assert [entry["score"] for entry in manifest] == [30.0, 20.0]
    assert len(list(evaluator.best_dir.glob("*.pt"))) == 2


def test_maybe_save_best_rejects_score_not_better_than_worst_kept(tmp_path: Path) -> None:
    evaluator = Evaluator(max_steps=20, keep_best=1, best_dir=tmp_path / "best")
    agent = _agent()

    assert evaluator.maybe_save_best(agent, {"episode": 1, "eval_score": 50.0}) is True
    assert evaluator.maybe_save_best(agent, {"episode": 2, "eval_score": 10.0}) is False

    manifest = evaluator._load_manifest(evaluator.best_dir / "manifest.json")
    assert len(manifest) == 1
    assert manifest[0]["score"] == 50.0
