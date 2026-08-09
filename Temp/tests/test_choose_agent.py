"""Tests for the independent saved-policy evaluators (Docs/ChooseAgent.md)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk.learning.choose_agent import (
    AgentMatchEvaluator,
    CheckpointEvaluator,
    _WandbLogger,
    _cached_policy_state,
    _play_game,
    show_checkpoint_win_rate_chart,
)


def _checkpoint_row(opponents, learner_seat, seed):
    return {
        "player_count": len(opponents) + 1,
        "learner_seat": learner_seat,
        "seed": seed,
        "opponent_kinds": list(opponents),
        "win": int(learner_seat == 0),
        "winner_kind": "learner" if learner_seat == 0 else "opponent",
        "winner_seat": learner_seat if learner_seat == 0 else 1,
        "reached_max_steps": False,
        "final_territory_count": 4,
        "final_army_count": 12,
        "territories_conquered": 2,
        "episode_reward_sum": 3.0,
        "agent_turns_survived": 5,
        "reward_per_agent_turn": 0.6,
        "step_count": 25,
    }


def _make_checkpoint_tree(tmp_path: Path) -> Path:
    run_dir = tmp_path / "DQN_103"
    for name in ("ep000200", "ep002000", "ep000050", "other"):
        (run_dir / name).mkdir(parents=True)
    return run_dir


def test_checkpoint_evaluator_filters_orders_and_writes_raw_schedule(tmp_path, monkeypatch) -> None:
    _make_checkpoint_tree(tmp_path)
    monkeypatch.setattr("risk.learning.choose_agent.CHECKPOINT_DIR", str(tmp_path))
    evaluator = CheckpointEvaluator(max_steps=30, seeds=(7,), use_wandb=False)
    calls = []
    monkeypatch.setattr(
        evaluator,
        "_play_checkpoint_game",
        lambda kind, policy_state, opponents, seat, seed, *_: calls.append((policy_state["name"], seat, seed))
        or _checkpoint_row(opponents, seat, seed),
    )

    monkeypatch.setattr(
        "risk.learning.choose_agent._cached_policy_state",
        lambda _cache, checkpoint: {"name": checkpoint.name},
    )
    result = evaluator.evaluate_run("DQN", 103, min_episode=100, max_episode=2000)

    assert list(result["checkpoints"]) == ["ep000200", "ep002000"]
    assert len(calls) == 2 * (3 + 4 + 5 + 6)
    assert all(entry["scheduled_games"] == 18 for entry in result["checkpoints"].values())
    saved = tmp_path / "DQN_103" / "evaluations" / "checkpoint_eval_ep000200_to_002000.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == result


def test_checkpoint_evaluator_resumes_individual_games(tmp_path, monkeypatch) -> None:
    _make_checkpoint_tree(tmp_path)
    monkeypatch.setattr("risk.learning.choose_agent.CHECKPOINT_DIR", str(tmp_path))
    output = tmp_path / "result.json"
    evaluator = CheckpointEvaluator(max_steps=30, seeds=(0,), use_wandb=False)
    calls = []
    monkeypatch.setattr(
        evaluator,
        "_play_checkpoint_game",
        lambda kind, policy_state, opponents, seat, seed, *_: calls.append((policy_state["name"], seat, seed))
        or _checkpoint_row(opponents, seat, seed),
    )

    monkeypatch.setattr(
        "risk.learning.choose_agent._cached_policy_state",
        lambda _cache, checkpoint: {"name": checkpoint.name},
    )
    first = evaluator.evaluate_run("DQN", 103, min_episode=200, max_episode=200, output_path=output)
    second = evaluator.evaluate_run("DQN", 103, min_episode=50, max_episode=200, output_path=output)

    assert len(calls) == (3 + 4 + 5 + 6) * 2
    assert first["checkpoints"]["ep000200"]["completed_games"] == 18
    assert second["checkpoints"]["ep000200"]["completed_games"] == 18
    assert second["checkpoints"]["ep000050"]["completed_games"] == 18


@pytest.mark.parametrize("minimum, maximum", [(-1, None), (None, -1), (3, 2)])
def test_checkpoint_evaluator_rejects_invalid_ranges(minimum, maximum) -> None:
    evaluator = CheckpointEvaluator(use_wandb=False)
    with pytest.raises(ValueError):
        evaluator.evaluate_run("DQN", 1, min_episode=minimum, max_episode=maximum)


def test_match_validation_and_cyclic_result_totals(tmp_path, monkeypatch) -> None:
    evaluator = AgentMatchEvaluator(max_steps=30, seeds=(0,), use_wandb=False)

    def fake_game(specs, seed, rotation):
        roster = specs[rotation:] + specs[:rotation]
        return {
            "seed": seed,
            "rotation": rotation,
            "roster": [spec["name"] for spec in roster],
            "winner": roster[0]["name"],
            "winner_seat": 0,
            "reached_max_steps": False,
            "step_count": 4,
            "participants": {
                spec["name"]: {
                    "win": int(index == 0), "agent_turns_survived": 1,
                    "territories_conquered": 0, "final_territory_count": 1,
                    "final_army_count": 1,
                }
                for index, spec in enumerate(roster)
            },
        }

    monkeypatch.setattr(evaluator, "_play_match_game", fake_game)
    participants = [
        {"name": "saved", "kind": "checkpoint", "agent_kind": "DQN", "checkpoint": "x.pt"},
        {"name": "raider", "kind": "heuristic", "agent_kind": "raider"},
        {"name": "killbot", "kind": "heuristic", "agent_kind": "killbot"},
    ]
    result = evaluator.evaluate(participants, output_path=tmp_path / "match.json")

    assert [game["roster"] for game in result["games"]] == [
        ["saved", "raider", "killbot"], ["raider", "killbot", "saved"], ["killbot", "saved", "raider"],
    ]
    assert all(total["total_wins"] == 1 for total in result["totals"].values())
    assert all(total["completed_games"] == 3 for total in result["totals"].values())


def test_match_rejects_bad_participants(tmp_path) -> None:
    evaluator = AgentMatchEvaluator(use_wandb=False)
    with pytest.raises(ValueError):
        evaluator.evaluate([], output_path=tmp_path / "match.json")
    with pytest.raises(ValueError):
        evaluator.evaluate(
            [{"name": "one", "kind": "heuristic", "agent_kind": "not-real"}] * 3,
            output_path=tmp_path / "match.json",
        )


def test_policy_cache_reads_a_checkpoint_once(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    reads = []
    monkeypatch.setattr(
        "risk.learning.choose_agent._read_policy_state",
        lambda path: reads.append(path) or {"weight": 1},
    )

    cache = {}
    first = _cached_policy_state(cache, checkpoint)
    second = _cached_policy_state(cache, checkpoint)

    assert first is second
    assert reads == [checkpoint.resolve()]


def test_end_of_turn_reward_is_added_when_opponent_eliminates_learner() -> None:
    class State:
        def __init__(self, current_player_index, eliminated=()):
            self.current_player_index = current_player_index
            self.eliminated = tuple(eliminated)
            self.owners = (0, 1)
            self.armies = (3, 3)

        def snapshot(self):
            return State(self.current_player_index, self.eliminated)

    first = State(0)
    second = State(1)
    terminal = State(1, eliminated=(0,))

    class Reward:
        def end_of_turn(self, before, after, seat):
            assert before.current_player_index == 0
            assert after is terminal
            assert seat == 0
            return 10.0

    class Environment:
        def __init__(self):
            self.state = first
            self.reward = Reward()
            self.calls = 0

        def is_terminal(self):
            return self.calls >= 2

        def current_state(self):
            return self.state

        def step(self, action, reward_player):
            self.calls += 1
            self.state = second if self.calls == 1 else terminal
            return SimpleNamespace(state=self.state, reward=float(self.calls), done=self.calls == 2)

        def winner(self):
            return 1 if self.calls >= 2 else None

    env = Environment()
    outcome = _play_game(
        SimpleNamespace(env=env),
        [lambda _unused, _state: object(), lambda _unused, _state: object()],
        tracked_seats=(0,), reward_seat=0, max_steps=3,
    )

    assert outcome["episode_reward_sum"] == 13.0


def test_wandb_bar_chart_uses_only_completed_checkpoints() -> None:
    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    class Plot:
        @staticmethod
        def bar(table, x, y, *, title):
            return {"table": table, "x": x, "y": y, "title": title}

    class Run:
        def __init__(self):
            self.rows = []

        def log(self, row):
            self.rows.append(row)

    logger = _WandbLogger(False, "unused", {})
    logger.wandb = SimpleNamespace(Table=Table, Image=lambda figure: "bar-chart")
    logger.run = Run()
    logger.log_checkpoint_win_rate_bar_chart(
        {
            "ep000200": {
                "episode": 200, "total_wins": 40, "total_win_rate": 0.74,
                "completed_games": 54, "scheduled_games": 54,
            },
            "ep000250": {
                "episode": 250, "total_wins": 10, "total_win_rate": 0.5,
                "completed_games": 20, "scheduled_games": 54,
            },
        }
    )

    assert logger.run.rows[0]["checkpoint_win_rate_bar_chart"] == "bar-chart"
    assert logger.run.rows[0]["checkpoint_win_rate_table"].data == [["ep000200", 200, 40, 0.74]]


def test_local_bar_chart_reads_completed_results(tmp_path, monkeypatch) -> None:
    results_path = tmp_path / "evaluation.json"
    results_path.write_text(
        json.dumps(
            {
                "checkpoints": {
                    "ep000200": {
                        "episode": 200, "total_win_rate": 0.74,
                        "completed_games": 54, "scheduled_games": 54,
                    },
                    "ep000250": {
                        "episode": 250, "total_win_rate": 0.5,
                        "completed_games": 10, "scheduled_games": 54,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    shown = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(True))

    output = show_checkpoint_win_rate_chart(results_path)

    assert shown == [True]
    assert output == tmp_path / "evaluation_win_rate.png"
    assert output.is_file()
