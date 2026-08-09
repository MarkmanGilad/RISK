"""Evaluate saved policies without changing the training evaluator.

``CheckpointEvaluator`` runs a fixed, comparable heuristic suite over every
checkpoint in one training run. ``AgentMatchEvaluator`` is the smaller custom
tool for a caller-chosen mix of saved policies and heuristic agents.  Both
write their raw games to JSON, so model selection stays an explicit later
decision instead of being hidden in the evaluator.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Support VS Code's Run button and direct execution by file path.  Module
# execution already puts the repository root on sys.path.
if __package__ in (None, ""):
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import torch

from risk.app.factory import GameFactory
from risk.game.actions import FortifyAction
from risk.learning.train_constants import CHECKPOINT_DIR, EVAL_MAX_STEPS
from risk.learning.trainer import build_learner_agent
from risk.ui.input.init_screen import InitScreenState


_FORMAT_VERSION = 1
_DEFAULT_SEEDS = (0, 1, 2)
_CHECKPOINT_SUITES = {
    3: ("raider", "killbot"),
    4: ("raider", "sentinel", "killbot"),
    5: ("random", "raider", "sentinel", "killbot"),
    6: ("random", "raider", "sentinel", "empire", "killbot"),
}
_HEURISTIC_KINDS = frozenset({"random", "raider", "sentinel", "empire", "killbot"})
_CHECKPOINT_NAME = re.compile(r"^ep(\d+)$")
_WANDB_EVALUATION_PROJECT = "Risk-Model-Evaluation"
_JSON_REPLACE_ATTEMPTS = 10
_JSON_REPLACE_RETRY_SECONDS = 0.25


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace ``path``, tolerating short-lived Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(_JSON_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == _JSON_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_JSON_REPLACE_RETRY_SECONDS)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_context(agent_kinds: Iterable[str], seed: int):
    state = InitScreenState()
    kinds = tuple(agent_kinds)
    state.set_player_count(len(kinds))
    for seat, kind in enumerate(kinds):
        state.set_agent_kind(seat, kind)
    return GameFactory.build(state.build_settings(seed=seed))


def _new_learned_agent(agent_kind: str, policy_state: dict[str, Any], ctx, seat: int):
    """Build a fresh policy and intentionally restore its online net only."""
    agent = build_learner_agent(agent_kind, ctx)
    agent.net.load_state_dict(policy_state)
    agent.attach(seat, ctx.env)
    agent.epsilon = 0.0
    agent.set_train_mode(False)
    return agent


def _read_policy_state(checkpoint: Path) -> dict[str, Any]:
    """Read only an online policy state dictionary from trusted local storage."""
    if checkpoint.is_dir():
        model_path = checkpoint / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"checkpoint has no model.pt: {checkpoint}")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        try:
            return payload["net"]
        except KeyError as exc:
            raise ValueError(f"checkpoint has no online net payload: {model_path}") from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"policy file does not exist: {checkpoint}")
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def _cached_policy_state(cache: dict[Path, dict[str, Any]], checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    if path not in cache:
        cache[path] = _read_policy_state(path)
    return cache[path]


def _participant_metrics(state, seats: Iterable[int]) -> dict[int, dict[str, int]]:
    return {
        seat: {
            "final_territory_count": sum(owner == seat for owner in state.owners),
            "final_army_count": sum(
                armies for owner, armies in zip(state.owners, state.armies) if owner == seat
            ),
        }
        for seat in seats
    }


def _play_game(
    ctx,
    agents: list,
    *,
    tracked_seats: Iterable[int],
    reward_seat: Optional[int],
    max_steps: int,
    progress_label: Optional[str] = None,
    progress_every_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Run one fresh headless game and return objective facts for tracked seats."""
    env = ctx.env
    tracked = tuple(tracked_seats)
    conquered = {seat: 0 for seat in tracked}
    turns = {seat: 0 for seat in tracked}
    reward_total = 0.0
    learner_turn_start = None
    learner_last_action = None
    step_count = 0
    done = False

    while step_count < max_steps and not env.is_terminal():
        before = env.current_state().snapshot()
        active_seat = before.current_player_index
        if active_seat in turns:
            turns[active_seat] += 1
        if reward_seat == active_seat and learner_turn_start is None:
            learner_turn_start = before

        action = agents[active_seat]((), before)
        result = env.step(action, reward_player=reward_seat)
        step_count += 1
        if (
            progress_label is not None
            and progress_every_steps is not None
            and step_count % progress_every_steps == 0
        ):
            print(f"  {progress_label}: step {step_count}/{max_steps}", flush=True)
        after = result.state
        done = result.done
        if reward_seat is not None:
            reward_total += result.reward
            if active_seat == reward_seat:
                learner_last_action = action

        for before_owner, after_owner in zip(before.owners, after.owners):
            if after_owner in conquered and before_owner != after_owner:
                conquered[after_owner] += 1

        learner_finished_turn = (
            reward_seat is not None
            and learner_turn_start is not None
            and (after.current_player_index == reward_seat or done or reward_seat in after.eliminated)
            and active_seat != reward_seat
        )
        learner_terminal_action = (
            reward_seat is not None
            and active_seat == reward_seat
            and (done or reward_seat in after.eliminated)
        )
        if learner_finished_turn or learner_terminal_action:
            if isinstance(learner_last_action, FortifyAction) or done or reward_seat in after.eliminated:
                reward_total += env.reward.end_of_turn(learner_turn_start, after, reward_seat)
            learner_turn_start = None
            learner_last_action = None

    state = env.current_state()
    winner = env.winner()
    metrics = _participant_metrics(state, tracked)
    for seat in tracked:
        metrics[seat].update(
            {
                "win": int(winner == seat),
                "territories_conquered": conquered[seat],
                "agent_turns_survived": turns[seat],
            }
        )
    return {
        "winner_seat": winner,
        "reached_max_steps": not env.is_terminal() and step_count >= max_steps,
        "step_count": step_count,
        "metrics": metrics,
        "episode_reward_sum": reward_total,
    }


def _validate_existing(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(f"existing result file has incompatible {key!r}")


def _checkpoint_game_key(game: dict[str, Any]) -> tuple[int, int, int]:
    return game["player_count"], game["learner_seat"], game["seed"]


def _matplotlib_pyplot():
    """Return a renderer that works even when this Python venv has no Tk GUI."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def show_checkpoint_win_rate_chart(
    results_path: str | Path, output_path: Optional[str | Path] = None
) -> Path:
    """Save and open a local Matplotlib bar chart from a checkpoint evaluation JSON."""
    results_path = Path(results_path)
    results = _read_json(results_path)
    checkpoints = results.get("checkpoints", {})
    rows = [
        (name, entry["total_win_rate"])
        for name, entry in sorted(checkpoints.items(), key=lambda item: item[1]["episode"])
        if entry["completed_games"] == entry["scheduled_games"]
    ]
    if not rows:
        raise ValueError("the result file has no fully completed checkpoints to plot")

    plt = _matplotlib_pyplot()

    names, win_rates = zip(*rows)
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.9), 5))
    bars = axis.bar(names, win_rates, color="#4f86e8")
    axis.set_title("Checkpoint win rate")
    axis.set_xlabel("Checkpoint / agent")
    axis.set_ylabel("Win rate")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=45)
    for bar, win_rate in zip(bars, win_rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            win_rate + 0.02,
            f"{win_rate:.1%}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    output = (
        Path(output_path)
        if output_path is not None
        else results_path.with_name(f"{results_path.stem}_win_rate.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


class _WandbLogger:
    """Small optional adapter; importing the evaluator never requires W&B."""

    def __init__(self, enabled: bool, name: str, config: dict[str, Any]) -> None:
        self.run = None
        self.wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging was requested but wandb is not installed") from exc
        self.wandb = wandb
        self.run = wandb.init(
            project=_WANDB_EVALUATION_PROJECT,
            name=name,
            job_type="evaluation",
            config=config,
        )

    def log(self, row: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(row)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def log_checkpoint_win_rate_bar_chart(
        self, checkpoints: dict[str, dict[str, Any]]
    ) -> None:
        """Log a real bar-chart image and its source table for completed checkpoints."""
        if self.run is None or self.wandb is None:
            return
        rows = [
            [name, entry["episode"], entry["total_wins"], entry["total_win_rate"]]
            for name, entry in sorted(checkpoints.items(), key=lambda item: item[1]["episode"])
            if entry["completed_games"] == entry["scheduled_games"]
        ]
        if not rows:
            return
        table = self.wandb.Table(
            columns=["agent", "episode", "total_wins", "total_win_rate"], data=rows
        )
        plt = _matplotlib_pyplot()

        names = [row[0] for row in rows]
        win_rates = [row[3] for row in rows]
        figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.9), 5))
        bars = axis.bar(names, win_rates, color="#4f86e8")
        axis.set_title("Checkpoint win rate")
        axis.set_xlabel("Checkpoint / agent")
        axis.set_ylabel("Win rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=45)
        for bar, win_rate in zip(bars, win_rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                win_rate + 0.02,
                f"{win_rate:.1%}",
                ha="center",
                va="bottom",
            )
        figure.tight_layout()
        self.run.log(
            {
                "checkpoint_win_rate_table": table,
                "checkpoint_win_rate_bar_chart": self.wandb.Image(figure),
            }
        )
        plt.close(figure)


class CheckpointEvaluator:
    """Evaluate every selected checkpoint against the fixed heuristic suites."""

    def __init__(
        self,
        *,
        max_steps: int = EVAL_MAX_STEPS,
        seeds: Iterable[int] = _DEFAULT_SEEDS,
        use_wandb: bool = False,
        progress_every_steps: int = 200,
    ) -> None:
        self.max_steps = int(max_steps)
        self.seeds = tuple(seeds)
        self.use_wandb = use_wandb
        self.progress_every_steps = int(progress_every_steps)
        self._policy_states: dict[Path, dict[str, Any]] = {}
        if self.max_steps <= 0 or not self.seeds or self.progress_every_steps <= 0:
            raise ValueError("max_steps and progress_every_steps must be positive, and seeds cannot be empty")

    def evaluate_run(
        self,
        agent_kind: str,
        run_id: int,
        *,
        min_episode: Optional[int] = None,
        max_episode: Optional[int] = None,
        output_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        self._validate_range(min_episode, max_episode)
        run_dir = Path(CHECKPOINT_DIR) / f"{agent_kind}_{run_id:03d}"
        checkpoints = self._find_checkpoints(run_dir, min_episode, max_episode)
        if not checkpoints:
            raise ValueError(f"no checkpoints found in requested range under {run_dir}")
        output = Path(output_path) if output_path is not None else self._default_output(run_dir, checkpoints)
        expected = self._metadata(agent_kind, run_id)
        results = self._open_results(output, expected)
        print(
            f"Checkpoint evaluation: {len(checkpoints)} checkpoints, "
            f"{len(checkpoints) * sum(players * len(self.seeds) for players in _CHECKPOINT_SUITES)} scheduled games.",
            flush=True,
        )
        print(f"Results: {output}", flush=True)
        logger = _WandbLogger(self.use_wandb, f"CheckpointEval_{agent_kind}_{run_id:03d}", expected)
        try:
            for checkpoint_number, (episode, checkpoint) in enumerate(checkpoints, start=1):
                entry = results["checkpoints"].setdefault(
                    checkpoint.name,
                    self._new_checkpoint_entry(episode, checkpoint),
                )
                self._evaluate_checkpoint(
                    entry,
                    checkpoint,
                    agent_kind,
                    results,
                    output,
                    logger,
                    checkpoint_number,
                    len(checkpoints),
                )
        finally:
            self._policy_states.clear()
            logger.finish()
        return results

    def _validate_range(self, minimum: Optional[int], maximum: Optional[int]) -> None:
        if minimum is not None and minimum < 0 or maximum is not None and maximum < 0:
            raise ValueError("episode bounds cannot be negative")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("min_episode cannot exceed max_episode")

    def _find_checkpoints(self, run_dir: Path, minimum: Optional[int], maximum: Optional[int]) -> list[tuple[int, Path]]:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"checkpoint run directory does not exist: {run_dir}")
        found = []
        for path in run_dir.iterdir():
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if path.is_dir() and match:
                episode = int(match.group(1))
                if (minimum is None or episode >= minimum) and (maximum is None or episode <= maximum):
                    found.append((episode, path))
        return sorted(found, key=lambda item: item[0])

    def _default_output(self, run_dir: Path, checkpoints: list[tuple[int, Path]]) -> Path:
        return run_dir / "evaluations" / (
            f"checkpoint_eval_ep{checkpoints[0][0]:06d}_to_{checkpoints[-1][0]:06d}.json"
        )

    def _metadata(self, agent_kind: str, run_id: int) -> dict[str, Any]:
        return {
            "format_version": _FORMAT_VERSION,
            "agent_kind": agent_kind,
            "run_id": run_id,
            "seeds": list(self.seeds),
            "max_steps": self.max_steps,
            "suites": {str(players): list(kinds) for players, kinds in _CHECKPOINT_SUITES.items()},
        }

    def _open_results(self, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
        if path.exists():
            results = _read_json(path)
            _validate_existing(results, expected)
            results.setdefault("checkpoints", {})
            return results
        return {**expected, "checkpoints": {}}

    def _new_checkpoint_entry(self, episode: int, path: Path) -> dict[str, Any]:
        return {
            "path": str(path), "episode": episode, "total_wins": 0,
            "scheduled_games": sum(players * len(self.seeds) for players in _CHECKPOINT_SUITES),
            "completed_games": 0, "total_win_rate": 0.0, "games": [],
        }

    def _evaluate_checkpoint(
        self,
        entry,
        checkpoint,
        agent_kind,
        results,
        output,
        logger,
        checkpoint_number,
        checkpoint_count,
    ) -> None:
        completed = {_checkpoint_game_key(game) for game in entry["games"]}
        policy_state = None
        print(
            f"[{checkpoint_number}/{checkpoint_count}] {checkpoint.name}: "
            f"{len(completed)}/{entry['scheduled_games']} games already complete.",
            flush=True,
        )
        try:
            for players, opponents in _CHECKPOINT_SUITES.items():
                for learner_seat in range(players):
                    for seed in self.seeds:
                        key = (players, learner_seat, seed)
                        if key in completed:
                            continue
                        game_number = len(entry["games"]) + 1
                        game_label = (
                            f"{checkpoint.name} game {game_number}/{entry['scheduled_games']} "
                            f"({players} players, learner seat {learner_seat}, seed {seed})"
                        )
                        print(f"  Starting {game_label}", flush=True)
                        if policy_state is None:
                            policy_state = _cached_policy_state(self._policy_states, checkpoint)
                        row = self._play_checkpoint_game(
                            agent_kind, policy_state, opponents, learner_seat, seed, game_label
                        )
                        entry["games"].append(row)
                        self._refresh_checkpoint_totals(entry)
                        _write_json(output, results)
                        print(
                            f"  Finished {game_label}: win={row['win']}, "
                            f"steps={row['step_count']}, total wins={entry['total_wins']}/{entry['completed_games']}",
                            flush=True,
                        )
                        logger.log({"row_type": "game", "checkpoint_episode": entry["episode"], **row})
            self._refresh_checkpoint_totals(entry)
            _write_json(output, results)
            logger.log_checkpoint_win_rate_bar_chart(results["checkpoints"])
            logger.log({"row_type": "checkpoint_summary", "checkpoint_episode": entry["episode"], "total_wins": entry["total_wins"], "completed_games": entry["completed_games"], "total_win_rate": entry["total_win_rate"]})
        finally:
            self._policy_states.pop(checkpoint.resolve(), None)

    def _play_checkpoint_game(
        self, agent_kind, policy_state, opponents, learner_seat, seed, game_label
    ) -> dict[str, Any]:
        kinds = list(opponents)
        kinds.insert(learner_seat, "ai")
        ctx = _build_context(kinds, seed)
        learner = _new_learned_agent(agent_kind, policy_state, ctx, learner_seat)
        ctx.agents[learner_seat] = learner
        outcome = _play_game(
            ctx,
            ctx.agents,
            tracked_seats=(learner_seat,),
            reward_seat=learner_seat,
            max_steps=self.max_steps,
            progress_label=game_label,
            progress_every_steps=self.progress_every_steps,
        )
        metric = outcome["metrics"][learner_seat]
        return {
            "player_count": len(kinds), "learner_seat": learner_seat, "seed": seed,
            "opponent_kinds": list(opponents), "win": metric["win"],
            "winner_kind": "learner" if outcome["winner_seat"] == learner_seat else "opponent" if outcome["winner_seat"] is not None else None,
            "winner_seat": outcome["winner_seat"], "reached_max_steps": outcome["reached_max_steps"],
            "final_territory_count": metric["final_territory_count"], "final_army_count": metric["final_army_count"],
            "territories_conquered": metric["territories_conquered"], "episode_reward_sum": outcome["episode_reward_sum"],
            "agent_turns_survived": metric["agent_turns_survived"],
            "reward_per_agent_turn": outcome["episode_reward_sum"] / max(metric["agent_turns_survived"], 1),
            "step_count": outcome["step_count"],
        }

    def _refresh_checkpoint_totals(self, entry: dict[str, Any]) -> None:
        entry["completed_games"] = len(entry["games"])
        entry["total_wins"] = sum(game["win"] for game in entry["games"])
        entry["total_win_rate"] = entry["total_wins"] / max(entry["completed_games"], 1)


class AgentMatchEvaluator:
    """Run caller-selected saved policies and heuristics through seat rotations."""

    def __init__(
        self,
        *,
        max_steps: int = EVAL_MAX_STEPS,
        seeds: Iterable[int] = _DEFAULT_SEEDS,
        use_wandb: bool = False,
        progress_every_steps: int = 200,
    ) -> None:
        self.max_steps = int(max_steps)
        self.seeds = tuple(seeds)
        self.use_wandb = use_wandb
        self.progress_every_steps = int(progress_every_steps)
        self._policy_states: dict[Path, dict[str, Any]] = {}
        if self.max_steps <= 0 or not self.seeds or self.progress_every_steps <= 0:
            raise ValueError("max_steps and progress_every_steps must be positive, and seeds cannot be empty")

    def evaluate(self, participants: Iterable[dict[str, Any]], *, output_path: str | Path) -> dict[str, Any]:
        specs = self._validate_participants(participants)
        path = Path(output_path)
        expected = {"format_version": _FORMAT_VERSION, "participants": specs, "seeds": list(self.seeds), "max_steps": self.max_steps}
        results = self._open_results(path, expected)
        print(f"Agent match: {len(specs) * len(self.seeds)} scheduled games. Results: {path}", flush=True)
        logger = _WandbLogger(self.use_wandb, f"AgentMatch_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}", expected)
        try:
            completed = {(game["seed"], game["rotation"]) for game in results["games"]}
            for seed in self.seeds:
                for rotation in range(len(specs)):
                    if (seed, rotation) in completed:
                        continue
                    print(f"  Starting match game seed {seed}, rotation {rotation}", flush=True)
                    row = self._play_match_game(specs, seed, rotation)
                    results["games"].append(row)
                    self._refresh_totals(results)
                    _write_json(path, results)
                    print(
                        f"  Finished match game seed {seed}, rotation {rotation}: "
                        f"winner={row['winner']}, steps={row['step_count']}",
                        flush=True,
                    )
                    logger.log({"row_type": "game", **row})
            self._refresh_totals(results)
            _write_json(path, results)
            for name, total in results["totals"].items():
                logger.log({"row_type": "participant_total", "participant": name, **total})
        finally:
            self._policy_states.clear()
            logger.finish()
        return results

    def _validate_participants(self, participants: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        specs = [dict(item) for item in participants]
        if not 3 <= len(specs) <= 6:
            raise ValueError("a match needs 3 to 6 participants")
        names = []
        for spec in specs:
            name, kind = spec.get("name"), spec.get("kind")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each participant needs a non-empty name")
            if kind == "heuristic":
                if spec.get("agent_kind") not in _HEURISTIC_KINDS:
                    raise ValueError("heuristic participants need a supported agent_kind")
                spec.pop("checkpoint", None)
            elif kind == "checkpoint":
                if not isinstance(spec.get("agent_kind"), str) or not spec.get("checkpoint"):
                    raise ValueError("checkpoint participants need agent_kind and checkpoint")
                spec["checkpoint"] = str(spec["checkpoint"])
            else:
                raise ValueError("participant kind must be 'heuristic' or 'checkpoint'")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("participant names must be unique")
        return specs

    def _open_results(self, path, expected):
        if path.exists():
            results = _read_json(path)
            _validate_existing(results, expected)
            results.setdefault("games", [])
            results.setdefault("totals", {})
            return results
        return {**expected, "games": [], "totals": {}}

    def _play_match_game(self, specs, seed, rotation) -> dict[str, Any]:
        roster = specs[rotation:] + specs[:rotation]
        kinds = [spec["agent_kind"] if spec["kind"] == "heuristic" else "ai" for spec in roster]
        ctx = _build_context(kinds, seed)
        for seat, spec in enumerate(roster):
            if spec["kind"] == "checkpoint":
                policy_state = _cached_policy_state(self._policy_states, spec["checkpoint"])
                ctx.agents[seat] = _new_learned_agent(spec["agent_kind"], policy_state, ctx, seat)
        outcome = _play_game(
            ctx,
            ctx.agents,
            tracked_seats=range(len(roster)),
            reward_seat=None,
            max_steps=self.max_steps,
            progress_label=f"match seed {seed}, rotation {rotation}",
            progress_every_steps=self.progress_every_steps,
        )
        by_name = {}
        for seat, spec in enumerate(roster):
            by_name[spec["name"]] = outcome["metrics"][seat]
        return {
            "seed": seed, "rotation": rotation,
            "roster": [spec["name"] for spec in roster], "winner": roster[outcome["winner_seat"]]["name"] if outcome["winner_seat"] is not None else None,
            "winner_seat": outcome["winner_seat"], "reached_max_steps": outcome["reached_max_steps"],
            "step_count": outcome["step_count"], "participants": by_name,
        }

    def _refresh_totals(self, results: dict[str, Any]) -> None:
        totals = {spec["name"]: {"total_wins": 0, "completed_games": 0, "total_win_rate": 0.0} for spec in results["participants"]}
        for game in results["games"]:
            for name, metric in game["participants"].items():
                totals[name]["total_wins"] += metric["win"]
                totals[name]["completed_games"] += 1
        for total in totals.values():
            total["total_win_rate"] = total["total_wins"] / max(total["completed_games"], 1)
        results["totals"] = totals


def main() -> None:
    """Run one saved-policy evaluation; edit the constants below in the code."""
    AGENT_KIND = "DQN"
    RUN_ID = 103
    MIN_EPISODE = 4_000  # Set to None to include every earlier checkpoint.
    MAX_EPISODE = 4_950  # Set to None to include every later checkpoint.
    USE_WANDB = True
    PROGRESS_EVERY_STEPS = 2000  # Print a live update every this many actions.
    # None resumes the default result file. Set a new label to start a separate
    # evaluation, for example: "reward_0_1_v2".
    EVALUATION_NAME = None
    # Use this existing JSON to add the 5000--6150 results to the completed
    # 6200--6700 results. Set back to None to use the normal output naming.
    RESULTS_PATH = (
        "Checkpoints/DQN_103/evaluations/"
        "checkpoint_eval_ep006200_to_006700.json"
    )
    # Set this to an existing evaluation JSON to show its local bar chart and
    # exit without running W&B or any games.
    PLOT_RESULTS_PATH = None

    if PLOT_RESULTS_PATH is not None:
        saved_chart = show_checkpoint_win_rate_chart(PLOT_RESULTS_PATH)
        print(f"Saved local bar chart: {saved_chart}", flush=True)
        return

    output_path = Path(RESULTS_PATH) if RESULTS_PATH is not None else None
    if output_path is None and EVALUATION_NAME is not None:
        output_path = (
            Path(CHECKPOINT_DIR)
            / f"{AGENT_KIND}_{RUN_ID:03d}"
            / "evaluations"
            / f"checkpoint_eval_{EVALUATION_NAME}.json"
        )
    print(
        f"Starting checkpoint evaluation: {AGENT_KIND} run {RUN_ID:03d}, "
        f"episodes {MIN_EPISODE} to {MAX_EPISODE}.",
        flush=True,
    )
    if output_path is not None:
        print(f"New named evaluation results: {output_path}", flush=True)
    results = CheckpointEvaluator(
        use_wandb=USE_WANDB,
        progress_every_steps=PROGRESS_EVERY_STEPS,
    ).evaluate_run(
        AGENT_KIND,
        RUN_ID,
        min_episode=MIN_EPISODE,
        max_episode=MAX_EPISODE,
        output_path=output_path,
    )
    for name, checkpoint in results["checkpoints"].items():
        print(
            f"{name}: {checkpoint['total_wins']}/{checkpoint['completed_games']} wins "
            f"({checkpoint['total_win_rate']:.1%})"
        )

    # To run a custom model-versus-agent match instead, comment out the
    # CheckpointEvaluator block above and uncomment/edit this block:
    #
    # results = AgentMatchEvaluator(
    #     use_wandb=USE_WANDB,
    #     progress_every_steps=PROGRESS_EVERY_STEPS,
    # ).evaluate(
    #     [
    #         {
    #             "name": "dqn_103_ep6700",
    #             "kind": "checkpoint",
    #             "agent_kind": "DQN",
    #             "checkpoint": "Checkpoints/DQN_103/ep006700",
    #         },
    #         {"name": "raider", "kind": "heuristic", "agent_kind": "raider"},
    #         {"name": "sentinel", "kind": "heuristic", "agent_kind": "sentinel"},
    #     ],
    #     output_path="Checkpoints/matches/dqn_103_ep6700_vs_heuristics.json",
    # )
    # print(results["totals"])


__all__ = [
    "CheckpointEvaluator",
    "AgentMatchEvaluator",
    "show_checkpoint_win_rate_chart",
    "main",
]


if __name__ == "__main__":
    main()
