"""Deterministic evaluation for `GNN_DQN_Agent` — see `Docs/Eval.md`.

`Evaluator.evaluate(...)` plays a fixed set of seeded, deterministic eval
games and returns aggregated metrics; `Evaluator.maybe_save_best(...)` keeps
the top-`keep_best` policy-only checkpoints (by eval score) in `best_dir`,
tracked through a `manifest.json` rather than by parsing filenames.

The rollout loop below is adapted from `Trainer.train()`
(`risk/learning/trainer.py`), not from `SelfPlay` — `SelfPlay.play_headless`
calls `env.step(action)` with no `reward_player`, which makes
`Environment.step` skip reward computation entirely, so it cannot produce
the reward metrics eval needs. Unlike `Trainer`, this loop never calls
`agent.remember(...)`/`agent.learn(...)` — eval does not feed the replay
buffer or train the net.
"""
from __future__ import annotations

import json
from pathlib import Path

from risk.app.factory import GameFactory
from risk.game.actions import FortifyAction
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.ui.input.init_screen import InitScreenState

# Two fixed eval suites (Docs/Eval.md "Eval games and opponents"): a small
# tactical game and a mixed full game, each played across the same fixed
# seeds every time so scores are comparable across evaluation calls.
# `killbot` is in both suites deliberately, so it's always part of eval
# rather than something that shows up only if randomly sampled.
_EVAL_SUITES: tuple[tuple[str, ...], ...] = (
    ("raider", "sentinel", "killbot"),
    ("random", "raider", "sentinel", "empire", "killbot"),
)
_EVAL_SEEDS: tuple[int, ...] = (0, 1, 2)
_LEARNER_SEAT = 0


class Evaluator:
    """Runs fixed deterministic eval suites and tracks the best-N policies."""

    def __init__(self, *, max_steps: int, keep_best: int, best_dir: str | Path) -> None:
        self.max_steps = max_steps
        self.keep_best = keep_best
        self.best_dir = Path(best_dir)

    def evaluate(self, agent: GNN_DQN_Agent, *, episode: int) -> dict:
        """Play every fixed eval game with a deterministic policy and
        aggregate the results. Restores `agent`'s `epsilon`/`train_mode`
        afterward regardless of outcome."""
        old_epsilon = agent.epsilon
        old_train_mode = agent.train_mode
        agent.epsilon = 0.0
        agent.set_train_mode(False)
        try:
            game_metrics = [
                self._play_one(agent, opponent_kinds, seed)
                for opponent_kinds in _EVAL_SUITES
                for seed in _EVAL_SEEDS
            ]
        finally:
            agent.epsilon = old_epsilon
            agent.set_train_mode(old_train_mode)

        return self._aggregate(game_metrics, episode=episode)

    def maybe_save_best(self, agent: GNN_DQN_Agent, eval_result: dict) -> bool:
        """Save `agent`'s current policy params if its score earns a spot
        among the top `keep_best` saved so far. Returns whether it saved."""
        episode = eval_result["episode"]
        score = eval_result["eval_score"]

        self.best_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.best_dir / "manifest.json"
        manifest = self._load_manifest(manifest_path)

        if len(manifest) >= self.keep_best and score <= min(e["score"] for e in manifest):
            return False

        filename = f"best_ep{episode:06d}_score{score:07.2f}.pt"
        agent.save_params(self.best_dir / filename)
        manifest.append({"path": filename, "episode": episode, "score": score})
        manifest.sort(key=lambda e: e["score"], reverse=True)

        while len(manifest) > self.keep_best:
            removed = manifest.pop()
            (self.best_dir / removed["path"]).unlink(missing_ok=True)

        self._write_manifest(manifest_path, manifest)
        return True

    # --- one eval game ----------------------------------------------------

    def _play_one(self, agent: GNN_DQN_Agent, opponent_kinds: tuple[str, ...], seed: int) -> dict:
        ctx = self._build_eval_context(opponent_kinds, seed)
        env, agents = ctx.env, ctx.agents
        seat = _LEARNER_SEAT
        agent.attach(seat, env)
        agents[seat] = agent

        step_count = 0
        agent_turns = 0
        episode_reward = 0.0
        territories_conquered = 0

        # Other seats may act before the learner's own seat on turn 1.
        while (
            step_count < self.max_steps
            and env.current_state().current_player_index != seat
        ):
            current_state = env.current_state()
            action = agents[current_state.current_player_index]((), current_state)
            env.step(action, reward_player=seat)
            step_count += 1

        while step_count < self.max_steps:
            if seat in env.current_state().eliminated:
                break

            current_state = env.current_state()
            state_before = current_state.snapshot()
            action = agents[seat]((), current_state)
            result = env.step(action, reward_player=seat)
            reward_total = result.reward
            step_count += 1
            agent_turns += 1

            while (
                step_count < self.max_steps
                and not result.done
                and seat not in result.state.eliminated
                and env.current_state().current_player_index != seat
            ):
                current_state = env.current_state()
                action_other = agents[current_state.current_player_index]((), current_state)
                result = env.step(action_other, reward_player=seat)
                reward_total += result.reward
                step_count += 1

            next_state = result.state
            done = result.done or seat in result.state.eliminated

            if isinstance(action, FortifyAction):
                reward_total += env.reward.end_of_turn(state_before, next_state, seat)

            territories_conquered += sum(
                1
                for before_owner, after_owner in zip(state_before.owners, next_state.owners)
                if before_owner != seat and after_owner == seat
            )

            episode_reward += reward_total
            if done:
                break

        win = int(env.winner() == seat)
        return {
            "win": win,
            "territories_conquered": territories_conquered,
            "episode_reward_sum": episode_reward,
            "agent_turns_survived": agent_turns,
            "reward_per_agent_turn": episode_reward / max(agent_turns, 1),
        }

    def _build_eval_context(self, opponent_kinds: tuple[str, ...], seed: int):
        state = InitScreenState()
        state.set_player_count(1 + len(opponent_kinds))
        state.set_agent_kind(_LEARNER_SEAT, "ai")  # placeholder; overridden by attach() above
        for i, kind in enumerate(opponent_kinds, start=1):
            state.set_agent_kind(i, kind)
        settings = state.build_settings(seed=seed)
        return GameFactory.build(settings)

    # --- aggregation / scoring ---------------------------------------------

    def _aggregate(self, game_metrics: list[dict], *, episode: int) -> dict:
        n = len(game_metrics)
        metrics = {
            "episode": episode,
            "eval_games": n,
            "eval_win_rate": sum(m["win"] for m in game_metrics) / n,
            "eval_avg_territories_conquered": (
                sum(m["territories_conquered"] for m in game_metrics) / n
            ),
            "eval_avg_reward_per_agent_turn": (
                sum(m["reward_per_agent_turn"] for m in game_metrics) / n
            ),
            "eval_avg_agent_turns_survived": (
                sum(m["agent_turns_survived"] for m in game_metrics) / n
            ),
        }
        metrics["eval_score"] = self._score(metrics)
        return metrics

    def _score(self, metrics: dict) -> float:
        return (
            100.0 * metrics["eval_win_rate"]
            + 2.0 * metrics["eval_avg_territories_conquered"]
            + 5.0 * metrics["eval_avg_reward_per_agent_turn"]
        )

    # --- manifest -----------------------------------------------------------

    def _load_manifest(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _write_manifest(self, path: Path, manifest: list[dict]) -> None:
        path.write_text(json.dumps(manifest, indent=2))


__all__ = ["Evaluator"]
