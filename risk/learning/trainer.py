"""Self-play trainer for DQN-family agents.

Built on top of `SelfPlay`, with an explicit loop similar to the example
PPO/DQN trainers. The trainer keeps orchestration obvious (build episode,
rollout, ingest, train, checkpoint), while `GNN_DQN_Agent` owns replay
ingest and gradient-step scheduling logic. Every episode:

- picks a random player count (`min_players`..`max_players`) and a random
    seat for the learner, then randomly assigns every non-learner seat to one
    of the configured random/heuristic opponent agents (`random`, `raider`,
    `sentinel`, `empire`, `killbot`) — no `HumanAgent` seats.
- plays to either a real game-over or the learner's own elimination, since
  once the learner is out there's nothing left for it to learn from that
  episode.
- reward (sparse terminal plus dense per-phase shaping, `Docs/Reward.md`) is
  computed by `Environment.step(...)` via `RewardCalculator` and stored on
  every learner transition via `GNN_DQN_Agent.remember(...)`.
- a gradient step is taken after every learner turn once the replay buffer
  holds at least `batch_size` transitions (`GNN_DQN_Agent.learn`).
- checkpointing and W&B/local logging are owned by `TrainingLogger`, called
  from this trainer.

Because the learner is reassigned a different seat (and `n_players`) every
episode, `GraphAdapter`'s `perspective` parameter (`Docs/GraphAdapter.md`)
is what keeps the net seeing one consistent "slot 0 is me" frame regardless
of which physical seat it actually occupies that episode — `GNN_DQN_Agent`
already threads this through every adapter call, and `ReplayBuffer` stores
each transition's `perspective` so it stays correct on replay even after
the learner moves seats again.

    from risk.learning.trainer import Trainer
    trainer = Trainer(run_id, agent=agent)
    trainer.train(n_episodes=2000)

Run as a script with: python -m risk.learning.trainer
"""
from __future__ import annotations

import random
from collections import deque
import math
from pathlib import Path
from typing import Optional

# Support being launched as a plain script (VS Code green Run button / F5),
# same bootstrap as self_play.py.
if __package__ in (None, ""):
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from risk.app.factory import GameFactory
from risk.app.setup import SetupStage
from risk.agents.heuristic_agent import EmpireAgent, KillbotAgent, RaiderAgent, SentinelAgent
from risk.agents.random_agent import RandomAgent
from risk.game.actions import FortifyAction
from risk.learning.dueling_dqn_agent import Dueling_DQN_Agent
from risk.learning.evaluator import Evaluator
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.ppo_agent import PPO_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EVAL_EVERY_EPISODES,
    EVAL_KEEP_BEST,
    EVAL_MAX_STEPS,
    MAX_PLAYERS,
    MAX_STEPS_PER_EPISODE,
    MIN_PLAYERS,
    ROLLING_WIN_RATE_WINDOW,
    TRAIN_OPPONENT_AGENT_KINDS,
    TRAIN_EPISODES,
)
from risk.learning.training_logger import TrainingLogger


def _aggregate_update_metrics(
    rows: list[dict[str, float]],
    *,
    weight_key: str | None = None,
    unweighted_fields: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Aggregate every agent update in an episode instead of keeping the last."""
    aggregated: dict[str, float] = {}
    nonfinite_count = 0
    keys = {key for row in rows for key in row}
    for key in keys:
        values: list[tuple[float, float]] = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                nonfinite_count += 1
                continue
            weight = 1.0
            if weight_key is not None and key not in unweighted_fields:
                try:
                    weight = float(row[weight_key])
                except (KeyError, TypeError, ValueError):
                    weight = 1.0
                if not math.isfinite(weight):
                    nonfinite_count += 1
                    weight = 1.0
            values.append((value, max(weight, 0.0)))
        if not values:
            continue
        if key.endswith("_max"):
            aggregated[key] = max(value for value, _ in values)
            continue
        total_weight = sum(weight for _, weight in values)
        aggregated[key] = (
            sum(value * weight for value, weight in values) / total_weight
            if total_weight > 0 else sum(value for value, _ in values) / len(values)
        )
    aggregated["update_metrics_nonfinite_count"] = float(nonfinite_count)
    return aggregated


class Trainer:
    """Runs self-play training episodes for one persistent `GNN_DQN_Agent`.

    The agent's `net`/`target_net`/`optimizer`/`replay_buffer` live for
    the whole `Trainer`'s lifetime; only its seat/`env` binding is rebuilt
    every episode (`GNN_DQN_Agent.attach`).
    """

    def __init__(
        self,
        run_id: int,
        *,
        agent: GNN_DQN_Agent,
        evaluator: Optional[Evaluator] = None,
        logger: Optional[TrainingLogger] = None,
        checkpoint_dir: Optional[str | Path] = None,
        use_wandb: bool = True,
        wandb_run_id: Optional[str] = None,
        wandb_step_from_episode: bool = False,
        resume: bool = True,
        notes: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.run_name = f"{agent.label}_{run_id:03d}"
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None
            else Path(CHECKPOINT_DIR) / self.run_name
        )
        self._rng = random.SystemRandom()
        self.episode = 0
        self.cumulative_learner_turns = 0
        self._recent_wins: deque[int] = deque(maxlen=ROLLING_WIN_RATE_WINDOW)

        self.agent = agent

        self.logger = logger or TrainingLogger(
            run_id,
            self.checkpoint_dir,
            use_wandb=use_wandb,
            run_name=self.run_name,
            wandb_run_id=wandb_run_id,
            wandb_step_from_episode=wandb_step_from_episode,
            resume=resume,
            notes=notes,
        )
        self.evaluator = evaluator or Evaluator(
            max_steps=EVAL_MAX_STEPS,
            keep_best=EVAL_KEEP_BEST,
            best_dir=self.checkpoint_dir / "best",
        )
        self.logger.start_run(agent=self.agent, trainer=self)
        resumed = self.logger.try_resume(agent=self.agent)
        if resumed is not None:
            self.episode = resumed["episode"]
            print(f"resumed from episode {self.episode}")

    def train(self, n_episodes: int) -> None:
        """Run the full training loop."""
        for _ in range(n_episodes):
            self.episode += 1
            self.agent.on_episode_start(self.episode)

            ctx, seat, n_players, opponent_kinds = self._build_episode_context()
            env, agents = ctx.env, ctx.agents
            step_count = 0
            agent_turns = 0
            episode_reward = 0.0
            territories_conquered = 0
            losses: list[float] = []
            update_metric_rows: list[dict[str, float]] = []
            learner_update_calls = 0
            optimizer_steps_before = self._agent_optimizer_steps()
            samples_processed_before = self._agent_samples_processed()
            done = False
            reward_components: dict[str, float] = {}
            
            #play other until agent's turn
            while (step_count < MAX_STEPS_PER_EPISODE and env.current_state().current_player_index != seat):
                current_state = env.current_state()
                action = agents[current_state.current_player_index]((), current_state)
                env.step(action, reward_player=seat)
                self._accumulate_reward_components(reward_components, env.reward.last_components)
                step_count += 1

            while step_count < MAX_STEPS_PER_EPISODE:
                if seat in env.current_state().eliminated:
                    break

                current_state = env.current_state()
                state = current_state.snapshot()
                action = agents[seat]((), current_state)
                result = env.step(action, reward_player=seat)
                self._accumulate_reward_components(reward_components, env.reward.last_components)
                reward_total = result.reward
                step_count += 1
                agent_turns += 1
                self._print_progress_line(n_episodes=n_episodes, step_count=step_count, agent_turns=agent_turns, seat=seat, current_state=result.state, topology=env.topology, )

                # Play until agent's turn again, the agent is eliminated, or the game ends.
                while (step_count < MAX_STEPS_PER_EPISODE and not result.done and seat not in result.state.eliminated and env.current_state().current_player_index != seat ):
                    current_state = env.current_state()
                    action_other = agents[current_state.current_player_index]((), current_state)
                    result = env.step(action_other, reward_player=seat)
                    self._accumulate_reward_components(reward_components, env.reward.last_components)
                    reward_total += result.reward
                    step_count += 1
                    self._print_progress_line(n_episodes=n_episodes, step_count=step_count, agent_turns=agent_turns, seat=seat, current_state=result.state, topology=env.topology,)

                # Store transition: state before agent acted → next_state (after all agents played)
                next_state = result.state
                done = result.done or seat in result.state.eliminated

                if isinstance(action, FortifyAction) or done:
                    reward_total += env.reward.end_of_turn(state, next_state, seat)
                    self._accumulate_reward_components(reward_components, env.reward.last_end_of_turn_components)

                territories_conquered += sum(1 for before_owner, after_owner in zip(state.owners, next_state.owners) if before_owner != seat and after_owner == seat)
                episode_reward += reward_total

                self.agent.remember(state, action, reward_total, next_state, done)
                reached_max_steps = step_count >= MAX_STEPS_PER_EPISODE
                step_losses = self.agent.learn(reached_max_steps=reached_max_steps)

                losses.extend(step_losses)
                if step_losses:
                    learner_update_calls += 1
                    optional_metrics = getattr(self.agent, "last_update_metrics", {})
                    if isinstance(optional_metrics, dict):
                        update_metric_rows.append(dict(optional_metrics))
                if done:
                    break

            self.cumulative_learner_turns += agent_turns
            progress_metrics = getattr(self.agent, "progress_metrics", None)
            agent_progress = progress_metrics() if callable(progress_metrics) else {}
            optimizer_steps = self._agent_optimizer_steps()
            samples_processed = self._agent_samples_processed()
            update_metrics = _aggregate_update_metrics(update_metric_rows, weight_key=getattr(self.agent, "update_metric_weight_key", None), 
                                                       unweighted_fields=getattr(self.agent, "unweighted_update_metrics", frozenset()),)
            winner = env.winner()
            win = int(winner == seat)
            self._recent_wins.append(win)
            reinforce_action_count = reward_components.get("reinforce_action_count", 0.0)
            reinforce_partial_action_count = reward_components.get("reinforce_partial_action_count", 0.0)
            reinforce_per_action = (reward_components.get("reinforce", 0.0) / reinforce_action_count if reinforce_action_count else 0.0)
            reinforce_ready_per_action = (reward_components.get("reinforce_ready", 0.0) / reinforce_action_count if reinforce_action_count else 0.0)
            reinforce_total_per_action = (reward_components.get("reinforce_total", 0.0) / reinforce_action_count if reinforce_action_count else 0.0)
            metrics = {
                "win": win,
                f"win_rate_last_{ROLLING_WIN_RATE_WINDOW}": (sum(self._recent_wins) / len(self._recent_wins)),
                "reward_per_agent_turn": episode_reward / max(agent_turns, 1),
                "learn_loss_mean": (sum(losses) / len(losses)) if losses else 0.0,
                "territories_conquered": territories_conquered,
                "agent_turns_survived": agent_turns,
                "cumulative_learner_turns": self.cumulative_learner_turns,
                "learner_update_calls_in_episode": learner_update_calls,
                "optimizer_steps_in_episode": optimizer_steps - optimizer_steps_before,
                "samples_processed_in_episode": samples_processed - samples_processed_before,
                "cumulative_optimizer_steps": optimizer_steps,
                "cumulative_samples_processed": samples_processed,
                "reinforce_action_count": reinforce_action_count,
                "reinforce_partial_action_count": reinforce_partial_action_count,
                "reward_component_reinforce_per_action": reinforce_per_action,
                "reward_component_reinforce_ready_per_action": reinforce_ready_per_action,
                "reward_component_reinforce_total_per_action": reinforce_total_per_action,
                **self._opponent_metrics(
                    opponent_kinds=opponent_kinds,
                    winner=winner,
                    learner_seat=seat,
                    n_players=n_players,
                ),
                **{
                    f"reward_component_{name}": total
                    for name, total in reward_components.items()
                    if name not in {
                        "reinforce_action_count",
                        "reinforce_partial_action_count",
                    }
                },
                **update_metrics,
                **agent_progress,
            }
            if self.episode % EVAL_EVERY_EPISODES == 0:
                eval_result = self.evaluator.evaluate(self.agent, episode=self.episode)
                saved_best = self.evaluator.maybe_save_best(self.agent, eval_result)
                eval_result["eval_saved_best"] = int(saved_best)
                metrics.update(eval_result)
            self.logger.log_episode(episode=self.episode, metrics=metrics)
            self.logger.checkpoint(episode=self.episode, agent=self.agent)
            if step_count > 0:
                print()

    def _accumulate_reward_components(self, totals: dict[str, float], components: dict[str, float]) -> None:
        """Add one `RewardCalculator` call's breakdown into the running
        per-episode totals logged as `reward_component_*` (`Docs/Reward.md`
        "per-component logging") — diagnostic only, no effect on training."""
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + value

    def _agent_optimizer_steps(self) -> int:
        return int(getattr(self.agent, "optimizer_steps", getattr(self.agent, "train_steps", 0)))

    def _agent_samples_processed(self) -> int:
        exact = getattr(self.agent, "samples_processed", None)
        return int(exact if exact is not None else self._agent_optimizer_steps() * BATCH_SIZE)

    def _build_episode_context(self):
        n_players = self._rng.randint(MIN_PLAYERS, MAX_PLAYERS)
        seat = self._rng.randrange(n_players)
        settings = SetupStage.default_settings(n=n_players, seed=None)
        ctx = GameFactory.build(settings)
        opponent_kinds = self._assign_random_opponents(ctx, learner_seat=seat)
        self.agent.attach(seat, ctx.env)
        ctx.agents[seat] = self.agent
        return ctx, seat, n_players, opponent_kinds

    def _assign_random_opponents(self, ctx, *, learner_seat: int) -> dict[int, str]:
        """Assign opponents and return their stable per-seat kind labels."""
        opponent_kinds: dict[int, str] = {}
        for player_id in range(len(ctx.agents)):
            if player_id == learner_seat:
                continue
            kind = self._rng.choice(TRAIN_OPPONENT_AGENT_KINDS)
            opponent_kinds[player_id] = kind
            seed = self._rng.randrange(2**32)
            if kind == "random":
                ctx.agents[player_id] = RandomAgent(player_id=player_id, env=ctx.env, seed=seed)
            elif kind == "raider":
                ctx.agents[player_id] = RaiderAgent(player_id=player_id, env=ctx.env, seed=seed)
            elif kind == "sentinel":
                ctx.agents[player_id] = SentinelAgent(player_id=player_id, env=ctx.env, seed=seed)
            elif kind == "empire":
                ctx.agents[player_id] = EmpireAgent(player_id=player_id, env=ctx.env, seed=seed)
            elif kind == "killbot":
                ctx.agents[player_id] = KillbotAgent(player_id=player_id, env=ctx.env, seed=seed)
            else:
                raise ValueError(f"Unknown training opponent kind {kind!r}")
        return opponent_kinds

    def _opponent_metrics(
        self,
        *,
        opponent_kinds: dict[int, str],
        winner: int | None,
        learner_seat: int,
        n_players: int,
    ) -> dict[str, int | str]:
        """Build roster and winner fields for per-episode W&B diagnostics."""
        winner_kind = "learner" if winner == learner_seat else opponent_kinds.get(winner, "none")
        roster = {
            player_id: (
                f"learner:{self.agent.label}" if player_id == learner_seat else opponent_kinds[player_id]
            )
            for player_id in range(n_players)
        }
        metrics: dict[str, int | str] = {
            "player_count": n_players,
            "winner_kind": winner_kind,
            "winner_seat": -1 if winner is None else winner,
            "roster": ", ".join(f"p{player_id}={kind}" for player_id, kind in roster.items()),
        }
        for kind in TRAIN_OPPONENT_AGENT_KINDS:
            count = sum(assigned_kind == kind for assigned_kind in opponent_kinds.values())
            metrics[f"opponent_count_{kind}"] = count
            metrics[f"winner_is_{kind}"] = int(winner_kind == kind)
            if count:
                metrics[f"opponent_{kind}_won_when_present"] = int(winner_kind == kind)
        return metrics

    def _print_progress_line(self, *, n_episodes: int, step_count: int, agent_turns: int, seat: int, current_state, topology, ) -> None:
        status_line = self.logger.format_status_line(topology=topology, episode=self.episode, n_episodes=n_episodes,
            step_count=step_count, agent_turns=agent_turns, seat=seat, current_state=current_state, )
        print(f"\033[K{status_line}", end="\r", flush=True)

def build_learner_agent(agent_kind: str, ctx):
    """Construct the selected learner against the temporary sizing environment."""
    if agent_kind == "DQN":
        return GNN_DQN_Agent(player_id=0, env=ctx.env, train_mode=True)
    if agent_kind == "Dueling_DQN":
        return Dueling_DQN_Agent(player_id=0, env=ctx.env, train_mode=True)
    if agent_kind == "PPO":
        return PPO_Agent(player_id=0, env=ctx.env, train_mode=True)
    raise ValueError(f"Unknown learner agent kind {agent_kind!r}")

def main() -> None:
    """Training entry point — change RUN_ID and choose one agent block per run.

    Run it with: python -m risk.learning.trainer
    """

    RUN_ID = 313

    ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
    agent = build_learner_agent("Dueling_DQN", ctx)
    trainer = Trainer(
        RUN_ID,
        agent=agent,
        use_wandb=True,
        resume=True,
        wandb_run_id="ph8ihqze",
    )
    trainer.train(n_episodes=TRAIN_EPISODES)
    trainer.logger.finish()


__all__ = ["Trainer", "main"]


if __name__ == "__main__":
    main()
