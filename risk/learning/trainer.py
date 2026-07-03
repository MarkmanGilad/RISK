"""Self-play trainer for `GNN_DQN_Agent` (Net A) — `Docs/RL-Prep-Changes.md`.

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
- checkpointing and W&B/local logging are owned by `TrainingLogger`
  (`Docs/Training-Logging-Plan.md`), called from this trainer.

Because the learner is reassigned a different seat (and `n_players`) every
episode, `GraphAdapter`'s `perspective` parameter (`Docs/GraphAdapter.md`)
is what keeps the net seeing one consistent "slot 0 is me" frame regardless
of which physical seat it actually occupies that episode — `GNN_DQN_Agent`
already threads this through every adapter call, and `ReplayBuffer` stores
each transition's `perspective` so it stays correct on replay even after
the learner moves seats again.

    from risk.learning.trainer import Trainer
    trainer = Trainer(seed=0)
    trainer.train(n_episodes=2000)

Run as a script with: python -m risk.learning.trainer
"""
from __future__ import annotations

import random
from collections import deque
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
from risk.learning.evaluator import Evaluator
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EVAL_EVERY_EPISODES,
    EVAL_KEEP_BEST,
    EVAL_MAX_STEPS,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    MAX_PLAYERS,
    MAX_STEPS_PER_EPISODE,
    MIN_PLAYERS,
    ROLLING_WIN_RATE_WINDOW,
    TRAIN_OPPONENT_AGENT_KINDS,
    TRAIN_EPISODES,
    TRAIN_STEPS_PER_CALL,
)
from risk.learning.training_logger import TrainingLogger


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
        agent: Optional[GNN_DQN_Agent] = None,
        evaluator: Optional[Evaluator] = None,
        logger: Optional[TrainingLogger] = None,
        use_wandb: bool = True,
        resume: bool = True,
        notes: Optional[str] = None,
        **agent_kwargs,
    ) -> None:
        self.run_id = run_id
        self.checkpoint_dir = Path(CHECKPOINT_DIR) / f"run_{run_id:03d}"
        self._rng = random.SystemRandom()
        self.episode = 0
        self._recent_wins: deque[int] = deque(maxlen=ROLLING_WIN_RATE_WINDOW)

        if agent is None:
            ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
            agent_kwargs.setdefault("epsilon", EPSILON_START)
            agent = GNN_DQN_Agent(player_id=0, env=ctx.env, train_mode=True, **agent_kwargs)
        self.agent = agent

        self.logger = logger or TrainingLogger(
            run_id, self.checkpoint_dir, use_wandb=use_wandb, resume=resume, notes=notes
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
            self.agent.epsilon = self._epsilon_for_episode(self.episode)

            ctx, seat, _ = self._build_episode_context()
            env, agents = ctx.env, ctx.agents
            step_count = 0
            agent_turns = 0
            episode_reward = 0.0
            territories_conquered = 0
            losses: list[float] = []
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
                self._print_progress_line(n_episodes=n_episodes, step_count=step_count, agent_turns=agent_turns,
                    seat=seat, current_state=result.state, topology=env.topology, )

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
                step_losses = self.agent.learn(batch_size=BATCH_SIZE, n_steps=TRAIN_STEPS_PER_CALL)

                losses.extend(step_losses)
                if done:
                    break

            winner = env.winner()
            win = int(winner == seat)
            self._recent_wins.append(win)
            metrics = {
                "win": win,
                f"win_rate_last_{ROLLING_WIN_RATE_WINDOW}": (sum(self._recent_wins) / len(self._recent_wins)),
                "reward_per_agent_turn": episode_reward / max(agent_turns, 1),
                "learn_loss_mean": (sum(losses) / len(losses)) if losses else 0.0,
                "territories_conquered": territories_conquered,
                "agent_turns_survived": agent_turns,
                **{
                    f"reward_component_{name}": total
                    for name, total in reward_components.items()
                },
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

    def _epsilon_for_episode(self, episode: int) -> float:
        """Linear decay from `EPSILON_START` to `EPSILON_END` over `EPSILON_DECAY_EPISODES`."""
        progress = min(episode / EPSILON_DECAY_EPISODES, 1.0)
        return EPSILON_START + (EPSILON_END - EPSILON_START) * progress

    def _build_episode_context(self):
        n_players = self._rng.randint(MIN_PLAYERS, MAX_PLAYERS)
        seat = self._rng.randrange(n_players)
        settings = SetupStage.default_settings(n=n_players, seed=None)
        ctx = GameFactory.build(settings)
        self._assign_random_opponents(ctx, learner_seat=seat)
        self.agent.attach(seat, ctx.env)
        ctx.agents[seat] = self.agent
        return ctx, seat, n_players

    def _assign_random_opponents(self, ctx, *, learner_seat: int) -> None:
        for player_id in range(len(ctx.agents)):
            if player_id == learner_seat:
                continue
            kind = self._rng.choice(TRAIN_OPPONENT_AGENT_KINDS)
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

    def _print_progress_line(self, *, n_episodes: int, step_count: int, agent_turns: int, seat: int, current_state, topology, ) -> None:
        status_line = self.logger.format_status_line(topology=topology, episode=self.episode, n_episodes=n_episodes,
            step_count=step_count, agent_turns=agent_turns, seat=seat, current_state=current_state, )
        print(f"\033[K{status_line}", end="\r", flush=True)


def main() -> None:
    """Training entry point — the only value you change per run is RUN_ID.

    Run it with: python -m risk.learning.trainer
    """
    RUN_ID = 22

    trainer = Trainer(RUN_ID)
    trainer.train(n_episodes=TRAIN_EPISODES)
    trainer.logger.finish()


__all__ = ["Trainer", "main"]


if __name__ == "__main__":
    main()
