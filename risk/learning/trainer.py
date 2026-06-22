"""Self-play trainer for `GNN_DQN_Agent` (Net A) — `Docs/RL-Prep-Changes.md`.

Built on top of `SelfPlay`, with an explicit loop similar to the example
PPO/DQN trainers. The trainer keeps orchestration obvious (build episode,
rollout, ingest, train, checkpoint), while `GNN_DQN_Agent` owns replay
ingest and gradient-step scheduling logic. Every episode:

- picks a random player count (`min_players`..`max_players`) and a random
  seat for the learner, then builds a fresh all-`RandomAgent` roster
  (`GameFactory.build`/`SetupStage.default_settings`) and reassigns the
  learner onto one seat (`GNN_DQN_Agent.attach`) — no `HumanAgent` seats.
- plays to either a real game-over or the learner's own elimination
  (`stop_when_player_eliminated`), since once the learner is out there's
  nothing left for it to learn from that episode.
- sparse terminal reward and `done` assignment are applied in
    `GNN_DQN_Agent.ingest_episode` on the episode's final transition.
- a gradient step is taken every episode once the replay buffer holds at
    least `batch_size` transitions (`GNN_DQN_Agent.learn`).
- checkpointing remains in this trainer.

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
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.train_constants import (
    BATCH_SIZE,
    CHECKPOINT_AFTER,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    MAX_PLAYERS,
    MAX_STEPS_PER_EPISODE,
    MIN_PLAYERS,
    TRAIN_EPISODES,
    TRAIN_STEPS_PER_CALL,
)


class Trainer:
    """Runs self-play training episodes for one persistent `GNN_DQN_Agent`.

    The agent's `net`/`target_net`/`optimizer`/`replay_buffer` live for
    the whole `Trainer`'s lifetime; only its seat/`env` binding is rebuilt
    every episode (`GNN_DQN_Agent.attach`).
    """

    def __init__(self, run_id: int, *, agent: Optional[GNN_DQN_Agent] = None, **agent_kwargs) -> None:
        self.run_id = run_id
        self.checkpoint_dir = Path(CHECKPOINT_DIR) / f"run_{run_id:03d}"
        self._rng = random.SystemRandom()
        self.episode = 0

        if agent is None:
            ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
            agent = GNN_DQN_Agent(player_id=0, env=ctx.env, train_mode=True, **agent_kwargs)
        self.agent = agent

    def train(self, n_episodes: int) -> None:
        """Run the full training loop."""
        for _ in range(n_episodes):
            self.episode += 1

            ctx, seat, _ = self._build_episode_context()
            env, agents = ctx.env, ctx.agents
            step_count = 0
            agent_turns = 0

            # Other seats may act before the learner's own seat on turn 1 —
            # play those opening moves now so the loop below only ever has to
            # deal with the agent's own turn.
            while env.current_state().current_player_index != seat:
                current_state = env.current_state()
                action = agents[current_state.current_player_index]((), current_state)
                env.step(action, reward_player=seat)
                step_count += 1

            for _ in range(MAX_STEPS_PER_EPISODE):
                if seat in env.current_state().eliminated:
                    break

                current_state = env.current_state()
                state = current_state.snapshot()
                action = agents[seat]((), current_state)
                result = env.step(action, reward_player=seat)
                step_count += 1
                agent_turns += 1
                print(f"ep={self.episode}/{n_episodes} steps={step_count} agent_turns={agent_turns}", end="\r", flush=True)

                # Play until agent's turn again, the agent is eliminated, or the game ends
                while (not result.done and seat not in result.state.eliminated and env.current_state().current_player_index != seat):
                    current_state = env.current_state()
                    action_other = agents[current_state.current_player_index]((), current_state)
                    result = env.step(action_other, reward_player=seat)
                    step_count += 1
                    print(f"ep={self.episode}/{n_episodes} steps={step_count} agent_turns={agent_turns}", end="\r", flush=True)

                # Store transition: state before agent acted → next_state (after all agents played)
                next_state = result.state
                done = result.done or seat in result.state.eliminated
                self.agent.remember(state, action, result.reward, next_state, done)
                self.agent.learn(batch_size=BATCH_SIZE, n_steps=TRAIN_STEPS_PER_CALL)

                if done:
                    break

            self._checkpoint()
            if step_count > 0:
                print()

    def _build_episode_context(self):
        n_players = self._rng.randint(MIN_PLAYERS, MAX_PLAYERS)
        seat = self._rng.randrange(n_players)
        settings = SetupStage.default_settings(n=n_players, seed=None)
        ctx = GameFactory.build(settings)
        self.agent.attach(seat, ctx.env)
        ctx.agents[seat] = self.agent
        return ctx, seat, n_players

    def _checkpoint(self) -> None:
        if self.episode < CHECKPOINT_AFTER:
            return
        if (self.episode - CHECKPOINT_AFTER) % CHECKPOINT_EVERY != 0:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"gnn_dqn_ep{self.episode:06d}.pt"
        self.agent.save_params(path)
        print(f"saved checkpoint: {path}")


def main() -> None:
    """Training entry point — the only value you change per run is RUN_ID.

    Run it with: python -m risk.learning.trainer
    """
    RUN_ID = 1

    trainer = Trainer(RUN_ID)
    trainer.train(n_episodes=TRAIN_EPISODES)


__all__ = ["Trainer", "main"]


if __name__ == "__main__":
    main()
