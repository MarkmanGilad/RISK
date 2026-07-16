# Trainer

`risk/learning/trainer.py` owns the high-level self-play training loop. It does
not own the learning algorithm itself: the persistent agent chooses actions,
stores replay transitions, runs gradient steps, and saves/loads its own model
state. The trainer's job is orchestration: build each game, attach the learner,
roll the environment forward, collect rewards and metrics, call evaluation, and
delegate logging/checkpointing.

## Current entry point

Run training with:

```bash
python -m risk.learning.trainer
```

`main()` is intentionally the place where the learner is selected. It creates
the agent against a temporary sizing environment, then passes it into
`Trainer`:

```python
ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
agent = build_learner_agent("DQN", ctx)

trainer = Trainer(RUN_ID, agent=agent)
trainer.train(n_episodes=TRAIN_EPISODES)
trainer.logger.finish()
```

`build_learner_agent` currently accepts `"DQN"`, `"Dueling_DQN"`, and `"PPO"`; an
unknown label raises `ValueError` immediately rather than failing later with
an unbound local variable.

There is no hidden default agent inside `Trainer.__init__`. This keeps the
trainer reusable for `GNN_DQN_Agent`, `Dueling_DQN_Agent`, and future agents
such as PPO or PDQN without silently changing behavior.

## Constructor responsibilities

`Trainer.__init__(run_id, *, agent, evaluator=None, logger=None,
checkpoint_dir=None, use_wandb=True, resume=True, notes=None)` sets up one
training run.

- `agent` is required and must already be constructed by the caller.
- `checkpoint_dir` defaults to `Checkpoints/<agent.label>_<run_id>` when not
  provided, for example `Checkpoints/DQN_030` or
  `Checkpoints/Dueling_DQN_030`. `label` is a plain class attribute each agent
  class declares itself (`GNN_DQN_Agent.label = "DQN"`,
  `Dueling_DQN_Agent.label = "Dueling_DQN"`) — `Trainer` just reads
  `agent.label`, it does not infer it from the class name or an isinstance
  check.
- `logger` defaults to `TrainingLogger`, which owns W&B init/log/finish and
  regular resume checkpoints.
- `evaluator` defaults to `Evaluator`, which periodically evaluates the current
  policy and saves best-policy checkpoints under `<checkpoint_dir>/best`.
- `resume=True` lets `TrainingLogger.try_resume(...)` restore the agent and
  trainer episode counter from the run checkpoint if one exists.

The trainer keeps one persistent agent object for the whole run. Each episode
creates a new environment and seat assignment, then calls `agent.attach(seat,
env)` so the same model/replay buffer continues training in the new game.

## Episode setup

For every episode, `_build_episode_context()`:

- samples a player count from `MIN_PLAYERS..MAX_PLAYERS`,
- samples the learner's physical seat,
- builds a fresh game with `SetupStage.default_settings(..., seed=None)`,
- replaces every non-learner seat with a random opponent kind from
  `TRAIN_OPPONENT_AGENT_KINDS`,
- attaches the persistent learner to the fresh environment,
- inserts the learner into `ctx.agents[seat]`.

The learner changes physical seat and player count across episodes. The agent's
graph adapter uses perspective-relative encoding so the model still sees a
stable "me versus others" frame; replay transitions store the perspective they
were collected under.

## Training loop

`Trainer.train(n_episodes)` is the only public training method. For each
episode it:

1. Increments `self.episode` and calls `agent.on_episode_start(self.episode)`
   — a no-op-by-default `BaseAgent` hook, the same pattern as the existing
   `on_turn_start`/`on_turn_end` hooks. `GNN_DQN_Agent`/`Dueling_DQN_Agent`
   override it to recompute their own `epsilon` from the linear decay
   schedule. Episode 1 uses `EPSILON_START`; it reaches `EPSILON_END` at
   episode `EPSILON_DECAY_EPISODES + 1`. `Trainer` itself no longer knows
   what epsilon is.
2. Builds a fresh episode context and advances opponents until the learner's
   first turn.
3. Repeats learner turns until game-over, learner elimination, or
   `MAX_STEPS_PER_EPISODE`.
4. On each learner turn, snapshots the state before the action, asks the agent
   for an action, steps the environment with `reward_player=seat`, then plays
   opponents until control returns to the learner or the game ends.
5. Sums all reward from the learner-action span, including reward triggered by
   opponent moves while `reward_player` is still the learner.
6. Adds `RewardCalculator.end_of_turn(...)` when the learner fortifies or the
   episode ends, because territory/army/continent delta rewards are computed at
   that boundary.
7. Stores exactly one transition for the learner turn with
   `agent.remember(state, action, reward_total, next_state, done)`.
8. Calls `agent.learn(reached_max_steps=...)`, where the flag reports whether
   this learner transition reached `MAX_STEPS_PER_EPISODE`; `done` continues to
   represent only a real terminal state.

The trainer does not inspect replay-buffer internals or optimizer internals.
Those are agent-owned responsibilities.

## Metrics

At the end of each episode, the trainer logs one metrics row through
`TrainingLogger.log_episode(...)`:

- `win`
- `win_rate_last_<ROLLING_WIN_RATE_WINDOW>`
- `reward_per_agent_turn`
- `learn_loss_mean`
- `territories_conquered`
- `agent_turns_survived`
- `cumulative_learner_turns` â€” running learner-turn budget for comparing
  sample efficiency across agents in one uninterrupted training run
- `learner_update_calls_in_episode` and `optimizer_steps_in_episode` â€” the
  agent-level learn calls and actual optimizer minibatches executed, which
  differ for PPO because one rollout update may contain many minibatches
- `samples_processed_in_episode`, `cumulative_optimizer_steps`, and
  `cumulative_samples_processed` â€” common compute/sample-presentation axes;
  PPO supplies exact counters while fixed-batch DQN derives samples from its
  optimizer-step count
- `reward_component_*` totals from `RewardCalculator.last_components` and
  `last_end_of_turn_components`

An agent may optionally provide `last_update_metrics` after a non-empty
`learn()` result and a `progress_metrics()` mapping. The trainer collects
**every** update mapping in the episode rather than retaining only the last:
ordinary fields are averaged and fields ending in `_max` keep their maximum.
Agents may nominate a per-update weight (PPO uses executed minibatches) so
minibatch-derived means are not biased toward short, early-stopped updates;
rollout-level fields remain equally weighted. A non-finite diagnostic count is
logged even though invalid values are excluded from numeric aggregates.
It merges the result generically, without checking an agent type. PPO uses
this for KL, loss/return/value diagnostics, entropy/action-set statistics,
gradient norms, rollout fill, and rollout-update count. `GNN_DQN_Agent`/
`Dueling_DQN_Agent` supply the DQN-family equivalent: `progress_metrics()`
reports `epsilon` (otherwise invisible in the logged metrics, `Docs/
Training-Logging-Plan.md`), replay buffer size, and steps since the last
target-network sync; `train_step`'s `last_update_metrics` reports TD-error
mean/std/max, Q-value and target-Q mean/std, and the pre-clip gradient norm
(`torch.nn.utils.clip_grad_norm_`'s return value, previously computed and
discarded) plus whether it was clipped — all prefixed `dqn_` so DQN and
Dueling DQN runs share one chart namespace for direct comparison.

If the episode hits the eval cadence, evaluation metrics are merged into the
same row before logging.

The live console status line is separate from W&B logging. It is a lightweight
progress signal printed during long episodes and formatted by `TrainingLogger`.

## Evaluation and checkpoints

Every `EVAL_EVERY_EPISODES`, the trainer calls:

```python
eval_result = self.evaluator.evaluate(self.agent, episode=self.episode)
saved_best = self.evaluator.maybe_save_best(self.agent, eval_result)
```

Evaluation is side-effect limited: it temporarily makes the agent deterministic,
runs fixed eval games, restores agent mode/epsilon, and returns metrics. Best
policy checkpoints live under the run's `best/` folder.

Regular resume checkpoints are handled separately by:

```python
self.logger.checkpoint(episode=self.episode, agent=self.agent)
```

`TrainingLogger` decides checkpoint cadence and calls the agent's
`save_checkpoint(...)` / `load_checkpoint(...)` methods. W&B run names and
local checkpoint folders use the same agent-labeled run name. W&B config records
`run_name`, `agent_class`, and `model_class`, so runs can distinguish
`GNN_DQN_Agent` from `Dueling_DQN_Agent` even when they share the same trainer.

## How to change agent

To train a different learner, change the agent construction in `main()` (or in
your own script) and keep `Trainer.train(...)` unchanged.

The new agent must provide the interface the trainer/logger/evaluator call:

- `attach(player_id, env)`
- callable action selection via `agent((), state)`
- `remember(state, action, reward, next_state, done)`
- `learn(*, reached_max_steps=False) -> list[float]` — the trainer supplies
  only the dynamic time-limit boundary of the transition it just stored. The
  DQN agents currently ignore it; PPO uses it to stop GAE crossing into the
  next reset game without changing the non-terminal `done=False` bootstrap.
  `batch_size`/`n_steps`/rollout length remain agent-owned constants in
  `train_constants.py`.
- `save_checkpoint(path)` and `load_checkpoint(path)`
- `save_params(path)` for best-policy checkpoints
- `set_train_mode(train)` for evaluation
- `epsilon` and `train_mode` attributes — `Evaluator` reads/zeroes/restores
  `epsilon` around eval games regardless of agent type, so every agent needs
  the attribute even if it's inert (e.g. `PPO_Agent`, `Docs/PPO.md`)
- `on_episode_start(episode)` — optional, inherited as a no-op from
  `BaseAgent`. `GNN_DQN_Agent`/`Dueling_DQN_Agent` override it to decay their
  own `epsilon`; an agent that doesn't need per-episode state (PPO) needs no
  code at all
- `net` and `target_net` attributes for logging identity/config
- `label` class attribute (a short string like `"DQN"`/`"Dueling_DQN"`) used
  to build the default `run_name`/`checkpoint_dir` — there is no fallback if
  it's missing, so any new agent class must declare its own

For a Dueling DQN run, build `Dueling_DQN_Agent` instead of `GNN_DQN_Agent` and
give it a distinct run id or checkpoint directory:

```python
ctx = GameFactory.build(SetupStage.default_settings(n=MIN_PLAYERS))
agent = Dueling_DQN_Agent(
    player_id=0,
    env=ctx.env,
    train_mode=True,
)

trainer = Trainer(RUN_ID, agent=agent, resume=False, notes="Dueling_DQN_Agent")
trainer.train(n_episodes=TRAIN_EPISODES)
trainer.logger.finish()
```

That default trainer call writes to `Checkpoints/Dueling_DQN_<run_id>` and uses
`Dueling_DQN_<run_id>` as the W&B run name. Pass `checkpoint_dir=...` only when
you need a custom storage location.

Do not add a new trainer just to change network architecture. Add a new trainer
only if the algorithm genuinely needs a different loop, such as a different
rollout format, replay sampling contract, policy-gradient update cadence, or
evaluation rule.

## Related docs

- `Docs/Training-Logging-Plan.md` covers `TrainingLogger`, W&B config, logged
  metrics, and checkpoint cadence.
- `Docs/Eval.md` covers evaluator behavior and best-policy checkpoints.
- `Docs/Reward.md` covers reward calculation and reward-component logging.
- `Docs/GraphAdapter.md` covers perspective-relative encoding.
- `Docs/RL-Prep-Changes.md` is the historical implementation log.
