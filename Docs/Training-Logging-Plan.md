# Training Logging Plan (implemented)

**Current state: implemented.** `risk/learning/training_logger.py`'s
`TrainingLogger`, `GNN_DQN_Agent.save_checkpoint`/`load_checkpoint`
(`gnn_dqn_agent.py`), and `Trainer` integration (`trainer.py`) are all in
place per the design below — see `Docs/RL-Prep-Changes.md` for the
implementation log. The console `\r` progress print stayed independent, as
decided. RNG-exact resume was explicitly dropped (episode randomness
diverging after a resume is fine; only model/optimizer/replay continuity
matters). The rest of this doc is now also a reference for how each piece
works — see the modules above for the actual code.

## Goal
Build one logging/checkpointing flow for RL training with these rules:

- Metrics are logged to W&B (not to local metric files).
- Local disk stores checkpoints needed for:
  - resuming training from the exact point it stopped,
  - loading trained policy params for play/inference.
- W&B run init stores all train constants and the network string so each run is reproducible.

This plan is based on ideas from `Temp/Examples/PPO_Trainer.py`, adapted to the current `risk/learning/trainer.py` + `GNN_DQN_Agent` architecture.

## Scope and non-scope

In scope:
- New logging class/module in `risk/learning/`.
- Integration into `Trainer` checkpoint flow.
- W&B config capture (`train_constants` + model string + run metadata).

Out of scope (for this pass):
- Changing reward logic.
- Local metric plotting/history files.
- UI changes.
- The existing `\r` console progress print in `Trainer.train(...)`
  (`ep=.../... steps=... agent_turns=...`). Stays exactly as-is, independent
  of `TrainingLogger` — it's the simple "is the trainer alive and roughly
  where is it" signal, distinct from W&B's episode-level metrics. Not folded
  into the logger, not replaced by it.

## Proposed architecture

## 1) `TrainingLogger` (single entry point)
Create `risk/learning/training_logger.py` with one orchestrator class:

```python
class TrainingLogger:
    def __init__(self, run_id: int, checkpoint_dir: Path, use_wandb: bool, ...):
        ...

    def start_run(self, *, agent, trainer, extra_config: dict | None = None) -> None:
        ...

    def log_episode(self, *, episode: int, metrics: dict[str, float]) -> None:
        ...

    def checkpoint(self, *, episode: int, agent) -> tuple[bool, Path | None]:
        ...

    def try_resume(self, *, agent) -> dict | None:
        ...

    def finish(self) -> None:
        ...
```

Responsibilities:
- W&B init/log/finish.
- Local checkpoint save/load and interval-based cadence.
- Keep `Trainer` thin (trainer computes metrics; logger handles all I/O and scheduling).

**Recommendation — don't have `TrainingLogger` reach into `agent.net`/
`agent.optimizer`/`agent.replay_buffer` directly to build the checkpoint
payload.** That repeats the exact coupling problem fixed in `Reward.md`
(an orchestrator computing/serializing things that belong to a class it
doesn't own). Cleaner split, matching how `Environment` calls
`RewardCalculator`:

- `GNN_DQN_Agent` gains `save_checkpoint(path)` / `load_checkpoint(path)`
  that bundle its own `net`/`target_net`/`optimizer`/`_train_steps`/
  `epsilon`/`replay_buffer` — it already owns all of these, it should own
  serializing them too.
- `TrainingLogger.checkpoint(episode, agent)` checks interval thresholds
  (`CHECKPOINT_AFTER`, `CHECKPOINT_EVERY`) and calls `agent.save_checkpoint(path)`
  if needed — no direct attribute access into the agent's internals, and trainer
  never sees interval logic.

## 2) W&B backend behavior
If `use_wandb=True` and `wandb` is available:
- `wandb.init(project=..., name=..., config=...)` in `start_run`.
- `wandb.log(metrics, step=episode)` in `log_episode`.
- `wandb.finish()` in `finish`.

If disabled/unavailable:
- No-op behavior (training still runs).

## 3) Local checkpoint behavior
Only local artifacts required:

1. Full training checkpoint (`resume`):
- `episode`
- `run_id`
- `agent.net.state_dict()`
- `agent.target_net.state_dict()`
- `agent.optimizer.state_dict()`
- `agent._train_steps`
- `agent.epsilon` — note: currently a fixed value passed at construction
  (`GNN_DQN_Agent.__init__`, `gnn_dqn_agent.py:52`), no decay schedule exists
  anywhere yet, so this field is just a constant for now, not evidence of a
  schedule already in place.
- replay buffer contents (`agent.replay_buffer.save(...)` or embedded payload)
- optional trainer counters (`episode`, any rolling stats needed for scheduling)

2. Policy-only checkpoint (`play`):
- `agent.net.state_dict()` only

This split keeps play-loading simple while preserving exact resume capability.

**Gap vs. current code — this is new work, not wiring onto existing
methods:**
- `GNN_DQN_Agent.save_params`/`load_params` (`gnn_dqn_agent.py:98-104`)
  today only handle `net.state_dict()` — no optimizer, no `target_net`, no
  `_train_steps`. `save_checkpoint`/`load_checkpoint` (see recommendation
  above) are new methods, not extensions of `save_params`/`load_params`,
  which should stay as the policy-only path described in (2).
- `ReplayBuffer` (`replay_buffer.py`) has `save(path)` but no `load(path)`
  — needed before resume can restore buffer contents at all.

## W&B config payload (what gets stored at init)

`start_run` should build config from:

1. `risk/learning/train_constants.py`
- collect all uppercase names (or use `__all__` if preferred for explicit control)
- include exact values used in this run

2. Model identity
- `model_class`: `type(agent.net).__name__`
- `model_str`: `str(agent.net)`
- `target_model_str`: `str(agent.target_net)`
- `param_count`: total trainable params
- device (`cpu`/`cuda`)

3. Run metadata
- `run_id`
- timestamp
- optional git commit hash (if available)
- optional remark/notes field

## Metrics to log to W&B (episode-level)

Minimal v1 set:
- `episode`
- `episode_steps`
- `agent_turns`
- `episode_reward`
- `win` (0/1)
- `eliminated` (0/1)
- `done` (0/1)
- `epsilon`
- `replay_size`
- `learn_loss_mean` (mean of losses returned by `agent.learn(...)` this episode)

Recommended additions:
- `n_players`
- `seat`
- `checkpoint_saved` (0/1)
- throughput (`steps_per_sec`)

## Monitoring strategy for slow training (hours per run)

Do not rely on win-rate alone. In multi-player Risk, the agent can improve
substantially before converting to wins, especially when late-game snowball
advantages are already against it.

### A) Two-track monitoring

Track 1: Training episodes (online, noisy)
- Purpose: see learning dynamics and behavior shift while training.

Track 2: Periodic evaluation episodes (offline, stable)
- Purpose: measure real policy quality.
- Rules for eval: no learning updates, exploration off (`epsilon=0`), fixed
    seed set, fixed opponent roster.

### B) Training-track metrics (log every episode)

Core progress signals:
- `episode_reward_sum`: full sum of step rewards from start phase to terminal
    or truncation (keeps your requested total signal).
- `reward_per_agent_turn = episode_reward_sum / max(agent_turns, 1)`
- `agent_turns_survived`
- `alive_players_when_eliminated` (or max value when not eliminated)

Board-strength at last observed state:
- `territory_share_end`
- `army_share_end`
- `continent_bonus_share_end`

Behavior signals:
- `attack_actions_per_turn`
- `conquests_per_turn`
- `cards_gained`

### C) Early-game checkpoint metrics (inside each episode)

Log board-quality snapshots at fixed own-turn milestones, for example
`k in {5, 10, 20, 40}`:
- `territory_share_at_k`
- `army_share_at_k`
- `reinforcement_income_at_k`

Why this helps: these signals move much earlier than win-rate and show if
the policy is establishing stronger positions even in episodes it later loses.

### D) Evaluation-track metrics (log every N training episodes)

Run evaluation every fixed interval (example: every 200 training episodes),
with 20 eval episodes:
- `eval_win_rate`
- `eval_avg_alive_players_when_eliminated` (or equivalent rank proxy)
- `eval_avg_episode_reward_sum`
- `eval_avg_territory_share_end`
- `eval_avg_army_share_end`

This should be the primary indicator of actual policy improvement.

### E) W&B dashboard and interpretation

Keep the dashboard intentionally small. The core chart set should stay at
4-6 metrics so it is readable during long runs. Suggested primary set:
- `eval_win_rate`
- `eval_avg_alive_players_when_eliminated`
- `episode_reward_sum`
- `reward_per_agent_turn`
- `territory_share_at_20`
- `learn_loss_mean`

Secondary diagnostics (log them, but do not build the whole dashboard
around them):
- `army_share_at_20`
- `epsilon`
- `replay_size`
- `checkpoint_saved`

Interpretation rule:
- If win-rate is still low but `reward_per_agent_turn`,
    `territory_share_at_20`, and eval rank proxy improve, the agent is getting
    better.
- If only `episode_reward_sum` rises while eval metrics stay flat, likely the
    policy is optimizing shaping patterns without real game-strength gains.

## Integration plan (no code yet)

1. Add `TrainingLogger` module.
2. Inject logger into `Trainer.__init__` with flags:
- `use_wandb: bool = True`
- `project_name: str = "Risk-GNN-DQN"`
- `resume: bool = True`
- `notes: str | None = None`
3. On trainer startup:
- call `logger.start_run(agent=self.agent, trainer=self, extra_config=...)`
- call `logger.try_resume(agent=self.agent)` and restore trainer counters if present.
4. In training loop:
- accumulate episode metrics
- call `logger.log_episode(...)` once per episode
- call `logger.checkpoint(episode=..., agent=...)` once per episode
  (logger internally checks cadence and skips if not at checkpoint time)
5. On exit:
- call `logger.finish()`

## Resume semantics

Resume should be exact for training continuity:
- same optimizer state
- same target net state
- same replay buffer
- same trainer episode counter

**RNG state is explicitly out of scope for resume.** `Trainer._build_episode_context`
already uses `random.SystemRandom()` for player-count/seat selection and
`SetupStage.default_settings(..., seed=None)` for the per-episode board deal
— both deliberately unseeded (OS entropy), and `SystemRandom` has no
state to capture even if we wanted to. Confirmed this is fine: episode
randomness not replaying identically across a resume is not a problem —
only the model/optimizer/replay continuity matters for training
correctness, not bit-exact episode reproduction.

If any resume component is missing/corrupt:
- load what is valid,
- print a clear warning,
- continue from best-effort state (do not hard crash unless user opts in).

## Dependency note

`wandb` is not currently listed in `requirements.txt`.
Plan for v1:
- keep W&B optional (`use_wandb=False` works without package),
- document/install `wandb` when enabling remote logging.

## Test plan (after implementation)

1. Unit tests (`Temp/tests/test_training_logger.py` — matches
   `risk/learning/training_logger.py` per `Docs/Testing.md`'s one-file-per-
   module convention):
- config builder includes constants and model string
- checkpoint save/load round-trip restores agent/trainer fields
- no-op mode works when W&B disabled

2. Integration smoke:
- run a short training (2-5 episodes) with W&B disabled
- verify checkpoint files are created
- resume and confirm episode counter continues

3. W&B smoke (manual):
- one short run with W&B enabled
- confirm config tab contains constants + model strings
- confirm episode metrics appear over time

## Implementation order

1. `TrainingLogger` skeleton + local checkpoint save/load.
2. Trainer integration with no-op W&B path.
3. W&B init/config/log/finish.
4. Resume-from-checkpoint exactness (replay + RNG).
5. Tests + short smoke runs.

## Acceptance criteria

- Training can run fully without W&B.
- W&B run shows constants + `str(net)` at init.
- Episode metrics are visible in W&B during training.
- Local resume continues from the saved episode with preserved optimizer/target/replay state.
- Policy-only checkpoint loads for play without trainer-only state.
