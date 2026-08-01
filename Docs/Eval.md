# Evaluation plan

**Current state: fully implemented.** `risk/learning/evaluator.py` has the
`Evaluator` class (`evaluate(...)`/`maybe_save_best(...)`); `EVAL_EVERY_EPISODES`/
`EVAL_KEEP_BEST`/`EVAL_MAX_STEPS` live in `risk/learning/train_constants.py`;
`Trainer` constructs and calls it as described in "Trainer integration"
below. Covered by `Temp/tests/test_evaluator.py`.

## Goal

Add a simple evaluation path for trained `GNN_DQN_Agent` models so "best
model" saving is based on measured play quality, not only on checkpoint
cadence. Regular checkpoints should still exist for resume, but best-model
selection should come from periodic deterministic evaluation.

This should let the current training loop stay mostly as-is. The loop already
stores one learner transition and runs one training update per learner
decision (`TRAIN_STEPS_PER_CALL = 1`), so the immediate problem is not "too
little training after each game." The immediate problem is "we save by episode
number, not by model quality."

## Proposed class

Create a new module:

```python
risk/learning/evaluator.py
```

with one main class:

```python
class Evaluator:
    def __init__(self, *, max_steps: int, keep_best: int, best_dir: str | Path) -> None:
        ...

    def evaluate(self, agent: GNN_DQN_Agent, *, episode: int) -> dict:
        ...

    def maybe_save_best(self, agent: GNN_DQN_Agent, eval_result: dict) -> bool:
        ...
```

Keep it plain and small. Do not introduce dataclasses for this pass. Return
plain dictionaries for metrics, matching the existing `TrainingLogger` style.

## Important design choice: copy `Trainer.train()`'s loop, not `SelfPlay`'s

Do not import `SelfPlay` for evaluation, and do not copy `self_play.py`'s
loop shape either — it calls `env.step(action)` with no `reward_player`,
which makes `Environment.step` skip reward computation entirely (it returns
`reward=0.0` unconditionally whenever `reward_player is None`,
`environment.py:150-155`). An evaluator built on that shape would have no
reward signal to report at all.

Copy `Trainer.train()`'s loop instead (`risk/learning/trainer.py:116-210`),
adapted for eval:

- build a fresh `GameContext`,
- reject human seats,
- call `agent.attach(learner_seat, env)`,
- play any opening moves before the learner's first turn,
- loop until terminal, learner elimination, or `max_steps`,
- on the learner's turn: snapshot `state`, get the action, call
  `env.step(action, reward_player=learner_seat)`,
- then play opponents' moves until control returns to the learner (or the
  game ends), summing `result.reward` across that whole span — same
  `reward_player=learner_seat` the entire time, so the sum is correct
  regardless of whose action triggered which shaping term,
- if the learner's action was a `FortifyAction`, also add
  `env.reward.end_of_turn(state, next_state, learner_seat)` — this is the
  *only* place `REWARD_TERRITORY_DELTA`/`REWARD_ARMY_DELTA_RELATIVE_SCALE`/
  `REWARD_CONTINENT_DELTA_RELATIVE` ever fire (`Docs/Reward.md`'s
  "End-of-turn" section); skipping this step would silently under-report
  `episode_reward_sum`/`reward_per_agent_turn` relative to the
  identically-named training metric, making them not comparable,
- accumulate `territories_conquered` the same way `Trainer` does — diffing
  owners across each learner-turn span (`state` vs `next_state`), not a
  single before/after diff over the whole game, since a territory can be
  conquered, lost, and re-conquered more than once,
- collect eval metrics,
- do **not** call `remember(...)`,
- do **not** call `learn(...)`.

Reason: eval needs the exact same reward/metric accounting `Trainer` already
gets right, just without the replay/training side effects. `SelfPlay` is
reward-agnostic by design (it's also used for plain non-training play), so
extending it with a `reward_player` knob would be the wrong module to grow —
`Trainer`'s loop is already the correct reference implementation to copy
from.

## Evaluation mode

Evaluation should temporarily make the learner deterministic:

```python
old_epsilon = agent.epsilon
old_train_mode = agent.train_mode
agent.epsilon = 0.0
agent.set_train_mode(False)
...
restore old values in finally
```

The evaluator should call `agent.attach(learner_seat, env)` for each eval game
the same way `Trainer` does, so the agent sees the correct board topology and
perspective.

Because `agent.attach(...)` mutates the agent's current `env`/seat binding,
eval should either run only at the end of a training episode (the planned v1)
or restore the previous binding afterward. In the planned integration, this is
safe because the next training episode calls `attach(...)` again with its own
fresh env and learner seat before the agent acts.

## Eval games and opponents

Do not use one eval episode with one player setup. Risk is too noisy for that;
one game can be decided by setup, opponent mix, or dice.

Recommended v1: **two fixed eval suites**, with fixed seeds. `killbot` is in
both suites deliberately (not one of several randomly-sampled kinds like in
`TRAIN_OPPONENT_AGENT_KINDS`) so it's always part of every eval run:

1. **Small tactical game**
   - 4 players total.
   - Learner + `raider` + `sentinel` + `killbot`.
   - Purpose: checks whether the model can survive and convert attacks in a
     lower-noise game.

2. **Mixed full game**
   - 6 players total.
   - Learner + `random` + `raider` + `sentinel` + `empire` + `killbot`.
   - Purpose: checks the real training-style environment with multiple
     opponent behaviors.

Start with 3 fixed seeds per suite, so each evaluation is 6 games total. If
runtime is too high, reduce to 2 seeds per suite. If best-model selection is
too noisy, increase to 5 seeds per suite.

Rotate the learner through fixed seats so evaluation does not reward always
acting from one position while remaining reproducible. For the three tactical
seeds, use learner seats `0`, `1`, and `2`; for the three full-game seeds, use
`0`, `2`, and `4`. The roster fills the other seats in its listed order.
`GraphAdapter`'s perspective-relative encoding ensures the learner keeps the
same input frame despite moving physical seats.

**Build each suite's roster via `GameFactory.build(settings)`, not by
hand-instantiating heuristic agents.** A fixed `GameSettings.seed` makes
`Environment`'s own RNG deterministic (territory dealing, card shuffling,
dice rolls all route through the one `self._rng` it seeds from
`settings.seed`, `environment.py:80`) — but `RaiderAgent`/`SentinelAgent`/
`EmpireAgent` each carry their own separate `random.Random(seed)` for
tie-breaking among equally-scored moves, defaulting to `seed=None` (OS
entropy) if not given one explicitly (`heuristic_agent.py:108`). Hand
-instantiating opponents the way `self_play.py`'s scratch `main()` does
(`RaiderAgent(player_id=0, env=ctx.env)`, no seed) would make eval
non-reproducible despite the "fixed seed" framing. `GameFactory.build_agents`
already derives a deterministic per-seat seed (`seed = (settings.seed or 0)
+ p.id + 1`, `factory.py:42`) for every agent it builds from `agent_kind` —
so build the roster the same way `self_play.py`'s `main()` was updated to
(`InitScreenState` → `set_agent_kind(...)` per seat → `state.build_settings
(seed=...)` → `GameFactory.build(...)`), and determinism comes for free.
The learner still needs a manual override afterward (`ctx.agents[learner_seat]
= agent`), same as every other manual-agent-injection call site, since there's
no `agent_kind` for `GNN_DQN_Agent`.

## Carr's methodology (arXiv:2009.06355) and our opponent split

Reference: Jamie Carr, *Using Graph Convolutional Networks and TD(λ) to play
the game of Risk* — saved locally at
`Docs/Carr-2020-GCN-TDlambda-Risk-DAD.pdf`. Two facts from that paper are
worth pinning here because they directly justify (and qualify) how we pick
eval opponents:

- **Carr did not use self-play.** He explicitly rejected it as infeasible
  ("Self-play was not viable due to game-tree search times", §3.4.3) and
  instead trained D.A.D via **offline TD(λ)** on ~200,000 turn end-states
  (~2000 matches) generated by six inbuilt Lux AIs: **Angry, Pixie, Cluster,
  Quo, Killbot, and Boscoe**. So **Killbot was in his training set.**
- **His benchmark reused one training opponent and held out two others.**
  The final evaluation (§3.8.1) was 283 games versus **Killbot, EvilPixie,
  and Bort** — of which EvilPixie and Bort were "AIs not included in the
  training data." So Killbot appears in **both** training and eval; the
  genuinely held-out generalization opponents were EvilPixie and Bort.
  Headline result: D.A.D placed 1st **35.3%** of the time, "nearly double
  that of Killbot," and Carr concludes Killbot "seems to be the strongest of
  the inbuilt Lux AIs."

Implications for our setup:

- Having `killbot` in **both** `TRAIN_OPPONENT_AGENT_KINDS` and the eval
  suites above is **consistent with Carr**, not a mistake — his strongest
  benchmark bot was also a training opponent. (Note the paradigm differs:
  Carr trained offline on *recorded* Killbot games; we train online RL
  playing *live* against `KillbotAgent`.)
- To also reproduce Carr's **held-out** benchmark, reserve at least one
  opponent kind for eval only — never placed in `TRAIN_OPPONENT_AGENT_KINDS`
  — so eval reports both a "seen" score (vs `killbot`) and an "unseen
  generalization" score (vs the held-out kind). A natural candidate is a
  future `EvilPixieAgent`, or simply holding `empire` (or another existing
  personality) out of training. This split is **not yet implemented**: today
  every eval opponent kind is also a training kind.

## Metrics to collect per eval game

Use metrics already meaningful in training/W&B:

- `win`: `1` if learner wins, else `0`.
- `territories_conquered`: count of territories that flip from not-learner to
  learner during the game.
- `episode_reward_sum`: sum of rewards attributed to the learner during eval.
- `agent_turns_survived`: learner decisions/actions before end/elimination.
- `reward_per_agent_turn`: `episode_reward_sum / max(agent_turns_survived, 1)`.

Optional but useful:

- `final_territory_count`
- `final_continent_bonus`
- `eliminated`: `1` if learner was eliminated.
- `step_count`

For v1 best-model scoring, keep the official score based only on the three
requested signals: win/loss, territories conquered, reward per agent turn.

## Score formula

Use a simple weighted score:

```python
score =
    100.0 * eval_win_rate
    + 2.0 * eval_avg_territories_conquered
    + 5.0 * eval_avg_reward_per_agent_turn
```

Why this shape:

- `win_rate` stays the anchor and dominates when the agent can actually win.
- `territories_conquered` gives a concrete behavior signal before wins are
  common.
- `reward_per_agent_turn` captures the dense reward design and should move
  earlier than win rate.

Keep this formula in one private method, for example:

```python
def _score(self, metrics: dict) -> float:
    ...
```

Do not make the formula too clever in v1. If it rewards the wrong behavior,
tune the three weights after looking at W&B.

## Aggregated eval metrics

`Evaluator.evaluate(...)` should return a dict that can be logged to W&B:

```python
{
    "eval_score": ...,
    "eval_win_rate": ...,
    "eval_avg_territories_conquered": ...,
    "eval_avg_reward_per_agent_turn": ...,
    "eval_avg_agent_turns_survived": ...,
    "eval_games": ...,
}
```

The trainer can call:

```python
eval_result = evaluator.evaluate(self.agent, episode=self.episode)
saved_best = evaluator.maybe_save_best(self.agent, eval_result)
eval_result["eval_saved_best"] = int(saved_best)
metrics.update(eval_result)  # merge into the per-episode training metrics dict
self.logger.log_episode(episode=self.episode, metrics=metrics)
```

Call `maybe_save_best(...)` before `log_episode(...)`, not after — that is
the only ordering where `eval_saved_best` exists in time to be added to
`eval_result`. Merge `eval_result` into the same `metrics` dict the episode's
own training metrics already populate, rather than a second standalone
`log_episode(...)` call — one call per episode either way, so eval and
training metrics always land in the same W&B row/step. (See "Trainer
integration" below for the exact same snippet in context.)

Use a prefix like `eval_` for all W&B metrics so they are visually separate
from training episode metrics.

## When to evaluate

Add constants in `train_constants.py`:

```python
EVAL_EVERY_EPISODES = 25
EVAL_KEEP_BEST = 5
EVAL_MAX_STEPS = MAX_STEPS_PER_EPISODE
```

For the 500-episode run horizon, evaluate every 25 episodes. This gives 20
policy measurements instead of 10 while retaining the deterministic six-game
suite. If evaluation proves too slow, use 50. This is separate
from resume checkpointing; both can exist:

- resume checkpoint: every `CHECKPOINT_EVERY`,
- best policy save: whenever eval score enters the top N.

## Keeping N best models

Best models should be policy-only saves, not full training checkpoints. Use
`GNN_DQN_Agent.save_params(...)` so loading a best model for play is simple.

Suggested directory:

```text
Checkpoints/run_013/best/
```

Save files like:

```text
best_ep000350_score0123.45.pt
best_ep000400_score0127.10.pt
manifest.json
```

`manifest.json` is the simple source of truth:

```json
[
  {"path": "best_ep000400_score0127.10.pt", "episode": 400, "score": 127.10},
  {"path": "best_ep000350_score0123.45.pt", "episode": 350, "score": 123.45}
]
```

On every eval:

1. Load manifest if it exists, otherwise start with `[]`.
2. If fewer than `EVAL_KEEP_BEST` entries exist, save the current params.
3. If the new score is higher than the worst saved score, save current params.
4. Sort descending by score.
5. Delete entries beyond `EVAL_KEEP_BEST` from disk.
6. Rewrite manifest.

This avoids guessing from filenames and keeps cleanup simple. Do not delete
regular resume checkpoints here; evaluator only manages the `best/` folder.

## Trainer integration

`Trainer.__init__` constructs the evaluator once, alongside `self.logger`:

```python
self.evaluator = Evaluator(
    max_steps=EVAL_MAX_STEPS,
    keep_best=EVAL_KEEP_BEST,
    best_dir=self.checkpoint_dir / "best",
)
```

Keep the training loop behavior unchanged except for periodic eval. Place
this block after `metrics` (the normal per-episode training metrics dict)
is built, merging `eval_result` into that same dict rather than calling
`self.logger.log_episode(...)` a second time — one `log_episode` call per
episode either way, so eval and training metrics always land in the same
W&B row/step on eval episodes, with no risk of the two calls landing on
different auto-incremented steps:

```python
if self.episode % EVAL_EVERY_EPISODES == 0:
    eval_result = self.evaluator.evaluate(self.agent, episode=self.episode)
    saved_best = self.evaluator.maybe_save_best(self.agent, eval_result)
    eval_result["eval_saved_best"] = int(saved_best)
    metrics.update(eval_result)
self.logger.log_episode(episode=self.episode, metrics=metrics)
self.logger.checkpoint(episode=self.episode, agent=self.agent)
```

This placement (before `checkpoint(...)`, not after) is still safe per
"Evaluation mode"'s no-leak argument: `checkpoint(...)` only ever persists
`net`/`target_net`/`optimizer`/`train_steps`/`epsilon`/the replay buffer —
never `agent.env`/`agent.player_id` — so it doesn't matter whether eval's
last `attach(...)` call (to some eval game's env) happens before or after
it. What actually matters is that the *next* episode's
`_build_episode_context()` calls `attach(...)` again before the agent acts,
which it always does regardless of where in the current episode's tail eval
ran.

This means:

- training still learns during normal episodes as today,
- eval does not add transitions to replay,
- eval does not train the net,
- best-model saving is based on deterministic eval quality.

## W&B

Log eval metrics only on eval episodes. W&B can handle sparse metrics; they
will appear as points every `EVAL_EVERY_EPISODES` episodes.

Recommended primary eval charts:

- `eval_score`
- `eval_win_rate`
- `eval_avg_territories_conquered`
- `eval_avg_reward_per_agent_turn`
- `eval_saved_best`

Training charts can stay as they are.

## Test plan

Read `Docs/Testing.md` before adding tests.

Suggested tests:

1. `Temp/tests/test_evaluator.py`
   - evaluates a tiny fixed setup with a simple agent,
   - returns expected metric keys,
   - restores learner `epsilon` and `train_mode` after eval.

2. Best-model retention test
   - feed fake eval scores,
   - assert only top `N` files remain,
   - assert `manifest.json` is sorted descending.

3. Trainer integration smoke
   - use a fake evaluator or very small eval interval,
   - confirm eval metrics are passed to logger,
   - confirm normal training calls still happen.

## Implementation order

1. Add `Docs/Eval.md` plan.
2. Add eval constants in `train_constants.py`.
3. Add `risk/learning/evaluator.py`, with its rollout loop adapted from
   `Trainer.train()` (see "Important design choice" above) — not from
   `self_play.py`.
4. Add best-model manifest/save/delete logic.
5. Integrate optional evaluator into `Trainer`.
6. Add focused tests.
7. Update `Docs/ChangeLog.md` after implementation.

## Open tuning choices

- Eval every 25 vs 50 episodes.
- 2 vs 3 fixed seeds per eval suite.
- Whether fixed learner seat is enough or eval should rotate learner seat.
- Whether `eval_score` should include final board strength later.

Recommended v1:

- `EVAL_EVERY_EPISODES = 25`
- `EVAL_KEEP_BEST = 5`
- two eval suites,
- 3 seeds per suite,
- fixed learner seat `0`,
- score from win rate, territories conquered, reward per agent turn only.
