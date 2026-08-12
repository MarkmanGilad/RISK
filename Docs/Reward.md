# Reward

`RewardCalculator` in `risk/learning/reward.py` calculates the learner reward
for `Environment.step(...)`. The constants in `risk/learning/train_constants.py`
are the source of truth for all weights.

## Pipeline

For each action, `compute(...)` adds the applicable trade-in, reinforcement,
attack, occupy, and fortify components. Phase shaping is clipped to `[-10, 10]`
and multiplied by `REWARD_SHAPING_SCALE`; terminal win/loss rewards are added
separately. Phase shaping applies only to learner actions.

At the end of a learner turn, `Trainer` calls `end_of_turn(...)`. This compares
the learner's pre-turn board with the board after opponents have acted and
adds territory, army, and continent changes to the transition reward.

## Observability

`RewardCalculator.last_components` and `last_end_of_turn_components` expose
the individual terms. `Trainer` aggregates them into `reward_component_*`
metrics and also logs reinforcement-action counts and per-action values.

## Verification

`Temp/tests/test_reward.py` covers terminal, phase, and end-of-turn behavior.
`Temp/tests/test_environment.py` covers the environment paths that produce the
reward inputs.
