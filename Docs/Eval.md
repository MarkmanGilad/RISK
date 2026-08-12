# Evaluation

`Evaluator` runs deterministic quality checks during training without adding
transitions to replay or changing model weights. It temporarily sets
`epsilon=0` and evaluation mode, restores both values afterward, and attaches
the learner to each fresh evaluation environment.

Each evaluation plays six fixed games: three 4-player games against Raider,
Sentinel, and Killbot, and three 6-player games against Random, Raider,
Sentinel, Empire, and Killbot. Seeds are `0`, `1`, and `2`; learner seats are
rotated in both suites.

The reported score is:

```text
100 * eval_win_rate
+ 2 * eval_avg_territories_conquered
+ 5 * eval_avg_reward_per_agent_turn
```

`Evaluator.maybe_save_best(...)` keeps the top `EVAL_KEEP_BEST` policy-only
files in `<checkpoint_dir>/best/` and records them in `manifest.json`.
`Trainer` runs evaluation every `EVAL_EVERY_EPISODES`, merges the `eval_*`
metrics into that episode's logging row, and saves a regular training
checkpoint separately.

Focused coverage is in `Temp/tests/test_evaluator.py`.
