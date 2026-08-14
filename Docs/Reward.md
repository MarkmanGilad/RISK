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

## Current constants

All values are `risk/learning/train_constants.py`'s live `REWARD_*` constants.

| Area | Constants | Values |
|---|---|---|
| Terminal | win / loss | `+300`, `-300` |
| Dense shaping | scale (terminal excluded) | `0.3` |
| Step safety | shaping cap | `±10` |
| Trade-in | early / territory match | `0.30`, `0.60` |
| Reinforce | ready scale / ready ratio / ready cap | `0.50`, `1.50`, `7.00` |
| Reinforce | total scale / total ratio / continent / interior / split | `0.50`, `2.00`, `5.00`, `-0.80`, `0.20` |
| Attack | fewer dice / ratio scale / ratio cap / ratio threshold | `-1.25`, `2.00`, `3.00`, `1.50` |
| Attack | continent domination / margin / advantage / army trade | `0.80`, `0.10`, `1.20`, `0.60` |
| Attack | eliminate base / per card / continent captured | `4.00`, `1.50`, `4.00` |
| Attack | conquer / conquer-with-card / card-territory match / stop without card / unfinished target | `1.20`, `1.00`, `0.60`, `-2.00`, `-0.50` |
| Occupy | forward momentum | `1.00` |
| Fortify | toward frontier / balance / continent push | `1.00`, `2.00`, `0.80` |
| End of turn | territory delta / army-share delta / continent delta | `20.00`, `0.10`, `2.50` |
| End of turn | territory hold / continent lost | `0.00`, `5.00` |

The constants have different input scales. Equal numeric values do not imply
equal reward strength: a board-share delta is much smaller than an
action-local fraction or an army-ratio term. `REWARD_TERRITORY_HOLD = 0.00`
means merely continuing to own territory pays nothing; only a share change
across the opponent round (`territory_delta`) or losing continent bonus
(`continent_lost`) contributes there.

## Reward by phase

All applicable terms in an action row are added, then the total **step
shaping** is clipped to `[-10, +10]` and multiplied by
`REWARD_SHAPING_SCALE = 0.3`. Terminal reward is not scaled. End-of-turn
shaping is calculated separately, at the learner's turn boundary, and
multiplied by the same scale.

| Action | Rewarded behavior / formula | Main scale |
|---|---|---:|
| `SkipTradeAction` | Optional three/four-card trade: `+0.30 x hand_factor / card_set_value` | small positive |
| `TradeInAction` | Optional trade: negative of the skip term; `+0.60` if the set matches an owned territory card | small |
| `ReinforcementAction` | Readiness against the weakest adjacent enemy and the total adjacent enemy force; gated contested-continent priority; interior and partial-action penalties | mixed |
| `AttackAction` | Ratio: `+2.0 x (min(attacker / defender, 3.0) - 1.5)`; army trade: `+0.60 x (defender losses - attacker losses)`; plus applicable continent, conquest, card, and elimination bonuses | largest / frequent |
| `StopAttackAction` | `-2.0` if a real attack remains and no territory was conquered this turn; if unfinished attack targets exist *after* a conquest, `-0.5` per unfinished target instead — the two penalties never stack | negative |
| `OccupyAction` | `+1.0 x moved / (source armies - 1)` | `0` to `+1` |
| `FortifyAction` | Toward frontier: positive moved fraction; away from frontier: negative fraction; two frontiers: negative threat-balance error; plus continent push | mixed |
| Skip fortify | No fortify shaping | `0` |

Attack-specific bonuses, when their conditions occur, are:

| Event during `AttackAction` | Extra reward |
|---|---:|
| Used fewer than maximum legal dice | `-1.25` |
| Continent domination / continent advantage | up to their formula-dependent values (`0.80`, `1.20` scales) |
| Eliminated a player | `4.0 + 1.5 x cards taken` |
| Conquered a territory | `+1.20` |
| First conquest/card of the turn | `+1.00` |
| Drawn card matches an owned territory | `+0.60` |
| Completed a continent | `+4.00` |

The `attack` W&B component excludes both the elimination and
unfinished-attack-target portions; they are logged separately as
`reward_component_eliminate` and `reward_component_unfinished_attack`.

## Observability

`RewardCalculator.last_components` and `last_end_of_turn_components` expose
the individual terms. `Trainer` aggregates them into `reward_component_*`
metrics and also logs reinforcement-action counts and per-action values.

## Verification

`Temp/tests/test_reward.py` covers terminal, phase, and end-of-turn behavior.
`Temp/tests/test_environment.py` covers the environment paths that produce the
reward inputs.
