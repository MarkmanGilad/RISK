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
the board immediately before the learner's turn-ending action with the board
after opponents have acted and adds territory, army, and continent changes to
the transition reward. This end-of-turn sum is scaled, but is not clipped.

## Current constants

All values are `risk/learning/train_constants.py`'s live `REWARD_*` constants.

| Area | Constants | Values |
|---|---|---|
| Terminal | win / loss | `+300`, `-300` |
| Dense shaping | scale (terminal excluded) | `0.3` |
| Step safety | shaping cap | `+/-10` |
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
multiplied by the same scale without clipping.

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

---

## Summary table: reward calculation by phase

For a learner action, the reward calculation is:

```text
step_reward = terminal + S(trade + reinforce + attack + occupy + fortify)
S(x) = 0.3 * clip(x, -10, +10)
```

At the learner's turn boundary, the trainer adds a separate board term:

```text
turn_boundary_reward = B(territory + army + continent terms)
B(x) = 0.3 * x
```

Terminal reward is neither clipped nor scaled. Every action-phase row marked
`S` is first summed with the other applicable action-phase rows, then the
whole sum is clipped once. Every board row marked `B` is summed separately,
scaled by `0.3`, and **not clipped**. Phase shaping applies only to learner
actions.

| Phase | Trigger | Raw terms before final scaling | Applied as |
|---|---|---|---|
| Terminal | Game ends | Learner wins: `+300`.<br>Learner does not win: `-300`. | Directly added |
| Trade-in | `SkipTradeAction` or `TradeInAction`; early-trade term only applies with 3 or 4 cards | Skip optional trade: `+0.30 * f(hand) / V`.<br>Make optional trade: `-0.30 * f(hand) / V`.<br>Set includes an owned-territory card: `+0.60`. | `S` |
| Reinforce | `ReinforcementAction` | Interior placement: `-0.80` per placed-on territory.<br>Readiness: `0.50 * [g_1.5(A_after / E_min) - g_1.5(A_before / E_min)]`.<br>Total coverage: `0.50 * [g_2.0(A_after / E_all) - g_2.0(A_before / E_all)]`.<br>Ready, partly owned continent: `5.0 * placed / (continent_armies + placed) * (territory_share + army_share) / continent_size`.<br>Partial budget: `-0.20` once. | `S` |
| Attack | `AttackAction` or `StopAttackAction` | **Attack:** fewer than max dice: `-1.25`; force ratio: `2.0 * (min(attacker_armies / defender_armies, 3.0) - 1.5)`; continent domination: `0.80 / (territories_not_owned + 1)`; continent advantage: `1.20 * continent_advantage / (territories_not_owned + 1)`; army trade: `0.60 * (defender_losses - attacker_losses)`; eliminate: `4.0 + 1.5 * cards_taken`; conquer: `+1.20`; complete continent: `+4.00`; first turn card: `+1.00`; useful drawn card: `+0.60`.<br><br>**Stop:** no conquest this turn while a real or unfinished attack remains: `-2.00`; after a conquest, unfinished targets remain: `-0.50 * unfinished_target_count`. | `S` |
| Occupy | `OccupyAction` after a conquest | `1.0 * moved / (source_armies_before - 1)`. | `S` |
| Fortify | Non-skip `FortifyAction` | Between two frontiers: `-2.0 * abs(destination_after - threat_target) / (source_after + destination_after)`.<br>Interior to frontier: `+1.0 * moved / (source_armies_before - 1)`.<br>Frontier to interior: `-1.0 * moved / (source_armies_before - 1)`.<br>Destination-continent push: `0.80 * (owned_in_destination_continent / continent_size) / continent_size`.<br>Skip fortify: `0`. | `S` |
| End-of-turn board | Learner transition closes; after opponents act if the game continues | Territory: `20.0 * (territory_share_after - territory_share_before)`.<br>Army share: `0.10 * (learner_army_share_after - learner_army_share_before)`.<br>Continent bonus: `2.50 * (bonus_share_after - bonus_share_before)`.<br>Continent lost: `5.0 * (bonus_after - bonus_before) / total_continent_bonus` (negative).<br>Territory hold: `0.0 * territory_share_after = 0`. | `B`; hold currently has no effect |

Definitions used in the table:

- `f(3) = 1`, `f(4) = 0.5`, otherwise `f = 0`; `V` is
  `card_set_value(cards_traded_in_count)`.
- `g_t(x) = max(0, min(x, 7) - t)`. `E_min` is the weakest adjacent
  enemy army count; `E_all` is the sum of adjacent enemy armies; `A_before`
  and `A_after` are the reinforced territory's army counts before and after
  placement.
- The reinforcement continent term applies only when the post-placement
  weakest-enemy ratio is at least `1.5` and the learner owns some, but not
  all, territories in the continent. Its shares are measured before placement.
- `continent_advantage` is the product of: positive territory share above
  `1 / alive_players`, positive troop share above `0.5`, and normalized
  continent value. This is the exact `_continent_advantage(...)` helper.
- `threat_target` assigns the combined post-fortify armies between two
  frontiers in proportion to their adjacent enemy-army threat.
