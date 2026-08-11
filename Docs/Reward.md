# Reward

This is the current reward reference for RL training. It describes what is
implemented now, not old plans or experiments. The source of truth is
`risk/learning/reward.py` and the values in
`risk/learning/train_constants.py`.

Historical reasoning and superseded proposals are in `Docs/ChangeLog.md` and
Git history.

## Reward pipeline

`Environment.step(...)` calls:

```python
RewardCalculator.compute(action, info, before, after, reward_player, done, winner)
```

for every actor's action. Phase shaping applies only when the current actor is
the learner (`before.current_player_index == reward_player`), so opponent
actions never receive reward as though the learner chose them. Terminal
reward is the exception: it is always about the learner's outcome.

```text
step_shaping = clip(
    trade_in + reinforce + attack + occupy + fortify,
    -10, +10,
)
step_reward = terminal + 0.3 x step_shaping
```

At the learner's turn boundary, `Trainer` also adds:

```python
RewardCalculator.end_of_turn(before_turn, after_turn, learner_seat)
```

This compares the board immediately after the learner's turn-ending action
with the board after every opponent has taken a turn. It is added when the
learner fortifies or when its action ends the episode. The latter means a
winning `AttackAction` or `OccupyAction` also receives end-of-turn reward.

One replay transition is stored per learner decision. Its reward is the step
reward above, plus `0.3 x` end-of-turn reward when that boundary is reached.

There is no terminal reward for a `MAX_STEPS_PER_EPISODE` timeout: it is a
truncation, not a terminal game state, so its replay transition keeps
`done=False` and may bootstrap.

## Current constants

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
| Attack | conquer / conquer-with-card / card-territory match / stop without card | `1.20`, `1.00`, `0.60`, `-2.00` |
| Occupy | forward momentum | `1.00` |
| Fortify | toward frontier / balance / continent push | `1.00`, `2.00`, `0.80` |
| End of turn | territory delta / army-share delta / continent delta | `1.00`, `0.10`, `2.50` |
| End of turn | territory hold / continent lost | `0.05`, `5.00` |

The constants have different input scales. Equal numeric values do not imply
equal reward strength: a board-share delta is much smaller than an
action-local fraction or an army-ratio term.

## Action reward summary

All applicable terms in an action row are added, then the total **step
shaping** is clipped to `[-10, +10]` and multiplied by
`REWARD_SHAPING_SCALE = 0.3`. Terminal reward is not scaled. End-of-turn
shaping is separately calculated and multiplied by the same scale.

| Action | Rewarded behavior / formula | Main scale |
|---|---|---:|
| `SkipTradeAction` | Optional three/four-card trade: `+0.30 x hand_factor / card_set_value` | small positive |
| `TradeInAction` | Optional trade: negative of skip term; `+0.60` if the set matches an owned territory card | small |
| `ReinforcementAction` | Readiness against the weakest adjacent enemy and the total adjacent enemy force; gated contested-continent priority; interior and partial-action penalties. See the complete formula below. | mixed |
| `AttackAction` | Ratio: `+2.0 x (min(attacker / defender, 3.0) - 1.5)`; army trade: `+0.60 x (defender losses - attacker losses)`; plus applicable continent, conquest, card, and elimination bonuses | largest / frequent |
| `StopAttackAction` | `-2.0` only if a real attack remains and no territory was conquered this turn | negative |
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

## Phase shaping

### Trade in

When trading is optional (three or four cards), skipping earns:

```text
REWARD_TRADE_IN_EARLY × hand_factor / card_set_value
```

where `hand_factor` is `1.0` for three cards and `0.5` for four. Trading in
that same situation receives the negative of this term. A forced trade gets
no early-trade term. A chosen trade set that contains a card for an owned
territory earns the territory-match bonus.

### Reinforce

For each frontier placement, the reward measures readiness against both the
weakest adjacent enemy and all adjacent enemy armies. A contested-continent
term applies only after the weak-neighbour readiness threshold is reached.
Interior placement receives a penalty, and an action that uses less than the
visible budget receives one fixed split penalty. The complete formulas and
constants are in [Reinforcement reward implementation](#reinforcement-reward-implementation).

### Attack

An `AttackAction` is rewarded in three layers. All applicable terms add
together before the per-step shaping cap is applied.

| Layer | What it asks | Terms |
|---|---|---|
| Decision | Was this a favourable attack to choose? | fewer-dice penalty, army-ratio reward, continent domination, continent advantage |
| Outcome | Did the dice exchange go well? | army-trade reward |
| Result | Did the attack advance the game? | conquest, card, continent-capture, and elimination bonuses |

The main decision-quality term is calculated **before** the dice roll:

```text
ratio_reward = 2.0 x (min(attacker_armies / defender_armies, 3.0) - 1.5)
```

It is negative below a 1.5:1 army ratio, zero at 1.5:1, positive above it,
and capped at `+3.0` so an overwhelming stack does not create an unlimited
bonus. It rewards selecting a favourable battle even if the dice roll later
goes badly.

The outcome term is:

```text
army_trade_reward = 0.60 x (defender_losses - attacker_losses)
```

It is positive when the defender loses more armies, negative when the learner
loses more, and zero for an even exchange.

On a normal successful conquest, the direct reward is at least `+1.20`, plus
the ratio and army-trade terms. A first conquest/card, continent completion,
or player elimination adds the corresponding strategic bonus from the table
above. The following `OccupyAction` is separate and can add up to `+1.0` for
moving armies forward.

For example, an attack with six armies against three, where the defender loses
one army, the learner loses none, and the territory is conquered, receives:

```text
ratio:       2.0 x (2.0 - 1.5) = +1.0
army trade:  0.60 x (1 - 0)    = +0.6
conquest:                          +1.2
------------------------------------------
attack reward:                    +2.8
```

`StopAttackAction` is separate: it receives `-2.0` only if a real attack is
still available and the learner has not conquered any territory during that
turn.

The `attack` W&B component excludes the elimination portion; it is logged
separately as `reward_component_eliminate`.

### Occupy

Forward momentum rewards the fraction of movable armies placed in the newly
conquered territory:

```text
1.0 × moved / (armies_at_source_before - 1)
```

### Fortify

- Moving from an interior territory toward a frontier earns a fraction-based
  positive reward; moving from a frontier into an interior territory gets the
  matching negative reward.
- When both source and destination are frontier territories, the reward is a
  negative penalty for deviating from a threat-weighted army balance.
- A continent-push term also applies to the destination.

Skipping fortification gets no fortify shaping.

## End-of-turn shaping

These terms are calculated once per learner turn boundary, not per action.
Their combined value is multiplied by `REWARD_SHAPING_SCALE = 0.3`; terminal
win/loss is not part of this calculation.

- `territory_delta`: learner territory-share change across the opponent round.
- `territory_hold`: `0.05 × territory share after the opponent round`.
- `army_delta`: learner army-share change across the opponent round.
- `continent_delta`: learner continent-bonus-share change across the opponent
  round.
- `continent_lost`: an additional negative penalty when that bonus share
  decreases; losing a high-value continent costs more than losing a
  low-value one.

## W&B diagnostics

`Trainer` accumulates the calculator's components per episode and writes them
as `reward_component_*` fields:

```text
trade_in, reinforce, attack, eliminate, occupy, fortify,
reinforce_ready, reinforce_total, reinforce_continent,
reinforce_interior, reinforce_split,
shaping_raw, shaping_clipped, terminal,
territory_delta, territory_hold, army_delta,
continent_delta, continent_lost
```

These component diagnostics retain their raw, pre-scale values so their
underlying behavior remains easy to inspect. `reward_per_agent_turn` is the
actual replay reward and includes `REWARD_SHAPING_SCALE`.

## Current combined-update changes

The current configuration uses `REWARD_TERRITORY_DELTA = 20.0` and
`REWARD_TERRITORY_HOLD = 0.0`: opponent-round territory gains and losses now
matter, while merely continuing to own territory does not pay repeatedly.

`State.unfinished_attack_targets_this_turn` records distinct enemy targets
attacked without conquest during the current turn and resets when the next
turn begins. On `StopAttackAction`, the old `-2.0` applies once when no
territory was conquered; after a conquest, each tracked unfinished target is
instead `-0.5`. The two penalties never stack. The graph exposes this
per-territory history and `conquered_this_turn`; W&B logs the latter penalty as
`reward_component_unfinished_attack`.

`reward_per_agent_turn` is useful for observing scale, but it is not the
objective. The primary success metric is win rate, with deterministic
evaluation used to remove epsilon-greedy action noise.

The `reward_component_*` metrics are raw, pre-scale totals accumulated over
the complete episode. A component value of `600` therefore does not mean that
one action received `600`, and it has not yet been multiplied by
`REWARD_SHAPING_SCALE`.

## Current interpretation

The latest DQN baseline, run 060, confirms that dense shaping is much larger
than the terminal result over a whole game: mean learner reward was about
`+688` per episode, including roughly `+477` even in lost games. Attack
shaping is the dominant component. A high total reward can therefore describe
active local play rather than a win.

For DQN_104, dense local rewards are multiplied by
`REWARD_SHAPING_SCALE = 0.5` before they reach replay, while terminal reward is
`+500/-500`. Both are five times their DQN_103 values, so their relative
balance is preserved while the reward and TD-target magnitudes increase.
Compare learning speed, win rate, Q/target scale, loss, and gradient clipping
against DQN_103.

DQN_105 uses the intermediate global scale `0.3` with terminal `+300/-300`.
This preserves the same dense-to-terminal ratio while reducing DQN_104's
absolute TD-target and gradient scale by 40%. Its reinforcement formulas are
the marginal correction documented below.

### DQN_104 reinforcement finding

At episode 311, DQN_104 logged `reward_component_reinforce = 879.6` across
691 learner decisions. Its raw reinforcement components were:

| Component | Episode total |
|---|---:|
| Weak-neighbour readiness | `533.9` |
| Whole-frontier strength | `371.2` |
| Continent priority | `9.7` |
| Interior placement | `0.0` |
| Split placement | `-35.2` |

The continent term is not causing the large total. The readiness terms are.
The `-35.2` split total also means that 176 partial reinforcement actions were
scored during the episode because each partial action costs only `-0.20`.

The absolute episode total is enlarged by the long episode, but the reward has
a real repeat-payment problem: each reinforcement action scores the complete
strength of the destination stack after placement. It does not score only the
improvement produced by that placement. A one-army action on an already strong
stack can therefore receive nearly the full readiness reward again, and the
environment permits repeated one/half/all-budget reinforcement actions.

After the global `0.5` scale, the episode's reinforcement shaping contributed
approximately `440` replay reward before interaction with the shared step cap.
That is close to the `+500` terminal win reward. Attack shaping was also large
(`1156.7` raw), so reinforcement is not the only episode-accumulation issue,
but it is the component with a clear repeated-state reward mechanism.

## Current reinforcement reward

The current reward encourages attack-ready frontier placement,
whole-frontier strength, and promising contested continents. It penalizes
interior placement and splitting the visible reinforcement budget.

### Complete formula

This reward does not affect `FortifyAction` or its separate continent-push
reward.

For every destination territory `x` in a `ReinforcementAction`, define:

```text
A_before            = armies on x before this placement
A_after             = armies on x after this placement
E                   = armies on direct enemy neighbours of x
armies_placed       = action.placements[x]
budget_before       = before.reinforcement_budget
c                   = continent containing x
territory_share(c)  = learner territories in c before placement / territories in c
army_share(c)       = learner armies in c before placement / all armies in c before placement
continent_armies(c) = all armies in c before this placement
continent_size(c)   = number of territories in c

ready_score(A, E) =
    max(0, min(A / min(E), REWARD_REINFORCE_READY_CAP)
           - REWARD_REINFORCE_READY_RATIO)

total_score(A, E) =
    max(0, min(A / sum(E), REWARD_REINFORCE_READY_CAP)
           - REWARD_REINFORCE_TOTAL_RATIO)
```

Use these named constants:

| Constant | Value | Meaning |
|---|---:|---|
| `REWARD_REINFORCE_READY_SCALE` | `0.50` | Strength of the reward for readiness against the easiest adjacent enemy. |
| `REWARD_REINFORCE_READY_RATIO` | `1.50` | Readiness improvement starts only above 1.5:1 against the weakest neighbour. |
| `REWARD_REINFORCE_TOTAL_SCALE` | `0.50` | Strength of the additional reward for being ready against the complete adjacent enemy force. |
| `REWARD_REINFORCE_TOTAL_RATIO` | `2.00` | Whole-frontier reward starts only when the stack is at least twice the sum of all directly adjacent enemy armies. |
| `REWARD_REINFORCE_READY_CAP` | `7.00` | Stops both readiness rewards growing beyond an overwhelming 7:1 advantage. |
| `REWARD_REINFORCE_CONTINENT_SCALE` | `5.00` | Strength of the contested-continent preference before placement and continent-size normalization. |
| `REWARD_REINFORCE_INTERIOR` | `-0.80` | Penalty for reinforcing an interior territory that cannot directly support an attack. |
| `REWARD_REINFORCE_SPLIT` | `0.20` | Positive magnitude of the one-time penalty for placing less than the full visible budget; the reward formula applies its negative sign. |

Calculate every reward from this table. Each row contains its complete
condition and formula.

| Reward | When it applies | Full raw formula |
|---|---|---|
| Weak-neighbour readiness | `E` is not empty | `REWARD_REINFORCE_READY_SCALE * (ready_score(A_after, E) - ready_score(A_before, E))` |
| Whole-frontier strength | `E` is not empty | `REWARD_REINFORCE_TOTAL_SCALE * (total_score(A_after, E) - total_score(A_before, E))` |
| Continent priority | `E` is not empty, `A_after / min(E) >= REWARD_REINFORCE_READY_RATIO`, and `0 < territory_share(c) < 1` | `(armies_placed / (continent_armies(c) + armies_placed)) * REWARD_REINFORCE_CONTINENT_SCALE * (territory_share(c) + army_share(c)) / continent_size(c)` |
| Interior placement | `E` is empty | `REWARD_REINFORCE_INTERIOR` |
| Split placement | Applied once per action when `action.total < budget_before` | `-REWARD_REINFORCE_SPLIT` |

For a frontier placement, add the weak-neighbour, whole-frontier, and eligible
continent rewards. For an interior placement, add only the interior reward;
the frontier and continent rewards are zero. After summing all destination
rewards, add the split penalty once for the complete action.

The ready and total terms reward only the improvement caused by the current
placement. Ready improvement is zero while both ratios remain at or below
1.5:1 against the weakest neighbour. Total improvement is zero while both
ratios remain at or below 2:1 against the sum of adjacent enemy armies. Both
potentials stop growing at 7:1, so adding armies to an already capped stack
earns no ready/total reward. The continent reward favors small contested
continents where the learner already has territory and army presence, but it
is available only after the reinforced territory reaches 1.5:1 readiness
against its weakest neighbour.

The zero floor is important: without it, the readiness threshold subtracts
out of the before/after difference and every increase in armies earns reward
even while the stack remains below 1.5:1. For repeated placements on one
territory with unchanged enemies, the summed marginal reward is bounded at
`2.75` ready, `2.50` total, and `5.25` combined for the complete sequence.

The continent scale is `5.00`. On the current board, the smallest contested
continent has four
territories, giving this term a theoretical upper bound below `2.19`. Together
with the maximum weak-neighbour (`2.75`) and whole-frontier (`2.50`) terms, the
maximum reinforcement shaping remains below about `7.44`, leaving room below
the shared `+10` step cap.

The split penalty is constant: leaving one army or most of the budget receives
the same penalty. It is applied once per action, including a custom action with
several destinations. Using the full visible budget receives no split penalty.
The marginal ready and total terms telescope: equivalent split placements on
the same territory with unchanged enemies sum to the same positive ready/total
reward as one full placement. Each partial action still pays the fixed split
penalty. The continent term remains action-local and is not guaranteed to be
split invariant, but DQN_104 showed that it was a very small contribution.

#### Aggregation and scaling

```text
reinforcement_raw =
    sum of all applicable destination rewards
    + one action-level split reward

step_shaping_raw =
    trade_in + reinforcement_raw + attack + occupy + fortify

step_shaping_clipped = clip(
    step_shaping_raw,
    -REWARD_SHAPING_STEP_CAP,
    +REWARD_SHAPING_STEP_CAP,
)

replay_step_reward =
    terminal_reward
    + REWARD_SHAPING_SCALE * step_shaping_clipped
```

All formulas in the table are raw shaping values. The terminal reward is not
scaled or clipped. The common step cap remains the final safety bound.

The combined `reward_component_reinforce` value and the raw contribution of
each term are logged under separate W&B components:

```text
reward_component_reinforce_ready
reward_component_reinforce_total
reward_component_reinforce_continent
reward_component_reinforce_interior
reward_component_reinforce_split
reinforce_action_count
reinforce_partial_action_count
reward_component_reinforce_per_action
reward_component_reinforce_ready_per_action
reward_component_reinforce_total_per_action
```

For every reinforcement action, the five component values must sum to the raw
reinforcement reward before the shared step-shaping cap and global shaping
scale. These fields are diagnostic only and do not alter replay reward.

### DQN_105 validation targets

Exact episode totals cannot be recovered from DQN_104's W&B aggregates because
the individual reinforcement before/after states were not logged. Fresh
DQN_105 should use these forecast and acceptance ranges rather than pretend to
have an exact predicted number:

| Metric | DQN_104 now | Expected after correction | Acceptance guidance |
|---|---:|---:|---|
| Raw reinforce per learner decision | `1.373` | `0.20` to `0.50` | Investigate if it remains above `0.50`. |
| Raw reinforce per episode at DQN_104's mean 435 decisions | `597` | about `90` to `220` | Always interpret with episode length. |
| Actual reinforce per learner decision | about `0.687` at DQN_104's `0.5` scale | `0.06` to `0.15` at DQN_105's `0.3` scale | Should remain clearly below attack. |
| Actual reinforce per equally long episode | about `299` at DQN_104's `0.5` scale | about `27` to `66` at DQN_105's `0.3` scale | Should be well below terminal `300`. |
| Share of positive shaping | `46.7%` | approximately `10%` to `20%` | Above `25%` indicates continued dominance. |
| Partial reinforcement actions per episode | about `133` | should fall materially | Track directly; do not set a fixed target until action counts are logged. |

These ranges assume roughly DQN_104-like episode length and attack behavior.
They are expected calibration ranges, not formula guarantees: the trained
policy and episode length will change in response to the corrected reward. The
mathematical sequence bounds above are the guarantees. Do not reduce
`REWARD_REINFORCE_READY_SCALE` or `REWARD_REINFORCE_TOTAL_SCALE` in the first
correction run. If the marginal formula still leaves reinforcement above 25%
of positive shaping after learning stabilizes, then test lower scales as a
separate experiment.

## Planned reward updates

The sections below remain unimplemented historical proposals. The older
combined-update experiment is recorded in `Docs/ChangeLog.md`; current shared
environment/action-space proposals live in `Docs/EnvironmentActionPlan.md`.

### Planned favorable-attack stop penalty

This plan is **not implemented**. When the learner selects `StopAttackAction`,
inspect the board immediately before that action. Score every distinct enemy
territory that remains legally attackable at a 2:1 or better army ratio. For
enemy territory `y`, use only its best currently available attacking source:

```text
best_ratio(y) = max over owned adjacent attackers x of
    armies[x] / armies[y]

stop_favorable_penalty =
    -REWARD_ATTACK_STOP_FAVORABLE_RATIO_SCALE x
    sum over distinct enemy targets y where best_ratio(y) >= 2.0 of
        best_ratio(y)
```

This is evaluated when the attack phase ends, rather than after fortification
or after opponents play, because only this state still says which attacks the
learner chose to leave unused. It applies whether or not the learner already
conquered another territory during the turn: a successful earlier attack does
not make abandoning a clearly favourable remaining attack desirable.

Thus a 3:1 target receives a larger penalty than a 2:1 target. Each enemy
territory counts once even if several owned territories can attack it; the
strongest available attacker defines that target's ratio. The sum means
leaving several favorable territories unconquered is worse than leaving one.

It must be a distinct, logged component and must not stack with the current
generic `StopAttackAction` penalties: this proportional favorable-attack
penalty replaces the already-**implemented** `-2.0` no-conquest penalty and
the already-implemented `unfinished_attack_targets_this_turn` /
`REWARD_ATTACK_UNFINISHED_TARGET = -0.5`-per-target penalty (see "Current
combined-update changes" above) for that stop — those are live code today,
not a past plan. The shared per-action `[-10, +10]` cap is the final safety
cap on the whole sum. Add focused tests for exactly 2:1, below-2:1,
ratio-proportional growth, multiple attacker sources for one target (count
once), multiple favorable targets (sum), a favorable target after an earlier
conquest, and no double penalty. Choose the scale constant after testing its
raw component distribution.

## Tests and related files

- `risk/learning/reward.py` — implementation.
- `risk/learning/train_constants.py` — current values.
- `Temp/tests/test_reward.py` — unit tests for terminal, phase, and
  end-of-turn terms.
- `risk/learning/trainer.py` — reward accumulation and replay storage.
- `Docs/Trainer.md` — training and W&B metric flow.
