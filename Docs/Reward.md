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
step_reward = terminal + step_shaping
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
reward above, plus end-of-turn reward when that boundary is reached.

There is no terminal reward for a `MAX_STEPS_PER_EPISODE` timeout: it is a
truncation, not a terminal game state, so its replay transition keeps
`done=False` and may bootstrap.

## Current constants

| Area | Constants | Values |
|---|---|---|
| Terminal | win / loss | `+100`, `-100` |
| Step safety | shaping cap | `±10` |
| Trade-in | early / territory match | `0.30`, `0.60` |
| Reinforce | concentration / attack readiness / ratio cap / no enemy / continent push | `1.20`, `1.50`, `2.50`, `-0.80`, `0.90` |
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
shaping** is clipped to `[-10, +10]`. Terminal and end-of-turn terms are
separate.

| Action | Rewarded behavior / formula | Main scale |
|---|---|---:|
| `SkipTradeAction` | Optional three/four-card trade: `+0.30 x hand_factor / card_set_value` | small positive |
| `TradeInAction` | Optional trade: negative of skip term; `+0.60` if the set matches an owned territory card | small |
| `ReinforcementAction` | Frontier concentration: `+1.20 x placed / remaining budget`; attack readiness: `+1.50 x (min(after / weakest enemy, 2.5) - 1)`; no enemy neighbour: `-0.80`; plus continent push | mixed |
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

For each placement:

- Placing on a frontier rewards the fraction of the remaining budget placed
  there.
- If an enemy neighbour exists, attack readiness is based on
  `armies_after / weakest_adjacent_enemy_armies`, capped at `2.5`, then
  shifted by `-1`.
- Placing where there is no enemy neighbour gets the no-enemy penalty.
- A small continent-push term rewards placing into a continent where the
  learner already owns a larger fraction.

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
shaping_raw, shaping_clipped, terminal,
territory_delta, territory_hold, army_delta,
continent_delta, continent_lost
```

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

## Current interpretation

The latest DQN baseline, run 060, confirms that dense shaping is much larger
than the terminal result over a whole game: mean learner reward was about
`+688` per episode, including roughly `+477` even in lost games. Attack
shaping is the dominant component. A high total reward can therefore describe
active local play rather than a win.

This is a reason to measure reward components alongside win rate, not a reason
to change several constants at once. Reward experiments should change one
clear hypothesis, retain the same DQN baseline and opponent setup, and compare
win rate, loss-game reward, Q/target scale, and gradient clipping.

## Planned reward updates

All unimplemented reward changes, their reasoning, exact code work, tests,
and experiment sequence are maintained in
[`Docs/Update_Plan.md`](Update_Plan.md). This file documents
only the current implemented reward system.

## Tests and related files

- `risk/learning/reward.py` — implementation.
- `risk/learning/train_constants.py` — current values.
- `Temp/tests/test_reward.py` — unit tests for terminal, phase, and
  end-of-turn terms.
- `risk/learning/trainer.py` — reward accumulation and replay storage.
- `Docs/Trainer.md` — training and W&B metric flow.
