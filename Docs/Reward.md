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
step_reward = terminal + 0.1 x step_shaping
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
reward above, plus `0.1 x` end-of-turn reward when that boundary is reached.

There is no terminal reward for a `MAX_STEPS_PER_EPISODE` timeout: it is a
truncation, not a terminal game state, so its replay transition keeps
`done=False` and may bootstrap.

## Current constants

| Area | Constants | Values |
|---|---|---|
| Terminal | win / loss | `+100`, `-100` |
| Dense shaping | scale (terminal excluded) | `0.1` |
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
shaping** is clipped to `[-10, +10]` and multiplied by
`REWARD_SHAPING_SCALE = 0.1`. Terminal reward is not scaled. End-of-turn
shaping is separately calculated and multiplied by the same scale.

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
Their combined value is multiplied by `REWARD_SHAPING_SCALE = 0.1`; terminal
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

## Current interpretation

The latest DQN baseline, run 060, confirms that dense shaping is much larger
than the terminal result over a whole game: mean learner reward was about
`+688` per episode, including roughly `+477` even in lost games. Attack
shaping is the dominant component. A high total reward can therefore describe
active local play rather than a win.

The dense local rewards are now multiplied by `REWARD_SHAPING_SCALE = 0.1`
before they reach replay, while terminal `+100/-100` remains unchanged. This
makes winning or losing dominate an episode's accumulated local shaping without
increasing TD-target magnitude. Compare win rate, loss-game reward, Q/target
scale, and gradient clipping against the previous baseline in a fresh run.

## Planned reward updates

All unimplemented reward changes, their reasoning, exact code work, tests,
and experiment sequence are maintained in
[`Docs/Update_Plan.md`](Update_Plan.md). This file documents
only the current implemented reward system.

### Planned reinforcement-shaping revision — simple draft

This plan is **not implemented**. It has three pieces: continent priority,
frontier readiness, and an interior-placement penalty.

#### Implementation policy summary

This replaces the current reinforcement concentration, capped readiness, and
continent-push rewards.  It is deliberately a small set of preferences, not a
script for the DQN: the agent must still learn which continent to pursue,
which frontier to reinforce, which enemy to attack, and when splitting force
is worthwhile.

For every `ReinforcementAction`, calculate the following from the destination
and the board state immediately before/after that one placement:

| Situation | Planned shaping | Policy expressed |
|---|---|---|
| Frontier with one or more enemy neighbours | Signed readiness against the weakest direct enemy: `weak_scale x (min(A / min(E), 7) - 1.5)` | Make at least one nearby attack viable; a larger advantage earns more up to 7:1. |
| Same frontier | Additional reward only above local-frontier dominance: `sum_scale x max(0, min(A / sum(E), 7) - 2.0)` | A strong stack that can cover its whole direct frontier is better, without rewarding overstacking beyond 7:1. |
| Same frontier, at least 1.5:1 ready, contested continent | `placement_fraction x continent_priority` | Prefer small continents in which the learner already has territory and army presence, so a conquest is plausible. |
| Destination has no enemy neighbour | `-0.8`; none of the positive terms above apply | Do not place armies in an interior territory that cannot immediately support an attack. |
| The action leaves part of the visible reinforcement budget | `-unused_scale x (budget_before - armies_placed)` | Put all currently available armies in one strong place rather than split them across redundant decisions. |

Here `A` is the destination army count **after** placement and `E` is the
list of its direct enemy-neighbour army counts.  The continent term applies
only when the destination is a frontier and the continent is contested
(`0 < territory_share < 1`) **and** `A / min(E) >= 1.5`; a fully owned
continent or a not-yet-ready frontier gets no reinforcement continent reward.
This preserves the policy that a placement which does not make any direct
attack viable remains negative. The unused-budget term is evaluated on every placement:
placing the entire remaining budget has zero penalty, while `30 -> [15, 15]`
is penalized on the first action and `30 -> [30]` is not.  The common
`[-10, +10]` step-shaping cap remains the final safety bound.

None of this touches `FortifyAction`. Its existing `_continent_push` term
(which rewards concentrating into an already-fully-owned continent, to help
hold it against recapture) is a separate calculation and is left exactly as
implemented today; this plan only replaces reinforcement's terms.

For a reinforced frontier territory `x`, use its army count after placement
and the armies of its direct enemy neighbours:

```text
A = armies[x] after reinforcement
E = armies of direct enemy neighbours of x

weak_ratio  = A / min(E)
total_ratio = A / sum(E)

weak_neighbour_reward =
    REWARD_REINFORCE_WEAK_NEIGHBOR_SCALE
    x (min(weak_ratio, 7.0) - 1.5)

total_frontier_reward =
    REWARD_REINFORCE_FRONTIER_SUM_SCALE
    x max(0, min(total_ratio, 7.0) - 2.0)
```

The first term is negative below 1.5:1 against the weakest neighbour, zero
at 1.5:1, and grows as that ratio improves up to 7:1. It ensures that
reinforcement is first capable of supporting at least one attack. The second
term is zero until the stack reaches 2:1 against the **sum** of all direct
enemy-neighbour armies; it then grows proportionally up to 7:1, so a force
that dominates the complete local frontier gets an additional reward without
paying for armies beyond that readiness. The shared per-action shaping cap is
still the final safety bound.

If `E` is empty, do not calculate either positive term; apply only the
existing `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR` negative reward.

For each placement into a **frontier** territory in a contested continent `c`,
calculate the following from the board **before** that placement:

```text
continent_priority(c) =
    REWARD_REINFORCE_CONTINENT_PRIORITY_SCALE
    x (territory_share(c) + army_share(c))
    / territory_count(c)

placement_fraction(c) =
    armies_placed
    / (total_armies_in_continent(c) + armies_placed)

continent_reward =
    placement_fraction(c)
    x continent_priority(c)
```

Apply it only if the destination has an enemy neighbour,
`A / min(E) >= 1.5`, and `0 < territory_share(c) < 1`. Thus a fully owned
continent or a not-yet-ready frontier receives no continent reinforcement
reward, and an interior placement cannot overcome the no-enemy-neighbour
penalty with a positive continent reward. Set
`REWARD_REINFORCE_CONTINENT_PRIORITY_SCALE = 10.0` for the first experiment;
it is an overall scale, not a weight favoring armies over territories.
Dividing by the continent's territory count favors small continents; high army
share makes a small, weakly defended enemy group attractive even if the
learner owns only one territory there. The bounded placement fraction prevents
a large early reinforcement into a nearly empty continent from becoming
arbitrarily large. The shared per-action shaping cap remains the final safety
bound.

This is not a tactical assessment: it says only "we already own territory and
armies here." The DQN still chooses the continent, target, attack order, dice
risk, and any multi-territory route. No battle probability, total-enemy-army,
frontier-allocation, or route formula is used.

After the continent and frontier terms, add a small non-concentration penalty:

```text
unused_armies = before.reinforcement_budget - action.total

unused_reinforcement_penalty =
    -REWARD_REINFORCE_UNUSED_ARMY_PENALTY x unused_armies
```

Using the entire remaining budget has zero penalty. Placing only part of it
and returning to reinforcement later is negative in proportion to the armies
left unused, so the learner cannot gain extra reward from redundant split
placements. This is Markov: the pre-action reinforcement budget is already in
the state and `action.total` is the proposed action amount. The normal legal
action set contains one destination per reinforcement action, so choosing the
full budget also concentrates it on that destination. If multi-destination
reinforcement actions are later exposed as ordinary legal candidates, add a
separate destination-count term rather than assuming this budget penalty alone
measures geographic concentration.

Implement this plan as a **replacement** for the current reinforcement
concentration, capped weakest-neighbour-ratio, and continent-push terms; do
not add the new terms on top of them. The agreed initial constants are:

```text
REWARD_REINFORCE_CONTINENT_PRIORITY_SCALE = 10.0
REWARD_REINFORCE_WEAK_NEIGHBOR_SCALE      = 0.50
REWARD_REINFORCE_FRONTIER_SUM_SCALE       = 0.50
REWARD_REINFORCE_FRONTIER_RATIO_CAP       = 7.0   # new name; distinct from the
                                                   # currently implemented
                                                   # REWARD_REINFORCE_RATIO_CAP
                                                   # (2.5), which this plan
                                                   # removes entirely
REWARD_REINFORCE_UNUSED_ARMY_PENALTY      = 0.05
REWARD_REINFORCE_NO_ENEMY_NEIGHBOR        = -0.8  # unchanged
```

`REWARD_REINFORCE_FRONTIER_RATIO_CAP` is deliberately a new constant name, not
a reuse of the existing `REWARD_REINFORCE_RATIO_CAP = 2.5`: the two cap
different terms at different scales (`2.5` shifted by `-1.0` today vs. `7.0`
shifted by `-1.5`/`-2.0` here), and reusing the name risks a stale reference
to the old semantics surviving the migration.

The 7:1 ratio ceiling keeps ordinary strong placements below the shared
`[-10, +10]` cap while preserving meaningful growth from 1.5:1 through 7:1.
It is a tactical readiness ceiling, not a second reward cap: further armies
may still be strategically useful, but reinforcement shaping stops paying for
an already overwhelming direct advantage. Seven-to-one leaves room to prepare
a stack for a stronger next target after taking an adjacent weak territory.
The frontier terms use the post-placement army count and are intentionally
continuous rather than one-time threshold bonuses. Test weak-neighbour ratios
below, at, and above 1.5:1; exactly 7:1 and above 7:1; total-frontier ratios below, at, and above
2:1; the weak-neighbour reward without the total-frontier reward; no enemy
neighbour; unused armies at zero, part, and all of the budget; and the shared
cap. Also test continent size/share ordering, a weak small continent with only
one learner territory, the not-yet-ready and fully-owned-continent exclusions,
and one-placement versus split-placement totals. Run only as a fresh
experiment after DQN 102 concludes.

#### Worked calibration examples

The values below are raw reinforcement shaping, followed by the replay value
after the global `REWARD_SHAPING_SCALE = 0.1`. They use the initial constants
above. `C` denotes the already-calculated continent reward where shown; it is
zero below the 1.5:1 readiness threshold.

| Case | Inputs and components | Raw -> replay |
|---|---|---:|
| Interior, full budget | no enemy neighbour: `-0.8` | `-0.80 -> -0.080` |
| Interior, split | no enemy; budget `20`, placed `10`: `-0.8 - 0.05 x 10` | `-1.30 -> -0.130` |
| Unready frontier | `A=3`, `E=[4, 5]`: `0.50 x (3/4 - 1.5)`; no `C` | `-0.375 -> -0.038` |
| Exactly ready | `A=6`, `E=[4, 8]`, `C=0.375`: weak and sum terms are zero | `+0.375 -> +0.038` |
| Ready against weakest only | `A=8`, `E=[4, 8]`, `C=0.375`: `0.50 x (2 - 1.5)` | `+0.625 -> +0.063` |
| Dominates weakest, reaches total threshold | `A=16`, `E=[4, 4]`, `C=0.375`: `0.50 x (4 - 1.5)`; total term zero at exactly `2:1` | `+1.625 -> +0.163` |
| Dominates the whole local frontier | `A=20`, `E=[4, 4]`, `C=0.375`: weak `0.50 x (5 - 1.5) = +1.750`, sum `0.50 x (2.5 - 2)` | `+2.375 -> +0.238` |
| Small, promising two-territory continent | `A=15`, `E=[1]`, `C=2.167`: weak `+2.750`, sum `+2.500` (both at 7:1 cap) | `+7.417 -> +0.742` |
| Good target but split placement | `A=16`, `E=[4,4]`, `C=0.500`, budget `30`, placed `15`: `+1.250 - 0.750 + C` | `+1.000 -> +0.100` |
| Same stack, tiny first split | `A=16`, `E=[4,4]`, `C=0.050`, budget `30`, placed `1`: `+1.250 - 1.450 + C` | `-0.150 -> -0.015` |
| Strong stack against three neighbours | `A=45`, `E=[5,5,5]`, `C=0.500`: weak `+2.750` (7:1 cap), sum `+0.500` | `+3.750 -> +0.375` |
| Extreme overstack: no extra reward beyond 7:1 | `A=62`, `E=[1]`, no `C`: weak `0.50 x (7 - 1.5)`, sum `0.50 x (7 - 2)` | `+5.250 -> +0.525` |

For the examples with `C=0.375`, the continent has four territories, the
learner owns two and 40% of its armies, has 30 total armies before placement,
and places six armies: `continent_priority = 10 x (0.5 + 0.4) / 4 = 2.25`
and `placement_fraction = 6 / (30 + 6)`, hence `C = 0.375`.

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

### Superseded reinforcement-shaping drafts

Two earlier drafts were superseded by the "simple draft" above. Per this
file's own policy (see the top of this section), superseded proposals are not
reproduced here in full — their complete formulas remain available in
`Docs/ChangeLog.md` and git history:

- **Detailed reinforcement draft.** A one-time `launch-ready` bonus on
  crossing a readiness threshold against the territory's *strongest* direct
  enemy neighbour, plus a separate `territory_share x army_share`
  continent-progress delta. Superseded because the simple draft's continuous
  weak-neighbour and sum-of-frontier terms reward partial progress toward
  readiness rather than only the crossing instant, and it separates the
  concentration incentive into its own explicit unused-budget penalty instead
  of relying on a one-time bonus's shape to discourage splitting.
- **Launch-value proposal.** A single Markov delta term,
  `battle_win_probability(A, weakest enemy) x frontier_security(sum of
  enemies)`. Superseded because it depends on the full-force battle-
  probability table that reward shaping otherwise avoids, and it folds
  weakest-neighbour and whole-frontier readiness into one multiplicative
  term where the simple draft's two additive terms are easier to test and
  calibrate independently. Its one useful correction — that reinforcement's
  continent term, not `FortifyAction`'s, should stop paying once a continent
  is fully owned — carried forward and is now stated directly in the "simple
  draft" section above.

## Tests and related files

- `risk/learning/reward.py` — implementation.
- `risk/learning/train_constants.py` — current values.
- `Temp/tests/test_reward.py` — unit tests for terminal, phase, and
  end-of-turn terms.
- `risk/learning/trainer.py` — reward accumulation and replay storage.
- `Docs/Trainer.md` — training and W&B metric flow.
