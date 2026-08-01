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

| Condition | Reward | What it teaches |
|---|---|---|
| Frontier placement, compared with its weakest direct enemy neighbour | Proportional reward above 1.5:1; negative below it | First make at least one attack plausible. |
| Frontier placement, compared with all direct enemy-neighbour armies combined | Additional proportional reward above 2:1 | Reward a stack that can dominate its whole local frontier. |
| Frontier reinforcement is in a contested continent | Placement fraction x continent priority | Build strength in small, partly owned continents where the learner already controls a larger share of armies. |
| Destination has no enemy neighbour | Existing `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR = -0.80` penalty | Do not reinforce a pure interior territory. |
| Reinforcement leaves armies in the current reinforcement budget | Small negative reward per unused army | Prefer committing the available force instead of making redundant split placements. |

For a reinforced frontier territory `x`, use its army count after placement
and the armies of its direct enemy neighbours:

```text
A = armies[x] after reinforcement
E = armies of direct enemy neighbours of x

weak_ratio  = A / min(E)
total_ratio = A / sum(E)

weak_neighbour_reward =
    REWARD_REINFORCE_WEAK_NEIGHBOR_SCALE
    x (weak_ratio - 1.5)

total_frontier_reward =
    REWARD_REINFORCE_FRONTIER_SUM_SCALE
    x max(0, total_ratio - 2.0)
```

The first term is negative below 1.5:1 against the weakest neighbour, zero
at 1.5:1, and grows as that ratio improves. It ensures that reinforcement is
first capable of supporting at least one attack. The second term is zero until
the stack reaches 2:1 against the **sum** of all direct enemy-neighbour
armies; it then grows proportionally, so a force that dominates the complete
local frontier gets an additional reward. The shared per-action shaping cap
is the only cap for these ratios.

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

Apply it only if the destination has an enemy neighbour and
`0 < territory_share(c) < 1`. Thus a fully owned continent receives no
continent reinforcement reward, and an interior placement cannot overcome the
no-enemy-neighbour penalty with a positive continent reward. Set
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
REWARD_REINFORCE_WEAK_NEIGHBOR_SCALE      = 0.5
REWARD_REINFORCE_FRONTIER_SUM_SCALE       = 0.5
REWARD_REINFORCE_UNUSED_ARMY_PENALTY      = 0.05
REWARD_REINFORCE_NO_ENEMY_NEIGHBOR        = -0.8  # unchanged
```

The frontier terms use the post-placement army count and are intentionally
continuous rather than one-time threshold bonuses. Test weak-neighbour ratios
below, at, and above 1.5:1; total-frontier ratios below, at, and above 2:1;
the weak-neighbour reward without the total-frontier reward; no enemy
neighbour; unused armies at zero, part, and all of the budget; and the shared
cap. Also test continent size/share ordering, a
weak small continent with only one learner territory,
fully-owned-continent exclusion, and one-placement versus split-placement
totals. Run only as a fresh experiment after DQN 102 concludes.

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
penalty replaces the applicable generic no-conquest or unfinished-target stop
penalty for that stop. The shared per-action `[-10, +10]` cap is the final
safety cap on the whole sum. Add focused tests for exactly 2:1, below-2:1,
ratio-proportional growth, multiple attacker sources for one target (count
once), multiple favorable targets (sum), a favorable target after an earlier
conquest, and no double penalty. Choose the scale constant after testing its
raw component distribution.

### Superseded detailed reinforcement draft

This plan is **not implemented**. It replaces the current
frontier-concentration and weakest-neighbour-ratio rewards with only three
coarse preferences. It must not prescribe attack targets, battle odds, or a
conquest route: those are the DQN's policy to learn from Q-values and the
existing attack, conquest, and terminal rewards.

| Reinforcement situation | Proposed shaping | Purpose |
|---|---|---|
| A frontier territory crosses from unready to ready against its strongest direct enemy neighbour | One small `REWARD_REINFORCE_LAUNCH_READY` bonus | Prefer concentration until a usable attack stack exists. |
| Two territories both cross the readiness threshold | One bonus for each territory | Splitting is allowed when it creates two real attack launches. |
| A placement leaves every affected frontier stack unready | No launch-ready bonus | Do not reward dividing armies before either attack is viable. |
| Placement is in a contested continent (`0 < territory_share < 1`) | Small change in `territory_share x army_share` | Prefer building military presence where the learner already has a plausible continent conquest. |
| Placement is in a fully owned continent | No reinforcement continent-progress reward | Do not pay merely for adding armies where there is nothing in that continent left to conquer. |
| Destination has no enemy neighbour | Existing `REWARD_REINFORCE_NO_ENEMY = -0.80` penalty | Discourage interior reinforcement with no immediate attack option. |
| Target, attack order, dice risk, future route, or exact battle odds | No reinforcement-specific formula | Leave these tactical choices to the DQN. |

#### 1. Do not split an unready attacking force

Give a small, one-time `launch-ready` bonus when reinforcement changes an
owned frontier territory from unready to ready. Readiness is deliberately a
simple direct-neighbour threshold:

```text
readiness_ratio(x, state) =
    armies[x] / max(armies[y] for adjacent enemy y)

launch_ready(x, state) = readiness_ratio(x, state) >= readiness_threshold

launch_reward = REWARD_REINFORCE_LAUNCH_READY
    when launch_ready changes from false to true
```

Start with `readiness_threshold = 1.5`, the point where the current attack
ratio term stops being negative. Choose the bonus magnitude only when the
revision is implemented and tested. There is no accumulating ratio,
battle-probability, or route-value score.

Two incomplete stacks receive no bonus; concentrating enough force on one
stack receives one bonus. Two bonuses are possible only when both stacks are
actually ready, which allows two attacks. A strong direct neighbour prevents
the territory from being ready even if another neighbour is weak. This is
Markov (it uses `before` and `after`) and cannot be farmed by splitting one
allocation into several actions.

#### 2. Reinforce a continent that is being conquered

For each contested continent `c`, use only its territory share and army share:

```text
territory_share(c) = learner territories in c / territories in c
army_share(c)      = learner armies in c / all armies in c
continent_progress(c) = territory_share(c) x army_share(c)

reinforcement_continent_reward = REWARD_REINFORCE_CONTINENT_PROGRESS x
    (continent_progress(c, after) - continent_progress(c, before))
```

Apply this only when `0 < territory_share(c) < 1`. It is a small bias toward
strengthening a plausible partly owned continent, not a rule that selects a
continent or attack target. The potential difference is split-action
invariant; the DQN decides whether concentrating there is worth it.

#### 3. Reinforce where an attack can be launched

The launch-ready bonus provides the positive signal. Retain
`REWARD_REINFORCE_NO_ENEMY = -0.80` when a destination has no enemy neighbour,
so interior placement remains discouraged. Do not add battle-win probability,
total hostile-army, frontier-allocation, or future-route terms: they are too
heuristic for shaping and should remain learnable policy decisions.

Keep the shared per-action `[-10, +10]` shaping cap and
`REWARD_SHAPING_SCALE = 0.1`. Test threshold crossing, sub-threshold
splitting, two genuinely ready stacks, the no-enemy penalty, contested
continent progress, fully-owned-continent exclusion, and split-action
invariance of the continent potential. Run only as a fresh experiment after
DQN 102 concludes.

### Superseded launch-value proposal

This plan is **not implemented**. It replaces only the frontier-concentration
and weakest-neighbour-ratio terms with one Markov launch-value term; the
continent-push term is kept (with a correctness fix, below). The launch value
does not depend on the reinforcement budget at the start of a turn: that
value is not represented in `State`, so using it would make the reward depend
on hidden history.

The frontier-concentration term is removed for two related reasons. Its
current denominator is the **remaining** budget, so the same allocation earns
more reward when split into several actions as the denominator shrinks. A
fixed turn-start denominator would remove that split incentive, but it is not
visible to the agent in the current state representation and would therefore
make the learning problem non-Markov. A fixed ratio against only the weakest
enemy is also insufficient: a stack may easily beat a one-army neighbour yet
be unable to continue through stronger enemy territories adjacent to its
launch point. The replacement below measures both immediate attack quality
and the total local hostile force.

For an owned territory `x` with adjacent enemy territories, define:

```text
best_battle(x, state) = max over adjacent enemies y of
    battle_win_probability(armies[x], armies[y])

frontier_security(x, state) = clamp(
    (armies[x] - 1) / sum(adjacent enemy armies),
    0,
    1,
)

launch_value(x, state) =
    best_battle(x, state) x frontier_security(x, state)

reinforcement_launch_reward = REWARD_REINFORCE_LAUNCH_VALUE_SCALE x (
    launch_value(x, after) - launch_value(x, before)
)
```

`battle_win_probability` is the exact full-force one-territory battle
probability already used by the heuristic agents. `frontier_security` uses
the **sum** of all direct enemy-neighbour armies, so an easy one-army target
cannot by itself make a launch stack look ready when the same territory also
faces stronger enemies. `REWARD_REINFORCE_LAUNCH_VALUE_SCALE` is a new
constant whose value must be chosen and documented with the implementation.

The difference between after and before launch value is essential. It is
Markov because both states are already inputs to reward calculation, and it
makes the total reward independent of how one allocation is split across
actions on the same territory: the intermediate launch values cancel. The
learner may therefore split armies when that is strategically useful, but
cannot earn extra reward merely for dividing a placement.

This is intentionally a *local* launch signal, not a hand-coded conquest
route planner. A later enemy territory not directly adjacent to `x` depends
on dice, occupation choices, and the next state after conquest. The DQN must
learn that multi-territory continuation through its Q-values and the existing
attack, conquest, and terminal rewards; reinforcement shaping should not
pretend to know the whole future route.

When a reinforced territory has no adjacent enemy, retain the existing
`-0.80` interior-reinforcement penalty. Continent-push still applies to every
placement, but the "already fully owned" fix applies to **reinforcement
only**, not to `FortifyAction`. The two phases want opposite behavior once a
continent is fully owned: reinforcement should stop being rewarded for
placing into a continent with nothing left to conquer, but fortify should
keep being rewarded for concentrating strength there, since holding an
already-completed continent against recapture (and its ongoing bonus) is
itself a goal. `_continent_push` currently scales with `owned / total`, which
peaks — rather than stops — once `owned == total`; that peak-at-full-ownership
shape is correct for fortify and should not change. For reinforcement, add a
guard so the term returns `0.0` once the continent is already fully owned by
the learner before the placement, without touching the formula or behavior
used by `FortifyAction`. Concretely, give `_continent_push` an
`allow_completed_continent` parameter (or an equivalent split into two
call sites) so reinforcement and fortify can diverge without duplicating the
`owned / total` calculation.

Keep the shared per-action `[-10, +10]` shaping cap and
`REWARD_SHAPING_SCALE = 0.1`. Add focused reward tests for the no-enemy
penalty, exact battle-probability improvement, total-adjacent-enemy security,
a weak neighbour beside stronger direct threats, split-action invariance on a
single territory, reinforcement's continent-push returning `0.0` once a
continent is fully owned, and a regression test that `FortifyAction`'s
continent-push is unchanged (still positive and peaking at full ownership).
Run this as a new, fresh experiment only after the current DQN 102 experiment
has concluded.

## Tests and related files

- `risk/learning/reward.py` — implementation.
- `risk/learning/train_constants.py` — current values.
- `Temp/tests/test_reward.py` — unit tests for terminal, phase, and
  end-of-turn terms.
- `risk/learning/trainer.py` — reward accumulation and replay storage.
- `Docs/Trainer.md` — training and W&B metric flow.
