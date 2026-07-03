# Reward design (implemented)

**Current state: fully implemented.** `Environment.step(...)` delegates all
reward computation to `RewardCalculator.compute(...)`
(`risk/learning/reward.py`), called from inside `step(...)` with no reward
arithmetic left in `environment.py` itself. `RewardCalculator` combines the
sparse terminal term (`REWARD_TERMINAL_WIN`/`REWARD_TERMINAL_LOSS`) with the
full set of per-phase dense shaping terms described below, gated so every
phase helper only scores the *current actor's own* decision
(`before.current_player_index == reward_player`) — `Trainer` calls
`Environment.step(...)` with `reward_player=<learner's seat>` for every
actor's turn, not just the learner's own, so without this gate an
opponent's action would get scored as if the learner had taken it.

`REWARD_TERMINAL_LOSS` fires on that player's own elimination too, even if
the game isn't over for everyone else yet — elimination is a real loss for
that player regardless. There is deliberately no terminal reward for a
`MAX_STEPS_PER_EPISODE` timeout (`trainer.py`'s episode loop, counted as
total environment steps including opponent actions): a timeout truncates
the episode without the underlying game state ever becoming
terminal, so it must never be paired with `done=True` — that would tell the
TD bootstrap target "nothing follows this state," which is false, and would
corrupt training. `trainer.py` gets this right: it just stops calling
`step()` when the cap is hit, leaving `done=False` on the last transition.

The end-of-turn opponent-impact terms (`REWARD_TERRITORY_DELTA`,
`REWARD_ARMY_DELTA_RELATIVE_SCALE`, `REWARD_CONTINENT_DELTA_RELATIVE`) are
the one piece `compute()` can't produce by itself — they need the state
right when the learner's `FortifyAction` is taken vs. the state once every
opponent has played their full turn, and no single `step()` call ever sees
both ends of that span. `Trainer.train(...)` calls
`Environment.reward.end_of_turn(...)` directly when the learner's own action
is a `FortifyAction`, adding the result into the transition's stored reward.

`REWARD_ATTACK_STOP_WITHOUT_CARD` needed turn-level memory ("was a card
drawn yet this turn"), now exposed as `State.conquered_this_turn` (set in
`Environment._apply_attack`, reset in `_begin_turn_for`) rather than a
private `Environment` attribute, so `RewardCalculator` can read it from a
`State` snapshot like everything else.

## Per-component logging (implemented)

`RewardCalculator.compute(...)` and `.end_of_turn(...)` still return the
same clipped float they always did — training behavior is unchanged — but
each call also stashes its term-by-term breakdown on
`self.last_components` (`trade_in`, `reinforce`, `attack`, `occupy`,
`fortify`, `shaping_raw`, `shaping_clipped`, `terminal`) and
`self.last_end_of_turn_components` (`territory_delta`, `army_delta`,
`continent_delta`) respectively — diagnostic-only, read after the fact.

`Trainer.train(...)` accumulates these into a per-episode running total
(`Trainer._accumulate_reward_components`) after every `env.step(...)`/
`env.reward.end_of_turn(...)` call in the episode loop, then logs each as
`reward_component_<name>` alongside the existing episode metrics
(`win`, `reward_per_agent_turn`, etc.) via `TrainingLogger.log_episode(...)`.
Comparing cumulative `reward_component_shaping_clipped` (and the terminal
`reward_component_terminal`, effectively `win`'s ±100) across episodes is
what lets a bad shaping constant — or the dense terms simply outgrowing the
one-time terminal reward as episodes get longer — show up directly in
training curves instead of only being inferred from behavior.

### Transition-level semantics (what actually gets remembered)

The intended behavior is:

- During the learner's own turn, each learner action is scored at that step:
  reinforce decisions get reinforce shaping, attack decisions get attack
  shaping, occupy/fortify decisions get their own phase shaping.
- The turn-ending learner action (`FortifyAction`, including skip) gets one
  additional end-of-turn term comparing:
  - `before_turn`: state when that fortify was taken,
  - `after_turn`: state after all opponents have played and control returns.

`Trainer` stores one replay transition per learner decision. For
non-turn-ending learner actions, that transition reward is effectively that
single learner-step reward. For turn-ending fortify, the stored reward is:

```
fortify_step_reward + end_of_turn(before_turn, after_opponent_round)
```

This matches the design goal: dense shaping on each learner decision, plus
one cross-round comparison term only at the learner's turn boundary.

Every value mentioned below is a constant in
`risk/learning/train_constants.py` (so it's tunable without touching
engine/trainer code). The rest of this doc is now also a reference for how
each term is computed — see `risk/learning/reward.py` for the actual code.

## Design principles to agree on first

1. **Sparse stays the anchor.** Win/lose/timeout keep being the dominant
  signal (large terminal magnitude, see values below). Dense shaping terms should be small enough per-step that
   they nudge behavior without letting the agent "farm" shaping reward
   instead of trying to win (classic reward-hacking risk — e.g. if
   conquering is rewarded richly enough, the agent may prefer drawn-out
   territory trading over closing out the game).
2. **Computed once, in the engine.** `Environment.step(...)`'s `_apply_*`
   methods already build an `info` dict per action (e.g. `_apply_attack`'s
   `conquered`/`eliminated`/`attacker_losses`/`defender_losses`). Shaping
   terms should read from that `info` plus before/after `State`, inside
   `step(...)`, not be recomputed ad hoc in the trainer — keeps one source
   of truth and means any future agent gets the same reward function for
   free.
3. **Symmetric bookkeeping, not just self.** Reward is computed for
   `reward_player`; shaping terms about *that player's own* territory/army
   changes are straightforward. Terms about *opponents* (e.g. "I eliminated
   someone") need to read `info["eliminated"]` etc., not just before/after
   diffs of `reward_player`'s own state.
4. **Every constant lives in `train_constants.py`.** No magic numbers in
   `environment.py` itself — the engine reads named constants so a tuning
   pass never touches game-rule code.
5. **No global-survivability gate on continent terms — keep shaping local.**
  An earlier draft of this doc gated every `*_CONTINENT_*` term behind a
  player-wide "is my weakest frontier territory dangerously outmatched"
  check. Dropped: shaping reward is a heuristic nudge, not a correctness
  guarantee, and that gate required scanning *every* frontier territory the
  player owns (not just the one the current action touches) to decide
  whether to suppress a single term — real complexity for a signal that
  doesn't need to be globally accurate. The network is expected to learn
  the actual "continent push vs. defend elsewhere" tradeoff itself from the
  sparse terminal reward plus the already-local terms
  (`REWARD_REINFORCE_ATTACK_READINESS`, `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR`,
  `REWARD_FORTIFY_BALANCE`'s own two-territory `frontier_threat`) — none of
  which need a global scan, since they only ever look at the territory/ies
  the action directly involves. Continent terms now apply unconditionally;
  no suppression, no extra gating constant.
6. **Composition rule is additive, with one shaping safety cap.** For each
  transition, sum all active shaping terms (respecting each term's own
  gates/mutual-exclusion rules), then clip that shaping subtotal to a
  configurable per-step bound. Add sparse terminal reward separately so
  win/loss remains the anchor signal:
  ```
  shaping_raw = sum(active_shaping_terms)
  shaping = clip(shaping_raw, -REWARD_SHAPING_STEP_CAP, REWARD_SHAPING_STEP_CAP)
  reward = sparse_terminal_reward + shaping
  ```
  This keeps the implementation simple ("add all applicable rewards") while
  preventing rare multi-trigger stacks from dominating learning.
7. **Step-range target.** Non-terminal step rewards are normalized via
  `REWARD_SHAPING_STEP_CAP` to stay in `[-10, 10]`. Terminal win/loss
  rewards are intentionally outside this range.

## Where each phase even has something to reward

Risk's 5 agent-facing phases (`Phase`, `risk/game/phase.py`) differ a lot in
what's worth shaping:

### `TRADE_IN`
`card_set_value(trade_in_index)` (`risk/constants.py`) is keyed to the
*global* trade-in count (`state.cards_traded_in_count`), not the trading
player's hand size. Its progressive value is capped at 50 armies to stop
very long games from exploding reinforcement budgets. Since opponents trading in sets also advances that
counter, waiting to trade is often a real edge — your own eventual trade
lands at a higher, more valuable index. So "reward trading, discourage
hoarding" would be backwards. Instead:

- **`REWARD_TRADE_IN_EARLY`** — applied on every `TRADE_IN` decision where
  `SkipTradeAction` is actually legal (hand size 3 or 4 —
  `MAX_CARDS_IN_HAND` forces a real trade at 5, where `SkipTradeAction`
  isn't even offered by `legal_actions()`, `environment.py:203`):
  - **Positive** when the agent chooses to skip (patience, riding the
    rising global trade-in index).
  - **Negative** when the agent chooses to trade anyway, cashing in early
    when it didn't have to. Scale the magnitude by how far from forced the
    hand was — hand=3 is the most avoidable trade (bigger penalty/bigger
    skip-reward), hand=4 is closer to forced regardless (smaller of each).
  - **Policy note (logic, not values):** default bias is still "wait when
    optional," but only when global trade-in value is still in its low/early
    regime (`state.cards_traded_in_count` / `card_set_value(...)`). Once the
    game is in a higher-value midgame regime, weaken or disable this
    wait-bias so optional early trade is no longer penalized by rule.
  - **Formula sketch:** use the current trade value `v = card_set_value(...)
    `, base constant `C = REWARD_TRADE_IN_EARLY`, and a hand-size factor
    `f(h)` so the wait bias automatically weakens as the trade value grows:
    ```
    r_early = C * f(h) / v

    f(h) = 1.0  if hand size == 3
         = 0.5  if hand size == 4
         = 0.0  if trade is forced
    ```
    `SkipTradeAction` gets `+r_early`, `TradeInAction` gets `-r_early`, and
    forced trade gets `0`. This keeps the rule strong in the early game and
    naturally weaker in the midgame, without needing to predict `v_next`.
  - `0`/not applicable at hand=5 — forced, not a real decision, so
    rewarding either choice there teaches nothing.
- **`REWARD_TRADE_IN_TERRITORY_MATCH`** — flat positive, applied whenever
  the agent *does* trade (forced or not) and the chosen set includes ≥1
  card for a territory it currently owns (`CARD_TERRITORY_BONUS_ARMIES`).
  Independent of `REWARD_TRADE_IN_EARLY` — rewards picking the better of
  several legal sets, a distinct sub-decision from whether to trade at all.

### `REINFORCE_PLACE`
Territory/army *count* doesn't change during this phase (placement only
moves armies the player already earned), so no count-delta term belongs
here — that belongs to `ATTACK`. What's worth shaping is *where* and *how*
the budget gets placed:

- **`REWARD_REINFORCE_CONCENTRATION`** — positive, scaled by `placed_amount
  / remaining_budget_before_this_action`, gated to frontier territories
  only (placing the full remaining budget in one action on one frontier
  territory scores highest; splitting it scores proportionally less).
  `_legal_reinforce` already offers a full-budget placement as one of its
  discretized options, so this rewards picking that option over spreading.
- **`REWARD_REINFORCE_ATTACK_READINESS`** — replaces the earlier, simpler
  "is there an enemy neighbor at all" framing with a sharper one: compare
  post-placement army count against the *weakest* adjacent enemy
  territory's army count (not strongest, not average) — if you can't beat
  even the easiest target, the territory isn't attack-ready regardless of
  whether that's because there's no enemy neighbor or every neighbor
  outguns you. Only computed when ≥1 adjacent enemy territory exists:
  ```
  ratio = armies_after_placement / weakest_adjacent_enemy_armies
  reward = REWARD_REINFORCE_ATTACK_READINESS_SCALE * (min(ratio, REWARD_REINFORCE_RATIO_CAP) - 1)
  ```
  Positive when you now outnumber the weakest neighbor (bigger margin =
  bigger reward), negative when even the easiest target still beats you.
  `REWARD_REINFORCE_RATIO_CAP` exists because the raw ratio can blow up
  when the weakest neighbor has very few armies (e.g. 1) — without a cap,
  one lucky matchup could dwarf every other reward term in the same step.
- **`REWARD_REINFORCE_NO_ENEMY_NEIGHBOR`** — flat negative, used only as
  the fallback for the case `REWARD_REINFORCE_ATTACK_READINESS` can't
  score (no adjacent enemy territory at all, so no ratio to compute) — an
  interior placement is currently wasted, those armies do nothing until a
  later `FORTIFY` moves them. Mutually exclusive with
  `REWARD_REINFORCE_ATTACK_READINESS` on any given placement, never both.
- **`REWARD_REINFORCE_CONTINENT_PUSH`** — positive, scaled by the
  placement territory's continent ownership fraction (`owned_territories_in_continent
  / total_territories_in_continent`, via `topology.territories_in(continent)`
  + `owners`) and inversely by continent size — smaller continents are
  cheaper to lock down and should be prioritized. Applies unconditionally
  (see principle 5 — no global survivability gate). Independent of the
  three terms above; rewards a different axis (continent-bonus strategy,
  not attack setup).

These three (concentration, attack-readiness/no-neighbor, continent-push)
are complementary, not redundant — an agent can concentrate into a
territory that's still outmatched (concentration reward, no
attack-readiness reward), or place a small amount onto an already-strong
territory (attack-readiness reward, less concentration reward), or push
into a continent for the long-term bonus regardless of immediate attack
plans.

Noted but deliberately not modeled yet: always rewarding maximum
concentration could teach the agent to turtle one border while neighbors
roll through its *other* borders in a 3-6 player free-for-all, since none
of these terms are aware of the agent's other frontier territories. Start
with these independent per-placement terms, see how self-play behaves, and
only add cross-territory awareness (e.g. don't reward over-stacking one
border if others are dangerously thin) if that failure mode actually shows
up — avoid engineering in complexity the sparse/delta terminal reward might
already teach on its own.

### `ATTACK`
The richest phase — most of the game's state actually changes here, and
where most of the engine bookkeeping (`_apply_attack`'s `info` dict,
`_conquered_this_turn`) already lives.

- **`REWARD_ATTACK_FEWER_DICE`** — negative when `action.dice <
  min(MAX_ATTACK_DICE, armies[from] - 1)`, i.e. the agent had the option to
  roll more dice and chose not to. More dice is the standard Risk
  heuristic (higher expected attrition advantage per round), so leaving a
  die unused should be a real penalty, not a tiny nudge. In particular, if
  the attacker could roll 3 dice and chooses 2, that should be a strong
  penalty. Gated the same way as the `TRADE_IN`/`OCCUPY` forced cases
  elsewhere in this doc: only penalize when fewer dice was a real choice,
  not when the territory's army count itself capped the max (e.g. a
  3-army territory can only ever roll 2 dice — that's not a suboptimal
  choice, it's the only option).
  **Independent of `REWARD_ATTACK_RATIO` below** — the two terms
  stack additively on the same action, neither gates or overrides the
  other. A 2-dice attack with a great force ratio (e.g. 4 armies attacking
  a 1-army territory) gets a strongly positive `REWARD_ATTACK_RATIO`
  *and* the fewer-dice penalty in the same step — the favorable ratio
  doesn't excuse leaving a die on the table.
- **`REWARD_ATTACK_RATIO`** — `ratio = armies[from] / armies[to]`
  (pre-attack), scaled continuously, same style as
  `REWARD_REINFORCE_ATTACK_READINESS`:
  ```
  reward = REWARD_ATTACK_RATIO_SCALE * (min(ratio, REWARD_ATTACK_RATIO_CAP) - REWARD_ATTACK_RATIO_THRESHOLD)
  ```
  Positive and growing the further `ratio` is above
  `REWARD_ATTACK_RATIO_THRESHOLD`, negative and growing the further below
  it — not just "good attack/bad attack" but "how good/bad," giving the
  network a denser gradient than a flat ±X would. `REWARD_ATTACK_RATIO_CAP`
  exists for the same reason as the reinforce term's cap: an
  overwhelmingly favorable ratio (e.g. attacking a 1-army territory with
  20) shouldn't produce an unbounded reward that dwarfs every other term in
  the same step.

  Advised default for the threshold: **1.5** — Risk strategy convention is
  that you generally want 1.5-2x the defender's strength before attacking
  is statistically favorable over a multi-round exchange, since ties favor
  the defender in 1v1 rolls and the defender's 2-dice cap gives it a
  positional edge even when out-numbered. Starting at 1.5 (not 2.0) errs
  toward not over-discouraging attacks, since it's easier to raise later
  than to recover from an agent that's learned not to attack at all.
- **`REWARD_ATTACK_CONTINENT_DOMINATION`** — positive when the attack
  target's continent is one the agent already dominates, using a simple
  low-compute gate that does not require scanning opponent shares:
  ```
  o = owned_territories_in_continent
  t = total_territories_in_continent
  p = number_of_alive_players
  m = t - o

  is_dominating = (o / t) >= (1 / p + REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN)
  reward = REWARD_ATTACK_CONTINENT_DOMINATION * (1 / (m + 1)) if is_dominating else 0
  ```
  This adapts to player count (`1/p` baseline), grows as the continent gets
  closer to full control (`1/(m+1)`), and stays cheap to compute. Applies
  unconditionally (see principle 5 — no global survivability gate).
- **`REWARD_ATTACK_ARMY_TRADE`** — net of `info["defender_losses"] -
  info["attacker_losses"]`, scaled by a small constant, on *every* attack
  regardless of outcome. Distinct from `REWARD_ATTACK_RATIO` —
  ratio assesses the *decision* before dice are rolled (was this a sound
  attack to start), this term assesses the *actual outcome* (did the dice
  favor you), so a sound attack with bad luck still gets useful negative
  feedback here even though the ratio term scored it positively going in.
- **`REWARD_ATTACK_ELIMINATE_OPPONENT`** — positive, on `info["eliminated"]
  is not None`, and scaled by how many cards are gained from that eliminated
  opponent:
  ```
  reward = REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD * (cards_taken_from_eliminated + 1)
  ```
  This keeps the signal aligned with actual elimination value: eliminating a
  player with more cards is worth more than eliminating one with none,
  while still giving a positive reward even when `cards_taken_from_eliminated == 0`.
  Separate from conquest itself — eliminating one of several opponents in a
  multi-player game is a real sub-goal (removes a future threat, transfers
  their cards/territories), not just "happened to be the attack that finished
  them off."
- **`REWARD_ATTACK_CONTINENT_CAPTURED`** — positive, when this specific
  conquest completes full ownership of a continent
  (`topology.owns_continent(...)` flips False→True for the agent).
  Distinct from `REWARD_ATTACK_CONTINENT_DOMINATION` above — domination
  rewards attacking *within* a continent already mostly controlled (an
  ongoing incentive across many attacks), this rewards the one-time
  moment of *finishing* a continent (a milestone, fires at most once per
  continent per game).
- **`REWARD_ATTACK_CONQUER_TERRITORY`** — positive, on every
  `info["conquered"]`, regardless of whether this is the turn's first
  conquest. Base "did this attack accomplish something" signal, so a turn
  that conquers 3 territories scores more than one that conquers 1.
- **`REWARD_ATTACK_CONQUER_WITH_CARD`** — additional positive, stacked on
  top of `REWARD_ATTACK_CONQUER_TERRITORY`, specifically on the conquest
  that also draws a card. Card draws only happen on a player's *first*
  conquest each turn (`_conquered_this_turn`, `environment.py:413`) — so
  the first conquest gets `REWARD_ATTACK_CONQUER_TERRITORY +
  REWARD_ATTACK_CONQUER_WITH_CARD`, later conquests in the same turn get
  `REWARD_ATTACK_CONQUER_TERRITORY` alone. Detect "drew a card" the same
  way the rest of this doc detects shaping signals, from a before/after
  diff (`len(hand_after) > len(hand_before)`), not by reaching into the
  engine's private flag.
- **`REWARD_ATTACK_CARD_TERRITORY_MATCH`** — additional positive, stacked
  on top of `REWARD_ATTACK_CONQUER_WITH_CARD`, when the specific card drawn
  has a `territory_id` the player currently owns (post-conquest). Same
  concept as `REWARD_TRADE_IN_TERRITORY_MATCH`, but at draw-time rather
  than trade-in-time — not double-counting the same decision, since one
  rewards the (lucky) draw event and the other rewards choosing that card
  later when assembling a set to trade in.
- **`REWARD_ATTACK_STOP_WITHOUT_CARD`** — negative, when the agent picks
  `StopAttackAction` and no card has been drawn yet this turn. Needs the
  same "was this actually avoidable" gate as `REWARD_ATTACK_FEWER_DICE`:
  `_legal_attack` always yields `StopAttackAction` unconditionally
  (`environment.py:459`), even when zero real `AttackAction`s exist (e.g.
  no territory has both `>=2` armies and an enemy neighbor) — so this
  penalty must only apply when at least one real `AttackAction` was also
  legal at that decision point, otherwise it punishes a forced stop the
  same way an ungated `TRADE_IN`/`dice` penalty would punish a forced move.
  Implementation note: unlike every other term above, this one needs
  *turn-level* memory ("was a card drawn at any point since this turn
  began"), not just a before/after diff on the single transition being
  scored — that's state the engine already tracks internally
  (`_conquered_this_turn`) but doesn't currently expose. Recommend exposing
  it (e.g. on `State`) rather than having the trainer reimplement the same
  bookkeeping, consistent with this doc's "computed once, in the engine"
  principle.

### `OCCUPY`
The engine hard-enforces that at least 1 army stays behind in the attacking
territory — `_occupy_bounds` (`environment.py:467-482`) computes
`available = armies[from] - 1` and only ever offers `count` in
`[lo, available]`, so "move everything" was never actually a legal option.
The real decision is how much of that `available` amount to send forward:

- **`REWARD_OCCUPY_FORWARD_MOMENTUM`** — positive, scaled by `count_moved /
  available` (the fraction of the movable amount actually sent into the
  conquered territory). Moving the max (`count == hi`) scores highest,
  moving only the minimum (`count == lo`) scores lowest. If the plan is to
  keep attacking, the conquered territory usually wants most of what's
  movable — it can become the next attacking base if it borders another
  enemy, and forces left behind in the origin are now one hop further from
  the front. Naturally bounded to `[0, 1]` before scaling, so unlike the
  ratio-based terms elsewhere in this doc it needs no separate cap. Keep this
  term intentionally simple: always score by ratio only, with no border/
  survivability gate.
  ```
  available = armies[from]_before - 1
  reward = REWARD_OCCUPY_FORWARD_MOMENTUM * (count_moved / available)
  ```

### `FORTIFY`
`FortifyAction` moves armies between two owned, connected territories. The
right reward shape depends on whether the *source* and *destination* are
each a "frontier" territory (≥1 adjacent enemy territory, same definition
used in `REINFORCE_PLACE`). To avoid contradictory signals, apply terms with
strict precedence:

1. If both source and destination are frontier: apply only
  `REWARD_FORTIFY_BALANCE`.
2. If exactly one side is frontier (XOR): apply only
  `REWARD_FORTIFY_TOWARD_FRONTIER`.
3. If neither side is frontier: no frontier/balance term.

- **`REWARD_FORTIFY_TOWARD_FRONTIER`** — the asymmetric case: destination
  is frontier, source is not (or vice versa). Positive, scaled by
  `count_moved / (armies[source]_before - 1)` (same "fraction of what's
  movable" pattern as `REWARD_OCCUPY_FORWARD_MOMENTUM`), when moving
  *toward* the frontier — an interior territory can safely give up most of
  its armies since nothing threatens it. **Negative**, same scaling, when
  moving *away* from a frontier into the interior — that strips a
  defended border for no reason. This term is intentionally amount-sensitive:
  moving 1 army gives a small signal; moving many armies gives a much larger
  signal (positive toward frontier, larger negative into interior).
  ```
  movable = armies[source]_before - 1
  move_frac = count_moved / movable

  if destination_is_frontier and not source_is_frontier:
      reward = +REWARD_FORTIFY_TOWARD_FRONTIER * move_frac
  elif source_is_frontier and not destination_is_frontier:
      reward = -REWARD_FORTIFY_TOWARD_FRONTIER * move_frac
  ```
  Only fires when exactly one side is frontier (XOR), so there is no
  overlap with `REWARD_FORTIFY_BALANCE`. The all-interior case is a real
  no-op (no term).
- **`REWARD_FORTIFY_BALANCE`** — the case you flagged: when *both* source
  and destination are frontier territories, concentrating onto one and
  draining the other is wrong — both face a threat, so the right move is
  to match armies to relative threat, not blindly force equal stacks.
  Negative, scaled by post-fortify deviation from a threat-weighted target:
  ```
  threat_src = frontier_threat(source)
  threat_dst = frontier_threat(dest)
  target_dst = (threat_dst / (threat_src + threat_dst + eps)) * (armies[source]_after + armies[dest]_after)
  reward = -REWARD_FORTIFY_BALANCE_SCALE * abs(armies[dest]_after - target_dst)
           / (armies[source]_after + armies[dest]_after)
  ```
  `frontier_threat(x)` should be an engine-computable local pressure proxy
  (e.g. sum of adjacent enemy armies, optionally weighted by adjacency/path).
  Normalization by total post-fortify armies keeps the penalty scale-invariant
  (a 10-army mismatch means something different between two 12-army stacks
  than between two 200-army stacks). `0` deviation from threat-weighted
  target scores best; moving everything to one side against the threat map
  scores worst.
  Mutually exclusive with `REWARD_FORTIFY_TOWARD_FRONTIER` by precedence
  above — when both are frontier, balance is the only active term.
- **`REWARD_FORTIFY_CONTINENT_PUSH`** — same shape as
  `REWARD_REINFORCE_CONTINENT_PUSH`: positive, scaled by the
  destination's continent ownership fraction and inversely by continent
  size. Applies unconditionally (see principle 5 — no global survivability
  gate). Independent of the frontier/balance terms above — a different axis
  (continent-bonus strategy, not immediate defense), so it stacks
  additively regardless of which of the two cases above fired.

## End-of-turn: opponent-impact terms

Every term above is scoped to a single action within the agent's own turn.
None of them ever see what the *opponents* did with their turns — and per
the loop trace earlier in this doc, there's exactly one transition per
agent turn where that's even observable: the turn-ending `FortifyAction`
(`_apply_fortify` always calls `_advance_turn`, `environment.py:544-546` —
skip or real move, every `FORTIFY` decision ends the turn). That
transition's `next_state` is the only one that spans the *entire opponent
round* (the trainer's inner `while` loop, `trainer.py:117-122`), since
every other action's `next_state` is just "immediately after my one
action," with opponents not having moved yet.

So these terms compute a before/after delta where **before** = the state
right when the agent takes its `FortifyAction` (i.e. after the agent's own
turn already happened) and **after** = `next_state` once every opponent has
played their full turn and control returns to the agent. That isolates
*the opponents' impact*, since the agent's own turn's effects are already
baked into "before" — and isolating that is exactly what no other term in
this doc does; every `ATTACK`/`REINFORCE_PLACE`/etc. term already rewards
the agent's own actions, never what happened to it while it wasn't looking.

- **`REWARD_TERRITORY_DELTA`** — simple relative-share version:
  ```
  my_share_before = my_territories_before / total_territories_before
  my_share_after = my_territories_after / total_territories_after
  reward = REWARD_TERRITORY_DELTA * (my_share_after - my_share_before)
  ```
  Because total territories is effectively fixed in Risk, this is equivalent
  to a normalized self-delta but clearer to reason about as "my board share
  got better/worse versus everyone else."
- **`REWARD_ARMY_DELTA_RELATIVE`** — use the same relative-share idea for
  armies (matches your 30/100 -> 25/80 example):
  ```
  my_share_before = my_total_armies_before / total_armies_before
  my_share_after = my_total_armies_after / total_armies_after
  reward = REWARD_ARMY_DELTA_RELATIVE_SCALE * (my_share_after - my_share_before)
  ```
  Positive when your fraction of all armies increased (even if your absolute
  army count fell), negative when your fraction fell.
- **`REWARD_CONTINENT_DELTA_RELATIVE`** — weight continent control by each
  continent's reinforcement value (bonus armies), not by raw continent count.
  Owning one low-value continent while opponents own a high-value one should
  not look equally good. Use bonus-share change:
  ```
  my_bonus_before = sum(continent_bonus(c) for c fully owned by me before)
  my_bonus_after = sum(continent_bonus(c) for c fully owned by me after)
  total_bonus = sum(continent_bonus(c) for all continents)

  my_share_before = my_bonus_before / total_bonus
  my_share_after = my_bonus_after / total_bonus
  reward = REWARD_CONTINENT_DELTA_RELATIVE * (my_share_after - my_share_before)
  ```
  This directly captures your example: owning a 2-bonus continent while an
  opponent owns a 5-bonus continent is relatively weak, and the reward will
  reflect that.

These fire once per agent turn (on the `FortifyAction` transition only),
unlike every phase-specific term above which fires on every matching
action — don't apply this formula to every transition, since outside of
that one turn-ending action `next_state` hasn't seen the opponents' moves
yet and the delta would just read `~0`.

## Full constant checklist (for the value-assignment pass)

Every constant this doc has introduced, grouped by where it fires. Use this
as the checklist when we move on to picking actual values.

- **`TRADE_IN`**: `REWARD_TRADE_IN_EARLY`, `REWARD_TRADE_IN_TERRITORY_MATCH`
- **`REINFORCE_PLACE`**: `REWARD_REINFORCE_CONCENTRATION`,
  `REWARD_REINFORCE_ATTACK_READINESS_SCALE`, `REWARD_REINFORCE_RATIO_CAP`,
  `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR`, `REWARD_REINFORCE_CONTINENT_PUSH`
- **`ATTACK`**: `REWARD_ATTACK_FEWER_DICE`, `REWARD_ATTACK_RATIO_SCALE`,
  `REWARD_ATTACK_RATIO_CAP`, `REWARD_ATTACK_RATIO_THRESHOLD`,
  `REWARD_ATTACK_CONTINENT_DOMINATION`,
  `REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN`,
  `REWARD_ATTACK_CONTINENT_ADVANTAGE`, `REWARD_ATTACK_ARMY_TRADE`,
  `REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD`, `REWARD_ATTACK_CONTINENT_CAPTURED`,
  `REWARD_ATTACK_CONQUER_TERRITORY`, `REWARD_ATTACK_CONQUER_WITH_CARD`,
  `REWARD_ATTACK_CARD_TERRITORY_MATCH`, `REWARD_ATTACK_STOP_WITHOUT_CARD`
- **`OCCUPY`**: `REWARD_OCCUPY_FORWARD_MOMENTUM`
- **`FORTIFY`**: `REWARD_FORTIFY_TOWARD_FRONTIER`,
  `REWARD_FORTIFY_BALANCE_SCALE`, `REWARD_FORTIFY_CONTINENT_PUSH`
- **End-of-turn**: `REWARD_TERRITORY_DELTA`, `REWARD_ARMY_DELTA_RELATIVE_SCALE`,
  `REWARD_CONTINENT_DELTA_RELATIVE`
- **Composition/safety**: `REWARD_SHAPING_STEP_CAP`
- **Terminal**: `REWARD_TERMINAL_WIN`, `REWARD_TERMINAL_LOSS` (implemented;
  no `REWARD_TERMINAL_TIMEOUT` — see intro)

~32 constants total. With this many additive terms, the shaping cap(s) need
explicit values, not implicit assumptions.

## Proposed v1 values (starting point)

These are concrete initial values to start implementation/tuning. They are
chosen to keep shaping bounded per step (after clipping) while making
terminal outcomes much larger than any single shaping event.

- **Global composition / terminal**
  - `REWARD_SHAPING_STEP_CAP = 10.0`
  - `REWARD_TERMINAL_WIN = 100.0` (implemented)
  - `REWARD_TERMINAL_LOSS = -100.0` (implemented)

- **`TRADE_IN`**
  - `REWARD_TRADE_IN_EARLY = 0.30` (sign applied by action: skip `+`, optional trade `-`)
  - `REWARD_TRADE_IN_TERRITORY_MATCH = 0.60`

- **`REINFORCE_PLACE`**
  - `REWARD_REINFORCE_CONCENTRATION = 1.20`
  - `REWARD_REINFORCE_ATTACK_READINESS_SCALE = 1.50`
  - `REWARD_REINFORCE_RATIO_CAP = 2.50`
  - `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR = -0.80`
  - `REWARD_REINFORCE_CONTINENT_PUSH = 0.90`

- **`ATTACK`**
  - `REWARD_ATTACK_FEWER_DICE = -1.25`
  - `REWARD_ATTACK_RATIO_SCALE = 2.00`
  - `REWARD_ATTACK_RATIO_CAP = 3.00`
  - `REWARD_ATTACK_RATIO_THRESHOLD = 1.50`
  - `REWARD_ATTACK_CONTINENT_DOMINATION = 0.80`
  - `REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN = 0.10`
  - `REWARD_ATTACK_CONTINENT_ADVANTAGE = 1.20`
  - `REWARD_ATTACK_ARMY_TRADE = 0.60`
  - `REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD = 1.25`
  - `REWARD_ATTACK_CONTINENT_CAPTURED = 4.00`
  - `REWARD_ATTACK_CONQUER_TERRITORY = 1.20`
  - `REWARD_ATTACK_CONQUER_WITH_CARD = 1.00`
  - `REWARD_ATTACK_CARD_TERRITORY_MATCH = 0.60`
  - `REWARD_ATTACK_STOP_WITHOUT_CARD = -2.00`

- **`OCCUPY`**
  - `REWARD_OCCUPY_FORWARD_MOMENTUM = 1.00`

- **`FORTIFY`**
  - `REWARD_FORTIFY_TOWARD_FRONTIER = 1.00`
  - `REWARD_FORTIFY_BALANCE_SCALE = 2.00`
  - `REWARD_FORTIFY_CONTINENT_PUSH = 0.80`

- **End-of-turn**
  - `REWARD_TERRITORY_DELTA = 1.00`
  - `REWARD_ARMY_DELTA_RELATIVE_SCALE = 0.10`
  - `REWARD_CONTINENT_DELTA_RELATIVE = 2.50`

Implementation note for normalization target: compute `shaping_raw` by adding
all active shaping terms, then clip once with `REWARD_SHAPING_STEP_CAP`. This
enforces non-terminal step rewards in `[-10, 10]` regardless of occasional
multi-trigger stacks.

## Open questions before coding

1. Sparse-only baseline first, then layer in shaping once it trains at all?
   Or build the shaping in from the start? (Recommend: get one sparse-only
   training run worth trusting first, since it's the simplest thing that
   could possibly work, then add shaping incrementally and compare.)
2. Should shaping reward be visible to `done` transitions only, or every
   transition? (Recommend: every transition — that's the point of dense
   shaping vs. the existing sparse terminal-only reward.)
3. Scale follow-up: keep only per-step clipping (`REWARD_SHAPING_STEP_CAP`),
  or also add an optional per-episode shaping cap later.
4. ~~Do we want per-component logging...~~ **Resolved/implemented** — see
   "Per-component logging (implemented)" above.

## Coding plan and structure

Goal: reward math lives in one dedicated class, fully outside `environment.py`.
`Environment.step(...)` calls that class and uses the number it returns —
it performs no reward arithmetic itself, not even the current sparse
+1/-1/0. This is a stricter reading of principle 2 ("computed once, in the
engine") than today's code: the engine still *produces* the raw ingredients
(`info` dicts, before/after `State`), but a separate class turns those
ingredients into a number.

### New file: `risk/learning/reward.py`

```python
class RewardCalculator:
    def __init__(self, topology: BoardTopology) -> None:
        self.topology = topology

    def compute(
        self,
        action: Action,
        info: dict,
        before: State,
        after: State,
        reward_player: int,
        done: bool,
        winner: Optional[int],
    ) -> float:
        ...
```

One public method, one call site. Everything else on the class is a private
helper (`_terminal`, `_trade_in`, `_reinforce`, `_attack`, `_occupy`,
`_fortify`, `_end_of_turn`, plus small shared primitives like
`_frontier_threat`). `compute` dispatches on `type(action)` /
`before.phase` to exactly one of the phase helpers, adds the terminal term,
clips, and returns — mirroring the `shaping_raw -> clip -> + terminal`
formula already specified under principle 6. No public API beyond
`compute`; the trainer and tests never call the phase helpers directly.

Instance methods throughout (per project style) — `topology` is the only
piece of fixed context the class needs (for `neighbors`, `territories_in`,
`owns_continent`, `continent_bonus`), so it's stored once at construction
instead of threaded through every call.

### `environment.py` changes

1. `Environment.__init__` builds `self._reward = RewardCalculator(self.topology)`.
2. In `step(...)`, capture `before = s.snapshot()` (already exists,
   `State.copy()`/`State.snapshot()`) right before dispatching to the
   `_apply_*` method — that's the only point where pre-action state is
   still available, since `_apply_*` mutates `s` in place.
3. Replace the existing inline `if reward_player is not None: ...` block
   with a single call:
   ```python
   if reward_player is not None:
       done = reward_player in s.eliminated or s.phase is Phase.GAME_OVER
       reward = self._reward.compute(
           action=action, info=info, before=before, after=s,
           reward_player=reward_player, done=done, winner=self.winner(),
       )
   ```
   No `+1.0`/`-1.0` literals left in `environment.py` — those become
   `REWARD_TERMINAL_WIN`/`REWARD_TERMINAL_LOSS` inside
   `RewardCalculator._terminal`.
4. `_apply_*` methods are untouched — they keep building `info` dicts
   exactly as today; `RewardCalculator` only ever reads them, never the
   other way around.

### Test placement

Per `Docs/Testing.md`'s convention (one `Temp/tests/*.py` per subsystem):
new file `Temp/tests/test_reward.py`, testing `RewardCalculator` directly
against hand-built `(before, after, info)` triples — no `Environment`
needed for most cases, which is the point of keeping it a separate class.
A handful of `test_environment.py` cases should confirm `step(...)` wires
`before`/`after`/`info` into `RewardCalculator` correctly, without
re-testing the reward math itself.

### Implementation order (answers open question 1) — all DONE

1. **DONE.** Landed the class with terminal-only logic, wired into
   `environment.py`, old inline `+1.0`/`-1.0` block deleted. Confirmed via
   `Temp/tests/test_reward.py` plus a real `Trainer.train(...)` smoke run.
2. **DONE.** All phase helpers implemented in one pass (`TRADE_IN` →
   `REINFORCE_PLACE` → `ATTACK` → `OCCUPY` → `FORTIFY` → end-of-turn), each
   gated on `before.current_player_index == reward_player`. Required two
   changes beyond `reward.py`/`environment.py`, both logged in
   [`RL-Prep-Changes.md`](RL-Prep-Changes.md): `trainer.py` now sums reward
   across its inner opponent-turn loop instead of overwriting it (the old
   sparse-only reward never exposed this bug, since every non-terminal step
   was `0.0` anyway), and calls `Environment.reward.end_of_turn(...)`
   directly on the learner's turn-ending `FortifyAction`; `State` gained a
   `conquered_this_turn` field so `RewardCalculator` can read "was a card
   drawn yet this turn" from a plain snapshot.
3. `Temp/tests/test_reward.py` has one case per phase helper plus
   `end_of_turn` (hand-built `before`/`after` `State` triples). Per-component
   logging (open question 4) is still future work — worth doing once a real
   training run is underway and a bad constant needs to be visible in the
   training curves rather than just inferred from behavior.

## Continent-advantage reward

Implementation status: implemented in `risk/learning/reward.py`, with
`REWARD_ATTACK_CONTINENT_ADVANTAGE` defined in
`risk/learning/train_constants.py`.

Observed problem: the trained agent learns which neighboring territory is
worth attacking, but it does not reliably convert a local advantage into a
continent plan. In particular, when it already owns most of a continent and
has enough troops there to finish it, the current shaping is too weak or too
indirect to say "keep taking territories in this continent now."

### Goal

Add a small dense reward that scores attacks by the strategic value of the
target's continent, not only by the immediate attack ratio. The reward should
prefer attacks into continents where the player has:

1. more owned territories out of the continent total,
2. more troops in that continent than all other players combined,
3. a higher continent bonus value.

This should be an incentive to finish continents where the player already has
a real advantage, not a blanket "always chase continents" bonus.

### Keep the first version cheap

Do not start by caching this in `GraphAdapter.u` or another global graph
attribute for reward calculation. Reward code already has everything it needs:
`State.owners`, `State.armies`, and `BoardTopology.territories_in(continent)`.
There are only 6 continents and 42 territories, so scanning the target
continent during `RewardCalculator._attack(...)` is cheap and much simpler
than adding cache invalidation or keeping graph features synchronized with the
rules engine.

Graph global attributes are still worth considering later as model inputs,
not as reward storage. If the agent still struggles after the reward change,
add per-continent summary features to `GraphAdapter.u` so the network can see
the same "continent opportunity" signal directly when scoring legal actions.
That would be a separate network-input change, documented in
`Docs/GraphAdapter.md`, and should update the `u` width and tests.

### Proposed formula

Add a helper in `RewardCalculator`, roughly:

```python
continent_advantage(state, player_id, continent) -> float
```

For the target continent:

```python
owned = owned_territories_in_continent
total = total_territories_in_continent
my_troops = troops_on_my_territories_in_continent
other_troops = troops_on_non_my_territories_in_continent
bonus = topology.continent_bonus(continent)
max_bonus = max(topology.continent_bonus(c) for c in topology.continents)

territory_score = owned / total
troop_score = my_troops / max(my_troops + other_troops, 1)
worth_score = bonus / max_bonus

baseline_territory_share = 1 / number_of_alive_players
territory_edge_raw = max(0, territory_score - baseline_territory_share)
troop_edge_raw = max(0, troop_score - 0.5)

# Rescale each edge to its own [0, 1] range before multiplying — see
# "Coefficient / making the signal stronger" below for why.
territory_edge = territory_edge_raw / max(1 - baseline_territory_share, _EPS)
troop_edge = troop_edge_raw / 0.5

advantage = territory_edge * troop_edge * worth_score
```

`_EPS` is `reward.py`'s existing module-level `_EPS = 1e-6`
(already used by `_fortify`'s balance term) — reuse it, don't add a second
epsilon constant. It only guards the degenerate `alive_players == 1`
case (already moot in practice, since the game ends before only one
player remains), not a realistic division.

`number_of_alive_players` should reuse `_attack(...)`'s existing
`alive_players = len(before.hands) - len(before.eliminated)`
(`reward.py:292`, already computed there for
`REWARD_ATTACK_CONTINENT_DOMINATION`) rather than recomputing it.

Multiplication is intentional: it keeps the reward low unless all three
signals agree. The `max(0, ...)` gates are also intentional: this is a reward
for actual advantage, not for any partial interest in a continent. The
territory edge only turns positive once the player owns more than their
expected share among surviving players, and the troop edge only turns positive
once the player has troop majority inside that continent. Owning most of a
continent with no troops there should not look as good as a real attack
opportunity, and having many troops in a low-progress continent should not
beat finishing a nearly controlled one.

### Coefficient / making the signal stronger

Gating both edges at `max(0, ...)` shrinks their usable range — `territory_edge_raw`
tops out at `1 - baseline_territory_share` (e.g. ~0.83 at 6 players, ~0.5 at
2), and `troop_edge_raw` tops out at `0.5`. Multiplied together with
`worth_score <= 1`, the *raw* product is compressed well below 1.0 even in
the best case (full ownership, full troop majority, max-bonus continent), so
a flat `REWARD_ATTACK_CONTINENT_ADVANTAGE` constant tuned against that
compressed ceiling would under-reward real advantage.

Rather than bolting on a second free-floating multiplier (which is just a
less legible way to do the same thing as raising the one constant — it has no
independent meaning of its own), rescale each raw edge by its own maximum
back into `[0, 1]` (as shown above: divide by `1 - baseline_territory_share`
and by `0.5` respectively). That restores `advantage`'s natural ceiling to
`worth_score`'s own `[0, 1]` range — same ceiling the original, ungated
formula had — so `REWARD_ATTACK_CONTINENT_ADVANTAGE` stays the single,
meaningful scale knob, comparable apples-to-apples against
`REWARD_ATTACK_CONQUER_TERRITORY`/`REWARD_ATTACK_CONTINENT_CAPTURED` without
needing a compensating bump once real data comes in. The `max(0, ...)` gate
behavior (zero reward below baseline) is unchanged; only the *shape* above
zero is restored to full strength.

`max_bonus` does not change during a game (it's a property of the fixed
board, not of state) — compute it once in `RewardCalculator.__init__`
alongside `self.topology` rather than re-scanning `topology.continents` on
every attack. Same "computed once" principle as the rest of this doc, and
free to do since nothing about it depends on `State`.

### Where it should fire

Use the target territory's continent for `AttackAction`.

Recommended v1 behavior:

- Add `REWARD_ATTACK_CONTINENT_ADVANTAGE` as a new term, additive alongside
  the existing `REWARD_ATTACK_CONTINENT_DOMINATION` — don't replace it. The
  two score different signals: `DOMINATION` is a hard ownership-threshold
  gate with no awareness of troop counts or continent value, while
  `ADVANTAGE` is a dense, troop/value-weighted closeness signal. `DOMINATION`
  is already tuned against the rest of the attack terms; removing it would
  lose that gate's behavior for no benefit, since the two terms are cheap
  to compute together and `compute()`'s composition rule is additive anyway
  (see principle 6).
- Apply it on every non-stop `AttackAction` into that continent, using the
  pre-attack state so it rewards the decision, not the dice result.
- Scale it by closeness to completion:
  ```python
  missing = total - owned
  reward = REWARD_ATTACK_CONTINENT_ADVANTAGE * advantage / (missing + 1)
  ```
- Keep `REWARD_ATTACK_CONTINENT_CAPTURED` as the one-time completion reward;
  this new term is the repeated "keep pushing here" signal before completion.

This gives the strongest reward when the agent owns most of a valuable
continent, has troop superiority there, and only has one or two territories
left to conquer.

### Avoid reward hacking

Keep the constant smaller than the direct conquest and capture rewards at
first. With the edge-rescaling above restoring `advantage`'s ceiling to
`[0, 1]`, the original starting point is back in scale:

```python
REWARD_ATTACK_CONTINENT_ADVANTAGE = 1.20
```

Then compare it against the existing values:

- `REWARD_ATTACK_CONQUER_TERRITORY = 1.20`
- `REWARD_ATTACK_CONTINENT_CAPTURED = 4.00`
- `REWARD_SHAPING_STEP_CAP = 10.0`

This should make continent pressure meaningful without letting the agent farm
continent intent instead of actually conquering territories.

### Possible graph feature follow-up

If reward-only shaping does not change behavior enough, add compact global
features for each continent to `GraphAdapter.u`, from the learner's
perspective:

- owned territory fraction,
- troop share,
- normalized continent bonus.

That is 18 new global features for 6 continents. It is more calculation than
the reward-only version and changes the model input shape, so it should be a
second step only after testing the reward change.

### Implementation and test notes

1. `RewardCalculator.__init__` caches the board-static max continent bonus.
2. `RewardCalculator._continent_advantage(...)` computes the gated,
   rescaled territory/troop/value signal.
3. `_attack(...)` adds the term for non-stop `AttackAction`, alongside the
   existing domination term.
4. Focused reward tests should stay in `Temp/tests/test_reward.py`; no new
   test file is needed for this subsystem.
