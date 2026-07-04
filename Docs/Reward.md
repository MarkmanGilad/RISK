# Reward design

**Status: fully implemented, actively being tuned.** `Environment.step(...)`
delegates all reward computation to `RewardCalculator.compute(...)`
(`risk/learning/reward.py`) — no reward arithmetic lives in `environment.py`.
`Trainer` additionally calls `RewardCalculator.end_of_turn(...)` directly at
the learner's turn boundary, since that term needs state spanning an entire
opponent round, which no single `step()` call ever sees.

This doc has two parts:

1. **Reference** — how the reward function works today (call sites, formulas,
   constants). Read this to understand or extend the reward.
2. **Analysis log** — dated findings from real training data and the changes
   they led to. Read this to understand *why* a constant has the value it
   has, and what's still open.

## How reward is computed

- `RewardCalculator.compute(action, info, before, after, reward_player, done, winner)`
  — called from `Environment.step(...)` on *every* actor's action (not just
  the learner's), gated so each phase helper only scores the *current
  actor's own* decision (`before.current_player_index == reward_player`) —
  otherwise an opponent's action would score as if the learner had taken it.
  Sums the active phase term(s), clips the shaping subtotal to
  `±REWARD_SHAPING_STEP_CAP`, and adds the terminal term (uncapped):
  ```
  shaping = clip(trade_in + reinforce + attack + occupy + fortify, ±REWARD_SHAPING_STEP_CAP)
  reward = terminal + shaping
  ```
- `RewardCalculator.end_of_turn(before_turn, after_turn, reward_player)` —
  called directly by `Trainer.train(...)` when the learner's own action ends
  its turn (`isinstance(action, FortifyAction) or done`), comparing state
  right when that action was taken vs. state once the entire opponent round
  has played out. Not clipped by `REWARD_SHAPING_STEP_CAP`. The `or done`
  half of that gate matters: it's what lets the learner's own game-ending
  action (an `AttackAction`/`OccupyAction` that never reaches `FortifyAction`)
  still score this term, instead of only scoring on `FortifyAction` turns.
- **Terminal**: `REWARD_TERMINAL_WIN`/`REWARD_TERMINAL_LOSS` fire once, on
  `done=True`. A player's own elimination is scored as a loss even if the
  game isn't over for everyone else. There is deliberately no terminal
  reward for a `MAX_STEPS_PER_EPISODE` timeout — a timeout truncates the
  episode without the game state ever becoming terminal, so pairing it with
  `done=True` would corrupt the TD bootstrap target. `Trainer` just stops
  calling `step()` when the cap is hit, leaving `done=False`.
- **Per-component logging**: every `compute()`/`end_of_turn()` call stashes
  its breakdown on `self.last_components`/`self.last_end_of_turn_components`
  — diagnostic only, doesn't affect the returned float.
  `Trainer._accumulate_reward_components` sums these into per-episode totals
  and logs each as `reward_component_<name>` via `TrainingLogger`, generically
  (any new key in either dict is picked up automatically, no trainer change
  needed to add a new term).

## Design principles

1. **Sparse stays the anchor.** Win/lose is meant to remain the dominant
   signal; dense shaping should nudge behavior without letting the agent
   farm shaping instead of trying to win. (Status: not fully holding —
   see Findings 5–6.)
2. **Computed once, in the engine.** Shaping reads from `info` dicts the
   `_apply_*` methods already build, plus before/after `State` snapshots —
   never recomputed ad hoc elsewhere.
3. **Symmetric bookkeeping.** Terms about the reward player's own state are
   simple before/after diffs; terms about *opponents* (eliminating one) read
   `info` instead.
4. **Every constant lives in `train_constants.py`.** No magic numbers in
   engine code.
5. **No global-survivability gate on continent terms.** Continent terms look
   only at the territory/continent the current action touches, never scan
   the player's whole frontier — kept as a cheap local heuristic, not a
   correctness guarantee.
6. **Composition is additive, with one per-step cap.** Sum active shaping
   terms, clip to `±REWARD_SHAPING_STEP_CAP`, add terminal separately.
7. **Step-range target.** Non-terminal *step* rewards stay in `[-10, 10]`
   (enforced). There is, notably, no equivalent *per-return* cap — see
   Finding 6.

## Current formulas, by phase

Every constant below is defined in `risk/learning/train_constants.py`; every
formula is implemented in the correspondingly-named method in
`risk/learning/reward.py`. Values as of the latest change are in the
[constants table](#current-constant-values) below.

### `TRADE_IN`

`card_set_value(trade_in_index)` is keyed to the *global* trade-in count, not
the trading player's hand size — since opponents trading in also advances
that counter, waiting to trade is often a real edge (your own eventual trade
lands at a higher index). So the reward encourages patience, not haste:

- **`REWARD_TRADE_IN_EARLY`** — only when `SkipTradeAction` is legal (hand
  size 3 or 4; forced at 5). Positive for skipping, negative for trading
  anyway, scaled by a hand-size factor (`1.0` at hand=3, `0.5` at hand=4) and
  inversely by the current trade value `v = card_set_value(...)`:
  `r = REWARD_TRADE_IN_EARLY * f(hand_size) / v`. Zero at hand=5 (forced,
  not a real decision).
- **`REWARD_TRADE_IN_TERRITORY_MATCH`** — flat positive whenever the agent
  trades (forced or not) and the set includes a card for a territory it
  owns. Independent of the term above — rewards picking the better of
  several legal sets, a different sub-decision from whether to trade at all.

### `REINFORCE_PLACE`

No count-delta term here — placement only moves armies already earned, it
doesn't change territory/army counts (that's `ATTACK`'s job). Three
complementary, independently-firing terms per placement:

- **`REWARD_REINFORCE_CONCENTRATION`** — positive, `placed_amount /
  remaining_budget_before`, gated to frontier territories. Rewards putting
  the full remaining budget on one frontier territory over splitting it.
- **`REWARD_REINFORCE_ATTACK_READINESS`** (scale
  `REWARD_REINFORCE_ATTACK_READINESS_SCALE`, cap `REWARD_REINFORCE_RATIO_CAP`)
  — only when ≥1 adjacent enemy territory exists:
  `reward = SCALE * (min(armies_after / weakest_adjacent_enemy_armies, CAP) - 1)`.
  Positive once you outnumber the weakest neighbor, negative otherwise.
- **`REWARD_REINFORCE_NO_ENEMY_NEIGHBOR`** — flat negative fallback when
  there's no adjacent enemy at all (an interior placement is currently
  wasted). Mutually exclusive with the term above.
- **`REWARD_REINFORCE_CONTINENT_PUSH`** — `scale * (owned/total) / total`
  (`_continent_push`) — positive, scaled by ownership fraction and divided
  by continent size *again* as an inverse-size weight. **Known weak spot**:
  this double division makes it negligible for large, valuable continents
  (Asia: `0.90 × 0.917 / 12 ≈ 0.069`) — see Finding 8.

### `ATTACK`

The richest phase — most engine state changes here.

- **`REWARD_ATTACK_FEWER_DICE`** — negative when the agent could have rolled
  more dice and didn't (only when that was a real choice, not when army
  count itself capped the max).
- **`REWARD_ATTACK_RATIO`** (scale `REWARD_ATTACK_RATIO_SCALE`, cap
  `REWARD_ATTACK_RATIO_CAP`, threshold `REWARD_ATTACK_RATIO_THRESHOLD`) —
  `reward = SCALE * (min(armies[from]/armies[to], CAP) - THRESHOLD)`.
  Continuous, not just good/bad — grows further positive/negative the
  further the pre-attack ratio is above/below the threshold.
- **`REWARD_ATTACK_CONTINENT_DOMINATION`** — positive when the target's
  continent is already dominated (`owned/total >= 1/alive_players +
  REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN`), scaled by
  `1/(territories_still_missing + 1)`. **Known weak spot**: the gate doesn't
  open until well into the conquest (Asia at 4 players: `owned >= 5` before
  this fires at all) — see Finding 8.
- **`REWARD_ATTACK_CONTINENT_ADVANTAGE`** — denser companion to the term
  above: `SCALE * advantage / (total - owned + 1)`, where `advantage =
  territory_edge * troop_edge * worth_score` (each gated to `[0,1]`,
  `_continent_advantage`) — rewards owning more of the continent *and*
  having troop majority in it *and* it being a high-bonus continent, not any
  one alone. Same gate-opens-late weak spot as domination.
- **`REWARD_ATTACK_ARMY_TRADE`** — `scale * (defender_losses -
  attacker_losses)`, on every attack regardless of outcome. Naturally
  zero-centered (unlike most attack terms) — scores the actual dice outcome,
  distinct from `_RATIO`'s pre-attack assessment of the decision.
- **`REWARD_ATTACK_ELIMINATE_OPPONENT_BASE` + `_PER_CARD`** — on
  `info["eliminated"]`: `reward = BASE + PER_CARD * cards_taken_from_them`.
  Flat base (on par with continent-capture) plus a per-card scale, so any
  elimination is a clearly big event regardless of the eliminated player's
  hand size. (Changed 2026-07-04 — see Finding 11.)
- **`REWARD_ATTACK_CONTINENT_CAPTURED`** — one-time positive, the specific
  conquest that flips continent ownership False→True. Distinct from
  `_DOMINATION` (an ongoing incentive while owning most of it) — this is the
  one-time completion milestone.
- **`REWARD_ATTACK_CONQUER_TERRITORY`** — positive on every conquest.
- **`REWARD_ATTACK_CONQUER_WITH_CARD`** — additional positive, stacked on
  the conquest that also draws a card (a turn's *first* conquest only).
- **`REWARD_ATTACK_CARD_TERRITORY_MATCH`** — additional positive, stacked
  further when the drawn card matches a territory the player now owns.
- **`REWARD_ATTACK_STOP_WITHOUT_CARD`** — negative for `StopAttackAction`
  when a real attack was available and no card drawn yet this turn (gated
  the same way as `_FEWER_DICE` — never penalizes a genuinely forced stop).

### `OCCUPY`

The engine already enforces ≥1 army stays behind — the real decision is how
much of the *movable* amount to send forward:

- **`REWARD_OCCUPY_FORWARD_MOMENTUM`** — positive, `count_moved /
  available`. Naturally bounded to `[0,1]`, no separate cap needed.

### `FORTIFY`

Shape depends on whether source/destination are "frontier" territories
(≥1 adjacent enemy), with strict precedence to avoid contradictory signals:
both frontier → balance only; exactly one frontier (XOR) → toward-frontier
only; neither → no term.

- **`REWARD_FORTIFY_TOWARD_FRONTIER`** — positive moving toward a frontier
  from a non-frontier source, negative moving away from a frontier into the
  interior. Scaled by `count_moved / movable`.
- **`REWARD_FORTIFY_BALANCE`** (scale `REWARD_FORTIFY_BALANCE_SCALE`) — when
  both sides are frontier, penalizes deviation from a threat-weighted target
  split (`frontier_threat` = sum of adjacent enemy armies) rather than
  rewarding raw concentration — both sides face a threat, so armies should
  match relative danger, not pile onto one side.
- **`REWARD_FORTIFY_CONTINENT_PUSH`** — same shape and same known weak spot
  as `REWARD_REINFORCE_CONTINENT_PUSH` above, applied to the destination.

### End-of-turn (opponent-impact + hold/lost terms)

Computed only at the learner's turn boundary (`end_of_turn`, see "How reward
is computed" above) — the only point where `after_turn` has seen the entire
opponent round.

- **`REWARD_TERRITORY_DELTA`** — `scale * (share_after - share_before)`,
  `share = my_territories / 42`. Symmetric, but structurally **can only be
  ≤ 0 on any turn except the episode-ending one** — opponents can only take
  from the learner during their round, never hand territory back, so a
  "successful defense" turn scores exactly `0`, never positive.
- **`REWARD_TERRITORY_HOLD`** — `scale * share_after`, unconditional, every
  turn. Added specifically because the term above can't reward successful
  defense (Finding 9). Deliberately small (0.05) since it's an always-on
  per-turn bonus — same farming-risk shape as `attack`'s dominance (Finding
  5–6), just much smaller magnitude.
- **`REWARD_ARMY_DELTA_RELATIVE`** — same relative-share idea for total
  armies. Unlike territory share, this *can* move positive on a pure
  opponent-round turn (if opponents lose armies fighting each other, the
  learner's army *share* can rise even with an unchanged absolute count).
- **`REWARD_CONTINENT_DELTA_RELATIVE`** — relative-share idea using
  continent-bonus share instead of raw continent count, so owning a
  low-value continent isn't scored the same as owning a high-value one.
  Symmetric (rewards gains and losses equally).
- **`REWARD_CONTINENT_LOST`** — extra penalty, fires *only* when continent
  bonus share dropped this round: `scale * (my_bonus_after -
  my_bonus_before) / total_bonus`. Stacks on top of the symmetric term
  above, making a loss sting more than the equivalent gain rewards, without
  adding any always-positive farming risk (it only ever subtracts). Added
  because the symmetric term alone was too small to make losing a
  hard-won continent feel costly (Finding 10).

These four fire once per learner turn (on the `end_of_turn` transition
only), unlike every phase term above which fires on every matching action.

## Current constant values

Source of truth is always `risk/learning/train_constants.py`; this table is
a snapshot for reference (as of 2026-07-04, after Findings 9–11).

| constant | value |
|---|---|
| `REWARD_SHAPING_STEP_CAP` | 10.00 |
| `REWARD_TERMINAL_WIN` / `_LOSS` | +100.00 / −100.00 |
| `REWARD_TRADE_IN_EARLY` | 0.30 |
| `REWARD_TRADE_IN_TERRITORY_MATCH` | 0.60 |
| `REWARD_REINFORCE_CONCENTRATION` | 1.20 |
| `REWARD_REINFORCE_ATTACK_READINESS_SCALE` | 1.50 |
| `REWARD_REINFORCE_RATIO_CAP` | 2.50 |
| `REWARD_REINFORCE_NO_ENEMY_NEIGHBOR` | −0.80 |
| `REWARD_REINFORCE_CONTINENT_PUSH` | 0.90 |
| `REWARD_ATTACK_FEWER_DICE` | −1.25 |
| `REWARD_ATTACK_RATIO_SCALE` | 2.00 |
| `REWARD_ATTACK_RATIO_CAP` | 3.00 |
| `REWARD_ATTACK_RATIO_THRESHOLD` | 1.50 |
| `REWARD_ATTACK_CONTINENT_DOMINATION` | 0.80 |
| `REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN` | 0.10 |
| `REWARD_ATTACK_CONTINENT_ADVANTAGE` | 1.20 |
| `REWARD_ATTACK_ARMY_TRADE` | 0.60 |
| `REWARD_ATTACK_ELIMINATE_OPPONENT_BASE` | **4.00** (new) |
| `REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD` | **1.50** (was 1.25, applied to `cards+1`) |
| `REWARD_ATTACK_CONTINENT_CAPTURED` | 4.00 |
| `REWARD_ATTACK_CONQUER_TERRITORY` | 1.20 |
| `REWARD_ATTACK_CONQUER_WITH_CARD` | 1.00 |
| `REWARD_ATTACK_CARD_TERRITORY_MATCH` | 0.60 |
| `REWARD_ATTACK_STOP_WITHOUT_CARD` | −2.00 |
| `REWARD_OCCUPY_FORWARD_MOMENTUM` | 1.00 |
| `REWARD_FORTIFY_TOWARD_FRONTIER` | 1.00 |
| `REWARD_FORTIFY_BALANCE_SCALE` | 2.00 |
| `REWARD_FORTIFY_CONTINENT_PUSH` | 0.80 |
| `REWARD_TERRITORY_DELTA` | 1.00 |
| `REWARD_ARMY_DELTA_RELATIVE_SCALE` | 0.10 |
| `REWARD_CONTINENT_DELTA_RELATIVE` | 2.50 |
| `REWARD_TERRITORY_HOLD` | **0.05** (new) |
| `REWARD_CONTINENT_LOST` | **5.00** (new) |

---

# Analysis log

Dated findings from real training data (`run_021`–`run_023`), in
chronological order. Kept as the record of *why* — skip to
["Status before the next test"](#status-before-the-next-test) for the
current bottom line.

## Findings 1–4 (2026-07-03, run_021, 7 episodes) — first look

- **Finding 1**: cumulative per-episode shaping averaged **8.6x** the ±100
  terminal magnitude. Not from a few extreme turns (per-step clipping was
  barely ever binding) — many ordinary turns each adding a small amount over
  hundreds of turns. Because this shaping is *not* potential-based, it's not
  guaranteed policy-invariant — a shaped return this much larger than the
  terminal can shift which policy is optimal, not just speed up learning.
- **Finding 2**: `attack` alone was **~84%** of all shaping, every episode,
  80–89% range, no exceptions — a structural property (attack fires many
  times per turn), not noise. Caveat: episode-*sum* data can't separate
  "fires often" from "each fire pays too much" — per-event-mean logging
  would be needed to tell those apart (**still not implemented**).
- **Finding 3**: the end-of-turn delta terms only ever saw the learner's
  *losses*, never its own winning turn — `Trainer` only called
  `end_of_turn(...)` on `FortifyAction`, and a win happens on an
  `AttackAction`/`OccupyAction` that ends the episode before reaching
  `FortifyAction`. **Fixed 2026-07-03**: gate changed to
  `isinstance(action, FortifyAction) or done`.
- **Finding 4**: constants aren't on comparable scales — `REWARD_TERRITORY_DELTA
  = 1.00` multiplies a *board-share* change (~1/42 per territory) while
  same-valued per-decision terms multiply fractions up to ~1–3. Comparing
  raw constants as "same number = same weight" is misleading.

## Findings 5–7 (2026-07-04, runs 022 & 023, ~500 episodes each) — post-fix, at scale

Confirmed Findings 1–2 hold at ~75x the sample size, not a 7-episode
artifact. Per agent-turn: `attack` **+1.70**, everything else combined
**+0.46**, the three delta terms together **−0.009** (still net-negative
even post-Finding-3-fix, since they see every mid-game opponent
counter-attack but the one winning swing is diluted across hundreds of
turns). Cumulative per-episode shaping was **14–18x** the terminal magnitude
(up from 8.6x) — driven by episodes getting longer as the agent improved
(130→~300 turns), not by per-decision magnitudes growing.

- **Finding 5**: attack dominance is a **problem**, not just a number —
  its per-turn reward is almost always positive (favorable attacks,
  conquest bonuses), so "keep attacking" is rewarded every turn regardless
  of whether the game is being closed out — classic farming risk.
- **Finding 6**: no per-*return* normalization exists, only per-step. With
  `gamma = 0.99` and ~1000+ transitions/episode, a terminal ±100 seen from
  an early state is discounted by `0.99^1000 ≈ 4e-5` — effectively
  invisible next to dense, immediate shaping.
- **Finding 7**: the eval_win_rate "decline" the user asked about is mostly
  **eval noise, not regression** — eval is only 6 fixed games (`evaluator.py`
  hardcodes 2 suites × 3 seeds), so every value is a multiple of ~1/6 with
  enormous variance. The trustworthy signal, `win_rate_last_50` (50-game
  rolling, against a randomized 3–6 opponent mix), rose from ~0.17–0.33 to
  ~0.47–0.54 and then **plateaued** — real learning, stalled around
  coin-flip despite strong board play. That plateau-with-strong-play is the
  behavioral fingerprint of Findings 5–6.

**Counterpoint, weighed before acting on any of the above:** `win_rate_last_50`
reaching ~0.5 against a heuristic pool (random baseline ~17–33% at 3–6
players) is a genuinely strong result. Attack dominance and the shaping/terminal
ratio are *latent* risks, not *demonstrated* failures — they'd only be active
problems if they were suppressing win rate, and the agent wins. **Decision
(2026-07-04): do not rewrite reward logic (no potential-based shaping/PBRS
for now); prefer the smallest reversible constant tweaks**, revisiting PBRS
only if the goal shifts to breaking past this plateau, beating specific
stronger opponents, or scaling to self-play. A "minimal tweak set" targeting
attack's positive bias was drafted (cut `_CONQUER_TERRITORY`/`_CONQUER_WITH_CARD`,
raise `_RATIO_THRESHOLD`) but **not yet applied** — still open, see below.

## Finding 8 (2026-07-04) — continent-completion incentive too weak, by formula

User's read of trained behavior: continent conquest "doesn't influence
enough." Confirmed by tracing the formulas, not just the W&B totals (continent
terms are folded into the single `attack`/`reinforce`/`fortify` buckets, not
separately logged):

- The domination gate (`REWARD_ATTACK_CONTINENT_DOMINATION`/`_ADVANTAGE`)
  doesn't open until `owned/total >= 1/alive_players + 0.10` — for Asia (12
  territories) at 4 players, the first **~5 conquests score zero**
  continent-specific reward, identical to attacking anywhere else.
- Even once open, the dribble (a few tenths) is small next to the ~5.4 a
  plain favorable attack already nets from `_RATIO`/`_ARMY_TRADE`/
  `_CONQUER_TERRITORY`/`_CONQUER_WITH_CARD` alone.
- The one big number, `REWARD_ATTACK_CONTINENT_CAPTURED = 4.00`, is a single
  event on the very last territory — too sparse to shape the many
  "which territory first" decisions leading up to it.
- `_continent_push` (feeds `REWARD_REINFORCE_CONTINENT_PUSH`/
  `_FORTIFY_CONTINENT_PUSH`) divides by continent size *twice*, so it's
  weakest for exactly the continents worth the most (Asia: `≈0.07` even at
  11/12 owned).

**Not yet applied** — proposed fix directions (loosen/remove the domination
gate threshold, weight `_continent_advantage`/`_continent_push` by
`continent_bonus` instead of diluting by size, or simply raise the
domination/advantage constants) are recorded but untested. **This is a
different problem from Finding 5** — Finding 5 is attack's share of *all*
shaping; this is how little of attack's *own* budget is continent-aware.
Only the *defense* half of "continent reward" has since been addressed
(Finding 10, `REWARD_CONTINENT_LOST`) — the *offense* half described here is
still open.

## Finding 9 (2026-07-04) — holding a territory scored nothing; implemented `REWARD_TERRITORY_HOLD`

Question: the game already grants more reinforcement armies for
territories/continents held — is that implicit incentive enough to teach
defense on its own? Checked by computing the per-*turn* (not per-episode)
territory/continent erosion rate across training:

| episodes | territory lost/turn | continent lost/turn | win rate |
|---|---|---|---|
| 1–100 | −0.0050 | −0.0013 | ~1% |
| 201–300 | −0.0059 | −0.0045 | 31–33% |
| 401–500 | −0.0049 | −0.0039 | 46–52% |
| 501–539 | −0.0046 | −0.0036 | 46–59% |

Win rate climbed ~50 points but the erosion rate stayed flat (territory) or
got *worse* (continent) — no sign the implicit "more armies next turn" path
teaches the network to value holding. The win-rate gain almost certainly
came from improved offense, not defense — the implicit path is not, in
practice, enough.

**Implemented:** `REWARD_TERRITORY_HOLD = 0.05`, unconditional every learner
turn (see the [end-of-turn reference](#end-of-turn-opponent-impact--holdlost-terms)
above for the formula). Tests:
`test_end_of_turn_hold_bonus_fires_even_without_territory_change`.

## Finding 10 (2026-07-04) — losing a continent barely stung; implemented `REWARD_CONTINENT_LOST`

Same weakness as Finding 9, for continents: `REWARD_CONTINENT_DELTA_RELATIVE`
is symmetric (penalizes a loss exactly as much as it rewards the equivalent
gain) and small, so giving up a hard-won continent barely registered against
the generic per-attack reward that built it.

**Implemented:** `REWARD_CONTINENT_LOST = 5.00`, fires only when continent
bonus share drops this round, stacking on the symmetric term (see the
[end-of-turn reference](#end-of-turn-opponent-impact--holdlost-terms) above).
Losing Asia (bonus 7 of 24 total) now costs `continent_delta (≈−0.73) +
continent_lost (≈−1.46) ≈ −2.19`, vs. only `≈+0.73` for gaining it — value
weighted for free by the same `/total_bonus` division, so losing Asia stings
~3.5x more than losing Australia. Unlike `territory_hold`, this adds **no**
new farming risk — it's a loss-only penalty, never an always-positive
per-turn bonus. Tests:
`test_end_of_turn_continent_loss_applies_extra_penalty`,
`test_end_of_turn_no_continent_loss_penalty_when_holding`.

## Finding 11 (2026-07-04) — eliminating an opponent underweighted; implemented base + per-card split

User's read: elimination happens rarely, and the old formula
(`PER_CARD * (cards_taken + 1)`, 1.25–5.00) felt like too small a push for
one of the game's real end-state objectives. Confirmed by magnitude: a
single *ordinary* favorable attack already nets ~5.4 from generic terms —
eliminating a whole opponent (permanently removing a threat, transferring
all their territories/cards) scored in that *same range*, and *less* than
finishing a continent (4.00) in the common 0–1 card case.

**Implemented:** split into `REWARD_ATTACK_ELIMINATE_OPPONENT_BASE = 4.00`
(new, flat, on par with continent-capture) + `_PER_CARD = 1.50` (was 1.25,
now applied to `cards_taken` directly, not `cards_taken + 1`). A 0-card
elimination goes from 1.25 → **4.00**; a 3-card elimination goes from 5.00 →
**8.50**, now clearly the single biggest per-attack event in the reward.
Test: `test_attack_eliminate_opponent_scales_with_cards_taken`.

**Resolved (2026-07-04):** `RewardCalculator._attack(...)` now returns
`(total_reward, eliminate_component)` — `compute()` subtracts the eliminate
portion out of the logged `attack` bucket and adds it as its own `eliminate`
bucket in `self.last_components` (the actual `reward` total, and hence
training behavior, is unchanged — this only affects what's logged). Shows up
automatically as `reward_component_eliminate` in W&B via the existing
generic `_accumulate_reward_components` loop, no trainer change needed.
Elimination frequency/magnitude is now directly visible, independent of
ordinary conquest reward. Test:
`test_attack_eliminate_opponent_scales_with_cards_taken` extended to assert
`last_components["eliminate"]`/`["attack"]` split correctly.

## Status before the next test (2026-07-04)

Three changes are stacked for the next run: `REWARD_TERRITORY_HOLD`
(Finding 9), `REWARD_CONTINENT_LOST` (Finding 10), and the eliminate
base+per-card split (Finding 11). Consistency check before running it:

- **They compose safely.** All three are purely additive, gated
  independently, no double-application of the same event — verified by the
  full test suite (240 passed, 1 skipped for an unrelated `torch_geometric`
  environment gap). Losing a continent-completing territory now triggers up
  to three end-of-turn terms at once (`territory_delta`, `continent_delta`,
  `continent_lost`) — intentional, not redundant: each scores a different
  axis (board-wide share, continent-bonus share, an extra loss-only
  kicker), and the combined magnitude (≈−2.2 for losing Asia) is
  proportioned against the +4.00 it cost to capture it, not runaway.
- **This is 3 changes in one test, against this doc's own rollout
  discipline** ("change one cluster at a time"). Reasonable here because
  the three are small, orthogonal, and each is now separately logged
  (`reward_component_territory_hold`, `_continent_lost`, `_eliminate`), so
  post-hoc attribution is possible for all three, even though the run
  itself still tests them together rather than one at a time.
- **Not addressed by these three changes, still open:**
  - **Finding 8** (continent completion too weak *on offense* — the
    domination gate, `_continent_advantage`, `_continent_push`). Only the
    *defense* half (Finding 10) has been done. If "continent reward" was
    expected to be fully fixed, it isn't yet.
  - **Findings 5–6** (attack ≈82% of shaping, shaping 14–18x the terminal
    anchor). None of the three changes reduce attack's share — the
    elimination boost (Finding 11) is itself part of the `attack` bucket, so
    it very slightly *adds* to attack's total magnitude, working against
    Finding 5/6's direction even while serving a different, legitimate goal
    (make elimination matter more). Not a conflict, just worth knowing
    these two efforts pull in different directions on that one metric.
  - **`REWARD_TERRITORY_HOLD` adds a new always-positive per-turn term**,
    which nudges the already-flagged shaping/terminal ratio (Finding 6)
    slightly further from the "sparse stays the anchor" principle, even
    though it's kept deliberately small.

**Suggestions before running the test:**

1. ~~Add a dedicated `reward_component_eliminate` breakout~~ **Done
   (2026-07-04)** — see Finding 11's "Resolved" note above.
2. When reviewing the run, check `reward_component_territory_hold`,
   `_continent_lost`, and `_eliminate` shares directly (all three now
   logged) rather than only `win_rate_last_50` — since 3 changes are
   stacked, the per-component breakdown is what lets you tell which one is
   actually doing something.
3. Findings 5–6 (attack dominance) and Finding 8 (continent-offense
   weakness) are both still fully open — worth deciding whether they're in
   scope for a follow-up test before or after this one, rather than by
   default, since every additional stacked change makes attribution harder.
4. Preserve the current best checkpoints under their existing run ids before
   starting a new one, so falling back to a known-good agent stays possible.
