# Environment action-space plan

This document records **future shared environment/action-space experiments**.
It is intentionally separate from algorithm-specific PPO/DQN plans. Any
change here alters the legal candidates seen by every learning agent.

## Planned: defer ordinary five-card trade-ins until the next turn

**Status:** implemented in `risk/game/environment.py` (2026-08-11). This
section is kept as the design record; see `Docs/ChangeLog.md` for the exact
files touched, including the consequential `MAX_TRANSIENT_HAND_SIZE` bump
(`risk/constants.py`) and its checkpoint-compatibility impact.

### Rule correction

When a player starts an attack turn with four cards and makes their first
ordinary conquest, the one conquest-card draw may bring the hand to five. The
player keeps that fifth card and completes the current attack/occupy/fortify
flow. At the start of that player's next turn, `TRADE_IN` requires a valid set
before reinforcement placement.

The existing immediate forced trade-in must remain for a different special
case: if a conquest eliminates another player and the eliminated player's
transferred cards leave the attacker with five or more cards, require trading
before resuming the parked `OCCUPY` action. The player must also place the
trade-in set armies before occupation resumes; otherwise the reinforcement
budget has no placement phase and is overwritten at a later turn start. This
avoids letting an elimination hand transfer bypass the maximum-card rule while
preserving both the pending conquest and the earned armies.

### Corrected defect

`Environment._apply_attack(...)` presently checks the final hand size after
both the first-conquest card draw and any eliminated-player card transfer. As
a result, an ordinary four-to-five-card conquest incorrectly switches from
`ATTACK` to `TRADE_IN` in the middle of the turn and the UI shows “you must
trade”. The check must distinguish an elimination transfer from a normal
conquest-card draw.

### Scope and implementation outline

1. In `Environment._apply_attack(...)`, record whether the defender was
   actually eliminated before deciding whether to interrupt the pending
   `OCCUPY` phase for a forced trade-in.
2. Preserve the existing first-conquest card draw and `conquered_this_turn`
   behavior. A normal conquest must proceed to `OCCUPY` regardless of whether
   the new hand size is five.
3. Preserve the elimination path: when an eliminated defender's transferred
   cards leave the attacker at five or more, enter `TRADE_IN` and keep
   `pending_attack`. While the hand remains at five or more, allow only valid
   `TradeInAction`s: each trade removes three cards, then re-check the hand
   before allowing placement or a resume. For example, nine cards require two
   consecutive valid trade-ins (9 -> 6 -> 3). After every required trade-in is
   complete, set
   `Phase.REINFORCE_PLACE` while retaining `pending_attack`; its budget is the
   accumulated trade value. This is the dedicated placement step. When that
   budget reaches zero, route directly back to `OCCUPY` rather than the normal
   `ATTACK` transition, and do not reset `conquered_this_turn`. The
   card-territory +2 bonuses remain immediate automatic placements on their
   matching owned territories, as they are today.
4. Do not use the ordinary turn-start reinforcement budget for this
   mid-attack placement: it must contain only the accumulated trade-in set
   value and must not recompute/reset the player's normal next-turn armies.
5. Keep the existing start-of-turn rule in `legal_actions()` and
   `_begin_turn_for(...)`: a five-or-more-card hand has no `SkipTradeAction`.
6. Read `Docs/Testing.md` before implementation and extend
   `Temp/tests/test_environment.py` with deterministic coverage for:
   - a player going from four to five cards by an ordinary conquest, remaining
     in `OCCUPY`, then being forced to trade only at their next turn; and
   - an elimination-with-card-transfer interruption where the selected trade
     set value is available for placement before `OCCUPY` resumes, then cannot
     be overwritten; and
   - a nine-card elimination hand that requires two consecutive trade-ins and
     exposes the sum of both set values for one placement step; and
   - the existing automatic +2 matching-territory-card bonus.
7. Update the card-limit comment in `risk/constants.py` and any card-rule
   documentation that currently describes a normal first-conquest draw as an
   immediate mid-attack forced trade-in.

## Planned: bucketed fortify amounts

**Status:** implemented 2026-08-11; fresh training run required.

### Current behavior

For each valid fortify source territory and reachable owned destination,
`Environment.legal_actions()` now offers:

- `skip fortify`; and
- the deduplicated `1 / half / maximum` transfer amounts.

The environment's rule validation continues to accept any positive amount up
to that maximum when an agent explicitly submits it. The bounded candidate set
lets learned agents choose a small move, rebalance, or maximum commitment
without enumerating every integer amount.

### Implemented behavior

For each valid source-destination pair, enumerate the deduplicated set:

```text
minimum = 1
maximum = armies_from - 1
middle = (minimum + maximum) // 2

amounts = sorted({minimum, middle, maximum})
```

Keep the existing skip action once per fortify decision. Do not change the game
rule: every submitted amount from `1` through `maximum` remains valid. This
only changes which actions are offered to learning agents.

Examples:

| Armies at source | Offered move amounts |
|---:|---|
| 2 | `1` |
| 3 | `1, 2` |
| 10 | `1, 5, 9` |

### Rationale

The present max-transfer-only choice can empty an important source border.
Adding small, medium, and maximum transfers gives the agent three meaningful
strategies—minor reinforcement, rebalance, or commitment—without enumerating
every integer army count.

Fortify candidates already span valid owned source-destination pairs, which
can be numerous. Enumerating all integer transfer amounts would grow with army
count and make action-injected graph scoring unnecessarily expensive. The
three-bucket scheme caps the extra non-skip candidates at roughly 3x the
current count.

### Implementation record

`Environment._legal_fortify(...)` emits the deduplicated amount buckets for
each existing valid pair. `FortifyAction` validation, action injection, game
rules, and agent classes remain unchanged because they already support every
valid count. `Temp/tests/test_environment.py` covers the `1 / half / maximum`
enumeration and a valid direct amount not in that set.

Measure legal-action counts and wall time during the fresh DQN_300 smoke run.
Action injection scores one graph per candidate, so this cost must be observed
rather than assumed negligible.

### Experiment rule

This is an environment/action-space change, not a PPO-only adjustment. Start
fresh DQN, Dueling DQN, or PPO runs after it; do not resume an existing
checkpoint and present it as the same experiment. Compare at matched learner
turns and processed samples, and report the candidate-action-count change.

The middle value is the integer floor `(1 + maximum) // 2`. Test a different
bucketing rule only as a separate experiment if it systematically
under-represents very large fortifications.
