# Heuristic Agents

`risk/agents/heuristic_agent.py` implements non-learning opponents used for
self-play and as a curriculum of RL training opponents. They all share the
`BaseAgent` contract (`risk/agents/base_agent.py`): callable as
`agent(events, state) -> Action | None`, ignoring `events`, reading legal
moves from the `Environment` they hold, and returning a move immediately.

## Battle math

### `attacker_roll_edge(attacker_dice, defender_dice)`
Looks up `ATTACKER_ROLL_EDGE` (`risk/constants.py`) — the exact probability
that the attacker inflicts more casualties than the defender on a single
roll, for each `(attacker_dice, defender_dice)` pair.

### `battle_win_probability(attacker_armies, defender_armies)`
Probability that the attacker eventually conquers the territory if both
sides always commit full force on every roll. Base cases: `0.0` if the
attacker has 1 army (can't roll), `1.0` if the defender has 0.

Computed bottom-up with a persistent module-level cache
(`_battle_win_cache`), not by direct recursion. Every entry in
`ROLL_OUTCOMES` (`risk/constants.py`) strictly reduces
`attacker_armies + defender_armies` by at least 1, so a naive recursive
version is correct but its call depth scales with total armies — large
army stacks in long games can exceed Python's recursion limit. The
iterative version fills `(a, d)` cache entries for `a` in increasing order
(and `d` increasing within each `a`), so every recursive sub-case it reads
(`_battle_win_lookup`) is already resolved, with no call stack involved.

## `HeuristicWeights`

A frozen dataclass of five weights (`attack_odds`, `attacker_surplus`,
`continent`, `bsr`, `compactness`) that every agent below combines
differently to score territories and attacks. Don't construct ad-hoc
dataclasses elsewhere for similar purposes — reuse this one.

## Agent hierarchy

All concrete agents subclass `AttackAgent`, overriding `weights`,
`attack_threshold`, and usually `_territory_score` to change what they
value.

### `AttackAgent` (base behavior)
- **Trade-in:** plays the first legal `TradeInAction` if any exist,
  otherwise skips.
- **Reinforce:** ranks owned territories by `_territory_score` (BSR +
  army count) and stacks the bulk of the budget onto the top ~3, front-
  loading the highest-ranked one.
- **Attack:** only considers attacks committing full available dice
  (`_is_full_force_attack`) whose `battle_win_probability` exceeds
  `attack_threshold`; scores candidates via `_attack_score` (blend of win
  odds, one-roll edge, post-attack army surplus, continent value,
  compactness) and picks the best, breaking ties randomly. Stops attacking
  if nothing clears the bar.
- **Occupy:** always moves the maximum allowed army count into the
  conquered territory.
- **Fortify:** moves spare armies (`armies - 1`) from the strongest
  non-border donor toward the weakest border territory, using BFS
  (`_owned_distances`) to find donors reachable through owned territory.

### `BSRAgent(AttackAgent)`
Re-weights territory scoring toward Border Security Ratio (enemy armies
adjacent / own armies) plus enemy-neighbor ratio — prioritizes shoring up
threatened borders over continent or shape concerns.

### `ContinentAgent(BSRAgent)`
Adds a bonus to `_territory_score` for territories in a continent the
agent already fully owns, scaled by that continent's bonus value —
nudges reinforcement/fortification toward defending completed continents.

### `ShapeAgent(BSRAgent)`
Adds an enemy-neighbor-ratio term scaled by `weights.compactness` —
prefers consolidating a compact, low-perimeter border.

### `CompositeAgent(ShapeAgent)`
Generic weighted blend of BSR, continent defense value
(`_continent_defense_value`), and enemy-neighbor ratio. Accepts an
optional `weights: HeuristicWeights` override at construction, making it
the configurable base for the named opponent personalities below.

### Named opponent personalities (all `CompositeAgent`)
| Agent | `attack_threshold` | Personality |
|---|---|---|
| `RaiderAgent` | 0.45 | Aggressive — high `attack_odds`/`attacker_surplus`, low `bsr`/`compactness`; takes marginal fights to expand fast. |
| `SentinelAgent` | 0.62 | Defensive — high `bsr`/`compactness`, low `attack_odds`; reinforces and waits. |
| `EmpireAgent` | 0.52 | Continent-focused — high `continent` weight; chases and defends continent bonuses. |

These three (plus `AttackAgent`/`BSRAgent`/`ContinentAgent`/`ShapeAgent`, and
`KillbotAgent` below) form the curriculum of fixed opponents for self-play
training.

### `KillbotAgent(AttackAgent)`

A Python reimplementation of the Lux *Killbot* heuristic (BetterPixie
continent strategy + Vulture weak-player elimination + SmartAgentBase
attack/card/HogWild utilities). Unlike `BSRAgent`/`ContinentAgent`/
`ShapeAgent`, which only override `_territory_score`, this one overrides
`_choose_trade`, `_reinforce`, `_attack`, and `_occupy` too, since it needs
per-turn state (which continents it's targeting, whether it's mid
weak-player-elimination) that the shared scoring helpers alone can't
express. See "`KillbotAgent` — Lux Killbot reimplementation" below for the
full spec, source references, and fidelity notes; short version:

- **Reinforce:** Vulture-first — if a clearly weakest opponent exists, is
  killable, and we already border their territory, dump the whole budget
  there and enter kill mode for this turn. Otherwise fall back to
  `AttackAgent._reinforce`'s top-3 stacking, but with `_territory_score`
  boosted for continents it's targeting or defending
  (`_select_continents`/`_continent_needs_help`).
- **Attack:** kill mode (greedy — always retarget the weakest reachable
  territory owned by the target, not a precomputed route) → opportunistic
  continent steal (`_continent_steal_attack`) → in-continent attacks →
  general attacks → guaranteed-conquest fallback (`_attack_for_card`) if
  nothing was taken yet this turn. HogWild (armies exceed all enemies
  combined) drops the threshold to 0.
- **Occupy:** moves in the full stack only if the just-conquered territory
  has ≤1 remaining enemy neighbor; otherwise moves in half and keeps a
  reserve.
- **Trade-in:** prefers the legal set that grants the most territory-bonus
  armies, instead of just the first valid set.

**Part of the training curriculum and always in eval.** Registered as
`agent_kind = "killbot"` (`risk/game/player.py`'s `AGENT_KIND_ORDER`, wired
into `GameFactory.build_agents`); included in
`risk/learning/train_constants.py`'s `TRAIN_OPPONENT_AGENT_KINDS` (so
`Trainer._assign_random_opponents` can sample it for any non-learner seat
like the other three), and added to **both** of `Evaluator`'s fixed eval
suites (`risk/learning/evaluator.py`'s `_EVAL_SUITES`) rather than left to
random sampling, so every eval run — not just some — measures the model
against it. See `Docs/Eval.md`'s "Eval games and opponents" for the updated
suite compositions.

## `KillbotAgent` — Lux Killbot reimplementation

> Status: **implemented** in `risk/agents/heuristic_agent.py` (see the agent
> hierarchy entry above for the short version). The rest of this section is
> the original design spec, source references, and the fidelity decisions
> made along the way — kept as the design record for why the code looks the
> way it does, not a to-do list.

### Why

Killbot is the strongest built-in AI in the *Lux* Risk engine and the main
non-learning benchmark in A. Carr's D.A.D thesis (D.A.D won ~35.3% in the
final benchmark, "nearly double" Killbot; Carr calls Killbot "the strongest
of the inbuilt Lux AIs"). Because it is a hand-coded heuristic rather than a
learning agent, we can reimplement it faithfully in Python as a strong,
fixed benchmark / curriculum opponent. It must be labelled **"a Python
reimplementation of the Lux Killbot heuristic"**, not "the original
Killbot", unless the unchanged Java agent is actually run.

Per the Lux SDK, Killbot is a composition rather than a single class:

```
Killbot = BetterPixie continent strategy
        + Vulture weak-player elimination strategy
        + generic SmartAgentBase attack / card / hogwild utilities
```

### Source references (Lux SDK)

- Overview of Lux AIs (Killbot = Vulture + BetterPixie):
  https://raw.githubusercontent.com/sillysoft/LuxSDK/master/AI_examples.html
- `BetterPixie` (continent selection, placement, attack, fortify):
  https://raw.githubusercontent.com/sillysoft/LuxSDK/master/src/com/sillysoft/lux/agent/BetterPixie.java
- `Vulture` (weak-player elimination, kill routes):
  https://raw.githubusercontent.com/sillysoft/LuxSDK/master/src/com/sillysoft/lux/agent/Vulture.java
- `SmartAgentBase` (card cashing, `attackForCard`, HogWild, utilities):
  https://raw.githubusercontent.com/sillysoft/LuxSDK/master/src/com/sillysoft/lux/agent/SmartAgentBase.java

### The Killbot heuristic (short version)

```
1. Prefer continents that are easy to take/hold (fewest border points).
2. Place armies to take or defend those continents.
3. Always look for weak players to eliminate, especially card-rich ones.
4. Attack only when the army ratio is favorable.
5. Try to conquer at least one territory per turn to earn a card.
6. Move armies toward borders / front lines.
7. If overwhelmingly stronger than everyone combined, attack all-out.
```

### The seven Killbot sub-heuristics (verified against the Lux source, 2026-07-04)

1. **Continent selection (BetterPixie `setupOurConts`).** Per turn, estimate
   armies needed for each continent as `enemyArmiesInCont * 1.3 -
   (ourArmiesInOrNearCont + routeArmies * 1.2)` — note the source discounts
   *both* nearby friendly armies **and** armies along the cheapest route
   into the continent (route armies weighted `1.2`, not just a flat
   subtraction of "nearby" armies as the earlier draft of this doc implied).
   Only commit to a continent whose cost fits a per-turn budget
   (`numberOfArmies / (numContinents/4)`) — this budget only gates *taking
   on additional* continents once we already hold one; if we own none, we
   always focus the single cheapest positive-cost continent regardless of
   budget. A continent already held is also re-checked via
   `continentNeedsHelp()`: it needs help if we don't fully own it yet, or if
   any of its border countries has fewer than `borderForce = 20` armies and
   we don't also control every continent surrounding it.
2. **Placement (Vulture first, else BetterPixie `placeArmiesToTakeCont` /
   `placeNearEnemies`).** `setToKillPlayer` discounts each opponent's army
   count by the card reward for killing them (`armies -= cardsWorth *
   cards/3`) and picks the lowest-discounted player, but only if they are
   also actually killable this way:
   `getPlayersBiggestArmyWithEnemyNeighbor(us).armies + numberOfArmies >
   theirTotalArmies + theirTerritoryCount` (our strongest attacking stack
   plus this turn's placement must be able to out-fight their whole empire
   territory-by-territory). If that holds **and** `ourArmies > 2 *
   weakestDiscounted`, Vulture stages armies on a single owned country
   chosen by comparing the target cluster's total defense against the
   staging route's cost (not simply "nearest territory") and records the
   route; sets a `placedToKill` flag. Otherwise delegate to BetterPixie:
   place into continents that still `continentNeedsHelp()` via
   `placeArmiesToTakeCont`, then spend any remainder with `placeNearEnemies
   (minimumToWin=true)`, splitting across enemy clusters weighted toward the
   weakest ones and stacking each share on our strongest neighboring
   country.
3. **Attack (kill route, else continent, else opportunistic continent
   steal).** If placed-to-kill, follow the saved route attacking till dead
   (`attackAlongRoute`). Otherwise attack inside chosen continents: first
   from countries with exactly one useful enemy neighbor, then any attack
   where `ourArmies > enemyArmies * outnumberBy` (`outnumberBy = 1` in
   BetterPixie, `1.3` in the more aggressive EvilPixie — confirmed exact
   constant/formula: `c.getArmies() > adjoining[i].getArmies() *
   outnumberBy`). Independently, `takeOutContinentCheck()` also opportunistically
   attacks a poorly-defended enemy-held continent whenever we have roughly
   2x its total army count, even outside our chosen `ourConts` — a
   continent-steal check the earlier draft of this doc omitted entirely.
4. **Cards.** Cash the best available set (`cashCardsIfPossible` /
   `Card.getBestSet`). Guarantee ≥1 conquest per turn via `attackForCard`:
   if `!board.tookOverACountry()` yet this turn, scan every owned country's
   neighbors for the best `us.armies / them.armies` ratio and attack the
   best pair if `bestUs.armies > bestThem.armies * outnumberTimes`. In our
   environment the "took over a country this turn" flag already exists as
   `State.conquered_this_turn` (maintained by `Environment`), so this gate
   needs no new bookkeeping. The Vulture card discount is also fully
   available: every player's hand is in `State.hands`, and the next set's
   value is `card_set_value(state.cards_traded_in_count)` (from
   `risk/constants.py`) — cards are **not** hidden information here.
5. **HogWild endgame.** `hogWildCheck`: if `ourArmies > sum(all enemy
   armies)` (confirmed exact: `armies[ID] > enemyArmies`, no margin), attack
   as much as possible. Crude but decisive; Carr notes the built-in Lux AIs
   need this because they do not search ahead.
6. **Occupy / move-in (`SmartAgentBase` `moveInMemory`).** Distinct from
   Fortify below — this fires immediately after each successful conquest,
   not once at end of turn. Lux does **not** always dump the maximum legal
   army count into a freshly conquered territory the way our current
   `AttackAgent.Occupy` does: if the remaining hostile countries around the
   new front form one unified cluster it moves the full attacking stack in,
   otherwise it moves in only `armies / 2` (`moveInMemory == -2`, "move
   half") to keep a reserve for the next fight. This is a real behavioral
   gap versus our agents, worth closing for fidelity.
7. **Fortify (end of turn).** If we own a continent, move interior armies
   outward to its border countries (`fortifyContinent`); on continents we
   don't own, move scattered armies toward positions that border the
   weakest enemy neighbor (`fortifyContinentScraps`).

### How Killbot compares to our current agents

Our `AttackAgent` family and Killbot share the "attack on favorable odds +
value continents + fortify to borders" skeleton, but differ sharply on the
Risk *economy* (cards, elimination) and on the fight math.

| Killbot pillar | Lux implementation | Our current agents | Gap |
|---|---|---|---|
| Battle decision | Army ratios: `attacker > defender * outnumberBy` (1.0–1.3); continent estimate `enemy * 1.3` | Exact multi-roll `battle_win_probability` + `attacker_roll_edge` tables, thresholded | **We are stronger** — our fight math is exact, not a ratio |
| Continent strategy | `setupOurConts`: per-turn affordability budget, `needed = enemy*1.3 - (ourNearby + route*1.2)`, focus easiest positive continent when owning none, `continentNeedsHelp` re-checks held continents against a `borderForce = 20` floor | `_continent_attack_value` / `_continent_defense_value` as weighted terms | Partial — we value continents but never compute per-turn affordability or a border-army floor |
| Continent steal | `takeOutContinentCheck`: opportunistically attacks a poorly-defended enemy continent at ~2x advantage, even outside chosen `ourConts` | None | **Missing** |
| Placement | Vulture-first kill route (gated by both the 2x-discounted-army check *and* a can-we-actually-out-fight-their-empire check), else BetterPixie continent placement | Rank owned territories by score, stack budget on top ~3 | **Missing the Vulture layer** |
| Player elimination for cards | Discount armies by `cardsWorth*cards/3`; kill if `ours > 2 * weakest` **and** our biggest bordering stack + this turn's placement can out-fight their whole empire | None | **Missing** — major strategic hole |
| Guaranteed card/turn | `attackForCard`: best safe attack if no conquest yet | Cash handed cards only; no forced conquest for a card | **Missing** |
| Card trade-in | `getBestSet` (best set) | First legal `TradeInAction` | Minor (env may auto-optimize) |
| HogWild endgame | If `ours > all enemies combined`, attack all-out | Keep attacking at the *same* threshold; no all-out mode | Partial — we never lower the bar to close |
| Multi-step kill routes | `attackAlongRoute` attack-till-dead | Greedy single best attack | **Missing** planning |
| Occupy / move-in | Full stack only if remaining hostiles form one cluster, else move in only `armies/2` to keep a reserve | `AttackAgent.Occupy` always moves the max allowed | **We over-commit** — always dumping max armies in wastes the "keep a reserve" behavior Killbot relies on |
| Fortify | Toward owned-continent borders; scraps toward weak enemies | BFS spare armies → weakest border | Roughly equivalent |

**Verdict.** Head-to-head, a faithful Killbot would likely beat our current
agents — not on tactics (our per-fight math is better) but on *economy*:
cards are Risk's largest army source, and Killbot both guarantees a card per
turn and hunts weak players to steal theirs, then closes with HogWild. Our
agents model none of that. For their actual purpose (a diverse, tunable RL
opponent curriculum) our agents remain the right design; Killbot is a single
monolithic benchmark policy, not a curriculum.

### Build plan

Reuse existing infrastructure wherever possible — do **not** duplicate the
battle math or the scoring helpers. Prefer our exact
`battle_win_probability` over Lux's crude ratios so the reimplementation is
both faithful in *behavior* and stronger in *fight selection*.

1. **Class shape.** Add `KillbotAgent(AttackAgent)` in
   `risk/agents/heuristic_agent.py`, reusing `battle_win_probability`,
   `_owned_indices`, `_owned_distances`, `_bsr`, `_continent_*` helpers, and
   `topology.continent_owner_counts` / `continent_bonus`. Keep the
   `BaseAgent` callable contract and phase dispatch from `AttackAgent.act`.
2. **Continent affordability (BetterPixie `setupOurConts` /
   `continentNeedsHelp`).** Add a per-turn method that, for each continent,
   estimates `needed = enemy_armies * 1.3 - (our_nearby_armies +
   route_armies * 1.2)`, and marks continents that fit a simple budget
   (`numberOfArmies / (numContinents/4)`) as "ours" for this turn — but only
   when we already hold at least one continent; with none held, always
   target the single cheapest positive-cost continent regardless of budget.
   Re-check continents we already fully own with a `borderForce = 20` floor
   on their border territories (skip the floor if we also hold every
   surrounding continent). Store the chosen set for the attack phase (reset
   each of our turns).
3. **Vulture placement (`setToKillPlayer` / `placeToKill`).** Add a
   pre-placement check. All inputs are directly available — cards are full
   information in our environment: discount each opponent's army total by
   `card_set_value(state.cards_traded_in_count) * (len(state.hands[i]) / 3)`
   (mirrors Lux's `cardsWorth * cards/3`), so no hidden-info fallback is
   needed. Gate the kill decision on **both** conditions Lux uses:
   `ours > 2 * weakestDiscounted`, and our strongest owned territory
   bordering that player plus this turn's placement budget must exceed
   their total armies plus their territory count (a can-we-actually-clear-
   their-empire check, not just an army-count comparison). If both hold,
   stage the whole placement budget on the owned territory that minimizes
   staging-route cost into their cluster (not simply "nearest") and record
   the kill route. Per **Decisions** below, the route is a **greedy
   nearest-enemy chain** (repeatedly step to the weakest adjacent
   target-owned territory, reusing `_owned_distances`-style BFS), not a true
   optimal `CountryRoute` — Lux itself bails on exact routes for large
   clusters. Else delegate to the continent placement of step 2.
4. **Attack policy.**
   - If placed-to-kill: walk the saved route, attacking each step at full
     force while `battle_win_probability` stays positive (attack-till-dead).
   - Else: attack inside chosen continents, prioritizing our territories
     with exactly one enemy neighbor, using `battle_win_probability >
     threshold`. Per **Decisions** below we standardize on
     `battle_win_probability` thresholds everywhere instead of porting Lux's
     `outnumberBy` ratios — there is no fixed ratio→probability mapping (it
     drifts with army scale, see Decisions), so set
     `KillbotAgent.attack_threshold` empirically like the other
     `CompositeAgent` personalities (0.45–0.62 range) rather than deriving it
     from `outnumberBy`.
   - Continent steal (`takeOutContinentCheck`): independent of `ourConts`,
     opportunistically attack any enemy-held continent when our nearby
     armies roughly double its total defense.
   - `attackForCard`: if no conquest happened this turn, make the single
     best positive-probability attack to secure a card.
   - HogWild: if our total armies exceed all enemies combined, drop the
     threshold toward 0 and keep attacking while any positive-EV attack
     exists.
5. **Occupy override (`moveInMemory` half-move).** Override
   `AttackAgent`'s always-max Occupy: after a conquest, keep a reserve when
   the new front still faces multiple enemies. Per **Decisions** below, v1
   uses the cheap **enemy-neighbor-count approximation** rather than an exact
   component flood-fill: if the just-conquered territory still borders more
   than one enemy, move in only half (`armies // 2`); otherwise move the full
   attacking stack. Clamp the chosen count to the legal `[attacker_dice,
   armies - 1]` range that `_legal_occupy`/`_occupy_bounds` already yields
   (confirmed `hi = armies[from] - 1`, `lo = attacker_dice` clamped). This
   needs a plain neighbor-count check, not `_enemy_neighbor_ratio` directly
   (that helper returns a fraction and discards the count — see Decisions);
   inline the same one-line `sum(...)` it uses internally instead. No new
   traversal either way. (The exact component enumeration — a flood fill
   seeded on one enemy neighbor, filtered on enemy ownership, checking the
   other bordering enemies all land in the reached set — is deferred; add it
   only if the reserve behavior measurably matters in play.)
6. **Fortify — implemented for free.** No override needed: `AttackAgent._fortify`
   ranks border targets via `_territory_score`, which `KillbotAgent` already
   overrides to favor `_our_conts`, so the continent bias BetterPixie wants
   falls out of the existing method without new fortify-specific code.
7. **Turn-state bookkeeping — implemented, simpler than planned.** Only
   three attributes ended up necessary: `_our_conts`, `_kill_target`,
   `_placed_to_kill`. No separate `_kill_route` list (per the greedy-chain
   Decision, the target player id plus the live board is all a greedy step
   needs) and no `_to_kill_player`/`_took_country_this_turn` (the latter is
   `State.conquered_this_turn`, already maintained by `Environment`). Reset
   happens in `act()` on `Phase.TRADE_IN`, which `Environment._begin_turn_for`
   guarantees fires exactly once per turn — no need to watch `events` or
   diff `state.current_player_index`.
8. **Tests — ad-hoc only so far.** Verified with a `Docs/Testing.md`
   "ad-hoc verification" style script (`SelfPlay.play_headless` over 30
   mixed-roster games, 3–6 players, `KillbotAgent` vs.
   `RaiderAgent`/`SentinelAgent`/`EmpireAgent`): zero crashes, `KillbotAgent`
   won 14/30 as seat 0. No checked-in test yet — per `Docs/Testing.md`'s
   convention, promote this into `Temp/tests/test_agents.py` (covering
   continent affordability selection, the `borderForce` re-check, the
   Vulture kill trigger, `takeOutContinentCheck`, the Occupy half-move
   behavior, `attackForCard`, and HogWild) once the shape has settled from
   actual use.
9. **Docs — done.** Moved into the agent hierarchy section above; **not**
   added to the curriculum table (`risk/learning/trainer.py`'s
   `TRAIN_OPPONENT_AGENT_KINDS`) per instruction — implemented for
   testing/benchmarking only for now.

### Resolved during code review (2026-07-04)

These earlier open questions were checked against the codebase and are no
longer blockers:

- **Opponent card visibility — resolved (full information).** `State.hands`
  is `list[list[Card]]` holding *every* player's hand, so opponent card
  counts are `len(state.hands[i])`. The next set's value is
  `card_set_value(state.cards_traded_in_count)` (`risk/constants.py`).
  Vulture's discount is therefore fully and faithfully computable; drop any
  hidden-info fallback.
- **`attackForCard` "took a country this turn" gate — resolved.** The
  environment already tracks `State.conquered_this_turn` (reset in
  `Environment._begin_turn_for`), the exact analogue of Lux's
  `board.tookOverACountry()`. No new per-turn flag needed.
- **Partial Occupy / half-move-in — resolved (supported).**
  `Environment._legal_occupy` yields every `count` in
  `[attacker_dice, armies_in_from - 1]`, so moving in `armies // 2` (clamped
  to that range) is directly expressible.
- **"Biggest army with an enemy neighbor" query — resolved (no new
  helper needed).** `heuristic_agent.py` already has the two pieces:
  `_owned_indices(state, player_id)` to list a player's territories and
  `_is_border(state, topology, index, player_id)` to test whether one has an
  enemy neighbor. `max((i for i in _owned_indices(state, opp_id) if
  _is_border(state, topology, i, opp_id)), key=lambda i: state.armies[i])`
  answers Vulture's `getPlayersBiggestArmyWithEnemyNeighbor` exactly — don't
  write a new traversal for this, just compose the two existing helpers.

### Open questions still to resolve before coding

None outstanding — see **Decisions (2026-07-04)** below.

### Decisions (2026-07-04)

The three remaining design choices are settled. All three favor the cheap
option, and all three are defensible fidelity choices rather than shortcuts
— two of them (exact battle math, simple reserve rule) arguably make the
agent play *better* than Lux Killbot, not worse.

- **Kill-route representation → greedy nearest-enemy chain** (not a true
  optimal route). Lux itself caps route search at 20 countries because the
  exact version "can hang the app on large search-spaces" and bails to the
  backer agent for big clusters — the original authors already treat the
  true route as too expensive. From the staged territory, repeatedly attack
  the weakest adjacent target-owned territory, using `battle_win_probability`
  to decide whether to continue. Captures Vulture's practical value (find a
  weak player, walk in, take the cards) at a fraction of the complexity,
  reusing the existing `_owned_distances`-style BFS. Revisit only if
  benchmarking shows Killbot leaving eliminations on the table.
- **`outnumberBy` → `battle_win_probability` thresholds everywhere** (not
  Lux's raw army ratios). This is the whole reason a Python reimplementation
  can be *stronger* than the Java original: our fight math is exact, Lux's is
  a proxy. **Correction:** an earlier draft of this doc guessed
  `outnumberBy = 1.0 → ≈0.5` and `outnumberBy = 1.3 → ≈0.6–0.65`; actually
  running `battle_win_probability` shows that's off, and worse, there is no
  fixed ratio→probability mapping at all — the win probability at a constant
  army *ratio* rises with absolute army size (attacker gets 3 dice vs the
  defender's 2, so the edge compounds over more rounds):
  `10v10 (ratio 1.0) → 0.48`, `20v20 → 0.577`, `30v30 → 0.633`;
  `13v10 (ratio 1.3) → 0.723`, `39v30 (ratio 1.3) → 0.913`. A single scalar
  `attack_threshold` can't reproduce a fixed ratio across scales by
  construction — that's an accepted, documented consequence of using exact
  odds instead of a ratio, not a bug. Don't try to back-solve a threshold
  from `outnumberBy`; instead pick `attack_threshold` the same way the other
  `CompositeAgent` personalities do (empirically, via the existing
  `RaiderAgent`/`SentinelAgent`/`EmpireAgent` values as reference points —
  0.45–0.62) and tune from self-play results.
- **Enemy-cluster connectivity for Occupy → enemy-neighbor-count
  approximation** (not an exact component flood-fill), at least for v1. Use
  "does the just-conquered territory still border more than one enemy → keep
  a reserve (move half); otherwise move the full stack." **Correction:** an
  earlier draft cited `_enemy_neighbor_ratio` as already sufficient for this
  — it isn't quite: that helper returns `enemies / len(neighbors)`, a
  *fraction*, and discards the raw enemy count, so `> 1` can't be tested
  against it directly (you'd have to multiply back by `len(neighbors)` and
  worry about rounding, which is worse than just counting). What's actually
  needed is the same one-line count `_enemy_neighbor_ratio` computes
  internally before dividing. **As implemented:** rather than inlining that
  sum at the call site, it's a proper module-level `_enemy_neighbor_count`
  helper (matching the style of every other predicate in this module —
  `_is_border`, `_enemy_neighbor_ratio`, etc.), so `KillbotAgent._occupy`
  just calls it. Still zero new *traversal* code (no BFS/DFS, just a
  neighbor scan identical in shape to what `_enemy_neighbor_ratio`/
  `_is_border` already do). Captures the actual intent — don't
  over-commit a stack into a spot still facing multiple fronts. The exact
  component enumeration is a marginal fidelity gain over what was itself an
  approximation in Lux; add the true flood-fill only if the reserve behavior
  measurably matters in play.

## Shared scoring helpers

Module-level functions used across the hierarchy:
- `_owned_indices` — territory indices owned by a player.
- `_is_border` — whether a territory has an enemy-owned neighbor.
- `_bsr` — Border Security Ratio: adjacent enemy armies / own armies.
- `_enemy_neighbor_ratio` — fraction of neighbors that are enemy-owned.
- `_enemy_neighbor_count` — raw enemy-neighbor count (used by
  `KillbotAgent._occupy`; `_enemy_neighbor_ratio` discards this by dividing).
- `_continent_attack_value` — reward for attacking into a continent the
  agent is close to completing (full bonus if it's the last enemy
  territory).
- `_continent_defense_value` — reward for holding territories in a
  continent the agent owns or partially owns.
- `_compactness_after_take` — fraction of a prospective conquest's
  neighbors that would already be friendly.
- `_owned_distances` — BFS distances from a target through
  same-owner territory only, used to find fortify donors.
- `_surrounding_continents` — continent ids adjacent to a continent's
  border, used by `KillbotAgent._continent_needs_help`.
- `_normalize` — clamps/rescales a value into `[0, 1]` given `(lo, hi)`.

## Related docs
See `Docs/RL-Prep-Changes.md` for the history of how dice-cap constants
and roll-outcome tables were centralized into `risk/constants.py` and
imported by this module.
