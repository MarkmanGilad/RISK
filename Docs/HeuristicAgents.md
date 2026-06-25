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

These three (plus `AttackAgent`/`BSRAgent`/`ContinentAgent`/`ShapeAgent`)
form the curriculum of fixed opponents for self-play training.

## Shared scoring helpers

Module-level functions used across the hierarchy:
- `_owned_indices` — territory indices owned by a player.
- `_is_border` — whether a territory has an enemy-owned neighbor.
- `_bsr` — Border Security Ratio: adjacent enemy armies / own armies.
- `_enemy_neighbor_ratio` — fraction of neighbors that are enemy-owned.
- `_continent_attack_value` — reward for attacking into a continent the
  agent is close to completing (full bonus if it's the last enemy
  territory).
- `_continent_defense_value` — reward for holding territories in a
  continent the agent owns or partially owns.
- `_compactness_after_take` — fraction of a prospective conquest's
  neighbors that would already be friendly.
- `_owned_distances` — BFS distances from a target through
  same-owner territory only, used to find fortify donors.
- `_normalize` — clamps/rescales a value into `[0, 1]` given `(lo, hi)`.

## Related docs
See `Docs/RL-Prep-Changes.md` for the history of how dice-cap constants
and roll-outcome tables were centralized into `risk/constants.py` and
imported by this module.
