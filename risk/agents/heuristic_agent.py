"""Heuristic Risk agents for self-play and RL opponent curricula.

The agents in this module use the same callable contract as RandomAgent, but
rank legal moves with classic Risk heuristics: attack odds, border security
ratio, continent completion pressure, and compactness.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.constants import (
    ATTACKER_ROLL_EDGE,
    MAX_ATTACK_DICE,
    MAX_DEFEND_DICE,
    ROLL_OUTCOMES,
    card_set_value,
)
from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    SkipTradeAction,
    StopAttackAction,
    TradeInAction,
)
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.state import State


@dataclass(frozen=True)
class HeuristicWeights:
    """Weights used by CompositeAgent to value territories and attacks."""

    attack_odds: float = 1.0
    attacker_surplus: float = 0.35
    continent: float = 0.7
    bsr: float = 0.8
    compactness: float = 0.35


def attacker_roll_edge(attacker_dice: int, defender_dice: int) -> float:
    """Return the attacker's chance to win more casualties than they lose."""

    return ATTACKER_ROLL_EDGE[(attacker_dice, defender_dice)]


_battle_win_cache: dict[tuple[int, int], float] = {}


def _battle_win_lookup(attacker_armies: int, defender_armies: int) -> float:
    if defender_armies <= 0:
        return 1.0
    if attacker_armies <= 1:
        return 0.0
    return _battle_win_cache[(attacker_armies, defender_armies)]


def battle_win_probability(attacker_armies: int, defender_armies: int) -> float:
    """Probability of conquering a territory while always rolling full force.

    Computed bottom-up (not via direct recursion): every roll outcome
    strictly reduces attacker_armies + defender_armies by at least 1, so a
    deep recursive version can exceed Python's call-stack limit once armies
    pile up into the hundreds during long training games.
    """

    if defender_armies <= 0:
        return 1.0
    if attacker_armies <= 1:
        return 0.0

    for a in range(2, attacker_armies + 1):
        for d in range(1, defender_armies + 1):
            if (a, d) in _battle_win_cache:
                continue
            attacker_dice = min(MAX_ATTACK_DICE, a - 1)
            defender_dice = min(MAX_DEFEND_DICE, d)
            _battle_win_cache[(a, d)] = sum(
                probability * _battle_win_lookup(a - attacker_losses, d - defender_losses)
                for attacker_losses, defender_losses, probability in ROLL_OUTCOMES[
                    (attacker_dice, defender_dice)
                ]
            )
    return _battle_win_cache[(attacker_armies, defender_armies)]


class AttackAgent(BaseAgent):
    """Attack when the one-roll odds are favorable; otherwise pass.

    Reinforcement and fortification still use light border-aware behavior so
    the agent can complete full games without falling back to random choices.
    """

    attack_threshold = 0.50
    weights = HeuristicWeights(
        attack_odds=1.0,
        attacker_surplus=0.2,
        continent=0.15,
        bsr=0.25,
        compactness=0.0,
    )

    def __init__(
        self,
        player_id: int,
        env: Optional[Environment] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id)
        self.env = env
        self._rng = random.Random(seed)

    def act(self, events: Sequence[object], state: State) -> Optional[Action]:
        if self.env is None:
            return None

        legal = self.env.legal_actions()
        if not legal:
            return None

        if state.phase is Phase.TRADE_IN:
            return self._choose_trade(legal) or SkipTradeAction()
        if state.phase is Phase.REINFORCE_PLACE:
            return self._reinforce(state)
        if state.phase is Phase.ATTACK:
            return self._attack(state, legal)
        if state.phase is Phase.OCCUPY:
            return self._occupy(legal)
        if state.phase is Phase.FORTIFY:
            return self._fortify(state)
        return self._rng.choice(list(legal))

    def _choose_trade(self, legal: Sequence[Action]) -> Optional[TradeInAction]:
        trades = [a for a in legal if isinstance(a, TradeInAction)]
        if trades:
            return trades[0]
        return None

    def _reinforce(self, state: State) -> Optional[ReinforcementAction]:
        assert self.env is not None
        pid = state.current_player_index
        budget = state.reinforcement_budget
        if budget <= 0:
            return None

        owned = _owned_indices(state, pid)
        if not owned:
            return None

        targets = sorted(
            owned,
            key=lambda i: self._territory_score(state, i),
            reverse=True,
        )
        if not targets:
            return None

        placements: dict[str, int] = {}
        remaining = budget
        top_count = min(3, len(targets))
        for rank, idx in enumerate(targets[:top_count]):
            if rank == top_count - 1:
                count = remaining
            else:
                count = max(1, budget // (2 + rank * 2))
                count = min(count, remaining - (top_count - rank - 1))
            remaining -= count
            placements[self.env.topology.territory_at(idx)] = count

        return ReinforcementAction(placements=placements)

    def _attack(self, state: State, legal: Sequence[Action]) -> Action:
        attacks = [a for a in legal if isinstance(a, AttackAction)]
        candidates = [
            (self._attack_score(state, a), a)
            for a in attacks
            if self._is_full_force_attack(state, a)
            and self._battle_odds(state, a) > self.attack_threshold
        ]
        if not candidates:
            return StopAttackAction()
        best_score = max(score for score, _ in candidates)
        best = [a for score, a in candidates if score == best_score]
        return self._rng.choice(best)

    def _occupy(self, legal: Sequence[Action]) -> Optional[OccupyAction]:
        occupiers = [a for a in legal if isinstance(a, OccupyAction)]
        if not occupiers:
            return None
        return max(occupiers, key=lambda a: a.count)

    def _fortify(self, state: State) -> FortifyAction:
        assert self.env is not None
        pid = state.current_player_index
        topology = self.env.topology
        owned = _owned_indices(state, pid)
        if not owned:
            return FortifyAction(None, None, 0)

        border = [i for i in owned if _is_border(state, topology, i, pid)]
        donors = [
            i
            for i in owned
            if state.armies[i] > 1 and i not in border
        ]
        if not border or not donors:
            return FortifyAction(None, None, 0)

        targets = sorted(
            border,
            key=lambda i: self._territory_score(state, i),
            reverse=True,
        )
        for target in targets:
            donor = self._nearest_donor(state, donors, target)
            if donor is None:
                continue
            count = state.armies[donor] - 1
            if count > 0:
                return FortifyAction(
                    topology.territory_at(donor),
                    topology.territory_at(target),
                    count,
                )
        return FortifyAction(None, None, 0)

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        return _bsr(state, self.env.topology, index) + 0.05 * state.armies[index]

    def _attack_score(self, state: State, action: AttackAction) -> float:
        assert self.env is not None
        topology = self.env.topology
        fi = topology.index_of(action.from_territory)
        ti = topology.index_of(action.to_territory)
        edge = self._attack_edge(state, action)
        surplus = (state.armies[fi] - 1) - state.armies[ti]
        return (
            self.weights.attack_odds * self._battle_odds(state, action)
            + 0.15 * edge
            + self.weights.attacker_surplus * _normalize(surplus, -5, 12)
            + self.weights.continent * _continent_attack_value(state, topology, self.player_id, ti)
            + self.weights.compactness * _compactness_after_take(state, topology, self.player_id, ti)
        )

    def _attack_edge(self, state: State, action: AttackAction) -> float:
        assert self.env is not None
        defender = self.env.topology.index_of(action.to_territory)
        defender_dice = min(MAX_DEFEND_DICE, state.armies[defender])
        return attacker_roll_edge(action.dice, defender_dice)

    def _battle_odds(self, state: State, action: AttackAction) -> float:
        assert self.env is not None
        topology = self.env.topology
        attacker = topology.index_of(action.from_territory)
        defender = topology.index_of(action.to_territory)
        return battle_win_probability(state.armies[attacker], state.armies[defender])

    def _is_full_force_attack(self, state: State, action: AttackAction) -> bool:
        assert self.env is not None
        attacker = self.env.topology.index_of(action.from_territory)
        return action.dice == min(MAX_ATTACK_DICE, state.armies[attacker] - 1)

    def _nearest_donor(
        self,
        state: State,
        donors: Sequence[int],
        target: int,
    ) -> Optional[int]:
        assert self.env is not None
        distances = _owned_distances(state, self.env.topology, target, self.player_id)
        reachable = [d for d in donors if d in distances]
        if not reachable:
            return None
        return max(reachable, key=lambda d: (state.armies[d], -distances[d]))


class BSRAgent(AttackAgent):
    """Prioritize high Border Security Ratio territories."""

    attack_threshold = 0.50
    weights = HeuristicWeights(
        attack_odds=0.8,
        attacker_surplus=0.25,
        continent=0.25,
        bsr=1.2,
        compactness=0.1,
    )

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        topology = self.env.topology
        return (
            self.weights.bsr * _bsr(state, topology, index)
            + 0.25 * _enemy_neighbor_ratio(state, topology, index, self.player_id)
        )


class ContinentAgent(BSRAgent):
    """Prefer finishing and defending continents."""

    weights = HeuristicWeights(
        attack_odds=0.85,
        attacker_surplus=0.25,
        continent=1.25,
        bsr=0.9,
        compactness=0.15,
    )

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        topology = self.env.topology
        continent = topology.continent_of(topology.territory_at(index))
        defend_bonus = 0.0
        if topology.owns_continent(state.owners, continent, self.player_id):
            defend_bonus = topology.continent_bonus(continent) / 7
        return super()._territory_score(state, index) + defend_bonus


class ShapeAgent(BSRAgent):
    """Prefer compact borders and attacks that reduce exposed perimeter."""

    weights = HeuristicWeights(
        attack_odds=0.75,
        attacker_surplus=0.2,
        continent=0.45,
        bsr=0.9,
        compactness=1.1,
    )

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        topology = self.env.topology
        return (
            super()._territory_score(state, index)
            + self.weights.compactness * _enemy_neighbor_ratio(state, topology, index, self.player_id)
        )


class CompositeAgent(ShapeAgent):
    """Weighted blend of attack odds, BSR, continent value, and compactness."""

    attack_threshold = 0.50

    def __init__(
        self,
        player_id: int,
        env: Optional[Environment] = None,
        seed: Optional[int] = None,
        weights: HeuristicWeights | None = None,
    ) -> None:
        super().__init__(player_id=player_id, env=env, seed=seed)
        if weights is not None:
            self.weights = weights

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        topology = self.env.topology
        return (
            self.weights.bsr * _bsr(state, topology, index)
            + self.weights.continent * _continent_defense_value(state, topology, self.player_id, index)
            + self.weights.compactness * _enemy_neighbor_ratio(state, topology, index, self.player_id)
        )


class RaiderAgent(CompositeAgent):
    """Aggressive opponent that accepts marginal fights to expand quickly."""

    attack_threshold = 0.45
    weights = HeuristicWeights(
        attack_odds=1.45,
        attacker_surplus=0.55,
        continent=0.25,
        bsr=0.35,
        compactness=0.05,
    )


class SentinelAgent(CompositeAgent):
    """Defensive opponent that reinforces threatened borders and waits."""

    attack_threshold = 0.62
    weights = HeuristicWeights(
        attack_odds=0.55,
        attacker_surplus=0.2,
        continent=0.45,
        bsr=1.65,
        compactness=0.75,
    )


class EmpireAgent(CompositeAgent):
    """Continent-focused opponent that chases bonuses and defends them."""

    attack_threshold = 0.52
    weights = HeuristicWeights(
        attack_odds=0.8,
        attacker_surplus=0.3,
        continent=1.75,
        bsr=0.8,
        compactness=0.35,
    )


class KillbotAgent(AttackAgent):
    """Python reimplementation of the Lux *Killbot* heuristic.

    Killbot = BetterPixie continent strategy + Vulture weak-player
    elimination + SmartAgentBase attack/card/HogWild utilities. Not the
    original Java agent — see `Docs/HeuristicAgents.md` for the full spec,
    source references, and fidelity notes. Three deliberate deviations from
    Lux (documented there as "Decisions"):
    - Uses exact `battle_win_probability` everywhere instead of Lux's
      `outnumberBy` army-ratio proxy.
    - Kill routes are walked greedily one attack at a time (always retarget
      the best currently-reachable territory owned by `_kill_target`)
      instead of Lux's precomputed `CountryRoute` — so there is no explicit
      route list to keep in sync as territories change hands.
    - The Occupy reserve rule approximates "is the remaining hostile
      territory one connected cluster" with a plain enemy-neighbor count
      instead of a full flood fill.
    """

    attack_threshold = 0.55
    weights = HeuristicWeights(
        attack_odds=1.0,
        attacker_surplus=0.3,
        continent=1.0,
        bsr=0.5,
        compactness=0.2,
    )

    # BetterPixie's setupOurConts constants.
    CONTINENT_ENEMY_MULTIPLIER = 1.3
    CONTINENT_ROUTE_MULTIPLIER = 1.2
    CONTINENT_BUDGET_DIVISOR = 4.0
    BORDER_FORCE_FLOOR = 20
    # Vulture's kill-trigger ratio and card-discount divisor.
    KILL_ARMY_RATIO = 2.0
    CARD_SET_DIVISOR = 3.0

    def __init__(
        self,
        player_id: int,
        env: Optional[Environment] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id=player_id, env=env, seed=seed)
        self._our_conts: set[str] = set()
        self._kill_target: Optional[int] = None
        self._placed_to_kill = False

    def act(self, events: Sequence[object], state: State) -> Optional[Action]:
        # TRADE_IN is entered exactly once per turn (Environment._begin_turn_for),
        # so this is a reliable once-per-turn reset point.
        if state.phase is Phase.TRADE_IN:
            self._our_conts = set()
            self._kill_target = None
            self._placed_to_kill = False
        return super().act(events, state)

    def _choose_trade(self, legal: Sequence[Action]) -> Optional[TradeInAction]:
        trades = [a for a in legal if isinstance(a, TradeInAction)]
        if not trades:
            return None
        assert self.env is not None
        state = self.env.current_state()
        topology = self.env.topology
        pid = self.player_id
        hand = state.hands[pid]

        def bonus_count(action: TradeInAction) -> int:
            return sum(
                1
                for i in action.card_indices
                if not hand[i].is_wild and state.owners[topology.index_of(hand[i].territory_id)] == pid
            )

        return max(trades, key=bonus_count)

    def _reinforce(self, state: State) -> Optional[ReinforcementAction]:
        assert self.env is not None
        self._our_conts = self._select_continents(state)
        if state.reinforcement_budget <= 0:
            self._placed_to_kill = False
            return None

        staging = self._find_kill_staging_territory(state)
        if staging is not None:
            self._placed_to_kill = True
            territory = self.env.topology.territory_at(staging)
            return ReinforcementAction(placements={territory: state.reinforcement_budget})

        self._placed_to_kill = False
        # BetterPixie's placeArmiesToTakeCont + placeNearEnemies collapse into
        # the base top-N stacking here, since `_territory_score` below already
        # boosts territories in `_our_conts` — no need to reimplement the
        # continent/remainder split as separate placement passes.
        return super()._reinforce(state)

    def _find_kill_staging_territory(self, state: State) -> Optional[int]:
        """Vulture's setToKillPlayer + placeToKill, simplified.

        Returns the owned territory to dump this turn's whole budget onto,
        or None if no kill is worth attempting. Requires an existing
        foothold bordering the target (no multi-hop route search — see the
        class docstring's kill-route deviation).
        """
        assert self.env is not None
        pid = self.player_id
        our_total = sum(state.armies[i] for i in _owned_indices(state, pid))

        target: Optional[int] = None
        target_discount: Optional[float] = None
        for opp in range(len(state.hands)):
            if opp == pid or opp in state.eliminated:
                continue
            discount = self._discounted_armies(state, opp)
            if target_discount is None or discount < target_discount:
                target, target_discount = opp, discount
        if target is None or target_discount is None:
            return None
        if our_total <= self.KILL_ARMY_RATIO * target_discount:
            return None

        biggest = self._biggest_army_with_enemy_neighbor(state, pid)
        if biggest is None:
            return None
        target_armies = sum(state.armies[i] for i in _owned_indices(state, target))
        target_territories = len(_owned_indices(state, target))
        if state.armies[biggest] + state.reinforcement_budget <= target_armies + target_territories:
            return None

        entry = self._biggest_army_bordering_player(state, target)
        if entry is None:
            return None
        self._kill_target = target
        return entry

    def _discounted_armies(self, state: State, player_id: int) -> float:
        total = sum(state.armies[i] for i in _owned_indices(state, player_id))
        next_set_value = card_set_value(state.cards_traded_in_count)
        hand_size = len(state.hands[player_id])
        return total - next_set_value * (hand_size / self.CARD_SET_DIVISOR)

    def _biggest_army_with_enemy_neighbor(self, state: State, player_id: int) -> Optional[int]:
        assert self.env is not None
        topology = self.env.topology
        candidates = [i for i in _owned_indices(state, player_id) if _is_border(state, topology, i, player_id)]
        if not candidates:
            return None
        return max(candidates, key=lambda i: state.armies[i])

    def _biggest_army_bordering_player(self, state: State, target_player: int) -> Optional[int]:
        assert self.env is not None
        topology = self.env.topology
        pid = self.player_id
        candidates = [
            i
            for i in _owned_indices(state, pid)
            if any(
                state.owners[topology.index_of(nb)] == target_player
                for nb in topology.neighbors(topology.territory_at(i))
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda i: state.armies[i])

    def _select_continents(self, state: State) -> set[str]:
        """BetterPixie's setupOurConts: continents we're targeting or defending this turn."""
        assert self.env is not None
        topology = self.env.topology
        pid = self.player_id
        owned_conts = {c for c in topology.continents if topology.owns_continent(state.owners, c, pid)}
        costs = self._continent_costs(state)
        if not owned_conts:
            if not costs:
                return set()
            return {min(costs, key=costs.get)}
        budget_per_cont = state.reinforcement_budget / max(
            1.0, len(topology.continents) / self.CONTINENT_BUDGET_DIVISOR
        )
        affordable = {c for c, cost in costs.items() if cost <= budget_per_cont}
        return owned_conts | affordable

    def _continent_costs(self, state: State) -> dict[str, float]:
        """needed = enemy_armies * 1.3 - (our_armies_in_cont + border_armies * 1.2)."""
        assert self.env is not None
        topology = self.env.topology
        pid = self.player_id
        owned_territories = _owned_indices(state, pid)
        costs: dict[str, float] = {}
        for continent in topology.continents:
            owned, total = topology.continent_owner_counts(state.owners, continent, pid)
            if owned == total:
                continue
            members = set(topology.continent_member_indices(continent))
            enemy_armies = sum(state.armies[i] for i in members if state.owners[i] != pid)
            our_armies = sum(state.armies[i] for i in members if state.owners[i] == pid)
            border_armies = sum(
                state.armies[i]
                for i in owned_territories
                if i not in members
                and any(topology.index_of(nb) in members for nb in topology.neighbors(topology.territory_at(i)))
            )
            costs[continent] = enemy_armies * self.CONTINENT_ENEMY_MULTIPLIER - (
                our_armies + border_armies * self.CONTINENT_ROUTE_MULTIPLIER
            )
        return costs

    def _continent_needs_help(self, state: State, continent: str) -> bool:
        assert self.env is not None
        topology = self.env.topology
        pid = self.player_id
        if not topology.owns_continent(state.owners, continent, pid):
            return True
        members = topology.continent_member_indices(continent)
        borders = [i for i in members if _is_border(state, topology, i, pid)]
        if not any(state.armies[i] < self.BORDER_FORCE_FLOOR for i in borders):
            return False
        surrounding = _surrounding_continents(topology, continent)
        if surrounding and all(topology.owns_continent(state.owners, c, pid) for c in surrounding):
            return False
        return True

    def _territory_score(self, state: State, index: int) -> float:
        assert self.env is not None
        topology = self.env.topology
        score = _bsr(state, topology, index) + 0.05 * state.armies[index]
        continent = topology.continent_of(topology.territory_at(index))
        if continent in self._our_conts:
            score += 1.0
            if self._continent_needs_help(state, continent):
                score += 1.0
        return score

    def _attack(self, state: State, legal: Sequence[Action]) -> Action:
        assert self.env is not None
        attacks = [a for a in legal if isinstance(a, AttackAction)]
        if not attacks:
            return StopAttackAction()

        if self._placed_to_kill and self._kill_target is not None:
            kill_action = self._best_kill_route_attack(state, attacks)
            if kill_action is not None:
                return kill_action
            # Target eliminated or no longer reachable this turn.
            self._placed_to_kill = False

        threshold = 0.0 if self._is_hogwild(state) else self.attack_threshold

        steal = self._continent_steal_attack(state, attacks)
        if steal is not None:
            return steal

        chosen = self._best_scored_attack(state, attacks, threshold, require_our_continent=True)
        if chosen is None:
            chosen = self._best_scored_attack(state, attacks, threshold, require_our_continent=False)
        if chosen is not None:
            return chosen

        if not state.conquered_this_turn:
            card_attack = self._attack_for_card(state, attacks)
            if card_attack is not None:
                return card_attack

        return StopAttackAction()

    def _best_kill_route_attack(self, state: State, attacks: Sequence[AttackAction]) -> Optional[AttackAction]:
        assert self.env is not None
        topology = self.env.topology
        options = [
            a
            for a in attacks
            if self._is_full_force_attack(state, a)
            and state.owners[topology.index_of(a.to_territory)] == self._kill_target
            and self._battle_odds(state, a) > 0.0
        ]
        if not options:
            return None
        best = max(self._battle_odds(state, a) for a in options)
        return self._rng.choice([a for a in options if self._battle_odds(state, a) == best])

    def _best_scored_attack(
        self,
        state: State,
        attacks: Sequence[AttackAction],
        threshold: float,
        *,
        require_our_continent: bool,
    ) -> Optional[AttackAction]:
        assert self.env is not None
        topology = self.env.topology
        candidates = [
            (self._attack_score(state, a), a)
            for a in attacks
            if self._is_full_force_attack(state, a)
            and self._battle_odds(state, a) > threshold
            and (not require_our_continent or topology.continent_of(a.to_territory) in self._our_conts)
        ]
        if not candidates:
            return None
        best_score = max(score for score, _ in candidates)
        return self._rng.choice([a for score, a in candidates if score == best_score])

    def _continent_steal_attack(self, state: State, attacks: Sequence[AttackAction]) -> Optional[AttackAction]:
        """takeOutContinentCheck: grab a poorly-defended enemy continent, independent of `_our_conts`."""
        assert self.env is not None
        topology = self.env.topology
        pid = self.player_id
        owned_territories = _owned_indices(state, pid)
        for continent in topology.continents:
            if topology.owns_continent(state.owners, continent, pid):
                continue
            members = set(topology.continent_member_indices(continent))
            enemy_defense = sum(
                state.armies[i] for i in members if state.owners[i] is not None and state.owners[i] != pid
            )
            if enemy_defense <= 0:
                continue
            our_nearby = sum(
                state.armies[i]
                for i in owned_territories
                if any(topology.index_of(nb) in members for nb in topology.neighbors(topology.territory_at(i)))
            )
            if our_nearby < self.KILL_ARMY_RATIO * enemy_defense:
                continue
            options = [
                a
                for a in attacks
                if self._is_full_force_attack(state, a)
                and topology.continent_of(a.to_territory) == continent
                and self._battle_odds(state, a) > 0.5
            ]
            if options:
                return max(options, key=lambda a: self._battle_odds(state, a))
        return None

    def _attack_for_card(self, state: State, attacks: Sequence[AttackAction]) -> Optional[AttackAction]:
        options = [a for a in attacks if self._is_full_force_attack(state, a) and self._battle_odds(state, a) > 0.5]
        if not options:
            return None
        return max(options, key=lambda a: self._battle_odds(state, a))

    def _is_hogwild(self, state: State) -> bool:
        pid = self.player_id
        our_armies = sum(state.armies[i] for i in _owned_indices(state, pid))
        enemy_armies = sum(
            armies for i, armies in enumerate(state.armies) if state.owners[i] is not None and state.owners[i] != pid
        )
        return our_armies > enemy_armies

    def _occupy(self, legal: Sequence[Action]) -> Optional[OccupyAction]:
        occupiers = [a for a in legal if isinstance(a, OccupyAction)]
        if not occupiers:
            return None
        assert self.env is not None
        state = self.env.current_state()
        pending = state.pending_attack
        counts = [a.count for a in occupiers]
        lo, hi = min(counts), max(counts)
        if pending is not None and _enemy_neighbor_count(state, self.env.topology, pending.to_index, self.player_id) > 1:
            desired = max(lo, hi // 2)
        else:
            desired = hi
        return min(occupiers, key=lambda a: abs(a.count - desired))


def _owned_indices(state: State, player_id: int) -> list[int]:
    return [i for i, owner in enumerate(state.owners) if owner == player_id]


def _is_border(state: State, topology, index: int, player_id: int) -> bool:
    territory = topology.territory_at(index)
    return any(
        state.owners[topology.index_of(nb)] != player_id
        for nb in topology.neighbors(territory)
    )


def _bsr(state: State, topology, index: int) -> float:
    owner = state.owners[index]
    if owner is None:
        return 0.0
    territory = topology.territory_at(index)
    enemy_armies = sum(
        state.armies[topology.index_of(nb)]
        for nb in topology.neighbors(territory)
        if state.owners[topology.index_of(nb)] != owner
    )
    return enemy_armies / max(1, state.armies[index])


def _enemy_neighbor_ratio(state: State, topology, index: int, player_id: int) -> float:
    territory = topology.territory_at(index)
    neighbors = topology.neighbors(territory)
    if not neighbors:
        return 0.0
    enemies = sum(
        1 for nb in neighbors if state.owners[topology.index_of(nb)] != player_id
    )
    return enemies / len(neighbors)


def _enemy_neighbor_count(state: State, topology, index: int, player_id: int) -> int:
    """Raw enemy-neighbor count — `_enemy_neighbor_ratio` discards this by dividing."""
    territory = topology.territory_at(index)
    return sum(
        1 for nb in topology.neighbors(territory) if state.owners[topology.index_of(nb)] != player_id
    )


def _continent_attack_value(state: State, topology, player_id: int, target_index: int) -> float:
    continent = topology.continent_of(topology.territory_at(target_index))
    owned, total = topology.continent_owner_counts(state.owners, continent, player_id)
    enemies = total - owned
    if enemies <= 0:
        return 0.0
    completion = owned / total
    takeover_bonus = 1.0 if enemies == 1 else 0.0
    return 0.7 * completion + 0.3 * takeover_bonus


def _continent_defense_value(state: State, topology, player_id: int, index: int) -> float:
    continent = topology.continent_of(topology.territory_at(index))
    owned, total = topology.continent_owner_counts(state.owners, continent, player_id)
    if owned == total:
        return topology.continent_bonus(continent) / 7
    return owned / total


def _compactness_after_take(state: State, topology, player_id: int, target_index: int) -> float:
    territory = topology.territory_at(target_index)
    friendly_neighbors = sum(
        1 for nb in topology.neighbors(territory)
        if state.owners[topology.index_of(nb)] == player_id
    )
    enemy_neighbors = sum(
        1 for nb in topology.neighbors(territory)
        if state.owners[topology.index_of(nb)] != player_id
    )
    total = friendly_neighbors + enemy_neighbors
    if total == 0:
        return 0.0
    return friendly_neighbors / total


def _surrounding_continents(topology, continent: str) -> set[str]:
    """Continent ids adjacent to `continent`'s border (used by `continentNeedsHelp`)."""
    members = set(topology.continent_member_indices(continent))
    surrounding: set[str] = set()
    for i in members:
        for nb in topology.neighbors(topology.territory_at(i)):
            ni = topology.index_of(nb)
            if ni not in members:
                surrounding.add(topology.continent_of(nb))
    return surrounding


def _owned_distances(state: State, topology, start: int, player_id: int) -> dict[int, int]:
    distances = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in topology.neighbors(topology.territory_at(cur)):
            ni = topology.index_of(nb)
            if ni in distances or state.owners[ni] != player_id:
                continue
            distances[ni] = distances[cur] + 1
            queue.append(ni)
    return distances


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


__all__ = [
    "ATTACKER_ROLL_EDGE",
    "AttackAgent",
    "BSRAgent",
    "CompositeAgent",
    "ContinentAgent",
    "EmpireAgent",
    "HeuristicWeights",
    "KillbotAgent",
    "ROLL_OUTCOMES",
    "RaiderAgent",
    "SentinelAgent",
    "ShapeAgent",
    "attacker_roll_edge",
    "battle_win_probability",
]
