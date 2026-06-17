"""Heuristic Risk agents for self-play and RL opponent curricula.

The agents in this module use the same callable contract as RandomAgent, but
rank legal moves with classic Risk heuristics: attack odds, border security
ratio, continent completion pressure, and compactness.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

from risk.agents.base_agent import BaseAgent
from risk.constants import ATTACKER_ROLL_EDGE, MAX_ATTACK_DICE, MAX_DEFEND_DICE, ROLL_OUTCOMES
from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
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


@lru_cache(maxsize=None)
def battle_win_probability(attacker_armies: int, defender_armies: int) -> float:
    """Probability of conquering a territory while always rolling full force."""

    if defender_armies <= 0:
        return 1.0
    if attacker_armies <= 1:
        return 0.0

    attacker_dice = min(MAX_ATTACK_DICE, attacker_armies - 1)
    defender_dice = min(MAX_DEFEND_DICE, defender_armies)
    return sum(
        probability
        * battle_win_probability(
            attacker_armies - attacker_losses,
            defender_armies - defender_losses,
        )
        for attacker_losses, defender_losses, probability in ROLL_OUTCOMES[
            (attacker_dice, defender_dice)
        ]
    )


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

        trade = self._choose_trade(legal)
        if trade is not None:
            return trade

        if state.phase is Phase.REINFORCE:
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
    "ROLL_OUTCOMES",
    "RaiderAgent",
    "SentinelAgent",
    "ShapeAgent",
    "attacker_roll_edge",
    "battle_win_probability",
]
