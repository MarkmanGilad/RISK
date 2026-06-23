"""Reward calculation for RL training, kept fully separate from `Environment`.

`Environment.step(...)` (`risk/game/environment.py`) calls `RewardCalculator.compute(...)`
and uses the returned float as-is — it performs no reward arithmetic itself.
`Trainer` (`risk/learning/trainer.py`) additionally calls `end_of_turn(...)`
directly when the learner's own action ends its turn, since that term needs
the state after the *entire opponent round*, which no single `step()` call
ever sees. See `Docs/Reward.md` for the full design and phased rollout plan.

Every phase helper below only ever scores the *current actor's own* decision
— gated on `before.current_player_index == reward_player` — so an opponent's
action taken with `reward_player=<learner>` (as `Environment.step` is called
for every actor in `Trainer`'s loop) never gets scored as if the learner had
made it. The terminal term is the one exception: win/loss is about
`reward_player`'s outcome regardless of whose action triggered it.
"""
from __future__ import annotations

from typing import Optional

from risk.constants import MAX_ATTACK_DICE, card_set_value
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
from risk.game.board_topology import BoardTopology
from risk.game.state import State
from risk.learning.train_constants import (
    REWARD_ARMY_DELTA_RELATIVE_SCALE,
    REWARD_ATTACK_ARMY_TRADE,
    REWARD_ATTACK_CARD_TERRITORY_MATCH,
    REWARD_ATTACK_CONTINENT_CAPTURED,
    REWARD_ATTACK_CONTINENT_DOMINATION,
    REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN,
    REWARD_ATTACK_CONQUER_TERRITORY,
    REWARD_ATTACK_CONQUER_WITH_CARD,
    REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD,
    REWARD_ATTACK_FEWER_DICE,
    REWARD_ATTACK_RATIO_CAP,
    REWARD_ATTACK_RATIO_SCALE,
    REWARD_ATTACK_RATIO_THRESHOLD,
    REWARD_ATTACK_STOP_WITHOUT_CARD,
    REWARD_CONTINENT_DELTA_RELATIVE,
    REWARD_FORTIFY_BALANCE_SCALE,
    REWARD_FORTIFY_CONTINENT_PUSH,
    REWARD_FORTIFY_TOWARD_FRONTIER,
    REWARD_OCCUPY_FORWARD_MOMENTUM,
    REWARD_REINFORCE_ATTACK_READINESS_SCALE,
    REWARD_REINFORCE_CONCENTRATION,
    REWARD_REINFORCE_CONTINENT_PUSH,
    REWARD_REINFORCE_NO_ENEMY_NEIGHBOR,
    REWARD_REINFORCE_RATIO_CAP,
    REWARD_SHAPING_STEP_CAP,
    REWARD_TERMINAL_LOSS,
    REWARD_TERMINAL_WIN,
    REWARD_TERRITORY_DELTA,
    REWARD_TRADE_IN_EARLY,
    REWARD_TRADE_IN_TERRITORY_MATCH,
)

_EPS = 1e-6


class RewardCalculator:
    """Turns engine-produced `info`/before/after `State` into a reward float."""

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
        terminal = self._terminal(reward_player, done, winner)

        shaping_raw = (
            self._trade_in(action, before, reward_player)
            + self._reinforce(action, before, after, reward_player)
            + self._attack(action, info, before, after, reward_player)
            + self._occupy(action, before, reward_player)
            + self._fortify(action, before, after, reward_player)
        )
        shaping = max(-REWARD_SHAPING_STEP_CAP, min(REWARD_SHAPING_STEP_CAP, shaping_raw))
        return terminal + shaping

    def end_of_turn(self, before_turn: State, after_turn: State, reward_player: int) -> float:
        """Opponent-impact terms — see `Docs/Reward.md`'s "End-of-turn" section.

        Called directly by `Trainer` when the learner's own action ends its
        turn (any `FortifyAction`), with `before_turn` = the state right
        when that action was taken and `after_turn` = the state once every
        opponent has played their full turn. No single `Environment.step()`
        call ever sees both ends of that span, so this can't be folded into
        `compute()`.
        """
        pid = reward_player

        my_terr_before = sum(1 for o in before_turn.owners if o == pid)
        my_terr_after = sum(1 for o in after_turn.owners if o == pid)
        share_before = my_terr_before / len(before_turn.owners)
        share_after = my_terr_after / len(after_turn.owners)
        reward = REWARD_TERRITORY_DELTA * (share_after - share_before)

        my_army_before = sum(a for o, a in zip(before_turn.owners, before_turn.armies) if o == pid)
        my_army_after = sum(a for o, a in zip(after_turn.owners, after_turn.armies) if o == pid)
        total_army_before = sum(before_turn.armies)
        total_army_after = sum(after_turn.armies)
        army_share_before = my_army_before / total_army_before if total_army_before else 0.0
        army_share_after = my_army_after / total_army_after if total_army_after else 0.0
        reward += REWARD_ARMY_DELTA_RELATIVE_SCALE * (army_share_after - army_share_before)

        total_bonus = sum(self.topology.continent_bonus(c) for c in self.topology.continents)
        if total_bonus:
            my_bonus_before = sum(
                self.topology.continent_bonus(c)
                for c in self.topology.continents
                if self.topology.owns_continent(before_turn.owners, c, pid)
            )
            my_bonus_after = sum(
                self.topology.continent_bonus(c)
                for c in self.topology.continents
                if self.topology.owns_continent(after_turn.owners, c, pid)
            )
            reward += REWARD_CONTINENT_DELTA_RELATIVE * (
                my_bonus_after / total_bonus - my_bonus_before / total_bonus
            )

        return reward

    # --- terminal -------------------------------------------------------

    def _terminal(self, reward_player: int, done: bool, winner: Optional[int]) -> float:
        if not done:
            return 0.0
        if winner == reward_player:
            return REWARD_TERMINAL_WIN
        return REWARD_TERMINAL_LOSS

    # --- shared primitives ------------------------------------------------

    def _is_frontier(self, state: State, idx: int) -> bool:
        pid = state.owners[idx]
        t = self.topology.territory_at(idx)
        return any(
            state.owners[self.topology.index_of(nb)] != pid for nb in self.topology.neighbors(t)
        )

    def _frontier_threat(self, state: State, idx: int) -> int:
        pid = state.owners[idx]
        t = self.topology.territory_at(idx)
        return sum(
            state.armies[self.topology.index_of(nb)]
            for nb in self.topology.neighbors(t)
            if state.owners[self.topology.index_of(nb)] != pid
        )

    def _weakest_adjacent_enemy_armies(self, state: State, idx: int) -> Optional[int]:
        pid = state.owners[idx]
        t = self.topology.territory_at(idx)
        enemy_armies = [
            state.armies[self.topology.index_of(nb)]
            for nb in self.topology.neighbors(t)
            if state.owners[self.topology.index_of(nb)] != pid
        ]
        return min(enemy_armies) if enemy_armies else None

    def _continent_push(self, state: State, territory: str, pid: int, scale: float) -> float:
        continent = self.topology.continent_of(territory)
        owned, total = self.topology.continent_owner_counts(state.owners, continent, pid)
        return scale * (owned / total) / total

    # --- TRADE_IN -----------------------------------------------------------

    @staticmethod
    def _trade_in_hand_factor(hand_size: int) -> float:
        if hand_size == 3:
            return 1.0
        if hand_size == 4:
            return 0.5
        return 0.0

    def _set_matches_owned_territory(self, state: State, pid: int, indices: tuple[int, ...]) -> bool:
        hand = state.hands[pid]
        for i in indices:
            card = hand[i]
            if card.is_wild:
                continue
            if state.owners[self.topology.index_of(card.territory_id)] == pid:
                return True
        return False

    def _trade_in(self, action: Action, before: State, reward_player: int) -> float:
        pid = before.current_player_index
        if pid != reward_player:
            return 0.0

        if isinstance(action, SkipTradeAction):
            f = self._trade_in_hand_factor(len(before.hands[pid]))
            if f == 0.0:
                return 0.0
            v = card_set_value(before.cards_traded_in_count)
            return REWARD_TRADE_IN_EARLY * f / v

        if isinstance(action, TradeInAction):
            reward = 0.0
            f = self._trade_in_hand_factor(len(before.hands[pid]))
            if f > 0.0:
                v = card_set_value(before.cards_traded_in_count)
                reward -= REWARD_TRADE_IN_EARLY * f / v
            if self._set_matches_owned_territory(before, pid, action.card_indices):
                reward += REWARD_TRADE_IN_TERRITORY_MATCH
            return reward

        return 0.0

    # --- REINFORCE_PLACE ---------------------------------------------------

    def _reinforce(self, action: Action, before: State, after: State, reward_player: int) -> float:
        pid = before.current_player_index
        if pid != reward_player or not isinstance(action, ReinforcementAction):
            return 0.0

        remaining_before = before.reinforcement_budget
        reward = 0.0
        for terr, count in action.placements.items():
            idx = self.topology.index_of(terr)

            if self._is_frontier(before, idx) and remaining_before > 0:
                reward += REWARD_REINFORCE_CONCENTRATION * (count / remaining_before)

            weakest = self._weakest_adjacent_enemy_armies(before, idx)
            if weakest is not None:
                ratio = after.armies[idx] / weakest
                reward += REWARD_REINFORCE_ATTACK_READINESS_SCALE * (
                    min(ratio, REWARD_REINFORCE_RATIO_CAP) - 1.0
                )
            else:
                reward += REWARD_REINFORCE_NO_ENEMY_NEIGHBOR

            reward += self._continent_push(after, terr, pid, REWARD_REINFORCE_CONTINENT_PUSH)

        return reward

    # --- ATTACK -------------------------------------------------------------

    def _attack(
        self, action: Action, info: dict, before: State, after: State, reward_player: int
    ) -> float:
        pid = before.current_player_index
        if pid != reward_player:
            return 0.0

        if isinstance(action, StopAttackAction):
            has_real_attack = any(
                before.owners[i] == pid and before.armies[i] >= 2 and self._is_frontier(before, i)
                for i in range(len(before.owners))
            )
            if has_real_attack and not before.conquered_this_turn:
                return REWARD_ATTACK_STOP_WITHOUT_CARD
            return 0.0

        if not isinstance(action, AttackAction):
            return 0.0

        fi = self.topology.index_of(action.from_territory)
        ti = self.topology.index_of(action.to_territory)
        reward = 0.0

        max_dice = min(MAX_ATTACK_DICE, before.armies[fi] - 1)
        if action.dice < max_dice:
            reward += REWARD_ATTACK_FEWER_DICE

        ratio = before.armies[fi] / before.armies[ti]
        reward += REWARD_ATTACK_RATIO_SCALE * (
            min(ratio, REWARD_ATTACK_RATIO_CAP) - REWARD_ATTACK_RATIO_THRESHOLD
        )

        continent = self.topology.continent_of(action.to_territory)
        owned, total = self.topology.continent_owner_counts(before.owners, continent, pid)
        alive_players = len(before.hands) - len(before.eliminated)
        if alive_players > 0 and (owned / total) >= (1 / alive_players + REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN):
            margin_left = total - owned
            reward += REWARD_ATTACK_CONTINENT_DOMINATION * (1.0 / (margin_left + 1))

        reward += REWARD_ATTACK_ARMY_TRADE * (
            info.get("defender_losses", 0) - info.get("attacker_losses", 0)
        )

        eliminated = info.get("eliminated")
        if eliminated is not None:
            cards_taken = len(before.hands[eliminated])
            reward += REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD * (cards_taken + 1)

        if info.get("conquered"):
            reward += REWARD_ATTACK_CONQUER_TERRITORY

            was_owned_before = self.topology.owns_continent(before.owners, continent, pid)
            is_owned_after = self.topology.owns_continent(after.owners, continent, pid)
            if is_owned_after and not was_owned_before:
                reward += REWARD_ATTACK_CONTINENT_CAPTURED

            drew_card = not before.conquered_this_turn
            if drew_card and len(after.hands[pid]) > len(before.hands[pid]):
                drawn_card = after.hands[pid][len(before.hands[pid])]
                reward += REWARD_ATTACK_CONQUER_WITH_CARD
                if not drawn_card.is_wild and after.owners[self.topology.index_of(drawn_card.territory_id)] == pid:
                    reward += REWARD_ATTACK_CARD_TERRITORY_MATCH

        return reward

    # --- OCCUPY -------------------------------------------------------------

    def _occupy(self, action: Action, before: State, reward_player: int) -> float:
        pid = before.current_player_index
        if pid != reward_player or not isinstance(action, OccupyAction) or before.pending_attack is None:
            return 0.0
        available = before.armies[before.pending_attack.from_index] - 1
        if available <= 0:
            return 0.0
        return REWARD_OCCUPY_FORWARD_MOMENTUM * (action.count / available)

    # --- FORTIFY -------------------------------------------------------------

    def _fortify(self, action: Action, before: State, after: State, reward_player: int) -> float:
        pid = before.current_player_index
        if pid != reward_player or not isinstance(action, FortifyAction) or action.is_skip:
            return 0.0

        fi = self.topology.index_of(action.from_territory)
        ti = self.topology.index_of(action.to_territory)
        src_frontier = self._is_frontier(before, fi)
        dst_frontier = self._is_frontier(before, ti)
        movable = before.armies[fi] - 1

        reward = 0.0
        if src_frontier and dst_frontier:
            threat_src = self._frontier_threat(before, fi)
            threat_dst = self._frontier_threat(before, ti)
            total_after = after.armies[fi] + after.armies[ti]
            target_dst = (threat_dst / (threat_src + threat_dst + _EPS)) * total_after
            reward -= REWARD_FORTIFY_BALANCE_SCALE * abs(after.armies[ti] - target_dst) / total_after
        elif dst_frontier and movable > 0:
            reward += REWARD_FORTIFY_TOWARD_FRONTIER * (action.count / movable)
        elif src_frontier and movable > 0:
            reward -= REWARD_FORTIFY_TOWARD_FRONTIER * (action.count / movable)

        reward += self._continent_push(after, action.to_territory, pid, REWARD_FORTIFY_CONTINENT_PUSH)

        return reward
