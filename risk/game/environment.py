"""The Risk rule engine.

`Environment` owns *all* legality and resolution. Agents and the game
loop only call: `reset`, `current_state`, `legal_actions`, `step`,
`is_terminal`, `winner`.

Deterministic given a seed in `GameSettings` (territory deal, starting-
army placement, dice, card draw).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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
from risk.constants import (
    CARD_SYMBOLS,
    CARD_TERRITORY_BONUS_ARMIES,
    DIE_SIDES,
    MAX_ATTACK_DICE,
    MAX_CARDS_IN_HAND,
    MAX_DEFEND_DICE,
    MIN_REINFORCEMENT,
    TERRITORIES_PER_REINFORCEMENT,
    WILD_SYMBOL,
    card_set_value,
    starting_armies_for,
)
from risk.game.card import Card, CardRules
from risk.game.phase import Phase
from risk.game.player import Player
from risk.game.settings import GameSettings
from risk.game.state import PendingAttack, State
from risk.learning.reward import RewardCalculator


# --- result type ----------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single `step` call."""

    state: State
    info: dict
    reward: float = 0.0
    done: bool = False


# --- environment ----------------------------------------------------------


class Environment:
    def __init__(self, topology: Optional[BoardTopology] = None) -> None:
        self.topology: BoardTopology = topology or BoardTopology()
        self._settings: Optional[GameSettings] = None
        self._state: Optional[State] = None
        self._rng: random.Random = random.Random()
        # Card deck: one card per territory + 2 wilds.
        self._deck: list[Card] = []
        self._discard: list[Card] = []
        self._reward = RewardCalculator(self.topology)

    # --- lifecycle --------------------------------------------------------

    def reset(self, settings: GameSettings) -> State:
        self._settings = settings
        self._rng = random.Random(settings.seed)
        n_players = settings.player_count

        # Build deck deterministically.
        self._deck = self._build_deck()
        self._discard = []

        state = State.initial(self.topology, n_players)

        # 1) Random territory distribution.
        territories = list(self.topology.territories)
        self._rng.shuffle(territories)
        for k, t in enumerate(territories):
            idx = self.topology.index_of(t)
            state.owners[idx] = k % n_players
            state.armies[idx] = 1

        # 2) Distribute remaining starting armies round-robin onto owned
        #    territories (random pick among each player's own).
        starting = starting_armies_for(n_players)
        owned_by: list[list[int]] = [[] for _ in range(n_players)]
        for i, o in enumerate(state.owners):
            assert o is not None
            owned_by[o].append(i)

        for p in range(n_players):
            remaining = starting - len(owned_by[p])
            assert remaining >= 0, "starting armies must cover every owned territory"
            for _ in range(remaining):
                idx = self._rng.choice(owned_by[p])
                state.armies[idx] += 1

        # 3) Begin player 0's turn (TRADE_IN phase first).
        state.current_player_index = 0
        self._state = state
        self._begin_turn_for(state, 0)
        return state

    # --- read-only views --------------------------------------------------

    @property
    def settings(self) -> GameSettings:
        assert self._settings is not None, "Environment.reset has not been called"
        return self._settings

    @property
    def reward(self) -> RewardCalculator:
        """The `RewardCalculator` this environment's `step(...)` calls.

        Exposed so callers needing terms `step(...)` itself can't see (e.g.
        `Trainer`'s end-of-turn opponent-impact reward, which spans multiple
        `step()` calls) can call it directly without recomputing the same
        math.
        """
        return self._reward

    def current_state(self) -> State:
        assert self._state is not None, "Environment.reset has not been called"
        return self._state

    def is_terminal(self) -> bool:
        return self.current_state().phase is Phase.GAME_OVER

    def winner(self) -> Optional[int]:
        s = self.current_state()
        alive = [p.id for p in self.settings.players if p.id not in s.eliminated]
        return alive[0] if len(alive) == 1 else None

    # --- step -------------------------------------------------------------

    def step(self, action: Action, reward_player: Optional[int] = None) -> StepResult:
        if self.is_terminal():
            raise RuntimeError("Cannot step a terminal Environment")
        s = self.current_state()
        action.validate_against(self.topology) if hasattr(action, "validate_against") else None
        before = s.snapshot() if reward_player is not None else None

        if isinstance(action, ReinforcementAction):
            info = self._apply_reinforce(s, action)
        elif isinstance(action, TradeInAction):
            info = self._apply_trade_in(s, action)
        elif isinstance(action, SkipTradeAction):
            info = self._apply_skip_trade(s)
        elif isinstance(action, AttackAction):
            info = self._apply_attack(s, action)
        elif isinstance(action, OccupyAction):
            info = self._apply_occupy(s, action)
        elif isinstance(action, StopAttackAction):
            info = self._apply_stop_attack(s)
        elif isinstance(action, FortifyAction):
            info = self._apply_fortify(s, action)
        else:
            raise ValueError(f"Unsupported action type: {type(action).__name__}")

        # Check for winner.
        w = self.winner()
        if w is not None and s.phase is not Phase.GAME_OVER:
            s.phase = Phase.GAME_OVER

        reward = 0.0
        done = False
        if reward_player is not None:
            done = reward_player in s.eliminated or s.phase is Phase.GAME_OVER
            reward = self._reward.compute(
                action=action,
                info=info,
                before=before,
                after=s,
                reward_player=reward_player,
                done=done,
                winner=self.winner(),
            )

        return StepResult(state=s, info=info, reward=reward, done=done)

    # --- legal actions ----------------------------------------------------

    def legal_actions(self, state: Optional[State] = None) -> list[Action]:
        """Return a representative list of legal actions for the current phase.

        `state` defaults to `self.current_state()`; pass an explicit snapshot
        to enumerate legal actions for a state other than the live one (e.g.
        a replay-buffer transition's `next_state` during training) — every
        `_legal_*` helper below is already a pure function of `(state,
        self.topology)`, so this never reads or mutates `self._state`.

        Note: ReinforcementAction is parameterized; enumerating *all* possible
        placements is exponential. For attacks/fortify we enumerate exactly.
        For reinforcement we yield a bucketed amount (one / half / all of the
        remaining budget) per owned territory — reinforcement is multi-step,
        so any split is still reachable over a few actions. Agents that want
        a specific split should construct their own ReinforcementAction and
        let `step` validate it.
        """
        s = state if state is not None else self.current_state()
        if s.phase is Phase.TRADE_IN:
            trades = list(self._legal_trade_ins(s))
            # If the player must trade (hand >= limit), only offer trade actions.
            if len(s.hands[s.current_player_index]) >= MAX_CARDS_IN_HAND:
                return trades
            return trades + [SkipTradeAction()]
        if s.phase is Phase.REINFORCE_PLACE:
            return list(self._legal_reinforce(s))
        if s.phase is Phase.ATTACK:
            return list(self._legal_attack(s))
        if s.phase is Phase.OCCUPY:
            return list(self._legal_occupy(s))
        if s.phase is Phase.FORTIFY:
            return list(self._legal_fortify(s))
        return []

    # ===== internals =====================================================

    # --- deck ------------------------------------------------------------

    def _build_deck(self) -> list[Card]:
        deck: list[Card] = []
        for i, t in enumerate(self.topology.territories):
            sym = CARD_SYMBOLS[i % len(CARD_SYMBOLS)]
            deck.append(Card(territory_id=t, symbol=sym))
        deck.append(Card(territory_id=None, symbol=WILD_SYMBOL))
        deck.append(Card(territory_id=None, symbol=WILD_SYMBOL))
        self._rng.shuffle(deck)
        return deck

    def _draw_card(self) -> Optional[Card]:
        if not self._deck:
            if not self._discard:
                return None
            self._deck = self._discard
            self._discard = []
            self._rng.shuffle(self._deck)
        return self._deck.pop()

    # --- reinforcement ---------------------------------------------------

    def _compute_reinforcement(self, s: State, player_id: int) -> int:
        owned = [i for i, o in enumerate(s.owners) if o == player_id]
        base = max(MIN_REINFORCEMENT, len(owned) // TERRITORIES_PER_REINFORCEMENT)
        bonus = 0
        for continent in self.topology.continents:
            if self.topology.owns_continent(s.owners, continent, player_id):
                bonus += self.topology.continent_bonus(continent)
        return base + bonus

    def _apply_reinforce(self, s: State, action: ReinforcementAction) -> dict:
        if s.phase is not Phase.REINFORCE_PLACE:
            raise ValueError("Reinforcement only allowed during REINFORCE_PLACE")
        pid = s.current_player_index
        # Validate territory ownership and shape.
        for terr, count in action.placements.items():
            if terr not in self.topology.territories:
                raise ValueError(f"Unknown territory in placements: {terr}")
            idx = self.topology.index_of(terr)
            if s.owners[idx] != pid:
                raise ValueError(f"Player {pid} does not own {terr}")
        if action.total > s.reinforcement_budget:
            raise ValueError(
                f"Reinforcement total {action.total} exceeds budget {s.reinforcement_budget}"
            )
        # Apply. Reinforcement is multi-step: a player may place part of the
        # budget and stay in REINFORCE_PLACE to place the rest in a later
        # action; the phase only advances once the whole budget is placed.
        for terr, count in action.placements.items():
            idx = self.topology.index_of(terr)
            s.armies[idx] += count
        s.reinforcement_budget -= action.total
        if s.reinforcement_budget == 0:
            s.phase = Phase.ATTACK
            s.conquered_this_turn = False
        return {"placed": dict(action.placements), "remaining_budget": s.reinforcement_budget}

    def _trade_in_set(self, s: State, pid: int, trio: tuple[Card, Card, Card]) -> int:
        for c in trio:
            s.hands[pid].remove(c)
            self._discard.append(c)
        value = card_set_value(s.cards_traded_in_count)
        s.cards_traded_in_count += 1
        # Territory bonus: a card depicting a territory the trading player
        # currently occupies grants 2 armies placed on it immediately, on
        # top of the set's reinforcement value.
        for c in trio:
            if c.is_wild:
                continue
            idx = self.topology.index_of(c.territory_id)
            if s.owners[idx] == pid:
                s.armies[idx] += CARD_TERRITORY_BONUS_ARMIES
        return value

    def _apply_trade_in(self, s: State, action: TradeInAction) -> dict:
        if s.phase is not Phase.TRADE_IN:
            raise ValueError("Trade-in only allowed during TRADE_IN")
        pid = s.current_player_index
        hand = s.hands[pid]
        idxs = action.card_indices
        if any(i >= len(hand) for i in idxs):
            raise ValueError("Card index out of range")
        trio = tuple(hand[i] for i in idxs)
        if not CardRules.is_valid_set(trio):
            raise ValueError("Selected cards are not a valid set")
        value = self._trade_in_set(s, pid, trio)  # type: ignore[arg-type]
        s.reinforcement_budget += value
        auto_skipped = not any(self._legal_trade_ins(s))
        if auto_skipped:
            s.phase = Phase.REINFORCE_PLACE
        return {
            "traded": value,
            "cards": len(hand) - 3,
            "auto_skipped_trade": auto_skipped,
        }

    def _apply_skip_trade(self, s: State) -> dict:
        s.phase = Phase.REINFORCE_PLACE
        return {}

    def _legal_trade_ins(self, s: State) -> Iterable[Action]:
        pid = s.current_player_index
        hand = s.hands[pid]
        n = len(hand)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if CardRules.is_valid_set((hand[i], hand[j], hand[k])):
                        yield TradeInAction(card_indices=(i, j, k))

    def trade_in_cards(self, cards: Sequence[Card]) -> int:
        """Optional explicit trade-in (callable during reinforcement)."""
        s = self.current_state()
        if s.phase is not Phase.TRADE_IN:
            raise ValueError("Trade-in only allowed during TRADE_IN")
        if not CardRules.is_valid_set(cards):
            raise ValueError("Not a valid trade-in set")
        pid = s.current_player_index
        hand_counter = Counter(id(c) for c in s.hands[pid])
        for c in cards:
            if id(c) not in hand_counter or hand_counter[id(c)] <= 0:
                raise ValueError("Card not in player's hand")
            hand_counter[id(c)] -= 1
        value = self._trade_in_set(s, pid, tuple(cards))  # type: ignore[arg-type]
        s.reinforcement_budget += value
        return value

    def _legal_reinforce(self, s: State) -> Iterable[Action]:
        pid = s.current_player_index
        owned = [self.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
        budget = s.reinforcement_budget
        if not owned or budget <= 0:
            return
        # Bucketed amounts per territory — one, half (//2), and all of the
        # remaining budget — instead of enumerating every integer amount
        # (which would scale with army count, not board size). Reinforcement
        # is multi-step, so any split is still reachable over a few actions;
        # agents may also construct their own placements and submit those.
        amounts = sorted({amt for amt in (1, budget // 2, budget) if amt > 0})
        for t in owned:
            for amt in amounts:
                yield ReinforcementAction(placements={t: amt})

    # --- attack ----------------------------------------------------------

    def _apply_attack(self, s: State, action: AttackAction) -> dict:
        pid = s.current_player_index
        fi = self.topology.index_of(action.from_territory)
        ti = self.topology.index_of(action.to_territory)
        if s.owners[fi] != pid:
            raise ValueError(f"Player {pid} does not own attacker {action.from_territory}")
        if s.owners[ti] == pid:
            raise ValueError(f"Cannot attack own territory {action.to_territory}")
        if not self.topology.are_adjacent(action.from_territory, action.to_territory):
            raise ValueError(f"{action.from_territory} and {action.to_territory} not adjacent")
        if s.armies[fi] < 2:
            raise ValueError(f"{action.from_territory} needs >= 2 armies to attack")
        max_atk = min(MAX_ATTACK_DICE, s.armies[fi] - 1)
        if action.dice > max_atk:
            raise ValueError(f"Attack dice {action.dice} exceeds max {max_atk}")

        defender_pid = s.owners[ti]
        d_dice = min(MAX_DEFEND_DICE, s.armies[ti])
        atk_rolls = sorted((self._rng.randint(1, DIE_SIDES) for _ in range(action.dice)), reverse=True)
        def_rolls = sorted((self._rng.randint(1, DIE_SIDES) for _ in range(d_dice)), reverse=True)
        atk_losses = def_losses = 0
        for a, d in zip(atk_rolls, def_rolls):
            if a > d:
                def_losses += 1
            else:
                atk_losses += 1
        s.armies[fi] -= atk_losses
        s.armies[ti] -= def_losses

        info = {
            "attacker_rolls": atk_rolls,
            "defender_rolls": def_rolls,
            "attacker_losses": atk_losses,
            "defender_losses": def_losses,
            "conquered": False,
            "eliminated": None,
            "winner": None,
        }

        if s.armies[ti] == 0:
            # Conquest: transfer ownership now, but defer the army move to
            # a separate OccupyAction. The attacker must move at least
            # `action.dice` armies in (or whatever is available if the
            # attacker has fewer), up to `armies_in_from - 1`.
            s.owners[ti] = pid
            info["conquered"] = True

            # Card draw on first conquest of the turn.
            if not s.conquered_this_turn:
                card = self._draw_card()
                if card is not None:
                    s.hands[pid].append(card)
                s.conquered_this_turn = True

            # Elimination check (done before occupy so cards transfer
            # correctly and the winner can be detected early).
            assert defender_pid is not None
            still_owns_any = any(o == defender_pid for o in s.owners)
            if not still_owns_any:
                s.eliminated.add(defender_pid)
                # Transfer cards.
                taken = s.hands[defender_pid]
                s.hands[pid].extend(taken)
                s.hands[defender_pid] = []
                info["eliminated"] = defender_pid
                # Cards transfer to the winner. They must trade down to < 5
                # before placing armies — the UI enforces this at reinforce.
                # (AI agents pick TradeInAction from legal_actions naturally.)

            # Park the attack so the next step is an OccupyAction.
            s.pending_attack = PendingAttack(
                from_index=fi, to_index=ti, attacker_dice=action.dice
            )
            s.phase = Phase.OCCUPY

        # Winner detection.
        w = self.winner()
        if w is not None:
            info["winner"] = w
        return info

    def _legal_attack(self, s: State) -> Iterable[Action]:
        pid = s.current_player_index
        for i, o in enumerate(s.owners):
            if o != pid or s.armies[i] < 2:
                continue
            t = self.topology.territory_at(i)
            for nb in self.topology.neighbors(t):
                ni = self.topology.index_of(nb)
                if s.owners[ni] == pid:
                    continue
                max_d = min(MAX_ATTACK_DICE, s.armies[i] - 1)
                for d in range(1, max_d + 1):
                    yield AttackAction(from_territory=t, to_territory=nb, dice=d)
        yield StopAttackAction()

    def _apply_stop_attack(self, s: State) -> dict:
        s.phase = Phase.FORTIFY
        return {}

    # --- occupy ----------------------------------------------------------

    def _occupy_bounds(self, s: State) -> tuple[int, int]:
        """Return ``(min_count, max_count)`` for the pending occupy."""
        pa = s.pending_attack
        if pa is None:
            raise ValueError("No pending conquest to occupy")
        fi = pa.from_index
        available = s.armies[fi] - 1
        if available < 1:
            # Edge case: only one army remains on the attacker square; the
            # rules require at least one to move into the conquered region.
            return (1, 1)
        lo = max(1, min(pa.attacker_dice, available))
        hi = available
        if lo > hi:
            lo = hi
        return (lo, hi)

    def _legal_occupy(self, s: State) -> Iterable[Action]:
        if s.pending_attack is None:
            return
        lo, hi = self._occupy_bounds(s)
        for c in range(lo, hi + 1):
            yield OccupyAction(count=c)

    def _apply_occupy(self, s: State, action: OccupyAction) -> dict:
        if s.phase is not Phase.OCCUPY or s.pending_attack is None:
            raise ValueError("OccupyAction is only legal during Phase.OCCUPY")
        lo, hi = self._occupy_bounds(s)
        if not (lo <= action.count <= hi):
            raise ValueError(
                f"OccupyAction count {action.count} not in [{lo}, {hi}]"
            )
        pa = s.pending_attack
        s.armies[pa.from_index] -= action.count
        s.armies[pa.to_index] += action.count
        s.pending_attack = None
        s.phase = Phase.ATTACK
        return {"occupied": action.count}

    # --- fortify ---------------------------------------------------------

    def _connected_through_owned(self, s: State, pid: int, start: str, goal: str) -> bool:
        start_i = self.topology.index_of(start)
        goal_i = self.topology.index_of(goal)
        if s.owners[start_i] != pid or s.owners[goal_i] != pid:
            return False
        seen = {start_i}
        stack = [start_i]
        while stack:
            cur = stack.pop()
            if cur == goal_i:
                return True
            for nb in self.topology.neighbors(self.topology.territory_at(cur)):
                ni = self.topology.index_of(nb)
                if ni in seen or s.owners[ni] != pid:
                    continue
                seen.add(ni)
                stack.append(ni)
        return False

    def _apply_fortify(self, s: State, action: FortifyAction) -> dict:
        pid = s.current_player_index
        if not action.is_skip:
            assert action.from_territory is not None and action.to_territory is not None
            fi = self.topology.index_of(action.from_territory)
            ti = self.topology.index_of(action.to_territory)
            if s.owners[fi] != pid or s.owners[ti] != pid:
                raise ValueError("Fortify between owned territories only")
            if action.count >= s.armies[fi]:
                raise ValueError(
                    f"Must leave >=1 army on source ({action.from_territory})"
                )
            if not self._connected_through_owned(s, pid, action.from_territory, action.to_territory):
                raise ValueError("Fortify endpoints not connected through owned territories")
            s.armies[fi] -= action.count
            s.armies[ti] += action.count

        # End of turn -> advance to next non-eliminated player.
        self._advance_turn(s)
        return {"moved": 0 if action.is_skip else action.count}

    def _legal_fortify(self, s: State) -> Iterable[Action]:
        pid = s.current_player_index
        # Always allow skipping.
        yield FortifyAction(from_territory=None, to_territory=None, count=0)
        # Enumerate move endpoints — only count=1..armies-1 for tractability;
        # agents that want larger counts can submit them directly.
        for i, o in enumerate(s.owners):
            if o != pid or s.armies[i] < 2:
                continue
            t = self.topology.territory_at(i)
            # Find all owned territories reachable through owned chain.
            for j, oj in enumerate(s.owners):
                if i == j or oj != pid:
                    continue
                u = self.topology.territory_at(j)
                if self._connected_through_owned(s, pid, t, u):
                    yield FortifyAction(from_territory=t, to_territory=u, count=s.armies[i] - 1)

    def _advance_turn(self, s: State) -> None:
        n = self.settings.player_count
        for _ in range(n):
            s.current_player_index = (s.current_player_index + 1) % n
            if s.current_player_index not in s.eliminated:
                break
        else:
            return  # only one alive — winner handles termination
        self._begin_turn_for(s, s.current_player_index)

    def _begin_turn_for(self, s: State, pid: int) -> None:
        s.phase = Phase.TRADE_IN
        s.reinforcement_budget = self._compute_reinforcement(s, pid)
        s.conquered_this_turn = False
        # If no valid set exists, there is no trade decision to make, so the
        # turn starts directly in REINFORCE_PLACE.
        if not any(self._legal_trade_ins(s)):
            s.phase = Phase.REINFORCE_PLACE


__all__ = ["Environment", "StepResult"]
