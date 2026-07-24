"""Tests for `RewardCalculator` terminal behavior and per-phase shaping."""
from __future__ import annotations

import pytest

from risk.constants import card_set_value
from risk.game.actions import (
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    SkipTradeAction,
    StopAttackAction,
    TradeInAction,
)
from risk.game.card import Card
from risk.game.state import PendingAttack
from risk.learning.reward import RewardCalculator
from risk.learning.train_constants import (
    REWARD_ARMY_DELTA_RELATIVE_SCALE,
    REWARD_ATTACK_ARMY_TRADE,
    REWARD_ATTACK_CONQUER_TERRITORY,
    REWARD_ATTACK_CONQUER_WITH_CARD,
    REWARD_ATTACK_CONTINENT_ADVANTAGE,
    REWARD_ATTACK_CONTINENT_DOMINATION,
    REWARD_ATTACK_ELIMINATE_OPPONENT_BASE,
    REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD,
    REWARD_ATTACK_FEWER_DICE,
    REWARD_ATTACK_RATIO_CAP,
    REWARD_ATTACK_RATIO_SCALE,
    REWARD_ATTACK_RATIO_THRESHOLD,
    REWARD_ATTACK_STOP_WITHOUT_CARD,
    REWARD_ATTACK_UNFINISHED_TARGET,
    REWARD_CONTINENT_DELTA_RELATIVE,
    REWARD_CONTINENT_LOST,
    REWARD_FORTIFY_CONTINENT_PUSH,
    REWARD_FORTIFY_TOWARD_FRONTIER,
    REWARD_OCCUPY_FORWARD_MOMENTUM,
    REWARD_REINFORCE_CONTINENT_PUSH,
    REWARD_REINFORCE_NO_ENEMY_NEIGHBOR,
    REWARD_TERMINAL_LOSS,
    REWARD_TERMINAL_WIN,
    REWARD_TERRITORY_DELTA,
    REWARD_TERRITORY_HOLD,
    REWARD_TRADE_IN_EARLY,
    REWARD_TRADE_IN_TERRITORY_MATCH,
)

from .conftest import make_env


def _calculator_and_state():
    env = make_env(n=3, seed=7, agent_kind="human")
    calc = RewardCalculator(env.topology)
    state = env.current_state().snapshot()
    return calc, state


def test_compute_returns_zero_when_not_done() -> None:
    calc, state = _calculator_and_state()

    reward = calc.compute(
        action=SkipTradeAction(),
        info={},
        before=state,
        after=state,
        reward_player=0,
        done=False,
        winner=None,
    )

    assert reward == 0.0


def test_compute_returns_win_when_reward_player_is_winner() -> None:
    calc, state = _calculator_and_state()

    reward = calc.compute(
        action=SkipTradeAction(),
        info={},
        before=state,
        after=state,
        reward_player=1,
        done=True,
        winner=1,
    )

    assert reward == REWARD_TERMINAL_WIN


def test_compute_returns_loss_when_other_player_wins() -> None:
    calc, state = _calculator_and_state()

    reward = calc.compute(
        action=SkipTradeAction(),
        info={},
        before=state,
        after=state,
        reward_player=1,
        done=True,
        winner=2,
    )

    assert reward == REWARD_TERMINAL_LOSS


def test_compute_returns_loss_when_eliminated_before_game_over() -> None:
    """`done=True` with `winner=None` means `reward_player` was eliminated
    while other players are still fighting it out — still a loss for them,
    not a neutral/timeout result."""
    calc, state = _calculator_and_state()

    reward = calc.compute(
        action=SkipTradeAction(),
        info={},
        before=state,
        after=state,
        reward_player=1,
        done=True,
        winner=None,
    )

    assert reward == REWARD_TERMINAL_LOSS


def _calc_env():
    env = make_env(n=3, seed=7, agent_kind="human")
    return RewardCalculator(env.topology), env


# --- TRADE_IN -------------------------------------------------------------


def test_trade_in_skip_rewards_patience_at_hand_three() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.current_player_index = 0
    state.hands[0] = [
        Card(territory_id=topo.territories[0], symbol="infantry"),
        Card(territory_id=topo.territories[1], symbol="infantry"),
        Card(territory_id=topo.territories[2], symbol="infantry"),
    ]
    state.cards_traded_in_count = 0

    reward = calc.compute(
        action=SkipTradeAction(), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(REWARD_TRADE_IN_EARLY / card_set_value(0))


def test_trade_in_trade_penalizes_when_optional_and_no_match() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.current_player_index = 0
    t0, t1, t2 = topo.territories[0], topo.territories[1], topo.territories[2]
    for t in (t0, t1, t2):
        state.owners[topo.index_of(t)] = 1  # not owned by the trading player
    state.hands[0] = [
        Card(territory_id=t0, symbol="infantry"),
        Card(territory_id=t1, symbol="infantry"),
        Card(territory_id=t2, symbol="infantry"),
    ]
    state.cards_traded_in_count = 0

    reward = calc.compute(
        action=TradeInAction(card_indices=(0, 1, 2)), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(-(REWARD_TRADE_IN_EARLY / card_set_value(0)))


def test_trade_in_territory_match_bonus() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.current_player_index = 0
    t0, t1, t2 = topo.territories[0], topo.territories[1], topo.territories[2]
    state.owners[topo.index_of(t0)] = 0  # trading player owns this card's territory
    state.owners[topo.index_of(t1)] = 1
    state.owners[topo.index_of(t2)] = 1
    state.hands[0] = [
        Card(territory_id=t0, symbol="infantry"),
        Card(territory_id=t1, symbol="infantry"),
        Card(territory_id=t2, symbol="infantry"),
    ]
    state.cards_traded_in_count = 0

    reward = calc.compute(
        action=TradeInAction(card_indices=(0, 1, 2)), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    expected = -(REWARD_TRADE_IN_EARLY / card_set_value(0)) + REWARD_TRADE_IN_TERRITORY_MATCH
    assert reward == pytest.approx(expected)


# --- REINFORCE_PLACE --------------------------------------------------------


def test_reinforce_no_enemy_neighbor_penalty() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.current_player_index = 0
    state.owners = [0] * len(state.owners)  # everyone owned by player 0 -> fully interior
    terr = topo.territories[0]
    idx = topo.index_of(terr)
    state.armies[idx] = 5
    state.reinforcement_budget = 10
    after = state.snapshot()
    after.armies[idx] = 5 + 3

    reward = calc.compute(
        action=ReinforcementAction(placements={terr: 3}), info={}, before=state, after=after,
        reward_player=0, done=False, winner=None,
    )

    continent = topo.continent_of(terr)
    owned, total = topo.continent_owner_counts(after.owners, continent, 0)
    expected = REWARD_REINFORCE_NO_ENEMY_NEIGHBOR + REWARD_REINFORCE_CONTINENT_PUSH * (owned / total) / total
    assert reward == pytest.approx(expected)


# --- ATTACK ------------------------------------------------------------------


def test_attack_fewer_dice_penalty_isolated() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)
    from_t, to_t = "Afghanistan", "China"
    fi, ti = topo.index_of(from_t), topo.index_of(to_t)
    state.owners[fi] = 0
    state.armies[fi] = 3
    state.armies[ti] = 2  # ratio = 1.5 == REWARD_ATTACK_RATIO_THRESHOLD -> ratio term is 0
    state.current_player_index = 0
    state.eliminated = set()
    state.hands = [[], [], []]
    state.conquered_this_turn = False

    action = AttackAction(from_territory=from_t, to_territory=to_t, dice=1)  # max dice would be 2
    info = {"defender_losses": 0, "attacker_losses": 0, "conquered": False, "eliminated": None}

    reward = calc.compute(
        action=action, info=info, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(REWARD_ATTACK_FEWER_DICE)


def test_attack_conquer_and_card_draw() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)
    from_t, to_t = "Afghanistan", "China"
    fi, ti = topo.index_of(from_t), topo.index_of(to_t)
    state.owners[fi] = 0
    state.armies[fi] = 3
    state.armies[ti] = 2  # ratio = 1.5 == REWARD_ATTACK_RATIO_THRESHOLD -> ratio term is 0
    state.current_player_index = 0
    state.eliminated = set()
    state.hands = [[], [], []]
    state.conquered_this_turn = False

    after = state.snapshot()
    after.owners[ti] = 0
    drawn = Card(territory_id="Alaska", symbol="infantry")
    after.owners[topo.index_of("Alaska")] = 1  # not owned by the attacker -> no card-territory-match bonus
    after.hands[0] = [drawn]

    action = AttackAction(from_territory=from_t, to_territory=to_t, dice=2)  # == max dice, no fewer-dice penalty
    info = {"defender_losses": 1, "attacker_losses": 0, "conquered": True, "eliminated": None}

    reward = calc.compute(
        action=action, info=info, before=state, after=after,
        reward_player=0, done=False, winner=None,
    )

    expected = REWARD_ATTACK_ARMY_TRADE * 1 + REWARD_ATTACK_CONQUER_TERRITORY + REWARD_ATTACK_CONQUER_WITH_CARD
    assert reward == pytest.approx(expected)


def test_attack_eliminate_opponent_scales_with_cards_taken() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)
    from_t, to_t = "Afghanistan", "China"
    fi, ti = topo.index_of(from_t), topo.index_of(to_t)
    state.owners[fi] = 0
    state.armies[fi] = 3
    state.armies[ti] = 2  # ratio == threshold -> ratio term is 0
    state.current_player_index = 0
    state.eliminated = set()
    state.hands = [
        [],
        [Card(territory_id="Alaska", symbol="infantry"), Card(territory_id="Iceland", symbol="cavalry")],
        [],
    ]
    state.conquered_this_turn = True  # already drew this turn -> no card-draw term to isolate the eliminate bonus

    after = state.snapshot()
    after.owners[ti] = 0

    action = AttackAction(from_territory=from_t, to_territory=to_t, dice=2)  # == max dice, no fewer-dice penalty
    info = {"defender_losses": 0, "attacker_losses": 0, "conquered": True, "eliminated": 1}

    reward = calc.compute(
        action=action, info=info, before=state, after=after,
        reward_player=0, done=False, winner=None,
    )

    expected_eliminate = REWARD_ATTACK_ELIMINATE_OPPONENT_BASE + REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD * 2
    expected = REWARD_ATTACK_CONQUER_TERRITORY + expected_eliminate
    assert reward == pytest.approx(expected)

    # The eliminate bonus is logged in its own bucket, split out of `attack`
    # (Docs/Reward.md, Finding 11's open item), so W&B can see elimination
    # frequency/magnitude independently of ordinary conquest reward.
    assert calc.last_components["eliminate"] == pytest.approx(expected_eliminate)
    assert calc.last_components["attack"] == pytest.approx(REWARD_ATTACK_CONQUER_TERRITORY)


def test_attack_stop_without_card_penalty() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)
    idx = topo.index_of("Afghanistan")
    state.owners[idx] = 0
    state.armies[idx] = 3  # >= 2 armies, and a frontier territory (its neighbors are owner 1)
    state.current_player_index = 0
    state.conquered_this_turn = False

    reward = calc.compute(
        action=StopAttackAction(), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(REWARD_ATTACK_STOP_WITHOUT_CARD)


def test_attack_stop_with_unfinished_targets_and_no_conquest_keeps_one_time_penalty() -> None:
    calc, env = _calc_env()
    state = env.current_state().snapshot()
    state.current_player_index = 0
    state.conquered_this_turn = False
    state.unfinished_attack_targets_this_turn = {1, 2}

    reward = calc.compute(
        action=StopAttackAction(), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(REWARD_ATTACK_STOP_WITHOUT_CARD)
    assert calc.last_components["unfinished_attack"] == pytest.approx(0.0)


def test_attack_stop_penalizes_each_unfinished_target_after_a_conquest() -> None:
    calc, env = _calc_env()
    state = env.current_state().snapshot()
    state.current_player_index = 0
    state.conquered_this_turn = True
    state.unfinished_attack_targets_this_turn = {1, 2}

    reward = calc.compute(
        action=StopAttackAction(), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    expected = 2 * REWARD_ATTACK_UNFINISHED_TARGET
    assert reward == pytest.approx(expected)
    assert calc.last_components["unfinished_attack"] == pytest.approx(expected)
    assert calc.last_components["attack"] == pytest.approx(0.0)


def test_attack_continent_advantage_rewards_real_edge() -> None:
    """Owning most of Australia with troop majority should add a positive,
    gated continent-advantage term on top of ratio/domination."""
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)

    eastern, new_guinea, western, indonesia = (
        "EasternAustralia", "NewGuinea", "WesternAustralia", "Indonesia"
    )
    for terr in (eastern, new_guinea, western):
        state.owners[topo.index_of(terr)] = 0
        state.armies[topo.index_of(terr)] = 5
    state.armies[topo.index_of(indonesia)] = 3  # stays owner 1 (the attack target)

    state.current_player_index = 0
    state.eliminated = set()
    state.hands = [[], [], []]
    state.conquered_this_turn = False

    fi, ti = topo.index_of(western), topo.index_of(indonesia)
    action = AttackAction(from_territory=western, to_territory=indonesia, dice=3)  # == max dice
    info = {"defender_losses": 0, "attacker_losses": 0, "conquered": False, "eliminated": None}

    reward = calc.compute(
        action=action, info=info, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    continent = topo.continent_of(indonesia)
    owned, total = topo.continent_owner_counts(state.owners, continent, 0)
    alive_players = 3
    my_troops = sum(state.armies[i] for i in topo.continent_member_indices(continent) if state.owners[i] == 0)
    other_troops = sum(state.armies[i] for i in topo.continent_member_indices(continent) if state.owners[i] != 0)
    baseline = 1 / alive_players
    territory_edge = max(0.0, owned / total - baseline) / (1 - baseline)
    troop_edge = max(0.0, my_troops / (my_troops + other_troops) - 0.5) / 0.5
    worth = topo.continent_bonus(continent) / max(topo.continent_bonus(c) for c in topo.continents)
    advantage = territory_edge * troop_edge * worth

    ratio = state.armies[fi] / state.armies[ti]
    expected_ratio = REWARD_ATTACK_RATIO_SCALE * (min(ratio, REWARD_ATTACK_RATIO_CAP) - REWARD_ATTACK_RATIO_THRESHOLD)
    margin_left = total - owned
    expected_domination = REWARD_ATTACK_CONTINENT_DOMINATION * (1.0 / (margin_left + 1))
    expected_advantage = REWARD_ATTACK_CONTINENT_ADVANTAGE * advantage / (margin_left + 1)

    assert advantage > 0  # sanity: the scenario really does cross both gates
    assert reward == pytest.approx(expected_ratio + expected_domination + expected_advantage)


def test_attack_continent_advantage_zero_below_territory_baseline() -> None:
    """Owning only 1 of 12 Asia territories (well under the 1/alive_players
    baseline) must contribute zero, regardless of troop majority there."""
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [1] * len(state.owners)
    from_t, to_t = "Afghanistan", "China"
    fi, ti = topo.index_of(from_t), topo.index_of(to_t)
    state.owners[fi] = 0
    state.armies[fi] = 20  # troop majority in Asia, but territory share is far below baseline
    state.armies[ti] = 2
    state.current_player_index = 0
    state.eliminated = set()
    state.hands = [[], [], []]
    state.conquered_this_turn = False

    action = AttackAction(from_territory=from_t, to_territory=to_t, dice=3)  # == max dice, no fewer-dice penalty
    info = {"defender_losses": 0, "attacker_losses": 0, "conquered": False, "eliminated": None}

    reward = calc.compute(
        action=action, info=info, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    ratio = state.armies[fi] / state.armies[ti]
    expected_ratio = REWARD_ATTACK_RATIO_SCALE * (min(ratio, REWARD_ATTACK_RATIO_CAP) - REWARD_ATTACK_RATIO_THRESHOLD)
    assert reward == pytest.approx(expected_ratio)  # no domination, no advantage term


# --- OCCUPY ------------------------------------------------------------------


def test_occupy_forward_momentum_scales_with_fraction() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    fi = topo.index_of(topo.territories[0])
    ti = topo.index_of(topo.territories[1])
    state.armies[fi] = 5  # available = armies - 1 = 4
    state.pending_attack = PendingAttack(from_index=fi, to_index=ti, attacker_dice=2)
    state.current_player_index = 0

    reward = calc.compute(
        action=OccupyAction(count=2), info={}, before=state, after=state,
        reward_player=0, done=False, winner=None,
    )

    assert reward == pytest.approx(REWARD_OCCUPY_FORWARD_MOMENTUM * (2 / 4))


# --- FORTIFY -----------------------------------------------------------------


def test_fortify_toward_frontier_positive() -> None:
    calc, env = _calc_env()
    topo = env.topology
    state = env.current_state().snapshot()
    state.owners = [0] * len(state.owners)
    dst = "China"
    enemy_neighbor = topo.neighbors(dst)[0]
    state.owners[topo.index_of(enemy_neighbor)] = 1  # makes `dst` a frontier territory
    src = next(
        t for t in topo.territories
        if t not in (dst, enemy_neighbor) and enemy_neighbor not in topo.neighbors(t)
    )
    state.current_player_index = 0
    fi, ti = topo.index_of(src), topo.index_of(dst)
    state.armies[fi] = 5  # movable = 4
    dst_armies_before = state.armies[ti]

    after = state.snapshot()
    after.armies[fi] = 1
    after.armies[ti] = dst_armies_before + 4

    reward = calc.compute(
        action=FortifyAction(from_territory=src, to_territory=dst, count=4), info={},
        before=state, after=after, reward_player=0, done=False, winner=None,
    )

    continent = topo.continent_of(dst)
    owned, total = topo.continent_owner_counts(after.owners, continent, 0)
    expected = REWARD_FORTIFY_TOWARD_FRONTIER + REWARD_FORTIFY_CONTINENT_PUSH * (owned / total) / total
    assert reward == pytest.approx(expected)


# --- end-of-turn opponent-impact terms --------------------------------------


def test_end_of_turn_combines_territory_army_and_continent_delta() -> None:
    calc, env = _calc_env()
    topo = env.topology
    n = len(topo)
    before_turn = env.current_state().snapshot()
    before_turn.owners = [1] * n
    before_turn.armies = [1] * n
    after_turn = before_turn.snapshot()
    gained = topo.territories[0]
    after_turn.owners[topo.index_of(gained)] = 0
    after_turn.armies[topo.index_of(gained)] = 5

    reward = calc.end_of_turn(before_turn, after_turn, 0)

    territory_term = REWARD_TERRITORY_DELTA * (1 / n - 0 / n)
    total_before, total_after = sum(before_turn.armies), sum(after_turn.armies)
    army_term = REWARD_ARMY_DELTA_RELATIVE_SCALE * (5 / total_after - 0 / total_before)
    total_bonus = sum(topo.continent_bonus(c) for c in topo.continents)
    my_bonus_after = sum(
        topo.continent_bonus(c) for c in topo.continents if topo.owns_continent(after_turn.owners, c, 0)
    )
    continent_term = REWARD_CONTINENT_DELTA_RELATIVE * (my_bonus_after / total_bonus - 0.0)
    hold_term = REWARD_TERRITORY_HOLD * (1 / n)

    assert reward == pytest.approx(territory_term + army_term + continent_term + hold_term)


def test_end_of_turn_hold_reward_is_zero_without_territory_change() -> None:
    calc, env = _calc_env()
    topo = env.topology
    n = len(topo)
    before_turn = env.current_state().snapshot()
    before_turn.owners = [0] * n
    before_turn.armies = [1] * n
    after_turn = before_turn.snapshot()

    reward = calc.end_of_turn(before_turn, after_turn, 0)

    assert calc.last_end_of_turn_components["territory_delta"] == pytest.approx(0.0)
    assert calc.last_end_of_turn_components["territory_hold"] == pytest.approx(REWARD_TERRITORY_HOLD * 1.0)
    assert reward == pytest.approx(0.0)


def test_end_of_turn_continent_loss_applies_extra_penalty() -> None:
    calc, env = _calc_env()
    topo = env.topology
    n = len(topo)
    # Fully own the smallest continent before the opponents' round, then lose
    # one of its territories so the continent bonus is given up.
    continent = min(topo.continents, key=lambda c: len(list(topo.continent_member_indices(c))))
    members = list(topo.continent_member_indices(continent))
    before_turn = env.current_state().snapshot()
    before_turn.owners = [1] * n
    before_turn.armies = [1] * n
    for idx in members:
        before_turn.owners[idx] = 0
    after_turn = before_turn.snapshot()
    after_turn.owners[members[0]] = 1  # opponent retakes one -> continent lost

    calc.end_of_turn(before_turn, after_turn, 0)

    total_bonus = sum(topo.continent_bonus(c) for c in topo.continents)
    expected_lost = REWARD_CONTINENT_LOST * (-topo.continent_bonus(continent) / total_bonus)
    assert calc.last_end_of_turn_components["continent_lost"] == pytest.approx(expected_lost)
    assert calc.last_end_of_turn_components["continent_lost"] < 0


def test_end_of_turn_no_continent_loss_penalty_when_holding() -> None:
    calc, env = _calc_env()
    topo = env.topology
    n = len(topo)
    before_turn = env.current_state().snapshot()
    before_turn.owners = [0] * n
    before_turn.armies = [1] * n
    after_turn = before_turn.snapshot()

    calc.end_of_turn(before_turn, after_turn, 0)

    assert calc.last_end_of_turn_components["continent_lost"] == pytest.approx(0.0)
