"""Tests for `Environment` (Phase 4)."""
from __future__ import annotations

import copy as _copy

import pytest

from risk.game.actions import (
    ActionStage,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    SkipTradeAction,
    StopAttackAction,
)
from .conftest import make_env, make_settings
from risk.game.board_topology import BoardTopology
from risk.constants import MIN_REINFORCEMENT
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.player import Player
from risk.game.settings import GameSettings


# --- helpers --------------------------------------------------------------


def _settings(n: int = 3, seed: int = 1234) -> GameSettings:
    return make_settings(n=n, seed=seed, agent_kind="human")


def _fresh_env(n: int = 3, seed: int = 1234) -> Environment:
    return make_env(n=n, seed=seed, agent_kind="human")


def _skip_to_reinforce_place(env: Environment) -> None:
    """Every fresh turn starts in TRADE_IN; skip it to reach REINFORCE_PLACE."""
    env.step(SkipTradeAction())


# --- reset ----------------------------------------------------------------


def test_reset_deterministic_with_seed() -> None:
    env_a = _fresh_env(seed=42)
    env_b = _fresh_env(seed=42)
    assert env_a.current_state().owners == env_b.current_state().owners
    assert env_a.current_state().armies == env_b.current_state().armies


def test_reset_different_seeds_differ() -> None:
    a = _fresh_env(seed=1).current_state()
    b = _fresh_env(seed=2).current_state()
    assert a.owners != b.owners or a.armies != b.armies


def test_reset_every_territory_owned_with_one_army_minimum() -> None:
    env = _fresh_env()
    s = env.current_state()
    assert all(o is not None for o in s.owners)
    assert all(a >= 1 for a in s.armies)


def test_reset_starting_armies_correct_per_player() -> None:
    from risk.constants import starting_armies_for

    for n in (3, 4, 5, 6):
        env = _fresh_env(n=n, seed=99)
        s = env.current_state()
        expected = starting_armies_for(n)
        for pid in range(n):
            total = sum(a for i, a in enumerate(s.armies) if s.owners[i] == pid)
            assert total == expected, f"player {pid} of {n}: got {total} expected {expected}"


def test_reset_phase_is_trade_in_for_player_zero() -> None:
    env = _fresh_env()
    s = env.current_state()
    assert s.phase is Phase.TRADE_IN
    assert s.current_player_index == 0
    assert s.reinforcement_budget >= MIN_REINFORCEMENT


def test_skip_trade_advances_to_reinforce_place() -> None:
    env = _fresh_env()
    s = env.current_state()
    budget_before = s.reinforcement_budget
    _skip_to_reinforce_place(env)
    assert s.phase is Phase.REINFORCE_PLACE
    assert s.current_player_index == 0
    assert s.reinforcement_budget == budget_before


# --- reinforcement counts -------------------------------------------------


def test_reinforcement_at_minimum_for_few_territories() -> None:
    env = _fresh_env()
    s = env.current_state()
    # Player 0 owns 14 territories in a 3-player game -> 14//3 = 4 > 3.
    assert s.reinforcement_budget >= MIN_REINFORCEMENT


def test_reinforcement_includes_continent_bonus() -> None:
    env = _fresh_env()
    s = env.current_state()
    pid = 0
    # Force-give player 0 all of Australia.
    aus = list(env.topology.territories_in("Australia"))
    for t in aus:
        i = env.topology.index_of(t)
        s.owners[i] = pid
        if s.armies[i] == 0:
            s.armies[i] = 1
    bonus = env.topology.continent_bonus("Australia")
    # Recompute by calling private (test-only access).
    expected = env._compute_reinforcement(s, pid)
    owned = sum(1 for o in s.owners if o == pid)
    assert expected >= max(MIN_REINFORCEMENT, owned // 3) + bonus


# --- multi-step reinforcement ----------------------------------------------


def test_reinforce_partial_placement_stays_in_reinforce() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = s.current_player_index
    budget = s.reinforcement_budget
    assert budget >= 2, "test needs a budget of at least 2 to split"
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]

    env.step(ReinforcementAction(placements={owned[0]: 1}))
    assert s.phase is Phase.REINFORCE_PLACE
    assert s.current_player_index == pid
    assert s.reinforcement_budget == budget - 1

    # Placing the rest finishes reinforcement and advances to ATTACK.
    env.step(ReinforcementAction(placements={owned[0]: budget - 1}))
    assert s.phase is Phase.ATTACK
    assert s.reinforcement_budget == 0


def test_reinforce_full_budget_in_one_step_still_ends_phase() -> None:
    """A human (or any agent) placing the whole budget at once keeps working
    exactly as before — multi-step is optional, not required."""
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == s.current_player_index]
    budget = s.reinforcement_budget

    env.step(ReinforcementAction(placements={owned[0]: budget}))
    assert s.phase is Phase.ATTACK
    assert s.reinforcement_budget == 0


def test_reinforce_rejects_total_over_budget() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == s.current_player_index]
    with pytest.raises(ValueError):
        env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget + 1}))


def test_reinforce_rejects_outside_reinforce_place() -> None:
    env = _fresh_env()
    s = env.current_state()
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == s.current_player_index]
    with pytest.raises(ValueError):
        env.step(ReinforcementAction(placements={owned[0]: 1}))


def test_legal_reinforce_offers_one_half_all_buckets() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    budget = s.reinforcement_budget
    assert budget >= 4, "test needs a budget where one/half/all are distinct"
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == s.current_player_index]

    acts = [a for a in env.legal_actions() if isinstance(a, ReinforcementAction)]
    amounts_for_first = sorted(
        a.total for a in acts if owned[0] in a.placements
    )
    assert amounts_for_first == sorted({1, budget // 2, budget})
    # Bounded: at most 3 amounts per owned territory, never one-per-army-unit.
    assert len(acts) == len(owned) * len(amounts_for_first)


# --- legal actions per phase ----------------------------------------------


def test_legal_actions_in_trade_in_only_trade_in() -> None:
    env = _fresh_env()
    acts = env.legal_actions()
    assert all(a.stage is ActionStage.TRADE_IN for a in acts)
    assert len(acts) > 0


def test_legal_actions_in_reinforce_place_only_reinforce() -> None:
    env = _fresh_env()
    _skip_to_reinforce_place(env)
    acts = env.legal_actions()
    assert all(isinstance(a, ReinforcementAction) for a in acts)
    assert len(acts) > 0


def test_legal_actions_in_attack_include_stop_and_adjacency() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    # Place all reinforcements on one owned territory and step to ATTACK.
    pid = 0
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget}))
    acts = env.legal_actions()
    assert any(isinstance(a, StopAttackAction) for a in acts)
    # All attack actions use adjacency.
    for a in acts:
        if isinstance(a, AttackAction):
            assert env.topology.are_adjacent(a.from_territory, a.to_territory)
            fi = env.topology.index_of(a.from_territory)
            ti = env.topology.index_of(a.to_territory)
            assert s.owners[fi] == pid
            assert s.owners[ti] != pid
            assert 1 <= a.dice <= 3


def test_legal_actions_in_fortify_includes_skip() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = s.current_player_index
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget}))
    env.step(StopAttackAction())
    acts = env.legal_actions()
    assert any(isinstance(a, FortifyAction) and a.is_skip for a in acts)


# --- illegal actions raise & don't mutate ---------------------------------


def _snapshot(s):
    return (list(s.owners), list(s.armies), s.phase, s.current_player_index,
            s.reinforcement_budget)


def test_illegal_reinforce_does_not_mutate() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    before = _snapshot(s)
    bad_terr = next(
        env.topology.territory_at(i)
        for i, o in enumerate(s.owners)
        if o != s.current_player_index
    )
    with pytest.raises(ValueError):
        env.step(ReinforcementAction(placements={bad_terr: s.reinforcement_budget}))
    assert _snapshot(s) == before


def test_illegal_attack_does_not_mutate() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = 0
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget}))
    before = _snapshot(s)
    # Attack a non-adjacent territory.
    far_target = None
    for j, o in enumerate(s.owners):
        if o == pid:
            continue
        t_to = env.topology.territory_at(j)
        if not env.topology.are_adjacent(owned[0], t_to):
            far_target = t_to
            break
    assert far_target is not None
    with pytest.raises(ValueError):
        env.step(AttackAction(from_territory=owned[0], to_territory=far_target, dice=1))
    assert _snapshot(s) == before


# --- dice & battle resolution --------------------------------------------


def test_dice_resolution_bounds_over_many_rolls() -> None:
    """Sanity: across many seeded reinforce->attack cycles, dice losses sum
    to 1 (1v1), 2 (3v2 etc.) per battle."""
    env = _fresh_env(seed=7)
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = 0
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    # Pour all reinforcements into one strong territory.
    strong = max(owned, key=lambda t: s.armies[env.topology.index_of(t)])
    env.step(ReinforcementAction(placements={strong: s.reinforcement_budget}))
    # Find an enemy neighbor.
    target = next(
        nb for nb in env.topology.neighbors(strong)
        if s.owners[env.topology.index_of(nb)] != pid
    )
    for _ in range(20):
        si = env.topology.index_of(strong)
        if s.armies[si] < 2:
            break
        ti = env.topology.index_of(target)
        if s.owners[ti] == pid:
            break
        max_a = min(3, s.armies[si] - 1)
        max_d = min(2, s.armies[ti])
        res = env.step(AttackAction(from_territory=strong, to_territory=target, dice=max_a))
        info = res.info
        assert info["attacker_losses"] + info["defender_losses"] == min(max_a, max_d)


def test_conquest_transfers_owner_and_armies() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    s = env.current_state()
    pid = 0
    # Engineer a trivial conquest: pick one owned territory with neighbor enemy,
    # then load the attacker very heavily and weaken the defender.
    attacker = next(
        env.topology.territory_at(i)
        for i, o in enumerate(s.owners) if o == pid
        and any(s.owners[env.topology.index_of(n)] != pid
                for n in env.topology.neighbors(env.topology.territory_at(i)))
    )
    target = next(
        nb for nb in env.topology.neighbors(attacker)
        if s.owners[env.topology.index_of(nb)] != pid
    )
    ai = env.topology.index_of(attacker)
    ti = env.topology.index_of(target)
    s.armies[ai] = 50
    s.armies[ti] = 1
    _skip_to_reinforce_place(env)
    env.step(ReinforcementAction(placements={attacker: s.reinforcement_budget}))
    # Hammer until conquered.
    for _ in range(50):
        if s.owners[ti] == pid:
            break
        env.step(AttackAction(from_territory=attacker, to_territory=target, dice=3))
        if s.phase is Phase.OCCUPY:
            env.step(OccupyAction(count=env.legal_actions()[0].count))
    assert s.owners[ti] == pid
    assert s.armies[ti] >= 1


def test_card_drawn_on_first_conquest_of_turn() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    s = env.current_state()
    pid = 0
    attacker = next(
        env.topology.territory_at(i)
        for i, o in enumerate(s.owners) if o == pid
        and any(s.owners[env.topology.index_of(n)] != pid
                for n in env.topology.neighbors(env.topology.territory_at(i)))
    )
    target = next(
        nb for nb in env.topology.neighbors(attacker)
        if s.owners[env.topology.index_of(nb)] != pid
    )
    ai = env.topology.index_of(attacker)
    ti = env.topology.index_of(target)
    s.armies[ai] = 50
    s.armies[ti] = 1
    hand_before = len(s.hands[pid])
    _skip_to_reinforce_place(env)
    env.step(ReinforcementAction(placements={attacker: s.reinforcement_budget}))
    for _ in range(50):
        if s.owners[ti] == pid:
            break
        env.step(AttackAction(from_territory=attacker, to_territory=target, dice=3))
        if s.phase is Phase.OCCUPY:
            env.step(OccupyAction(count=env.legal_actions()[0].count))
    assert len(s.hands[pid]) == hand_before + 1


# --- occupy --------------------------------------------------------------


def _force_conquest(env: Environment, pid: int) -> tuple[str, str, int]:
    """Drive `env` to a conquest by `pid` and return (attacker, target, dice).

    Leaves the env in `Phase.OCCUPY` with a pending attack ready to resolve.
    """
    s = env.current_state()
    attacker = next(
        env.topology.territory_at(i)
        for i, o in enumerate(s.owners) if o == pid
        and any(s.owners[env.topology.index_of(n)] != pid
                for n in env.topology.neighbors(env.topology.territory_at(i)))
    )
    target = next(
        nb for nb in env.topology.neighbors(attacker)
        if s.owners[env.topology.index_of(nb)] != pid
    )
    ai = env.topology.index_of(attacker)
    ti = env.topology.index_of(target)
    s.armies[ai] = 50
    s.armies[ti] = 1
    _skip_to_reinforce_place(env)
    env.step(ReinforcementAction(placements={attacker: s.reinforcement_budget}))
    while s.phase is not Phase.OCCUPY:
        env.step(AttackAction(from_territory=attacker, to_territory=target, dice=3))
    return attacker, target, 3


def test_conquest_enters_occupy_phase_and_does_not_move_armies() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    pid = 0
    _attacker, target, _dice = _force_conquest(env, pid)
    s = env.current_state()
    ti = env.topology.index_of(target)
    assert s.phase is Phase.OCCUPY
    assert s.owners[ti] == pid
    # Conquered territory has 0 armies until the OccupyAction lands.
    assert s.armies[ti] == 0
    assert s.pending_attack is not None
    assert s.pending_attack.to_index == ti


def test_legal_actions_in_occupy_span_dice_to_armies_minus_one() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    pid = 0
    attacker, _target, dice = _force_conquest(env, pid)
    s = env.current_state()
    ai = env.topology.index_of(attacker)
    counts = sorted(a.count for a in env.legal_actions())
    assert counts == list(range(dice, s.armies[ai]))


def test_occupy_moves_armies_and_returns_to_attack() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    pid = 0
    attacker, target, _dice = _force_conquest(env, pid)
    s = env.current_state()
    ai = env.topology.index_of(attacker)
    ti = env.topology.index_of(target)
    armies_before = s.armies[ai]
    env.step(OccupyAction(count=5))
    assert s.armies[ai] == armies_before - 5
    assert s.armies[ti] == 5
    assert s.phase is Phase.ATTACK
    assert s.pending_attack is None


def test_occupy_rejects_count_below_dice() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    pid = 0
    _attacker, _target, dice = _force_conquest(env, pid)
    with pytest.raises(ValueError):
        env.step(OccupyAction(count=dice - 1))


def test_occupy_rejects_count_above_available() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    pid = 0
    attacker, _target, _dice = _force_conquest(env, pid)
    s = env.current_state()
    ai = env.topology.index_of(attacker)
    with pytest.raises(ValueError):
        env.step(OccupyAction(count=s.armies[ai]))  # would leave 0 behind


# --- fortify & winner -----------------------------------------------------


def test_fortify_skip_advances_turn() -> None:
    env = _fresh_env()
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = s.current_player_index
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget}))
    env.step(StopAttackAction())
    env.step(FortifyAction(from_territory=None, to_territory=None, count=0))
    s2 = env.current_state()
    assert s2.current_player_index != pid
    assert s2.phase is Phase.TRADE_IN


def test_fortify_requires_connected_owned_chain() -> None:
    env = _fresh_env(seed=0)
    s = env.current_state()
    _skip_to_reinforce_place(env)
    pid = s.current_player_index
    owned = [env.topology.territory_at(i) for i, o in enumerate(s.owners) if o == pid]
    env.step(ReinforcementAction(placements={owned[0]: s.reinforcement_budget}))
    env.step(StopAttackAction())
    # Find two owned territories not connected through owned chain.
    disconnected = None
    for a in owned:
        for b in owned:
            if a == b:
                continue
            if not env._connected_through_owned(s, pid, a, b):
                disconnected = (a, b)
                break
        if disconnected:
            break
    if disconnected is None:
        pytest.skip("Lucky seed: all owned territories connected.")
    a, b = disconnected
    ai = env.topology.index_of(a)
    if s.armies[ai] < 2:
        s.armies[ai] = 5
    with pytest.raises(ValueError):
        env.step(FortifyAction(from_territory=a, to_territory=b, count=1))


def test_winner_detected_when_only_one_alive() -> None:
    env = Environment()
    env.reset(_settings(n=3, seed=0))
    s = env.current_state()
    # Force everyone but player 0 eliminated.
    for i in range(len(s.owners)):
        s.owners[i] = 0
    s.eliminated = {1, 2}
    assert env.winner() == 0


# --- terminal step rejection ---------------------------------------------


def test_step_on_terminal_raises() -> None:
    env = _fresh_env()
    env.current_state().phase = Phase.GAME_OVER
    with pytest.raises(RuntimeError):
        env.step(StopAttackAction())
