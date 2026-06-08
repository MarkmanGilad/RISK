"""Tests for `risk.ui.human_input.HumanInputController` (Phase 10)."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from risk.agents.human_agent import HumanAgent
from risk.agents.random_agent import RandomAgent
from risk.game.actions import (
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    StopAttackAction,
    TradeInAction,
)
from risk.game.card import Card
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.player import Player
from risk.game.settings import GameSettings
from risk.ui.input.human_input import (
    HudActionPanelModel,
    HumanInputController,
)


# --- fixtures ------------------------------------------------------------


def _settings(human_ids: set[int], n: int = 3, seed: int = 1) -> GameSettings:
    return GameSettings(
        players=tuple(
            Player(
                id=i,
                name=f"P{i}",
                color=(i * 30 + 10, i * 40 + 10, i * 50 + 10),
                agent_kind="human" if i in human_ids else "ai",
            )
            for i in range(n)
        ),
        seed=seed,
    )


def _build(human_ids: set[int], seed: int = 1, n: int = 3):
    """Build env + agents + controller with the given seats marked human.

    The controller is owned by the first human seat (or seat 0 for all-AI
    tests, where `is_human_turn` is always False).
    """
    settings = _settings(human_ids, n=n, seed=seed)
    env = Environment()
    env.reset(settings)
    agents = []
    for p in settings.players:
        if p.agent_kind == "human":
            agents.append(HumanAgent(player_id=p.id, env=env, settings=settings))
        else:
            agents.append(RandomAgent(player_id=p.id, env=env, seed=seed + p.id + 1))
    owner = next((a for a in agents if isinstance(a, HumanAgent)), agents[0])
    controller = HumanInputController(env, owner, settings)
    return env, agents, controller


def _own_indices(env, pid: int) -> list[int]:
    s = env.current_state()
    return [i for i, o in enumerate(s.owners) if o == pid]


def _enemy_indices(env, pid: int) -> list[int]:
    s = env.current_state()
    return [i for i, o in enumerate(s.owners) if o != pid]


def _drive_to_attack_phase(env, agents, controller) -> None:
    """Submit a valid reinforce so the env transitions to ATTACK for the
    current human seat."""
    s = env.current_state()
    assert s.phase == Phase.REINFORCE
    pid = s.current_player_index
    owned = _own_indices(env, pid)
    assert owned
    terr = env.topology.territory_at(owned[0])
    placements = {terr: s.reinforcement_budget}
    agents[pid].submit(ReinforcementAction(placements=placements))
    # Tick the env via direct step (the controller test does not run
    # Game.tick; we drive the env directly for determinism).
    env.step(agents[pid]._pending if False else ReinforcementAction(placements=placements))
    # Re-sync the agent's pending after manual step.
    agents[pid].clear()
    controller.on_turn_change(env.current_state())


# --- on_turn_change & is_human_turn -------------------------------------


def test_is_human_turn_true_for_human_seat():
    env, agents, controller = _build(human_ids={0})
    state = env.current_state()
    assert state.current_player_index == 0
    assert controller.is_human_turn(state) is True


def test_is_human_turn_false_for_ai_seat():
    env, agents, controller = _build(human_ids=set())  # all AI
    assert controller.is_human_turn(env.current_state()) is False


def test_on_turn_change_clears_state_when_phase_changes():
    env, agents, controller = _build(human_ids={0})
    # Dirty the controller.
    controller.pending_placements = {"Ural": 3}
    controller.selected_from = 5
    controller.selected_to = 7
    # Simulate a transition to a new (player, phase).
    state = env.current_state()
    controller.on_turn_change(state)  # first call records the key
    # Mutate phase and verify reset happens on the next on_turn_change.
    state.phase = Phase.ATTACK
    controller.on_turn_change(state)
    assert controller.pending_placements == {}
    assert controller.selected_from is None
    assert controller.selected_to is None


# --- widgets() ----------------------------------------------------------


def test_widgets_inactive_for_ai_turn():
    env, agents, controller = _build(human_ids=set())
    m = controller.widgets(env.current_state())
    assert isinstance(m, HudActionPanelModel)
    assert m.header == ""
    assert m.is_active is False


def test_widgets_reinforce_shape():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    m = controller.widgets(s)
    assert m.header == "Your Turn (REINFORCE)"
    assert m.is_active is True
    assert m.info_lines[0].endswith(f"/ {s.reinforcement_budget}")
    ids = {b.id for b in m.buttons}
    assert ids == {"place_armies", "clear_all", "toggle_cards"}
    place_btn = next(b for b in m.buttons if b.id == "place_armies")
    assert place_btn.enabled is False  # no placements yet
    assert place_btn.primary is True


# --- REINFORCE flow -----------------------------------------------------


def test_reinforce_left_click_increments_and_button_submits():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    budget = s.reinforcement_budget
    owned = _own_indices(env, 0)[0]
    terr = env.topology.territory_at(owned)

    for _ in range(budget):
        controller.on_territory_click(owned, button=1)
    assert controller.pending_placements == {terr: budget}

    # Place Armies button now enabled in the model.
    m = controller.widgets(env.current_state())
    place_btn = next(b for b in m.buttons if b.id == "place_armies")
    assert place_btn.enabled is True

    controller.on_hud_button("place_armies")
    assert agents[0]._pending is not None
    assert isinstance(agents[0]._pending, ReinforcementAction)
    assert agents[0]._pending.placements == {terr: budget}
    # Pending placements emptied on successful submit.
    assert controller.pending_placements == {}


def test_reinforce_left_click_caps_at_budget():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    owned = _own_indices(env, 0)[0]
    for _ in range(s.reinforcement_budget + 5):
        controller.on_territory_click(owned, button=1)
    assert sum(controller.pending_placements.values()) == s.reinforcement_budget


def test_reinforce_right_click_decrements_and_never_negative():
    env, agents, controller = _build(human_ids={0})
    owned = _own_indices(env, 0)[0]
    controller.on_territory_click(owned, button=1)
    controller.on_territory_click(owned, button=1)
    controller.on_territory_click(owned, button=3)  # right-click decrements
    terr = env.topology.territory_at(owned)
    assert controller.pending_placements[terr] == 1
    controller.on_territory_click(owned, button=3)
    controller.on_territory_click(owned, button=3)  # third dec is a no-op
    assert terr not in controller.pending_placements


def test_reinforce_enemy_click_is_ignored():
    env, agents, controller = _build(human_ids={0})
    enemy = next(i for i in _enemy_indices(env, 0))
    controller.on_territory_click(enemy, button=1)
    assert controller.pending_placements == {}
    assert agents[0]._pending is None


def test_reinforce_clear_all_button_empties_pending():
    env, agents, controller = _build(human_ids={0})
    owned = _own_indices(env, 0)[0]
    controller.on_territory_click(owned, button=1)
    controller.on_territory_click(owned, button=1)
    assert controller.pending_placements
    controller.on_hud_button("clear_all")
    assert controller.pending_placements == {}
    assert agents[0]._pending is None


def test_reinforce_place_armies_does_nothing_until_budget_matched():
    env, agents, controller = _build(human_ids={0})
    owned = _own_indices(env, 0)[0]
    controller.on_territory_click(owned, button=1)  # 1 placed, budget > 1
    controller.on_hud_button("place_armies")
    assert agents[0]._pending is None


def test_reinforce_via_hud_field_increments():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    owned = _own_indices(env, 0)[0]
    terr = env.topology.territory_at(owned)
    for _ in range(s.reinforcement_budget):
        controller.on_hud_field("placements_inc", terr)
    assert controller.pending_placements[terr] == s.reinforcement_budget
    controller.on_hud_field("placements_dec", terr)
    assert controller.pending_placements[terr] == s.reinforcement_budget - 1
    controller.on_hud_field("placements_clear", terr)
    assert terr not in controller.pending_placements


# --- ATTACK flow --------------------------------------------------------


def _advance_to_attack(env, agents, controller, pid: int) -> None:
    """Step env directly: submit a legal reinforce so it transitions."""
    s = env.current_state()
    assert s.current_player_index == pid and s.phase == Phase.REINFORCE
    owned = _own_indices(env, pid)[0]
    terr = env.topology.territory_at(owned)
    env.step(ReinforcementAction(placements={terr: s.reinforcement_budget}))
    controller.on_turn_change(env.current_state())
    assert env.current_state().phase == Phase.ATTACK
    assert env.current_state().current_player_index == pid


def test_attack_select_from_then_to_then_submit():
    env, agents, controller = _build(human_ids={0})
    _advance_to_attack(env, agents, controller, pid=0)
    s = env.current_state()
    # Find an attackable pair for player 0.
    attack_action = next(
        (a for a in env.legal_actions() if isinstance(a, AttackAction)), None
    )
    if attack_action is None:
        pytest.skip("Seed produced no legal attack on first turn.")

    fi = env.topology.index_of(attack_action.from_territory)
    ti = env.topology.index_of(attack_action.to_territory)
    controller.on_territory_click(fi, button=1)
    assert controller.selected_from == fi
    controller.on_territory_click(ti, button=1)
    assert controller.selected_to == ti

    # Dice spinner exposed; bump to legal max.
    m = controller.widgets(env.current_state())
    assert m.dice_max >= 1
    for _ in range(3):
        controller.on_hud_field("dice_inc")
    expected_dice = min(3, s.armies[fi] - 1)
    assert controller.attack_dice == expected_dice

    controller.on_hud_button("attack")
    assert isinstance(agents[0]._pending, AttackAction)
    assert agents[0]._pending.from_territory == attack_action.from_territory
    assert agents[0]._pending.to_territory == attack_action.to_territory
    assert agents[0]._pending.dice == expected_dice
    # `to` cleared after a submitted attack.
    assert controller.selected_to is None
    assert controller.selected_from == fi


def test_attack_non_adjacent_click_ignored():
    env, agents, controller = _build(human_ids={0})
    _advance_to_attack(env, agents, controller, pid=0)
    own = _own_indices(env, 0)
    # Pick an own territory with >= 2 armies as `from`.
    s = env.current_state()
    from_idx = next((i for i in own if s.armies[i] >= 2), own[0])
    controller.on_territory_click(from_idx, button=1)
    from_terr = env.topology.territory_at(from_idx)
    # Pick an enemy that is NOT adjacent.
    non_adj = next(
        i for i in _enemy_indices(env, 0)
        if not env.topology.are_adjacent(
            from_terr, env.topology.territory_at(i)
        )
    )
    controller.on_territory_click(non_adj, button=1)
    assert controller.selected_to is None


def test_end_attack_button_submits_stop_attack():
    env, agents, controller = _build(human_ids={0})
    _advance_to_attack(env, agents, controller, pid=0)
    controller.on_hud_button("end_attack")
    assert isinstance(agents[0]._pending, StopAttackAction)


# --- FORTIFY flow -------------------------------------------------------


def _advance_to_fortify(env, agents, controller, pid: int) -> None:
    _advance_to_attack(env, agents, controller, pid)
    env.step(StopAttackAction())
    controller.on_turn_change(env.current_state())
    assert env.current_state().phase == Phase.FORTIFY
    assert env.current_state().current_player_index == pid


def test_skip_fortify_button_submits_skip():
    env, agents, controller = _build(human_ids={0})
    _advance_to_fortify(env, agents, controller, pid=0)
    controller.on_hud_button("skip_fortify")
    assert isinstance(agents[0]._pending, FortifyAction)
    assert agents[0]._pending.is_skip is True


def test_fortify_move_button_submits_legal_action():
    env, agents, controller = _build(human_ids={0})
    _advance_to_fortify(env, agents, controller, pid=0)
    s = env.current_state()
    # Find any legal non-skip fortify.
    legal_move = next(
        (a for a in env.legal_actions()
         if isinstance(a, FortifyAction) and not a.is_skip),
        None,
    )
    if legal_move is None:
        pytest.skip("Seed produced no legal fortify move on first turn.")

    fi = env.topology.index_of(legal_move.from_territory)
    ti = env.topology.index_of(legal_move.to_territory)
    controller.on_territory_click(fi, button=1)
    controller.on_territory_click(ti, button=1)
    assert controller.selected_from == fi
    assert controller.selected_to == ti
    # Bump count to 2 if possible, otherwise stay at 1.
    if s.armies[fi] - 1 >= 2:
        controller.on_hud_field("count_inc")
    controller.on_hud_button("move_armies")
    assert isinstance(agents[0]._pending, FortifyAction)
    assert agents[0]._pending.from_territory == legal_move.from_territory
    assert agents[0]._pending.to_territory == legal_move.to_territory
    assert agents[0]._pending.count >= 1


def test_fortify_disconnected_click_ignored():
    env, agents, controller = _build(human_ids={0})
    _advance_to_fortify(env, agents, controller, pid=0)
    s = env.current_state()
    own = [i for i in _own_indices(env, 0) if s.armies[i] >= 2]
    if not own:
        pytest.skip("No fortify-capable own territory.")
    src = own[0]
    controller.on_territory_click(src, button=1)
    # Find an enemy or a disconnected own territory to click. Easiest:
    # an enemy index -> ignored (must be own).
    enemy = _enemy_indices(env, 0)[0]
    controller.on_territory_click(enemy, button=1)
    assert controller.selected_to is None


# --- HUD button gating --------------------------------------------------


def test_hud_buttons_ignored_when_ai_turn():
    env, agents, controller = _build(human_ids=set())
    # No human seat -> every button is a no-op.
    controller.on_hud_button("place_armies")
    controller.on_hud_button("clear_all")
    controller.on_hud_button("attack")
    controller.on_hud_button("end_attack")
    controller.on_hud_button("move_armies")
    controller.on_hud_button("skip_fortify")
    # No HumanAgent to inspect; just confirm no exceptions and state intact.
    assert controller.pending_placements == {}


# --- OCCUPY flow --------------------------------------------------------


def _force_conquest_for_pid(env, pid: int) -> tuple[int, int]:
    """Drive env to Phase.OCCUPY for `pid`. Returns (from_idx, to_idx)."""
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
    env.step(ReinforcementAction(placements={attacker: s.reinforcement_budget}))
    from risk.game.phase import Phase as _Phase
    while s.phase is not _Phase.OCCUPY:
        env.step(AttackAction(from_territory=attacker, to_territory=target, dice=3))
    return ai, ti


def test_widgets_occupy_shape_and_bounds():
    env, agents, controller = _build(human_ids={0})
    _force_conquest_for_pid(env, pid=0)
    controller.on_turn_change(env.current_state())
    s = env.current_state()
    m = controller.widgets(s)
    assert m.header == "Your Turn (OCCUPY)"
    assert m.is_active is True
    assert m.count == 3  # min(dice, available) = min(3, 49)
    assert m.count_max == s.armies[s.pending_attack.from_index] - 1
    ids = {b.id for b in m.buttons}
    assert ids == {"occupy", "toggle_cards"}
    btn = m.buttons[0]
    assert btn.enabled is True
    assert btn.primary is True


def test_occupy_count_inc_dec_clamp_to_bounds():
    env, agents, controller = _build(human_ids={0})
    fi, _ti = _force_conquest_for_pid(env, pid=0)
    controller.on_turn_change(env.current_state())
    assert controller.occupy_count == 3
    controller.on_hud_field("count_dec")  # already at min, stays
    assert controller.occupy_count == 3
    controller.on_hud_field("count_inc")
    assert controller.occupy_count == 4
    # Cap at upper bound.
    s = env.current_state()
    hi = s.armies[fi] - 1
    for _ in range(hi + 5):
        controller.on_hud_field("count_inc")
    assert controller.occupy_count == hi


def test_occupy_button_submits_occupy_action():
    env, agents, controller = _build(human_ids={0})
    fi, ti = _force_conquest_for_pid(env, pid=0)
    controller.on_turn_change(env.current_state())
    controller.on_hud_field("count_inc")  # 3 -> 4
    controller.on_hud_field("count_inc")  # 4 -> 5
    controller.on_hud_button("occupy")
    assert isinstance(agents[0]._pending, OccupyAction)
    assert agents[0]._pending.count == 5


def test_after_occupy_selected_from_carries_to_conquered_square():
    env, agents, controller = _build(human_ids={0})
    fi, ti = _force_conquest_for_pid(env, pid=0)
    controller.on_turn_change(env.current_state())
    # Submit + manually advance env to mimic Game.tick.
    controller.on_hud_button("occupy")
    env.step(agents[0]._pending)
    agents[0].clear()
    controller.on_turn_change(env.current_state())
    from risk.game.phase import Phase as _Phase
    assert env.current_state().phase is _Phase.ATTACK
    assert controller.selected_from == ti
    assert controller.selected_to is None


# --- card screen / trade-in ----------------------------------------------


def _give_set(env, pid: int) -> None:
    """Replace pid's hand with one tradeable set (one of each symbol)."""
    s = env.current_state()
    s.hands[pid] = [
        Card(territory_id="Alaska", symbol="infantry"),
        Card(territory_id="Alberta", symbol="cavalry"),
        Card(territory_id="Ontario", symbol="artillery"),
    ]


def test_cards_button_present_on_human_turn():
    env, agents, controller = _build(human_ids={0})
    m = controller.widgets(env.current_state())
    assert any(b.id == "toggle_cards" for b in m.buttons)
    toggle = next(b for b in m.buttons if b.id == "toggle_cards")
    assert toggle.label.startswith("Cards (")


def test_toggle_cards_opens_card_screen():
    env, agents, controller = _build(human_ids={0})
    _give_set(env, pid=0)
    assert controller.widgets(env.current_state()).show_cards is False
    controller.on_hud_button("toggle_cards")
    m = controller.widgets(env.current_state())
    assert m.show_cards is True
    assert len(m.cards) == 3
    assert all(sel is False for _label, sel in m.cards)
    # Toggle button now offers to hide.
    assert any(b.label.startswith("Hide Cards") for b in m.buttons)


def test_select_three_valid_cards_enables_trade():
    env, agents, controller = _build(human_ids={0})
    _give_set(env, pid=0)
    controller.on_hud_button("toggle_cards")
    controller.on_hud_field("card_toggle", "0")
    controller.on_hud_field("card_toggle", "1")
    controller.on_hud_field("card_toggle", "2")
    m = controller.widgets(env.current_state())
    assert m.trade_value == 4  # first trade-in is worth 4
    trade_btn = next(b for b in m.buttons if b.id == "trade_cards")
    assert trade_btn.enabled is True


def test_selection_capped_at_three():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    s.hands[0] = [
        Card(territory_id="Alaska", symbol="infantry"),
        Card(territory_id="Alberta", symbol="cavalry"),
        Card(territory_id="Ontario", symbol="artillery"),
        Card(territory_id="Quebec", symbol="infantry"),
    ]
    controller.on_hud_button("toggle_cards")
    for i in range(4):
        controller.on_hud_field("card_toggle", str(i))
    assert len(controller.selected_cards) == 3


def test_trade_button_submits_trade_in_action():
    env, agents, controller = _build(human_ids={0})
    _give_set(env, pid=0)
    controller.on_hud_button("toggle_cards")
    for i in range(3):
        controller.on_hud_field("card_toggle", str(i))
    controller.on_hud_button("trade_cards")
    assert isinstance(agents[0]._pending, TradeInAction)
    assert agents[0]._pending.card_indices == (0, 1, 2)
    assert controller.selected_cards == set()


def test_trade_in_step_adds_budget_and_removes_cards():
    env, agents, controller = _build(human_ids={0})
    _give_set(env, pid=0)
    budget_before = env.current_state().reinforcement_budget
    env.step(TradeInAction(card_indices=(0, 1, 2)))
    s = env.current_state()
    assert s.reinforcement_budget == budget_before + 4
    assert len(s.hands[0]) == 0


def test_invalid_set_does_not_enable_trade():
    env, agents, controller = _build(human_ids={0})
    s = env.current_state()
    s.hands[0] = [
        Card(territory_id="Alaska", symbol="infantry"),
        Card(territory_id="Alberta", symbol="infantry"),
        Card(territory_id="Ontario", symbol="cavalry"),
    ]
    controller.on_hud_button("toggle_cards")
    for i in range(3):
        controller.on_hud_field("card_toggle", str(i))
    m = controller.widgets(env.current_state())
    assert m.trade_value is None
    trade_btn = next(b for b in m.buttons if b.id == "trade_cards")
    assert trade_btn.enabled is False


def test_cards_button_absent_when_not_human_turn():
    # Seat 0 is human, but it's seat 0's turn; check an AI seat yields nothing.
    env, agents, controller = _build(human_ids={1})
    # Controller owned by seat 1, but current player is seat 0 (AI).
    m = controller.widgets(env.current_state())
    assert m.is_active is False
    assert m.show_cards is False

