"""Human-player input controller (decision builder).

Pygame-free core. It is owned by a single :class:`HumanAgent` and turns
decoded UI gestures into a complete :class:`Action`:

- `on_territory_click(index, button)` for map clicks (1=left, 3=right).
- `on_hud_button(button_id)` for side-panel primary / secondary buttons.
- `on_hud_field(field_id, value=None)` for side-panel +/- / clear /
  per-row controls.

Once a gesture assembles a legal action the controller forwards it to its
owning agent via `agent.submit(...)` (pre-validated against
`env.legal_actions()`). The renderer reads `widgets(state)` to draw the HUD
action panel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from risk.game.actions import (
    Action,
    AttackAction,
    FortifyAction,
    OccupyAction,
    ReinforcementAction,
    StopAttackAction,
    TradeInAction,
)
from risk.game.card import is_valid_set
from risk.game.constants import card_set_value
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.settings import GameSettings
from risk.game.state import State


# --- view-model rendered by HudPanel.render_action_panel -----------------


@dataclass(frozen=True)
class HudButton:
    id: str
    label: str
    enabled: bool
    primary: bool = False


@dataclass(frozen=True)
class HudActionPanelModel:
    """Pure-data description of the action panel for the current state."""

    header: str = ""
    info_lines: tuple[str, ...] = ()
    # Reinforce: per-territory placement rows (territory_name, count).
    placement_rows: tuple[tuple[str, int], ...] = ()
    # Attack / Fortify selection fields. None means not yet selected.
    from_label: Optional[str] = None
    to_label: Optional[str] = None
    # Numeric dropdowns. value is None when N/A.
    dice: Optional[int] = None
    dice_max: int = 0
    count: Optional[int] = None
    count_max: int = 0
    # Card screen (toggled by the "Cards" button on the current human's turn).
    show_cards: bool = False
    # Per-card rows: (label, selected).
    cards: tuple[tuple[str, bool], ...] = ()
    # Value of the currently-selected set, or None if the selection is not a
    # tradeable set (or trading is not allowed in this phase).
    trade_value: Optional[int] = None
    # Footer buttons.
    buttons: tuple[HudButton, ...] = ()

    @property
    def is_active(self) -> bool:
        return bool(self.header)


# --- controller ----------------------------------------------------------


class HumanInputController:
    """Per-player decision builder. Belongs to one :class:`HumanAgent`."""

    def __init__(self, env: Environment, agent: Any, settings: GameSettings) -> None:
        self.env = env
        self.topology = env.topology
        self.agent = agent
        self.settings = settings
        self._reset_ephemeral()
        self._last_turn_key: Optional[tuple[int, int]] = None

    # --- lifecycle -------------------------------------------------------

    def _reset_ephemeral(self) -> None:
        self.pending_placements: dict[str, int] = {}
        self.selected_from: Optional[int] = None
        self.selected_to: Optional[int] = None
        self.attack_dice: int = 1
        self.fortify_count: int = 1
        self.occupy_count: int = 1
        self.show_cards: bool = False
        self.selected_cards: set[int] = set()

    def on_turn_change(self, state: State) -> None:
        """Update ephemeral selections in response to phase / player changes.

        Mostly a hard reset, but the OCCUPY sub-phase is treated specially so
        the player can fluidly chain attacks from a freshly conquered region.
        """
        key = (state.current_player_index, int(state.phase))
        prev = self._last_turn_key
        self._last_turn_key = key
        if prev == key:
            return
        same_player = prev is not None and prev[0] == state.current_player_index
        prev_phase = prev[1] if prev is not None else None

        if (
            state.phase == Phase.OCCUPY
            and state.pending_attack is not None
        ):
            pa = state.pending_attack
            self.pending_placements = {}
            self.selected_from = pa.from_index
            self.selected_to = pa.to_index
            self.attack_dice = 1
            self.fortify_count = 1
            self.show_cards = False
            self.selected_cards = set()
            lo, _hi = self._occupy_bounds(state)
            self.occupy_count = lo
            return

        if (
            state.phase == Phase.ATTACK
            and prev_phase == int(Phase.OCCUPY)
            and same_player
            and self.selected_to is not None
        ):
            # Hand the conquered square back as the next attacker.
            new_from = self.selected_to
            self._reset_ephemeral()
            self.selected_from = new_from
            # Default to the maximum number of dice (3 when possible).
            self.attack_dice = max(1, min(3, state.armies[new_from] - 1))
            return

        self._reset_ephemeral()

    # --- queries ---------------------------------------------------------

    def is_human_turn(self, state: State) -> bool:
        if state.phase == Phase.GAME_OVER:
            return False
        pid = state.current_player_index
        if pid in state.eliminated:
            return False
        if pid != self.agent.player_id:
            return False
        return self.settings.players[pid].agent_kind == "human"

    # --- map clicks ------------------------------------------------------

    def on_territory_click(self, territory_index: int, button: int) -> None:
        state = self.env.current_state()
        if not self.is_human_turn(state):
            return
        if not (0 <= territory_index < len(self.topology.territories)):
            return
        phase = state.phase
        if phase == Phase.REINFORCE:
            self._reinforce_click(state, territory_index, button)
        elif phase == Phase.ATTACK:
            self._attack_click(state, territory_index, button)
        elif phase == Phase.FORTIFY:
            self._fortify_click(state, territory_index, button)

    def _reinforce_click(self, state: State, idx: int, button: int) -> None:
        pid = state.current_player_index
        terr = self.topology.territory_at(idx)
        if state.owners[idx] != pid:
            return
        if button == 1:  # left -> +1
            placed = sum(self.pending_placements.values())
            if placed >= state.reinforcement_budget:
                return
            self.pending_placements[terr] = self.pending_placements.get(terr, 0) + 1
        elif button == 3:  # right -> -1
            self._dec_placement(terr)

    def _dec_placement(self, terr: str) -> None:
        if terr not in self.pending_placements:
            return
        self.pending_placements[terr] -= 1
        if self.pending_placements[terr] <= 0:
            del self.pending_placements[terr]

    def _attack_click(self, state: State, idx: int, button: int) -> None:
        if button != 1:
            return
        pid = state.current_player_index
        if state.owners[idx] == pid:
            if state.armies[idx] >= 2:
                self.selected_from = idx
                self.selected_to = None
                limit = min(3, state.armies[idx] - 1)
                # Default to the maximum number of dice (3 when possible).
                self.attack_dice = max(1, limit)
            return
        # Click on an enemy / neutral territory: requires a chosen "from"
        # and adjacency.
        if self.selected_from is None:
            return
        from_terr = self.topology.territory_at(self.selected_from)
        if not self.topology.are_adjacent(from_terr, self.topology.territory_at(idx)):
            return
        self.selected_to = idx

    def _fortify_click(self, state: State, idx: int, button: int) -> None:
        if button != 1:
            return
        pid = state.current_player_index
        if state.owners[idx] != pid:
            return
        # First click (or click on already-selected `from`) sets `from`.
        if self.selected_from is None or idx == self.selected_from:
            if state.armies[idx] >= 2:
                self.selected_from = idx
                self.selected_to = None
                self.fortify_count = 1
            return
        # Second click selects `to` if connected through owned territories.
        if self._connected_through_owned(state, pid, self.selected_from, idx):
            self.selected_to = idx
            self.fortify_count = 1

    def _connected_through_owned(
        self, state: State, pid: int, src_idx: int, dst_idx: int
    ) -> bool:
        if src_idx == dst_idx:
            return False
        seen = {src_idx}
        stack = [src_idx]
        while stack:
            cur = stack.pop()
            if cur == dst_idx:
                return True
            for nb in self.topology.neighbors(self.topology.territory_at(cur)):
                ni = self.topology.index_of(nb)
                if ni in seen or state.owners[ni] != pid:
                    continue
                seen.add(ni)
                stack.append(ni)
        return False

    # --- HUD buttons / fields -------------------------------------------

    def on_hud_button(self, button_id: str) -> None:
        state = self.env.current_state()
        if not self.is_human_turn(state):
            return

        if button_id == "place_armies":
            self._submit_reinforce(state)
        elif button_id == "clear_all":
            self._reset_ephemeral()
        elif button_id == "toggle_cards":
            self.show_cards = not self.show_cards
            if not self.show_cards:
                self.selected_cards = set()
        elif button_id == "trade_cards":
            self._submit_trade(state)
        elif button_id == "attack":
            self._submit_attack(state)
        elif button_id == "end_attack":
            self._submit_stop_attack(state)
        elif button_id == "occupy":
            self._submit_occupy(state)
        elif button_id == "move_armies":
            self._submit_fortify(state)
        elif button_id == "skip_fortify":
            self._submit_skip_fortify(state)

    def on_hud_field(self, field_id: str, value: Any = None) -> None:
        state = self.env.current_state()
        if not self.is_human_turn(state):
            return
        if field_id == "placements_inc" and isinstance(value, str):
            idx = self.topology.index_of(value)
            self._reinforce_click(state, idx, button=1)
        elif field_id == "card_toggle" and value is not None:
            self._toggle_card(state, int(value))
        elif field_id == "placements_dec" and isinstance(value, str):
            self._dec_placement(value)
        elif field_id == "placements_clear" and isinstance(value, str):
            self.pending_placements.pop(value, None)
        elif field_id == "attack_from_clear":
            self.selected_from = None
            self.selected_to = None
        elif field_id == "attack_to_clear":
            self.selected_to = None
        elif field_id == "fortify_from_clear":
            self.selected_from = None
            self.selected_to = None
        elif field_id == "fortify_to_clear":
            self.selected_to = None
        elif field_id == "dice_inc":
            if self.selected_from is None:
                return
            limit = min(3, state.armies[self.selected_from] - 1)
            self.attack_dice = max(1, min(self.attack_dice + 1, limit))
        elif field_id == "dice_dec":
            self.attack_dice = max(1, self.attack_dice - 1)
        elif field_id == "count_inc":
            if state.phase == Phase.OCCUPY:
                lo, hi = self._occupy_bounds(state)
                self.occupy_count = max(lo, min(self.occupy_count + 1, hi))
                return
            if self.selected_from is None:
                return
            limit = state.armies[self.selected_from] - 1
            if limit < 1:
                return
            self.fortify_count = max(1, min(self.fortify_count + 1, limit))
        elif field_id == "count_dec":
            if state.phase == Phase.OCCUPY:
                lo, _hi = self._occupy_bounds(state)
                self.occupy_count = max(lo, self.occupy_count - 1)
                return
            self.fortify_count = max(1, self.fortify_count - 1)

    # --- submission helpers ---------------------------------------------

    def _is_legal(self, action: Action) -> bool:
        target = action.to_dict()
        for a in self.env.legal_actions():
            if a.to_dict() == target:
                return True
        return False

    def _submit_reinforce(self, state: State) -> None:
        if state.phase != Phase.REINFORCE:
            return
        placed = sum(self.pending_placements.values())
        if placed == 0 or placed != state.reinforcement_budget:
            return
        action = ReinforcementAction(placements=dict(self.pending_placements))
        if self._is_legal(action):
            self.agent.submit(action)
            self.pending_placements = {}

    def _toggle_card(self, state: State, idx: int) -> None:
        hand = state.hands[self.agent.player_id]
        if not (0 <= idx < len(hand)):
            return
        if idx in self.selected_cards:
            self.selected_cards.discard(idx)
        elif len(self.selected_cards) < 3:
            self.selected_cards.add(idx)

    def _submit_trade(self, state: State) -> None:
        if state.phase != Phase.REINFORCE:
            return
        if len(self.selected_cards) != 3:
            return
        hand = state.hands[self.agent.player_id]
        idxs = tuple(sorted(self.selected_cards))
        if any(i >= len(hand) for i in idxs):
            return
        action = TradeInAction(card_indices=idxs)  # type: ignore[arg-type]
        if self._is_legal(action):
            self.agent.submit(action)
            self.selected_cards = set()

    def _submit_attack(self, state: State) -> None:
        if state.phase != Phase.ATTACK:
            return
        if self.selected_from is None or self.selected_to is None:
            return
        limit = min(3, state.armies[self.selected_from] - 1)
        if limit < 1:
            return
        dice = max(1, min(self.attack_dice, limit))
        action = AttackAction(
            from_territory=self.topology.territory_at(self.selected_from),
            to_territory=self.topology.territory_at(self.selected_to),
            dice=dice,
        )
        if self._is_legal(action):
            self.agent.submit(action)
            # Defender square will resolve next tick. Keep `from`, drop `to`
            # so the player re-picks deliberately.
            self.selected_to = None

    def _submit_stop_attack(self, state: State) -> None:
        if state.phase != Phase.ATTACK:
            return
        action = StopAttackAction()
        if self._is_legal(action):
            self.agent.submit(action)
            self.selected_from = None
            self.selected_to = None

    def _occupy_bounds(self, state: State) -> tuple[int, int]:
        """Mirror `Environment._occupy_bounds` for view-model + validation."""
        pa = state.pending_attack
        if pa is None:
            return (1, 1)
        available = state.armies[pa.from_index] - 1
        if available < 1:
            return (1, 1)
        lo = max(1, min(pa.attacker_dice, available))
        hi = available
        if lo > hi:
            lo = hi
        return (lo, hi)

    def _submit_occupy(self, state: State) -> None:
        if state.phase != Phase.OCCUPY or state.pending_attack is None:
            return
        lo, hi = self._occupy_bounds(state)
        count = max(lo, min(self.occupy_count, hi))
        action = OccupyAction(count=count)
        if self._is_legal(action):
            self.agent.submit(action)
            # Leave `selected_to` set to the conquered index so the next
            # ATTACK transition can promote it to `selected_from`.

    def _submit_fortify(self, state: State) -> None:
        if state.phase != Phase.FORTIFY:
            return
        if self.selected_from is None or self.selected_to is None:
            return
        limit = state.armies[self.selected_from] - 1
        if limit < 1:
            return
        count = max(1, min(self.fortify_count, limit))
        action = FortifyAction(
            from_territory=self.topology.territory_at(self.selected_from),
            to_territory=self.topology.territory_at(self.selected_to),
            count=count,
        )
        if self._is_legal(action):
            self.agent.submit(action)

    def _submit_skip_fortify(self, state: State) -> None:
        if state.phase != Phase.FORTIFY:
            return
        action = FortifyAction(from_territory=None, to_territory=None, count=0)
        if self._is_legal(action):
            self.agent.submit(action)

    # --- view-model -----------------------------------------------------

    def widgets(self, state: State) -> HudActionPanelModel:
        if not self.is_human_turn(state):
            return HudActionPanelModel()
        model = self._phase_widgets(state)
        return self._with_cards(state, model)

    def _card_label(self, card) -> str:
        if card.is_wild:
            return "WILD"
        return f"{card.symbol[:3].upper()} {card.territory_id[:24]}"

    def _with_cards(self, state: State, model: HudActionPanelModel) -> HudActionPanelModel:
        """Append the 'Cards' toggle (and, when open, the card screen)."""
        if not model.is_active:
            return model
        hand = state.hands[self.agent.player_id]
        can_trade = state.phase == Phase.REINFORCE
        toggle_label = (
            f"Hide Cards ({len(hand)})" if self.show_cards else f"Cards ({len(hand)})"
        )
        buttons = model.buttons + (
            HudButton(id="toggle_cards", label=toggle_label, enabled=True),
        )
        if not self.show_cards:
            return replace(model, buttons=buttons)

        # Drop selections that no longer exist (e.g. after a trade).
        self.selected_cards = {i for i in self.selected_cards if i < len(hand)}
        rows = tuple(
            (self._card_label(c), i in self.selected_cards)
            for i, c in enumerate(hand)
        )
        trade_value: Optional[int] = None
        if can_trade and len(self.selected_cards) == 3:
            trio = tuple(hand[i] for i in sorted(self.selected_cards))
            if is_valid_set(trio):
                trade_value = card_set_value(state.cards_traded_in_count)
        if can_trade:
            label = f"Trade Set (+{trade_value})" if trade_value else "Trade Set"
            buttons = buttons + (
                HudButton(
                    id="trade_cards",
                    label=label,
                    enabled=trade_value is not None,
                    primary=True,
                ),
            )
        return replace(
            model,
            show_cards=True,
            cards=rows,
            trade_value=trade_value,
            buttons=buttons,
        )

    def _phase_widgets(self, state: State) -> HudActionPanelModel:
        if not self.is_human_turn(state):
            return HudActionPanelModel()

        if state.phase == Phase.REINFORCE:
            placed = sum(self.pending_placements.values())
            budget = state.reinforcement_budget
            rows = tuple(self.pending_placements.items())
            return HudActionPanelModel(
                header="Your Turn (REINFORCE)",
                info_lines=(f"Armies to place: {placed} / {budget}",),
                placement_rows=rows,
                buttons=(
                    HudButton(
                        id="place_armies",
                        label="Place Armies",
                        enabled=(placed == budget and budget > 0),
                        primary=True,
                    ),
                    HudButton(
                        id="clear_all",
                        label="Clear All",
                        enabled=bool(self.pending_placements),
                    ),
                ),
            )

        if state.phase == Phase.ATTACK:
            from_label = self._label_for(state, self.selected_from)
            to_label = self._label_for(state, self.selected_to)
            dice_max = 0
            if self.selected_from is not None:
                dice_max = min(3, max(0, state.armies[self.selected_from] - 1))
            attack_enabled = (
                self.selected_from is not None
                and self.selected_to is not None
                and dice_max >= 1
            )
            return HudActionPanelModel(
                header="Your Turn (ATTACK)",
                from_label=from_label,
                to_label=to_label,
                dice=self.attack_dice if dice_max >= 1 else None,
                dice_max=dice_max,
                buttons=(
                    HudButton(
                        id="attack", label="Attack!", enabled=attack_enabled, primary=True
                    ),
                    HudButton(
                        id="end_attack", label="End Attack Phase", enabled=True
                    ),
                ),
            )

        if state.phase == Phase.OCCUPY and state.pending_attack is not None:
            pa = state.pending_attack
            lo, hi = self._occupy_bounds(state)
            from_label = self._label_for(state, pa.from_index)
            to_label = self._label_for(state, pa.to_index)
            return HudActionPanelModel(
                header="Your Turn (OCCUPY)",
                info_lines=(
                    f"Move between {lo} and {hi} armies into "
                    f"{self.topology.territory_at(pa.to_index)}.",
                ),
                from_label=from_label,
                to_label=to_label,
                count=self.occupy_count,
                count_max=hi,
                buttons=(
                    HudButton(
                        id="occupy",
                        label="Move Armies",
                        enabled=lo <= self.occupy_count <= hi,
                        primary=True,
                    ),
                ),
            )

        if state.phase == Phase.FORTIFY:
            from_label = self._label_for(state, self.selected_from)
            to_label = self._label_for(state, self.selected_to)
            count_max = 0
            if self.selected_from is not None:
                count_max = max(0, state.armies[self.selected_from] - 1)
            move_enabled = (
                self.selected_from is not None
                and self.selected_to is not None
                and count_max >= 1
            )
            info_lines: tuple[str, ...] = ()
            if self.selected_from is not None and self.selected_to is not None:
                info_lines = (
                    f"Path: {self.topology.territory_at(self.selected_from)}"
                    f" -> {self.topology.territory_at(self.selected_to)}",
                )
            return HudActionPanelModel(
                header="Your Turn (FORTIFY)",
                info_lines=info_lines,
                from_label=from_label,
                to_label=to_label,
                count=self.fortify_count if count_max >= 1 else None,
                count_max=count_max,
                buttons=(
                    HudButton(
                        id="move_armies",
                        label="Move Armies",
                        enabled=move_enabled,
                        primary=True,
                    ),
                    HudButton(
                        id="skip_fortify", label="Skip Fortify", enabled=True
                    ),
                ),
            )

        return HudActionPanelModel()

    def _label_for(self, state: State, idx: Optional[int]) -> Optional[str]:
        if idx is None:
            return None
        return f"{self.topology.territory_at(idx)} ({state.armies[idx]} armies)"


__all__ = ["HudActionPanelModel", "HudButton", "HumanInputController"]
