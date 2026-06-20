"""Human agent: turns pygame input events into actions.

The loop calls `agent(events, state)` every frame. The agent owns a
`HumanInputController` (the decision state machine) and a reference to the
`view` (for pixel -> territory hit-testing and the HUD click regions). It
decodes mouse events into controller gestures and returns the assembled
`Action` once a decision is complete — `None` otherwise.

`submit(action)` remains as a programmatic entry point (used by tests and
any non-pygame driver).
"""
from __future__ import annotations

from typing import Optional, Sequence

import pygame

from risk.agents.base_agent import BaseAgent
from risk.agents.human_input import HudActionPanelModel, HumanInputController
from risk.game.actions import Action, FortifyAction, ReinforcementAction
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.state import State


class HumanAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        env: Optional[Environment] = None,
        settings=None,
        view=None,
    ) -> None:
        super().__init__(player_id)
        self.env = env
        self.settings = settings
        self.view = view
        self._pending: Optional[Action] = None
        self.controller: Optional[HumanInputController] = None
        self._ensure_controller()

    # --- wiring ----------------------------------------------------------

    def set_view(self, view) -> None:
        self.view = view

    def _ensure_controller(self) -> Optional[HumanInputController]:
        if self.controller is None and self.env is not None:
            self.controller = HumanInputController(self.env, self, self.settings)
        return self.controller

    # --- programmatic entry (tests / non-pygame drivers) ----------------

    def submit(self, action: Action) -> None:
        """Record a fully-assembled decision to be returned by `act`."""
        self._pending = action

    def clear(self) -> None:
        self._pending = None

    # --- agent contract --------------------------------------------------

    def act(self, events: Sequence[object], state: State) -> Optional[Action]:
        controller = self._ensure_controller()
        if controller is not None:
            controller.on_turn_change(state)
            if self.view is not None:
                self._consume_events(events)
        if self._pending is not None:
            chosen, self._pending = self._pending, None
            if self.env is not None and not self._is_legal(chosen):
                return None
            return chosen
        return None

    def widgets(self, state: State) -> HudActionPanelModel:
        controller = self._ensure_controller()
        if controller is None:
            return HudActionPanelModel()
        controller.on_turn_change(state)
        return controller.widgets(state)

    # --- event decoding --------------------------------------------------

    def _consume_events(self, events: Sequence[object]) -> None:
        regions = getattr(self.view, "hud_regions", None) or []
        for event in events:
            if getattr(event, "type", None) == pygame.MOUSEBUTTONDOWN:
                self._dispatch_click(event, regions)

    def _dispatch_click(self, event, regions) -> None:
        for rect, region_id in regions:
            if rect.collidepoint(event.pos):
                self._dispatch_hud(region_id)
                return
        terr_name = self.view.territory_at(event.pos)
        if terr_name is None:
            return
        try:
            idx = self.env.topology.index_of(terr_name)
        except (KeyError, ValueError):
            return
        assert self.controller is not None
        self.controller.on_territory_click(idx, event.button)

    def _dispatch_hud(self, region_id: str) -> None:
        assert self.controller is not None
        if region_id.startswith("button:"):
            self.controller.on_hud_button(region_id[len("button:") :])
            return
        if region_id.startswith("field:"):
            rest = region_id[len("field:") :]
            if ":" in rest:
                fid, payload = rest.split(":", 1)
                self.controller.on_hud_field(fid, payload)
            else:
                self.controller.on_hud_field(rest)

    # --- helpers ---------------------------------------------------------

    def _is_legal(self, action: Action) -> bool:
        # Reinforcement is a continuous (parameterized) space: `legal_actions()`
        # only enumerates the canonical one-territory form, so a mixed
        # placement won't appear there. Validate it structurally instead.
        if isinstance(action, ReinforcementAction):
            return self._is_legal_reinforce(action)
        # FortifyAction count is user-chosen (1..armies-1); legal_actions() only
        # yields count=armies-1 so an exact match would always fail. Bypass.
        if isinstance(action, FortifyAction):
            return True
        target = action.to_dict()
        return any(a.to_dict() == target for a in self.env.legal_actions())

    def _is_legal_reinforce(self, action: ReinforcementAction) -> bool:
        state = self.env.current_state()
        if state.phase is not Phase.REINFORCE_PLACE:
            return False
        if action.total > state.reinforcement_budget:
            return False
        pid = state.current_player_index
        topology = self.env.topology
        for terr in action.placements:
            if terr not in topology.territories:
                return False
            if state.owners[topology.index_of(terr)] != pid:
                return False
        return True


__all__ = ["HumanAgent"]
