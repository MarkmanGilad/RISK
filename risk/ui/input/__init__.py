"""pygame-free input layer: builds actions from clicks/keys, hit-testing, setup state."""
from .hit_test import TerritoryHitTester
from .human_input import HudActionPanelModel, HumanInputController
from .init_screen import DEFAULT_COLORS, InitScreenState

__all__ = [
    "TerritoryHitTester",
    "HudActionPanelModel",
    "HumanInputController",
    "DEFAULT_COLORS",
    "InitScreenState",
]
