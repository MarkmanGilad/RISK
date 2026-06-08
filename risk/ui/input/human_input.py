"""Backward-compatible re-export.

The human input controller and its HUD view-model now live in the agents
layer (`risk.agents.human_input`) because they are owned by `HumanAgent`.
This module keeps the historical `risk.ui.input.human_input` import path
working for the renderer and tests.
"""
from __future__ import annotations

from risk.agents.human_input import (
    HudActionPanelModel,
    HudButton,
    HumanInputController,
)

__all__ = ["HudActionPanelModel", "HudButton", "HumanInputController"]
