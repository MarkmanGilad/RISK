"""Training utilities for Risk."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from risk.learning.self_play import StepHook, play_headless, play_rendered


def __getattr__(name: str) -> Any:
    if name in __all__:
        self_play = import_module("risk.learning.self_play")
        return getattr(self_play, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["play_headless", "play_rendered", "StepHook"]
