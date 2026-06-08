"""Pace AI ticks so each move is visible in `play` mode.

Pure timing logic, no pygame. `delay_ms <= 0` means "never wait".
"""
from __future__ import annotations


class AITickPacer:
    def __init__(self, delay_ms: int) -> None:
        self.delay_ms = delay_ms
        self._last_tick_ms = -delay_ms  # allow an immediate first AI tick

    def may_tick(self, now_ms: int, is_ai: bool) -> bool:
        if not is_ai or self.delay_ms <= 0:
            return True
        return (now_ms - self._last_tick_ms) >= self.delay_ms

    def note_tick(self, now_ms: int) -> None:
        self._last_tick_ms = now_ms


__all__ = ["AITickPacer"]
