"""Player value object.

A `Player` is an immutable seat description: id, display name, color, and
agent kind. Dynamic per-player state (elimination flag, card hand,
ownership counts) lives in `State`, not here, so `Player` can be
shared safely without copy-on-write concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


ALLOWED_AGENT_KINDS: Final[frozenset[str]] = frozenset({"human", "ai"})


@dataclass(frozen=True, slots=True)
class Player:
    id: int
    name: str
    color: tuple[int, int, int]
    agent_kind: str

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError(f"Player id must be non-negative, got {self.id}")
        if not self.name:
            raise ValueError("Player name must be a non-empty string")
        if self.agent_kind not in ALLOWED_AGENT_KINDS:
            raise ValueError(
                f"Unknown agent_kind {self.agent_kind!r}; "
                f"expected one of {sorted(ALLOWED_AGENT_KINDS)}"
            )
        if (
            not isinstance(self.color, tuple)
            or len(self.color) != 3
            or not all(isinstance(c, int) and 0 <= c <= 255 for c in self.color)
        ):
            raise ValueError(
                f"Player color must be a 3-tuple of ints in 0..255, got {self.color!r}"
            )


__all__ = ["Player", "ALLOWED_AGENT_KINDS"]
