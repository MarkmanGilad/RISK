"""GameSettings: the immutable match configuration.

Built once by the setup screen and handed to `Environment.reset(settings)`.
Carries the player roster and the RNG seed (for deterministic setup,
deterministic dice rolls, and reproducible RL episodes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from risk.game.constants import MAX_PLAYERS, MIN_PLAYERS
from risk.game.player import Player


@dataclass(frozen=True, slots=True)
class GameSettings:
    players: tuple[Player, ...]
    seed: Optional[int] = None
    rule_toggles: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize: accept any Sequence[Player] and freeze to a tuple.
        if not isinstance(self.players, tuple):
            object.__setattr__(self, "players", tuple(self.players))
        if not isinstance(self.rule_toggles, Mapping):
            raise ValueError("rule_toggles must be a mapping")
        # Freeze rule_toggles to a read-only view.
        object.__setattr__(self, "rule_toggles", dict(self.rule_toggles))

        n = len(self.players)
        if not (MIN_PLAYERS <= n <= MAX_PLAYERS):
            raise ValueError(
                f"GameSettings requires {MIN_PLAYERS}..{MAX_PLAYERS} players, got {n}"
            )

        ids = [p.id for p in self.players]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate player ids in GameSettings: {ids}")
        if any(not isinstance(p, Player) for p in self.players):
            raise ValueError("All entries in players must be Player instances")

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be int or None, got {type(self.seed).__name__}")

    @property
    def player_count(self) -> int:
        return len(self.players)


__all__ = ["GameSettings"]
