"""Typed action objects.

Actions describe *what* a player wants to do; they do not check
legality vs. ownership / armies / adjacency. That validation belongs
to `Environment` (Phase 4). Construction only enforces structural
validity (field types, value ranges, known territory ids).

Round-trips through `to_dict` / `ActionCodec.from_dict` for replay, logging,
and future RL trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from risk.constants import MAX_ATTACK_DICE, MAX_DEFEND_DICE
from risk.game.phase import Phase


# --- base -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Action:
    """Abstract base for all player actions. Subclasses set `phase`."""

    phase: ClassVar[Phase]

    @staticmethod
    def _check_territory(name: str, topology) -> None:
        if topology is None:
            return
        if name not in topology.territories:
            raise ValueError(f"Unknown territory: {name!r}")

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


# --- reinforcement --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReinforcementAction(Action):
    """Place armies on owned territories. `placements: {territory_id: count}`."""

    placements: dict[str, int]
    phase: ClassVar[Phase] = Phase.REINFORCE

    def __post_init__(self) -> None:
        if not isinstance(self.placements, dict):
            raise ValueError("placements must be a dict[str, int]")
        if not self.placements:
            raise ValueError("placements must be non-empty")
        for terr, count in self.placements.items():
            if not isinstance(terr, str) or not terr:
                raise ValueError(f"Bad territory key: {terr!r}")
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError(f"placements[{terr}] must be int, got {type(count).__name__}")
            if count < 0:
                raise ValueError(f"placements[{terr}] is negative: {count}")
        if self.total == 0:
            raise ValueError("ReinforcementAction with all-zero placements is meaningless")

    @property
    def total(self) -> int:
        return sum(self.placements.values())

    def validate_against(self, topology, budget: Optional[int] = None) -> None:
        for terr in self.placements:
            self._check_territory(terr, topology)
        if budget is not None and self.total != budget:
            raise ValueError(
                f"ReinforcementAction total {self.total} does not match budget {budget}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "reinforce", "placements": dict(self.placements)}


# --- attack ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttackAction(Action):
    from_territory: str
    to_territory: str
    dice: int
    phase: ClassVar[Phase] = Phase.ATTACK

    def __post_init__(self) -> None:
        if not isinstance(self.from_territory, str) or not self.from_territory:
            raise ValueError("from_territory must be a non-empty string")
        if not isinstance(self.to_territory, str) or not self.to_territory:
            raise ValueError("to_territory must be a non-empty string")
        if self.from_territory == self.to_territory:
            raise ValueError("AttackAction cannot target the same territory")
        if not isinstance(self.dice, int) or isinstance(self.dice, bool):
            raise ValueError("dice must be int")
        if not (1 <= self.dice <= MAX_ATTACK_DICE):
            raise ValueError(f"dice must be in 1..{MAX_ATTACK_DICE}, got {self.dice}")

    def validate_against(self, topology) -> None:
        self._check_territory(self.from_territory, topology)
        self._check_territory(self.to_territory, topology)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "attack",
            "from_territory": self.from_territory,
            "to_territory": self.to_territory,
            "dice": self.dice,
        }


# --- stop attack ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StopAttackAction(Action):
    """Signal end-of-attack phase. Transitions to FORTIFY."""

    phase: ClassVar[Phase] = Phase.ATTACK

    def to_dict(self) -> dict[str, Any]:
        return {"type": "stop_attack"}


# --- trade in cards -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeInAction(Action):
    """Cash in three cards from the current player's hand during REINFORCE.

    `card_indices` are positions into the player's current hand. The
    environment validates that they form a valid set and adds the set value
    to the reinforcement budget.
    """

    card_indices: tuple[int, int, int]
    phase: ClassVar[Phase] = Phase.REINFORCE

    def __post_init__(self) -> None:
        if not isinstance(self.card_indices, tuple) or len(self.card_indices) != 3:
            raise ValueError("card_indices must be a tuple of 3 ints")
        for i in self.card_indices:
            if not isinstance(i, int) or isinstance(i, bool):
                raise ValueError("card index must be int")
            if i < 0:
                raise ValueError(f"card index must be >= 0, got {i}")
        if len(set(self.card_indices)) != 3:
            raise ValueError("card_indices must be distinct")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "trade_in", "card_indices": list(self.card_indices)}


# --- occupy ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OccupyAction(Action):
    """Move armies from the attacker into the just-conquered territory.

    Issued exclusively during `Phase.OCCUPY`, which the environment enters
    immediately after a successful conquest. `count` must satisfy
    ``attacker_dice <= count <= armies_in_from - 1``.
    """

    count: int
    phase: ClassVar[Phase] = Phase.OCCUPY

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise ValueError("count must be int")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "occupy", "count": self.count}


# --- fortify --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FortifyAction(Action):
    """Move armies between owned, connected territories. Pass `count=0` to skip."""

    from_territory: Optional[str]
    to_territory: Optional[str]
    count: int
    phase: ClassVar[Phase] = Phase.FORTIFY

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise ValueError("count must be int")
        if self.count < 0:
            raise ValueError(f"count must be >= 0, got {self.count}")
        if self.count == 0:
            # Skip fortify: territories may be None.
            return
        if not isinstance(self.from_territory, str) or not self.from_territory:
            raise ValueError("from_territory required when count > 0")
        if not isinstance(self.to_territory, str) or not self.to_territory:
            raise ValueError("to_territory required when count > 0")
        if self.from_territory == self.to_territory:
            raise ValueError("FortifyAction cannot move within the same territory")

    @property
    def is_skip(self) -> bool:
        return self.count == 0

    def validate_against(self, topology) -> None:
        if self.is_skip:
            return
        self._check_territory(self.from_territory, topology)
        self._check_territory(self.to_territory, topology)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "fortify",
            "from_territory": self.from_territory,
            "to_territory": self.to_territory,
            "count": self.count,
        }


# --- round-trip -----------------------------------------------------------


class ActionCodec:
    """Round-trips an `Action` through its `to_dict()` representation."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Action:
        t = data.get("type")
        if t == "reinforce":
            return ReinforcementAction(placements=dict(data["placements"]))
        if t == "attack":
            return AttackAction(
                from_territory=data["from_territory"],
                to_territory=data["to_territory"],
                dice=int(data["dice"]),
            )
        if t == "stop_attack":
            return StopAttackAction()
        if t == "trade_in":
            return TradeInAction(card_indices=tuple(int(i) for i in data["card_indices"]))
        if t == "occupy":
            return OccupyAction(count=int(data["count"]))
        if t == "fortify":
            return FortifyAction(
                from_territory=data.get("from_territory"),
                to_territory=data.get("to_territory"),
                count=int(data["count"]),
            )
        raise ValueError(f"Unknown action type: {t!r}")


__all__ = [
    "Action",
    "ActionCodec",
    "ReinforcementAction",
    "AttackAction",
    "StopAttackAction",
    "TradeInAction",
    "OccupyAction",
    "FortifyAction",
]
