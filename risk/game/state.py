"""Dynamic game state for Risk.

`State` is the mutable, per-turn data of a match. It is intentionally
decoupled from `BoardTopology` (which is the static graph) so that the
same topology can back many concurrent states (RL rollouts, planning,
undo stacks, ...).

Arrays are aligned with `topology.territories`: `owners[i]` and
`armies[i]` describe `topology.territory_at(i)`.

The class is pygame-free and torch-free. The future GNN adapter will
consume `(topology, state)` via `to_features` and emit tensors.
"""
from __future__ import annotations

import copy as _copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from risk.game.card import Card
from risk.game.phase import Phase


@dataclass
class PendingAttack:
    """Bookkeeping for an in-progress attack (between dice resolution steps)."""

    from_index: int
    to_index: int
    attacker_dice: int


@dataclass
class State:
    """Mutable per-match dynamic state."""

    owners: list[Optional[int]]
    armies: list[int]
    current_player_index: int
    phase: Phase
    reinforcement_budget: int
    hands: list[list[Card]]
    eliminated: set[int]
    cards_traded_in_count: int = 0
    pending_attack: Optional[PendingAttack] = None
    # Whether the current player has drawn a conquest card this turn —
    # cards only draw on a turn's *first* conquest, so this also gates
    # later conquests' card draws. Reset each turn in
    # `Environment._begin_turn_for`. Exposed on `State` (rather than kept
    # as a private `Environment` attribute) so `RewardCalculator` can read
    # it from a `before`/`after` snapshot without `Environment` needing to
    # compute any reward itself.
    conquered_this_turn: bool = False

    # --- constructors ------------------------------------------------------

    @classmethod
    def initial(cls, topology, player_count: int) -> "State":
        """Build an empty `State` sized to `topology`, with no territories owned."""
        n = len(topology)
        return cls(
            owners=[None] * n,
            armies=[0] * n,
            current_player_index=0,
            phase=Phase.SETUP,
            reinforcement_budget=0,
            hands=[[] for _ in range(player_count)],
            eliminated=set(),
            cards_traded_in_count=0,
            pending_attack=None,
            conquered_this_turn=False,
        )

    # --- copy --------------------------------------------------------------

    def copy(self) -> "State":
        """Return an independent copy safe for rollouts."""
        return State(
            owners=list(self.owners),
            armies=list(self.armies),
            current_player_index=self.current_player_index,
            phase=self.phase,
            reinforcement_budget=self.reinforcement_budget,
            hands=[list(h) for h in self.hands],  # Card is frozen, shallow inner copy is fine
            eliminated=set(self.eliminated),
            cards_traded_in_count=self.cards_traded_in_count,
            pending_attack=_copy.copy(self.pending_attack) if self.pending_attack else None,
            conquered_this_turn=self.conquered_this_turn,
        )

    def snapshot(self) -> "State":
        """Alias for `copy()` — preferred name when storing for replay/undo."""
        return self.copy()

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "owners": list(self.owners),
            "armies": list(self.armies),
            "current_player_index": self.current_player_index,
            "phase": int(self.phase),
            "reinforcement_budget": self.reinforcement_budget,
            "hands": [
                [{"territory_id": c.territory_id, "symbol": c.symbol} for c in hand]
                for hand in self.hands
            ],
            "eliminated": sorted(self.eliminated),
            "cards_traded_in_count": self.cards_traded_in_count,
            "conquered_this_turn": self.conquered_this_turn,
            "pending_attack": (
                {
                    "from_index": self.pending_attack.from_index,
                    "to_index": self.pending_attack.to_index,
                    "attacker_dice": self.pending_attack.attacker_dice,
                }
                if self.pending_attack is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        pa = data.get("pending_attack")
        return cls(
            owners=list(data["owners"]),
            armies=list(data["armies"]),
            current_player_index=int(data["current_player_index"]),
            phase=Phase(int(data["phase"])),
            reinforcement_budget=int(data["reinforcement_budget"]),
            hands=[
                [Card(territory_id=c["territory_id"], symbol=c["symbol"]) for c in hand]
                for hand in data["hands"]
            ],
            eliminated=set(data["eliminated"]),
            cards_traded_in_count=int(data["cards_traded_in_count"]),
            conquered_this_turn=bool(data.get("conquered_this_turn", False)),
            pending_attack=(
                PendingAttack(
                    from_index=int(pa["from_index"]),
                    to_index=int(pa["to_index"]),
                    attacker_dice=int(pa["attacker_dice"]),
                )
                if pa is not None
                else None
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    # --- equality (value-based) -------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # State is mutable

    # --- learning hook -----------------------------------------------------

    def to_features(self, topology) -> Any:  # noqa: ANN401 - returns tensors later
        """Future hook for the GNN adapter (see `risk/learning/graph_adapter.py`).

        Will return node features (owner one-hot + army counts + flags),
        edge_index from `topology.edge_index()`, and per-turn globals.
        Not implemented in the rules-only milestone.
        """
        raise NotImplementedError(
            "State.to_features is reserved for the GNN adapter (Phase: learning)."
        )


__all__ = ["State", "PendingAttack"]
