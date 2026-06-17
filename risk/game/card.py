"""Risk territory card and trade-in detection.

`Card` is an immutable value object carrying the territory id and the
symbol stamped on the card (`infantry`, `cavalry`, `artillery`, or
`wild`). The wild card has no territory.

`is_valid_set(cards)` decides whether three cards form a tradeable set
under classic Risk rules:

- three cards of the same symbol, or
- one of each of the three regular symbols,

with wilds substituting for any symbol.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from risk.constants import CARD_SYMBOLS, WILD_SYMBOL


_ALL_SYMBOLS: frozenset[str] = frozenset(CARD_SYMBOLS) | {WILD_SYMBOL}


@dataclass(frozen=True, slots=True)
class Card:
    territory_id: Optional[str]
    symbol: str

    def __post_init__(self) -> None:
        if self.symbol not in _ALL_SYMBOLS:
            raise ValueError(
                f"Unknown card symbol {self.symbol!r}; "
                f"expected one of {sorted(_ALL_SYMBOLS)}"
            )
        if self.symbol == WILD_SYMBOL:
            if self.territory_id is not None:
                raise ValueError("Wild card must have territory_id=None")
        else:
            if not isinstance(self.territory_id, str) or not self.territory_id:
                raise ValueError(
                    f"Non-wild card requires a territory_id, got {self.territory_id!r}"
                )

    @property
    def is_wild(self) -> bool:
        return self.symbol == WILD_SYMBOL


class CardRules:
    """Trade-in and validity rules for `Card`s."""

    @staticmethod
    def validate_against_topology(card: Card, topology) -> None:
        """Raise if a non-wild card references a territory not in the topology."""
        if card.is_wild:
            return
        if card.territory_id not in topology.territories:
            raise ValueError(
                f"Card references unknown territory {card.territory_id!r}"
            )

    @staticmethod
    def is_valid_set(cards: Sequence[Card]) -> bool:
        """Return True if these exactly three cards form a tradeable set."""
        if len(cards) != 3:
            return False

        wilds = sum(1 for c in cards if c.is_wild)
        non_wild = Counter(c.symbol for c in cards if not c.is_wild)
        distinct = len(non_wild)

        # Three of a kind (wilds collapse to a single symbol).
        if distinct <= 1:
            return True

        # One of each regular symbol, wilds filling in the rest.
        if distinct + wilds == 3 and all(v == 1 for v in non_wild.values()):
            return True

        return False

    @classmethod
    def find_valid_set(cls, hand: Iterable[Card]) -> Optional[tuple[Card, Card, Card]]:
        """Return any 3-card valid subset of `hand`, or None if none exists."""
        cards = list(hand)
        n = len(cards)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    trio = (cards[i], cards[j], cards[k])
                    if cls.is_valid_set(trio):
                        return trio
        return None


__all__ = ["Card", "CardRules"]
