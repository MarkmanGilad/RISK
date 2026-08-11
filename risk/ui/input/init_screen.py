"""Initial setup screen.

Headless-friendly state machine that the UI binds events into and the
loop renders. The data model is fully testable without pygame.

Use:
- `setup = InitScreenState()` — defaults to 3 human players.
- `setup.set_player_count(n)` — clamp to 3..6.
- `setup.set_name(i, name)`, `set_color(i, rgb)`, `set_agent_kind(i, kind)`.
- `setup.build_settings(seed=None)` -> `GameSettings`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from risk.constants import MAX_PLAYERS, MIN_PLAYERS
from risk.game.player import AGENT_KIND_ORDER, ALLOWED_AGENT_KINDS, Player
from risk.game.settings import GameSettings
from risk.ui_constants import DEFAULT_PLAYER_COLORS as DEFAULT_COLORS


@dataclass
class _Seat:
    name: str
    color: tuple[int, int, int]
    agent_kind: str


@dataclass
class InitScreenState:
    seats: list[_Seat] = field(default_factory=list)
    learned_selections: dict[int, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seats:
            self.seats = [
                _Seat(name=f"Player {i + 1}", color=DEFAULT_COLORS[i], agent_kind="human")
                for i in range(MIN_PLAYERS)
            ]

    # --- mutators ---------------------------------------------------------

    def set_player_count(self, n: int) -> None:
        if not (MIN_PLAYERS <= n <= MAX_PLAYERS):
            raise ValueError(f"player count must be {MIN_PLAYERS}..{MAX_PLAYERS}")
        while len(self.seats) < n:
            i = len(self.seats)
            self.seats.append(
                _Seat(name=f"Player {i + 1}", color=DEFAULT_COLORS[i], agent_kind="human")
            )
        while len(self.seats) > n:
            self.seats.pop()
        self.learned_selections = {
            seat: selection for seat, selection in self.learned_selections.items() if seat < n
        }

    def set_name(self, index: int, name: str) -> None:
        self._check_index(index)
        name = name.strip()
        if not name:
            raise ValueError("name cannot be empty")
        self.seats[index].name = name

    def set_color(self, index: int, color: tuple[int, int, int]) -> None:
        self._check_index(index)
        if (
            not isinstance(color, tuple)
            or len(color) != 3
            or not all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        ):
            raise ValueError(f"invalid RGB tuple: {color!r}")
        self.seats[index].color = color

    def set_agent_kind(self, index: int, kind: str) -> None:
        self._check_index(index)
        if kind not in ALLOWED_AGENT_KINDS:
            raise ValueError(
                f"agent_kind must be one of {sorted(ALLOWED_AGENT_KINDS)}, got {kind!r}"
            )
        self.seats[index].agent_kind = kind

    def next_agent_kind(self, index: int) -> str:
        self._check_index(index)
        current = self.seats[index].agent_kind
        if current not in AGENT_KIND_ORDER:
            current = "random"
        next_index = (AGENT_KIND_ORDER.index(current) + 1) % len(AGENT_KIND_ORDER)
        self.seats[index].agent_kind = AGENT_KIND_ORDER[next_index]
        return self.seats[index].agent_kind

    def next_visible_agent_kind(self, index: int) -> str:
        """Cycle the setup-screen types, including its UI-only learned seat."""
        self._check_index(index)
        if index in self.learned_selections:
            self.learned_selections.pop(index)
            self.seats[index].agent_kind = "human"
            return "human"
        if self.seats[index].agent_kind == "killbot":
            self.seats[index].agent_kind = "ai"
            self.learned_selections[index] = {
                "source": "",
                "checkpoint": "",
                "agent_kind": "DQN",
                "label": "",
                "preset_id": "",
            }
            return "learned"
        return self.next_agent_kind(index)

    def set_learned_selection(
        self, index: int, *, source: str, checkpoint: str, agent_kind: str,
        label: str, preset_id: str = "",
    ) -> None:
        self._check_index(index)
        if source not in {"manual", "preset"}:
            raise ValueError("learned source must be manual or preset")
        if agent_kind not in {"DQN", "Dueling_DQN", "PPO"}:
            raise ValueError("unsupported learned agent kind")
        self.seats[index].agent_kind = "ai"
        self.learned_selections[index] = {
            "source": source,
            "checkpoint": checkpoint,
            "agent_kind": agent_kind,
            "label": label,
            "preset_id": preset_id,
        }

    def is_learned(self, index: int) -> bool:
        self._check_index(index)
        return index in self.learned_selections

    def _check_index(self, index: int) -> None:
        if not (0 <= index < len(self.seats)):
            raise IndexError(f"seat index out of range: {index}")

    # --- queries ----------------------------------------------------------

    @property
    def player_count(self) -> int:
        return len(self.seats)

    def can_start(self) -> tuple[bool, str]:
        if not (MIN_PLAYERS <= self.player_count <= MAX_PLAYERS):
            return False, f"Need {MIN_PLAYERS}..{MAX_PLAYERS} players."
        names = [s.name for s in self.seats]
        if any(not n.strip() for n in names):
            return False, "All players need a name."
        if len(set(c for c in (s.color for s in self.seats))) != len(self.seats):
            return False, "Player colors must be unique."
        return True, ""

    def build_settings(self, seed: Optional[int] = None) -> GameSettings:
        ok, why = self.can_start()
        if not ok:
            raise ValueError(why)
        players = tuple(
            Player(id=i, name=s.name, color=s.color, agent_kind=s.agent_kind)
            for i, s in enumerate(self.seats)
        )
        return GameSettings(players=players, seed=seed)


__all__ = ["InitScreenState", "DEFAULT_COLORS"]
