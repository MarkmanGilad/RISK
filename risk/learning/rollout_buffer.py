"""Ordered, disposable on-policy storage for PPO."""
from __future__ import annotations

from typing import NamedTuple

from risk.game.actions import Action
from risk.game.state import State


class RolloutTransition(NamedTuple):
    state: State
    action: Action
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    next_state: State
    done: bool
    gae_boundary: bool = False


class RolloutBuffer:
    """A rollout in collection order; it is never sampled or persisted."""

    def __init__(self) -> None:
        self._transitions: list[RolloutTransition] = []

    def push(self, state: State, action: Action, action_index: int, log_prob: float,
             value: float, reward: float, next_state: State, done: bool) -> None:
        self._transitions.append(RolloutTransition(
            state, action, int(action_index), float(log_prob), float(value),
            float(reward), next_state, bool(done),
        ))

    def mark_last_boundary(self) -> None:
        if not self._transitions:
            return
        transition = self._transitions[-1]
        self._transitions[-1] = RolloutTransition(
            transition.state, transition.action, transition.action_index,
            transition.old_log_prob, transition.old_value, transition.reward,
            transition.next_state, transition.done, True,
        )

    def all(self) -> tuple[RolloutTransition, ...]:
        return tuple(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)


__all__ = ["RolloutBuffer", "RolloutTransition"]
