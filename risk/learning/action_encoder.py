"""Encodes `Action` candidates into tensors for DQN `Q(s, a)` scoring.

Each `Action` subclass already knows how to encode itself as a
`(stage, t1, t2, n)` integer tuple via `Action.dqn_index()` (see
`risk/game/actions.py` and `Docs/Action.md`'s "Representing actions for
DQN" section). This module just batches those tuples into a tensor — it's
the only place in this pipeline that imports torch for actions, mirroring
how `risk/learning/graph_adapter.py` is the only place that imports torch
for state.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from risk.game.actions import Action
from risk.game.board_topology import BoardTopology
from risk.game.environment import Environment
from risk.game.state import State


class ActionEncoder:
    """Converts `Action` candidates into `(stage, t1, t2, n)` tensors."""

    @staticmethod
    def encode_one(
        action: Action, topology: BoardTopology, state: Optional[State] = None
    ) -> tuple[int, int, int, int]:
        """One action -> its `(stage, t1, t2, n)` tuple (see `Action.dqn_index`)."""
        return action.dqn_index(topology, state)

    @classmethod
    def encode_many(
        cls,
        actions: Sequence[Action],
        topology: BoardTopology,
        state: Optional[State] = None,
    ) -> torch.Tensor:
        """A batch of actions -> a `[len(actions), 4]` long tensor.

        Row order matches `actions`' order, so `tensor[i]` is the encoding
        of `actions[i]` — pair them back up after scoring to know which
        `Action` object to `env.step(...)`.
        """
        if not actions:
            return torch.empty((0, 4), dtype=torch.long)
        rows = [cls.encode_one(a, topology, state) for a in actions]
        return torch.tensor(rows, dtype=torch.long)

    @classmethod
    def encode_legal(cls, env: Environment) -> tuple[list[Action], torch.Tensor]:
        """Convenience: `env.legal_actions()` for the current state -> `(candidates, tensor)`.

        `tensor[i]` encodes `candidates[i]`; after scoring, index back into
        `candidates` with the winning row to get the real `Action` to step.
        """
        candidates = env.legal_actions()
        tensor = cls.encode_many(candidates, env.topology, env.current_state())
        return candidates, tensor


__all__ = ["ActionEncoder"]
