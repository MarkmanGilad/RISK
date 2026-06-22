"""Encodes legal `Action`s into tensors for DQN `Q(s, a)` scoring.

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
from risk.game.environment import Environment
from risk.game.state import State


class ActionEncoder:
    """Converts legal `Action`s into `(stage, t1, t2, n)` tensors.

    Built around one `Environment` so it can always pull the current
    `topology`/`state` automatically:

        encoder = ActionEncoder(env)
        legal_actions, tensor = encoder()   # == encoder.encode_legal()
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

    def __call__(self) -> tuple[list[Action], torch.Tensor]:
        return self.encode_legal()

    def encode_one(
        self, action: Action, state: Optional[State] = None
    ) -> tuple[int, int, int, int]:
        """One action -> its `(stage, t1, t2, n)` tuple (see `Action.dqn_index`).

        `state` defaults to `self.env.current_state()`; pass it explicitly
        only if encoding against a different snapshot.
        """
        if state is None:
            state = self.env.current_state()
        return action.dqn_index(self.env.topology, state)

    def encode_many(
        self, actions: Sequence[Action], state: Optional[State] = None
    ) -> torch.Tensor:
        """A batch of actions -> a `[len(actions), 4]` long tensor.

        Row order matches `actions`' order, so `tensor[i]` is the encoding
        of `actions[i]` — pair them back up after scoring to know which
        `Action` object to `env.step(...)`.
        """
        if not actions:
            return torch.empty((0, 4), dtype=torch.long)
        rows = [self.encode_one(a, state) for a in actions]
        return torch.tensor(rows, dtype=torch.long)

    def encode_batch(
        self, actions: Sequence[Action], states: Sequence[State]
    ) -> torch.Tensor:
        """Like `encode_many`, but each action has its own state.

        For a replay-buffer minibatch, where every transition's action was
        taken against a different state — unlike `encode_many`'s single
        shared `state` for every candidate of one decision.

        `tensor[i]` encodes `(actions[i], states[i])`; row order matches.
        """
        if not actions:
            return torch.empty((0, 4), dtype=torch.long)
        rows = [self.encode_one(a, s) for a, s in zip(actions, states)]
        return torch.tensor(rows, dtype=torch.long)

    def encode_legal(self) -> tuple[list[Action], torch.Tensor]:
        """`self.env.legal_actions()` for the current state -> `(legal_actions, tensor)`.

        `tensor[i]` encodes `legal_actions[i]`; after scoring, index back
        into `legal_actions` with the winning row to get the real `Action`
        to step.
        """
        legal_actions = self.env.legal_actions()
        tensor = self.encode_many(legal_actions, self.env.current_state())
        return legal_actions, tensor


__all__ = ["ActionEncoder"]
