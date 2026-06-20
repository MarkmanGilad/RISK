"""Off-policy replay buffer for the injection DQN (Net A, `Docs/NetworkArchitectures.md`).

Stores plain `(state, action, reward, next_state, done)` transitions, the
same shape as a classic DQN buffer — `state`/`next_state` are raw `State`
snapshots (`risk/game/state.py`), `action` is the raw `Action` taken
(`risk/game/actions.py`), nothing graph- or tensor-shaped at all. No
dependency on `GraphAdapter`, `ActionGraphBuilder`, `ActionEncoder`, or
`Environment` — this module only stores and randomly samples.

Why so little is precomputed: building a `Data` graph (`GraphAdapter`) or
injecting one action's perturbation into it (`ActionGraphBuilder`) is
cheap — a copy of a ~42-node graph plus writing 1-2 tensor entries — next to
the GNN forward pass that has to run over it at train time regardless. So
there's nothing to gain by paying that cost once at push time and storing
the result; it's cheaper, in memory terms, to redo it lazily every time a
transition is sampled. The training loop owns all of that: building
`state`'s/`next_state`'s graphs, encoding `action`, enumerating `next_state`'s
legal actions via `env.legal_actions(next_state)` (which takes an explicit
`state` precisely so it can be called against a stored snapshot rather than
the live env, see `Environment.legal_actions`), and injecting/batching
whatever legal actions it needs for `Q(s, a)` and the DQN target.
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Optional, Union

import torch

from risk.game.actions import Action
from risk.game.state import State

DEFAULT_CAPACITY = 500_000

Transition = tuple   # (state: State, action: Action, reward: float, next_state: State, done: bool)


class ReplayBuffer:
    """Fixed-capacity ring buffer of `(state, action, reward, next_state, done)`.

        buffer = ReplayBuffer()
        buffer.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buffer.sample(256)

    `states`/`actions`/`next_states` come back as plain tuples of domain
    objects (`State`/`Action`) — turning them into graphs/tensors is the
    training loop's job, not the buffer's. `rewards`/`dones` are the only
    fields uniformly numeric, so those come back as tensors.
    """

    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, path: Optional[Union[str, Path]] = None
    ) -> None:
        self.capacity = capacity
        if path is not None:
            self.buffer = deque(torch.load(path), maxlen=capacity)
        else:
            self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        # `Environment` mutates one `State` object in place for the whole
        # game (see `risk/learning/self_play.py`), so without snapshotting
        # here every stored transition would end up aliasing whatever that
        # object looks like by the time training reads it back, regardless
        # of whether the caller already snapshotted. `action` is an
        # immutable frozen dataclass (`Docs/Action.md`) — nothing to copy.
        self.buffer.append(
            (state.snapshot(), action, float(reward), next_state.snapshot(), bool(done))
        )

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        reward = torch.tensor(rewards, dtype=torch.float32)
        done = torch.tensor(dones, dtype=torch.bool)
        return states, actions, reward, next_states, done

    def save(self, path: Union[str, Path]) -> None:
        torch.save(list(self.buffer), path)

    def __len__(self) -> int:
        return len(self.buffer)


__all__ = ["ReplayBuffer", "Transition", "DEFAULT_CAPACITY"]
