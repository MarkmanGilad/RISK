"""Off-policy replay buffer for the injection DQN (Net A, `Docs/NetworkArchitectures.md`).

Stores `(state, action, reward, next_state, done, stage, next_stage)`
transitions — `state`/`next_state` are raw `State` snapshots
(`risk/game/state.py`), `action` is the raw `Action` taken
(`risk/game/actions.py`), nothing graph-shaped at all. No dependency on
`GraphAdapter`, `ActionGraphBuilder`, `ActionEncoder`, or `Environment` —
this module only stores and randomly samples.

`push()` trusts the caller to hand over `state`/`next_state` objects that
are already independent (not aliasing the live, still-mutating `State`
`Environment` keeps reusing for the whole game) — it stores exactly what
it's given, no copying of its own. `GNN_DQN_Agent.remember()` is the one
real caller, and it both snapshots and tags its own `.perspective`
attribute onto each `State` *before* calling `push()` (`Docs/GraphAdapter.md`'s
perspective-rotation player, needed because a self-play trainer reassigns
the learner to a different seat every episode — `Docs/RL-Prep-Changes.md`'s
trainer — so it can't be re-read off the live agent at sample time and has
to travel with the transition). That's deliberately invisible to this
module: `perspective` is RL-training-only metadata about *who observed*
a state, not a property of the transition event itself, so it doesn't
belong in `ReplayBuffer`'s stored shape at all — see `remember()`'s own
docstring for how it's attached and read back.

`stage`/`next_stage` *are* stored (unlike everything graph-shaped below) —
`action.phase` and `next_state.phase` are single attribute reads, basically
free, so there's nothing to gain by deferring them: a transition is pushed
once but typically sampled many times over training (the whole point of an
off-policy buffer), so computing them once at push time does strictly less
total work than re-deriving them on every one of those resamples.

Why everything graph-shaped *isn't* precomputed, by contrast: building a
`Data` graph (`GraphAdapter`) or injecting one action's perturbation into
it (`ActionGraphBuilder`) is real, nontrivial work — a copy of a ~42-node
graph plus writing 1-2 tensor entries — and pre-building/storing one per
transition would multiply that cost by however many times it gets sampled
*and* pay extra memory for the stored result. Cheap-to-derive (stage) and
expensive-to-derive (graphs) get opposite treatment for the same reason:
minimize total work over the buffer's lifetime. The training loop owns
building `state`'s/`next_state`'s graphs, encoding `action`, enumerating
`next_state`'s legal actions via `env.legal_actions(next_state)` (which
takes an explicit `state` precisely so it can be called against a stored
snapshot rather than the live env, see `Environment.legal_actions`), and
injecting/batching whatever legal actions it needs for `Q(s, a)` and the
DQN target.
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

Transition = tuple   # (state, action, reward, next_state, done, stage, next_stage)


class ReplayBuffer:
    """Fixed-capacity ring buffer of `(state, action, reward, next_state,
    done, stage, next_stage)` transitions.

        buffer = ReplayBuffer()
        buffer.push(state, action, reward, next_state, done)
        (states, actions, rewards, next_states,
         dones, stages, next_stages) = buffer.sample(256)

    `states`/`actions`/`next_states` come back as plain tuples of domain
    objects (`State`/`Action`) — turning them into graphs/tensors is the
    training loop's job, not the buffer's. `rewards`/`dones`/`stages`/
    `next_stages` are numeric, so those come back as tensors. `stage`/
    `next_stage` aren't part of `push()`'s signature — they're computed
    from `action.phase`/`next_state.phase` once, inside `push()`, and
    stored alongside them (see the module docstring for why this one's
    eager, unlike everything graph-shaped). A sampled minibatch's
    transitions span many different decision points, so `stages`/
    `next_stages` are naturally mixed (unlike one decision's candidate set,
    which is single-stage since `Phase` doubles as the DQN
    action-representation stage directly — `Docs/RL-Prep-Changes.md`);
    they're what lets the training loop route each row to its head after
    one shared `Encoder` call, via a boolean mask per stage, without
    re-deriving it from the `Action` objects or
    `Environment.legal_actions(next_state)` first. `next_stage` is
    `int(Phase.GAME_OVER)` wherever `next_state` is terminal — no sentinel
    needed, `GAME_OVER`'s own value already distinguishes it from every
    real decision phase, and those rows are `done` anyway.
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
        # No copying here — `state`/`next_state` must already be
        # independent of `Environment`'s live, still-mutating `State`
        # (`GNN_DQN_Agent.remember()`'s job, the one real caller). `action`
        # is an immutable frozen dataclass (`Docs/Action.md`) — nothing to
        # copy regardless.
        self.buffer.append((
            state,
            action,
            float(reward),
            next_state,
            bool(done),
            int(action.phase),
            int(next_state.phase),
        ))

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        states, actions, rewards, next_states, dones, stages, next_stages = zip(
            *random.sample(self.buffer, batch_size)
        )
        reward = torch.tensor(rewards, dtype=torch.float32)
        done = torch.tensor(dones, dtype=torch.bool)
        stage = torch.tensor(stages, dtype=torch.long)
        next_stage = torch.tensor(next_stages, dtype=torch.long)
        return states, actions, reward, next_states, done, stage, next_stage

    def save(self, path: Union[str, Path]) -> None:
        torch.save(list(self.buffer), path)

    def __len__(self) -> int:
        return len(self.buffer)


__all__ = ["ReplayBuffer", "Transition", "DEFAULT_CAPACITY"]
