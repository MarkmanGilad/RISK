"""GNN+DQN inference agent (Net A) for self-play/training rollouts.

This class is the game-logic wrapper around `GNN_DQN`:

- asks `env.legal_actions(state)`
- builds one graph row per legal action (`ActionGraphBuilder` for graph
  stages; base graph for `TRADE_IN`)
- batches and scores all rows with `GNN_DQN`
- returns the selected `Action`

Training/update code is intentionally deferred; this class provides action
selection and owns a `ReplayBuffer` so training can be plugged in later.
"""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.environment import Environment
from risk.game.phase import Phase
from risk.game.state import State
from risk.learning.action_encoder import ActionEncoder
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.gnn_dqn import GNN_DQN
from risk.learning.graph_adapter import EDGE_ATTR_DIM, GraphAdapter
from risk.learning.replay_buffer import ReplayBuffer


def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """Return the runtime device: explicit override, else CUDA if available."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class GNN_DQN_Agent(BaseAgent):
    """`BaseAgent` wrapper for `GNN_DQN` action scoring."""

    def __init__(self, player_id: int, env: Environment, *,
                 replay_buffer: ReplayBuffer | None = None, device: torch.device | None = None,
                 epsilon: float = 0.0, train_mode: bool = False, seed: int | None = None,
                 gamma: float = 0.99, lr: float = 1e-4, target_update_every: int = 1000,) -> None:
        super().__init__(player_id)
        self.env = env
        self.device = resolve_device(device)
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)
        self.gamma = float(gamma)
        self.target_update_every = int(target_update_every)
        self._train_steps = 0

        sample = self.adapter(env.current_state(), perspective=self.player_id)
        self.net = GNN_DQN(
            in_dim=sample.x.shape[1],
            hidden_dim=64,
            edge_dim=EDGE_ATTR_DIM,
            u_dim=sample.u.shape[1],
        ).to(self.device)

        # Kept for DQN training parity; synchronized now, updated during training later.
        self.target_net = deepcopy(self.net).to(self.device)
        self.target_net.eval()
        self.set_train_mode(train_mode)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    @property
    def train_steps(self) -> int:
        return self._train_steps

    def attach(self, player_id: int, env: Environment) -> None:
        """Rebind this agent to a new episode's seat/env.

        `net`/`target_net`/`optimizer`/`replay_buffer` persist across the
        call (that's the whole point — one learner trained over many
        self-play episodes); only the per-episode bindings are rebuilt,
        the same way `__init__` builds them the first time. Lets a trainer
        reuse one `GNN_DQN_Agent` while reassigning it to a different
        physical seat (and a freshly built `env`/`GameContext`) every
        episode — see `Docs/RL-Prep-Changes.md`'s trainer.
        """
        self.player_id = player_id
        self.env = env
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)

    def load_params(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def save_params(self, path: str | Path) -> None:
        """Policy-only checkpoint: just the net weights, for play/inference."""
        torch.save(self.net.state_dict(), path)

    def save_checkpoint(self, dir_path: str | Path) -> None:
        """Full training checkpoint: everything needed to resume exactly.

        Writes two files under `dir_path`: `model.pt` (net/target_net/
        optimizer/train_steps/epsilon state) and `replay.pt` (the replay
        buffer, via `ReplayBuffer.save`, kept separate since it can be much
        larger than the model state).
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
                "epsilon": self.epsilon,
            },
            dir_path / "model.pt",
        )
        self.replay_buffer.save(dir_path / "replay.pt")

    def load_checkpoint(self, dir_path: str | Path) -> None:
        """Inverse of `save_checkpoint` — restores net/target/optimizer/
        train_steps/epsilon and replaces `self.replay_buffer` with the
        saved one."""
        dir_path = Path(dir_path)
        payload = torch.load(dir_path / "model.pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._train_steps = int(payload["train_steps"])
        self.epsilon = float(payload["epsilon"])
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer.capacity, path=dir_path / "replay.pt")

    def set_train_mode(self, train: bool) -> None:
        self.train_mode = bool(train)
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

    def device_report(self, state: State | None = None) -> dict[str, str]:
        """Small runtime report to verify model/tensors are on the selected device."""
        if state is None:
            state = self.env.current_state()

        report = {
            "selected_device": str(self.device),
            "cuda_available": str(torch.cuda.is_available()),
            "net_device": str(next(self.net.parameters()).device),
        }

        legal_actions = self.env.legal_actions(state)
        if not legal_actions:
            return report

        base = self.adapter(state, perspective=self.player_id)
        rows = [self.builder(base, legal_actions[0], state)]
        batch = Batch.from_data_list(rows).to(self.device)
        encoded = self.action_encoder.encode_many(legal_actions[:1], state).to(self.device)
        phase = encoded[:, 0]
        card_indices = encoded[:, 1:4]

        report["batch_device"] = str(batch.x.device)
        report["phase_device"] = str(phase.device)
        report["card_indices_device"] = str(card_indices.device)
        return report

    def act(self, events: Sequence[object], state: State) -> Action | None:
        del events
        legal_actions = self.env.legal_actions(state)
        if not legal_actions:
            return None

        if self.train_mode and self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)

        q_values = self.score_actions(state, legal_actions)
        best_index = int(torch.argmax(q_values).item())
        return legal_actions[best_index]

    def score_actions(self, state: State, legal_actions: Sequence[Action]) -> torch.Tensor:
        """Score each legal action with the online net and return `[N]` Q values."""
        base = self.adapter(state, perspective=self.player_id)
        rows: list[Data] = []
        for action in legal_actions:
            rows.append(self.builder(base, action, state))

        batch = Batch.from_data_list(rows).to(self.device)
        encoded = self.action_encoder.encode_many(legal_actions, state).to(self.device)
        phase = encoded[:, 0]
        card_indices = encoded[:, 1:4]

        # Keep device placement explicit; this catches accidental CPU/GPU mismatches early.
        net_device = next(self.net.parameters()).device
        if net_device != self.device:
            raise RuntimeError(f"Net is on {net_device}, expected {self.device}")
        if batch.x.device != self.device:
            raise RuntimeError(f"Batch is on {batch.x.device}, expected {self.device}")
        if phase.device != self.device or card_indices.device != self.device:
            raise RuntimeError(
                "Action tensors must be on "
                f"{self.device}, got phase={phase.device}, card_indices={card_indices.device}"
            )

        with torch.no_grad():
            q_values = self.net(batch, phase, card_indices)
        return q_values

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool,) -> None:
        """Snapshot and store one transition.

        Snapshots here (not in `ReplayBuffer.push`, which trusts its
        caller) so it can also tag each copy with `.perspective =
        self.player_id` — this agent's *current* seat, captured now rather
        than re-read at training time, since a self-play trainer reassigns
        this agent to a different seat every episode (`attach`). `State` is
        a plain, unfrozen dataclass, so this is a normal attribute set, not
        a declared field — invisible to `to_dict()`/`__eq__`, read back by
        `_q_value`/`_max_next_q` as `state.perspective` when building that
        transition's graph.
        """
        state_snapshot = state.snapshot()
        next_state_snapshot = next_state.snapshot()
        state_snapshot.perspective = self.player_id
        next_state_snapshot.perspective = self.player_id
        self.replay_buffer.push(state_snapshot, action, reward, next_state_snapshot, done)

    def can_train(self, batch_size: int) -> bool:
        return len(self.replay_buffer) >= batch_size

    def learn_steps(self, batch_size: int, n_steps: int) -> list[float]:
        losses: list[float] = []
        for _ in range(max(0, n_steps)):
            losses.append(self.train_step(batch_size))
        return losses

    def learn(self, *, batch_size: int, n_steps: int) -> list[float]:
        if not self.can_train(batch_size):
            return []
        return self.learn_steps(batch_size, n_steps)

    def _score(self, net: torch.nn.Module, rows: list[Data], phase: torch.Tensor,
               card_indices: torch.Tensor) -> torch.Tensor:
        """Batch `rows` and score them with `net`. Shared by the current-Q
        and target-Q passes below — only the rows/net/grad-tracking differ."""
        batch = Batch.from_data_list(rows).to(self.device)
        return net(batch, phase.to(self.device), card_indices.to(self.device))

    def _q_value(self, states: Sequence[State], actions: Sequence[Action],
                 stage: torch.Tensor) -> torch.Tensor:
        """`Q(s, a)` for the taken `(state, action)` pairs, via the online net.

        Each transition has its own `state`/`action` pair, unlike
        `score_actions`'s "one state, many candidate actions" shape — so
        rows/`card_indices` are built via `encode_batch` (one state per
        action) rather than `encode_many` (one shared state for every
        action). Each `state`'s own `.perspective` (set by
        `ReplayBuffer.push`, `Docs/RL-Prep-Changes.md`'s trainer reassigns
        this agent's seat every episode) is used to build its graph, not
        this agent's current seat.
        """
        rows = [self.builder(self.adapter(s, perspective=s.perspective), a, s)
                for s, a in zip(states, actions)]
        card_indices = self.action_encoder.encode_batch(actions, states)[:, 1:4]
        return self._score(self.net, rows, stage, card_indices)

    def _max_next_q(self, next_states: Sequence[State], done: torch.Tensor,
                     next_stage: torch.Tensor) -> torch.Tensor:
        """`max_a' Q_target(next_state, a')` per transition, `0` where `done`.

        Each `next_state` has its own candidate count, so every candidate
        across the whole minibatch is built/batched/scored in one shot, then
        reduced back to one max per transition with a per-row group index
        (`torch_geometric.utils.scatter(..., reduce="max")`) rather than
        looping a separate forward pass per transition. Every legal action
        of one `next_state` shares that state's `phase` (`Phase`/decision-
        stage are 1:1 — `Docs/RL-Prep-Changes.md`), so the per-row `phase`
        the head-routing needs is just `next_stage` broadcast per group,
        not re-derived from each candidate's own `dqn_index()`. Likewise
        every candidate of one `next_state` is built from that state's own
        `.perspective` (set by `ReplayBuffer.push`), not this agent's
        current seat.
        """
        max_q = torch.zeros(len(next_states), dtype=torch.float32, device=self.device)
        rows: list[Data] = []
        groups: list[int] = []
        card_rows: list[tuple[int, int, int]] = []
        for i, ns in enumerate(next_states):
            if bool(done[i]):
                continue
            legal = self.env.legal_actions(ns)
            if not legal:
                continue
            base = self.adapter(ns, perspective=ns.perspective)
            for a in legal:
                rows.append(self.builder(base, a, ns))
                groups.append(i)
                card_rows.append(a.dqn_index(self.env.topology, ns)[1:])

        if not rows:
            return max_q

        group_index = torch.tensor(groups, dtype=torch.long)
        phase = next_stage[group_index]
        card_indices = torch.tensor(card_rows, dtype=torch.long)
        with torch.no_grad():
            q_all = self._score(self.target_net, rows, phase, card_indices)
        group_index = group_index.to(self.device)
        per_group_max = scatter(q_all, group_index, dim=0, dim_size=len(next_states), reduce="max")
        touched = torch.zeros(len(next_states), dtype=torch.bool, device=self.device)
        touched[group_index.unique()] = True
        max_q[touched] = per_group_max[touched]
        return max_q

    def train_step(self, batch_size: int) -> float:
        """One DQN TD-error update from a sampled replay minibatch.

        `Q(s, a)` (online net, gradients on) vs. `r + gamma * (1 - done) *
        max_a' Q(s', a')` (target net, no gradients) — standard DQN, Huber
        loss. Returns the scalar loss. Target net is hard-synced to the
        online net every `target_update_every` calls.
        """
        states, actions, reward, next_states, done, stage, next_stage = (
            self.replay_buffer.sample(batch_size)
        )
        reward = reward.to(self.device)
        done = done.to(self.device)

        q_value = self._q_value(states, actions, stage)
        max_next_q = self._max_next_q(next_states, done, next_stage)
        target_q = reward + self.gamma * (~done).float() * max_next_q

        loss = F.mse_loss(q_value, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return float(loss.item())


__all__ = ["GNN_DQN_Agent"]
