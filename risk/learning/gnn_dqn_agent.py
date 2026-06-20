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
from torch_geometric.data import Batch, Data

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
                 epsilon: float = 0.0, train_mode: bool = False, seed: int | None = None,) -> None:
        super().__init__(player_id)
        self.env = env
        self.device = resolve_device(device)
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)

        sample = self.adapter(env.current_state())
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

    def load_params(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def save_params(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

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

        base = self.adapter(state)
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
        base = self.adapter(state)
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
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, *args, **kwargs) -> None:
        """Training loop intentionally deferred; implemented in a later pass."""
        raise NotImplementedError("GNN_DQN_Agent.train_step will be implemented later.")


__all__ = ["GNN_DQN_Agent"]
