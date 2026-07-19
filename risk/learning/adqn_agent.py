"""Advantage Dueling DQN agent — the implementation of ``Docs/ADQN.md``.

ADQN owns an independent copy of the raw dueling ``(V, A)`` network and agent
plumbing. It follows Dueling DQN's replay, Double-DQN target, epsilon-greedy
action selection, and Bellman loss, then adds the bounded centered-advantage
objective described in the design. ADQN and PQN are sibling algorithms; this
module does not inherit from or construct either PQN class.
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
from risk.game.state import State
from risk.learning.action_encoder import ActionEncoder
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.adqn import ADQN
from risk.learning.graph_adapter import EDGE_ATTR_DIM, GraphAdapter
from risk.learning.replay_buffer import ReplayBuffer
from risk.learning.train_constants import (
    ADQN_ADVANTAGE_LOSS_COEF,
    ADQN_ADVANTAGE_WEIGHT_SCALE,
    ADQN_ADVANTAGE_WEIGHT_SATURATION,
    ADQN_GRAD_DIAGNOSTIC_EVERY,
    ADQN_LOSS_BALANCE_EPSILON,
    ADQN_MAX_ADVANTAGE_LOSS_FRACTION,
    BATCH_SIZE,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    GRAD_CLIP_MAX_NORM,
    TRAIN_STEPS_PER_CALL,
)


def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class ADQN_Agent(BaseAgent):
    """Dueling Q learner with a bounded, signed centered-advantage loss."""

    label = "ADQN"

    def __init__(
        self,
        player_id: int,
        env: Environment,
        *,
        replay_buffer: ReplayBuffer | None = None,
        device: torch.device | None = None,
        epsilon: float = EPSILON_START,
        advantage_loss_coef: float = ADQN_ADVANTAGE_LOSS_COEF,
        max_advantage_loss_fraction: float = ADQN_MAX_ADVANTAGE_LOSS_FRACTION,
        advantage_weight_scale: float = ADQN_ADVANTAGE_WEIGHT_SCALE,
        advantage_weight_saturation: float = ADQN_ADVANTAGE_WEIGHT_SATURATION,
        grad_diagnostic_every: int = ADQN_GRAD_DIAGNOSTIC_EVERY,
        loss_balance_epsilon: float = ADQN_LOSS_BALANCE_EPSILON,
        train_mode: bool = False,
        seed: int | None = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        target_update_every: int = 1000,
    ) -> None:
        if advantage_loss_coef < 0:
            raise ValueError("advantage_loss_coef must be non-negative")
        if max_advantage_loss_fraction < 0:
            raise ValueError("max_advantage_loss_fraction must be non-negative")
        if advantage_weight_scale <= 0:
            raise ValueError("advantage_weight_scale must be positive")
        if not 0 <= advantage_weight_saturation <= 1:
            raise ValueError("advantage_weight_saturation must lie in [0, 1]")
        if grad_diagnostic_every < 1:
            raise ValueError("grad_diagnostic_every must be at least 1")
        if loss_balance_epsilon <= 0:
            raise ValueError("loss_balance_epsilon must be positive")

        super().__init__(player_id)
        self.env = env
        self.device = resolve_device(device)
        self.action_selection = "epsilon_greedy_q"
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)
        self.gamma = float(gamma)
        self.target_update_every = int(target_update_every)
        self._train_steps = 0
        self.last_update_metrics: dict[str, float] = {}

        sample = self.adapter(env.current_state(), perspective=self.player_id)
        self.net = ADQN(
            in_dim=sample.x.shape[1],
            hidden_dim=64,
            edge_dim=EDGE_ATTR_DIM,
            u_dim=sample.u.shape[1],
        ).to(self.device)
        self.target_net = deepcopy(self.net).to(self.device)
        self.target_net.eval()
        self.set_train_mode(train_mode)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.advantage_loss_coef = float(advantage_loss_coef)
        self.max_advantage_loss_fraction = float(max_advantage_loss_fraction)
        self.advantage_weight_scale = float(advantage_weight_scale)
        self.advantage_weight_saturation = float(advantage_weight_saturation)
        self.grad_diagnostic_every = int(grad_diagnostic_every)
        self.loss_balance_epsilon = float(loss_balance_epsilon)

    @property
    def train_steps(self) -> int:
        return self._train_steps

    def progress_metrics(self) -> dict[str, float]:
        return {
            "epsilon": self.epsilon,
            "adqn_advantage_loss_coef": self.advantage_loss_coef,
            "adqn_max_advantage_loss_fraction": self.max_advantage_loss_fraction,
            "adqn_advantage_weight_scale": self.advantage_weight_scale,
            "adqn_replay_buffer_size": float(len(self.replay_buffer)),
            "adqn_train_steps_since_target_sync": float(
                self._train_steps % self.target_update_every
            ),
        }

    def attach(self, player_id: int, env: Environment) -> None:
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
        torch.save(self.net.state_dict(), path)

    def save_checkpoint(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
                "epsilon": self.epsilon,
                "advantage_loss_coef": self.advantage_loss_coef,
                "max_advantage_loss_fraction": self.max_advantage_loss_fraction,
                "advantage_weight_scale": self.advantage_weight_scale,
                "advantage_weight_saturation": self.advantage_weight_saturation,
                "grad_diagnostic_every": self.grad_diagnostic_every,
                "loss_balance_epsilon": self.loss_balance_epsilon,
            },
            dir_path / "model.pt",
        )
        self.replay_buffer.save(dir_path / "replay.pt")

    def load_checkpoint(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        payload = torch.load(
            dir_path / "model.pt", map_location=self.device, weights_only=False
        )
        self.net.load_state_dict(payload["net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._train_steps = int(payload["train_steps"])
        self.epsilon = float(payload.get("epsilon", EPSILON_START))
        self.advantage_loss_coef = float(
            payload.get("advantage_loss_coef", ADQN_ADVANTAGE_LOSS_COEF)
        )
        self.max_advantage_loss_fraction = float(
            payload.get(
                "max_advantage_loss_fraction", ADQN_MAX_ADVANTAGE_LOSS_FRACTION
            )
        )
        # Checkpoints written before this setting existed used tanh(td_advantage),
        # which is exactly the scaled formula with a scale of 1.0.
        self.advantage_weight_scale = float(
            payload.get("advantage_weight_scale", 1.0)
        )
        self.advantage_weight_saturation = float(
            payload.get(
                "advantage_weight_saturation", ADQN_ADVANTAGE_WEIGHT_SATURATION
            )
        )
        self.grad_diagnostic_every = int(
            payload.get("grad_diagnostic_every", ADQN_GRAD_DIAGNOSTIC_EVERY)
        )
        self.loss_balance_epsilon = float(
            payload.get("loss_balance_epsilon", ADQN_LOSS_BALANCE_EPSILON)
        )
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer.capacity, path=dir_path / "replay.pt"
        )

    def on_episode_start(self, episode: int) -> None:
        progress = min(max(episode - 1, 0) / EPSILON_DECAY_EPISODES, 1.0)
        self.epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * progress

    def set_train_mode(self, train: bool) -> None:
        self.train_mode = bool(train)
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

    def act(self, events: Sequence[object], state: State) -> Action | None:
        del events
        legal_actions = self.env.legal_actions(state)
        if not legal_actions:
            return None
        if self.train_mode and self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)

        value, advantage = self.score_actions(state, legal_actions)
        group_index = torch.zeros(
            len(legal_actions), dtype=torch.long, device=self.device
        )
        q_values = self._combine_q(value, advantage, group_index)
        return legal_actions[int(torch.argmax(q_values).item())]

    def score_actions(
        self, state: State, legal_actions: Sequence[Action]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.adapter(state, perspective=self.player_id)
        rows: list[Data] = [base]
        for action in legal_actions:
            rows.append(self.builder(base, action, state))

        batch = Batch.from_data_list(rows).to(self.device)
        encoded = self.action_encoder.encode_many(legal_actions, state).to(self.device)
        phase = torch.cat([encoded[:1, 0], encoded[:, 0]])
        card_indices = torch.cat(
            [
                torch.zeros((1, 3), dtype=torch.long, device=self.device),
                encoded[:, 1:4],
            ]
        )
        group_index = torch.zeros(len(rows), dtype=torch.long, device=self.device)
        value_mask = torch.zeros(len(rows), dtype=torch.bool, device=self.device)
        value_mask[0] = True
        with torch.no_grad():
            return self.net(
                batch,
                phase,
                card_indices,
                value_mask=value_mask,
                group_index=group_index,
            )

    def remember(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        state_snapshot = state.snapshot()
        next_state_snapshot = next_state.snapshot()
        state_snapshot.perspective = self.player_id
        next_state_snapshot.perspective = self.player_id
        self.replay_buffer.push(
            state_snapshot, action, reward, next_state_snapshot, done
        )

    def can_train(self) -> bool:
        return len(self.replay_buffer) >= BATCH_SIZE

    def learn_steps(self) -> list[float]:
        return [
            self.train_step(BATCH_SIZE)
            for _ in range(max(0, TRAIN_STEPS_PER_CALL))
        ]

    def learn(self, *, reached_max_steps: bool = False) -> list[float]:
        del reached_max_steps
        if not self.can_train():
            return []
        return self.learn_steps()

    def _combine_q(
        self,
        value: torch.Tensor,
        advantage: torch.Tensor,
        group_index: torch.Tensor,
    ) -> torch.Tensor:
        advantage_mean = scatter(
            advantage,
            group_index,
            dim=0,
            dim_size=value.shape[0],
            reduce="mean",
        )
        return value[group_index] + advantage - advantage_mean[group_index]

    def _score(
        self,
        net: torch.nn.Module,
        rows: list[Data],
        phase: torch.Tensor,
        card_indices: torch.Tensor,
        value_mask: torch.Tensor,
        group_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = Batch.from_data_list(rows).to(self.device)
        return net(
            batch,
            phase.to(self.device),
            card_indices.to(self.device),
            value_mask=value_mask.to(self.device),
            group_index=group_index.to(self.device),
        )

    def _current_state_terms(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        stage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return taken Q, V, and taken centered A for every replay state."""
        rows: list[Data] = []
        groups: list[int] = []
        value_rows: list[bool] = []
        card_rows: list[tuple[int, int, int]] = []
        taken_local_indices: list[int] = []
        legal_counts: list[int] = []

        for group, (state, action) in enumerate(zip(states, actions)):
            legal = self.env.legal_actions(state)
            legal_counts.append(len(legal))
            taken_local_indices.append(legal.index(action))
            base = self.adapter(state, perspective=state.perspective)
            rows.append(base)
            groups.append(group)
            value_rows.append(True)
            card_rows.append((0, 0, 0))
            for legal_action in legal:
                rows.append(self.builder(base, legal_action, state))
                groups.append(group)
                value_rows.append(False)
                card_rows.append(
                    legal_action.dqn_index(self.env.topology, state)[1:]
                )

        group_index = torch.tensor(groups, dtype=torch.long)
        value_mask = torch.tensor(value_rows, dtype=torch.bool)
        phase = stage[group_index]
        card_indices = torch.tensor(card_rows, dtype=torch.long)
        value, advantage = self._score(
            self.net, rows, phase, card_indices, value_mask, group_index
        )

        action_group_index = group_index[~value_mask].to(self.device)
        q_all = self._combine_q(value, advantage, action_group_index)
        centered_all = q_all - value[action_group_index]

        q_taken = torch.empty(len(states), dtype=torch.float32, device=self.device)
        centered_taken = torch.empty_like(q_taken)
        offset = 0
        for group, count in enumerate(legal_counts):
            taken = offset + taken_local_indices[group]
            q_taken[group] = q_all[taken]
            centered_taken[group] = centered_all[taken]
            offset += count
        return q_taken, value, centered_taken

    def _next_state_terms(
        self,
        next_states: Sequence[State],
        done: torch.Tensor,
        next_stage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return online V(s') and DDQN target Q(s', a*) per transition."""
        v_online = torch.zeros(
            len(next_states), dtype=torch.float32, device=self.device
        )
        q_target_at_astar = torch.zeros_like(v_online)
        rows: list[Data] = []
        groups: list[int] = []
        value_rows: list[bool] = []
        card_rows: list[tuple[int, int, int]] = []
        active_indices: list[int] = []

        for original_index, next_state in enumerate(next_states):
            if bool(done[original_index]):
                continue
            legal = self.env.legal_actions(next_state)
            if not legal:
                continue
            local_group = len(active_indices)
            active_indices.append(original_index)
            base = self.adapter(next_state, perspective=next_state.perspective)
            rows.append(base)
            groups.append(local_group)
            value_rows.append(True)
            card_rows.append((0, 0, 0))
            for action in legal:
                rows.append(self.builder(base, action, next_state))
                groups.append(local_group)
                value_rows.append(False)
                card_rows.append(
                    action.dqn_index(self.env.topology, next_state)[1:]
                )

        if not rows:
            return v_online, q_target_at_astar

        group_index = torch.tensor(groups, dtype=torch.long)
        active_index_tensor = torch.tensor(active_indices, dtype=torch.long)
        phase = next_stage[active_index_tensor][group_index]
        card_indices = torch.tensor(card_rows, dtype=torch.long)
        value_mask = torch.tensor(value_rows, dtype=torch.bool)

        with torch.no_grad():
            value_online, advantage_online = self._score(
                self.net,
                rows,
                phase,
                card_indices,
                value_mask,
                group_index,
            )
            value_target, advantage_target = self._score(
                self.target_net,
                rows,
                phase,
                card_indices,
                value_mask,
                group_index,
            )

        action_group_index = group_index[~value_mask].to(self.device)
        q_online = self._combine_q(
            value_online, advantage_online, action_group_index
        )
        q_target = self._combine_q(
            value_target, advantage_target, action_group_index
        )

        best_online_by_group: dict[int, float] = {}
        best_target_by_group: dict[int, float] = {}
        for group, online_value, target_value in zip(
            action_group_index.tolist(), q_online, q_target
        ):
            online_scalar = float(online_value.item())
            if (
                group not in best_online_by_group
                or online_scalar > best_online_by_group[group]
            ):
                best_online_by_group[group] = online_scalar
                best_target_by_group[group] = float(target_value.item())

        for local_group, original_index in enumerate(active_indices):
            v_online[original_index] = value_online[local_group]
            q_target_at_astar[original_index] = best_target_by_group[local_group]
        return v_online, q_target_at_astar

    def _advantage_loss_terms(
        self,
        td_advantage: torch.Tensor,
        centered_advantage_taken: torch.Tensor,
        q_loss: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Calculate the detached scaled-tanh weight and bounded auxiliary loss."""
        advantage_weight = (
            self.advantage_weight_scale
            * torch.tanh(td_advantage / self.advantage_weight_scale)
        ).detach()
        per_sample_loss = -advantage_weight * centered_advantage_taken
        advantage_loss = per_sample_loss.mean()
        advantage_loss_abs_mean = per_sample_loss.abs().mean()
        cap_coefficient = (
            self.max_advantage_loss_fraction
            * q_loss.detach()
            / (advantage_loss_abs_mean.detach() + self.loss_balance_epsilon)
        )
        effective_coefficient = torch.minimum(
            cap_coefficient,
            q_loss.new_tensor(self.advantage_loss_coef),
        )
        weighted_advantage_loss = effective_coefficient * advantage_loss
        return (
            advantage_weight,
            per_sample_loss,
            advantage_loss,
            advantage_loss_abs_mean,
            effective_coefficient,
            weighted_advantage_loss,
        )

    def _encoder_gradient_diagnostic(
        self, q_loss: torch.Tensor, weighted_advantage_loss: torch.Tensor
    ) -> tuple[float, float, float]:
        """Measure the two encoder gradients without writing parameter ``.grad``."""
        parameters = tuple(self.net.encoder.parameters())
        q_grads = torch.autograd.grad(
            q_loss, parameters, retain_graph=True, allow_unused=True
        )
        advantage_grads = torch.autograd.grad(
            weighted_advantage_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        q_sq = q_loss.new_zeros(())
        advantage_sq = q_loss.new_zeros(())
        dot = q_loss.new_zeros(())
        for q_grad, advantage_grad, parameter in zip(
            q_grads, advantage_grads, parameters
        ):
            q_tensor = torch.zeros_like(parameter) if q_grad is None else q_grad
            advantage_tensor = (
                torch.zeros_like(parameter)
                if advantage_grad is None
                else advantage_grad
            )
            q_sq = q_sq + q_tensor.square().sum()
            advantage_sq = advantage_sq + advantage_tensor.square().sum()
            dot = dot + (q_tensor * advantage_tensor).sum()

        q_norm = q_sq.sqrt()
        advantage_norm = advantage_sq.sqrt()
        if float(q_norm.detach()) == 0.0 or float(advantage_norm.detach()) == 0.0:
            cosine = q_loss.new_zeros(())
        else:
            cosine = dot / (
                q_norm * advantage_norm + self.loss_balance_epsilon
            )
            cosine = cosine.clamp(-1.0, 1.0)
        return (
            float(q_norm.detach()),
            float(advantage_norm.detach()),
            float(cosine.detach()),
        )

    def _correlation(self, left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.detach().float().flatten()
        right = right.detach().float().flatten()
        if left.numel() < 2 or right.numel() < 2:
            return 0.0
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
        if float(denominator) == 0.0:
            return 0.0
        return float((left * right).sum().div(denominator).clamp(-1.0, 1.0))

    def train_step(self, batch_size: int) -> float:
        states, actions, reward, next_states, done, stage, next_stage = (
            self.replay_buffer.sample(batch_size)
        )
        reward = reward.to(self.device)
        done = done.to(self.device)

        q_online_taken, v_online_s, centered_taken = self._current_state_terms(
            states, actions, stage
        )
        v_online_next, q_target_at_astar = self._next_state_terms(
            next_states, done, next_stage
        )
        y = reward + self.gamma * (~done).float() * q_target_at_astar
        q_loss = F.smooth_l1_loss(q_online_taken, y.detach())

        td_advantage = (
            reward + self.gamma * (~done).float() * v_online_next - v_online_s
        )
        (
            advantage_weight,
            _per_sample_loss,
            advantage_loss,
            advantage_loss_abs_mean,
            effective_coefficient,
            weighted_advantage_loss,
        ) = self._advantage_loss_terms(td_advantage, centered_taken, q_loss)
        total_loss = q_loss + weighted_advantage_loss

        self.optimizer.zero_grad()
        gradient_metrics: dict[str, float] = {}
        if (self._train_steps + 1) % self.grad_diagnostic_every == 0:
            q_norm, advantage_norm, cosine = self._encoder_gradient_diagnostic(
                q_loss, weighted_advantage_loss
            )
            gradient_metrics = {
                "adqn_q_encoder_grad_norm": q_norm,
                "adqn_advantage_encoder_grad_norm": advantage_norm,
                "adqn_encoder_gradient_cosine_similarity": cosine,
            }
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), GRAD_CLIP_MAX_NORM
        )
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        detached_q_loss = q_loss.detach()
        detached_weighted = weighted_advantage_loss.detach()
        detached_activity = (
            effective_coefficient.detach() * advantage_loss_abs_mean.detach()
        )
        denominator = detached_q_loss + self.loss_balance_epsilon
        td_error = y - q_online_taken.detach()
        abs_weight = advantage_weight.abs()
        self.last_update_metrics = {
            "adqn_q_loss": float(detached_q_loss),
            "adqn_advantage_loss": float(advantage_loss.detach()),
            "adqn_advantage_loss_abs_mean": float(
                advantage_loss_abs_mean.detach()
            ),
            "adqn_weighted_advantage_loss": float(detached_weighted),
            "adqn_total_loss": float(total_loss.detach()),
            "adqn_advantage_loss_coef": self.advantage_loss_coef,
            "adqn_max_advantage_loss_fraction": self.max_advantage_loss_fraction,
            "adqn_advantage_weight_scale": self.advantage_weight_scale,
            "adqn_effective_advantage_coef": float(
                effective_coefficient.detach()
            ),
            "adqn_effective_advantage_coef_max": float(
                effective_coefficient.detach()
            ),
            "adqn_weighted_advantage_to_q_loss_ratio": float(
                detached_weighted.abs() / denominator
            ),
            "adqn_advantage_activity_to_q_loss_ratio": float(
                detached_activity / denominator
            ),
            "adqn_td_advantage_mean": float(td_advantage.detach().mean()),
            "adqn_td_advantage_abs_mean": float(
                td_advantage.detach().abs().mean()
            ),
            "adqn_td_error_mean": float(td_error.mean()),
            "adqn_td_error_abs_mean": float(td_error.abs().mean()),
            "adqn_q_value_mean": float(q_online_taken.detach().mean()),
            "adqn_target_q_mean": float(y.mean()),
            "adqn_v_online_mean": float(v_online_s.detach().mean()),
            "adqn_v_online_abs_mean": float(v_online_s.detach().abs().mean()),
            "adqn_a_centered_taken_mean": float(centered_taken.detach().mean()),
            "adqn_a_centered_taken_abs_mean": float(
                centered_taken.detach().abs().mean()
            ),
            "adqn_a_centered_taken_max": float(
                centered_taken.detach().abs().max()
            ),
            "adqn_advantage_weight_positive_fraction": float(
                (advantage_weight > 0).float().mean()
            ),
            "adqn_advantage_weight_negative_fraction": float(
                (advantage_weight < 0).float().mean()
            ),
            "adqn_advantage_weight_saturated_fraction": float(
                (
                    abs_weight
                    >= self.advantage_weight_scale
                    * self.advantage_weight_saturation
                )
                .float()
                .mean()
            ),
            "adqn_advantage_weight_td_error_correlation": self._correlation(
                advantage_weight, td_error
            ),
            "adqn_grad_norm": float(grad_norm),
            "adqn_grad_norm_clipped": float(grad_norm > GRAD_CLIP_MAX_NORM),
            **gradient_metrics,
        }
        return float(total_loss.item())


__all__ = ["ADQN_Agent"]
