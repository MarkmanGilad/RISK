"""PPO learner using the same injected-action graphs as the DQN agents."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.environment import Environment
from risk.game.state import State
from risk.learning.action_encoder import ActionEncoder
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.dueling_dqn_agent import resolve_device
from risk.learning.graph_adapter import EDGE_ATTR_DIM, GraphAdapter
from risk.learning.ppo_net import PPO_Net
from risk.learning.rollout_buffer import RolloutBuffer, RolloutTransition
from risk.learning.train_constants import (
    GRAD_CLIP_MAX_NORM,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_LR,
    PPO_MINIBATCH_SIZE,
    PPO_ROLLOUT_LENGTH,
    PPO_TARGET_KL,
    PPO_VALUE_HUBER_BETA,
    PPO_VALUE_LOSS_COEF,
)


class PPO_Agent(BaseAgent):
    """On-policy actor-critic with a fixed-length, boundary-aware rollout."""

    label = "PPO"
    update_metric_weight_key = "ppo_optimizer_steps_per_update"
    unweighted_update_metrics = frozenset(
        {
            "ppo_legal_actions_mean",
            "ppo_forced_action_fraction",
            "ppo_explained_variance",
            "ppo_return_mean",
            "ppo_return_std",
            "ppo_old_value_mean",
            "ppo_old_value_std",
            "ppo_advantage_mean",
            "ppo_advantage_std",
            "ppo_policy_encoder_grad_norm",
            "ppo_value_encoder_grad_norm",
            "ppo_epochs_completed",
            "ppo_early_stopped",
            "ppo_early_stop_kl",
            "ppo_optimizer_steps_per_update",
            "ppo_samples_processed_per_update",
        }
    )

    def __init__(self, player_id: int, env: Environment, *, device: torch.device | None = None,
                 train_mode: bool = False, seed: int | None = None, gamma: float = 0.99,
                 lr: float = PPO_LR, rollout_length: int = PPO_ROLLOUT_LENGTH,
                 target_kl: float = PPO_TARGET_KL) -> None:
        super().__init__(player_id)
        self.env, self.device, self.gamma = env, resolve_device(device), float(gamma)
        self.rollout_length = int(rollout_length)
        self.target_kl = float(target_kl)
        self.epsilon = 0.0  # Evaluator compatibility; PPO never consults it.
        self._rng = random.Random(seed)
        self.rollout_buffer = RolloutBuffer()
        self._pending: tuple[float, float, int] | None = None
        self._train_steps = 0
        self._optimizer_steps = 0
        self._samples_processed = 0
        self._samples_processed_estimated = False
        self.last_update_metrics: dict[str, float] = {}
        self._bind_environment(env)
        sample = self.adapter(env.current_state(), perspective=player_id)
        self.net = PPO_Net(sample.x.shape[1], 64, EDGE_ATTR_DIM, sample.u.shape[1]).to(self.device)
        self.target_net = self.net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.set_train_mode(train_mode)

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    @property
    def samples_processed(self) -> int:
        return self._samples_processed

    def progress_metrics(self) -> dict[str, float]:
        """Cheap rollout-state metrics for the generic trainer logger."""
        rollout_fill = len(self.rollout_buffer)
        return {
            "ppo_rollout_fill": float(rollout_fill),
            "ppo_rollout_fill_fraction": rollout_fill / self.rollout_length,
            "ppo_rollout_updates": float(self._train_steps),
            "ppo_samples_processed_estimated": float(self._samples_processed_estimated),
        }

    def _bind_environment(self, env: Environment) -> None:
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)

    def attach(self, player_id: int, env: Environment) -> None:
        self.player_id, self.env = player_id, env
        self._bind_environment(env)

    def set_train_mode(self, train: bool) -> None:
        self.train_mode = bool(train)
        self.net.train(train)

    def save_params(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load_params(self, path: str | Path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def save_checkpoint(self, dir_path: str | Path) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
                "optimizer_steps": self._optimizer_steps,
                "samples_processed": self._samples_processed,
                "samples_processed_estimated": self._samples_processed_estimated,
            },
            path / "model.pt",
        )

    def load_checkpoint(self, dir_path: str | Path) -> None:
        payload = torch.load(Path(dir_path) / "model.pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._train_steps = int(payload["train_steps"])
        self._optimizer_steps = int(payload.get("optimizer_steps", self._optimizer_steps_from_state()))
        if "samples_processed" in payload:
            self._samples_processed = int(payload["samples_processed"])
            self._samples_processed_estimated = bool(
                payload.get("samples_processed_estimated", False)
            )
        else:
            self._samples_processed = self._optimizer_steps * PPO_MINIBATCH_SIZE
            self._samples_processed_estimated = True
        self.rollout_buffer.clear()

    def _optimizer_steps_from_state(self) -> int:
        """Recover Adam's step count when loading a pre-counter checkpoint."""
        steps = []
        for state in self.optimizer.state.values():
            step = state.get("step")
            if step is not None:
                steps.append(int(step.item() if torch.is_tensor(step) else step))
        return max(steps, default=0)

    def _decision_rows(self, state: State, actions: Sequence[Action], perspective: int) -> tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.adapter(state, perspective=perspective)
        rows = [base] + [self.builder(base, action, state) for action in actions]
        encoded = self.action_encoder.encode_many(actions, state)
        phase = torch.cat([encoded[:1, 0], encoded[:, 0]])
        cards = torch.cat([torch.zeros((1, 3), dtype=torch.long), encoded[:, 1:4]])
        groups = torch.zeros(len(rows), dtype=torch.long)
        value_mask = torch.tensor([True] + [False] * len(actions), dtype=torch.bool)
        return rows, phase, cards, groups, value_mask

    def _forward_actions(self, state: State, actions: Sequence[Action], perspective: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows, phase, cards, groups, value_mask = self._decision_rows(state, actions, perspective)
        return self.net(Batch.from_data_list(rows).to(self.device), phase.to(self.device), cards.to(self.device), groups.to(self.device), value_mask.to(self.device))

    def act(self, events: Sequence[object], state: State) -> Action | None:
        del events
        legal = self.env.legal_actions(state)
        if not legal:
            return None
        with torch.no_grad():
            logits, values = self._forward_actions(state, legal, self.player_id)
            distribution = Categorical(logits=logits)
            index = distribution.sample() if self.train_mode else torch.argmax(logits)
            self._pending = (float(distribution.log_prob(index)), float(values[0]), int(index))
        return legal[int(index)]

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        if self._pending is None:
            raise RuntimeError("PPO_Agent.remember() requires the preceding action from PPO_Agent.act()")
        log_prob, value, index = self._pending
        self._pending = None
        state_copy, next_copy = state.snapshot(), next_state.snapshot()
        state_copy.perspective = self.player_id
        next_copy.perspective = self.player_id
        self.rollout_buffer.push(state_copy, action, index, log_prob, value, reward, next_copy, done)

    def _forward_grouped(self, entries: Sequence[tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch many decisions' rows into one `PPO_Net` forward call.

        `entries`: `(rows, phase, cards, value_mask)` per decision, group id
        implied by position. Returns `(logits, values, action_group_index)`
        where `action_group_index[i]` is which decision `logits[i]` belongs
        to — mirrors `Docs/DuelingDQN.md`'s flattened-minibatch shape, reused
        here so a whole minibatch costs one encoder forward pass, not one
        per transition (`Docs/PPO.md`'s "build once" requirement).
        """
        all_rows: list[Data] = []
        all_phase: list[torch.Tensor] = []
        all_cards: list[torch.Tensor] = []
        all_groups: list[torch.Tensor] = []
        all_value_mask: list[torch.Tensor] = []
        for group_id, (rows, phase, cards, value_mask) in enumerate(entries):
            all_rows.extend(rows)
            all_phase.append(phase)
            all_cards.append(cards)
            all_groups.append(torch.full((len(rows),), group_id, dtype=torch.long))
            all_value_mask.append(value_mask)

        batch = Batch.from_data_list(all_rows).to(self.device)
        phase = torch.cat(all_phase).to(self.device)
        cards = torch.cat(all_cards).to(self.device)
        groups = torch.cat(all_groups).to(self.device)
        value_mask = torch.cat(all_value_mask).to(self.device)

        logits, values = self.net(batch, phase, cards, groups, value_mask)
        action_group_index = groups[~value_mask]
        return logits, values, action_group_index

    def _next_values(self, transitions: Sequence[RolloutTransition]) -> torch.Tensor:
        values = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        entries = []
        positions = []
        for i, transition in enumerate(transitions):
            if transition.done:
                continue
            legal = self.env.legal_actions(transition.next_state)
            if not legal:
                raise RuntimeError("non-terminal PPO next_state has no legal actions")
            rows, phase, cards, _, value_mask = self._decision_rows(transition.next_state, legal, transition.next_state.perspective)
            entries.append((rows, phase, cards, value_mask))
            positions.append(i)
        if entries:
            with torch.no_grad():
                _, group_values, _ = self._forward_grouped(entries)
            for group_id, position in enumerate(positions):
                values[position] = group_values[group_id]
        return values

    def _gae(self, transitions: Sequence[RolloutTransition], next_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.old_value for t in transitions], dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(rewards)
        carry = torch.tensor(0.0, device=self.device)
        for i in range(len(transitions) - 1, -1, -1):
            transition = transitions[i]
            bootstrap = 0.0 if transition.done else next_values[i]
            delta = rewards[i] + self.gamma * bootstrap - old_values[i]
            carry = delta + (0.0 if transition.done or transition.gae_boundary else self.gamma * PPO_GAE_LAMBDA) * carry
            advantages[i] = carry
        return advantages, advantages + old_values

    def _cache_transition_entry(self, transition: RolloutTransition) -> tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one transition's graph rows once, up front, so `learn()`'s
        `PPO_EPOCHS` passes reuse them instead of rebuilding from
        `env.legal_actions`/`GraphAdapter`/`ActionGraphBuilder` every epoch
        (`Docs/PPO.md`: "building the batch once and reusing it across
        epochs is a real, not premature, optimization here"). Also where the
        stored action-index integrity check runs — once, not every epoch."""
        legal = self.env.legal_actions(transition.state)
        if transition.action_index >= len(legal) or legal[transition.action_index] != transition.action:
            raise RuntimeError("stored PPO action no longer matches its legal-action index")
        rows, phase, cards, _, value_mask = self._decision_rows(transition.state, legal, transition.state.perspective)
        return rows, phase, cards, value_mask

    def _evaluate_indices(self, cached_entries: Sequence[tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor]],
                           transitions: Sequence[RolloutTransition], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entries = [cached_entries[i] for i in indices]
        logits, values, action_group_index = self._forward_grouped(entries)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        max_entropies: list[torch.Tensor] = []
        for group_id, i in enumerate(indices):
            group_logits = logits[action_group_index == group_id]
            distribution = Categorical(logits=group_logits)
            action_index = torch.tensor(transitions[i].action_index, device=self.device)
            log_probs.append(distribution.log_prob(action_index))
            entropies.append(distribution.entropy())
            max_entropies.append(group_logits.new_tensor(float(group_logits.numel())).log())
        return torch.stack(log_probs), values, torch.stack(entropies), torch.stack(max_entropies)

    def _k3_approx_kl(self, ratio: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        """Non-negative sampled estimator of KL(old policy || current policy)."""
        return (ratio - 1 - log_ratio).mean()

    def _gradient_norm(self, group: str) -> torch.Tensor:
        """L2 norm for one parameter group after backward, before clipping."""
        gradient_norms: list[torch.Tensor] = []
        for name, parameter in self.net.named_parameters():
            if parameter.grad is None:
                continue
            is_encoder = name.startswith("encoder.")
            is_value = name.startswith("value_head.")
            if group == "encoder" and not is_encoder:
                continue
            if group == "value_head" and not is_value:
                continue
            if group == "policy_heads" and (is_encoder or is_value):
                continue
            gradient_norms.append(parameter.grad.detach().float().norm(2))
        if not gradient_norms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(gradient_norms).norm(2)

    def _detached_gradient_norm(self, gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
        norms = [gradient.detach().float().norm(2) for gradient in gradients if gradient is not None]
        if not norms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(norms).norm(2)

    def _mean_tensor(self, values: Sequence[torch.Tensor]) -> float:
        return float(torch.stack(list(values)).mean())

    def learn(self, *, reached_max_steps: bool = False) -> list[float]:
        if reached_max_steps:
            self.rollout_buffer.mark_last_boundary()
        if len(self.rollout_buffer) < self.rollout_length:
            return []
        transitions = self.rollout_buffer.all()
        cached_entries = [self._cache_transition_entry(t) for t in transitions]
        legal_action_counts = [len(entry[0]) - 1 for entry in cached_entries]
        next_values = self._next_values(transitions)
        advantages, returns = self._gae(transitions, next_values)
        raw_advantages = advantages.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_log_probs = torch.tensor([t.old_log_prob for t in transitions], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.old_value for t in transitions], dtype=torch.float32, device=self.device)
        losses: list[float] = []
        kls: list[float] = []
        clips: list[float] = []
        entropies: list[float] = []
        normalized_entropy_sums: list[torch.Tensor] = []
        normalized_entropy_counts: list[torch.Tensor] = []
        value_mse_losses: list[float] = []
        value_huber_losses: list[float] = []
        policy_losses: list[float] = []
        weighted_value_losses: list[float] = []
        entropy_bonuses: list[float] = []
        total_grad_norms: list[torch.Tensor] = []
        encoder_grad_norms: list[torch.Tensor] = []
        policy_head_grad_norms: list[torch.Tensor] = []
        value_head_grad_norms: list[torch.Tensor] = []
        clipped_gradients: list[torch.Tensor] = []
        policy_encoder_grad_norm = torch.tensor(0.0, device=self.device)
        value_encoder_grad_norm = torch.tensor(0.0, device=self.device)
        measured_component_encoder_grads = False
        completed_epochs = 0
        early_stopped = False
        early_stop_kl = 0.0
        samples_before = self._samples_processed
        optimizer_steps_before = self._optimizer_steps
        for _ in range(PPO_EPOCHS):
            order = torch.randperm(len(transitions), device=self.device)
            for indices_tensor in order.split(PPO_MINIBATCH_SIZE):
                indices = indices_tensor.tolist()
                log_probs, values, entropy, max_entropy = self._evaluate_indices(
                    cached_entries, transitions, indices
                )
                log_ratio = log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                approximate_kl = float(self._k3_approx_kl(ratio, log_ratio).detach())
                # The first minibatch has KL near zero. Later minibatches
                # include drift from the preceding optimizer steps.
                if losses and approximate_kl > self.target_kl:
                    early_stopped = True
                    early_stop_kl = approximate_kl
                    break
                surrogate = torch.minimum(ratio * advantages[indices], ratio.clamp(1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages[indices])
                value_mse = torch.nn.functional.mse_loss(values, returns[indices])
                value_loss = torch.nn.functional.smooth_l1_loss(
                    values, returns[indices], beta=PPO_VALUE_HUBER_BETA
                )
                policy_loss = -surrogate.mean()
                weighted_value_loss = PPO_VALUE_LOSS_COEF * value_loss
                entropy_bonus = PPO_ENTROPY_COEF * entropy.mean()
                actor_loss = policy_loss - entropy_bonus
                loss = actor_loss + weighted_value_loss
                self.optimizer.zero_grad()
                if not measured_component_encoder_grads:
                    encoder_parameters = tuple(self.net.encoder.parameters())
                    policy_encoder_grad_norm = self._detached_gradient_norm(
                        torch.autograd.grad(
                            actor_loss,
                            encoder_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )
                    value_encoder_grad_norm = self._detached_gradient_norm(
                        torch.autograd.grad(
                            weighted_value_loss,
                            encoder_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )
                    measured_component_encoder_grads = True
                loss.backward()
                encoder_grad_norms.append(self._gradient_norm("encoder"))
                policy_head_grad_norms.append(self._gradient_norm("policy_heads"))
                value_head_grad_norms.append(self._gradient_norm("value_head"))
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), GRAD_CLIP_MAX_NORM
                ).detach()
                total_grad_norms.append(total_grad_norm)
                clipped_gradients.append((total_grad_norm > GRAD_CLIP_MAX_NORM).float())
                self.optimizer.step()
                self._optimizer_steps += 1
                self._samples_processed += len(indices)
                losses.append(float(loss.detach()))
                kls.append(approximate_kl)
                clips.append(float((ratio.sub(1).abs() > PPO_CLIP_EPS).float().mean().detach()))
                entropies.append(float(entropy.mean().detach()))
                variable_action_mask = max_entropy > 0
                if variable_action_mask.any():
                    normalized_entropy_sums.append(
                        (entropy[variable_action_mask] / max_entropy[variable_action_mask])
                        .detach()
                        .sum()
                    )
                    normalized_entropy_counts.append(variable_action_mask.sum().detach())
                value_mse_losses.append(float(value_mse.detach()))
                value_huber_losses.append(float(value_loss.detach()))
                policy_losses.append(float(policy_loss.detach()))
                weighted_value_losses.append(float(weighted_value_loss.detach()))
                entropy_bonuses.append(float(entropy_bonus.detach()))
            if early_stopped:
                break
            completed_epochs += 1
        self._train_steps += 1
        explained = 1 - torch.var(returns - old_values) / (torch.var(returns) + 1e-8)
        mean_value_mse = sum(value_mse_losses) / len(value_mse_losses)
        mean_value_huber_loss = sum(value_huber_losses) / len(value_huber_losses)
        self.last_update_metrics = {
            "ppo_approx_kl": sum(kls) / len(kls),
            "ppo_clip_fraction": sum(clips) / len(clips),
            "ppo_entropy": sum(entropies) / len(entropies),
            "ppo_normalized_entropy": (
                float(torch.stack(normalized_entropy_sums).sum()
                      / torch.stack(normalized_entropy_counts).sum())
                if normalized_entropy_sums else 0.0
            ),
            "ppo_legal_actions_mean": sum(legal_action_counts) / len(legal_action_counts),
            "ppo_legal_actions_max": float(max(legal_action_counts)),
            "ppo_forced_action_fraction": (
                sum(count == 1 for count in legal_action_counts) / len(legal_action_counts)
            ),
            "ppo_policy_loss": sum(policy_losses) / len(policy_losses),
            # Keep the historical key as raw MSE so PPO_045 charts remain
            # comparable with PPO_043/044; Huber is the optimized objective.
            "ppo_value_loss": mean_value_mse,
            "ppo_value_mse": mean_value_mse,
            "ppo_value_huber_loss": mean_value_huber_loss,
            "ppo_value_rmse": mean_value_mse ** 0.5,
            "ppo_weighted_value_loss": sum(weighted_value_losses) / len(weighted_value_losses),
            "ppo_entropy_bonus": sum(entropy_bonuses) / len(entropy_bonuses),
            "ppo_explained_variance": float(explained),
            "ppo_return_mean": float(returns.mean()),
            "ppo_return_std": float(returns.std(unbiased=False)),
            "ppo_old_value_mean": float(old_values.mean()),
            "ppo_old_value_std": float(old_values.std(unbiased=False)),
            "ppo_advantage_mean": float(raw_advantages.mean()),
            "ppo_advantage_std": float(raw_advantages.std(unbiased=False)),
            "ppo_grad_norm": self._mean_tensor(total_grad_norms),
            "ppo_grad_norm_max": float(torch.stack(total_grad_norms).max()),
            "ppo_encoder_grad_norm": self._mean_tensor(encoder_grad_norms),
            "ppo_policy_head_grad_norm": self._mean_tensor(policy_head_grad_norms),
            "ppo_value_head_grad_norm": self._mean_tensor(value_head_grad_norms),
            "ppo_policy_encoder_grad_norm": float(policy_encoder_grad_norm),
            "ppo_value_encoder_grad_norm": float(value_encoder_grad_norm),
            "ppo_gradient_clip_fraction": self._mean_tensor(clipped_gradients),
            "ppo_epochs_completed": float(completed_epochs),
            "ppo_early_stopped": float(early_stopped),
            "ppo_early_stop_kl": early_stop_kl,
            "ppo_early_stop_kl_max": early_stop_kl,
            "ppo_optimizer_steps_per_update": float(self._optimizer_steps - optimizer_steps_before),
            "ppo_samples_processed_per_update": float(self._samples_processed - samples_before),
        }
        self.rollout_buffer.clear()
        return losses


__all__ = ["PPO_Agent"]
