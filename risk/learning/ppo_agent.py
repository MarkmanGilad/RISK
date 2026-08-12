"""PPO learner using the same injected-action graphs as the DQN agents."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
    PPO_LR,
    PPO_MINIBATCH_SIZE,
    PPO_N_STEP,
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
            "ppo_target_horizon_mean",
            "ppo_target_bootstrap_fraction",
            "ppo_policy_encoder_grad_norm",
            "ppo_value_encoder_grad_norm",
            "ppo_value_to_policy_encoder_grad_ratio",
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
        """Initialize PPO state, network, optimizer, and environment adapters."""
        super().__init__(player_id)
        self.env, self.device, self.gamma = env, resolve_device(device), float(gamma)
        if PPO_N_STEP < 1:
            raise ValueError("PPO_N_STEP must be at least 1")
        self.rollout_length = int(rollout_length)
        self.target_kl = float(target_kl)
        self.epsilon = 0.0  # Evaluator compatibility; PPO never consults it.
        self.rollout_buffer = RolloutBuffer()
        self._pending: tuple[float, float, int] | None = None
        self.train_steps = 0
        self.optimizer_steps = 0
        self.samples_processed = 0
        self._samples_processed_estimated = False
        self.last_update_metrics: dict[str, float] = {}
        self._bind_environment(env)
        sample = self.adapter(env.current_state(), perspective=player_id)
        self.net = PPO_Net(sample.x.shape[1], 64, EDGE_ATTR_DIM, sample.u.shape[1]).to(self.device)
        self.target_net = self.net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.set_train_mode(train_mode)

    def progress_metrics(self) -> dict[str, float]:
        """Cheap rollout-state metrics for the generic trainer logger."""
        rollout_fill = len(self.rollout_buffer)
        return {
            "ppo_rollout_fill": float(rollout_fill),
            "ppo_rollout_fill_fraction": rollout_fill / self.rollout_length,
            "ppo_rollout_updates": float(self.train_steps),
            "ppo_samples_processed_estimated": float(self._samples_processed_estimated),
        }

    def _bind_environment(self, env: Environment) -> None:
        """Create graph and action encoders bound to this environment."""
        self.adapter = GraphAdapter(env.topology, env.settings)
        self.builder = ActionGraphBuilder(env.topology)
        self.action_encoder = ActionEncoder(env)

    def attach(self, player_id: int, env: Environment) -> None:
        """Attach the agent to a player and refresh environment-specific helpers."""
        self.player_id, self.env = player_id, env
        self._bind_environment(env)

    def set_train_mode(self, train: bool) -> None:
        """Set stochastic action selection and network training mode."""
        self.train_mode = bool(train)
        self.net.train(train)

    def save_params(self, path: str | Path) -> None:
        """Save only the policy-value network parameters."""
        torch.save(self.net.state_dict(), path)

    def load_params(self, path: str | Path) -> None:
        """Load only the policy-value network parameters."""
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def save_checkpoint(self, dir_path: str | Path) -> None:
        """Save network, optimizer, and PPO progress counters."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "optimizer_steps": self.optimizer_steps,
                "samples_processed": self.samples_processed,
                "samples_processed_estimated": self._samples_processed_estimated,
            },
            path / "model.pt",
        )

    def load_checkpoint(self, dir_path: str | Path) -> None:
        """Restore network, optimizer, and PPO progress counters."""
        payload = torch.load(Path(dir_path) / "model.pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.train_steps = int(payload["train_steps"])
        self.optimizer_steps = int(payload.get("optimizer_steps", self._optimizer_steps_from_state()))
        if "samples_processed" in payload:
            self.samples_processed = int(payload["samples_processed"])
            self._samples_processed_estimated = bool(
                payload.get("samples_processed_estimated", False)
            )
        else:
            self.samples_processed = self.optimizer_steps * PPO_MINIBATCH_SIZE
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
        """Build graph-network inputs for one state and its legal actions."""
        base = self.adapter(state, perspective=perspective)
        rows = [base] + [self.builder(base, action, state) for action in actions]
        encoded = self.action_encoder.encode_many(actions, state)
        phase = torch.cat([encoded[:1, 0], encoded[:, 0]])
        cards = torch.cat([torch.zeros((1, 3), dtype=torch.long), encoded[:, 1:4]])
        groups = torch.zeros(len(rows), dtype=torch.long)
        value_mask = torch.tensor([True] + [False] * len(actions), dtype=torch.bool)
        return rows, phase, cards, groups, value_mask

    def _forward_actions(self, state: State, actions: Sequence[Action], perspective: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate action logits and the state value for one decision."""
        rows, phase, cards, groups, value_mask = self._decision_rows(state, actions, perspective)
        return self.net(Batch.from_data_list(rows).to(self.device), phase.to(self.device), cards.to(self.device), groups.to(self.device), value_mask.to(self.device))

    def act(self, events: Sequence[object], state: State) -> Action | None:
        """Select an action and cache its policy data for the rollout."""
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
        """Store the completed decision using the data cached by `act()`."""
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

    def _clean_value_entry(self, state: State, perspective: int) -> tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one clean graph row for a value-only bootstrap evaluation."""
        base = self.adapter(state, perspective=perspective)
        return [base], torch.zeros(1, dtype=torch.long), torch.zeros((1, 3), dtype=torch.long), torch.tensor([True], dtype=torch.bool)

    def _boundary_values(self, transitions: Sequence[RolloutTransition]) -> torch.Tensor:
        """Evaluate only non-terminal cutoff and rollout-tail bootstrap states."""
        values = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        entries = []
        positions = []
        for i, transition in enumerate(transitions):
            if transition.done or (not transition.gae_boundary and i != len(transitions) - 1):
                continue
            entries.append(self._clean_value_entry(transition.next_state, self.player_id))
            positions.append(i)
        if entries:
            with torch.no_grad():
                _, group_values, _ = self._forward_grouped(entries)
            for group_id, position in enumerate(positions):
                values[position] = group_values[group_id]
        return values

    def _n_step_targets(self, transitions: Sequence[RolloutTransition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fixed-horizon targets using stored values and boundary bootstraps."""
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.old_value for t in transitions], dtype=torch.float32, device=self.device)
        targets = torch.zeros_like(rewards)
        horizons = torch.zeros_like(rewards)
        uses_bootstrap = torch.zeros(len(transitions), dtype=torch.bool, device=self.device)
        boundary_values = self._boundary_values(transitions)
        for start in range(len(transitions)):
            discount = 1.0
            for offset in range(PPO_N_STEP):
                index = start + offset
                if index >= len(transitions):
                    break
                transition = transitions[index]
                targets[start] += discount * rewards[index]
                horizon = offset + 1
                horizons[start] = horizon
                discount *= self.gamma

                if transition.done:
                    break
                if transition.gae_boundary or index == len(transitions) - 1:
                    targets[start] += discount * boundary_values[index]
                    uses_bootstrap[start] = True
                    break
                if horizon == PPO_N_STEP:
                    targets[start] += discount * old_values[index + 1]
                    uses_bootstrap[start] = True
                    break
        return targets - old_values, targets, horizons, uses_bootstrap

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
        """Evaluate log-probabilities, values, and entropies for a minibatch."""
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
        """Return the detached combined L2 norm of optional gradients."""
        norms = [gradient.detach().float().norm(2) for gradient in gradients if gradient is not None]
        if not norms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(norms).norm(2)

    def _value_to_policy_encoder_grad_ratio(self, value_norm: float, policy_norm: float) -> float:
        """Return the finite critic-to-actor encoder-gradient ratio."""
        return value_norm / max(policy_norm, 1e-12)

    def _run_minibatch(self, cached_entries: Sequence[tuple[list[Data], torch.Tensor, torch.Tensor, torch.Tensor]],
            transitions: Sequence[RolloutTransition], indices: Sequence[int], old_log_probs: torch.Tensor,
            advantages: torch.Tensor, returns: torch.Tensor, *, has_previous_step: bool, ) -> tuple[dict[str, float] | None, float]:
        """Run one PPO optimizer step, or report its pre-step KL early stop."""
        idx = list(indices)
        log_probs, values, entropy, max_entropy = self._evaluate_indices(cached_entries, transitions, idx)
        log_ratio = log_probs - old_log_probs[idx]
        ratio = log_ratio.exp()
        approximate_kl = float(self._k3_approx_kl(ratio, log_ratio).detach())
        if has_previous_step and approximate_kl > self.target_kl:
            return None, approximate_kl

        surrogate = torch.minimum(ratio * advantages[idx], ratio.clamp(1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages[idx], )
        value_mse = torch.nn.functional.mse_loss(values, returns[idx])
        value_loss = torch.nn.functional.smooth_l1_loss(values, returns[idx], beta=PPO_VALUE_HUBER_BETA)
        policy_loss = -surrogate.mean()
        weighted_value_loss = PPO_VALUE_LOSS_COEF * value_loss
        entropy_bonus = PPO_ENTROPY_COEF * entropy.mean()
        actor_loss = policy_loss - entropy_bonus
        loss = actor_loss + weighted_value_loss

        self.optimizer.zero_grad()
        # region Encoder-gradient diagnostics
        encoder_parameters = tuple(self.net.encoder.parameters())
        policy_encoder_grad_norm = self._detached_gradient_norm(torch.autograd.grad(actor_loss, encoder_parameters, retain_graph=True, allow_unused=True))
        value_encoder_grad_norm = self._detached_gradient_norm(torch.autograd.grad(weighted_value_loss, encoder_parameters, retain_graph=True, allow_unused=True))
        # endregion
        loss.backward()
        # region Post-backward gradient diagnostics
        encoder_grad_norm = self._gradient_norm("encoder")
        policy_head_grad_norm = self._gradient_norm("policy_heads")
        value_head_grad_norm = self._gradient_norm("value_head")
        # endregion
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP_MAX_NORM).detach()
        self.optimizer.step()
        self.optimizer_steps += 1
        self.samples_processed += len(idx)

        # region Minibatch logging metrics
        variable_action_mask = max_entropy > 0
        normalized_entropy_sum = (float((entropy[variable_action_mask] / max_entropy[variable_action_mask]).detach().sum()) if variable_action_mask.any() else 0.0 )
        return {
            "loss": float(loss.detach()),
            "approximate_kl": approximate_kl,
            "clip_fraction": float((ratio.sub(1).abs() > PPO_CLIP_EPS).float().mean().detach()),
            "entropy": float(entropy.mean().detach()),
            "normalized_entropy_sum": normalized_entropy_sum,
            "normalized_entropy_count": float(variable_action_mask.sum()),
            "value_mse": float(value_mse.detach()),
            "value_huber_loss": float(value_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "weighted_value_loss": float(weighted_value_loss.detach()),
            "entropy_bonus": float(entropy_bonus.detach()),
            "grad_norm": float(total_grad_norm),
            "encoder_grad_norm": float(encoder_grad_norm),
            "policy_head_grad_norm": float(policy_head_grad_norm),
            "value_head_grad_norm": float(value_head_grad_norm),
            "policy_encoder_grad_norm": float(policy_encoder_grad_norm),
            "value_encoder_grad_norm": float(value_encoder_grad_norm),
            "gradient_clipped": float(total_grad_norm > GRAD_CLIP_MAX_NORM),
            "sample_count": float(len(idx)),
        }, approximate_kl
        # endregion

    def _mean_minibatch_metric(self, minibatches: Sequence[dict[str, float]], name: str) -> float:
        """Return the unweighted mean for one recorded minibatch metric."""
        return sum(minibatch[name] for minibatch in minibatches) / len(minibatches)

    def _sample_weighted_minibatch_metric(self, minibatches: Sequence[dict[str, float]], name: str) -> float:
        """Return the sample-weighted mean for one minibatch metric."""
        total_samples = sum(minibatch["sample_count"] for minibatch in minibatches)
        return sum(minibatch[name] * minibatch["sample_count"] for minibatch in minibatches) / total_samples

    def _summarize_update(self, minibatches: Sequence[dict[str, float]], *, legal_action_counts: Sequence[int], returns: torch.Tensor,
            old_values: torch.Tensor, raw_advantages: torch.Tensor, completed_epochs: int, early_stopped: bool, early_stop_kl: float, optimizer_steps_before: int,
            samples_before: int, target_horizons: torch.Tensor, target_bootstraps: torch.Tensor, ) -> dict[str, float]:
        """Preserve the public PPO metric schema while hiding collection detail."""
        value_mse = self._mean_minibatch_metric(minibatches, "value_mse")
        policy_encoder_grad_norm = self._sample_weighted_minibatch_metric(minibatches, "policy_encoder_grad_norm")
        value_encoder_grad_norm = self._sample_weighted_minibatch_metric(minibatches, "value_encoder_grad_norm")
        normalized_entropy_count = sum(minibatch["normalized_entropy_count"] for minibatch in minibatches)
        explained = 1 - torch.var(returns - old_values) / (torch.var(returns) + 1e-8)
        return {
            "ppo_approx_kl": self._mean_minibatch_metric(minibatches, "approximate_kl"),
            "ppo_clip_fraction": self._mean_minibatch_metric(minibatches, "clip_fraction"),
            "ppo_entropy": self._mean_minibatch_metric(minibatches, "entropy"),
            "ppo_normalized_entropy": (sum(minibatch["normalized_entropy_sum"] for minibatch in minibatches) / normalized_entropy_count if normalized_entropy_count else 0.0),
            "ppo_legal_actions_mean": sum(legal_action_counts) / len(legal_action_counts),
            "ppo_legal_actions_max": float(max(legal_action_counts)),
            "ppo_forced_action_fraction": (sum(count == 1 for count in legal_action_counts) / len(legal_action_counts)),
            "ppo_policy_loss": self._mean_minibatch_metric(minibatches, "policy_loss"),
            # Keep the historical key as raw MSE so PPO_045 charts remain
            # comparable with PPO_043/044; Huber is the optimized objective.
            "ppo_value_loss": value_mse,
            "ppo_value_mse": value_mse,
            "ppo_value_huber_loss": self._mean_minibatch_metric(minibatches, "value_huber_loss"),
            "ppo_value_rmse": value_mse ** 0.5,
            "ppo_weighted_value_loss": self._mean_minibatch_metric(minibatches, "weighted_value_loss"),
            "ppo_entropy_bonus": self._mean_minibatch_metric(minibatches, "entropy_bonus"),
            "ppo_explained_variance": float(explained),
            "ppo_return_mean": float(returns.mean()),
            "ppo_return_std": float(returns.std(unbiased=False)),
            "ppo_old_value_mean": float(old_values.mean()),
            "ppo_old_value_std": float(old_values.std(unbiased=False)),
            "ppo_advantage_mean": float(raw_advantages.mean()),
            "ppo_advantage_std": float(raw_advantages.std(unbiased=False)),
            "ppo_target_horizon_mean": float(target_horizons.mean()),
            "ppo_target_bootstrap_fraction": float(target_bootstraps.float().mean()),
            "ppo_grad_norm": self._mean_minibatch_metric(minibatches, "grad_norm"),
            "ppo_grad_norm_max": max(minibatch["grad_norm"] for minibatch in minibatches),
            "ppo_encoder_grad_norm": self._mean_minibatch_metric(minibatches, "encoder_grad_norm"),
            "ppo_policy_head_grad_norm": self._mean_minibatch_metric(minibatches, "policy_head_grad_norm"),
            "ppo_value_head_grad_norm": self._mean_minibatch_metric(minibatches, "value_head_grad_norm"),
            "ppo_policy_encoder_grad_norm": policy_encoder_grad_norm,
            "ppo_value_encoder_grad_norm": value_encoder_grad_norm,
            "ppo_value_to_policy_encoder_grad_ratio": (self._value_to_policy_encoder_grad_ratio(value_encoder_grad_norm, policy_encoder_grad_norm)),
            "ppo_gradient_clip_fraction": self._mean_minibatch_metric(minibatches, "gradient_clipped"),
            "ppo_epochs_completed": float(completed_epochs),
            "ppo_early_stopped": float(early_stopped),
            "ppo_early_stop_kl": early_stop_kl,
            "ppo_early_stop_kl_max": early_stop_kl,
            "ppo_optimizer_steps_per_update": float(self.optimizer_steps - optimizer_steps_before),
            "ppo_samples_processed_per_update": float(self.samples_processed - samples_before),
        }

    def _prepare_rollout_update(self, transitions: Sequence[RolloutTransition]) -> dict[str, list | torch.Tensor]:
        """Build the fixed rollout targets and cached inputs for one PPO update."""
        advantages, returns, target_horizons, target_bootstraps = self._n_step_targets(transitions)
        cached_entries = [self._cache_transition_entry(transition) for transition in transitions]
        raw_advantages = advantages.clone()

        return {
            "cached_entries": cached_entries,
            "legal_action_counts": [len(entry[0]) - 1 for entry in cached_entries],
            "advantages": (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8),
            "returns": returns,
            "raw_advantages": raw_advantages,
            "target_horizons": target_horizons,
            "target_bootstraps": target_bootstraps,
            "old_log_probs": torch.tensor([transition.old_log_prob for transition in transitions], dtype=torch.float32, device=self.device, ),
            "old_values": torch.tensor([transition.old_value for transition in transitions], dtype=torch.float32, device=self.device, ),
        }

    def _run_update_epochs(self, transitions: Sequence[RolloutTransition], update: dict[str, list | torch.Tensor]) -> tuple[list[dict[str, float]], int, bool, float]:
        """Run PPO epochs until completion or the KL guard stops the update."""
        minibatches: list[dict[str, float]] = []
        completed_epochs = 0
        early_stopped = False
        early_stop_kl = 0.0

        for _ in range(PPO_EPOCHS):
            order = torch.randperm(len(transitions), device=self.device)
            for indices_tensor in order.split(PPO_MINIBATCH_SIZE):
                minibatch, approximate_kl = self._run_minibatch(update["cached_entries"], transitions, indices_tensor.tolist(), update["old_log_probs"],
                    update["advantages"], update["returns"], has_previous_step=bool(minibatches), )
                if minibatch is None:
                    early_stopped = True
                    early_stop_kl = approximate_kl
                    break
                minibatches.append(minibatch)
            if early_stopped:
                break
            completed_epochs += 1

        return minibatches, completed_epochs, early_stopped, early_stop_kl

    def learn(self, *, reached_max_steps: bool = False) -> list[float]:
        """Run one complete PPO rollout update when enough turns are collected."""
        if reached_max_steps:
            self.rollout_buffer.mark_last_boundary()
        if len(self.rollout_buffer) < self.rollout_length:
            return []

        transitions = self.rollout_buffer.all()
        update = self._prepare_rollout_update(transitions)
        samples_before = self.samples_processed
        optimizer_steps_before = self.optimizer_steps
        minibatches, completed_epochs, early_stopped, early_stop_kl = self._run_update_epochs(transitions, update)

        self.train_steps += 1
        self.last_update_metrics = self._summarize_update(minibatches, legal_action_counts=update["legal_action_counts"], returns=update["returns"],
            old_values=update["old_values"], raw_advantages=update["raw_advantages"], completed_epochs=completed_epochs, early_stopped=early_stopped,
            early_stop_kl=early_stop_kl, optimizer_steps_before=optimizer_steps_before, samples_before=samples_before,
            target_horizons=update["target_horizons"], target_bootstraps=update["target_bootstraps"],)
        self.rollout_buffer.clear()
        return [minibatch["loss"] for minibatch in minibatches]


__all__ = ["PPO_Agent"]
