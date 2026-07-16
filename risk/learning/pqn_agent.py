"""PQN inference agent for self-play/training rollouts — `Docs/PQN.md`.

Copy of `Dueling_DQN_Agent` (`dueling_dqn_agent.py`) wired to `PQN`
(`pqn.py`) instead of `Dueling_DQN`. `PQN.forward` returns the raw
`(V(s), A(s, a))` streams uncombined — this agent combines them into `Q` via
`_combine_q` wherever the Bellman loss needs it, and reads the policy
straight from `A` via a per-decision softmax, needing no `Q` for that at all
(`Docs/PQN.md` §14: `softmax(Q) == softmax(A)` exactly, so there is no
reason to reconstruct `Q` first just to build the policy).

This class is the game-logic wrapper around `PQN`:

- asks `env.legal_actions(state)`
- builds one graph row per legal action (`ActionGraphBuilder` for graph
  stages; base graph for `TRADE_IN`)
- batches and scores all rows with `PQN`
- samples (training) or argmaxes (evaluation) the resulting per-decision
  categorical policy over `A`, and returns the selected `Action`
"""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from risk.agents.base_agent import BaseAgent
from risk.game.actions import Action
from risk.game.environment import Environment
from risk.game.state import State
from risk.learning.action_encoder import ActionEncoder
from risk.learning.action_graph_builder import ActionGraphBuilder
from risk.learning.graph_adapter import EDGE_ATTR_DIM, GraphAdapter
from risk.learning.pqn import PQN
from risk.learning.replay_buffer import ReplayBuffer
from risk.learning.train_constants import (
    BATCH_SIZE,
    GRAD_CLIP_MAX_NORM,
    PQN_POLICY_LOSS_COEF,
    TRAIN_STEPS_PER_CALL,
)


def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """Return the runtime device: explicit override, else CUDA if available."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class PQN_Agent(BaseAgent):
    """`BaseAgent` wrapper for `PQN` action scoring."""

    label = "PQN"

    def __init__(self, player_id: int, env: Environment, *,
                 replay_buffer: ReplayBuffer | None = None, device: torch.device | None = None,
                 train_mode: bool = False, seed: int | None = None,
                 gamma: float = 0.99, lr: float = 1e-4, target_update_every: int = 1000,) -> None:
        super().__init__(player_id)
        self.env = env
        self.device = resolve_device(device)
        self.epsilon = 0.0  # Evaluator compatibility only; PQN never consults it — it samples the policy, not epsilon-greedy (Docs/PQN.md).
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
        self.net = PQN(
            in_dim=sample.x.shape[1],
            hidden_dim=64,
            edge_dim=EDGE_ATTR_DIM,
            u_dim=sample.u.shape[1],
        ).to(self.device)

        # Kept for Double-DQN target parity with Dueling; synchronized now,
        # updated during training later.
        self.target_net = deepcopy(self.net).to(self.device)
        self.target_net.eval()
        self.set_train_mode(train_mode)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    @property
    def train_steps(self) -> int:
        return self._train_steps

    def progress_metrics(self) -> dict[str, float]:
        """Cheap per-episode diagnostics for the generic trainer logger
        (`Docs/Trainer.md`) — mirrors `Dueling_DQN_Agent`'s, minus
        `epsilon` (permanently inert for PQN, not worth charting)."""
        return {
            "pqn_replay_buffer_size": float(len(self.replay_buffer)),
            "pqn_train_steps_since_target_sync": float(self._train_steps % self.target_update_every),
        }

    def attach(self, player_id: int, env: Environment) -> None:
        """Rebind this agent to a new episode's seat/env — see
        `Dueling_DQN_Agent.attach`'s docstring; identical reasoning."""
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

        No `epsilon` in the payload (unlike `Dueling_DQN_Agent`'s) — it's
        permanently `0.0` and inert for PQN, nothing to restore."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self._train_steps,
            },
            dir_path / "model.pt",
        )
        self.replay_buffer.save(dir_path / "replay.pt")

    def load_checkpoint(self, dir_path: str | Path) -> None:
        """Inverse of `save_checkpoint` — restores net/target/optimizer/
        train_steps and replaces `self.replay_buffer` with the saved one."""
        dir_path = Path(dir_path)
        payload = torch.load(dir_path / "model.pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._train_steps = int(payload["train_steps"])
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer.capacity, path=dir_path / "replay.pt")

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

        _, advantage = self.score_actions(state, legal_actions)
        if self.train_mode:
            index = int(Categorical(logits=advantage).sample().item())
        else:
            index = int(torch.argmax(advantage).item())
        return legal_actions[index]

    def score_actions(self, state: State, legal_actions: Sequence[Action]) -> tuple[torch.Tensor, torch.Tensor]:
        """Score each legal action with the online net.

        Returns `(value, advantage)` uncombined: `value` is a single-element
        tensor (`V(s)`), `advantage` is `[N]` — one `A(s, a)` per legal
        action, in `legal_actions` order (`Docs/PQN.md` §24.A). `act()`
        samples/argmaxes `advantage` directly; nothing here needs `Q`."""
        base = self.adapter(state, perspective=self.player_id)
        rows: list[Data] = [base]
        for action in legal_actions:
            rows.append(self.builder(base, action, state))

        batch = Batch.from_data_list(rows).to(self.device)
        encoded = self.action_encoder.encode_many(legal_actions, state).to(self.device)
        phase = torch.cat([encoded[:1, 0], encoded[:, 0]])
        card_indices = torch.cat([torch.zeros((1, 3), dtype=torch.long, device=self.device), encoded[:, 1:4]])
        group_index = torch.zeros(len(rows), dtype=torch.long, device=self.device)
        value_mask = torch.zeros(len(rows), dtype=torch.bool, device=self.device)
        value_mask[0] = True

        with torch.no_grad():
            value, advantage = self.net(batch, phase, card_indices, value_mask=value_mask, group_index=group_index)
        return value, advantage

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool,) -> None:
        """Snapshot and store one transition — same snapshot-and-tag
        pattern as `Dueling_DQN_Agent.remember`; see its docstring."""
        state_snapshot = state.snapshot()
        next_state_snapshot = next_state.snapshot()
        state_snapshot.perspective = self.player_id
        next_state_snapshot.perspective = self.player_id
        self.replay_buffer.push(state_snapshot, action, reward, next_state_snapshot, done)

    def can_train(self) -> bool:
        return len(self.replay_buffer) >= BATCH_SIZE

    def learn_steps(self) -> list[float]:
        losses: list[float] = []
        for _ in range(max(0, TRAIN_STEPS_PER_CALL)):
            loss = self.train_step(BATCH_SIZE)
            losses.append(loss)
        return losses

    def learn(self, *, reached_max_steps: bool = False) -> list[float]:
        """Run PQN updates; the cutoff flag is reserved for on-policy agents."""
        del reached_max_steps
        if not self.can_train():
            return []
        return self.learn_steps()

    def _combine_q(self, value: torch.Tensor, advantage: torch.Tensor,
                    group_index: torch.Tensor) -> torch.Tensor:
        """`Q(s, a_i) = V(s) + A(s, a_i) - mean_j(A(s, a_j))`, grouped per
        decision — the same `scatter(..., reduce="mean")` pattern already
        validated in `dueling_dqn.py`/`ppo_net.py`'s `_forward_grouped`.
        Kept in this one place so `train_step`'s three call sites (current
        state, next state online, next state target) never each rewrite
        the grouped mean themselves — the old, removed `value_mask is None`
        fallback (`Docs/DuelingDQN.md`'s dead-code review) got exactly this
        formula wrong once already by averaging over the wrong rows."""
        n_groups = value.shape[0]
        advantage_mean = scatter(advantage, group_index, dim=0, dim_size=n_groups, reduce="mean")
        return value[group_index] + advantage - advantage_mean[group_index]

    def _score(self, net: torch.nn.Module, rows: list[Data], phase: torch.Tensor,
               card_indices: torch.Tensor, value_mask: torch.Tensor,
               group_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch `rows` and score them with `net`, returning the raw
        `(value, advantage)` streams — shared by the current-state and
        next-state passes below, same split as `Dueling_DQN_Agent._score`."""
        batch = Batch.from_data_list(rows).to(self.device)
        return net(
            batch,
            phase.to(self.device),
            card_indices.to(self.device),
            value_mask=value_mask.to(self.device),
            group_index=group_index.to(self.device),
        )

    def _current_state_terms(self, states: Sequence[State], actions: Sequence[Action],
                              stage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`Docs/PQN.md` §24.B's "current state" calculation, via the online
        net: for each sampled `(state, action)` pair, the taken action's
        combined `Q_online(s, a)` (Bellman prediction), the clean-row
        `V_online(s)` (TD-advantage baseline), and `log pi(a | s)` (policy
        term, from a per-decision softmax over `A`). Builds every legal
        action of every sampled state in one flattened batch, exactly like
        `Dueling_DQN_Agent._q_value` — the dueling mean needs the whole
        legal-action set, not just the taken action.
        """
        rows: list[Data] = []
        groups: list[int] = []
        value_rows: list[bool] = []
        card_rows: list[tuple[int, int, int]] = []
        taken_local_index: list[int] = []
        legal_counts: list[int] = []
        for i, (s, a) in enumerate(zip(states, actions)):
            legal = self.env.legal_actions(s)
            legal_counts.append(len(legal))
            base = self.adapter(s, perspective=s.perspective)
            rows.append(base)
            groups.append(i)
            value_rows.append(True)
            card_rows.append((0, 0, 0))
            taken_local_index.append(legal.index(a))
            for legal_action in legal:
                rows.append(self.builder(base, legal_action, s))
                groups.append(i)
                value_rows.append(False)
                card_rows.append(legal_action.dqn_index(self.env.topology, s)[1:])

        group_index = torch.tensor(groups, dtype=torch.long)
        phase = stage[group_index]
        card_indices = torch.tensor(card_rows, dtype=torch.long)
        value_mask = torch.tensor(value_rows, dtype=torch.bool)
        value, advantage = self._score(self.net, rows, phase, card_indices, value_mask, group_index)

        action_group_index = group_index[~value_mask].to(self.device)
        q_all = self._combine_q(value, advantage, action_group_index)

        q_taken = torch.empty(len(states), dtype=torch.float32, device=self.device)
        log_pi_taken = torch.empty(len(states), dtype=torch.float32, device=self.device)
        offset = 0
        for i in range(len(states)):
            k = legal_counts[i]
            group_slice = slice(offset, offset + k)
            offset += k
            q_taken[i] = q_all[group_slice][taken_local_index[i]]
            distribution = Categorical(logits=advantage[group_slice])
            log_pi_taken[i] = distribution.log_prob(
                torch.tensor(taken_local_index[i], device=self.device)
            )
        return q_taken, value, log_pi_taken

    def _next_state_terms(self, next_states: Sequence[State], done: torch.Tensor,
                           next_stage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`Docs/PQN.md` §24.B's "next state" calculation: `V_online(s')`
        (policy baseline, zero for terminals/no-legal-actions) and the
        Double-DQN `Q_target(s', a*)` bootstrap — online net selects `a*`,
        target net evaluates it, same online-selects/target-evaluates split
        as `Dueling_DQN_Agent._max_next_ddqn_q`, just built from the raw
        `(V, A)` streams instead of an already-fused `Q`.
        """
        v_online = torch.zeros(len(next_states), dtype=torch.float32, device=self.device)
        q_target_at_astar = torch.zeros(len(next_states), dtype=torch.float32, device=self.device)

        rows: list[Data] = []
        groups: list[int] = []
        value_rows: list[bool] = []
        card_rows: list[tuple[int, int, int]] = []
        active_indices: list[int] = []
        for i, ns in enumerate(next_states):
            if bool(done[i]):
                continue
            legal = self.env.legal_actions(ns)
            if not legal:
                continue
            local_group = len(active_indices)
            active_indices.append(i)
            base = self.adapter(ns, perspective=ns.perspective)
            rows.append(base)
            groups.append(local_group)
            value_rows.append(True)
            card_rows.append((0, 0, 0))
            for a in legal:
                rows.append(self.builder(base, a, ns))
                groups.append(local_group)
                value_rows.append(False)
                card_rows.append(a.dqn_index(self.env.topology, ns)[1:])

        if not rows:
            return v_online, q_target_at_astar

        group_index = torch.tensor(groups, dtype=torch.long)
        active_next_stage = next_stage[torch.tensor(active_indices, dtype=torch.long)]
        phase = active_next_stage[group_index]
        card_indices = torch.tensor(card_rows, dtype=torch.long)
        value_mask = torch.tensor(value_rows, dtype=torch.bool)

        with torch.no_grad():
            value_online, advantage_online = self._score(
                self.net, rows, phase, card_indices, value_mask, group_index
            )
            value_target, advantage_target = self._score(
                self.target_net, rows, phase, card_indices, value_mask, group_index
            )

        action_group_index = group_index[~value_mask].to(self.device)
        q_online = self._combine_q(value_online, advantage_online, action_group_index)
        q_target = self._combine_q(value_target, advantage_target, action_group_index)

        best_online_by_group: dict[int, float] = {}
        best_target_by_group: dict[int, float] = {}
        for group, online_value, target_value in zip(action_group_index.tolist(), q_online, q_target):
            online_scalar = float(online_value.item())
            if group not in best_online_by_group or online_scalar > best_online_by_group[group]:
                best_online_by_group[group] = online_scalar
                best_target_by_group[group] = float(target_value.item())

        for local_group, original_index in enumerate(active_indices):
            v_online[original_index] = value_online[local_group]
            q_target_at_astar[original_index] = best_target_by_group[local_group]

        return v_online, q_target_at_astar

    def _policy_loss(self, td_advantage: torch.Tensor,
                     log_pi_taken: torch.Tensor) -> torch.Tensor:
        """Return replay-weighted policy loss with a fixed TD weight."""
        return torch.mean(-td_advantage.detach() * log_pi_taken)

    def train_step(self, batch_size: int) -> float:
        """One PQN update from a sampled replay minibatch.

        Bellman loss unchanged from Dueling DQN (`Docs/DuelingDQN.md`'s
        "Loss" section): Smooth L1 against the Double-DQN target. Adds a
        replay-based policy-improvement term (`Docs/PQN.md` §24.C):
        `td_advantage` is detached before weighting `-log pi(a|s)`, so
        policy-loss gradients never flow into either value estimate — the
        value stream only ever learns through `q_loss`.
        """
        states, actions, reward, next_states, done, stage, next_stage = (
            self.replay_buffer.sample(batch_size)
        )
        reward = reward.to(self.device)
        done = done.to(self.device)

        q_online_taken, v_online_s, log_pi_taken = self._current_state_terms(states, actions, stage)
        v_online_next, q_target_at_astar = self._next_state_terms(next_states, done, next_stage)

        y = reward + self.gamma * (~done).float() * q_target_at_astar
        q_loss = F.smooth_l1_loss(q_online_taken, y.detach())

        td_advantage = reward + self.gamma * (~done).float() * v_online_next - v_online_s
        policy_loss = self._policy_loss(td_advantage, log_pi_taken)

        loss = q_loss + PQN_POLICY_LOSS_COEF * policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP_MAX_NORM)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        td_error = y - q_online_taken.detach()
        self.last_update_metrics = {
            "pqn_q_loss": float(q_loss.detach()),
            "pqn_policy_loss": float(policy_loss.detach()),
            "pqn_total_loss": float(loss.detach()),
            "pqn_td_error_mean": float(td_error.mean()),
            "pqn_td_error_abs_mean": float(td_error.abs().mean()),
            "pqn_td_advantage_mean": float(td_advantage.detach().mean()),
            "pqn_td_advantage_abs_mean": float(td_advantage.detach().abs().mean()),
            "pqn_q_value_mean": float(q_online_taken.detach().mean()),
            "pqn_value_mean": float(v_online_s.detach().mean()),
            "pqn_target_q_mean": float(y.mean()),
            "pqn_grad_norm": float(grad_norm),
            "pqn_grad_norm_clipped": float(grad_norm > GRAD_CLIP_MAX_NORM),
        }

        return float(loss.item())


__all__ = ["PQN_Agent"]
