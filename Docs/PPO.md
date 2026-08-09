# PPO build plan — injected-action PPO

Goal: add a third learner, **`PPO_Agent`**, alongside `GNN_DQN_Agent` and
`Dueling_DQN_Agent`. The comparison roadmap is DQN, Dueling DQN, PPO, and
PQN, as described in `Docs/NetworkArchitectures.md` and `Docs/PQN.md`. PPO
uses the same per-action graph injection that `GNN_DQN` and `Dueling_DQN`
already use.

**Status: implemented as standalone PPO modules, reviewed and smoke-tested.**
`risk/learning/ppo_net.py`, `ppo_agent.py`, and `rollout_buffer.py` provide
the injected-action network, agent, and ordered on-policy storage. PPO
defaults live with every other learner setting in `train_constants.py`.
`learn()` builds and validates every
rollout transition's graph once per update (`_cache_transition_entry`) and
batches whole minibatches into a single `PPO_Net` forward call
(`_forward_grouped`) rather than rebuilding/forwarding one transition at a
time — the earlier draft did the latter, which cost thousands of unbatched
forward passes per update; fixed and verified with an ad-hoc smoke script
(loss decreases within an update, weights change, checkpoint round-trips).
`Temp/tests/test_ppo.py` covers its focused regression cases. PPO can be
selected by `build_learner_agent("PPO", ctx)`; the shared trainer aggregates
its update diagnostics without a PPO-specific branch.

**Current recommendation:** keep this as a plan until the Dueling run has at
least one meaningful comparison window. PPO is a larger change than Dueling: it
keeps the public agent interface, but changes the data contract from off-policy
replay to on-policy rollouts.

**Settled:** action injection is the only action representation used by the
learning roadmap. This doc assumes that decision and focuses only on what PPO
changes: the policy/value objective and the on-policy rollout update.

---

## Design decision

Reuse the pieces already validated by `Dueling_DQN`/`Dueling_DQN_Agent`: the
same `GraphAdapter` base graph, `ActionGraphBuilder` injection, `Encoder`,
per-phase heads, and clean-row-plus-injected-rows batch shape
(`Docs/DuelingDQN.md`'s "Value stream"/`value_mask` section). The real changes
from Dueling are the **loss** and the **meaning of the per-phase heads**:
PPO uses clipped surrogate + value loss + entropy instead of Bellman TD, and
the phase heads output policy logits instead of advantages. The graph batch
construction is already solved.

The same minimal-diff policy from Dueling applies here: copy first, then change
only what PPO's objective requires. Do not touch `GNN_DQN_Agent` or
`Dueling_DQN_Agent`. The only planned shared-loop changes are generic ones:
notifying an agent when an episode ends and forwarding optional agent-supplied
update diagnostics; neither may branch on PPO.

**Comment:** this is the right first PPO variant for this codebase because it
keeps the state/action representation fixed and changes only the learning
objective. If PPO behaves differently, the main reason should be the on-policy
policy-gradient update, not a new action encoder or a new trainer.

---

## Keep PPO-specific behavior inside the agent

PPO is on-policy: it collects a fixed-length rollout, runs several epochs of
minibatch updates over that exact rollout, then discards it. That rhythm is
different from DQN's "one gradient step per learner turn from a random replay
sample." The goal is to hide that difference from `Trainer` and keep the same
duck-typed call sites that `Trainer.train()` already uses:

```python
self.agent.remember(state, action, reward_total, next_state, done)
step_losses = self.agent.learn(reached_max_steps=reached_max_steps)
```

`learn()` receives only the dynamic max-step boundary flag
(`Docs/Trainer.md`'s required-interface list); `PPO_Agent` reads its own
training constants, same as the other two:

1. **`remember(state, action, reward, next_state, done)` keeps its exact signature.** `PPO_Agent` appends to an ordered rollout list instead of a capacity-bounded `ReplayBuffer`; order matters for GAE, unlike DQN's random sampling.
2. **`learn(*, reached_max_steps=False)` takes only the cutoff flag.** When it
   is `True`, `PPO_Agent` marks its latest rollout entry as a GAE boundary but
   leaves its stored `done=False`, preserving the `V(next_state)` bootstrap.
   PPO checks its own rollout length against `PPO_ROLLOUT_LENGTH` (see
   "Constants" below). Below that length it returns `[]`, the same shape
   `GNN_DQN_Agent.learn` returns when `can_train()` is `False`. Once the
   rollout is full, it runs the full PPO update (GAE, `PPO_EPOCHS` passes over
   shuffled minibatches, clipped surrogate + value loss + entropy), clears the
   rollout, and returns the real per-minibatch losses.
3. **PPO has no epsilon-greedy behavior.** It inherits the no-op
   `BaseAgent.on_episode_start(...)` hook and sets `epsilon = 0.0` only for
   evaluator-interface compatibility: `Evaluator` saves, zeroes, and restores
   that attribute for every agent type. PPO never reads it. `train_mode` is
   the meaningful control: `set_train_mode(False)` makes `act()` take the
   argmax rather than sample the categorical policy, which keeps evaluation
   deterministic.

None of this requires `Trainer` to know anything about PPO. That is the point:
the interface stays shared, while the algorithm-specific internals stay on the
agent. `Dueling_DQN_Agent` already follows that split; PPO just relies on it
more heavily.

**Comment:** the shared loop can work for PPO, but `PPO_Agent.learn()` must own
the update boundary carefully. A rollout may span multiple game episodes, and
an update may happen in the middle of a game after one learner turn. That is
acceptable as a truncated on-policy rollout: the policy is fixed while those
transitions are collected, `next_state` provides the bootstrap point, and the
next transition after the update starts a fresh rollout under the new policy.

---

## Store collection-time log-prob and value

PPO's clipped ratio needs $\pi_{old}(a|s)$: the probability of the action under
the policy that actually chose it. It must not be recomputed later from the
current, already-updated network, because that would make the ratio trivially 1
and defeat clipping. The same rule applies to the value estimate used to compute
advantages: store the value estimate from collection time.

`remember(...)`'s signature should not grow new parameters, so `act()` stores
the values it computed when it sampled the categorical policy and read the
value head:

```python
self._pending_log_prob = dist.log_prob(chosen_index).detach()
self._pending_value = value.detach()
self._pending_action_index = int(chosen_index)
```

Then `remember(...)` reads and clears them onto the rollout entry. This is safe
because `Trainer.train()` currently calls `act()` then `remember()` once per
learner turn for the same transition, without interleaving a second action
selection. Re-check this assumption if the training loop ever changes.

**Important:** store detached tensors or plain Python floats. Do not keep the
collection-time computation graph alive for the whole rollout. PPO recomputes
current logits/values during `learn()`; stored old log-probs/values are fixed
targets, not tensors that should receive gradients.

**Store the chosen legal-action index, plus the selected `Action` already
required by the shared transition contract.** `act()` must also stash the
sampled/argmax index, and `remember()` must put it in the rollout entry. During
learning, each decision's logits are flattened with the other decisions'
legal-action sets; the saved index is the unambiguous selector for that
decision. Rebuild legal actions from the stored state in the same deterministic
environment order, then validate `legal[action_index] == action` before
gathering. The index is the selector; the action equality check is an
integrity assertion that catches a future ordering or action-representation
change instead of silently training on the wrong logit.

---

## Network plan: `PPO_Net`

**Copy `dueling_dqn.py`, not `gnn_dqn.py`.** Dueling already has the shape PPO
needs: one clean base row routed to a value head, one injected row per legal
action routed to per-phase heads, and `group_index` grouping rows by decision.
The only structural change is: **do not combine into `Q`.** Return the two
streams separately instead of fusing them:

```python
def forward(self, state, phase, card_indices, group_index=None, value_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    logits = <same per-phase-head routing as Dueling's advantage stream>   # [N_action_rows]
    value = <same value_head(g[value_mask]) as Dueling>                   # [n_groups]
    return logits, value
```

- `logits`: one scalar logit per legal action, **not** an advantage
  — `Categorical(logits=logits[group_mask])` over one decision's legal
  actions gives a proper $\pi(a|s)$ directly. No mean-subtraction needed
  here (unlike Dueling's `Q`), since softmax already normalizes.
- `value`: the clean row's value-head output, one scalar per decision
  group — identical computation to Dueling's value stream.
- The per-phase heads (`ScoringHead`/`TradeInHead`) need no code changes,
  just as they needed no structural change when Dueling reinterpreted them as
  advantage heads. Only the meaning of the scalar changes.
- Keep the `value_mask` argument from Dueling's clean-row implementation even
  if the sketch above omits it for brevity. PPO should compute `V(s)` from the
  clean base row, not from action-injected rows.

---

## Agent plan: `PPO_Agent`

- **`act(...)`**: build the clean base row + one injected row per legal
  action (identical to `score_actions` in the other two agents), forward
  through `PPO_Net` to get `(logits, value)`, then:
  - `train_mode=True`: sample `Categorical(logits=logits)`, stash
    `log_prob`/`value` for the upcoming `remember()` call.
  - `train_mode=False` (eval): take `argmax(logits)` — deterministic,
    matching what `Evaluator` expects when it calls `set_train_mode(False)`.
- **`remember(...)`**: attach the stashed `log_prob`/`value`, append
  `(state, action, action_index, old_log_prob, old_value, reward, next_state, done)` to
  an ordered rollout buffer (see below). No `.perspective` handling changes
  — same snapshot-and-tag pattern as the other two agents.
  If `remember()` is called without pending values, raise a clear error; that
  catches any future trainer change that violates the `act()` then
  `remember()` contract.
- **`learn(...)`**: gate on rollout length (see "Keep PPO-specific behavior
  inside the agent" above). If
  full:
  1. Build every stored transition's graphs **once** (not once per epoch —
     PPO reuses the same fixed rollout across `PPO_EPOCHS` passes, so
     building the batch once and reusing it across epochs is a real,
     not premature, optimization here, unlike `_q_value`'s per-call
     rebuild in Dueling which genuinely can't be cached that way).
   2. Compute $V(s')$ for every **non-terminal** transition directly from its own stored
     `next_state`, not by shifting the value array by one index the way
     the reference implementation in `Temp/Examples/PPO_Agent.py` does.
     Our `remember(state, action, reward, next_state, done)` already
     carries `next_state` per transition (DQN's replay buffer needed the
     same field), so there is no need to separately ask anything external
     for a "one more bootstrap value" the way a bare `(s, a, r, done)`
     rollout would — this sidesteps an off-by-one class of bug entirely.
  3. GAE: $\delta_t = r_t + \gamma V(s'_t)(1-done_t) - V(s_t)$, then the
     usual backward recursive sum with `PPO_GAE_LAMBDA`, resetting the
     recursion at `done_t=True` boundaries (same zero-bootstrap idea
     `_max_next_ddqn_q` already uses via `(~done).float() * max_next_q`).
  4. Up to `PPO_EPOCHS` passes over `PPO_MINIBATCH_SIZE`-sized shuffled chunks of
     the rollout: recompute `(logits, value)` with the **current** online
     net, get the ratio $r_t(\theta) = \exp(\log\pi_\theta - \log\pi_{old})$,
     clipped surrogate loss, Smooth-L1/Huber value loss against the raw GAE
     returns, entropy
     bonus, combine with `PPO_VALUE_LOSS_COEF`/`PPO_ENTROPY_COEF`, clip
     gradients (reuse `GRAD_CLIP_MAX_NORM`), step the optimizer. Before each
     later minibatch, stop the remaining passes when its sampled non-negative
     k3 approximate KL, `mean(ratio - 1 - log(ratio))`, exceeds
     `PPO_TARGET_KL`; the update that caused the drift is retained. k3 is a
     sampled estimate of `KL(old policy || current policy)` and avoids the
     negative finite-minibatch readings of `mean(old_log_prob - log_prob)`.
   5. Clear the rollout buffer, return the list of per-minibatch losses.

  **Required boundary details:** for a real `done=True`, use a zero bootstrap
  and do not build a next-state action batch: a game-over state may have no
  legal actions. A `MAX_STEPS_PER_EPISODE` cutoff is not terminal: leave its
  stored `done=False` and compute the normal `V(next_state)` bootstrap. It is,
  however, an episode boundary. Store an internal `gae_boundary` marker on
  that rollout entry and reset GAE's recursive accumulator there, so its
  advantage does not include the first transition of the next reset game.
  The marker affects only the recursive GAE carry-over, not the immediate
  bootstrap term. It need not change the public
  `remember(state, action, reward, next_state, done)` signature.

  Normalize advantages over the rollout before minibatching (with a small
  epsilon in the standard-deviation denominator); returns remain unnormalized
  value targets. After each full update, set an agent-owned
  `last_update_metrics: dict[str, float]` with approximate KL, clip fraction,
  raw/normalized entropy and legal-action counts; policy/Huber-value/entropy
  loss components plus raw value MSE/RMSE for cross-run comparison;
  return, old-value, advantage, and explained-variance statistics; pre-clip
  total/encoder/policy-head/value-head gradient norms;
  one-minibatch actor-versus-critic gradient norms on the shared encoder;
  completed epochs, early-stop diagnostics, and exact optimizer/sample counts.
  These metrics do not alter parameter updates. Combined/head norms reuse the
  normal backward pass; the actor-versus-critic encoder comparison performs
  two diagnostic `autograd.grad` calls on only the first accepted minibatch of
  each rollout update.
  `Trainer` generically merges this optional mapping into
  the episode metrics it forwards to `TrainingLogger`, without inspecting the
  agent type. This makes PPO health visible in W&B while preserving the
  shared `learn(*, reached_max_steps=False) -> list[float]` contract and
  avoiding a PPO branch. `progress_metrics()` additionally reports rollout
  fill/fraction and completed rollout updates on every episode row.
- **No target network.** PPO doesn't bootstrap off a frozen copy the way
  Double-DQN does. The clipped ratio against the rollout's old log-probs is the
  stabilizer, not a separate frozen net. This has one concrete interface
  consequence: `TrainingLogger._build_config` reads `agent.target_net`
  (`target_model_str`, plus `Docs/Trainer.md`'s required-interface list). The
  simplest fix that needs no `Trainer`/logger changes is
  `PPO_Agent.target_net = self.net` (an alias, not a real copy). That satisfies
  the interface, and `target_model_str` in W&B config correctly shows the same
  architecture.
- **`label = "PPO"`** class attribute, same mechanism as the other two
  agents (`Docs/Trainer.md`'s required-interface list) — default
  checkpoint dir becomes `Checkpoints/PPO_<id>`, W&B run name `PPO_<id>`.

### New file: rollout buffer

`ReplayBuffer` is the wrong shape for PPO. It is a capacity-bounded ring buffer
for random sampling, while PPO needs ordered, full-rollout access that is
cleared after every update. Add a small `risk/learning/rollout_buffer.py` with
a `RolloutBuffer` class:

```python
buffer = RolloutBuffer()
buffer.push(state, action, action_index, log_prob, value, reward, next_state, done)
transitions = buffer.all()   # ordered, not sampled
buffer.clear()
```

Use the same split as `ReplayBuffer`: store raw domain objects and build graphs
later at learn time. The reasoning is the same. Building/injecting action graphs
is real work, so defer it until the data is actually used.

Store the selected legal-action index from `act()` alongside the action. The
flattened training batch must preserve each decision's action-row offset so it
can gather exactly that selected logit. Rebuilding candidates from a stored
state must use `env.legal_actions(state_snapshot)` in its deterministic order
and assert that its indexed action equals the stored action before gathering.
For a forced episode cutoff, `learn(reached_max_steps=True)` records
`gae_boundary=True` while preserving the trainer-provided `done=False`; PPO
then bootstraps from that transition's `next_state` without joining GAE to the
next game.

---

## Checkpointing

Checkpointing is simpler than DQN/Dueling because there is no large persistent
replay buffer to serialize. The rollout buffer is empty right after a successful
`learn()` call, and a partially filled rollout is not worth resuming. Losing it
on restart means collecting a fresh rollout, not losing trained progress.
`save_checkpoint`/`load_checkpoint` store `net`, `optimizer`, rollout-update
count, optimizer-step count, and processed-sample count. Legacy checkpoints
recover the optimizer-step count from Adam state and visibly mark their
minibatch-size-based sample count as estimated. There is no `replay.pt` file.
`save_params` remains the policy-only net-weights dump, same as the other
agents.

**Comment:** dropping a partial rollout on resume is the right v1 tradeoff.
Trying to resume half-collected on-policy data would add statefulness and edge
cases for very little value.

---

## Trainer notes for PPO

Same story as Dueling (`Docs/DuelingDQN.md`'s "Trainer notes for dueling",
`Docs/Trainer.md`) except for one generic observability hook: `Trainer` must
merge an optional `agent.last_update_metrics` mapping into the episode metrics
before calling `TrainingLogger`, without inspecting the agent type. Do not
change a max-step cutoff's `done` value: `Trainer` already leaves it `False`,
which preserves DQN's and PPO's valid bootstrap. `Trainer` passes
`reached_max_steps=True` to the existing `learn()` call for that transition;
PPO records its internal `gae_boundary` marker there without changing
`remember()` or branching the trainer on PPO. Build `PPO_Agent` in the caller
and pass it into the existing `Trainer`, relying on
`label`/`agent_class`/`model_class` in the checkpoint path and W&B config to
distinguish it. Use `resume=False` and a fresh run id for the first PPO run,
same as Dueling.

**Superseded (2026-08-09):** `trainer.py`'s `main()` does not actually have a
commented second/third agent block today — it only ever builds one active
agent. The paragraph above's "third commented block next to DQN/Dueling"
described an aspiration, not the current file. See "Planned PPO restart:
PPO_200"'s "Comment (wiring)" below for the concrete decision: selecting
PPO_200 in `main()` replaces `DQN_105`'s active configuration outright: keep
`DQN_105`'s `RUN_ID`/`wandb_run_id` as a comment rather than a live block.

Beyond the optional-metrics hook, the trainer loop should stay unchanged.
Changing `Trainer.train(...)` to branch on PPO is not permitted.

---

## Constants

These live in `risk/learning/train_constants.py`, alongside the shared DQN
settings, so a run's PPO and DQN knobs have one source of truth.

- `PPO_ROLLOUT_LENGTH` — learner turns collected before an update. Note the
  unit: **learner turns** (one `remember()` call each), the same
  granularity DQN's replay buffer uses — not raw `env.step()` calls, and
  likely spans multiple episodes to fill one rollout, which is expected
  for a persistent on-policy agent reused across episodes exactly like the
  other two agents.
- `PPO_EPOCHS` — passes over the same rollout per update.
- `PPO_MINIBATCH_SIZE` — chunk size within one epoch's shuffle.
- `PPO_CLIP_EPS` — surrogate clip range.
- `PPO_TARGET_KL` — sampled approximate-KL limit that stops the remaining
  minibatches of a rollout once prior optimizer steps have moved the policy
  too far from the policy that collected it.
- `PPO_GAE_LAMBDA`.
- `PPO_VALUE_LOSS_COEF`.
- `PPO_VALUE_HUBER_BETA` — transition point between Huber's quadratic and
  linear regions; PPO_045 uses Huber to bound critic outlier gradients while
  retaining raw MSE/RMSE only as diagnostics.
- `PPO_ENTROPY_COEF` — fixed for v1, no decay schedule (the reference
  `Temp/Examples/PPO_Agent.py` decays this; keep v1 simple and add decay
  later only if entropy collapses too fast in practice — same "simple
  first, optimize if needed" stance the Dueling doc already took on
  `_q_value`'s cost).
- `PPO_LR` — separate from DQN's `lr` default; actor and critic share one
  optimizer here (one network, two heads), so only one LR knob is needed,
  unlike the reference example's separate actor/critic optimizers.

All of these are unvalidated starting guesses. The first real experiment will
need empirical tuning, especially for `PPO_ROLLOUT_LENGTH`, because the useful
value depends on how many learner turns a typical episode produces.

Suggested v1 defaults to start discussion, not final values:

| Constant | Starting point | Comment |
|---|---:|---|
| `PPO_ROLLOUT_LENGTH` | 256 learner turns | Updates after roughly two current-length games instead of seven; still enough data for GAE. |
| `PPO_EPOCHS` | 4 | Conservative; avoid overfitting a small rollout. |
| `PPO_MINIBATCH_SIZE` | 64 | Four minibatches per rollout, doubling optimizer-step frequency without increasing epoch reuse. |
| `PPO_CLIP_EPS` | 0.2 | Standard PPO starting point. |
| `PPO_TARGET_KL` | 0.02 | Stops extra epochs when the policy shift becomes unsafe; does not roll back the crossing update. |
| `PPO_GAE_LAMBDA` | 0.95 | Standard bias/variance tradeoff. |
| `PPO_VALUE_LOSS_COEF` | 0.5 | Standard shared-network value weight. |
| `PPO_VALUE_HUBER_BETA` | 1.0 | Bounds large critic-error gradients; raw MSE and RMSE remain logged. |
| `PPO_ENTROPY_COEF` | 0.01 | Enough to discourage early policy collapse. |
| `PPO_LR` | 1e-4 | Reduced after PPO_043's KL early stop saturated and prevented most planned minibatches. |

---

## Tests

Mirror `Temp/tests/test_dueling_dqn.py`'s shape — narrow, focused, added to
a new `Temp/tests/test_ppo.py` per `Docs/Testing.md`'s convention:

1. `PPO_Net.forward` returns `(logits, value)` with the right shapes —
   `logits` one per action row, `value` one per decision group.
2. GAE math sanity check on a tiny hand-constructed rollout (fixed
   rewards/values/dones) compared against a manually-computed expected
   advantage/return sequence — same spirit as `test_reward.py`'s
   per-component checks.
3. `learn()` returns `[]` before `PPO_ROLLOUT_LENGTH` transitions are
   collected, and a non-empty loss list once it's reached.
4. `set_train_mode(False)` makes `act()` deterministic (always the argmax
   action for a fixed state), `set_train_mode(True)` allows sampling to
   vary.
5. `save_checkpoint`/`load_checkpoint` round-trips net/optimizer/
   train_steps (no replay buffer to check, per "Checkpointing" above).
6. The collection-time log-prob/value round-trip correctly:
   `act()` followed by `remember()` produces a rollout entry whose stored
   `old_log_prob` matches what `act()` computed, not a value recomputed
   from a since-updated network.
7. Stored `old_log_prob`/`old_value` are detached (`requires_grad=False`) so
   rollout storage does not retain collection graphs.
8. Selected-action index and flattened batch offsets gather the same logit
   that `act()` used, across all represented phases; deliberately mismatching
   the rebuilt indexed action and stored action raises a clear error.
9. A `MAX_STEPS_PER_EPISODE` cutoff remains `done=False`, bootstraps from its
   `next_state`, and causes `learn(reached_max_steps=True)` to set
   `gae_boundary=True`; GAE must not incorporate the first transition from the
   next reset game.
10. Terminal transitions use a zero bootstrap without attempting to build a
    no-legal-action game-over batch.
11. A rollout whose sampled KL exceeds its configured target performs its
    first minibatch, then skips the remaining PPO epochs and reports the
    early-stop diagnostics.
12. The k3 KL estimator is non-negative for representative log-ratios.
13. Diagnostic counters survive checkpoints, and KL-stop tests verify exact
    optimizer/sample counts plus bounded normalized entropy.
14. The optimized Huber critic loss remains below raw MSE, while the historical
    `ppo_value_loss` key and RMSE continue to report MSE-scale accuracy.
15. **(Added for the PPO_200 restart, not part of v1.)**
    `ppo_value_to_policy_encoder_grad_ratio` is present and finite in
    `last_update_metrics` under normal conditions, and its `1e-12` epsilon
    floor is exercised by a case with a (near-)zero policy-encoder gradient,
    confirming a large-but-finite ratio rather than `NaN`/`Inf` or a silently
    substituted zero. See the PPO_200 restart section's implementation step 6.

---

## Rollout plan

1. Create `ppo_net.py`/`ppo_agent.py`/`rollout_buffer.py` by copying
   `dueling_dqn.py`/`dueling_dqn_agent.py` as starting points; strip the
   `Q` combination, return `(logits, value)`.
2. Wire `act()` to sample/argmax correctly depending on `train_mode`, with
   the log-prob/value smuggling into `remember()`.
3. Implement `RolloutBuffer` (ordered push/all/clear).
4. Implement boundary-aware GAE + normalized advantages + clipped-surrogate
   + value-loss + entropy `learn()`, gated on `PPO_ROLLOUT_LENGTH`.
5. Add `PPO_*` constants to `train_constants.py`.
6. Add `label = "PPO"`, `target_net = self.net` alias, checkpoint methods.
7. Add the focused tests above; validate GAE and detach behavior against
   hand-computed/small examples before trusting it on real rollouts.
8. Add the optional commented PPO import/construction block in
   `trainer.py`'s `main()` only after `PPO_Agent` exists.
9. Pass `reached_max_steps` through the shared `learn()` call and add the
   generic optional `last_update_metrics` hook to `Trainer`/`TrainingLogger`,
   without inspecting an agent type. PPO records cutoff boundaries without
   changing `done`.
10. Smoke-test through the existing `Trainer`, exactly like Dueling's rollout
    plan did: one short run, `use_wandb=False`, confirming `remember`/`learn`
    round-trip behavior, metrics forwarding, and that the rollout-length gate
    withholds losses until the rollout is full.
11. Run a first real comparison run under its own run id.

---

## Success criteria

- `GNN_DQN_Agent`/`Dueling_DQN_Agent` remain completely untouched and still
  train/evaluate/resume as before.
- `PPO_Agent` selects actions in the same environments via the same
  `BaseAgent` contract.
- `Trainer.train()` requires zero special-casing for PPO — the same loop
  that runs DQN/Dueling runs PPO.
- One full rollout update changes `PPO_Net`'s parameters; a partial,
  not-yet-full rollout does not (verified by the rollout-length gate test).
- Checkpoints save/load independently under `Checkpoints/PPO_<id>`,
  distinguishable from DQN/Dueling in W&B config the same way.
- First comparison run reports the same `win`/`win_rate_last_50`/reward
  components/eval metrics shape as the other two agents.

## Non-goals for v1

- Do not change PQN here. PQN and PQN_e are separate implemented replay-based
  learners; their behavior-policy comparison belongs in `Docs/PQN.md`.
- Do not revisit action representation here — `Docs/NetworkArchitectures.md`
  records the settled injection-only roadmap.
- Keep the DQN agents' learning behavior unchanged: they accept and currently
  ignore `reached_max_steps`. The only planned shared addition beyond that is
  an optional update-metrics hook; do not change max-step cutoff terminal
  semantics or branch `Trainer`/`TrainingLogger` on PPO.
- Do not add an entropy-coefficient decay schedule, learning-rate
  scheduler, or value-clipping in v1 — the reference example has all
  three; start without them and add only if training shows a specific need.
- Do not change the reward function or action encoding.

---

## Planned PPO restart: PPO_200

**Status: PPO-specific configuration and diagnostics are implemented. The
local smoke run passed through episode 250 (27 rollout updates, 426 optimizer
steps, 109,056 processed samples), and the launcher is now configured for a
fresh W&B-backed PPO_200 run.**

**Confirmed (2026-08-09):** run id `PPO_200`, not `PPO_104`/`PPO_106`. It
deliberately starts a fresh numbering block, separate from the shared
DQN/Dueling run-id sequence (`DQN_105` is the latest of those) and from the
legacy `PPO_041--045` runs, so a chart name or checkpoint directory alone
signals which era a run belongs to.

**Comment (wiring):** `trainer.py`'s `main()` only ever selects one active
agent — there is currently no commented second block despite earlier text in
this doc assuming one. Launching `PPO_200` there **replaces** `DQN_105`'s
current resume configuration (`resume=True`, pinned
`wandb_run_id="5b66yunb"`, restart checkpoint episode 400 — see
`Docs/ChangeLog.md`'s 2026-08-09 entry), it does not run alongside it. When
implementing step 3 below, keep `DQN_105`'s exact `RUN_ID`/`wandb_run_id` as a
commented reference next to the new PPO block rather than deleting it outright,
so resuming `DQN_105` later doesn't require re-deriving those values.

The historical PPO runs are not a clean test of PPO under the current task.
They used the old 13-column graph/action representation, the old territory
reward, and dense shaping at its unscaled effective strength.  The next PPO
experiment must start fresh with the same current 15-column representation and
the current shared DQN_105 reward regime, including its implemented
reinforcement formulas.  It must not load an old PPO checkpoint or rollout.

### Evidence from PPO_041--PPO_045

The previous runs establish an optimization problem, not that PPO is incapable
of learning the game:

| Run | Main configuration/change | Observed result |
|---|---|---|
| `PPO_041` | rollout 1024, minibatch 128, LR `3e-4`, no KL stop | 0 wins through episode 599 |
| `PPO_042` | rollout 256, minibatch 64, LR `3e-4` | began learning, but only 16% wins in episodes 1001--2000 |
| `PPO_043` | added target-KL stop `0.02` | early-stopped about 94% of updates and regressed after its early peak |
| `PPO_044` | reduced LR to `1e-4` | critic gradients remained extremely large; almost no wins |
| `PPO_045` | replaced critic MSE optimization with Huber | improved steadily to about 24% late training wins, still far below DQN at a matched learner-turn budget |

The decisive PPO_045 diagnostic is shared-encoder gradient imbalance.  Late in
the run, its mean actor encoder gradient norm was about `4.8`, versus about
`169` for the already-weighted critic encoder gradient: roughly 35:1 in favor
of the critic.  The combined gradient was clipped on almost every optimizer
step.  PPO_045's normalized entropy also fell to about `0.25` while the policy
was still weak.  The restart therefore targets critic dominance, correlated
small rollouts, and premature policy narrowing.

Historical W&B runs:

- `PPO_041`: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/1dvbyora`
- `PPO_042`: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/y8xgfnqx`
- `PPO_043`: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/1rku19b6`
- `PPO_044`: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/xnfzycub`
- `PPO_045`: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/dvskjyqh`

### PPO_200 hypothesis

Keep the current shared DQN_105 reward constants unchanged: dense shaping
scale `0.3` and terminal rewards `+300/-300`.  DQN_105 is currently the most
promising DQN run, so PPO_200 must not alter the general reward regime while it
is still training.  This makes PPO_200 a comparison under the same live task,
not a reward experiment.  Any later PPO-only reward profile would be a
separately planned experiment, because the present reward calculator is shared
by every learner.

PPO is on-policy and cannot replay rare wins.  Increase rollout diversity so
each update includes several games and terminal outcomes, and lower the
critic's shared-encoder influence.  The initial restart configuration is:

| Constant | PPO_200 value | Reason |
|---|---:|---|
| shared reward regime | DQN_105: shaping `0.3`, terminal `+300/-300` | Keep the currently promising DQN task unchanged; do not confound PPO tuning with a reward change. |
| `PPO_ROLLOUT_LENGTH` | `1024` | More games, outcomes, seats, player counts, and rosters per on-policy update. |
| `PPO_MINIBATCH_SIZE` | `256` | Lower-variance gradients; four minibatches per epoch. |
| `PPO_EPOCHS` | `4` | At most four presentations of each on-policy sample. |
| `PPO_LR` | `1e-4` | PPO_044/045's safer policy step; let KL diagnostics decide whether it is still too large. |
| `PPO_CLIP_EPS` | `0.2` | Retain the established surrogate bound. |
| `PPO_TARGET_KL` | `0.02` | Retain the corrected k3 early-stop safety check. |
| `PPO_GAE_LAMBDA` | `0.95` | Retain the existing bias/variance tradeoff. |
| `PPO_VALUE_LOSS_COEF` | **`0.1`** | Fivefold reduction from PPO_045 to address measured critic dominance. |
| `PPO_VALUE_HUBER_BETA` | `1.0` | Retain bounded critic outlier gradients. |
| `PPO_ENTROPY_COEF` | `0.01` | Keep one initial entropy setting; react only to measured collapse. |
| `GRAD_CLIP_MAX_NORM` | `10.0` | Keep the existing safety bound and avoid a second confounding change. |

**Comment (current code state, checked 2026-08-09):** `train_constants.py`
now matches the PPO_200 target: `PPO_ROLLOUT_LENGTH=1024`,
`PPO_MINIBATCH_SIZE=256`, and `PPO_VALUE_LOSS_COEF=0.1`. `PPO_CLIP_EPS`, `PPO_TARGET_KL`,
`PPO_GAE_LAMBDA`, `PPO_VALUE_HUBER_BETA`, `PPO_ENTROPY_COEF`, `PPO_LR`, and
`GRAD_CLIP_MAX_NORM` already match this table today and need no change.
`GRAD_CLIP_MAX_NORM` is also shared with `GNN_DQN_Agent`/`Dueling_DQN_Agent`/
`PQN_Agent`/`ADQN_Agent`, so leaving it untouched is required, not just
convenient, to avoid a cross-agent confound.

The reinforcement-reward revision in `Docs/Reward.md` is implemented. PPO_200
must retain the current DQN_105 formulas and constants, and capture their exact
values in W&B.  Do not implement a reward redesign as part of PPO tuning: that
would alter DQN_105 and make a PPO regression ambiguous.

### Implementation steps

1. Change only the PPO constants listed above in
   `risk/learning/train_constants.py`; do not modify any shared reward
   constant or DQN-family setting.
2. Change `PPO_Agent`'s encoder-component diagnostics to aggregate every
   executed minibatch in an update (weighted by minibatch size), rather than
   reporting only the first minibatch.  Log a finite critic-to-actor ratio so
   the primary PPO_045 failure signal is representative of the full rollout.

   Name the new metric `ppo_value_to_policy_encoder_grad_ratio` and calculate
   it as the update-wide value norm divided by
   `max(update_wide_policy_norm, 1e-12)`.  This preserves a finite, visible
   failure signal if the policy encoder gradient vanishes; do not silently drop
   the ratio or substitute zero in that case. Add
   `"ppo_value_to_policy_encoder_grad_ratio"` to
   `PPO_Agent.unweighted_update_metrics` alongside its two inputs
   (`ppo_policy_encoder_grad_norm`/`ppo_value_encoder_grad_norm`, already
   listed there): it is a per-update-computed ratio, not a raw per-minibatch
   loss, so `Trainer` must average it across multiple rollout updates in one
   episode with equal per-update weight, not weighted by
   `ppo_optimizer_steps_per_update` the way loss/return fields are
   (`Docs/Trainer.md`'s "rollout-level fields remain equally weighted").
   Omitting it from that set would not error; it would silently mis-weight the
   ratio the same way its inputs would be mis-weighted if they were omitted.

   **Comment (cost/design of this change):** today's code
   (`measured_component_encoder_grads` in `ppo_agent.py`) runs the two
   diagnostic `torch.autograd.grad(..., retain_graph=True)` calls exactly once
   per update, on the update's first minibatch. That minibatch always has
   KL≈0 by construction (nothing has moved yet), so it is the *least*
   representative sample of the drift-induced imbalance the diagnostic exists
   to catch — this is a correctness fix to the measurement, not only more
   logging. Running it on every minibatch instead of once raises the diagnostic
   cost from two `autograd.grad` calls to up to 32 per update at the current
   PPO_200 sizing: two component gradients for each of `PPO_EPOCHS *
   ceil(PPO_ROLLOUT_LENGTH / PPO_MINIBATCH_SIZE)` = `4 * 4` minibatches (fewer
   if KL early-stops). That figure is specific to today's
   `PPO_ROLLOUT_LENGTH`/`PPO_MINIBATCH_SIZE`/`PPO_EPOCHS` values and will
   change if any of the three does later — it is not a fixed property of the
   design. That is a real, bounded compute-cost increase, accepted here
   because the two calls are diagnostic-only (no parameter update depends on
   them) and because an update-wide reading is the only way to tell whether
   critic dominance holds across the whole rollout rather than just its
   least-drifted minibatch. Weight each minibatch's contribution by its
   executed sample count (`len(indices)`) rather than averaging minibatches
   equally — at `PPO_ROLLOUT_LENGTH=1024` and `PPO_MINIBATCH_SIZE=256` every
   minibatch happens to be the same size, so the weighting is inert at these
   particular values, but keep it so the code stays correct if either constant
   changes later.
3. Select `build_learner_agent("PPO", ctx)` in `trainer.py`, set
   `RUN_ID = 200`, and use `resume=False`. **Implemented:** after the local
   smoke run passed, `main()` now builds PPO_200 with W&B enabled and no W&B
   run id, creating a fresh cloud run rather than resuming local or cloud state.

   **Comment:** this step also touches `Temp/tests/test_ppo.py`, which is not
   separately listed but is covered by step 6. Specifically,
   `test_kl_limit_stops_remaining_ppo_epochs` hardcodes the current
   coefficient (`ppo_weighted_value_loss == 0.5 * ppo_value_huber_loss`); once
   `PPO_VALUE_LOSS_COEF` becomes `0.1` that assertion must change to match, or
   it will fail for the right reason but an unexpected one if not anticipated.
4. Confirm the run resolves to new namespaces `Checkpoints/PPO_200` and
   `PPO_200`, with a randomly initialized policy/value network and empty
   rollout.
5. Confirm the current graph/action widths are derived from the current
   environment rather than hardcoded or adapted from a legacy PPO checkpoint.
6. Add/update focused tests in the existing `Temp/tests/test_ppo.py` and
   relevant constants/logger tests; do not add a new test file for this
   subsystem.  Read `Docs/Testing.md` before changing or running tests. At
   minimum, update the existing `test_kl_limit_stops_remaining_ppo_epochs`
   assertion for the new `PPO_VALUE_LOSS_COEF` (step 3's comment above), and
   add a new case exercising `ppo_value_to_policy_encoder_grad_ratio`
   end to end: assert it is present and finite in `last_update_metrics` under
   normal conditions, and separately construct or mock a case where the
   policy-encoder gradient is (near) zero to confirm the ratio reports a
   large-but-finite value through the `1e-12` floor rather than `NaN`/`Inf`
   or a silently substituted zero. This is item 15 in the "Tests" section
   above, added there for a feature that section predates.
7. Update this document, `Docs/Trainer.md`, and `Docs/ChangeLog.md` with the
   implemented launcher/configuration before starting the real run.
8. Run the focused PPO/trainer/logger tests, then the full suite using the
   project environment documented in `Docs/Testing.md`.
9. Perform a short `use_wandb=False` smoke run that fills at least one complete
   1024-turn rollout, executes an update, clears the rollout, and emits finite
   diagnostics.

   **Comment (cadence):** `CHECKPOINT_AFTER`/`CHECKPOINT_EVERY`/
   `EVAL_EVERY_EPISODES` are shared, episode-counted constants this plan does
   not propose changing. A 1024-turn rollout is 4x PPO_045's 256, so PPO_200
   completes fewer rollout updates per episode-counted checkpoint/eval
   interval than PPO_045 did. Not a reason to change those constants
   preemptively — just confirm during this smoke run that the interval still
   captures a reasonable number of rollout updates, and only revisit if it
   doesn't.
10. **Implemented:** start the fresh W&B run only after the smoke run and
   configuration review pass. The 250-episode smoke checkpoint is preserved as
   `Checkpoints/PPO_200_smoke_ep000250`; PPO_200 itself begins with an empty
   checkpoint namespace and random network initialization.

### Required monitoring

Compare by `cumulative_learner_turns` first, not only by episode.  PPO and DQN
present samples to their optimizers at very different rates.  Review PPO_200 at
approximately 250K, 500K, 1M, and 1.5M learner turns.

Track at minimum:

- training and deterministic evaluation win rate, including player-count and
  opponent-roster breakdowns;
- update-wide, minibatch-size-weighted `ppo_policy_encoder_grad_norm` and
  `ppo_value_encoder_grad_norm`, plus
  `ppo_value_to_policy_encoder_grad_ratio`;
- total/head gradient norms and `ppo_gradient_clip_fraction`;
- approximate KL, surrogate clip fraction, completed epochs, and KL early-stop
  fraction;
- normalized entropy, legal-action count, and forced-action fraction;
- value Huber loss, raw value RMSE/MSE, return/value scale, and explained
  variance;
- cumulative learner turns, optimizer steps, processed samples, and wall time.

**Comment (undercorrection risk):** cutting `PPO_VALUE_LOSS_COEF` fivefold
reduces critic dominance of the shared encoder, but it also slows how fast the
value function tracks returns. Explained variance and value RMSE (already
listed above) are the guard against overshooting into an undertrained critic
— a falling explained variance alongside a falling gradient ratio would mean
the fix went too far, not that it is working. Watch both together, not the
gradient ratio alone.

The old 35:1 critic/actor encoder-gradient ratio is the primary failure signal.
For PPO_200, a sustained ratio below 5:1 is the target; below 10:1 is acceptable
early; a sustained ratio above 20:1 means the critic still controls the shared
representation.  Do not react to one minibatch or episode--use rollout-update
windows.

Normalized entropy should not fall below about `0.4` while evaluation remains
weak.  If it does, first confirm that the decline occurs across phases/action
set sizes.  The next isolated experiment may then increase entropy pressure or
normalize the entropy bonus by each decision's maximum categorical entropy;
do not change entropy mid-run and call it the same experiment.

If KL early stopping fires on most updates and completed epochs remain near or
below one, the next isolated experiment should reduce PPO LR to `5e-5`.  Do not
raise the KL target merely to force all epochs through.  If the critic remains
dominant despite coefficient `0.1`, the next isolated value coefficient is
`0.05`; separate actor/critic encoders or optimizers are later structural
experiments, not part of PPO_200.

### Continue/stop criteria

- At 250K turns, require finite metrics, no sustained value/return explosion,
  and evidence that the actor receives a meaningful shared-encoder gradient.
- At 500K turns, require improving evaluation score or win rate and no
  premature entropy collapse.  A low absolute win rate alone is not a stop
  condition because a from-scratch PPO run can learn slowly.
- At 1M turns, PPO_200 must clearly exceed PPO_045's roughly 24% late-training
  win level or show a strong, consistent evaluation trend that justifies more
  time.  Compare against DQN_105 at the same learner-turn budget as the main
  target once that run reaches the matched budget; use DQN_103 only as the
  historical secondary reference until then.
- Continue to 1.5M turns when the robust evaluation trend is still positive.
  Stop or fork an isolated hyperparameter correction when performance is flat
  and a diagnostic identifies critic dominance, KL saturation, or entropy
  collapse.

Evaluation points contain only six games today and are too coarse for a final
claim.  Promotion of a best PPO checkpoint or a claim of DQN parity requires a
larger fixed suite (at least 100 games, balanced across seats, player counts,
and representative rosters) plus held-out randomized games.

### Success definition and fallback

The restart succeeds operationally when it removes the old optimization
pathology: the actor materially influences the encoder, entropy remains useful
until the policy becomes competent, KL permits useful sample reuse, and
evaluation improves at matched learner-turn budgets.  The long-run goal is to
match or exceed DQN/Dueling evaluation performance, but PPO_200 is the first
controlled test under the current task, not proof that one hyperparameter set
must achieve parity.

If a stable from-scratch PPO remains far behind after the planned budget, keep
that result as the fair algorithm comparison.  The practical follow-up is a
separate transfer experiment: behavior-clone PPO's policy from DQN_103 on a
held-out state/action dataset, initialize the critic from observed returns,
then fine-tune on-policy.  Label that run as DQN-assisted PPO; do not present it
as a from-scratch PPO comparison.
