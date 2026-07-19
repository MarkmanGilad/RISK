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
`remember()` or branching the trainer on PPO. Build
`PPO_Agent` in the caller (or in `trainer.py`'s `main()` as a third commented
block next to DQN/Dueling), pass it into the existing `Trainer`, and rely on
`label`/`agent_class`/`model_class` in the checkpoint path and W&B config to
distinguish it. Use `resume=False` and a fresh run id for the first PPO run,
same as Dueling.

Beyond the optional-metrics hook, the trainer loop should stay unchanged.
Adding a commented PPO import and construction block in `main()` is fine when
implementation starts; changing `Trainer.train(...)` to branch on PPO is not.

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
