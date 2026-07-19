# VQN — Value Q-Learning Network

VQN is a proposed experiment, not an implemented agent.  It keeps Dueling
DQN's action-value learning and replaces PQN's replay policy-gradient loss
with an auxiliary Bellman loss for the *derived value of a policy*.  The
experiment asks a narrower question than PQN did:

> Does making the Q network internally consistent with the value of its own
> action-selection policy improve learning, without applying a replayed
> `log pi` policy-gradient update?

VQN starts from Dueling DQN, just as ADQN did.  Its network is an independent
copy of Dueling's raw `(V, A)` architecture, and its agent is an independent
sibling with Dueling's graph batching, replay, Double-DQN target,
epsilon/rollout plumbing, checkpointing, and logging copied explicitly.  No
implementation is authorized by this document yet.

## 1. Intuition and history

Dueling DQN learns an action value for each legal action:

```text
Q(s, a) = V_head(s) + A(s, a) - mean_a' A(s, a')
```

Its ordinary Smooth L1 Bellman loss is already the primary learning signal:

```text
y_Q      = r + gamma * (1 - done) * Q_target(s', a*)
a*       = argmax_a Q_online(s', a)
q_loss   = SmoothL1(Q_online(s, a_taken), stop_gradient(y_Q))
```

PQN kept that loss and added a replay policy-improvement loss:

```text
policy_loss = -mean(stop_gradient(td_advantage) * log pi(a_taken | s))
```

The first PQN comparison was worse than Dueling DQN.  The likely failure mode
is not simply a bad scalar coefficient: `log pi` makes the update magnitude
depend on the current probability of a replayed action, while replay contains
actions selected by older policies and epsilon exploration.  ADQN removed
that probability-dependent objective entirely and instead updates centered
advantages directly.

VQN takes a different route.  It also removes the `log pi` loss, but asks
whether a policy-value Bellman constraint can improve the shared Q
representation.  This resembles the useful part of the "equation in the
loss" intuition: alongside fitting individual action Q targets, fit the
expected Q of a clearly defined policy to its one-step Bellman target.

## 2. Do not reuse the raw Dueling V head as a policy value

The raw Dueling head is not automatically the value of the softmax policy.
Mean-centering fixes its meaning to the uniform average of the legal-action
Q values:

```text
V_head(s) = mean_a Q(s, a)
```

For a policy `pi`, the desired state value is instead:

```text
V^pi(s) = sum_a pi(a | s) * Q(s, a)
```

These are equal only for a uniform policy (or a special set of equal Q
values).  VQN must therefore derive `V^pi` from every legal-action Q value;
it must not apply a new Bellman loss directly to `V_head`.

### Computation cost relative to this project's Dueling DQN

VQN does **not** require an additional legal-action GNN batch. This project's
Dueling DQN already scores every legal action of each current replay state
`s`: its centered-Q reconstruction needs the mean advantage over that whole
legal-action group before it can select `Q(s, a_taken)`. It already also
scores every legal action of `s'` with both online and target networks for
Double-DQN selection and evaluation.

VQN reuses those existing raw advantages and reconstructed Q values. Its
derived policy value adds only inexpensive vector operations within each
already-scored legal-action group:

```text
pi  = softmax(A)
Vpi = sum(pi * Q)
```

The fresh-rollout design in Section 4 can add environment-collection cost,
but the `Vpi` calculation itself does not add a network head, a separate GNN
forward pass, or per-action optimizer updates relative to the existing
Dueling training batch.

## 3. Policy and derived value

The initial VQN experiment uses a temperature-one softmax policy over the
raw action advantages:

```text
pi_online(a_i | s) = softmax_i(A_online(s, a_i))

Q_online(s, a_i)
    = V_head_online(s)
      + A_online(s, a_i)
      - mean_j A_online(s, a_j)

Vpi_online(s)
    = sum_i pi_online(a_i | s) * Q_online(s, a_i)
```

For bootstrapping, calculate the same quantity with a frozen value-bootstrap
network:

```text
pi_bootstrap(a_j | s') = softmax_j(A_bootstrap(s', a_j))

Vpi_bootstrap(s')
    = sum_j pi_bootstrap(a_j | s') * Q_bootstrap(s', a_j)
```

The value target and auxiliary loss are:

```text
y_V      = r + gamma * (1 - done) * Vpi_bootstrap(s')
v_loss   = SmoothL1(Vpi_online(s), stop_gradient(y_V))
```

Unlike PQN's policy loss, `v_loss` is a non-negative prediction error.  It
has no `log pi` term, no signed TD multiplier, and no direct instruction to
increase or decrease only the sampled action.  Its gradient flows through all
current-state legal-action Qs and the softmax weighting that defines the
policy value.

## 4. On-policy data is required for the first experiment

For an observed transition `(s, a, r, s')`, `y_V` is a sample of the policy
value target only if `a` was sampled from the policy whose value is being
learned.  Old DQN replay does not meet this condition: it contains
epsilon-greedy and older-network actions.  Using it unchanged would quietly
turn the new auxiliary loss into an uncontrolled off-policy value estimate.

The first VQN version therefore uses two data paths:

1. **DQN replay path.** Keep Dueling DQN's existing replay buffer and its
   unchanged Double-DQN `q_loss`.
2. **Fresh policy-value rollout path.** At the start of a short rollout,
   freeze a snapshot of the online network.  Sample every training action
   from that snapshot's softmax policy, and retain those transitions only for
   one VQN value update.  Use the same snapshot as the detached
   value-bootstrap network.  Then discard the rollout transitions and collect
   a fresh rollout before the next VQN value update.

This one-update-per-frozen-rollout rule keeps the policy that generated the
action, the online policy evaluated on the left side before its update, and
the bootstrap policy aligned.  It avoids pretending that stale replay is
on-policy.  It is intentionally conservative and may be slower than DQN
replay; measuring that cost is part of the experiment.

A later named variant may use stale replay only after it stores each action's
behavior probability and applies an explicitly chosen off-policy correction
(for example, clipped importance sampling, Retrace, or V-trace).  That is not
part of VQN version 1.

## 5. Total loss and safeguards

The VQN update combines the unchanged DQN loss with the new value loss:

```text
effective_value_coef
    = min(
        VQN_VALUE_LOSS_COEF,
        VQN_MAX_VALUE_LOSS_FRACTION * stop_gradient(q_loss)
        / (stop_gradient(v_loss) + VQN_LOSS_BALANCE_EPSILON)
      )

total_loss = q_loss + effective_value_coef * v_loss
```

The detached Bellman-relative cap follows ADQN's safety principle: an
auxiliary experiment must not silently overwhelm the proven Q-learning
objective.  Since `v_loss` is non-negative, cancellation is not a concern;
the cap is only loss-balance protection.  Initial constants, if implementation
is approved, should be deliberately conservative and tuned only against a
matched control:

```text
VQN_VALUE_LOSS_COEF          = 0.05     # proposed starting point
VQN_MAX_VALUE_LOSS_FRACTION  = 0.25
VQN_LOSS_BALANCE_EPSILON     = 1e-8
```

`VQN_VALUE_LOSS_COEF = 0` must reproduce the matched Dueling control's
optimizer update exactly, subject only to the separate rollout collection
mode.

## 6. Changes from Dueling DQN

The eventual implementation should start from Dueling DQN and make only
these changes:

1. Add `risk/learning/vqn.py` with a standalone `VQN` network copied from
   Dueling's raw `(V, A)` network.  It returns the same raw value and action
   advantage outputs; it must not subclass or import `PQN` or `ADQN`.
2. Add `risk/learning/vqn_agent.py` with a standalone `VQN_Agent`, copying
   Dueling agent plumbing rather than inheriting from PQN or ADQN.
3. Add helpers that batch all legal actions for a state and calculate both
   centered Q values and `Vpi = sum(pi * Q)`.
4. Preserve DQN replay and Double-DQN `q_loss` without modification.
5. Add the short frozen softmax-rollout buffer described in Section 4, used
   only for `v_loss`; do not put its stale entries back into the policy-value
   path after an update.
6. Snapshot the network before each value rollout and use that snapshot as the
   detached value-bootstrap network for the one subsequent VQN update.
7. Add VQN constants, trainer selection `build_learner_agent("VQN", ctx)`,
   the `VQN_<run_id>` label, checkpoint persistence, and W&B configuration.
8. Log `vqn_q_loss`, `vqn_value_loss`, `vqn_weighted_value_loss`,
   `vqn_total_loss`, `vqn_effective_value_coef`,
   `vqn_value_activity_to_q_loss_ratio`, `vqn_vpi_online_mean`,
   `vqn_vpi_target_mean`, `vqn_vpi_td_error_mean`, policy entropy, and the
   existing Q/gradient diagnostics.  Add separate encoder-gradient norms and
   their cosine similarity for Q versus value loss, using the same sparse
   diagnostic cadence as ADQN.

## 7. Required tests before a run

1. Centered advantages have mean zero separately for every legal-action
   group, and reconstructed Q equals Dueling's Q calculation.
2. `Vpi` equals `sum(softmax(A) * Q)` for each state, including groups with
   different numbers of legal actions.
3. The raw Dueling `V_head` and derived `Vpi` are demonstrably different for
   a non-uniform softmax and non-equal Q values.
4. `v_loss` is finite, non-negative, and uses detached bootstrap values.
5. Its gradients reach current-state Q/advantage outputs but not the frozen
   bootstrap snapshot.
6. A VQN value rollout samples actions from its frozen softmax snapshot and
   is consumed after exactly one value update.
7. With `VQN_VALUE_LOSS_COEF = 0`, the VQN update equals the matched
   Dueling-control update.
8. The effective coefficient is finite and the value-loss activity ratio does
   not exceed `VQN_MAX_VALUE_LOSS_FRACTION` apart from numerical epsilon.
9. The Q and value encoder-gradient diagnostics are finite; their cosine is
   in `[-1, 1]`, with a defined zero result if either norm is zero.
10. Full checkpoint round-trip preserves online/target nets, optimizer,
    ordinary replay, any pending value rollout/snapshot state, constants, and
    counters.

## 8. Experiment plan and success criteria

Use matched seeds, reward settings, model size, training budget, and
evaluation protocol.  The comparison must include:

1. **Dueling-softmax control:** the copied Dueling learner with the same
   frozen softmax rollout collection but `VQN_VALUE_LOSS_COEF = 0`.  This
   isolates changed behavior collection from the new loss.
2. **VQN:** the same setup with the proposed value auxiliary loss enabled.
3. **Existing Dueling epsilon-greedy baseline:** retained as a practical
   reference, but not used alone to attribute a difference to `v_loss`.

Judge VQN by evaluation win rate/score, sample efficiency measured in game
steps, Q loss and TD-error stability, value-loss activity, gradient alignment,
and wall-clock cost.  Do not call VQN an improvement merely because
`v_loss` declines: it succeeds only if it is at least as stable as the
softmax Dueling control and improves held-out evaluation results across
repeated seeds.

## 9. Open risks

1. **Redundancy.** DQN's Q loss may already supply all useful value
   information; the new loss may only duplicate or dilute it.
2. **Behavior change.** Softmax rollout collection is a material change from
   epsilon-greedy Dueling.  The matched zero-coefficient control is mandatory.
3. **Cost.** Fresh, one-update rollouts can be much less sample-efficient than
   replay.  Any performance comparison must report game steps and wall-clock
   time.
4. **Gradient conflict.** A positive scalar auxiliary loss can still conflict
   with the Q objective in the shared encoder.  Use the specified gradient
   cosine rather than inferring cooperation from scalar losses.
5. **Policy definition.** Temperature, entropy regularization, or an
   off-policy correction each define a different algorithm.  Do not add any
   of them to VQN version 1 without a separately named design change.
