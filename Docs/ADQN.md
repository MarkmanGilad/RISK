# ADQN — Advantage Dueling DQN

ADQN starts by copying Dueling DQN, exactly as PQN did. It keeps Dueling
DQN's network architecture, replay buffer, Double-DQN target, epsilon-greedy
action selection, target-network synchronization, and Bellman loss. The only
algorithmic addition is a second loss that trains the stored action's centered
advantage directly.

This is the normative implementation specification and now describes the
implemented `ADQN` network and `ADQN_Agent` in `risk/learning/adqn.py` and
`risk/learning/adqn_agent.py`. They are independent sibling implementations:
ADQN does not construct `PQN` and `ADQN_Agent` does not inherit `PQN_Agent`.
For now the raw dueling network and agent plumbing are intentionally copied;
a common base for Dueling DQN, PQN, and ADQN may be considered later.
Sections A-G define the implementation, while Section H lists only the risks
to monitor during the first training run.

## Why change PQN to ADQN?

PQN adds this replay policy loss to the Bellman loss:

```text
policy_loss
    = -mean(stop_gradient(td_advantage) * log pi(a | s))
```

In replay, this creates a practical "bias" toward negative-advantage samples.
The word "bias" here means unequal replay impact, not that the gradient
increases the probability of bad actions. A negative-advantage update still
decreases the bad action's probability.

The unequal impact comes from the current action probability. A good action
usually already has high probability, so `abs(log pi)` is small. A bad action
usually has low probability, so `abs(log pi)` is large:

```text
td_advantage = +2, pi = 0.8
policy-loss contribution = -(+2) * log(0.8) = +0.446

td_advantage = -2, pi = 0.2
policy-loss contribution = -(-2) * log(0.2) = -3.219
```

On-policy sampling partly compensates for this because the 80%-probability
action is sampled much more often than the 20%-probability action. A replay
buffer does not sample according to the current policy: it repeatedly returns
actions selected by older policies and by epsilon exploration. Old bad actions
can therefore receive much more influence than their current probability
would give them on-policy.

**This is a gradient asymmetry, not only a logged-value asymmetry.**
Differentiating `-td_advantage * log pi(a|s)` with respect to
the taken action's raw advantage output gives
`-td_advantage * (1 - pi(a))`. A confident, already-good action (`pi(a)`
near 1) gets a *shrinking* corrective push as it approaches certainty, while
a rare/surprising action (`pi(a)` near 0) gets pushed with close to full
strength. Separately, `td_advantage` itself is unbounded — this game's
rewards run to +-100 for terminal outcomes — so it also acts as an
unclamped linear multiplier on that gradient. ADQN's scaled `tanh` weight
fixes both in one change: dropping `log pi` removes the
`(1 - pi(a))` term entirely (every taken action now gets the same bounded
push regardless of the policy's current confidence in it), and the scaled
`tanh` bounds the previously-unclamped `td_advantage` multiplier while
preserving useful magnitude information over a wider range.

Run `PQN_e_047` showed this failure mode. At the review snapshot, 143 of 151
logged episode-mean policy losses were negative. The negative policy term also
grew large enough to cancel a substantial part of the positive Bellman loss.

ADQN removes `log pi(a | s)` completely. For a replay state `s`, the network
first produces one state value and one raw advantage for every legal action:

```text
V_online(s)
A_online(s, a_i)    for every legal action a_i
```

The agent centers those raw advantages within that one legal-action decision
and reconstructs Q exactly as Dueling DQN does:

```text
mean_A_online(s)
    = mean_i A_online(s, a_i)

A_centered_online(s, a_i)
    = A_online(s, a_i) - mean_A_online(s)

Q_online(s, a_i)
    = V_online(s) + A_centered_online(s, a_i)
```

For the stored replay action `a`, ADQN calculates whether its transition was
better or worse than the online state-value baseline:

```text
td_advantage
    = r
      + gamma * (1 - done) * V_online(s')
      - V_online(s)

advantage_weight
    = stop_gradient(
        ADQN_ADVANTAGE_WEIGHT_SCALE
        * tanh(td_advantage / ADQN_ADVANTAGE_WEIGHT_SCALE)
      )
```

`stop_gradient` means PyTorch `.detach()`. The new loss applies this bounded,
signed weight directly to the stored action's centered advantage:

```text
per_sample_advantage_loss
    = -advantage_weight * A_centered_online(s, a)

advantage_loss
    = mean(per_sample_advantage_loss)

advantage_loss_abs_mean
    = mean(abs(per_sample_advantage_loss))
```

The signed `advantage_loss` supplies gradients. The positive
`advantage_loss_abs_mean` is detached and used only to prevent mixed-sign
samples from hiding the auxiliary activity when calculating its coefficient:

```text
effective_advantage_coef
    = min(
        ADQN_ADVANTAGE_LOSS_COEF,
        ADQN_MAX_ADVANTAGE_LOSS_FRACTION * stop_gradient(q_loss)
        / (stop_gradient(advantage_loss_abs_mean) + epsilon)
      )

total_loss
    = q_loss + effective_advantage_coef * advantage_loss
```

When `td_advantage > 0`, minimizing the new loss raises the stored action's
centered advantage relative to the other legal actions. When
`td_advantage < 0`, it lowers it. Equal and opposite TD advantages produce
equal and opposite bounded weights, independent of the current action
probability. ADQN still uses replay, so this does not make it an on-policy
algorithm or remove every form of replay-distribution bias. It specifically
removes PQN's probability-based asymmetry between old negative- and
positive-advantage actions. Sections A-D define the same calculations in the
exact batching order used for implementation.

## A. Network-head outputs

For one state `s`, the heads produce the same raw values as the Dueling/PQN
network:

```text
V_online(s)                 one scalar from the clean state row
A_online(s, a_i)            one raw advantage for every legal action a_i
```

The target network produces the equivalent values:

```text
V_target(s)
A_target(s, a_i)
```

ADQN needs access to the raw `V` and `A` streams before they are combined.
As in PQN, the network should return those raw streams and the agent should do
the calculations below. The heads and their sizes do not change.

## B. Calculations for `s` and `s'`

For the sampled replay state `s`, calculate all legal-action advantages:

```text
mean_A_online(s) = mean_i A_online(s, a_i)

A_centered_online(s, a_i)
    = A_online(s, a_i) - mean_A_online(s)

Q_online(s, a_i)
    = V_online(s) + A_centered_online(s, a_i)
```

The Bellman prediction uses the stored action `a`:

```text
Q_online_taken = Q_online(s, a)
```

For the next state `s'`, the online network selects the DDQN action:

```text
mean_A_online(s') = mean_j A_online(s', a'_j)

Q_online(s', a'_j)
    = V_online(s')
      + A_online(s', a'_j)
      - mean_A_online(s')

a* = argmax_j Q_online(s', a'_j)
```

The target network evaluates that selected action:

```text
mean_A_target(s') = mean_j A_target(s', a'_j)

Q_target(s', a*)
    = V_target(s')
      + A_target(s', a*)
      - mean_A_target(s')
```

The Bellman target and TD advantage are:

```text
y = r + gamma * (1 - done) * Q_target(s', a*)

td_advantage
    = r
      + gamma * (1 - done) * V_online(s')
      - V_online(s)
```

`td_advantage` is used only as a detached weight for the new advantage loss.
No gradient from that loss may flow through `V_online(s)` or
`V_online(s')`.

## C. Bounded TD-advantage weight

ADQN must preserve the sign:

```text
positive weight   increase the stored action's centered advantage
negative weight   decrease the stored action's centered advantage
```

Use a scaled `tanh` to preserve the sign, remain approximately linear near
zero, and bound the weight. Let `C = ADQN_ADVANTAGE_WEIGHT_SCALE`:

```text
advantage_weight
    = stop_gradient(C * tanh(td_advantage / C))
```

Therefore:

```text
-C < advantage_weight < C
```

The initial scale is `C = 5`. Near zero,
`C * tanh(td_advantage / C) approximately equals td_advantage`; large values
approach `-C` or `+C`. This differs from merely multiplying
`tanh(td_advantage)` by `C`: dividing the input by `C` delays saturation and
retains TD-magnitude information over the wider interval.

`ADQN_ADVANTAGE_WEIGHT_SATURATION = 0.95` is a fraction of the configured
bound, not an absolute weight. Treat a sample as saturated when
`abs(advantage_weight) >= C * 0.95`; use this same definition for
`adqn_advantage_weight_saturated_fraction` and its tests.

This keeps positive and negative updates, prevents a large TD advantage from
giving an arbitrarily large auxiliary gradient, and avoids the replay
`log pi(a|s)` problem seen in PQN. It also avoids introducing a
batch-dependent normalization scale.

`tanh` and the Bellman-relative loss-fraction limit in Section D are both
needed because they control different levels of the update. `tanh` acts on
each replay sample before the batch mean: it bounds that sample's detached TD
weight, and therefore its direct gradient multiplier on the stored action's
centered advantage, to the interval `(-C, C)`. The loss-fraction limit acts
later through one detached coefficient for the whole minibatch: it moderates
the aggregate auxiliary-loss activity relative to `q_loss`, but it does not
bound how strongly one unusually large raw TD advantage can dominate the
other samples inside that minibatch. Loss magnitude also does not determine
the exact gradient magnitude; for example, a centered advantage near zero can
make the per-sample scalar loss small even though an unbounded raw TD advantage
would still be its gradient multiplier. Therefore the fraction limit does not
make `tanh` redundant.

## D. Loss

The Bellman loss remains unchanged:

```text
q_loss = SmoothL1(Q_online(s, a), stop_gradient(y))
```

First calculate the signed loss separately for every replay sample:

```text
per_sample_advantage_loss
    = -advantage_weight * A_centered_online(s, a)

advantage_loss
    = mean(per_sample_advantage_loss)

advantage_loss_abs_mean
    = mean(abs(per_sample_advantage_loss))
```

`advantage_loss` is the signed objective that supplies gradients.
`advantage_loss_abs_mean` is a detached magnitude reference used only to
moderate the effective coefficient. It prevents positive and negative samples
from canceling before their activity is measured.

The total loss is:

```text
effective_advantage_coef
    = min(
        ADQN_ADVANTAGE_LOSS_COEF,
        ADQN_MAX_ADVANTAGE_LOSS_FRACTION
        * stop_gradient(q_loss)
        / (stop_gradient(advantage_loss_abs_mean) + epsilon)
      )

weighted_advantage_loss = effective_advantage_coef * advantage_loss

total_loss = q_loss + weighted_advantage_loss
```

`ADQN_ADVANTAGE_LOSS_COEF` is the normal requested coefficient.
`ADQN_MAX_ADVANTAGE_LOSS_FRACTION` limits the weighted auxiliary loss to a
fraction of the current Bellman loss. The initial comparison value should be
`0.25`, meaning:

```text
abs(weighted_advantage_loss) <= 0.25 * q_loss
```

This is a minibatch-level scalar-activity limit. It complements rather than
replaces Section C's per-sample `tanh` bound.

Use these initial constants:

```text
ADQN_ADVANTAGE_LOSS_COEF            = 0.25
ADQN_MAX_ADVANTAGE_LOSS_FRACTION    = 0.25
ADQN_LOSS_BALANCE_EPSILON           = 1e-8
ADQN_ADVANTAGE_WEIGHT_SCALE         = 5.0
ADQN_ADVANTAGE_WEIGHT_SATURATION    = 0.95
ADQN_GRAD_DIAGNOSTIC_EVERY          = 100 optimizer updates
```

`ADQN_ADVANTAGE_LOSS_COEF` is now `0.25` as parameter tuning after the first
ADQN comparison run; changing it does not alter the loss definition.
`ADQN_MAX_ADVANTAGE_LOSS_FRACTION` bounds
the auxiliary scalar activity relative to `q_loss`, so an excessively large
base coefficient is moderated when the fraction cap binds. This does not
bound the auxiliary-to-Bellman gradient ratio. Use
`adqn_advantage_activity_to_q_loss_ratio`, the two encoder-gradient norms, and
their cosine similarity to judge whether `0.25` is appropriate. If the
activity ratio stays well below the maximum fraction, `0.25` is the binding
constraint; if it sits at the maximum, the fraction cap is binding.

For example:

```text
q_loss                       = 40
advantage_loss               = -500
advantage_loss_abs_mean      = 500
ADQN_ADVANTAGE_LOSS_COEF     = 0.25
maximum permitted magnitude = 0.25 * 40 = 10

effective_advantage_coef     = min(0.25, 10 / 500) = 0.02
weighted_advantage_loss      = 0.02 * -500 = -10
total_loss                   = 40 - 10 = 30
```

`q_loss` and `advantage_loss_abs_mean` are detached only while calculating
`effective_advantage_coef`. The real signed `advantage_loss` in
`weighted_advantage_loss` remains connected to the graph, so it still supplies
a scaled gradient. Do not replace the optimized signed mean with the positive
absolute mean.

Do not implement this by applying `torch.clamp` directly to the weighted loss.
Outside the clamp interval, that would give the auxiliary branch zero gradient.
The detached adaptive coefficient moderates the gradient instead of abruptly
turning it off.

There is no softmax, `Categorical` distribution, `log pi`, probability ratio,
or entropy term in ADQN. The bounded weight controls the auxiliary gradient;
the existing global gradient-norm clipping remains in place after the two
losses are combined.

The auxiliary loss can be negative. That is acceptable: it is a directional
objective, not an error magnitude. Its gradient weight is bounded by -1 and
1, and its weighted scalar contribution is additionally limited relative to
the Bellman loss. This cap controls scalar loss balance, not the exact ratio
of the two gradient norms; the existing global gradient clipping is still
required.

### Gradient-direction diagnostic

Scalar loss ratios cannot show whether the two objectives cooperate or fight
over the shared representation. Before the combined backward pass, measure
their gradients separately on the shared encoder parameters:

```text
q_encoder_grad
    = grad(q_loss, encoder_parameters)

advantage_encoder_grad
    = grad(weighted_advantage_loss, encoder_parameters)

encoder_gradient_cosine_similarity
    = dot(q_encoder_grad, advantage_encoder_grad)
      / (
          norm(q_encoder_grad)
          * norm(advantage_encoder_grad)
          + epsilon
        )
```

Log:

```text
adqn_q_encoder_grad_norm
adqn_advantage_encoder_grad_norm
adqn_encoder_gradient_cosine_similarity
```

Interpret the cosine as:

```text
close to +1   the auxiliary update supports Bellman learning
close to  0   the objectives are mostly independent
below     0   the auxiliary update conflicts with Bellman learning
close to -1   they push the shared encoder in opposite directions
```

Use the already-scaled `weighted_advantage_loss` for this diagnostic so the
logged auxiliary gradient reflects `effective_advantage_coef`. Calculate the
diagnostic before `total_loss.backward()` with graph retention, and do not
write these diagnostic gradients into `.grad`. Because two separate gradient
queries add training cost, run them every
`ADQN_GRAD_DIAGNOSTIC_EVERY = 100` optimizer updates. A zero norm in either
branch must produce a defined cosine value of `0`, not NaN.

### Diagnostic definitions

All diagnostic inputs are detached. Use these exact per-update definitions:

```text
td_error = y - Q_online_taken

adqn_weighted_advantage_to_q_loss_ratio
    = abs(weighted_advantage_loss)
      / (q_loss + ADQN_LOSS_BALANCE_EPSILON)

adqn_advantage_activity_to_q_loss_ratio
    = effective_advantage_coef * advantage_loss_abs_mean
      / (q_loss + ADQN_LOSS_BALANCE_EPSILON)

adqn_advantage_weight_positive_fraction
    = mean(advantage_weight > 0)

adqn_advantage_weight_negative_fraction
    = mean(advantage_weight < 0)

adqn_advantage_weight_saturated_fraction
    = mean(
        abs(advantage_weight)
        >= ADQN_ADVANTAGE_WEIGHT_SCALE
           * ADQN_ADVANTAGE_WEIGHT_SATURATION
      )
```

`adqn_advantage_weight_td_error_correlation` is the Pearson correlation
between detached `advantage_weight` and detached `td_error`. Return `0` when
the minibatch has fewer than two samples or either input has zero variance.

Log the following ordinary means/absolute means directly from their detached
tensors: TD advantage, TD error, online taken Q, Bellman target Q, online V,
and taken centered A. `adqn_a_centered_taken_max` means the maximum absolute
taken centered advantage. On each update,
`adqn_effective_advantage_coef_max` has the same scalar value as
`adqn_effective_advantage_coef`; the trainer's existing `_max` suffix rule
turns the former into the true maximum across the episode while averaging the
latter.

The activity ratio is the quantity guaranteed to stay at or below
`ADQN_MAX_ADVANTAGE_LOSS_FRACTION` (apart from numerical epsilon). The signed
weighted-loss ratio may be smaller because positive and negative samples can
cancel in `advantage_loss`; log both so cancellation remains visible.

Encoder-gradient fields are emitted only on the configured diagnostic update.
They may be absent on other updates; do not fill skipped updates with zeros,
because that would bias episode aggregation.

## E. Action selection and replay

Copy these parts from Dueling DQN without modification:

```text
training action = random legal action with probability epsilon
                = argmax_a Q_online(s, a) otherwise

evaluation action = argmax_a Q_online(s, a)
```

Use the same epsilon schedule, replay capacity, minibatch size, training
frequency, target synchronization, DDQN selection/evaluation split, and
checkpoint cadence as Dueling DQN. This keeps the comparison focused on the
new centered-advantage loss.

## F. Changes from Dueling DQN

The implementation starts from Dueling DQN behavior and makes only these
changes:

1. **Independent sibling network and agent.** `risk/learning/adqn.py` owns
   class `ADQN`, an intentional copy of the raw dueling `(V, A)` architecture
   with the same `forward(state, phase, card_indices, value_mask,
   group_index=None) -> (value_mean, advantage)` contract. It does not import
   or subclass `PQN`. `risk/learning/adqn_agent.py` owns a standalone
   `ADQN_Agent(BaseAgent)` with copied graph batching, grouped Q
   reconstruction, replay, DDQN next-state calculation, epsilon-greedy action
   selection, checkpointing, and training plumbing. It does not inherit from
   `PQN_Agent` or `Dueling_DQN_Agent`. This duplication is deliberate for the
   first implementation so ADQN, PQN, and Dueling DQN remain separate
   algorithm siblings. A shared raw-dueling network/agent base is a possible
   later refactor, not part of this implementation.
2. Keep calculating the exact centered advantage over all legal actions.
3. Calculate `td_advantage` from `V_online(s)` and `V_online(s')`.
4. Convert it to the detached, bounded `advantage_weight` in Section C.
5. Add `advantage_loss` to the unchanged Bellman `q_loss`.
6. Add the six ADQN constants specified in Section D. Store the base
   coefficient, maximum fraction, weight scale, saturation fraction, and
   diagnostic cadence on the agent so future comparison variants can override
   them without changing module globals.
7. Add trainer selection `build_learner_agent("ADQN", ctx)` and use the
   `ADQN_<run_id>` run/checkpoint label.
8. Log `adqn_q_loss`, `adqn_advantage_loss`,
   `adqn_advantage_loss_abs_mean`, `adqn_weighted_advantage_loss`,
   `adqn_total_loss`, the configured base coefficient, maximum fraction, and
   `adqn_advantage_weight_scale`,
   `adqn_effective_advantage_coef` and
   `adqn_effective_advantage_coef_max`,
   `adqn_weighted_advantage_to_q_loss_ratio`,
   `adqn_advantage_activity_to_q_loss_ratio`,
   `adqn_td_advantage_mean`, `adqn_td_advantage_abs_mean`,
   `adqn_td_error_mean`, `adqn_td_error_abs_mean`,
   `adqn_q_value_mean`, `adqn_target_q_mean`, `adqn_grad_norm`, and
   `adqn_grad_norm_clipped`,
   `adqn_advantage_weight_positive_fraction`,
   `adqn_advantage_weight_negative_fraction`,
   `adqn_advantage_weight_saturated_fraction`, and
   `adqn_advantage_weight_td_error_correlation`. To detect a stable Q value
   hiding V/A decomposition drift, also log `adqn_v_online_mean`,
   `adqn_v_online_abs_mean`, `adqn_a_centered_taken_mean`,
   `adqn_a_centered_taken_abs_mean`, and `adqn_a_centered_taken_max`.
   Retain Q diagnostics and the combined gradient norm before clipping, plus
   `adqn_q_encoder_grad_norm`,
   `adqn_advantage_encoder_grad_norm`, and
   `adqn_encoder_gradient_cosine_similarity`.
9. Persist the base coefficient, maximum fraction, weight scale, saturation
   fraction, diagnostic cadence, and all existing Dueling training state in
   full checkpoints. W&B config must record every effective ADQN setting.
   A legacy checkpoint without the weight-scale field must restore scale `1.0`
   because that exactly preserves the former `tanh(td_advantage)` behavior.
10. Add focused coverage in `Temp/tests/test_adqn.py`; extend the existing
    trainer and training-logger tests for factory selection, metric
    aggregation, and W&B configuration. Update the relevant `Docs/` files and
    `Docs/ChangeLog.md` with the implementation.

## G. Required tests before training

1. With `ADQN_ADVANTAGE_LOSS_COEF = 0`, one update matches Dueling DQN's
   Bellman-only loss and action behavior.
2. `A_centered_online` has mean zero separately for every sampled state.
3. `advantage_weight` is detached and always lies between
   `-ADQN_ADVANTAGE_WEIGHT_SCALE` and `+ADQN_ADVANTAGE_WEIGHT_SCALE`.
4. Positive TD advantage increases the stored action's centered advantage;
   negative TD advantage decreases it.
5. No advantage-loss gradient reaches either value output through
   `td_advantage`.
6. DDQN online selection and target evaluation match Dueling DQN.
7. Epsilon decay and greedy evaluation match Dueling DQN exactly.
8. Checkpoint round-trip preserves every configurable ADQN setting, optimizer,
   online/target nets, replay buffer, epsilon, and training counters.
9. `effective_advantage_coef` equals the base coefficient when
   `advantage_loss_abs_mean` is already below the configured fraction of
   `q_loss`.
10. Large positive and negative auxiliary losses are both limited to the
    configured fraction of `q_loss`, while their gradients remain nonzero.
11. The loss-ratio calculation is finite when `advantage_loss_abs_mean` or
    `q_loss` is zero.
12. A minibatch with large equal-and-opposite per-sample losses uses their
    large `advantage_loss_abs_mean` to reduce `effective_advantage_coef`, even
    when the signed `advantage_loss` is zero.
13. Separate encoder-gradient norms are finite and the cosine is in
    `[-1, 1]`; if either encoder-gradient norm is zero, the logged cosine is
    exactly zero rather than NaN.
14. Computing the diagnostic does not populate or change parameter `.grad`
    values and does not change the subsequent combined optimizer update.
15. V and centered-A diagnostic metrics are present and finite; a synthetic
    case where V and centered A move in opposite directions while Q stays
    fixed is visible in those metrics.
16. Advantage-weight positive, negative, and saturated fractions are within
    `[0, 1]`, and their applicable categories are internally consistent.
17. Advantage-weight/Bellman-TD-error correlation is finite and lies in
    `[-1, 1]`; zero variance in either input produces a defined value of zero.
18. Episode aggregation reports the weighted mean
    `adqn_effective_advantage_coef` and the true per-episode maximum
    `adqn_effective_advantage_coef_max`.
19. `adqn_advantage_activity_to_q_loss_ratio` does not exceed the configured
    maximum fraction, including a mixed-sign canceling minibatch, while the
    signed weighted-loss ratio is allowed to be smaller.
20. Skipped gradient-diagnostic updates omit the three encoder-gradient fields
    rather than logging false zeros; diagnostic updates aggregate normally.

## H. First-run monitoring and stop conditions

Sections A-G are the implementation decisions. The following are monitored
risks, not unresolved design comments:

1. **Linear-loss drift.** The bounded weight and adaptive coefficient limit
   each update, but the linear advantage objective has no finite minimum.
   Watch V and centered-A diagnostics together with Q. Stop the run if V and
   centered A diverge in opposite directions while Q stays stable or if any
   value becomes nonfinite. A finite margin/Huber auxiliary loss is a future
   algorithm, not part of this implementation.

2. **Tanh saturation.** Large TD advantages produce weights near the configured
   `-5` or `+5` bounds. Use the specified saturation fraction to measure how
   often the auxiliary signal becomes binary. Record the scale with every run;
   changing it defines a different comparison setting.

3. **Adaptive-coefficient variability.** The coefficient is deliberately tied
   to the current minibatch Q loss and can become noisy or approach zero.
   Compare its episode mean and maximum. EMA smoothing is a possible later
   experiment, not part of the first implementation.

4. **Gradient conflict.** The encoder-gradient cosine is the direct diagnostic:
   positive values indicate cooperation and negative values indicate conflict.
   If it remains negative while Q loss and learning performance are worse than
   the Dueling/PQN_e0 control, treat the auxiliary loss as harmful rather than
   increasing its coefficient.

5. **TD-weight definition.** The first implementation uses the V-based
   TD advantage from Section B. A bounded Bellman-error weight would define a
   different algorithm and must be tested as a separate named variant rather
   than substituted silently.
