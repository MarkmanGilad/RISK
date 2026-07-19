# Policy Duel DQN (PDDQN)

Policy Duel DQN (**PDDQN**) is a proposed, unimplemented hybrid of PPO-style
on-policy policy learning and Dueling Double-DQN replay learning.  It is not a
rename of PQN and it is not Q-Prop.  Its central constraint is architectural:

> Reuse the existing Dueling raw `(V, A)` network outputs for Q values, policy
> logits, and a derived policy value.  Do not add a separate actor network,
> Q network, or learned policy-value head.

The proposed implementation starts from Dueling DQN, just as ADQN did.  It
would own an independent copied raw-dueling network and agent plumbing; no
implementation is authorized by this document.

## 1. Motivation and history

Dueling DQN is sample-efficient because it retains transitions in replay and
learns an action-value Bellman target from them.  PPO is stable because it
trains a stochastic policy only from a recent rollout collected by that
policy, then discards the rollout once it has been reused for a small number
of epochs.

PQN attempted to use both ideas with one network, but applied a policy loss to
random replay samples:

```text
-stop_gradient(td_advantage) * log pi(a | s)
```

That makes old epsilon and older-policy actions directly train the current
policy.  The first PQN comparison was worse than Dueling DQN, so replayed
policy-gradient updates are not the starting point for a new hybrid.

PDDQN keeps the useful separation:

```text
fresh rollout  -> PPO policy and policy-value losses only
replay buffer  -> Double-DQN Q loss only
```

Every fresh transition may be appended to both stores.  The short rollout is
cleared after its PPO update; the replay copy remains available for later Q
learning.  Thus PDDQN does not throw away the environment experience, while
it never applies the PPO actor loss to stale replay data.

## 2. Compact network contract

For every decision, the network returns exactly Dueling's existing raw
outputs:

```text
V_head(s)                    one scalar state output
A_raw(s, a_i)                one scalar for each legal action a_i
```

For a legal-action group of size `N`, derive Dueling Q values exactly as
usual:

```text
A_centered(s, a_i)
    = A_raw(s, a_i) - mean_j A_raw(s, a_j)

Q(s, a_i)
    = V_head(s) + A_centered(s, a_i)
```

Use the same raw action scores as categorical policy logits:

```text
pi(a_i | s) = softmax_i(A_raw(s, a_i))
```

Finally derive the policy state value from values that already exist:

```text
Vpi(s) = sum_i pi(a_i | s) * Q(s, a_i)
```

`Vpi` is not a new learned head.  It is only grouped softmax, centering,
multiplication, and summation after the raw Dueling output.  This project’s
Dueling implementation already batches every legal action for both the
current state and the next state; PDDQN adds no new GNN head and no separate
actor/critic network forward for these calculations.

PDDQN must not claim that `V_head` itself is `Vpi`: mean-centered Dueling
makes `V_head` the uniform mean of legal-action Q values, whereas `Vpi` is
weighted by the softmax policy.

## 3. Data collection

Training actions are sampled from the categorical policy:

```text
a_t ~ Categorical(logits=A_raw(s_t, .))
```

For every transition, retain:

```text
state, action, reward, next_state, done
old_log_prob = log pi_behavior(action | state)
old_vpi      = Vpi_behavior(state)
```

Append the transition to both:

1. an ordered, short on-policy rollout buffer; and
2. the persistent DQN replay buffer.

PDDQN version 1 must not add epsilon-random actions during this collection.
The stored `old_log_prob` must describe the policy that actually selected the
action for PPO's ratio to be meaningful.  Exploration comes from categorical
sampling and the PPO entropy bonus.  A future mixed softmax/epsilon behavior
policy would need to store the probability under that exact mixture; it is a
separate design.

## 4. Fresh-rollout PPO objective

For the fresh rollout only, calculate GAE from the derived policy value:

```text
delta_t
    = r_t + gamma * (1 - done_t) * Vpi(s_{t+1}) - Vpi(s_t)

advantage_t = GAE(delta_t, gamma, lambda)
return_t    = advantage_t + old_vpi_t
```

Normalize advantages before the policy objective.  Recompute the current
policy from `A_raw` and form PPO's usual ratio:

```text
ratio_t
    = exp(log pi_current(a_t | s_t) - old_log_prob_t)

policy_loss
    = -mean(min(
        ratio_t * advantage_t,
        clip(ratio_t, 1 - eps, 1 + eps) * advantage_t
      ))

value_loss
    = SmoothL1(Vpi_current(s_t), stop_gradient(return_t))

entropy_bonus
    = entropy(pi_current(. | s_t))

ppo_loss
    = policy_loss
      + PPO_VALUE_LOSS_COEF * value_loss
      - PPO_ENTROPY_COEF * entropy_bonus
```

The stored rollout is reused for the existing small number of PPO epochs,
with ratio clipping and approximate-KL early stopping.  It is then cleared.
This is the only place in PDDQN where `log pi` is used for optimization.

## 5. Replay Double-DQN objective

Replay transitions train only the Q interpretation of the same raw outputs:

```text
a* = argmax_a Q_online(s', a)

y_Q
    = r + gamma * (1 - done) * Q_target(s', a*)

q_loss
    = SmoothL1(Q_online(s, a_taken), stop_gradient(y_Q))
```

This is Dueling Double-DQN without a policy-gradient term.  It is valid for
the older transitions because Q learning is off-policy.  Do not calculate
PPO ratios, GAE, entropy, or policy loss on replay samples.

## 6. Update schedule

The first experiment uses sequential updates, not one blended backward pass:

```text
1. Collect a full fresh rollout while sampling the softmax policy.
2. Run the PPO update on that rollout only; clear the rollout.
3. Run a configured, limited number of Double-DQN replay updates.
4. Collect the next rollout from the network after those replay updates.
```

Conceptually PDDQN optimizes both `ppo_loss` and `q_loss`, but sequential
updates preserve PPO's meaning: during its multi-epoch pass, no replay Q
update changes the policy between PPO minibatches.  A single blended loss
would be a later named variant, because it makes PPO's KL/ratio diagnostics
harder to interpret.

Replay Q updates still change `A_raw` and therefore change the policy logits.
This is intentional, but it is PDDQN's central risk.  Measure the policy KL
before and after the replay-Q phase; collect a new rollout immediately after
that phase rather than continuing an old one.

## 7. Relationship to PQN and Q-Prop

### PQN

PQN uses the same compact raw `(V, A)` output and derives Q and softmax policy
in the same way.  Its difference is where the policy loss is evaluated:

```text
PQN:    replay Q loss + replay log-pi policy loss
PDDQN:  replay Q loss + fresh-rollout PPO policy/value losses
```

PDDQN removes PQN's replayed `td_advantage * log pi` term completely.  It
uses PPO's stored behavior log probability, ratio clip, and KL stop only on
fresh policy-sampled data.

### Q-Prop

Q-Prop also combines an on-policy policy gradient with an off-policy Q
critic, but it normally has separate actor and critic networks and uses the
critic as a control variate for the actor gradient.  It was developed mainly
for continuous actions.  PDDQN instead uses one Dueling action-score stream
for both a discrete categorical policy and Q values; its PPO policy gradient
is unchanged rather than Q-Prop's Taylor/control-variate estimator.

PDDQN is therefore cheaper in network structure but more tightly coupled:
replay Q updates can directly move its policy logits.  It must be evaluated as
an experimental multi-objective learner, not assumed to inherit Q-Prop's
stability claims.

## 8. Required diagnostics

Log every existing DQN and PPO metric, plus:

```text
pddqn_q_loss
pddqn_policy_loss
pddqn_value_loss
pddqn_entropy_bonus
pddqn_vpi_online_mean
pddqn_vpi_return_mean
pddqn_ppo_approx_kl
pddqn_ppo_clip_fraction
pddqn_replay_updates_per_rollout
pddqn_policy_kl_after_replay_q_phase
pddqn_q_encoder_grad_norm
pddqn_ppo_actor_encoder_grad_norm
pddqn_ppo_value_encoder_grad_norm
pddqn_q_vs_actor_encoder_cosine
pddqn_q_vs_value_encoder_cosine
```

The policy-KL-after-replay metric is essential: it measures the policy change
that PPO's own ratio clip did not directly limit.  Compute sparse
branch-gradient diagnostics before the respective backward passes, without
writing them into parameter `.grad`.

## 9. Required tests before training

1. Raw network output and Dueling Q reconstruction exactly match Dueling DQN
   when PDDQN policy/replay extras are disabled.
2. `pi` is `softmax(A_raw)` separately for every legal-action group.
3. `Vpi` equals `sum(pi * Q)` and differs from `V_head` for a non-uniform
   policy and non-equal Q values.
4. A collected action's stored `old_log_prob` equals the sampling policy's
   log probability; no epsilon action can enter version-1 PPO rollouts.
5. PPO loss is evaluated only on the ordered fresh rollout, never on replay.
6. Q loss is evaluated on replay and matches existing Dueling DDQN targets.
7. PPO's GAE uses derived `Vpi`, including correct terminal and cutoff
   boundaries.
8. No replay Q optimizer step occurs during a PPO multi-epoch update.
9. Replay Q updates can change policy logits, and the post-replay policy-KL
   diagnostic detects that change.
10. Checkpoint round-trip preserves network, target network, optimizer,
    ordinary replay, pending rollout, counters, and all PDDQN settings.

## 10. Comparison plan and stop conditions

Use repeated matched seeds and report both environment steps and optimizer
steps.  Required controls are:

1. **PPO-only compact control:** same raw network and softmax collection,
   but no replay Q updates.
2. **Dueling-policy-collection control:** same softmax collection and replay
   Q updates, but no PPO updates.
3. **PDDQN:** both phases enabled.

Success requires better held-out evaluation at comparable environment-step
budgets than both controls, without persistent nonfinite values, Q-target
divergence, policy-KL jumps after replay, or sustained negative Q-versus-PPO
encoder-gradient alignment.  Do not judge success from a decreasing PPO or Q
loss alone.

Stop and inspect the design if any of the following persists:

- replay-Q phases repeatedly create large policy KL jumps;
- PPO immediately hits its KL stop or clip saturation after replay phases;
- online Q and target Q separate or become nonfinite;
- one objective's encoder gradient consistently opposes the other; or
- performance is no better than either single-objective control.
