# PPO

`PPO_Agent` is an on-policy, injected-action actor-critic. It shares the graph
representation with the DQN learners, but it collects an ordered rollout with
one frozen policy, trains on that rollout for several epochs, then discards it.
PPO-specific behavior is contained in `risk/learning/ppo_agent.py`; it does
not alter the environment, trainer, game rules, or the DQN-family learners.

## Network and collection

`PPO_Net` receives one clean graph row for the state and one action-injected
row per legal action. It returns one logit per legal action and one state value
`V(s)` from the clean graph row.

During training, `act()` samples `Categorical(logits=...)`; during evaluation,
it takes the highest-logit action. At collection time it stores detached
collection-policy data:

```text
(state, action, action_index, old_log_prob, old_value,
 reward, next_state, done, gae_boundary)
```

`old_log_prob` is `log pi_old(action | state)` and is the fixed reference for
the PPO ratio. `old_value` is `V_old(state)`, used as the critic baseline and
as a later bootstrap value. The buffer preserves collection order.

One stored transition is one **learner transition**: it begins when the learner
acts and ends when the learner is next due to act, after opponents have played.
Its reward already aggregates that interval. The rollout contains 1,024 learner
transitions.

True game termination has `done=True`. A maximum-step cutoff remains
non-terminal and is marked `gae_boundary=True`, preventing a target from
including rewards after the next reset game.

## 16-step bootstrapped targets

PPO uses `PPO_N_STEP = 16`. For every transition `t`, it sums up to sixteen
consecutive learner rewards and then uses a critic bootstrap:

$$
G_t^{(h)} = \sum_{k=0}^{h-1} \gamma^k r_{t+k}
             + \gamma^h V_{\mathrm{next},t}
$$

`h` is at most 16. The advantage and value target are:

$$
\hat A_t = G_t^{(h)} - V_{\mathrm{old}}(s_t),
\qquad
V_{\mathrm{target},t} = G_t^{(h)}
$$

The bootstrap source is selected as follows:

- **Ordinary 16-step continuation:** reuse `transitions[t + 16].old_value`.
  This is already `V_old(s[t + 16])`, stored when that later action was
  collected -- no new forward pass is required.
- **Terminal inside the reward window:** include that terminal transition's
  reward, then bootstrap with zero.
- **Forced episode boundary inside the reward window:** include its reward,
  then use one clean value-only evaluation of that transition's `next_state`.
- **Rollout tail before 16 rewards:** use one clean value-only evaluation of
  the last transition's `next_state`.

A terminal or boundary at exactly `t + 16` is beyond the target's reward
window; the ordinary stored `old_value[t + 16]` is used. If both `done` and
`gae_boundary` are true, `done` wins and the bootstrap is zero.

All clean boundary/tail values are deduplicated and evaluated under
`torch.no_grad()` before the first optimizer step, so they are values from the
unchanged collection-time critic. `_clean_value_entry(...)` and
`_forward_grouped(...)` batch these clean rows without building legal actions
or action-injected graphs. Thus PPO never evaluates all 1,024 successor states
with their legal-action sets.

## PPO optimization

Advantages are normalized over the rollout. PPO makes up to four shuffled
epochs of 64-transition minibatches. For each minibatch it recomputes the
current policy and critic and applies the clipped surrogate:

$$
r_t(\theta) = \exp\!\left(
  \log \pi_\theta(a_t \mid s_t)
  - \log \pi_{\mathrm{old}}(a_t \mid s_t)
\right)
$$

$$
L^{\mathrm{CLIP}} =
\min\!\left(
  r_t(\theta)\hat A_t,
  \operatorname{clip}\!\left(r_t(\theta), 0.8, 1.2\right)\hat A_t
\right)
$$

The total loss is clipped policy loss, plus weighted Smooth-L1 critic loss
against `G_t^(16)`, minus an entropy bonus. Gradients are clipped. After the
first optimizer step, remaining minibatches are skipped when sampled k3
approximate KL exceeds `0.02`; the crossing minibatch is not applied.

The agent reports ordinary PPO optimization diagnostics plus:

- `ppo_target_horizon_mean`: mean number of rewards in each target;
- `ppo_target_bootstrap_fraction`: fraction structurally using a value
  continuation, whether or not the predicted scalar happens to be zero.

These are rollout-level values and are averaged equally across rollout updates
by the existing trainer metrics aggregation.

| Setting | Value |
| --- | ---: |
| rollout length | 1,024 learner transitions |
| n-step horizon | 16 learner transitions |
| epochs | 4 |
| minibatch size | 64 |
| clipping epsilon | 0.2 |
| target KL | 0.02 |
| value-loss coefficient | 0.1 |
| entropy coefficient | 0.01 |
| learning rate | 7.5e-5 |

## Relation to the PPO paper

This remains PPO: it uses an on-policy frozen rollout, a value baseline,
multiple minibatch epochs, and the clipped policy objective. The original PPO
paper's practical actor-critic configuration uses GAE, which is a
lambda-weighted blend of n-step estimates rather than this one fixed 16-step
cutoff. The fixed horizon is a simpler variance-control experiment; it retains
the critic bootstrap without the previous redundant successor action-graph
work. See [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Focused verification

`Temp/tests/test_ppo.py` covers stored successor-value reuse, terminal and
boundary semantics, clean value-only tail evaluation, no reset-game
propagation, PPO update metrics, and checkpointing. The existing
`Temp/tests/test_training_logger.py` configuration check ensures the exported
`PPO_N_STEP` setting reaches W&B configuration.
