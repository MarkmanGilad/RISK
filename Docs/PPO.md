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
against `G_t^(16)`, minus an entropy bonus. Gradients are clipped. After every
complete epoch, PPO recomputes sample-weighted k3 approximate KL across the
full cached rollout under `torch.no_grad()`. If it exceeds `0.02`, the
completed epoch is retained and PPO does not begin another epoch.

### Global post-epoch KL stopping (`PPO_312`)

**Status: implemented for fresh PPO_312; PPO_311 is unchanged.** PPO_311 shows that the 16-step
targets learn, but its current minibatch-level KL check stops every late update
after only a few of the 16 minibatches in one epoch. A single 64-transition
minibatch is a noisy KL estimate because legal-action counts and advantages
vary substantially. Its rejected batch can exceed `0.02` even while the mean
KL of applied minibatches is near `0.003`; the current mean omits the rejected
batch, so it cannot diagnose that mismatch by itself.

PPO_312 retains `PPO_TARGET_KL = 0.02`, the
16-step targets, learning rate, clipping, and gradient clipping. It changes
only when KL decides whether another epoch may begin:

1. Run all 16 shuffled 64-transition minibatches in the first epoch. Do not
   use the normal target KL to abort a minibatch or a partial epoch.
2. After every complete epoch, recompute the chosen-action log-probability for
   the complete cached rollout under `torch.no_grad()`, in the same safe
   64-transition batches. Compare it to the stored `old_log_prob` values and
   calculate the sample-weighted global k3 estimate:

   $$
   \widehat{\mathrm{KL}} =
   \frac{1}{N}\sum_{t=1}^{N}
   \left(r_t - 1 - \log r_t\right)
   $$

3. If that full-rollout KL exceeds `PPO_TARGET_KL`, retain the completed epoch
   and do not start the next one. Otherwise begin the next shuffled epoch, up
   to `PPO_EPOCHS`.
4. The full-rollout KL pass must reuse the rollout's cached graph entries and
   stored action indices. It must not rebuild legal actions, evaluate successor
   states, or create a 1,024-transition GPU batch.
5. Preserve existing clipping and gradient clipping. Do not add a normal
   minibatch-level target-KL stop to this experiment. If a later measurement
   demonstrates a genuinely unsafe within-epoch jump, add a separate,
   explicitly named emergency guard in a new isolated experiment rather than
   silently turning the normal noisy check back on.

This guarantees at least one full use of every on-policy sample per rollout,
instead of PPO_311's late behavior of applying roughly four minibatches
(about 25% of a rollout) before stopping. It still limits the policy change
before later epochs. The post-epoch pass also runs after the final epoch as a
consistent diagnostic. `ppo_post_epoch_kl` is the final completed epoch's KL,
`ppo_post_epoch_kl_max` is the maximum across completed epochs, and
`ppo_kl_stop_epoch` is the one-based epoch after which a high KL blocked a
further epoch; it is `PPO_EPOCHS` when all epochs run.

The implementation is limited to `ppo_agent.py` and its focused PPO tests;
the launcher selects fresh PPO_312. The environment, DQN, Dueling DQN, and
other learners remain unchanged.

Focused tests must prove that a minibatch above `PPO_TARGET_KL` does not stop a
first epoch, the global sample-weighted KL is computed across all cached
transitions, a high global KL prevents epoch two, and a low global KL permits
it. Keep `ppo_approx_kl` as the applied-minibatch diagnostic for chart
continuity; add `ppo_post_epoch_kl`, `ppo_post_epoch_kl_max`, and a 1-based
`ppo_kl_stop_epoch` for the new decision. `ppo_early_stopped` and
`ppo_early_stop_kl` continue to report whether and why future epochs were
skipped, but now refer to global post-epoch KL.

**Implementation decisions:**

- **Register `ppo_post_epoch_kl` and `ppo_kl_stop_epoch` in
  `unweighted_update_metrics` — but not `ppo_post_epoch_kl_max`.** Confirmed
  by reading `trainer.py`'s `_aggregate_update_metrics`: a key ending in
  `_max` takes the maximum unconditionally, before any weight is ever
  applied, so `unweighted_update_metrics` membership is inert for it. This
  matches the existing precedent — `ppo_early_stop_kl_max` is *not* in that
  set today, on purpose. "Register all new rollout-level metrics" as
  currently worded would put `ppo_post_epoch_kl_max` there too, diverging
  from that precedent rather than following it; only the two non-`_max`
  names actually need registration to avoid being weighted by
  `ppo_optimizer_steps_per_update` like a per-minibatch loss.
- **Define `ppo_kl_stop_epoch` for the no-early-stop case.** It's only
  meaningful paired with `ppo_early_stopped=True`. When all `PPO_EPOCHS`
  complete without the global KL ever exceeding the target, set it to
  `PPO_EPOCHS` rather than leaving it `0`/unset — a `0` would misread as
  "stopped after epoch 0" on a chart. `ppo_early_stopped=False` stays the
  unambiguous "did it stop at all" signal either way.
- **Compute the global KL as one running sum over all `N` cached transitions
  divided by `N`, not an equal-weighted average of the 16 per-chunk means.**
  The two coincide only because `PPO_ROLLOUT_LENGTH = 1024` divides evenly by
  `PPO_MINIBATCH_SIZE = 64` today. If either constant changes later and the
  last chunk becomes a different size, an equal-weighted chunk average would
  silently misweight it — the same bug class already fixed once for
  `ppo_value_to_policy_encoder_grad_ratio`'s minibatch weighting.
- **Reuse `_evaluate_indices(...)`/`_forward_grouped(...)` for the
  log-probability recompute** rather than adding a new log-prob-only path.
  They already return both `values` and `log_probs` from one shared-encoder
  forward pass per chunk; this check simply ignores the `values` it doesn't
  need, instead of duplicating the forward-pass logic.
- **The post-epoch KL pass runs after the final epoch too.** There is no later
  epoch to gate, but retaining this diagnostic makes every update comparable.
- **Monitoring note for the first real run under this change:** epoch 1 now
  always completes unconditionally, where the old per-minibatch gate could
  previously interrupt it early. That's standard PPO practice (checking KL
  between epochs, not within one), not a regression — but the run should
  specifically watch whether an unconstrained first epoch ever produces an
  unusually large KL or gradient-norm jump that the old check would have
  caught mid-epoch.

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
propagation, post-epoch KL stopping, PPO update metrics, and checkpointing. The existing
`Temp/tests/test_training_logger.py` configuration check ensures the exported
`PPO_N_STEP` setting reaches W&B configuration.
