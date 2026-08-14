# DPQN — Deep Policy Q-learning Network

**Status: proposed design; not implemented.** DPQN is a hybrid learner that
keeps the project's working replay-based DQN objective while adding a small
on-policy policy-gradient objective. It is intended to reuse the expensive
graph encoding calculation rather than train separate DQN and PPO networks.

After warm-up, the policy head is the sole actor. It samples every learner
action in successive frozen 32-transition policy blocks, while the DQN Q head
continues to learn off-policy from those same transitions in replay.

The first version deliberately does **not** use PPO clipping, importance
sampling, or SAC-style soft-Q targets. Its policy loss is an auxiliary,
Q-guided policy-improvement signal with actor-only entropy regularization.

## Shared network

DPQN v1 uses the standard DQN action-injected layout: one injected graph row
for each legal candidate action, with no separate clean-state row. A single
graph encoder and pooling stack produces one embedding per action row, shared
by two shallow output heads.

    legal action-injected graph rows
                 |
         shared graph encoder
                 |
         +-------+----------------+
         |                        |
      DQN Q path             policy path
       Q(s, a)          one logit per legal action

The policy path applies a masked softmax over legal-action logits:

$$
\pi_\theta(a \mid s) = \operatorname{softmax}(\text{policy logits})_a.
$$

The policy head is the actor: it samples every action saved in recent policy
memory. The Q head is the replay-trained critic. The two paths have independent
final heads and both losses may update the shared encoder, but the policy loss
must not back-propagate through its Q-derived return weight.

## Two sources of experience

DPQN keeps two deliberately different stores of transitions.

| Store | Contents and use | Learning role |
| --- | --- | --- |
| Replay buffer | All learner transitions: DQN epsilon-greedy warm-up actions and post-warm-up policy-head actions, sampled in random minibatches. | Off-policy DQN TD learning. |
| Recent policy memory | An ordered segment of the latest 32 actions sampled from the policy head, including state, selected action/index, legal-action mask, reward, next state, and terminal/boundary information. | One on-policy policy-gradient update, then discard or replace it. |

Every policy-generated transition **must** enter both recent policy memory and
replay. The actor loss consumes only the former; DQN can learn from both kinds
of replay transition because its TD objective is off-policy.

## Training cadence for v1

DPQN v1 uses a 32-transition policy-collection block and a 4-step
bootstrapped policy return:

| Setting | Value |
| --- | ---: |
| policy collection length | 32 learner transitions |
| policy return horizon | 4 learner transitions |
| policy optimizer updates per collection block | 1 |
| DQN optimizer updates per collection block | 32 |
| DQN replay batch size | 64 |
| DQN replay eligibility | at least 64 transitions |
| actor-loss eligibility | `episode > EPSILON_DECAY_EPISODES` (epsilon at `EPSILON_END`) |
| policy-gradient coefficient | tune $\lambda_\pi$ |
| policy entropy coefficient | tune $\lambda_H$ |

The policy must remain fixed while collecting the 32 transitions. Because the
encoder is shared, this means no optimizer step may run during collection:
even a DQN-only update would change the policy's state representation.

At the end of a collection block:

1. Form 4-step bootstrapped returns for all 32 policy-memory transitions.
   A terminal inside a return makes its final Q continuation zero.
2. Make one **joint** optimizer update: the policy loss uses all 32 recent
   transitions and the DQN loss uses an independent 64-transition replay
   batch.
3. Make 31 further DQN-only replay updates of batch size 64, for 32 DQN
   updates in total.
4. Discard the policy memory and begin collecting the next 32 actions from the
   newly updated policy.

The joint update must be first. Once a DQN-only update changes the shared
encoder, the stored actions no longer come from the current policy; reusing
them would require an importance ratio or PPO-style correction.

### DQN warm-up before actor learning

Replay size 64 is only the minimum needed to form a DQN batch; it is not
evidence that Q values are mature enough to guide the actor. DPQN therefore
uses a separate actor-loss gate. Before that gate opens:

1. Collect transitions into replay using the existing DQN epsilon-greedy
   behavior.
2. Once replay has at least 64 transitions, run DQN-only updates using the
   established DQN update rhythm.
3. Do not add policy-gradient or entropy losses, and do not retain a
   policy-memory segment for actor learning.

Enable the actor once the existing exploration schedule has fully decayed:
gate on `episode > EPSILON_DECAY_EPISODES` (episode 101 onward), the same
point at which `on_episode_start` already sets `epsilon = EPSILON_END`. This
reuses an already-tuned schedule instead of introducing new warm-up constants.
At up to `MAX_STEPS_PER_EPISODE` learner turns per episode, 100 episodes
produces far more than the minimum 64 DQN updates needed to clear replay/batch
eligibility, comfortably satisfying "substantially larger than the batch
minimum."

Episode count is a schedule, not a direct measurement of Q accuracy, and
episode length varies with the randomly sampled player count and opponent
mix, so the number of DQN updates behind this gate is not fixed run to run.
Log `cumulative_optimizer_steps` and `cumulative_learner_turns` at the moment
the gate opens, so the Q-policy alignment diagnostics can be read against how
much DQN training actually preceded them. If regret stays high immediately
after warm-up, add an explicit minimum-update-count condition alongside the
episode gate rather than only lengthening it. When the gate opens, discard any
warm-up transition metadata and begin a fresh, frozen 32-transition
policy-sampled collection block.

At that exact transition, training action selection switches completely from
DQN epsilon-greedy selection to sampling the masked categorical distribution
from the policy head. Epsilon-greedy Q actions must not be mixed into an actor
collection block: entropy regularization is the actor's exploration mechanism,
and all actions used by the raw policy loss must have been sampled from that
policy. Every actor-generated transition enters replay, so DQN continues to
learn off-policy from the policy's experience. During evaluation, choose the
highest-probability legal action from the policy head rather than sampling.

After actor learning begins, target-network synchronization remains counted in
DQN optimizer steps, so each regular cycle advances that counter by 32.

The replay sampling intensity matches the current DQN's raw sample exposure:

$$
\frac{32 \times 64}{32} = 64
\quad\text{replay samples per fresh learner transition,}
$$

which is exactly one 64-transition replay update after every learner
transition — the same batch size and update frequency as the current
standalone DQN's replay rhythm (`BATCH_SIZE = 64`, `TRAIN_STEPS_PER_CALL = 1`).
DPQN does not need a different batch size or update cadence to preserve that
exposure: it only redirects one update in every block of 32 to be the joint
actor/DQN step instead of a DQN-only step.

## Unchanged DQN loss

The DQN part retains the current replay, target-network, Smooth-L1, gradient
clipping, and Double-DQN behavior. In particular, the online Q network selects
the next legal action and the target Q network evaluates it:

$$
y_t^{\mathrm{DQN}} = r_t + \gamma
Q_{\bar\theta}\!\left(s_{t+1},
  \arg\max_{a' \in \mathrm{legal}(s_{t+1})} Q_\theta(s_{t+1},a')\right).
$$

$$
L_{\mathrm{DQN}} =
\operatorname{SmoothL1}\!\left(Q_\theta(s_t,a_t),
y_t^{\mathrm{DQN}}\right).
$$

For a terminal transition, define the next-state Q continuation as zero, so
the target is simply the final reward: $y_t^{\mathrm{DQN}} = r_t$. The live
implementation enforces this with its done mask; it is omitted from the
displayed equation only to keep the notation simple.

The barred-theta symbol denotes the delayed target-network parameters. DPQN
must preserve the existing target-update cadence and legal-action scoring used
to select and evaluate the next action.

## Bootstrapped policy-gradient loss

For each policy-memory transition, v1 forms a 4-step return, stopping early
at a true terminal or collection boundary. The equation writes the horizon as
h, with h = 4 in v1:

$$
\hat G_t^{(h)} =
\sum_{i=0}^{h-1}\gamma^i r_{t+i}
+ \gamma^h B_{\bar\theta}(s_{t+h}).
$$

At a terminal, the bootstrap is zero. Initially, the bootstrap is the same
Double-DQN continuation used by the Q learner:

$$
B_{\bar\theta}(s) =
Q_{\bar\theta}\!\left(s,
  \arg\max_{a \in \mathrm{legal}(s)}Q_\theta(s,a)\right).
$$

Version 1 uses a detached expected-Q baseline without adding a separate value
head:

$$
b_t =
\operatorname{stopgrad}\left[
\sum_{a \in \mathrm{legal}(s_t)}
\pi_\theta(a \mid s_t) Q_\theta(s_t,a)
\right].
$$

The policy advantage and policy-gradient loss are:

$$
\hat A_t =
\operatorname{stopgrad}\left(\hat G_t^{(h)} - b_t\right),
$$

$$
L_{\mathrm{PG}} = -\operatorname{mean}_t\left[
\hat A_t
\log \pi_\theta(a_t \mid s_t)
\right].
$$

This baseline is action-independent for a given state and is fully detached,
so it reduces score-function variance without training the Q head through the
policy loss. It uses Q values already scored for the policy's legal-action
group and therefore adds no graph-encoder pass. It does not make the
Q-guided target into $Q^\pi$; that remains a separate source of bias.

Log raw return, baseline, and advantage mean/std, including by game phase,
alongside separate policy and DQN shared-encoder gradient norms. These
diagnostics identify phase-scale problems or an actor signal that is too weak
to affect the shared representation.

## Policy entropy regularization

The actor receives an entropy bonus to prevent its legal-action distribution
from collapsing too early:

$$
H_t = -\sum_{a \in \mathrm{legal}(s_t)}
\pi_\theta(a \mid s_t)\log\pi_\theta(a \mid s_t),
\qquad
\bar H = \operatorname{mean}_t H_t.
$$

Entropy is computed after the legal-action mask and grouped softmax. It is
averaged once per policy state, not once per action row, so states with many
legal actions do not receive extra loss weight merely because they have more
candidate graphs.

The actor and total objectives are:

$$
L_{\mathrm{actor}} =
\lambda_\pi L_{\mathrm{PG}} - \lambda_H \bar H,
$$

$$
L_{\mathrm{DPQN}} = L_{\mathrm{DQN}} + L_{\mathrm{actor}}.
$$

Because optimization minimizes loss, the negative entropy term rewards higher
policy entropy. It updates the policy head and shared encoder only; it does
not alter the DQN Q head, replay target, or reward definition. This is an
exploration bonus for the actor, not SAC-style entropy-regularized Q learning.

## Q-policy alignment diagnostics

The policy head has logits and probabilities, not a scalar value, so raw
policy logits must not be subtracted from DQN Q values. Instead, compare the
Q value expected under the policy with the greedy DQN Q value over the same
legal-action group:

$$
V_Q^\pi(s) =
\sum_{a \in \mathrm{legal}(s)}\pi_\theta(a \mid s)Q_\theta(s,a),
\qquad
V_Q^{\mathrm{greedy}}(s) =
\max_{a \in \mathrm{legal}(s)} Q_\theta(s,a).
$$

$$
\Delta_Q(s) = V_Q^{\mathrm{greedy}}(s) - V_Q^\pi(s).
$$

**dpqn_policy_q_regret** is the mean of $\Delta_Q$ across the policy-memory
states. It measures the DQN-predicted value sacrificed by the actor's
stochastic action distribution. It should be read with entropy: a nonzero gap
is expected while entropy regularization intentionally preserves exploration.

Log the following detached diagnostics from one grouped forward pass over the
same legal actions:

| Metric | Meaning |
| --- | --- |
| **dpqn_policy_q_expected** | Mean $V_Q^\pi(s)$: DQN Q value expected under the policy. |
| **dpqn_q_greedy_value** | Mean $V_Q^{\mathrm{greedy}}(s)$: best current DQN Q value. |
| **dpqn_policy_q_regret** | Mean $\Delta_Q(s)$: expected Q loss from policy stochasticity. |
| **dpqn_sampled_q_regret** | Mean $\max_a Q(s,a) - Q(s,a_{\mathrm{sampled}})$: Q loss of the action actually sampled for collection. |
| **dpqn_policy_q_argmax_agreement** | Fraction of states where the policy's most likely action equals DQN's greedy action. |
| **dpqn_policy_q_rank_correlation** | Mean within-state rank correlation between policy logits and Q values; skip one-action or constant-value groups. |
| **dpqn_policy_entropy** and normalized entropy | Exploration context needed to interpret the Q-regret values. |

These are diagnostics only: calculate them under no gradient or detach all
inputs so they do not alter either head. Initially use the online Q head,
because the question is whether the current actor is aligned with the current
DQN preferences.

If DQN TD error is stable but Q regret remains high, evaluate a later,
separate Q-to-policy distillation experiment. Define a temperature-smoothed
DQN teacher distribution:

$$
p_Q(a \mid s) =
\operatorname{softmax}\left(
\frac{\operatorname{stopgrad}(Q_\theta(s,a))}{\tau}
\right),
$$

and train the policy with:

$$
L_{\mathrm{distill}} =
-\sum_a\operatorname{stopgrad}(p_Q(a \mid s))
\log\pi_\theta(a \mid s).
$$

This is deliberately not part of DPQN v1. It would add a Q-imitation
objective, so use the diagnostics to justify it before changing the loss.

## What this policy signal means

The DQN bootstrap includes a greedy next-action choice, so it estimates an
optimal-control, Q-star-style continuation. A textbook on-policy policy
gradient instead weights actions by Q-pi, whose future actions are sampled
from the current policy:

$$
Q^\pi(s,a) = \mathbb{E}\left[
r + \gamma\sum_{a'}\pi(a' \mid s')Q^\pi(s',a')
\right].
$$

Therefore this first DPQN loss is not an unbiased pure policy-gradient
estimator. It says: increase the probability of actions that DQN predicts lead
toward high-value greedy future play. That is the intentional meaning of a
**Q-guided policy-improvement loss**. The discrepancy becomes smaller as the
policy approaches a greedy policy with respect to Q.

## Combined objective and v1 boundaries

The policy-gradient and entropy coefficients need tuning so the auxiliary
actor gradient encourages exploration without overwhelming the stable DQN
signal.

Version 1 intentionally excludes:

- PPO ratios, clipping, and repeated epochs over a policy segment.
- Policy-gradient updates from arbitrary replay samples.
- Time-based alternation between DQN and policy action selection after the
  actor gate; the policy head is the sole actor.
- A SAC-style soft-Q target.
- Simultaneously training the same Q head with both DQN's greedy target and a
  SAC soft-Q target; those targets give Q different meanings.

The actor-only entropy bonus does not change the Q semantics. Converting the Q
bootstrap itself to SAC-style entropy regularization would be a separate
experiment, because it would replace—not supplement—the standard DQN target.

## Open implementation choices

- Tune the policy-loss coefficient using policy and DQN gradient norms on the
  shared encoder; do not infer it directly from the 32:1 update count.
- Choose and, if useful, anneal the entropy coefficient while monitoring mean
  entropy, normalized entropy, and legal-action counts.
- The actor-loss gate reuses `EPSILON_DECAY_EPISODES` (see
  [DQN warm-up before actor learning](#dqn-warm-up-before-actor-learning));
  confirm via `cumulative_optimizer_steps`/Q-policy alignment diagnostics that
  100 episodes is actually enough DQN training once real run data exists, and
  add an explicit update-count condition if not.
- Review Q-policy alignment diagnostics before deciding whether a later
  Q-to-policy distillation loss is justified.
- The DQN replay batch now matches the already-validated standalone DQN batch
  size (64), so no separate device-memory check is needed there. Still verify
  that the policy path's per-update grouped forward — every legal action for
  all 32 policy-memory states at once — fits in device memory; that group can
  be wider than a single replay batch row when many actions are legal.
- Treat Dueling DQN plus the same actor protocol as a later comparison, after
  standard-DQN DPQN establishes whether the hybrid objective helps at all.
- Add focused tests for terminal/boundary bootstrapping, legal-action masking,
  detached advantages, target-network use, the 32/4/32 cadence, warm-up
  gating, routing every policy transition to both replay and one-use policy
  memory, policy-head whole-game evaluation, and shared encoder gradients
  before enabling DPQN in the learner factory.

## Review notes

**2026-08-14** — Second pass, after the warm-up-gate, self-baseline, and
post-warm-up action-selector revisions.

- Both concerns from the first pass are resolved. The actor-loss warm-up gate
  (see [DQN warm-up before actor learning](#dqn-warm-up-before-actor-learning))
  is now separate from DQN batch eligibility, so the actor no longer starts
  imitating an untrained Q head. The detached expected-Q baseline $b_t$ (see
  [Bootstrapped policy-gradient loss](#bootstrapped-policy-gradient-loss))
  removes the raw-REINFORCE variance problem without a value head or an extra
  encoder pass: subtracting a state-only, fully-detached quantity keeps the
  gradient unbiased while cutting variance, effectively using the Q head as
  its own implicit critic.
- The post-warm-up behavior is now explicit: the policy head samples every
  learner action in successive frozen blocks, while every actor transition
  enters both replay and one-use actor memory. Evaluation takes the policy
  argmax for every learner action.
- The Q-policy alignment diagnostics section sequences the work well: it
  measures regret, argmax agreement, and rank correlation before reaching for
  the deferred distillation loss, rather than adding that loss speculatively.
- Resolved since this pass: the DQN replay batch size changed to 64, matching
  the already-validated standalone DQN batch, so that device-memory question
  no longer needs separate verification. The joint update's other grouped
  forward — every legal action for all 32 policy-memory states at once
  (needed for both $b_t$ and the entropy term) — remains unverified; Risk
  states with many simultaneous legal reinforcement or fortify targets could
  make that group wider than a typical replay batch row, so it is still worth
  checking that width.

No open correctness concerns remain from this review. The design looks
implementation-ready pending the warm-up thresholds and coefficient tuning
already tracked above.

## Related literature and novelty

DPQN belongs to an established family of methods that combine an explicit
policy with an off-policy value learner. The broad idea of combining policy
gradient and Q-learning is therefore not new. The most relevant prior work is:

| Paper | Overlap with DPQN | Important difference and lesson |
| --- | --- | --- |
| [Actor-Advisor (2019)](https://arxiv.org/abs/1902.02556) | A separate policy-gradient actor is guided by an independently learned off-policy critic; its experiments include a replay-trained Double DQN critic in a discrete-action setting. | This is the closest structural precedent. The actor deliberately learns from Monte-Carlo returns, while the critic's softmax policy advises action selection. It is a direct alternative to DPQN's choice to put a short Q-star-style bootstrap in the actor return. |
| [PGQL: Combining Policy Gradient and Q-learning (2017)](https://arxiv.org/html/1611.01626) | Combines entropy-regularized on-policy policy gradient with replay Q-learning. | The canonical policy-gradient/Q-learning hybrid, but its Q estimate is derived from the policy's action preferences and value head. It does not test independent DQN and actor heads. Its Q-learning term is an optimizing, biased critic, matching DPQN's Q-guided rather than unbiased-policy-gradient interpretation. |
| [BDPI: Sample-Efficient Model-Free RL with Off-Policy Critics (2019)](https://arxiv.org/abs/1903.04193) | Uses an explicit stochastic actor with replay-trained off-policy critics that target Q-star values in discrete actions. | The actor slowly imitates the critics' greedy policies rather than using REINFORCE or a fresh on-policy block. It is the closest precedent for learning Q-star off-policy and making an actor follow it. |
| [ACER (2017)](https://arxiv.org/abs/1611.01224) | Discrete actor-critic with replay and a stochastic policy. | ACER corrects actor updates drawn from replay with truncated importance sampling and a trust region. It is relevant only if a later DPQN version reuses policy samples after the one fresh-block update. |
| [Q-Prop (2017)](https://arxiv.org/abs/1611.02247) | Preserves an on-policy policy-gradient signal while exploiting an off-policy critic. | It uses the critic as a variance-reducing control variate in continuous control. It motivates DPQN's detached baseline, but it does not use a DQN Q-star target as the actor return. [AWR](https://arxiv.org/abs/1910.00177)/AWAC (Peng et al., 2019; Nair et al., 2020) extend the same off-policy-critic-as-weight idea to discrete/offline settings, but *exponentiate* the advantage ($w=\exp(A/\beta)$) instead of using it raw; DPQN v1's un-exponentiated $\hat A_t$ is closer to plain n-step A2C, and swapping in exponentiated weighting is a candidate later comparison against tuning $\lambda_\pi$/$\lambda_H$ directly. |
| [Mean Actor-Critic (2017)](https://arxiv.org/abs/1709.00503) | Discrete-action policy gradient can use all legal-action Q values, instead of only the sampled action. | A useful later variance-reduction experiment once the Q-policy diagnostics show alignment. Its guarantees assume Q-pi, not the Q-star-style critic used by DPQN v1. This full-expectation baseline is the same object as the classical [Expected Sarsa](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf) identity $V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)]$ (van Seijen et al., 2009): DPQN's $b_t$ is literally that quantity, computed for free from Q values the policy forward pass already needs. |
| [Reactor (2018)](https://arxiv.org/abs/1704.04651) | One network exposes both a policy head $\pi(a\mid x)$ and a Q-value head $Q(x,a)$; its "β-leave-one-out" actor gradient uses Q as an action-value baseline. | The closest existing precedent for sharing one trunk between a policy and a Q head — this qualifies, rather than fully supports, the novelty bullet below about a shared encoder. The difference is what Reactor couples to that shared trunk: distributional Retrace off-policy correction feeding a gradient derived jointly with the actor, versus DPQN's unmodified standard Double-DQN target and independently-derived REINFORCE term, on a graph encoder built from injected legal-action rows rather than a generic state encoding. |
| [Discrete SAC (2019)](https://arxiv.org/abs/1910.07207) | A discrete stochastic actor, replay critic, and entropy objective. | It changes the critic to soft policy evaluation. DPQN keeps the ordinary Double-DQN max-Q target and uses entropy only in the actor loss, so it is not SAC. A newer [discrete-SAC analysis](https://arxiv.org/abs/2509.09838) is also useful background for keeping actor and critic entropy semantics separate. |

The exact combination below was not found in this literature search:

- an independent Double-DQN Q head and categorical actor head sharing one
  trunk — Reactor establishes precedent for a shared policy/Q trunk in
  general, but couples it to distributional Retrace rather than an
  unmodified, standard Double-DQN target;
- replay-only DQN optimization alongside strictly fresh, frozen on-policy
  actor blocks, so the policy loss needs no importance correction;
- a four-step Double-DQN bootstrap and detached policy-expected-Q baseline for
  the actor;
- action-injected graph rows, dynamically masked legal-action distributions,
  and the planned Q-policy regret/agreement diagnostics before any
  distillation is added.

This supports a limited and accurate novelty claim: **DPQN is a new
project-specific hybrid configuration and empirical experiment for this
action-injected graph setting.** It does not support claiming a new general
reinforcement-learning family or the first combination of policy gradients
with Q-learning. The Q-guided actor target remains an intentional biased
policy-improvement heuristic, not a new unbiased policy-gradient theorem.

For research-level algorithmic novelty, the project would need evidence beyond
the architecture: controlled ablations against DQN, Actor-Advisor-style Q
advice, and a standard actor-critic baseline, plus a result or theoretical
argument that attributes a benefit specifically to the frozen on-policy block,
shared encoder, or Q-policy diagnostics.
