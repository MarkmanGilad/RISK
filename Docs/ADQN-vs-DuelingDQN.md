# ADQN_050 vs. Dueling_DQN_040 — early-run analysis

> **Historical configuration:** this analysis describes `ADQN_050`, which
> used `ADQN_ADVANTAGE_LOSS_COEF = 0.1` and the original unscaled
> `tanh(td_advantage)` weight (equivalent to weight scale `1.0`). New ADQN runs
> default to coefficient `0.25` and
> `ADQN_ADVANTAGE_WEIGHT_SCALE = 5.0`; those settings are not represented by
> the results below.

This document compares the first real ADQN run (`ADQN_050`, W&B
`intfs3ew`) with the reference Dueling DQN run (`Dueling_DQN_040`, W&B
`gxlxeems`). It records an early-training snapshot: ADQN had logged 549
episodes and Dueling DQN 951, out of a configured 10,000. Performance
comparisons restrict Dueling to ADQN's episode range unless stated otherwise.

`Docs/ADQN.md` remains the normative algorithm specification. This file is an
interpretation of one pair of stochastic runs, not a change to that spec.

## Executive conclusion

The new ADQN loss is active, but its measured influence on the shared encoder
is small and it has not produced a clearly separable performance result yet.

The evidence supports four statements:

1. **ADQN with coefficient zero is Dueling DQN.** In a controlled one-update
   comparison, Q-values, greedy action, DDQN target, Bellman loss, and final
   parameters matched exactly—even when ADQN's gradient diagnostic ran.
2. **The coefficient-0.1 auxiliary loss is not zero or capped away.** Its
   scalar activity averages about 8.4% of Q-loss magnitude, and the effective
   coefficient is almost always the full `0.1`.
3. **Its shared-encoder gradient is small.** The weighted auxiliary/Q encoder-
   gradient ratio has a 1.4% median and 2.15% mean; the latest-100 mean is
   1.31%.
4. **Performance is inconclusive.** Dueling leads cumulative training wins
   through episode 500, while ADQN has the higher mean evaluation score at
   common checkpoints. Six-game evaluations and unpaired stochastic runs are
   too noisy to establish an improvement or regression.

Therefore, similar-looking curves do not mean the loss has no influence. They
mean its current influence—especially on the shared encoder—is modest and has
not created a reliable outcome difference in this first run.

## 1. What is actually being compared?

### Shared algorithm and configuration

Both runs use the same 522,710-parameter dueling architecture, replay
minibatch size, Double-DQN target, epsilon schedule, learning rate,
target-network synchronization cadence, reward settings, gradient clipping,
and epsilon-greedy Q action rule. ADQN adds the centered-advantage loss with
base coefficient `0.1`.

The implementations place the same calculation differently:

```text
Dueling DQN: network returns Q = V + A - mean(A)
ADQN:        network returns raw (V, A); agent calculates Q
```

### Sources of run-to-run variation

These are not paired deterministic trials. The runs have different:

- initial weights;
- learner seats and player counts;
- opponent rosters and game trajectories;
- replay contents and sampled minibatches;
- training-code snapshots.

Dueling_040 evaluates every 50 episodes; ADQN_050 evaluates every 25. The
extra ADQN evaluations change measurement density, not training updates.
Because the training setup is stochastic, a difference between these two
curves may be no larger than the difference between two ordinary Dueling runs.

## 2. Coefficient-zero equivalence

ADQN with `advantage_loss_coef = 0` should be functionally identical to
Dueling DQN. This was checked directly rather than inferred from long-run
curves.

A controlled CPU comparison used the same initialization seed, state/action
sets, replay transitions, sampled minibatch, and optimizer settings. ADQN's
normally periodic encoder-gradient diagnostic was forced to run during the
tested update to verify that it is read-only.

```text
check                                      result
state-dict keys                            identical
initial parameters                        exact match
parameter count                           522,710 each
maximum legal-action Q difference         0.0
greedy action                             identical
stored-action Q difference                0.0
Double-DQN target difference              0.0
one-update Bellman-loss difference        0.0
post-update parameter difference          0.0
ADQN weighted auxiliary loss              0.0
```

At coefficient zero, ADQN's extra work affects runtime and logging only. A
long ADQN-zero run is not a distinct algorithmic experiment; it is another
stochastic Dueling run under the current implementation.

## 3. Performance comparison

### Training wins

Raw training wins in matched 100-episode windows are:

```text
episodes       ADQN_050    Dueling_DQN_040
  1-100          0.0%           1.0%
101-200          2.0%           8.0%
201-300         15.0%          27.0%
301-400         30.0%          28.0%
401-500         27.0%          40.0%
```

Across episodes 1-500:

```text
ADQN_050             14.8%
Dueling_DQN_040      20.8%
```

Dueling leads the cumulative measure, but not every interval: ADQN is slightly
ahead during episodes 301-400. In the short 501-548 window, ADQN records 50.0%
and Dueling 43.75%; 48 episodes are too few to call that a reversal.

### Evaluation results

At the ten common checkpoints from episode 50 through 500:

```text
metric                                      ADQN_050    Dueling_DQN_040
mean eval score                              149.83          139.17
median eval score                            157.28          136.87
mean eval win rate                            15.0%           18.3%
episode-500 eval score                       240.21          243.05
episode-500 eval win rate                     33.3%           66.7%
```

Each evaluation is only six games. At episode 500, the scores are almost
equal, while the win-rate difference is two wins versus four. That sample is
too small to treat either metric as decisive.

### What performance currently says

The outcome evidence is mixed:

- Dueling has more cumulative training wins through episode 500.
- ADQN has the higher mean and median evaluation score.
- Dueling has the slightly higher mean evaluation win rate.
- The episode-500 evaluation scores are essentially tied.

This does not demonstrate that ADQN helps, but it also does not demonstrate
that ADQN behaves identically to Dueling. The comparison is currently inside
the noise level of a single stochastic run pair.

## 4. Is the ADQN loss active?

Yes. ADQN begins logging update diagnostics after the replay buffer becomes
trainable, producing 516 episode-level diagnostic rows in this snapshot.

```text
metric                                      mean       median       range
effective advantage coefficient          0.09995      0.1000   0.09697-0.1000
advantage activity / Q-loss ratio         0.08372      0.08465  0.05175-0.12732
abs(signed weighted loss) / Q-loss        0.05948      0.05980  0.03715-0.11227
weighted advantage loss                  -0.10940     -0.11366 -0.16995--0.04255
Q loss                                    2.53737      2.44087  1.35293-3.94205
```

### The safety cap is almost never controlling the loss

The per-episode maximum effective coefficient is `0.1` in every diagnostic
row. The episode mean falls below `0.099` in only about 0.6% of rows. Thus:

```text
binding setting:       ADQN_ADVANTAGE_LOSS_COEF = 0.1
usually not binding:   ADQN_MAX_ADVANTAGE_LOSS_FRACTION = 0.25
```

The weak behavioral separation cannot be explained by the cap suppressing the
auxiliary branch.

### Scalar contribution and gradient contribution are different

The auxiliary activity is about 8.4% of Q-loss magnitude. Mixed-sign samples
cancel in the signed batch mean, leaving an average absolute signed
contribution of about 5.9%.

These ratios describe scalar losses. They do not tell us how much the network
parameters move; the gradient measurements below are the relevant evidence
for that question.

## 5. How strongly does the loss influence optimization?

### Shared-encoder gradient magnitude

```text
mean Q encoder-gradient norm                  77.98
mean weighted-advantage encoder norm           1.06
advantage/Q norm ratio, mean                    2.15%
advantage/Q norm ratio, median                  1.40%
advantage/Q norm ratio, latest 100 episodes     1.31%
```

The auxiliary loss is therefore a small perturbation to the Bellman gradient
on the shared representation. This is the most direct explanation for ADQN
remaining behaviorally close to Dueling at coefficient `0.1`.

### Gradient alignment

```text
mean cosine similarity                 +0.050
median cosine similarity               +0.051
range                              -0.926 to +0.827
positive diagnostic rows                60.1%
negative diagnostic rows                39.9%
strongly negative (< -0.25)              8.5%
latest-100 mean cosine                  +0.018
```

The shared-encoder objectives are mostly close to orthogonal, with a slight
positive mean. The auxiliary gradient does not consistently reinforce
Bellman learning, and sometimes conflicts with it. Orthogonality does not
make the auxiliary signal meaningless: it could teach useful independent
features. The current performance data simply does not show a dependable
benefit from that direction.

### Important measurement gap: the action heads

The diagnostic above covers only the shared encoder. ADQN directly trains the
selected action's centered advantage, so its relative gradient on the
advantage heads may be larger than 1-2%. Separate Bellman and auxiliary
gradient norms are not currently logged for those heads.

Therefore, it is correct to say that the measured encoder influence is small.
It is not yet correct to say that the entire auxiliary update is only 1-2% of
the Bellman update.

## 6. Stability and scale signals

### Tanh saturation

```text
advantage-weight saturation fraction         49.8% overall
advantage-weight saturation, latest 100       56.3%
```

About half the `ADQN_050` samples have `|tanh(td_advantage)| >= 0.95`. For those samples,
the auxiliary loss mainly preserves the sign of the TD advantage rather than
its magnitude. This was an expected first-run risk in `Docs/ADQN.md` §H.

### Global gradient clipping

```text
combined gradient clipped                     97.4% overall
combined gradient clipped, latest 100         100.0%
```

The Q branch dominates the combined gradient norm. Clipping scales both
branches together, so it does not selectively erase ADQN's auxiliary
gradient, but it reduces the absolute update from both.

### Centered-advantage outliers

The taken centered-advantage absolute mean is approximately `2.84` over the
latest 100 episodes. However, `adqn_a_centered_taken_max` averages about `256`
over those episodes and reaches `339` in this snapshot. These rare large
values deserve monitoring because the linear auxiliary objective has no
finite minimum.

This is not the same as comparing `A` with `V`. A large `V(s)` does not dilute
the action ranking:

```text
Q(s,a) = V(s) + A_centered(s,a)
argmax_a Q(s,a) = argmax_a A_centered(s,a)
```

`V(s)` is shared by all legal actions and cancels from `argmax`. Even small
centered-advantage differences completely determine the greedy action. The
`|A|/|V|` ratio is therefore not a measure of behavioral influence.

## 7. What can and cannot be concluded?

### Supported by the current evidence

- ADQN-zero and Dueling DQN are functionally equivalent.
- ADQN_050's auxiliary loss is active and almost always uses coefficient
  `0.1` without hitting the 25% cap.
- Its shared-encoder gradient is small relative to the Bellman gradient.
- Its encoder alignment is weak and inconsistent rather than strongly
  cooperative.
- Tanh saturation, global clipping, and rare centered-advantage outliers are
  real monitoring concerns.
- No clear performance benefit is visible yet.

### Not supported by the current evidence

- That the auxiliary loss has no influence.
- That ADQN is definitively worse or better than Dueling DQN.
- That a small `A/V` ratio weakens greedy action selection.
- That encoder-gradient orthogonality makes the auxiliary signal noise.
- That the 25% cap is responsible for the loss's modest gradient influence.

## 8. Recommended next evidence

1. Continue comparing at matched episode and optimizer-step horizons.
2. Watch whether the auxiliary/Q encoder-gradient ratio remains near 1-2%.
3. Add separate Q-versus-auxiliary gradient norms for the advantage heads if
   the goal is to quantify the full optimization influence.
4. Monitor centered-advantage maxima and tanh saturation before increasing the
   coefficient.
5. Use multiple independent runs per coefficient for a causal performance
   conclusion. ADQN-zero is not a different algorithm; it is only a
   contemporaneous stochastic Dueling control.

## Run references

- ADQN_050: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/intfs3ew`
- Dueling_DQN_040: `https://wandb.ai/giladmarkman/Risk-GNN-DQN/runs/gxlxeems`
- Local checkpoints: `Checkpoints/ADQN_050` and
  `Checkpoints/Dueling_DQN_040`
