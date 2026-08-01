# Change log

Dated record of what changed in code and docs, and why. The point is
**cross-session/cross-agent visibility**: if you're picking up this repo
in a fresh session (or a different agent is working on it in parallel),
read the top few entries here before assuming you know the current state
of `risk/learning/` — something may have moved since your last context.

Newest entry first. Keep entries short — a few bullets, not a narrative.
Each bullet should name the actual files touched, so "what changed" is
verifiable by reading the diff, not just this summary. This is a log, not
a design doc — the *why* behind a decision belongs in the relevant
`Docs/*.md` (linked from these bullets), not repeated here.

---

## 2026-08-01

- **Completed the planned reinforcement-policy contract for implementation.**
  `Docs/Reward.md` now requires continent reward to be both contested and
  frontier-only, states that a fully owned continent receives none, records
  the agreed initial constants, and makes the replacement (not additive)
  scope explicit. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Planned a Markov penalty for deferred reinforcement.** `Docs/Reward.md`
  now specifies a small negative term per army left in the visible
  reinforcement budget after a placement, discouraging redundant split moves
  while leaving a full-budget placement unpenalized. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Specified the planned proportional frontier reinforcement terms.**
  `Docs/Reward.md` replaces the earlier one-time readiness-crossing draft
  with a signed 1.5:1 weakest-neighbour term and an additional positive 2:1
  total-adjacent-enemy term; no-neighbour reinforcement remains negative.
  Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Specified the planned continent reinforcement formula.** `Docs/Reward.md`
  now uses a named `REWARD_REINFORCE_CONTINENT_PRIORITY_SCALE` multiplier
  (initially `10.0`) over territory share plus army share, divided by continent
  territory count and multiplied by a bounded fraction of armies placed; the
  split-placement test remains required. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Planned a proportional penalty for ending attacks while leaving a 2:1
  opportunity.** `Docs/Reward.md` now specifies an unimplemented,
  proportional per-target `StopAttackAction` penalty: each distinct enemy
  target uses its strongest available attacker; the existing shared shaping
  cap, rather than a separate ratio cap, bounds the total. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Replaced the detailed reinforcement proposal with a minimal policy
  draft.** `Docs/Reward.md` now reduces the unimplemented plan to a one-time
  readiness crossing, a plain average of contested-continent territory and
  army shares, and the existing interior penalty. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Added a compact reinforcement-policy summary.** `Docs/Reward.md` now
  tabulates the proposed readiness, contested-continent, interior-placement,
  and intentionally unshaped cases so the boundaries of the DQN's learned
  policy are explicit. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Simplified the planned reinforcement policy to three DQN-friendly
  preferences.** `Docs/Reward.md` rejects the launch-value/battle-probability
  proposal in favor of a one-time direct-neighbour readiness crossing,
  contested-continent territory/army-share progress, and the existing
  interior-placement penalty. Route and target selection remain learned.
  Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Replaced the planned capped-ratio reinforcement reward with a launch-value
  policy.** `Docs/Reward.md` now specifies a Markov potential difference that
  combines exact full-force battle probability with security against the sum
  of all directly adjacent enemy armies. This prevents a single weak neighbour
  from making a source territory appear campaign-ready; future route decisions
  remain the DQN's responsibility. The plan remains unimplemented. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Split the continent-push "already fully owned" fix so it only applies to
  reinforcement, not fortify.** `Docs/Reward.md`'s planned
  reinforcement-shaping revision previously had the fix to `_continent_push`
  (stop rewarding placement into a continent with nothing left to conquer)
  apply to both call sites via the shared helper. Fortify wants the opposite:
  concentrating strength into an already-completed continent should keep
  being rewarded, since holding it against recapture is a real goal.
  `_continent_push` now needs a parameter (e.g. `allow_completed_continent`)
  so the two phases can diverge. Still entirely unimplemented. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Reinstated continent-push and softened the readiness exponent in the
  planned reinforcement-shaping revision.** `Docs/Reward.md` no longer drops
  the continent-push term for reinforcement; it keeps it but fixes
  `_continent_push` (shared with `FortifyAction`) to stop rewarding
  placements into a continent the learner already fully owns. The readiness
  progress exponent is now a tunable `REWARD_REINFORCE_READINESS_EXPONENT`
  (suggested starting value `1.5`, not the earlier hardcoded `2`): squaring
  is kept in spirit — concentrating force into one strong stack should still
  beat spreading armies thin — but the effect was judged too extreme at `2`.
  Still entirely unimplemented. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Revised the planned reinforcement shaping to stay Markov.**
  `Docs/Reward.md` now replaces the hidden turn-start-budget proposal with a
  squared, 1.5:1--5:1 capped army-ratio progress difference. It removes the
  concentration/continent terms, keeps the interior-placement penalty, and
  explains that the remaining-budget denominator rewarded action splitting.
  The replacement remains split-action invariant without adding state history.
  The plan is explicitly unimplemented and names the required focused tests
  and fresh-run constraint. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Configured a fresh classic-DQN reward/exploration experiment.** The active
  `trainer.py` launcher now starts `DQN_102` with `resume=False`, so it cannot
  mix the new scaled reward regime with an older replay buffer. Set the shared
  epsilon floor to `0.1` (from `0.05`) while preserving the 100-episode decay.
  Documented the effective launcher configuration. Files:
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Docs/Trainer.md`, `Docs/ChangeLog.md`.

## 2026-07-30

- **Reduced full checkpoint cadence to every 50 episodes after episode 200.**
  The first regular checkpoint remains episode 200; later checkpoints are at
  250, 300, and so on, limiting lost training without saving early state.
  Added the constant-value regression test and documented the cadence. Files:
  `risk/learning/train_constants.py`,
  `Temp/tests/test_training_logger.py`, `Docs/Trainer.md`,
  `Docs/ChangeLog.md`.

- **Scaled dense reward shaping to make ending the game matter more.** Added
  `REWARD_SHAPING_SCALE = 0.1`, applied after per-action shaping is clipped
  and to the combined end-of-turn shaping; terminal win/loss remains
  `+100/-100`. Updated reward expectations and the reward reference. Files:
  `risk/learning/train_constants.py`, `risk/learning/reward.py`,
  `Temp/tests/test_reward.py`, `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Removed fixed-seat bias from periodic evaluation.** Each seeded tactical
  evaluation now places the learner at seats 0/1/2, and each full-game
  evaluation uses seats 0/2/4; opponent rosters fill the remaining seats in a
  deterministic order. Added schedule coverage. Files:
  `risk/learning/evaluator.py`, `Temp/tests/test_evaluator.py`,
  `Docs/Eval.md`, `Docs/ChangeLog.md`.

- **Made interrupted Run 101 resumable without splitting its W&B charts.**
  `TrainingLogger` can now require resumption of an explicit W&B run id;
  it can also use the restored trainer episode as W&B's true step axis for a
  new cloud run. `trainer.py` now resumes `Dueling_DQN_101` from its latest
  full checkpoint into a new W&B run beginning at episode 601. Added focused
  logger coverage and documented both procedures. Files:
  `risk/learning/training_logger.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_training_logger.py`, `Docs/Trainer.md`,
  `Docs/ChangeLog.md`.

## 2026-07-24

- **Corrected the action-injection documentation to match the signed-delta
  implementation.** `Action.md` now describes injected candidate graphs and
  correctly scopes `dqn_index()` to an internal locator; `ActionGraphBuilder.md`
  now consistently documents preserved real army counts, signed
  `proposed_army_delta` changes, unmodified trade-in copies, and the shared
  representation used by every implemented learner. Files:
  `Docs/Action.md`, `Docs/ActionGraphBuilder.md`, `Docs/ChangeLog.md`.

- **Reconciled `Update_Plan.md` with the actual `codex/history-aware-injection`
  implementation.** Verified against the real diff and a full test run (350
  passed, 1 skipped): steps 2-5 of "Combined implementation and run" are done
  (branch, code, `Dueling_DQN_100` launcher wiring, tests), step 1 (a matched
  control run) is still pending — `DQN_060`/`Dueling_DQN_040` are both only at
  episode 800 — and step 11's doc checklist was wrong: `Testing.md` was
  updated but unlisted, while `NetworkArchitectures.md` (stale
  `Data(x=[42, 13], ..., u=[1, 34])` example) and `Trainer.md` were listed but
  not actually touched. Removed the stale "nothing here is coded yet" intro
  sentence. Files: `Docs/Update_Plan.md`.

- **Implemented the history-aware reward and action-injection update on
  `codex/history-aware-injection`.** `State` now serializes/copies per-turn
  unfinished attack targets; `Environment` maintains and resets them;
  `RewardCalculator` applies and logs the non-stacking unfinished-target stop
  penalty; graph inputs expose the history plus a signed proposed-army-delta
  column; action injection preserves real army counts; and `trainer.py` now
  selects fresh `Dueling_DQN_100`. Added state, environment, reward, and graph
  regression coverage. Files: `risk/game/state.py`,
  `risk/game/environment.py`, `risk/learning/reward.py`,
  `risk/learning/train_constants.py`, `risk/learning/graph_adapter.py`,
  `risk/learning/action_graph_builder.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_state.py`, `Temp/tests/test_environment.py`,
  `Temp/tests/test_reward.py`, `Temp/tests/test_graph_representation.py`,
  `Docs/Update_Plan.md`, `Docs/Reward.md`, `Docs/GraphAdapter.md`,
  `Docs/ActionGraphBuilder.md`, `Docs/DuelingDQN.md`, `Docs/Testing.md`,
  `Docs/ChangeLog.md`.

- **Added the Dueling launch and documentation requirements to the update
  plan.** `Docs/Update_Plan.md` now requires explicitly selecting
  `Dueling_DQN_Agent` and verifying the `Dueling_DQN_100` run name, preventing
  the plain-DQN `trainer.py` default from being launched by mistake. It also
  adds `Docs/DuelingDQN.md` to the required current-document updates. Files:
  `Docs/Update_Plan.md`, `Docs/ChangeLog.md`.

- **Fixed singular "the chosen control" wording after step 1's fork.**
  `Docs/Update_Plan.md` steps 7-8 now name both branches of step 1's choice
  (a fresh matched Dueling control, or both `Dueling_040` and `DQN_060` as
  cautious references) instead of assuming a single control. Files:
  `Docs/Update_Plan.md`.

- **Made Dueling DQN the recommended base for the combined update.**
  `Docs/Update_Plan.md` now explains why preserving real armies and injecting
  `proposed_army_delta` is particularly suited to Dueling's separate
  `V(s)`/`A(s, a)` representation, recommends `Dueling_DQN_100`, and records
  the unmatched-epsilon caveat when comparing against Dueling 040. Files:
  `Docs/Update_Plan.md`, `Docs/ChangeLog.md`.

- **Clarified the implementation details and comparison controls for the
  combined update.** `Docs/Update_Plan.md` now specifies the final node/global
  layout and named feature offsets, separate unfinished-target W&B component,
  `codex/history-aware-injection` branch, DQN_060/DQN_100 as the recommended
  matched pair, evaluation invariants, roster-difficulty diagnostics, and
  current-document updates after implementation. Files:
  `Docs/Update_Plan.md`, `Docs/ChangeLog.md`.

- **Made the combined run's base agent (DQN vs. Dueling DQN) an open choice
  with a matched control.** `Docs/Update_Plan.md` no longer hardcodes `DQN_060`
  as the control: it now requires whichever base agent is chosen (plain DQN or
  Dueling DQN) to use the matching control (`DQN_060` or `Dueling_DQN_040`)
  throughout, and the fresh run ID prefix (`DQN_100`/`Dueling_DQN_100`) to
  match. Avoids confounding the reward/injection experiment with an
  unrelated architecture change. Files: `Docs/Update_Plan.md`.

- **Named the combined-update branch and run ID.** `Docs/Update_Plan.md` now
  specifies branch `history-aware-injection` and run `DQN_100` (combined-update
  runs use IDs 100 and up), replacing the placeholder branch name and generic
  "new run ID" wording. Files: `Docs/Update_Plan.md`.

- **Added branch protection to the combined-update run plan.**
  `Docs/Update_Plan.md` now requires creating `history-aware-injection` from
  the baseline and committing the complete implementation there before
  training, so the DQN 060 code stays easy to restore if the experiment
  regresses. Files: `Docs/Update_Plan.md`, `Docs/ChangeLog.md`.

- **Made the planned update intentionally breaking.**
  `Docs/Update_Plan.md` now requires a fresh model, replay buffer, run ID, and
  checkpoint directory with `resume=False`. It explicitly rules out loading
  old checkpoints, adapting old replay/state data, or maintaining old graph
  input compatibility. Files: `Docs/Update_Plan.md`, `Docs/ChangeLog.md`.

- **Approved `Update_Plan.md` for implementation as one combined change
  set.** No code changed yet, but the plan is no longer provisional: the
  territory-reward, unfinished-target/state-tracking, and action-injection
  pieces all proceed together as originally scoped, including the
  action-injection piece despite no dedicated evidence of the disentanglement
  failure mode it targets. Files: `Docs/Update_Plan.md`.

- **Flagged that `Update_Plan.md`'s action-injection change reopens a prior
  design decision without citing new evidence.** `Docs/ActionGraphBuilder.md`
  already recorded choosing the direct army-column write over a parallel
  `proposed_delta` column, deferring the latter until training shows the
  network can't disentangle proposed from actual army counts. `Update_Plan.md`
  now cross-links that note so the plan isn't read as a fresh idea. Files:
  `Docs/Update_Plan.md`.

- **Merged the ADQN first-run comparison into the ADQN reference.**
  `Docs/ADQN.md` now ends with a clearly marked historical appendix for
  ADQN_050 versus Dueling_DQN_040, preserving the old-settings warning,
  conclusions, performance/stability evidence, and run links. Removed the
  redundant standalone `Docs/ADQN-vs-DuelingDQN.md`. Files: `Docs/ADQN.md`,
  `Docs/ADQN-vs-DuelingDQN.md`, `Docs/ChangeLog.md`.

- **Removed three obsolete planning/history summaries.**
  `Docs/Training-Logging-Plan.md` duplicated the implemented trainer/logging
  reference, `Docs/RL-Prep-Changes.md` duplicated historical change records,
  and `Docs/summarization.md` duplicated the graph/network reference docs.
  Their current material is covered by `Docs/Trainer.md`, `Docs/Eval.md`,
  `Docs/ChangeLog.md`, `Docs/GraphAdapter.md`, `Docs/ActionGraphBuilder.md`,
  and `Docs/NetworkArchitectures.md`. Updated current documentation links.
  Files: `Docs/Training-Logging-Plan.md`, `Docs/RL-Prep-Changes.md`,
  `Docs/summarization.md`, `Docs/Trainer.md`, `Docs/Eval.md`,
  `Docs/GraphAdapter.md`, `Docs/HeuristicAgents.md`, `Docs/Testing.md`,
  `risk/game/phase.py`, `risk/learning/trainer.py`,
  `risk/learning/training_logger.py`, `risk/learning/gnn_dqn_agent.py`,
  `risk/learning/dueling_dqn_agent.py`, `risk/learning/graph_adapter.py`,
  `risk/learning/replay_buffer.py`, `Temp/tests/test_self_play.py`,
  `Docs/ChangeLog.md`.

- **Created one general plan for the next combined update.** New
  `Docs/Update_Plan.md` combines the planned reward update and action-injection
  representation update under separate sections, with one code-change and
  training run plan. It replaces `Docs/Reward_Update_Plan.md`; `Docs/Reward.md`
  now points to the general plan. Files: `Docs/Update_Plan.md`,
  `Docs/Reward_Update_Plan.md`, `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Changed the reward update plan to one combined experiment.**
  `Docs/Reward_Update_Plan.md` now treats territory outcome balance and the
  Markov-safe unfinished-target penalty as one coordinated code change and
  one matched training run, explicitly recording that it sacrifices isolated
  attribution to avoid multiple multi-day experiments. Files:
  `Docs/Reward_Update_Plan.md`, `Docs/ChangeLog.md`.

- **Consolidated every proposed reward change into one update plan.** New
  `Docs/Reward_Update_Plan.md` is now the sole plan for both the reward-only
  territory experiment and the Markov-safe unfinished-target/state-observation
  experiment, including code work, tests, and run order. `Docs/Reward.md`
  now only points to this plan; the superseded
  `Docs/Attack_History_Observation.md` was removed. Files:
  `Docs/Reward_Update_Plan.md`, `Docs/Reward.md`,
  `Docs/Attack_History_Observation.md`, `Docs/ChangeLog.md`.

- **Separated the attack-history model change from reward tuning.** New
  `Docs/Attack_History_Observation.md` contains the Markov-safe
  unfinished-target design: state history, graph features, network dimensions,
  reward rule, tests, and its own matched experiment. `Docs/Reward.md` now
  links to it and keeps the next territory experiment reward-only. Files:
  `Docs/Attack_History_Observation.md`, `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Corrected the unfinished-target reward plan for the Markov requirement.**
  `Docs/Reward.md` now requires a per-territory unfinished-target graph
  feature and a global `conquered_this_turn` graph feature alongside the
  history-based stop reward. It explains that the reward must not be added
  without those observable inputs, and adds the corresponding adapter,
  network-dimension, and test work. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Made the next reward experiment implementation-ready.** `Docs/Reward.md`
  now states the three proposed reward changes up front, then gives an exact
  file-by-file code plan for constants, per-turn state, environment updates,
  stop-reward decision table, W&B component, tests, and the matched training
  run. No source code or current training run changed. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Expanded the next reward experiment with the unfinished-target stop
  penalty.** `Docs/Reward.md` now plans tracking targets that were attacked
  but not conquered within a turn, preserving the existing one-time `-2`
  no-conquest/no-card stop penalty, and using a proposed `-0.5` per-target
  penalty only when a different conquest already earned a card. It records
  the required state lifecycle, reward-component logging, and test coverage;
  no source code or current training run changed. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Replaced the historical reward design document with a concise current
  reference.** `Docs/Reward.md` now documents only the implemented reward
  pipeline, formulas, current constants, W&B components, current diagnostic
  interpretation, and test/source locations. Superseded plans and historical
  findings remain traceable through Git and this change log. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Added an action-by-action reward table to the current reference.**
  `Docs/Reward.md` now summarizes every action's active reward terms and the
  conditional attack-event bonuses in one scan-friendly table. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Explained the implemented attack-reward composition.** `Docs/Reward.md`
  now separates pre-roll decision quality, dice outcome, and strategic-result
  bonuses, with formulas and a worked conquest example. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Consolidated the attack-reward reference.** `Docs/Reward.md` now keeps
  the phase summary and detailed attack explanation in one `Attack` section.
  Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Recorded the next isolated reward experiment.** `Docs/Reward.md` now
  specifies the post-DQN-060 outcome-balance experiment: zero
  `REWARD_TERRITORY_HOLD`, raise `REWARD_TERRITORY_DELTA` from 1 to the
  candidate 20, preserve meaningful attack/card rewards, and compare
  territory-trading behavior, wins, reward scale, and clipping without
  stacking other changes. It records the observed positive loss-return versus
  win-return ratio, the territory-cycle incentive, the retained negative
  low-ratio attack signal, and future constant/test work without applying it.
  Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Added per-episode training-roster and winner logging for W&B.** `Trainer`
  now records a readable seat roster, winner kind/seat, player count, each
  opponent-kind count, each kind's overall winner indicator, and its
  conditional winner indicator when present. This makes it possible to see
  which heuristic wins and whether learner performance changes with roster
  strength. Added focused trainer assertions and documented the fields.
  Files: `risk/learning/trainer.py`, `Temp/tests/test_trainer.py`,
  `Docs/Trainer.md`, `Docs/Training-Logging-Plan.md`, `Docs/ChangeLog.md`.

## 2026-07-19

- **Added the pre-implementation Policy Duel DQN (PDDQN) hybrid design.**
  `Docs/Policy_Duel_DQN.md` specifies a compact raw-Dueling `(V, A)` network
  that derives Q values, softmax policy logits, and `Vpi = sum(pi * Q)` with
  no new learned heads. It separates fresh-rollout PPO policy/value updates
  from replay-only Double-DQN Q updates, documents the sequential schedule,
  PQN/Q-Prop differences, diagnostics, tests, controls, and stop conditions.
  No code was changed. Files: `Docs/Policy_Duel_DQN.md`,
  `Docs/ChangeLog.md`.

- **Clarified VQN's neural computation cost relative to the existing Dueling
  DQN implementation.** `Docs/VQN.md` now records that Dueling already
  batches every legal action for both current and next replay states, so VQN
  reuses those Q/advantage outputs and adds only grouped softmax and weighted
  sum operations; its fresh rollout can add collection cost, not an extra GNN
  action batch. Files: `Docs/VQN.md`, `Docs/ChangeLog.md`.

- **Added the pre-implementation VQN design and experiment plan.**
  `Docs/VQN.md` defines Value Q-Learning Network as an independent
  Dueling-DQN-derived experiment: it removes PQN's replay `log pi` policy
  loss and proposes a bounded auxiliary Bellman loss for a policy value
  derived as `sum(pi * Q)`, rather than reusing Dueling's raw mean-Q value
  head. The plan requires a fresh frozen softmax rollout for the first
  on-policy value-loss experiment, preserves ordinary DQN replay for Q loss,
  and specifies safeguards, tests, matched controls, and risks. No code was
  changed. Files: `Docs/VQN.md`, `Docs/ChangeLog.md`.

## 2026-07-18

- **Increased ADQN's bounded TD-weight range and tuned its base coefficient.**
  Added `ADQN_ADVANTAGE_WEIGHT_SCALE = 5.0` and changed the detached weight to
  `scale * tanh(td_advantage / scale)`, preserving near-zero magnitude while
  smoothly bounding each sample at `+-5`; changed the base auxiliary
  coefficient from `0.1` to `0.25`. Updated saturation to mean 95% of the
  configured bound, persisted/logged the scale, preserved scale `1.0` when
  loading legacy checkpoints, expanded focused tests, and marked the old
  settings in the `ADQN_050` analysis. Epsilon decay remains unchanged at
  200 episodes to preserve the immediate comparison. Files:
  `risk/learning/train_constants.py`, `risk/learning/adqn_agent.py`,
  `risk/learning/training_logger.py`, `Temp/tests/test_adqn.py`,
  `Temp/tests/test_training_logger.py`, `Docs/ADQN.md`,
  `Docs/ADQN-vs-DuelingDQN.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Clarified why ADQN uses both `tanh` and the Bellman-relative loss cap.**
  Documented that `tanh` bounds each replay sample's TD-based gradient weight,
  while the detached effective coefficient limits aggregate minibatch scalar
  activity and cannot prevent one unbounded raw TD advantage from dominating
  within the batch. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Restructured the ADQN-versus-Dueling analysis around answerable
  questions.** Added an executive conclusion; separated comparability,
  coefficient-zero equivalence, training/evaluation outcomes, scalar loss
  activity, encoder-gradient magnitude/alignment, action-head measurement
  limits, and stability signals; and split supported conclusions from claims
  the first stochastic run pair cannot establish. Files:
  `Docs/ADQN-vs-DuelingDQN.md`, `Docs/ChangeLog.md`.

- **Corrected and expanded the `ADQN_050` versus `Dueling_DQN_040`
  analysis.** Replaced the false claim that Dueling leads every matched
  interval with the mixed result actually in the histories: Dueling leads
  cumulative training wins through episode 500, while ADQN leads mean eval
  score at the common checkpoints. Removed the incorrect inference that a
  small `|A_centered|/|V|` ratio weakens greedy action selection (`V` cancels
  from `argmax_a Q`). Added the controlled coefficient-zero equivalence result
  (identical Q, DDQN target, loss, action, and post-update parameters), exact
  scalar-loss/cap statistics, the more direct 1-2% auxiliary/Q encoder-
  gradient ratio, action-head diagnostic limitation, saturation/clipping
  results, and appropriately cautious single-run interpretation. Files:
  `Docs/ADQN-vs-DuelingDQN.md`, `Docs/ChangeLog.md`.

- **Identified the learner's randomized seat in live training output.** The
  console status line now includes `learner p<seat>`, removing the need to
  infer it from the omitted player in the opponent-territory breakdown. Added
  focused formatter coverage and updated the trainer/logging documentation.
  Files: `risk/learning/training_logger.py`,
  `Temp/tests/test_training_logger.py`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`, `Docs/ChangeLog.md`.

## 2026-07-17

- **Implemented ADQN as an independent sibling network and agent.** Added a
  standalone `ADQN` raw `(V, A)` network and `ADQN_Agent(BaseAgent)` with
  intentionally copied Dueling-style batching/replay/DDQN/epsilon plumbing;
  neither class constructs or inherits a PQN class. The agent adds detached
  tanh TD weights, the signed centered-advantage loss and adaptive Bellman-
  relative cap, complete diagnostics, checkpoint/config persistence, and
  trainer factory selection. Added focused
  ADQN, trainer, and logger coverage and updated the implementation/testing/
  logging docs. Files: `risk/learning/adqn.py`, `risk/learning/adqn_agent.py`,
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `risk/learning/training_logger.py`, `Temp/tests/test_adqn.py`,
  `Temp/tests/test_trainer.py`, `Temp/tests/test_training_logger.py`,
  `Docs/ADQN.md`, `Docs/NetworkArchitectures.md`, `Docs/Testing.md`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`, `Docs/ChangeLog.md`.

- **Made ADQN's opening proposal self-contained and corrected its coefficient
  rationale.** The motivation now defines raw and centered advantages, Q,
  V-based TD advantage, detached tanh weight, per-sample/signed/absolute-mean
  losses, adaptive coefficient, total loss, and positive/negative update
  directions before the detailed specification. Also corrected `0.1` from a
  claimed tuned/safe value to a guarded initial value whose scalar cap does
  not guarantee a gradient-ratio bound. Files: `Docs/ADQN.md`,
  `Docs/ChangeLog.md`.

- **Corrected `Docs/ADQN.md` now that it's the normative implementation
  spec, not a discussion record.** **Superseded by the independent-sibling
  implementation decision above:** this entry records the earlier design
  decision and is no longer the current implementation rule. Two fixes at
  that time: (1) Section F item 1 told an
  implementer to add a new `risk/learning/adqn.py` network file "starting
  from Dueling DQN" — but Section A's raw `(V, A)` contract is not merely
  similar to `PQN` (`risk/learning/pqn.py`), it's the exact same
  architecture and `forward()` signature with zero differences, so a new
  file would be a byte-for-byte duplicate. Changed the instruction to
  import and construct `PQN` directly from `adqn_agent.py`; only the agent
  file is new, matching the project's minimal-diff-per-sibling-agent
  convention at the agent level (each of `Dueling_DQN_Agent`/`PQN_Agent`
  already carries its own copied helpers) without applying it to a network
  class that has no diff to make. (2) An earlier review round had
  explicitly warned against reusing PQN's tuned `0.1` coefficient blindly,
  since the two losses live on different scales — that warning quietly
  disappeared when the doc's discussion history was condensed, and the
  constants block just hardcodes `ADQN_ADVANTAGE_LOSS_COEF = 0.1` with no
  stated justification. Added a note explaining why this is actually safe:
  `ADQN_MAX_ADVANTAGE_LOSS_FRACTION` bounds the effective contribution
  regardless of the base coefficient's exact value, so `0.1` being "wrong"
  for ADQN's scale is a soft failure mode, not a silent one — and pointed
  at `adqn_advantage_activity_to_q_loss_ratio` as the first-run signal for
  whether `0.1` was actually the binding constraint. Files: `Docs/ADQN.md`.

- **Made the ADQN document implementation-ready.** Removed the resolved
  13-item review history, fixed concrete initial constants and diagnostic
  cadence, and retained only five first-run monitored risks. Added exact
  diagnostic formulas, cancellation-visible activity ratio, correlation and
  zero-variance behavior, sparse gradient-diagnostic aggregation rules,
  explicit metric names and implementation/test files, checkpointed settings,
  and corresponding tests. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Promoted all remaining ADQN pre-implementation diagnostics into the
  normative plan.** Section F now requires V/centered-A decomposition metrics,
  effective-coefficient mean/max, advantage-weight sign and saturation
  fractions, and correlation with Bellman TD error; Section G now requires
  finite/range, zero-variance, drift-visibility, and episode-aggregation tests.
  Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Re-reviewed `Docs/ADQN.md` after its gradient-conflict-diagnostic
  revision; added one review note, no code.** Confirmed item 10's
  cancellation fix (§D/F/G) and the new encoder-gradient cosine-similarity
  diagnostic are both correct — verified the diagnostic's design choice of
  using the *scaled* `weighted_advantage_loss` (not raw `advantage_loss`) is
  right, since `effective_advantage_coef` is detached and
  `grad(weighted_advantage_loss) = effective_advantage_coef *
  grad(advantage_loss)` exactly matches what the real optimizer step
  applies. Flagged that item 7's own proposed diagnostic — logging `V`
  mean/absolute mean and `A_centered_taken` mean/absolute mean/max, because
  Q staying stable doesn't rule out V and A_centered drifting in opposite
  directions underneath it — never made the same "prose to normative
  section" trip that item 10's fix did: section F's logging list has no
  V/A_centered-specific fields, and none of section G's 14 tests check for
  this. Recommended adding `adqn_v_online_mean`/`_abs_mean` and
  `adqn_a_centered_taken_mean`/`_abs_mean`/`_max` to section F item 8.
  Smaller version of the same gap for items 8-9's logging asks; suggested
  reusing `Docs/Trainer.md`'s existing `_max`-suffix aggregation convention
  (already takes the max across an episode's updates) to get item 9's
  "distribution, not a point value" ask for `adqn_effective_advantage_coef`
  without new aggregation code. Files: `Docs/ADQN.md`.

- **Added ADQN gradient-conflict diagnostics to the plan.** Specified separate
  Bellman and weighted-advantage encoder-gradient norms plus their cosine
  similarity, including interpretation, zero-norm handling, configurable
  sampling frequency to control overhead, logging names, and tests ensuring
  diagnostics do not alter the optimizer update. Files: `Docs/ADQN.md`,
  `Docs/ChangeLog.md`.

- **Moved ADQN's cancellation fix into the normative loss specification.**
  Section D now defines per-sample signed losses, optimizes their signed mean,
  and calculates `effective_advantage_coef` from detached
  `mean(abs(per_sample_advantage_loss))`. Updated the worked example, logging
  fields, tests, and review notes so equal-and-opposite samples cannot silently
  bypass the proposed cap. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Renamed ADQN's adaptive coefficient for clarity.** Replaced
  `lambda_effective` with `effective_advantage_coef` throughout the design,
  including the proposed W&B field
  `adqn_effective_advantage_coef`; this is an ordinary scalar coefficient,
  not a Python lambda function. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Re-reviewed `Docs/ADQN.md` after its cancellation-bug fix (item 10); added
  one review note, no code.** Numerically confirmed item 10's finding with a
  sharper example: `per_sample_advantage_loss = [2000, -2000, 2000, -2000]`
  gives a signed mean of exactly `0`, so the *old* formula's denominator
  collapses to `eps` and `effective_advantage_coef` resolves to the full unmoderated
  coefficient (`0.1`), while item 10's `mean(abs(...))` fix correctly
  resolves it to `0.005` — a 20x difference in the actual gradient-scaling
  factor applied, invisible in a loss-value log since `weighted_advantage_
  loss` reports `0.0` either way. Flagged that this fix exists only as prose
  in review section H — section D's formula block and worked example still
  use the old, buggy `abs(advantage_loss)` denominator unchanged, so an
  implementer following section D literally would reproduce the bug.
  Recommended folding the fix into section D directly, adding
  `adqn_advantage_loss_abs_mean` as its own logged field distinct from the
  signed mean (section F item 8), and adding the canceling-minibatch case as
  an explicit required test (section G) rather than leaving it as a prose
  aside. Files: `Docs/ADQN.md`.

- **Added two further ADQN review comments.** Flagged that using
  `abs(mean(per_sample_advantage_loss))` in `effective_advantage_coef` allows large
  positive/negative sample contributions to cancel and bypass the scalar
  cap; recommended detached `mean(abs(per_sample_advantage_loss))` as the cap
  reference while retaining the signed mean as the optimized objective. Also
  documented why V remains necessary for the absolute Bellman-return level
  and the TD-advantage baseline even though it cancels from greedy action
  selection. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Re-reviewed `Docs/ADQN.md` after its adaptive-loss-balancing revision;
  added one review note, no code.** Numerically verified the doc's item 6
  (centered-advantage gradient split sums to exactly zero; the uncentered
  version doesn't, confirmed with a worked `w=0.7, N=5` check summing to
  `-0.7`) and confirmed items 7-8's math is sound. Flagged a new concern
  with the `effective_advantage_coef` adaptive cap added in section D: it scales
  inversely with the *current batch's* `q_loss`, which (a) makes the cap
  least restrictive early in training when `q_loss` is largest and value
  estimates are least reliable — backwards from where caution is likely
  wanted, (b) drives `effective_advantage_coef` to ~0 whenever a batch's `q_loss` is
  already small, silently disabling the ranking-improvement signal exactly
  when the value fit is good, and (c) makes `effective_advantage_coef` itself a
  noisy, batch-fluctuating quantity — a new source of training variance.
  Recommended logging `adqn_effective_advantage_coef`'s distribution (not just a
  point value) and considering an EMA-smoothed `q_loss` reference if this
  turns out to matter in practice. Files: `Docs/ADQN.md`.

- **Added adaptive ADQN loss balancing to the plan.** The proposed
  `effective_advantage_coef` now caps the signed weighted advantage-loss contribution
  at a configurable fraction of detached Bellman loss, initially 25%, while
  preserving a scaled nonzero gradient instead of directly clamping the loss.
  Added the formula, worked example, logging fields, configuration settings,
  limitations, and required tests. Files: `Docs/ADQN.md`,
  `Docs/ChangeLog.md`.

- **Added a second mathematical review of the ADQN plan.** Confirmed that
  the centered auxiliary loss produces the intended zero-sum ranking
  gradient across legal actions. Flagged that `tanh` bounds only the detached
  weight: the linear loss remains unbounded and may let `V` and centered `A`
  drift in opposite directions while preserving Q. Added required stream
  diagnostics and identified bounded Bellman error as a distinct future
  comparison, not a silent replacement for the planned V-based TD advantage.
  Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Reviewed `Docs/ADQN.md`; added review notes, no code.** Confirmed PQN's
  `log pi(a|s)` weighting is a genuine gradient asymmetry (differentiating
  gives `-td_advantage * (1 - pi(a))`, so confident-good actions get a
  shrinking push while rare/surprising ones get near-full-strength pushback)
  compounded by `td_advantage` itself being unbounded at this game's reward
  scale — both fixed by ADQN's `tanh(td_advantage)` weight in one change.
  Flagged for discussion: ADQN drops PQN's "one score is both Q and policy"
  premise entirely (now Dueling DQN plus an auxiliary loss, not a policy
  network); `tanh` will saturate to about +-1 for nearly every terminal-scale
  transition given +-100 rewards, worth measuring once it runs; `A_centered`
  has no natural ceiling the way a softmax probability does, so drift depends
  entirely on `q_loss`'s indirect counter-pressure; no starting value is
  proposed for `ADQN_ADVANTAGE_LOSS_COEF` and PQN's tuned `0.1` likely doesn't
  transfer (different loss scale) — recommended measuring `adqn_advantage_loss`
  vs. `adqn_q_loss` at coefficient 0 first. Noted that the planned coef-0
  Dueling-equivalence test (§G.1) can reuse the same pattern already proven
  for `PQN_e0`
  (`test_pqn_e0_reproduces_dueling_dqn_given_identical_weights`). Files:
  `Docs/ADQN.md`.

- **Documented why ADQN replaces PQN's replay policy loss.** Added the
  run-047 evidence and the positive/negative-advantage example showing how
  `log pi` gives old low-probability negative-advantage actions unequal replay
  impact. Clarified that ADQN removes this probability-based asymmetry with a
  bounded signed `tanh(td_advantage)` weight, without claiming to remove all
  off-policy replay bias. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Simplified ADQN's bounded advantage weight.** Removed mean-absolute
  division from the plan; ADQN now uses the direct detached expression
  `tanh(td_advantage)`, preserving the sign while bounding the auxiliary
  weight in `(-1, 1)` without batch-dependent scaling. Files: `Docs/ADQN.md`,
  `Docs/ChangeLog.md`.

- **Added the ADQN design plan.** `ADQN` starts as a Dueling-DQN copy and
  adds only a linear loss on the stored action's exact centered advantage,
  weighted by a detached TD advantage. The plan uses direct `tanh` to retain
  a bounded signed weight in `(-1, 1)`. It records the exact
  `s`/`s'` calculations, loss, minimal implementation delta, diagnostics, and
  required tests. Files: `Docs/ADQN.md`, `Docs/ChangeLog.md`.

- **Verified `PQN_e0` is algorithmically equivalent to `Dueling_DQN_Agent`.**
  Confirmed empirically rather than by inspection alone:
  `Dueling_DQN`/`PQN`'s `state_dict()` keys and shapes match 1:1 (weights
  copy straight across); with copied weights, `Q_dueling` vs. combined
  `Q_pqn_e0` on a live state agree to `1.5e-7` and share the same `argmax`;
  `act()` picked the identical action on 20/20 draws under a shared
  epsilon-greedy RNG stream; one identical `train_step` (same transitions,
  same sampled minibatch) produced the same loss (`0.0` relative diff) and
  post-step weights within `1.3e-5` — the same order as the GNN forward-pass
  float noise already documented in `Docs/Testing.md`, not an algorithmic
  gap. `PQN_e0`'s always-computed `policy_loss` contributes exactly zero
  since its coefficient is `0.0`. Added
  `test_pqn_e0_reproduces_dueling_dqn_given_identical_weights` to
  `Temp/tests/test_pqn.py` to pin this down permanently. Full suite green
  (329 passed, 1 skipped). Files: `Temp/tests/test_pqn.py`.

- **Added a first-class `PQN_e0` Bellman-only comparison variant.** The trainer
  factory now builds epsilon-greedy PQN with a per-agent policy-loss
  coefficient of zero; the agent uses that value in its total loss, label,
  progress metrics, and checkpoint state, while W&B records the effective
  value instead of the module default. Added factory/loss/checkpoint/logger
  coverage and documented the comparison contract. Files:
  `risk/learning/pqn_agent.py`, `risk/learning/trainer.py`,
  `risk/learning/training_logger.py`, `Temp/tests/test_pqn.py`,
  `Temp/tests/test_trainer.py`, `Temp/tests/test_training_logger.py`,
  `Docs/PQN.md`, `Docs/Trainer.md`, `Docs/Training-Logging-Plan.md`,
  `Docs/NetworkArchitectures.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Reviewed the implemented `PQN`/`PQN_e` action-selection modes; fixed one
  stale docstring.** Checked `pqn_agent.py`'s `__init__`/`act`/
  `on_episode_start`/`save_checkpoint`/`load_checkpoint` against `Docs/PQN.md`
  §24.D's plan: constructor surface, greedy-branch logic (epsilon-random vs.
  `argmax Q_online`, both training and eval), Dueling-identical decay,
  checkpoint round-trip with legacy-checkpoint defaulting, and
  `build_learner_agent("PQN"/"PQN_e", ctx)` wiring in `trainer.py` all match
  the plan and behave correctly — full suite green (325 passed, 1 skipped)
  plus a fresh `Trainer.train()` smoke run in `PQN_e` mode (epsilon decayed
  0.9905 after 3 episodes, matching Dueling's formula; all `pqn_*` diagnostics
  populated). Found one real defect: `progress_metrics()`'s docstring said
  `epsilon` was omitted ("permanently inert, not worth charting") while the
  method actually returns it — leftover from before the `PQN_e` mode existed.
  Fixed the docstring in `pqn_agent.py` and the matching stale line in
  `Docs/PQN.md` §24.E to state `progress_metrics()` reports `epsilon` in both
  modes. No behavior change — `epsilon` was already being returned and
  already covered by `test_pqn_agent_progress_metrics_reports_replay_state_
  and_epsilon`. Files: `risk/learning/pqn_agent.py`, `Docs/PQN.md`.

- **Removed an obsolete duplicate trainer source from `Docs/`.**
  `Docs/trainer.py` was an unused, stale Dueling-DQN-040-era copy; the only
  active trainer is `risk/learning/trainer.py`. Removing it prevents editing
  or running the wrong file. Files: `Docs/trainer.py` (deleted),
  `Docs/ChangeLog.md`.

- **Implemented Dueling-comparable PQN_e behavior selection.** `PQN` retains
  original sampled-softmax actions; `PQN_e` uses the same PQN network/losses
  with Dueling DQN's exact epsilon decay, uniform-random exploration, and
  greedy combined-Q action choice. The selected mode determines the instance
  label/run/checkpoint path, persists in full checkpoints (with legacy PQN
  checkpoints defaulting to the original mode), and is included in W&B config.
  Added coverage for both modes, decay, checkpoint compatibility, trainer
  factory selection, and logger config. Files: `risk/learning/pqn_agent.py`,
  `risk/learning/trainer.py`, `risk/learning/training_logger.py`,
  `Temp/tests/test_pqn.py`, `Temp/tests/test_trainer.py`,
  `Temp/tests/test_training_logger.py`, `Docs/PQN.md`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`, `Docs/NetworkArchitectures.md`,
  `Docs/PPO.md`, `Docs/Testing.md`.

- **Made the planned `epsilon_greedy_q` PQN mode concrete.** §24.D previously
  said only that the mode/epsilon would be "added to `PQN_Agent`" without
  specifying how; now specifies the actual proposed surface: a constructor
  `action_selection: str = "policy_sample"` parameter, `epsilon` promoted from
  a hardcoded inert `0.0` to a real constructor parameter consulted only in
  `epsilon_greedy_q` mode, `on_episode_start` overridden unconditionally with
  Dueling's exact decay formula (no mode branching in the decay itself),
  checkpoint fields with backward-compatible defaults for older PQN
  checkpoints, and `progress_metrics()` reporting `epsilon` unconditionally.
  Cross-referenced from §24.E so the "no epsilon" implementation-status note
  doesn't read as final. Still a plan — no code changed. Files: `Docs/PQN.md`.

- **Planned an opt-in Dueling-comparable PQN behavior policy.** The next PQN
  implementation step will preserve the sampled-softmax mode and add an
  `epsilon_greedy_q` mode that uses Dueling DQN's exact epsilon schedule and
  greedy `Q_online` action rule, isolating PQN's extra policy loss in a
  comparison run. No code changed. Files: `Docs/PQN.md`.

## 2026-07-16

- **Added a PQN policy-entropy diagnostic without changing its loss.** The
  current-state `Categorical(logits=advantage)` now supplies both the existing
  selected-action log-probability and a per-state entropy tensor. Its detached
  replay-minibatch mean is logged as `pqn_policy_entropy`; the connected tensor
  is retained for a future, separate entropy-regularization experiment. Files:
  `risk/learning/pqn_agent.py`, `Temp/tests/test_pqn.py`, `Docs/PQN.md`,
  `Docs/Testing.md`.

- **Pinned PQN's detached TD-advantage contract.** `PQN_Agent` now centralizes
  the replay policy loss in `_policy_loss(...)`, and a focused test proves its
  value-weight input receives no policy-loss gradient. Corrected the policy
  coefficient's stale documentation reference. Files:
  `risk/learning/pqn_agent.py`, `risk/learning/train_constants.py`,
  `Temp/tests/test_pqn.py`, `Docs/PQN.md`, `Docs/Testing.md`.

- **Implemented PQN and wired it into the trainer.** New `risk/learning/pqn.py`
  (`PQN` network — copy of `Dueling_DQN` returning raw `(value_mean,
  advantage)` per `Docs/PQN.md` §24.A) and `risk/learning/pqn_agent.py`
  (`PQN_Agent` — `_combine_q`, `_current_state_terms`/`_next_state_terms`,
  `train_step` implementing §24.C's Bellman + replay-based policy-improvement
  loss, `act()` sampling/argmaxing `softmax(advantage)`, no `epsilon`).
  `train_constants.py` gained `PQN_POLICY_LOSS_COEF = 0.1` (the initial
  policy-loss weight from §24.C). `trainer.py`'s `build_learner_agent(...)` gained a `"PQN"`
  branch and the matching import — the only change made to that file (all
  other trainer/agent classes left untouched per explicit scope). Fixed one
  real bug found via a `Trainer.train()` smoke run: `_next_state_terms` built
  a compact per-transition `group_index` (skipping `done`/no-legal-action
  transitions) but then indexed the full-length `next_stage` tensor with it
  directly, silently reading the wrong transition's phase for any batch with
  a skipped transition — fixed by remapping `next_stage` through
  `active_indices` first. New `Temp/tests/test_pqn.py` (11 tests: network
  output shape, `_combine_q` on one/two groups, `act()` sampling vs. argmax,
  learn/threshold parity with Dueling, `reached_max_steps` inertness,
  `progress_metrics`/`last_update_metrics` keys, checkpoint round-trip).
  Full suite green (314 passed, 1 skipped) and a 3-episode `Trainer` smoke
  run completed without incident. Not yet run as a real training experiment.
  Files: `risk/learning/pqn.py` (new), `risk/learning/pqn_agent.py` (new),
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_pqn.py` (new), `Docs/PQN.md`, `Docs/Testing.md`.

- **Reversed course again on PQN's network return contract: raw `(V, A)`,
  agent combines.** `pqn.py`'s `forward(...)` now returns the two
  uncombined head outputs instead of a fused `Q` — no new head, no new
  computation, just a different return signature — matching `heads.py`'s
  own stated precedent that orchestration ("which head", now also "how to
  combine/read them") is the agent's job, not the net's. `pqn_agent.py`
  gains one shared `_combine_q(value, advantage, group_index)` helper
  (reusing the `scatter(..., reduce="mean")` pattern already validated in
  `dueling_dqn.py`/`ppo_net.py`) so the grouped-mean formula is written
  once and reused at all three call sites (`s` online, `s'` online, `s'`
  target) instead of risking a repeat of the already-fixed
  `value_mask is None` bug. Policy logits now read directly from `A`
  (`softmax(A)`, per group) rather than from `Q`. Updated §24.A-B
  accordingly; §24.C's loss formulas were already written in terms of the
  derived `Q_online`/`V_online`/`pi`, so they needed no changes. Doc-only,
  no code. Files: `Docs/PQN.md`.

- **Deleted the now-redundant `mean_i(Q) = V(s)` derivation from `Docs/PQN.md`.**
  Since `pqn.py` returns `V(s)` directly (previous entry), the algebraic
  "recover it by averaging Q instead" alternative has no reason to stay in
  the doc — it documented a path not taken. §24.A now states only the
  chosen design. Doc-only, no code. Files: `Docs/PQN.md`.

- **Reversed the PQN plan's "recover V(s) by averaging Q" approach in favor
  of returning it directly.** Both are mathematically exact
  (`mean_i(Q(s,aᵢ)) = V(s)` is an unconditional algebraic identity given
  Dueling's combination formula), but computing `V(s)` → folding it into
  `Q` → recovering it by averaging `Q` back is a round trip with no real
  benefit over just returning it. `pqn.py`'s `forward(...)` now returns
  `(Q, V)` — `V` is `value_mean`, already computed internally from the
  clean row and previously discarded, so this is a small additive change
  (no new computation, no new head), not a redesign. Policy logits still
  use `Q` directly (`softmax(Q) == softmax(A)`, §14), so raw `A(s,a_i)` is
  never needed standalone anywhere in the agent. Updated §24.A-C
  accordingly. Doc-only, no code. Files: `Docs/PQN.md`.

- **Deferred PQN entropy regularization.** The initial PQN objective now has
  only Bellman and policy losses; entropy is documented as a future isolated
  experiment after the baseline has a comparison window. Files: `Docs/PQN.md`.

- **Specified PQN policy-loss reduction over replay samples.** The policy-loss
  formula now takes the minibatch mean, making it a scalar compatible with the
  total loss. Files: `Docs/PQN.md`.

- **Reviewed the PQN design doc for correctness; fixed two inconsistencies,
  added one simplification.** (1) §13's Bellman loss said plain squared
  error, contradicting §24.C's Smooth-L1 — fixed to Smooth-L1, matching
  Dueling DQN's actual loss. (2) §16's total loss added the entropy term
  (`+ λ_H H(π)`), which under gradient descent would push the policy toward
  *lower* entropy — the opposite of the intent; fixed to `-`, matching
  §24.C and PPO's actual `- PPO_ENTROPY_COEF * entropy.mean()`. (3) Added a
  derivation showing `V(s)` and the policy are both exact algebraic
  identities (`mean`/`softmax`) over Dueling's existing single fused-`Q`
  output — no network return-signature change is needed at all, `pqn.py`
  can be an unmodified copy of `dueling_dqn.py`. Also flagged (not
  specified) that diagnostics/tests/checkpoint format are still open,
  pointing at PPO's encoder-gradient-norm split as precedent given PQN
  trains two losses through one shared encoder. Doc-only, no code. Files:
  `Docs/PQN.md`.

- **Made PQN's stop-gradient operation concrete.** The policy-loss formula now
  shows `td_advantage.detach()` directly and explains that it blocks policy
  gradients from changing the value estimate. Files: `Docs/PQN.md`.

- **Separated PQN head outputs from derived calculations.** The plan now
  identifies value and per-action advantage as the only network-head outputs,
  then derives Q-values, logits, Softmax policy probabilities, and replayed
  action terms explicitly. Files: `Docs/PQN.md`.

- **Deleted the superseded PQN implementation plan section.** The old
  "section 24" (a more elaborate copy-Dueling-then-add-policy-learning plan)
  was already marked as replaced by the newer, simpler section 25; removed
  it and renumbered the replacement down to 24 so the doc has no gap or
  dangling "(replaces section 24)" reference. Files: `Docs/PQN.md`.

- **Simplified the PQN implementation plan around the existing Dueling net.**
  The replacement section states only the unchanged Dueling head outputs, the
  required current/next-state calculations, and the added Softmax and
  policy-loss calculation. Files: `Docs/PQN.md`.

- **Added the missing "Loss" section to `Docs/DuelingDQN.md`.** The doc
  documented network architecture and training-detail bookkeeping but never
  spelled out the actual objective: Smooth L1 against the Double-DQN
  bootstrap target, Adam + gradient clipping, hard target-network sync
  cadence, and the `dqn_*` diagnostics now logged (`Docs/Trainer.md`). Doc
  gap only — no code changed. Files: `Docs/DuelingDQN.md`.

- **Added a copy-from-Dueling PQN implementation plan.** `Docs/PQN.md` now
  specifies the new files, preserves Dueling's clean-row/grouped batch
  contract, and defines the policy, replay-loss, logging, checkpoint, test,
  and staged-rollout changes required for PQN without modifying DQN or
  Dueling. Files: `Docs/PQN.md`.

- **Brought DQN/Dueling DQN training diagnostics up to parity with PPO's.**
  `GNN_DQN_Agent`/`Dueling_DQN_Agent` previously logged nothing beyond
  `learn_loss_mean` — no `epsilon`, no gradient norm (already computed by
  `clip_grad_norm_` and discarded), no TD-error/Q-value visibility. Both
  agents now implement the same generic `progress_metrics()`/
  `last_update_metrics` hooks PPO already used, needing zero `Trainer`
  changes: `progress_metrics()` reports `epsilon`, `dqn_replay_buffer_size`,
  `dqn_train_steps_since_target_sync`; `train_step()` reports
  `dqn_td_error_mean`/`_abs_mean`/`_std`/`_abs_max`, `dqn_q_value_mean`/`_std`,
  `dqn_target_q_mean`/`_std`, `dqn_grad_norm`/`_clipped` — same `dqn_` prefix
  for both agents so their runs land in one shared chart namespace. Verified
  end to end through a real `Trainer` run for both agent classes. Files:
  `risk/learning/gnn_dqn_agent.py`, `risk/learning/dueling_dqn_agent.py`,
  `Temp/tests/test_agents.py`, `Temp/tests/test_dueling_dqn.py`,
  `Docs/Trainer.md`, `Docs/Training-Logging-Plan.md`.

- **Corrected the test-environment PATH guidance.** The documented
  `C:\\venvs\\ai-rl` interpreter is valid, but the system `python` can still
  resolve first; agent guidance now requires the explicit interpreter command.
  Files: `AGENTS.md`, `CLAUDE.md`.

- **Corrected the documented test venv path — it pointed at an incomplete
  environment.** `CLAUDE.md`/`AGENTS.md` told sessions to use
  `C:\Users\Gilad\venvs\ai-rl`, but that venv is missing `svg.path`, which
  spuriously fails `test_game_loop.py`/`test_ui.py` (`ModuleNotFoundError:
  No module named 'svg'` from `risk/ui/render/risk_map.py`). The complete
  venv is `C:\venvs\ai-rl` — already on `PATH`, so plain `python`/`pytest`
  resolves to it — confirmed via a full run: 299 passed, 1 skipped, 0
  failures. Files: `CLAUDE.md`, `AGENTS.md`.

- **Recorded the project test environment in agent guidance.** `AGENTS.md`
  and `CLAUDE.md` now point to the recovered `ai-rl` Python environment and
  give full-suite and focused-test commands, preventing future sessions from
  using the dependency-free system interpreter. Files: `AGENTS.md`,
  `CLAUDE.md`.

- **Validated the Dueling value-mask cleanup against DQN and PPO.** Using the
  recovered `ai-rl` environment, the focused Dueling DQN, classic DQN, and PPO
  regression files all passed (42 tests). The change stays confined to
  Dueling's private wrapper; no DQN or PPO behavior regressed. Files:
  `Temp/tests/test_dueling_dqn.py`, `Temp/tests/test_agents.py`,
  `Temp/tests/test_ppo.py`, `Docs/DuelingDQN.md`.

- **Aligned Dueling's private scoring wrapper with the required value-mask
  contract.** `_score(...)` no longer accepts or forwards `None` for
  `value_mask`, matching `Dueling_DQN.forward(...)`; all four Dueling training
  call sites already pass it explicitly. DQN's separate `_score(...)` and
  PPO's independent network interface are untouched. Files:
  `risk/learning/dueling_dqn_agent.py`, `Docs/DuelingDQN.md`.

- **Removed the dead `value_mask is None` fallback from `Dueling_DQN.forward`.**
  `value_mask` is now a required argument (`group_index` stays optional,
  unchanged); the old fallback branch that approximated `V(s)` by averaging
  over action-injected rows is gone, since every real call site
  (`dueling_dqn_agent.py`'s `score_actions`/`_score`/`_q_value`/
  `_max_next_q`/`_max_next_ddqn_q`) and all of `test_dueling_dqn.py` already
  passed `value_mask` explicitly — the branch was unreachable in practice.
  Docstrings updated to match; `Docs/DuelingDQN.md`'s "Recommended network
  API" text and `Docs/NetworkArchitectures.md` already described `value_mask`
  as required, so no doc-text changes were needed there beyond the review
  note. Could not execute `Temp/tests/test_dueling_dqn.py` in this session's
  sandbox — no network access to `download.pytorch.org` to install
  `torch`/`torch_geometric` — so this is unverified by an actual pytest run;
  the change is a pure deletion of an already-unreachable branch (no
  reachable code path changed), but running the suite locally is recommended
  before treating this as fully verified. Files: `risk/learning/dueling_dqn.py`,
  `Docs/DuelingDQN.md`.
- **Confirmed the Dueling value-mask fallback is removable.** The documented
  repository-wide call-site review confirms every current Dueling forward pass
  supplies clean value rows and an explicit `value_mask`; removal will make
  unsupported external calls fail fast instead of using the superseded
  action-injected value approximation. Corrected the review's stale test count
  and scoped its reachability claim to this repository. Files:
  `Docs/DuelingDQN.md`.

## 2026-07-12

- **Bounded PPO critic outlier gradients for PPO_045.** The critic now trains
  with Smooth-L1/Huber (`beta = 1.0`) instead of raw MSE while preserving raw
  MSE under both `ppo_value_loss` and `ppo_value_mse`, plus RMSE, for direct
  comparison with PPO_043/044. The actor objective, rewards, LR, KL target,
  rollout, and coefficients are unchanged; the trainer entry point advances to
  run 45. Files: `risk/learning/ppo_agent.py`,
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_ppo.py`, `Docs/PPO.md`, `Docs/Testing.md`,
  `Docs/Training-Logging-Plan.md`.
- **Expanded PPO and common compute diagnostics.** PPO now records exact
  optimizer minibatches/sample presentations, policy/value/entropy loss
  components, return/value/advantage scale, normalized entropy and legal-action
  counts, and pre-clip total/encoder/policy-head/value-head gradient norms.
  One minibatch per rollout also measures actor and critic gradients separately
  on the shared encoder, allowing the critic-dominance hypothesis to be tested.
  Checkpoints persist the new counters and recover optimizer steps from legacy
  Adam state. Files: `risk/learning/ppo_agent.py`, `Temp/tests/test_ppo.py`,
  `Docs/PPO.md`, `Docs/Testing.md`.
- **Aggregated all updates within each training episode.** Trainer no longer
  overwrites earlier PPO diagnostics when an episode contains multiple rollout
  updates; it weights optimizer-derived means by executed minibatches,
  preserves rollout-level means and `_max` maxima, flags non-finite values, and logs
  common per-episode/cumulative optimizer and sample-presentation axes for PPO,
  DQN, and Dueling DQN. Files: `risk/learning/trainer.py`,
  `Temp/tests/test_trainer.py`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`.
- **Recorded the PPO_044 run settings.** The user-selected `PPO_LR = 1e-4`
  and `RUN_ID = 44` are now reflected in the constants documentation and
  trainer entry point after PPO_043's near-universal KL early stopping. Files:
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`, `Docs/PPO.md`.

## 2026-07-11

- **Corrected PPO's KL diagnostic and early-stop estimator.** PPO now uses
  non-negative k3, `mean(ratio - 1 - log(ratio))`, rather than the
  sign-flipping finite-minibatch k1 estimate. This leaves the PPO objective
  unchanged but makes the logged KL and `PPO_TARGET_KL` stop condition
  trustworthy. Files: `risk/learning/ppo_agent.py`, `Temp/tests/test_ppo.py`,
  `Docs/PPO.md`, `Docs/Testing.md`.

- **Added PPO KL early stopping.** `PPO_Agent.learn()` now stops the remaining
  rollout minibatches when sampled approximate KL exceeds the new
  `PPO_TARGET_KL = 0.02`, retaining the crossing update while avoiding further
  policy drift in that rollout. It logs completed epochs, whether the stop
  fired, and the KL that triggered it. Added a focused regression test and
  documented the setting. Files:
  `risk/learning/train_constants.py`, `risk/learning/ppo_agent.py`,
  `Temp/tests/test_ppo.py`, `Docs/PPO.md`.

## 2026-07-11

- **Made PPO updates more frequent.** Reduced the rollout from 1,024 to 256
  learner turns and the PPO minibatch from 128 to 64 while retaining four
  epochs. PPO now starts updating after roughly two current-length games and
  takes one optimizer step per 16 learner turns rather than per 32. Files:
  `risk/learning/train_constants.py`, `Docs/PPO.md`.

- **Logged PPO diagnostics and a common sample-efficiency axis.** Trainer now
  logs `cumulative_learner_turns`, generically forwards an agent's fresh
  `last_update_metrics`, and accepts optional progress metrics. PPO supplies
  KL, clip fraction, entropy, value diagnostics, rollout fill/fraction, and
  completed rollout count without changing DQN/Dueling learning behavior.
  Files: `risk/learning/trainer.py`, `risk/learning/ppo_agent.py`,
  `Temp/tests/test_trainer.py`, `Temp/tests/test_ppo.py`,
  `Docs/Trainer.md`, `Docs/PPO.md`, `Docs/Testing.md`.

- **Increased evaluation cadence for 500-episode runs.** Evaluation now runs
  every 25 episodes rather than 50, producing 20 deterministic six-game
  measurements per run instead of 10. Files:
  `risk/learning/train_constants.py`, `Docs/Eval.md`.

- **Centralized replay capacity with the other training knobs.**
  `REPLAY_BUFFER_CAPACITY` now lives in `train_constants.py`; `ReplayBuffer`
  imports it for its default instead of declaring its own constant. Game-rule,
  UI, and graph-schema constants remain owned by their separate subsystems.
  Files: `risk/learning/train_constants.py`,
  `risk/learning/replay_buffer.py`, `Docs/Training-Logging-Plan.md`.

- **Made training constants collapsible by concern.** Added editor-recognized
  `# region` blocks for episode setup, DQN, PPO, exploration, run control,
  checkpointing, evaluation, reward settings, and exports. Files:
  `risk/learning/train_constants.py`.

- **Unified PPO and DQN training settings.** Moved all `PPO_*` knobs into
  the shared `train_constants.py`, updated PPO to import them there, and
  removed the duplicate `ppo_constants.py` module. Files:
  `risk/learning/train_constants.py`, `risk/learning/ppo_agent.py`,
  `risk/learning/ppo_constants.py` (removed), `Docs/PPO.md`.

- **Enabled PPO selection through the shared trainer factory.**
  `build_learner_agent("PPO", ctx)` now constructs `PPO_Agent` with the same
  temporary sizing environment used by DQN variants, without adding a
  PPO-specific trainer branch. Files: `risk/learning/trainer.py`,
  `Docs/Trainer.md`.

- **Added focused PPO regression coverage.** New tests verify policy/value
  output shapes, boundary-aware GAE bootstrap behavior, detached collection
  metadata, the rollout gate, cached legal-action index validation, grouped
  PPO forwards, and checkpoint restoration. Files: `Temp/tests/test_ppo.py`,
  `Docs/Testing.md`.

- **Fixed PPO rebuilding every transition's graph on every epoch instead of
  once.** `PPO_Agent.learn()` was calling `env.legal_actions` +
  `GraphAdapter`/`ActionGraphBuilder` + a full `PPO_Net` forward pass one
  transition at a time, inside the `PPO_EPOCHS × minibatch` loop — for the
  default constants that's 4000+ unbatched forward passes per update instead
  of the ~32 batched ones `Docs/PPO.md` calls for ("build once, not once per
  epoch"). Added `_cache_transition_entry` (builds + validates each
  transition's graph rows once, before the epoch loop) and `_forward_grouped`
  (batches many transitions' rows into one `PPO_Net` call, reusing the same
  flattened multi-group shape `Dueling_DQN` already supports), and rewired
  `_next_values`/`learn()` to use them; removed the old per-transition
  `_evaluate_batch`. Verified with an ad-hoc smoke script (not committed):
  loss decreases sensibly within an update, net weights change, checkpoint
  round-trips, `last_update_metrics` populates with sane values. Files:
  `risk/learning/ppo_agent.py`.

- **Implemented standalone injected-action PPO without modifying existing
  Python files.** Added `PPO_Net`, `PPO_Agent`, ordered `RolloutBuffer`, and
  PPO defaults in new `risk/learning/ppo_*.py` modules. PPO collects detached
  collection-time policy/value data, computes boundary-aware GAE, and runs
  clipped-surrogate/value/entropy updates; shared trainer/logging wiring stays
  pending explicit permission. Files: `risk/learning/ppo_net.py`,
  `risk/learning/ppo_agent.py`, `risk/learning/rollout_buffer.py`,
  `risk/learning/ppo_constants.py`, `Docs/PPO.md`.

- **Added regression tests confirming the epsilon/`learn()` refactor didn't
  change DQN/Dueling behavior.** New `Temp/tests/test_trainer.py` (previously
  zero coverage for `Trainer`) locks down the `reached_max_steps` contract
  (only the final `learn()` call of a truncated episode gets `True`) and runs
  short end-to-end smoke trainings for both agents. `test_agents.py`/
  `test_dueling_dqn.py` gained threshold-preservation tests (`can_train()`/
  `learn()` still flip exactly at `BATCH_SIZE`) and tests proving
  `reached_max_steps` is fully inert for DQN-family agents. Also ran an
  ad-hoc empirical check (`git worktree` at `HEAD`, not committed, per
  `Docs/Testing.md`'s "Ad-hoc verification" convention): identical seeded
  `remember()`+`learn()` runs against `HEAD` vs. the current tree produced
  matching loss values; small (~1e-5 relative) differences in raw net
  weights turned out to reproduce even running the *current* code twice in a
  row — confirmed as pre-existing `torch_geometric` scatter-aggregation
  nondeterminism, not something this session's refactor introduced. Files:
  `Temp/tests/test_trainer.py`, `Temp/tests/test_agents.py`,
  `Temp/tests/test_dueling_dqn.py`, `Docs/Testing.md`.

- **Restored the guarded learner factory after an accidental local revert.**
  `build_learner_agent` again validates the requested agent kind and raises a
  clear error for unsupported labels. Files: `risk/learning/trainer.py`.

- **Hardened learner selection in the trainer entry point.**
  `build_learner_agent` now uses the accurate `agent_kind` name, returns each
  supported learner directly, and raises a clear error for an unknown label.
  Files: `risk/learning/trainer.py`, `Docs/Trainer.md`.

- **Passed the max-step boundary through the shared learning call.**
  `Trainer` now calls `learn(reached_max_steps=...)`; DQN/Dueling accept and
  ignore it, while planned PPO will use it to separate a GAE boundary from a
  terminal `done`. Files: `risk/learning/trainer.py`,
  `risk/learning/gnn_dqn_agent.py`, `risk/learning/dueling_dqn_agent.py`,
  `Temp/tests/test_agents.py`, `Temp/tests/test_dueling_dqn.py`,
  `Docs/Trainer.md`, `Docs/PPO.md`.

- **Corrected PPO time-limit semantics.** `Docs/PPO.md` now preserves
  `done=False` and the `V(next_state)` bootstrap at a max-step cutoff. A
  planned generic episode-end callback records PPO's internal GAE boundary so
  returns cannot cross into a reset game. Files: `Docs/PPO.md`.

- **Removed obsolete epsilon-greedy planning from PPO.** `Docs/PPO.md` now
  records only PPO's inert evaluator-compatibility attribute and no longer
  includes it in PPO checkpoint state. Files: `Docs/PPO.md`.

- **Kept epsilon fully at its configured start value for the first episode.**
  DQN and Dueling now begin decay after episode 1, so the documented
  `EPSILON_START` value is actually used before the 200-transition decay.
  Tests cover both endpoints. Files: `risk/learning/gnn_dqn_agent.py`,
  `risk/learning/dueling_dqn_agent.py`, `Temp/tests/test_agents.py`,
  `Temp/tests/test_dueling_dqn.py`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`.

- **Moved epsilon-greedy decay out of `Trainer` and into the DQN agents.**
  `Trainer` no longer imports `EPSILON_*` constants or computes the decay
  schedule; it just calls a new no-op-by-default `BaseAgent.on_episode_start
  (episode)` hook once per episode (same pattern as the existing
  `on_turn_start`/`on_turn_end` hooks). `GNN_DQN_Agent`/`Dueling_DQN_Agent`
  override it to recompute their own `epsilon`; `PPO_Agent` (`Docs/PPO.md`)
  needs no code, it inherits the no-op. Rationale: epsilon-greedy exploration
  is a DQN-family concern that PPO doesn't use, so `Trainer` shouldn't own it
  — same "agent owns its own algorithm-specific state" split already used for
  `learn()`'s constants. Files: `risk/agents/base_agent.py`,
  `risk/learning/trainer.py`, `risk/learning/gnn_dqn_agent.py`,
  `risk/learning/dueling_dqn_agent.py`, `Docs/Trainer.md`,
  `Docs/Training-Logging-Plan.md`, `Docs/PPO.md`.

- **Clarified PPO's shared-boundary and observability contracts.**
  `Docs/PPO.md` now requires the cutoff-aware `done` expression, a generic
  `last_update_metrics` forwarding hook, and an action-index integrity check
  when rebuilding legal actions. Files: `Docs/PPO.md`.

- **Hardened the PPO design before implementation.** `Docs/PPO.md` now makes
  selected legal-action indices mandatory, handles terminal bootstrap without
  constructing a game-over action batch, and marks forced time-limit cutoffs
  terminal so GAE cannot cross into a reset game. It also specifies advantage
  normalization, PPO diagnostics, and focused regression coverage for those
  cases. Files: `Docs/PPO.md`.

## 2026-07-05

- **`learn()` made no-arg for every training agent.** `GNN_DQN_Agent`/
  `Dueling_DQN_Agent`'s `can_train()`/`learn_steps()`/`learn()` no longer
  take `batch_size`/`n_steps` — they read `BATCH_SIZE`/`TRAIN_STEPS_PER_CALL`
  from `train_constants.py` directly. `Trainer.train()`'s call site is now
  `self.agent.learn()`. Rationale: `Trainer` was only ever forwarding two
  static constants unchanged, never computing or varying them, so the
  parameters were pure ceremony — and it removes the need for `PPO_Agent`
  (`Docs/PPO.md`) to accept-and-ignore them. `epsilon` stays a real
  cross-interface value since it genuinely is Trainer-owned/dynamic.
  Files: `risk/learning/gnn_dqn_agent.py`, `risk/learning/dueling_dqn_agent.py`,
  `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/PPO.md`.
- **Decided: injection only, no lookup action representation.** Net A
  (DQN+inject) and Dueling (built on the same foundation) already train
  well; Net B/Net D (lookup) are not being built for any algorithm. Kept
  the Net B/D writeups as historical reference, clearly marked as not
  being pursued, rather than deleting them.
  Files: `Docs/NetworkArchitectures.md`, `Docs/PPO.md` (resolved its
  "Net C vs Net D" open question).

## 2026-07-04

- **Built `Dueling_DQN`/`Dueling_DQN_Agent`** — a second learner alongside
  `GNN_DQN_Agent`, sharing `GraphAdapter`/`ActionGraphBuilder`/`Encoder`/
  per-phase heads, adding a value head + `group_index`-grouped advantage
  mean (`Q = V + A - mean(A)`). `V(s)` is computed from a clean,
  non-injected base row (`value_mask`), not averaged over action-injected
  rows — this went through one redesign mid-build once that distinction
  was flagged. Minimal-diff policy vs. `GNN_DQN_Agent` throughout; the old
  agent was not modified.
  Files: `risk/learning/dueling_dqn.py`, `risk/learning/dueling_dqn_agent.py`,
  `Temp/tests/test_dueling_dqn.py`, `Docs/DuelingDQN.md`.
- **`Trainer`/`TrainingLogger` identity/storage hooks** (additive, classic
  agent behavior unchanged by default): `Trainer.__init__` gained
  `checkpoint_dir` override; `TrainingLogger` gained `run_name`;
  `_build_config` gained `agent_class`. Checkpoint dir and W&B run name
  now default to `Checkpoints/<agent.label>_<run_id>`
  (e.g. `DQN_030`/`Dueling_DQN_030`) instead of `Checkpoints/run_<id>`.
  Each agent declares its own `label` class attribute — no
  isinstance-based lookup in `Trainer`.
  Files: `risk/learning/trainer.py`, `risk/learning/training_logger.py`,
  `risk/learning/gnn_dqn_agent.py`, `risk/learning/dueling_dqn_agent.py`,
  `Docs/Trainer.md`, `Docs/Training-Logging-Plan.md`.
- **Wrote `Docs/PQN.md`** — design-only doc for a unified value/policy net
  (Dueling architecture, `Q`/`π` from one scoring function). No code.
- **Wrote `Docs/PPO.md`** — design-only doc for a third learner, `PPO_Agent`
  (Net C, PPO + injection). Key mechanism: push all on-policy rhythm
  (rollout buffer, update cadence, log-prob/value smuggling from `act()`
  into `remember()`) inside the agent so `Trainer.train()` needs zero
  PPO-specific changes. No code yet — status is explicitly "wait for a
  meaningful Dueling comparison window" before starting implementation.

---

## Convention for new entries

- Add new entries at the **top**, under today's date (new `## YYYY-MM-DD`
  heading if today doesn't have one yet).
- One bullet per logical change, bold a short label, then 1-3 sentences of
  *why* (not a restatement of the diff), then a `Files:` line.
- Link to the relevant `Docs/*.md` for full design rationale instead of
  duplicating it here.
- This file itself doesn't need a changelog entry when you edit it.
