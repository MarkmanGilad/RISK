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

## 2026-07-16

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
