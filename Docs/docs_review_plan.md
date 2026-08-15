# Documentation review and restoration plan

**Status:** planning only. This document records a read-only audit; it does
not authorize changes to implementation or current reference documents.

**Verification pass (2026-08-15):** every claim below was re-checked against
current code and current `Docs/*.md` content, not just the original
pre-cleanup snapshot. Two things came out of that pass that matter for anyone
executing this plan:

1. **Several items were already partially or fully fixed** by doc edits made
   after this plan was first written — this repo has more than one session
   touching these docs concurrently. Re-read the current doc before acting on
   any item below; don't assume the gap described here is still exactly as
   described.
2. **Every line count in the original document-by-document table was already
   wrong.** Docs that keep moving make a "pre-cleanup -> current" count stale
   almost immediately. The same problem applies to anything else that names a
   specific current run id, episode number, or test count as if it were a
   durable fact — `Trainer.md`'s launcher example and `PPO.md`'s launcher
   claim are the concrete cases below, and both were *already* wrong at
   verification time. Treat any such number as an illustrative example (e.g.
   `Dueling_DQN_NNN`, "whichever learner `main()` currently builds") rather
   than the value of the moment, both when reading this plan and when writing
   the fixes it proposes.

## Scope and baseline

The audit compares every current `Docs/*.md` file with commit `4d13eca`, the
snapshot immediately before the `529fb4c` documentation cleanup. The review
checks two things:

1. whether the cleanup removed material needed to describe the **current**
   implementation; and
2. whether a current document is factually stale even when it was not removed
   by the cleanup.

The goal is a compact, trustworthy reference for live behavior. Historical
plans, retired learners, superseded experiments, and build diaries should stay
retired unless they contain a fact not documented anywhere else. Numbers that
name *this week's* specific run, checkpoint, or test count are not "live
behavior" in that sense — write the mechanism, not the instance.

## Audit conclusion

The cleanup correctly retired PQN, ADQN, VQN, Policy-Duel-DQN, model-selection
planning, conflict copies, and old experiment notes. Those documents should
not be restored wholesale.

It did, however, over-compress several active references. The main recovery
work is to add concise current-state documentation, not to resurrect the old
plans verbatim. The verification pass found one additional failure mode worth
watching for going forward: **restoring lost material can itself reintroduce
redundancy** if it's added alongside content that already covers the same
ground. `Reward.md`'s restored phase table briefly existed twice, in two
different levels of formula detail, before being deduplicated during this
audit (see `ChangeLog.md`). Check for an existing table/section covering the
same fact before adding one back.

## Document-by-document record

Line counts below are a snapshot taken during the 2026-08-15 verification
pass, not a maintained fact — they will drift the next time any of these
files is edited. Rows marked "not re-verified" were outside the scope of this
pass and reflect the original audit only.

| Document | Pre-cleanup -> snapshot | Verdict |
|---|---:|---|
| `Action.md` | 286 -> 286 (not re-verified) | Unchanged and current. It already preserves the removed environment-plan facts for trade flow and fortify candidates. |
| `ActionGraphBuilder.md` | 103 -> 168 | Partially fixed since the original audit. Its old trade-in sketch was replaced by a better current limitation/deferred-plan explanation. Its batched-example graph shape (`x=[2856, 13]`, `u=[68, 34]`) is still stale; see Priority 1. |
| `BoardTopology.md` | 103 -> 136 | No cleanup loss, but now internally self-contradictory, not just stale; see Priority 2. |
| `ChangeLog.md` | 2,101 -> growing (not re-verified) | Append-only in effect; no active material lost. |
| `ChooseAgent.md` | 244 -> 311 | No cleanup loss, but several statements are stale; see Priority 2. |
| `Content.md` | added after cleanup (not re-verified) | Correct active-document index. It should index a future learned-agent-play reference. |
| `DPQN.md` | added after cleanup (not re-verified) | Correctly marked as a proposal, not a current implementation reference. |
| `DuelingDQN.md` | 447 -> 27 | Partially fixed since the original audit: it already states the Double-DQN target and Smooth-L1 loss. Still missing the numeric hyperparameters and checkpoint filenames; see Priority 1 (narrower than originally scoped). |
| `Environment.md` | added by cleanup -> 33 | Useful short reference. One card-flow sentence needs correction, and is wrong in a more specific way than first described; see Priority 2. |
| `Eval.md` | 360 -> 27 | Partially fixed since the original audit: no-replay/no-learning, fixed suites, seed/seat rotation, cadence, and retention/manifest are already stated. A narrower set of facts is still missing, and one real (not just documentation) reward-boundary behavior gap was confirmed; see Priority 1 (narrower than originally scoped). |
| `GraphAdapter.md` | 159 -> 197 | Partially fixed since the original audit: the top summary prose already states the correct 15-column/35-value shapes. Its detailed feature tables still contradict that prose; see Priority 1. |
| `GraphAttentionNetwork.md` | 154 -> 223 | No loss; it correctly documents the live 15-node-feature / 35-global-feature encoder and needs no changes. |
| `HeuristicAgents.md` | 454 -> 504 | The Killbot-absent claim from the original audit is **already fixed**. A different, newly confirmed problem remains: an obsolete build-plan section now contradicts the doc's own corrected text; see Priority 2. |
| `NetworkArchitectures.md` | 230 -> 49 (not re-verified) | No required restoration. Removed PQN/ADQN/roadmap material is retired; detailed live graph mechanics belong to the graph-specific documents. |
| `Poster.md` | 242 -> 177 (not re-verified; under active separate revision this session) | No implementation-reference loss confirmed at time of original audit. Removed material was poster layout/asset guidance, not live code behavior. |
| `PPO.md` | 111 -> 244 | Still stale, and in a more specific way than first described: the launcher doesn't run an older PPO version, it runs Dueling DQN, a different learner family entirely; see Priority 2. |
| `Reward.md` | 416 -> 109 | The phase table was restored, then briefly duplicated, then deduplicated during this session (see `ChangeLog.md`). Several live terminal/timeout/attack conditions remain confirmed missing; see Priority 1. |
| `Testing.md` | 116 -> 35 | Two concrete gaps confirmed. A third ("restore compact fixture/convention guidance") is not concretely falsifiable against the current doc; see Priority 2. |
| `Trainer.md` | 286 -> 314 | Removed material is correctly retired PQN/ADQN behavior and an old run note. Two live wording details need correction, one of which the plan itself gave a now-stale example number for; see Priority 2. |

## Priority 1: restore missing current reference material

### 1. Correct the shared graph feature contract

**Scope corrected by verification:** this is now a finish-the-job task, not a
full restore. `GraphAdapter.md`'s top summary paragraph already states the
correct shapes and features — someone already fixed the headline claim. What
remains stale is everything below that paragraph, which still contradicts it:

- the detailed node-feature table still shows only 3 rows ending at column 12
  and headers/examples still say `[42, 13]`; it needs rows for
  `unfinished_attack_target` (column 13) and `proposed_army_delta`
  (column 14), and every `[42, 13]` string corrected to `[42, 15]`;
- the `u` (global) table still stops at `u[28:34]` with no row for `u[34]`;
  add `conquered_this_turn` there, and correct every `[1, 34]` string to
  `[1, 35]`;
- the `u[0]` table row still states its source as `settings.player_count`;
  correct it to the live state's player count (`len(state.hands)`) — this one
  is a direct contradiction of the doc's own corrected prose a few lines
  above it, not just an omission;
- `ActionGraphBuilder.md`'s batched-shape verification example
  (`DataBatch(x=[2856, 13], ...)`, `u=[68, 34]`) is still stale and needs the
  same 13->15 / 34->35 correction.

`GraphAttentionNetwork.md` needs no changes — it already gives correct,
internally consistent dimensions and remains the right cross-document
reference.

### 2. Rebuild `DuelingDQN.md` as a compact live learner reference

**Scope corrected by verification:** the doc (now 27 lines) already states
the Double-DQN target and Smooth-L1 loss facts. Add only what's actually
still missing, verified from `risk/learning/dueling_dqn_agent.py` and
`train_constants.py`:

- replay batch size 64 (`BATCH_SIZE`) and one optimizer update per ready
  learner transition (`TRAIN_STEPS_PER_CALL = 1`);
- Adam learning rate `1e-4`, gradient norm clip `10`
  (`GRAD_CLIP_MAX_NORM`), hard target-network sync every 1,000 updates
  (`target_update_every`);
- the checkpoint filenames by name: `model.pt` and `replay.pt`.

Do not restore build plans, rollout/smoke logs, old test counts, or the dead
`value_mask is None` investigation.

### 3. Rebuild `Eval.md` around the implemented evaluator contract

**Scope corrected by verification:** most of the originally-listed facts are
already present in the current 27-line doc — no replay/learning during eval,
the two fixed suites, seed/seat rotation, cadence, and retention/manifest
behavior are all already stated. Add only what's still genuinely missing:

- name `GameFactory` as the mechanism for fresh eval contexts (as opposed to
  `SelfPlay`);
- the max step count (`EVAL_MAX_STEPS = MAX_STEPS_PER_EPISODE`);
- the complete metric-key list — the doc currently names only the three keys
  used in the score formula and omits `episode`, `eval_games`,
  `eval_avg_agent_turns_survived`, and `eval_score` itself;
- `Trainer`'s `eval_saved_best` handling;
- the inference that reattachment to a fresh environment means evaluation
  belongs at an episode boundary.

**The reward-boundary discrepancy is a confirmed real behavior gap, not just
an underdocumented detail.** `Evaluator` adds end-of-turn reward only on a
literal `FortifyAction`, unconditionally of `done`. `Trainer` and
`CheckpointEvaluator` (`risk/learning/choose_agent.py`, already documented by
name in `ChooseAgent.md`) both add it on `FortifyAction or done`. That means
`Evaluator` can silently skip end-of-turn reward when a learner's turn ends by
elimination or game-over through a non-fortify action — for example a winning
`AttackAction`/`OccupyAction`. Document this honestly, as originally planned.
Whether this asymmetry should also be fixed in `evaluator.py` itself is a
code question outside this documentation-only plan's scope — flag it for a
separate decision rather than silently aligning the doc to either behavior.

### 4. Create a factual learned-agent-play reference

**Confirmed accurate as originally scoped — no changes needed to this item.**
Every claimed behavior verified true against `risk/app/learned_agent_play.py`,
`risk/app/main.py`, `risk/ui/render/init_screen_view.py`, and
`risk/learning/choose_agent.py`. Do **not** restore the old
`PlayLearnedAgents.md` plan. Create a new `LearnedAgentPlay.md` that records:

- learned selections are UI-only placeholders until game construction;
- raw policy `.pt` and `epNNNNNN/model.pt` loading behavior;
- epsilon-zero/eval-mode setup and actual-seat attachment;
- invalid/incompatible file validation and Start-button behavior;
- presets from `Params/play_agents.json`;
- fresh, independent agent instances for each learned seat.

Index that new reference in `Content.md` once it exists. Neither the doc nor
the `Content.md` index entry exist yet.

### 5. Complete the live `Reward.md` semantics

**Confirmed still open.** `Reward.md`'s per-phase table was restored and then
deduplicated earlier today (it briefly existed as two overlapping tables at
different levels of formula detail — see `ChangeLog.md`); that work did not
touch any of these five gaps, all still confirmed against the current
109-line doc:

- terminal `-300` applies when the learner is eliminated mid-game
  (`reward_player in state.eliminated`), independent of full game-over
  (`Phase.GAME_OVER`) — confirmed in `environment.py`'s `done` computation.
  The doc's terminal row currently only distinguishes "wins" from "does not
  win," not "learner eliminated while others keep playing" from "game fully
  ends";
- max-step cutoff keeps `done=False` (confirmed: `done`'s computation has no
  step-count term at all), so that transition is non-terminal and may
  bootstrap; this fact currently lives only in `Trainer.md`, with no
  cross-reference from `Reward.md`;
- continent-domination reward is *gated* on
  `owned/total >= 1 / alive_players + 0.10` — the `0.10` margin value is
  already in the constants table, but the gating condition itself is still
  never written out, only the payout formula;
- `continent_advantage`'s doc prose still omits its two normalization
  divisions (`territory_edge / max(1 - baseline_share, eps)` and
  `troop_edge / 0.5`) applied before the final multiply — it should either
  state those or stop calling the definition exact;
- still no cross-reference explaining that one stored transition can span the
  learner's action plus intervening opponent actions before control returns.

## Priority 2: factual drift and maintainability repairs

| Document | Required current-state repair |
|---|---|
| `Trainer.md` | Confirmed still stale, and the plan's own original example number is already wrong too (a live illustration of why not to hard-code these). Update the launcher example to describe the mechanism rather than a specific run id — "`main()` selects the learner and run id; read `trainer.py` for the current one" — instead of naming today's value. Also confirmed missing: state that `done=True` covers game-over **or** learner elimination (`reward_player in state.eliminated`), with the max-step cutoff separate and non-terminal (`done`'s computation has no step-count term). |
| `PPO.md` | Confirmed still stale, and worse than "an older PPO version": the current launcher doesn't start any PPO run at all, it builds Dueling DQN. Remove the specific-run claim rather than swapping in a new one — state that the launcher's selected learner changes over time and should be read from `trainer.py`, not asserted here. Clarify terminal versus max-step semantics as above. |
| `HeuristicAgents.md` | The original "Killbot is absent from training" claim is **already fixed** — current doc correctly states Killbot is in both the training curriculum and eval rosters, matching `TRAIN_OPPONENT_AGENT_KINDS` and the evaluator's fixed suites. Newly confirmed instead: a leftover "Build plan" section (steps 1-9) doesn't just read as historical, it **directly contradicts** the doc's own corrected text elsewhere — it says Killbot is "ad-hoc only," "not added to the curriculum," "for testing/benchmarking only for now." Archive or remove that section; a self-contradictory document is worse than the original staleness. |
| `Testing.md` | Confirmed still missing: `test_player_card_settings.py` (the file exists; it's absent from the test map) and an explicit note that `test_agents.py` covers base `GNN_DQN_Agent`/DQN specifically (it does, with roughly a dozen tests, but the doc's current listing is generic). Drop the "restore compact fixture/convention guidance" sub-item as written — the doc already has a Conventions section covering fixtures and DQN-encoder-comparison guidance, and no concrete missing convention was identified. Only re-add it if a specific missing fact is found. |
| `ChooseAgent.md` | Confirmed still stale on all three counts. The loader description still claims `load_params` is used; the code calls `agent.net.load_state_dict(...)` directly, with no `load_params` anywhere in the load path — correct the description, don't just soften it. The "Add `test_choose_agent.py`" sentence is now dead future-tense: that file already exists and is already listed in `Testing.md`, so delete the sentence rather than rephrasing it — there's nothing left to plan for. The `"max_steps": 1000` example is still stale against the actual `2000` default (`EVAL_MAX_STEPS = MAX_STEPS_PER_EPISODE`). |
| `Environment.md` | Confirmed still stale, in a more specific way than first described. When no valid card set exists, the engine sets the phase directly to `REINFORCE_PLACE` before any action is taken — no `SkipTradeAction` is ever legal or exposed that turn. The current wording ("a skip action advances to reinforcement") misstates this as something the learner does; it should say the phase change is automatic, with no trade-in-related action available. |
| `BoardTopology.md` | No longer optional — confirmed self-contradictory. One sentence about `edge_index()` still says "future GNN support," while a separate "GNN Hook" section in the *same file* already correctly documents that `graph_adapter.py` uses it live and unmodified today. Fix the stale sentence; it now contradicts its own document, not just external code. |

## Material that should remain retired

Keep these out of the active reference set:

- `ADQN.md`, `PQN.md`, `Policy_Duel_DQN.md`, and `VQN.md`;
- `ModelSelection.md` and old checkpoint-selection planning;
- old environment smoke-run/checkpoint advice;
- DQN_104/DQN_105 findings, forecast ranges, and unimplemented reward ideas;
- duplicate conflict copies and former poster asset/layout instructions.

The applicable current rules from the former `EnvironmentActionPlan.md` are
already retained in `Action.md`: ordinary versus elimination card-trade flow,
`pending_attack`, multi-trade behavior, and bounded fortify candidates.

These stay retired specifically *because* they're tied to a past state of the
project (a specific run's findings, a specific superseded plan) rather than
describing current mechanism — the same reasoning that argues against writing
new current-state docs around today's specific run numbers.

## Proposed execution order after explicit approval

1. Correct all Priority 1 factual errors in graph, reward, and terminal
   semantics — using the corrected, narrower scope above, not the original
   line-count-driven estimate of how much is missing.
2. Expand Dueling DQN and evaluation documentation using only current code.
3. Add the new learned-agent-play reference and index it.
4. Apply the Priority 2 drift repairs and trim/archive obsolete heuristic-plan
   prose. Fix `HeuristicAgents.md`'s self-contradiction as part of this step,
   not as an afterthought — it's now actively misleading, not merely stale.
5. Re-run a documentation-to-code audit of every numeric shape, constant,
   learner label, test path, and evaluation setting before committing. Given
   how much drifted between this plan's first draft and its verification pass
   the same day, budget for this step catching new drift too, not just
   confirming the list above.

## Implementation gate

This plan is not an instruction to modify existing documentation. Make these
changes only after a specific user request to implement the approved subset.
