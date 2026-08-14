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

## 2026-08-14

- **Produced a readability-focused poster revision without changing the unfinished results band.**
  `Docs/Risk_poster_readability_update.pptx` shortens the central claim,
  makes the encoder flow explicit, replaces the unreadable attention-matrix
  illustration with a large Q/K/V sparse-attention flow, and increases the
  historical DQN_103 reward table to 22 px body text while retaining its key
  reward signals. `Docs/Poster.md` now mirrors the revised claim, numbered
  attention explanation, and poster-summary reward table. The source
  `Docs/Risk.pptx` remains untouched. Files: Docs/Risk_poster_readability_update.pptx,
  Docs/Poster.md, Docs/ChangeLog.md.

- **Expanded the poster's DQN_103 reward table into a readable phase summary.**
  The reward panel now uses the full available space for the historical reward
  equation and phase/trigger rows with key calculations and plain-language
  intent. It replaces unexplained labels such as “frontier + readiness” with
  concrete language about enemy borders and army strength versus the weakest
  adjacent enemy. Files: Docs/Risk.pptx, Docs/Poster.md, Docs/ChangeLog.md.

- **Added a phase-grouped reward-calculation reference.** The end of
  `Docs/Reward.md` now gives each phase's trigger, raw equations, and final
  scaling path for terminal, trade, reinforcement, attack, occupy, fortify,
  and board-progress rewards. It also makes explicit that action shaping is
  jointly clipped before scale `0.3`, while end-of-turn shaping is scaled but
  not clipped, and corrects the board comparison's timing. Files:
  Docs/Reward.md, Docs/ChangeLog.md.

- **Restored Reward.md's per-phase reward table, deleted in `529fb4c`.**
  That commit trimmed Reward.md from 526 to 28 lines and dropped the
  "Current constants" and per-action reward tables along with a large amount
  of superseded experiment history (DQN_104/105 investigation notes, an
  unimplemented favorable-attack-stop proposal). Restored only the two
  tables — verified line-by-line against the live `REWARD_*` constants and
  `reward.py` first, not copy-pasted as-is: the old table's end-of-turn row
  had gone stale even before the trim (`REWARD_TERRITORY_DELTA` documented
  as `1.00`, actually `20.00`; `REWARD_TERRITORY_HOLD` documented as `0.05`,
  actually `0.00`), and the `StopAttackAction` row needed updating to the
  current unfinished-attack-target tiering. The historical/planning sections
  were intentionally left out as superseded; ask if those should come back
  too. A follow-up line-by-line check against `reward.py` found one further
  error in the restored prose (not the tables): the logged `attack` W&B
  component excludes both `eliminate` and `unfinished_attack`, not just
  `eliminate` as first written — corrected. Files: Docs/Reward.md,
  Docs/ChangeLog.md.

- **Reworked the poster around reward shaping and held-out DQN_103 evidence.**
  The right-center technical panel now documents the historical DQN_103
  reward regime that produced the reported checkpoint result (terminal
  `+100/-100`, dense scale `0.1`); the DQN_103 3–6-player evaluation moved
  to the third bottom card. The middle card is now explicitly reserved for the
  planned DQN-versus-Dueling-DQN-versus-PPO comparison rather than labelled as
  a Dueling-only result. Files: Docs/Risk.pptx, Docs/Poster.md,
  Docs/ChangeLog.md.

- **Restored the policy head as DPQN's sole post-warm-up actor.** The plan no
  longer alternates DQN-controlled and policy-controlled collection. After the
  warm-up gate, the policy head samples every learner action and its 32-action
  frozen blocks feed both replay and one-use actor memory; DQN remains the
  off-policy learner and evaluation uses policy argmax. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Made DPQN DQN-primary after the actor gate.** The proposed cadence now
  alternates 32 epsilon-greedy DQN-controlled learner transitions with a
  frozen 32-transition policy block. Every policy-block transition enters both
  replay and one-use actor memory; the 32 delayed replay updates preserve the
  standalone DQN's 64 replay samples per fresh transition. Checkpoints are
  evaluated in separate whole-game Q-only and policy-only modes, never by
  switching heads during a game. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Merged a second, duplicate "related work" pass into the existing
  literature/novelty section.** Two overlapping related-work write-ups ended
  up in DPQN.md (this session added one citing Reactor/PGQL/Q-Prop/AWR/
  Expected Sarsa; a "Related literature and novelty" section citing
  Actor-Advisor/PGQL/BDPI/ACER/Mean Actor-Critic/Discrete SAC was already
  present, apparently from a parallel edit — Docs/DPQN.md is untracked, so
  there is no git history to confirm provenance). Kept the table-based
  section and folded in the non-redundant points: added a Reactor row (the
  closest existing precedent for one trunk feeding both a policy and a Q
  head, which qualifies rather than fully supports the "shared encoder"
  novelty bullet), tied the Mean Actor-Critic row to the classical Expected
  Sarsa identity behind DPQN's $b_t$ baseline, and noted AWR/AWAC's
  exponentiated-advantage weighting as an alternative to DPQN's raw-advantage
  REINFORCE next to the Q-Prop row. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Added DPQN related work and scoped its novelty claim.** The design now
  cites the closest policy-gradient/Q-learning, replay-actor, and
  discrete-entropy precedents, identifies Actor-Advisor as the strongest
  structural baseline, and distinguishes DPQN's new project-specific
  configuration from a new general RL algorithm family. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Set DPQN's actor-loss warm-up gate to the existing epsilon schedule.**
  The actor now enables at `episode > EPSILON_DECAY_EPISODES` (epsilon at
  `EPSILON_END`, i.e. after episode 100) instead of an unspecified placeholder
  threshold, reusing the already-tuned decay schedule rather than new warm-up
  constants. The doc now also calls for logging `cumulative_optimizer_steps`/
  `cumulative_learner_turns` at gate-open time, since episode count is a
  schedule rather than a direct Q-maturity measurement and episode length
  varies with the randomly sampled player count/opponent mix. Files:
  Docs/DPQN.md, Docs/ChangeLog.md.

- **Switched DPQN's DQN replay batch to 64, matching standalone DQN.** The
  cadence changed from 16 updates of batch 128 to 32 updates of batch 64 per
  32-transition collection block (1 joint + 31 DQN-only), so DPQN now reuses
  the exact batch size and per-transition update frequency already validated
  by the live DQN/Dueling DQN agents (`BATCH_SIZE = 64`,
  `TRAIN_STEPS_PER_CALL = 1`) instead of an untested batch size, while
  preserving the same 64-samples-per-transition replay exposure. Replay and
  actor-loss eligibility thresholds, the target-sync counter step, and the
  device-memory open item were updated to match. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Added reviewer notes to DPQN.md.** Second-pass review confirms the
  warm-up gate and detached expected-Q baseline resolved the two open
  correctness concerns from the first pass, and flags one implementation-time
  follow-up: the "device memory" open item should also cover the 32-state
  legal-action group forward, not just the 128-transition DQN replay batch.
  Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Specified DPQN's post-warm-up action selector.** Training switches from
  epsilon-greedy DQN behavior to masked policy-head sampling when the actor
  gate opens; evaluation takes the policy argmax, and DQN remains off-policy
  on the resulting replay transitions. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Strengthened the DPQN actor signal and warm-up plan.** DPQN now uses a
  detached policy-expected Q baseline rather than raw returns, and defers all
  actor/entropy updates until dedicated replay-size and DQN-update warm-up
  thresholds are met; DQN can train alone using its established behavior
  first. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Added DPQN Q-policy alignment diagnostics to the plan.** The design now
  logs detached policy-expected versus greedy-Q values, regret, sampled-action
  regret, argmax agreement, rank correlation, and entropy before considering
  any later Q-to-policy distillation loss. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Set DPQN's policy bootstrap horizon to four learner transitions.** This
  balances real short-horizon rewards with the stable DQN continuation while
  retaining the 32-transition collection block and 16 replay updates. Files:
  Docs/DPQN.md, Docs/ChangeLog.md.

- **Added actor-only entropy regularization to the DPQN plan.** The policy
  objective now rewards entropy across each state's legal action distribution
  with its own coefficient, while the DQN target and Q-loss semantics remain
  unchanged. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Defined DPQN v1's training cadence.** The plan now collects 32 frozen
  policy actions, uses 8-step returns, takes one joint actor/DQN update, then
  15 DQN-only updates of replay batch 128; this retains DQN's 64 replay-sample
  presentations per new learner transition. Files: Docs/DPQN.md,
  Docs/ChangeLog.md.

- **Restored DPQN's compact bootstrap shorthand.** The n-step policy return
  again uses B for the final DQN continuation, with its Double-DQN definition
  directly below the equation. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Simplified DPQN's displayed DQN target notation.** The equation now
  defines the terminal continuation as zero and states the terminal
  reward-only case in prose, while the implementation continues to use its
  done mask internally. Files: Docs/DPQN.md, Docs/ChangeLog.md.

- **Simplified DPQN v1 to standard DQN plus an actor head.** The proposal now
  removes Dueling's clean-state/value/advantage machinery and starts with a
  detached bootstrapped return rather than a learned baseline, isolating the
  DQN-plus-policy experiment; Dueling DPQN is deferred as a later comparison.
  Files: Docs/DPQN.md, Docs/Content.md, Docs/ChangeLog.md.

- **Documented the proposed DPQN hybrid learner.** The design preserves the
  working Double-DQN replay objective while adding a one-use recent-policy
  memory and detached, bootstrapped Q-guided policy loss through a shared graph
  encoder; it explicitly remains unimplemented pending the listed design
  choices. Files: Docs/DPQN.md, Docs/Content.md, Docs/ChangeLog.md.

## 2026-08-13

- **Configured a fresh Dueling DDQN comparison run.** The trainer launcher now
  builds `Dueling_DQN_Agent` for `Dueling_DQN_304`, without resuming the
  interrupted DQN_303 checkpoint or its W&B history. Training was not started.
  Files: `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/ChangeLog.md`.

- **Added course, lecturer, and institutional branding to the A0 poster.**
  The poster header now identifies *Practical Deep Learning for Science*,
  Prof. Eilam Gross, Gilad Markman, and 2026, and includes the supplied
  Weizmann Institute of Science logo. Files: `Docs/Risk.pptx`,
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Configured the trainer to resume interrupted `DQN_303`.** The launcher now
  restores the newest local DQN checkpoint and resumes the original W&B history
  (`rujxk3x7`) after the Windows Update restart stopped its process. Files:
  `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/ChangeLog.md`.

- **Raised the default training-episode count from 10,000 to 100,000.**
  `TRAIN_EPISODES` in `risk/learning/train_constants.py` now defaults
  `python -m risk.learning.trainer` to a 100,000-episode run; no other
  constant or code path references the old literal, and no test hardcodes
  the value. Files: `risk/learning/train_constants.py`, `Docs/ChangeLog.md`.

- **Implemented PPO's global post-epoch KL gate for fresh `PPO_312`.** PPO now
  completes an epoch before measuring sample-weighted k3 KL across its cached
  rollout in safe minibatches, blocking only a later epoch when that global
  value exceeds the target. Added post-epoch KL and stop-epoch metrics, focused
  coverage, and the fresh launcher. Files: `risk/learning/ppo_agent.py`,
  `risk/learning/trainer.py`, `Temp/tests/test_ppo.py`, `Docs/PPO.md`,
  `Docs/Trainer.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Logged a deferred correction for trade-in action representation; no learner
  code changed.** Current trade-in scoring repeats the base graph and embeds
  hand-slot positions rather than actual cards, so it cannot observe selected
  card identities, territory matches, symbols, wilds, or retained cards. The
  next-model plan is selected-card node injection, a real order-invariant hand
  encoder, focused tests, input-versioning, and retraining of DQN, Dueling
  DQN, and PPO. Current checkpoints and poster results remain unchanged.
  Files: `Docs/ActionGraphBuilder.md`, `Docs/NetworkArchitectures.md`,
  `Docs/ChangeLog.md`.

- **Made the current trade-in observation explicit in its deferred plan.** The
  plan now lists the only card-related inputs available today — next-set value,
  per-player card counts, accumulated reinforcement budget, and positional
  `(i, j, k)` slots — and contrasts them with the missing candidate-specific
  card, territory-match, wild, and retained-hand information. Files:
  `Docs/ActionGraphBuilder.md`, `Docs/ChangeLog.md`.

- **Raised two small A0-poster captions to a readable size.** In
  `Docs/Risk.pptx`, the GitHub QR caption and the fixed-checkpoint evaluation
  caveat are now 18 pt (up from 12.75 pt and 13.5 pt); their positions and all
  other poster content are unchanged. Files: `Docs/Risk.pptx`,
  `Docs/ChangeLog.md`.

- **Clarified the poster's GitHub QR label.** `Docs/Risk.pptx` retains the
  top-right project-repository QR and now pairs it with `PROJECT REPOSITORY`
  and GitHub's white lockup, without implying that the experiment results are
  published on GitHub. Files: `Docs/Risk.pptx`,
  `Assets/GitHub_Lockup_White.png`, `Docs/ChangeLog.md`.

## 2026-08-12

- **Added a fixed DQN_103 checkpoint-evaluation panel to the print poster.**
  `Docs/Risk.pptx` now uses the middle band for a sparse action-injection
  explanation and the colorful top-five DQN_103 win-rate chart across 3–6
  players. It states the 54-game, three-seed, rotated-seat, epsilon-0,
  2,000-step protocol; reports `ep006700` as 46/54 wins (85.2%) with zero
  timeouts; and labels the evidence as checkpoint-selection, not a
  DQN/Dueling-DQN/PPO comparison. `Docs/Poster.md` mirrors the figure order,
  method wording, and scope caveat. Files: `Docs/Risk.pptx`,
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Refocused the print poster on the scientific method and locked its
  three-chart results geometry.** `Docs/Risk.pptx` removes the player/agent
  selection UI and heuristic roster, reduces the board-to-graph card, and
  enlarges the action-injection and shared GATN panels. Its results band now
  has equal DQN, Dueling DQN, and PPO slots; the latter two visibly repeat
  `Assets/DQN Win.png` only as labelled DQN-data placeholders until matched
  curves exist. `Docs/Poster.md` mirrors the revised figure order and the
  explicit placeholder disclosure. Files: `Docs/Risk.pptx`,
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Put the DQN win-rate chart into the print poster's shared results frame.**
  Docs/Risk.pptx now replaces its repeated learner-description cards with
  Assets/DQN Win.png, labelled truthfully as a temporary DQN-only placeholder
  for the future DQN/Dueling DQN/PPO comparison. Docs/Poster.md uses the same
  wording so the design brief and visible poster agree. Files: Docs/Risk.pptx,
  Docs/Poster.md, Docs/ChangeLog.md.

- **Reviewed the planned global post-epoch KL fix and closed several open
  items.** `Docs/PPO.md`'s "Planned: global post-epoch KL stopping" section
  (addressing `PPO_311`'s noisy per-minibatch KL check stopping ~75% of each
  rollout's minibatches) was directionally sound; added: `_max`-suffixed
  metrics don't need `unweighted_update_metrics` registration (confirmed by
  reading `trainer.py`'s `_aggregate_update_metrics` — the `_max` branch
  ignores weight entirely, matching the existing `ppo_early_stop_kl_max`
  precedent, which the plan's blanket instruction would have contradicted); a
  defined value for `ppo_kl_stop_epoch` when no early stop occurs; a
  running-sum (not equal-weighted chunk-average) requirement for the global
  KL, future-proofing against `PPO_ROLLOUT_LENGTH`/`PPO_MINIBATCH_SIZE` not
  dividing evenly later; a pointer to reuse `_evaluate_indices`/
  `_forward_grouped` instead of a new log-prob-only path; and a monitoring
  note about epoch 1 now always running unconstrained. No code changed —
  plan-only, per instruction not to implement yet. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Added `Assets/DQN Win.png` to the poster and the pptx deck.**
  `Docs/Poster.md`'s Figure 7 placeholder now shows the actual DQN rolling
  win-rate chart (DQN_103, DQN_105, DQN_303), captioned as DQN's own
  training progress pending the Dueling DQN/PPO comparison. `Docs/Risk.pptx`
  is a fixed-layout, print-ready A0 single slide with no open space in its
  Results band, so rather than disturb that layout the image was added as a
  new second slide (title, context line, image, caption) — a supporting
  slide, not part of the print poster itself. Installed `python-pptx` in
  `C:\venvs\ai-rl` to make the edit (previously absent from that
  environment). Files: `Docs/Poster.md`, `Docs/Risk.pptx`, `Docs/ChangeLog.md`.

- **Planned global post-epoch PPO KL stopping.** `Docs/PPO.md` now scopes a
  fresh-run follow-up for PPO_311's noisy minibatch KL stops: complete each
  epoch, measure sample-weighted k3 KL across the cached rollout in safe
  batches, and decide whether to begin the next epoch. It preserves the
  16-step targets and requires no environment, trainer, or other-learner
  changes. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Created an A0 PowerPoint version of the project poster.** `Risk.pptx`
  turns `Docs/Poster.md` into a one-slide landscape poster using the local
  board, UI, action-injection, GATN, and encoder-summary visuals. Files:
  `Risk.pptx`, `Docs/ChangeLog.md`.

- **Simplified the README table of contents.** It now links only the main
  sections, without nested subsection entries. Files: `README.md`,
  `Docs/ChangeLog.md`.

- **Added a README table of contents.** The front page now links its game,
  GATN/RL, training, and documentation sections from the introduction. Files:
  `README.md`, `Docs/ChangeLog.md`.

- **Linked contributor instructions to the documentation index.** `AGENTS.md`
  and `CLAUDE.md` now direct future sessions to `Docs/Content.md` before
  locating subsystem documentation. Files: `AGENTS.md`, `CLAUDE.md`,
  `Docs/ChangeLog.md`.

- **Added the full documentation index to the README.** The repository front
  page now links every active reference document by topic, matching
  `Docs/Content.md`. Files: `README.md`, `Docs/ChangeLog.md`.

- **Made the README the active guide for interactive learned-agent play.**
  Added its current policy-loading behavior to the player-selection section,
  indexed `README.md` as the primary app-use document, and moved the completed
  `PlayLearnedAgents.md` implementation plan to `Temp/retired_documents/plans/`.
  Files: `README.md`, `Docs/Content.md`,
  `Temp/retired_documents/plans/PlayLearnedAgents.md`,
  `Temp/tests/test_learned_agent_play.py`, `Docs/ChangeLog.md`.

- **Grouped the documentation index by topic.** `Docs/Content.md` now
  separates game/environment, agents/UI, graph/GATN, RL/evaluation, and
  project-support references. Files: `Docs/Content.md`, `Docs/ChangeLog.md`.

- **Added an active-document index.** `Docs/Content.md` links every current
  documentation file and summarizes its purpose, while separating retired
  material under `Temp/` from the active reference set. Files:
  `Docs/Content.md`, `Docs/ChangeLog.md`.

- **Declared pytest as a test dependency.** `requirements.txt` now includes
  `pytest`, matching the documented full-suite command. Files:
  `requirements.txt`, `Docs/ChangeLog.md`.

- **Reframed the README opening around the GATN training project.** The title,
  RL/GATN badges, and opening description now foreground legal-action graph
  injection, self-play training, and the DQN/Dueling-DQN/PPO comparison.
  Files: `README.md`, `Docs/ChangeLog.md`.

- **Added a concise graph-attention RL overview to the README.** It now
  explains the board graph, legal-action injection, sparse GATN encoder, and
  the controlled DQN/Dueling-DQN/PPO comparison, with graph and injection
  visuals from the poster assets. Files: `README.md`, `Docs/ChangeLog.md`.

- **Expanded the README game guide with poster-based explanation and UI
  visuals.** It now explains Risk's objective, dice combat, turn phases,
  player types, and legal-action flow, and embeds the playable-board and
  player-selection images. Files: `README.md`, `Docs/ChangeLog.md`.

- **Reduced the encoder-reference image widths.** The four embedded encoder
  diagrams in `Docs/GraphAttentionNetwork.md` now render at 1,200 pixels wide
  for a more compact reading layout. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Added encoder calculation visuals to the graph-attention reference.**
  `Docs/GraphAttentionNetwork.md` now embeds the three detailed encoder
  calculation pages and the concise matrix summary used on the poster. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Split the poster headline callout into two lines.** `Docs/Poster.md`'s
  "That encoder is trained..." sentence now starts on its own blockquote
  paragraph instead of running on from the injection sentence. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Added the RL training angle to the poster's headline callout.**
  `Docs/Poster.md`'s main claim covered only the graph/injection
  representation; it now adds a second sentence stating the encoder is
  trained through reinforcement learning, compared across DQN, Dueling DQN,
  and PPO — the user flagged this as a very important part missing from the
  headline. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Reverted the poster's headline callout, tuned rather than rewritten.**
  The full plain-language rewrite (previous entry) dropped "injects," which
  the user wanted kept. `Docs/Poster.md` now restores the original
  "injects each legal Risk move into the board graph..." wording, with only
  its opening clause tuned ("predicting from one enormous fixed action
  space" → "scoring from one enormous fixed list of moves") rather than
  reworded end to end. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Simplified the poster's headline callout for a first-time reader.**
  `Docs/Poster.md`'s main claim previously used jargon ("fixed action space,"
  "state–action graph," "graph-attention encoder") a reader hasn't been
  introduced to yet; it now says the same thing in plain terms — the network
  looks at the map with one legal move drawn on it and scores just that
  option, repeated per legal move with the same shared network. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Made the poster's opening sentence more general.** `Docs/Poster.md`'s
  subtitle now reads "A graph-attention reinforcement-learning approach to
  Risk, compared across DQN, Dueling DQN, and PPO" instead of leading with
  phase-level implementation detail (trade-in/reinforce/attack/occupy/
  fortify), combining the user's two suggested directions — a general
  framing plus naming the three compared learners. Files: `Docs/Poster.md`,
  `Docs/ChangeLog.md`.

- **Added practical player-selection and training instructions to the README.**
  It now gives the setup-screen steps for choosing human, heuristic, and
  learned seats, plus the exact trainer settings to edit before launching a
  DQN, Dueling DQN, or PPO run. Files: `README.md`, `Docs/ChangeLog.md`.

- **Rewrote the README as a current project guide.** Removed stale future-RL
  and historical-roadmap language; documented the current game architecture,
  supported learner set, trainer/checkpoint workflow, verified test command,
  and active reference documents. Files: `README.md`, `Docs/ChangeLog.md`.

- **Rewrote `Docs/Poster.md` as poster content instead of a production
  brief.** Stripped all instructional/meta commentary (placement directions,
  "reader-facing copy" labels, production rules, asset register, results
  wording-rule table) and left only the words and images that would actually
  appear on the poster, plus the layout diagram/region-share table (kept per
  explicit exception). Added a new "problem" panel (huge, mostly-illegal
  action space; a flat vector losing the board's relational structure) ahead
  of the graph/injection "idea" panel, since the prior version never stated
  the problem before its solution. All 6 image assets are still referenced
  (none dropped); Figures 7–9 remain reserved, caption-only placeholders for
  the not-yet-available training-result charts. Files: `Docs/Poster.md`,
  `Docs/ChangeLog.md`.

- **Replaced completed planning documents with current-code references.**
  `Docs/Environment.md` now documents the implemented card-trade and fortify
  behavior; `Docs/Reward.md` now documents only the live reward pipeline.
  Moved the completed environment checklist and unimplemented model-selection
  proposal to `Temp/retired_documents/plans/`. Files: `Docs/Environment.md`,
  `Docs/Reward.md`, `Temp/retired_documents/plans/`, `Docs/ChangeLog.md`.

- **Moved the board-screenshot Figure 1 into the poster's game-background
  primer.** `Docs/Poster.md` now places Figure 1 (the colored playable-board
  screenshot) in the header, next to the new Risk rules primer, instead of
  alongside Figure 2 in section 1. Updated section 1's heading/text (now just
  Figure 2, the board-to-graph transformation), the layout ASCII diagram,
  the header/game-visuals region shares (9%→11%, 43%→41%), and the asset
  register to match. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Retired ADQN/PQN and archived the related experiment records.** The live
  learner factory now accepts only DQN, Dueling DQN, and PPO; removed ADQN/PQN
  constants, logger wiring, and active test coverage. Their code, tests, and
  documents now live under `Temp/retired_algorithms/`; stale conflicted docs
  moved to `Temp/retired_documents/conflicts/`. Rewrote the active learning,
  architecture, trainer, evaluation, test, and README references to match the
  supported implementation. Files: `risk/learning/trainer.py`,
  `risk/learning/train_constants.py`, `risk/learning/training_logger.py`,
  `Temp/tests/test_trainer.py`, `Temp/tests/test_training_logger.py`,
  `Docs/DuelingDQN.md`, `Docs/Eval.md`, `Docs/NetworkArchitectures.md`,
  `Docs/Testing.md`, `Docs/Trainer.md`, `README.md`,
  `Temp/retired_algorithms/`.

- **Added a Risk rules primer to the poster brief.** `Docs/Poster.md` now
  opens with a short "Game background" section (goal, dice-combat mechanics,
  and the five turn phases in one sentence) for readers unfamiliar with Risk,
  placed before "The one idea a reader should remember" with a note that it
  belongs near the header/Figure 1 as a caption, not a full poster section.
  Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Rebuilt the poster into a visual-first evidence story.** Docs/Poster.md
  now uses the original high-resolution board, graph, UI, action-injection,
  network, and encoder assets in a coherent A0 layout, with a full-width
  three-chart results band reserved for matched DQN, Dueling DQN, and PPO
  comparisons. The player-selector roster remains beside its UI image; dense
  encoder mathematics is now a secondary inset. Files: Docs/Poster.md.

- **Moved the heuristic-agent table beside the player-selection UI.** The
  poster now explains Random, Raider, Sentinel, Empire, and Killbot immediately
  after the start-screen image, where readers first encounter those options.
  Files: `Docs/Poster.md`.

- **Added a poster-ready encoder matrix summary.** `Docs/Poster.md` now embeds
  the concise sparse-attention overview beneath the network pipeline, keeping
  the exact action-injection idea visible without adding the full derivation to
  the poster. Files: `Docs/Poster.md`, `Assets/encoder_matrix_summary.png`,
  `Assets/RiskMap/encoder_matrix_summary.png`.

- **Corrected PPO equation formatting.** `Docs/PPO.md` now uses consistently
  rendered display-math delimiters and conventional subscripts, spacing, and
  policy-conditioning notation for the n-step target and clipped PPO loss.
  Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Implemented PPO's 16-step bootstrapped targets.** `PPO_Agent` now uses
  stored `old_value[t + 16]` for ordinary continuations, performs clean
  value-only forwards only at non-terminal boundaries and tails, and logs
  target horizon/bootstrap diagnostics. `PPO_N_STEP = 16` is exported for W&B;
  no environment, trainer, or other learner changed. Focused PPO/logger tests:
  29 passed. Files: `risk/learning/ppo_agent.py`,
  `risk/learning/train_constants.py`, `Temp/tests/test_ppo.py`,
  `Docs/PPO.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Verified the 16-step PPO plan's newest details and closed a doc-sync
  gap.** Confirmed `test_training_logger.py`'s config test iterates
  `train_constants.__all__` generically (no hardcoded key list), so adding
  `PPO_N_STEP` there needs no test edits, just a run — the plan's step 6 was
  already correct on that point. Confirmed step 2's `t + n` boundary
  reasoning (old_value reuse is valid regardless of what transition `t + n`
  itself later does, since it values the *pre-action* state) is sound. Found
  one real gap: `Docs/Testing.md`'s `test_ppo.py` description still describes
  "complete-episode return targets," the Monte-Carlo design step 6 replaces;
  `Docs/PPO.md` now calls for updating that description alongside the test
  changes. No code changed — plan-only, per instruction not to implement yet.
  Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Closed indexing and experiment-config gaps in the 16-step PPO plan.**
  `Docs/PPO.md` now makes terminal/boundary handling exact at the `t + n`
  edge, requires `PPO_N_STEP` to be exported so the existing logger records it
  in W&B, and defines the bootstrap diagnostic structurally rather than by a
  value's incidental numeric magnitude. The scoped tests now include the
  existing logger configuration check as well as PPO coverage. No code changed.
  Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Closed a metric-weighting gap in the 16-step PPO plan.** Its two new
  diagnostics (`ppo_target_horizon_mean`, `ppo_target_bootstrap_fraction`)
  are rollout-wide, once-per-update values, the same kind as
  `ppo_return_mean`/`ppo_advantage_mean` — which are already registered in
  `PPO_Agent.unweighted_update_metrics` so `Trainer` averages them equally
  across rollout updates instead of weighting them like a per-minibatch loss.
  The plan's step 5 named the two new diagnostics but not that requirement;
  `Docs/PPO.md` now calls it out explicitly. Also pointed step 3 at reusing
  the existing `_clean_value_entry(...)`/`_forward_grouped(...)` helpers for
  boundary/tail evaluation instead of implying a new value-only path — they
  already do exactly this. No code changed — plan-only, per instruction not
  to implement yet. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Clarified the 16-step PPO plan's time scale, old-policy bootstrap timing,
  and diagnostics.** `Docs/PPO.md` now defines 16 as learner transitions,
  requires deduplicated boundary/tail values to be evaluated under `no_grad`
  before optimizer updates, and adds horizon/bootstrap observability so the
  experiment's shortened targets can be measured. No code changed. Files:
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Reviewed and closed gaps in the 16-step bootstrapped PPO plan.**
  `Docs/PPO.md`'s new n-step plan (a response to `PPO_310` showing no
  learning under full Monte-Carlo returns) was directionally sound but
  missing: precedence when a transition is both `done=True` and
  `gae_boundary=True` (same edge case already fixed for the Monte-Carlo
  implementation — `done` must win), a test for its own "boundary value
  shared across every earlier target" requirement, and a note not to resume
  `PPO_310`'s checkpoint since its critic was trained against a different
  regression target. Also clarified that this plan approximates the paper's
  GAE architecturally but not as an estimator (one fixed cutoff vs. a
  $\lambda$-weighted blend), and that a bigger fixed n is not the right
  escalation if 16 doesn't work — real GAE is. No code changed — plan-only,
  per instruction not to implement yet. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Changed the next PPO target experiment from GAE to configurable 16-step
  bootstrapping.** `Docs/PPO.md` now specifies `PPO_N_STEP = 16`, precise
  terminal/boundary/tail handling, and reuse of the stored value at `t + 16`;
  it continues to forbid all-successor action-graph evaluation and leaves code
  unchanged. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Replaced PPO's accumulated design history with a current-state reference
  and a paper-style GAE plan.** `Docs/PPO.md` now documents the implemented
  Monte-Carlo return targets, their boundary bootstrap behaviour, and the
  existing clipped PPO optimization without preserving superseded plans. It
  adds a scoped follow-up plan to restore efficient GAE by reusing stored
  successor values and evaluating only missing boundary/tail values; no code
  changed. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

## 2026-08-11

- **Added a one-sentence-per-heuristic-agent table to the poster brief.**
  `Docs/Poster.md`'s "Training opponents" section now lists Random, Raider,
  Sentinel, Empire, and Killbot each in one sentence, sourced from
  `risk/agents/random_agent.py` and `risk/agents/heuristic_agent.py`. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Fixed a stale comment in `ActionGraphBuilder`.** Its module docstring
  claimed non-attack actions (reinforce/occupy/fortify) write into `x`'s
  army-count column; the actual code (and `Docs/GraphAttentionNetwork.md`)
  writes into the separate proposed-army-delta column instead — verified
  `GraphAttentionNetwork.md` against the running code (dimensions, PyG
  `TransformerConv` internals, parameter counts) while answering a question
  about its accuracy, and found the doc correct but this comment out of
  date. Files: `risk/learning/action_graph_builder.py`, `Docs/ChangeLog.md`.

- **Linked the sparse-attention notation to PyG's actual execution.** The
  graph-attention guide now notes that `message(...)` computes the segment
  softmax during `propagate(...)`, while `propagate(..., aggr="add")`
  automatically performs the segment sum. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Adopted column-vector notation for sparse attention weights.** The
  graph-attention guide now treats `alpha` as `[166 x 1]`, making the
  weighted-message multiplication directly `alpha * V_edge`. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Defined `target_index` in the graph-attention input table.** The sparse
  encoder guide now introduces `target_index = edge_index[1]` before its use
  in the segment-softmax and segment-sum operations. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Rewrote the graph-attention guide to match the actual sparse encoder.**
  `Docs/GraphAttentionNetwork.md` now describes `TransformerConv`'s 166-edge
  gather, edge projection, segment softmax, segment aggregation, and internal
  plus outer residual paths rather than a dense masked-attention analogy.
  Files: `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Removed the orphaned `PPO_GAE_LAMBDA` constant.** It was PPO-only (no
  other learner ever read it) and, since `ppo_agent.py`'s complete-episode
  return-target fix, no longer imported or used at all — left in place it was
  exactly the silent-inactive-knob risk `Docs/PPO.md` had flagged: a value
  sitting at `0.95` that looked live but did nothing. User confirmed PPO-only
  constants in the shared `train_constants.py` are fair game to change/delete
  for this fix, superseding the earlier "shared constants are out of scope"
  boundary. Updated `Docs/PPO.md`'s "Constants" table, the corrected
  return-target plan's `PPO_GAE_LAMBDA` bullet, and the "Implementation
  status" note to match; left the historical `PPO_200`/`PPO_301` config
  tables untouched since those record what those runs actually used at the
  time. Verified with the focused suite only (`Temp/tests/test_ppo.py`, 14
  passed) per instruction not to run unnecessary tests. Files:
  `risk/learning/train_constants.py`, `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Verified the implemented PPO return-target fix against the finalized plan
  and fixed one real regression along the way.** Read `ppo_agent.py` fresh
  against `Docs/PPO.md`'s corrected design: `_next_values`/`_gae` are gone,
  replaced by `_return_targets`/`_boundary_values`/`_clean_value_entry`
  exactly as planned, and all four new `test_ppo.py` cases (interior boundary,
  implicit last-position boundary, clean-graph-only bootstrap, zero-boundary
  no-forward) pass, confirming every bug found during plan review is actually
  fixed. Along the way, briefly changed `_boundary_values` to read
  `transition.next_state.perspective` instead of `self.player_id` for
  consistency with `_cache_transition_entry`'s pattern — this broke
  `test_boundary_values_build_clean_rows_without_legal_actions`, because
  `State` has no real `perspective` field (it's only ever bolted on by
  `remember()`, and this test pushes transitions directly into
  `RolloutBuffer`, bypassing that). Reverted; `self.player_id` was correct
  as originally implemented. Separately, briefly removed `PPO_GAE_LAMBDA`
  from `train_constants.py` as an orphaned constant, then reverted that too
  after finding it was a deliberate, already-documented scope decision (this
  changelog's "Constrained the PPO return-target fix to PPO-owned code"
  entry) to leave shared constants untouched — `risk/learning/ppo_agent.py`
  and `Docs/PPO.md` are the only files this fix is allowed to touch. Full
  suite: 414 passed, 1 skipped. Files: `risk/learning/ppo_agent.py` (net
  no-op), `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Implemented PPO's boundary-only return targets.** `PPO_Agent` now builds
  full episode returns backward, evaluates clean critic graphs only for forced
  cutoffs and an unfinished rollout tail, and no longer expands every
  next-state legal-action set. Focused tests cover terminal, cutoff, tail, and
  value-only batching behavior. The scoped change leaves the environment,
  trainer, shared constants, and other learners untouched. Files:
  `risk/learning/ppo_agent.py`, `Temp/tests/test_ppo.py`, `Docs/PPO.md`,
  `Docs/Testing.md`, `Docs/ChangeLog.md`. Verified: focused PPO suite 14
  passed; full suite 414 passed, 1 skipped.

- **Constrained the PPO return-target fix to PPO-owned code.** `Docs/PPO.md`
  now limits implementation to `ppo_agent.py` and its focused tests; it
  explicitly excludes the environment, trainer, shared constants, graph/action
  code, and all other learners. The legacy GAE constant remains untouched in
  the shared module and is no longer an active PPO return-target setting. The
  older rollout and PPO_200 plans are marked historical so their old shared
  file references cannot be mistaken for current authorization. Files:
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Closed a zero-boundary edge case in the PPO GAE bug-fix plan before coding
  it.** A rollout can legitimately have no forced cutoffs and happen to end
  its last transition on a real terminal, needing no bootstrap anywhere;
  calling the planned value-only batched forward with an empty boundary list
  in that case would hit `PPO_Net.forward`'s `not value_mask.any()` guard
  (`ppo_net.py:43`) and raise `ValueError`. `Docs/PPO.md` now calls for
  skipping the value-only forward entirely when the collected boundary list is
  empty, plus a test for that exact scenario. No code changed — plan-only, per
  request to check the plan again before implementing. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Closed an implementability gap in the PPO GAE bug-fix plan's clean-graph
  boundary evaluation, before coding it.** `Docs/PPO.md` required boundary
  bootstraps to use clean base graphs with no injected action rows, but the
  existing `_decision_rows(...)` helper it would naturally reuse breaks on an
  empty action list — its clean-row `phase` value is borrowed from the first
  legal action's encoding, which doesn't exist when there are no actions,
  producing a 0-length `phase` tensor against a 1-length `rows` list. Verified
  `PPO_Net.forward` itself already tolerates an all-clean, no-action batch
  without changes (its per-phase-head loop no-ops when the action mask is
  empty), so the plan now calls for a small dedicated clean-row builder
  instead of reusing `_decision_rows` with `actions=[]`. No code changed —
  plan-only, per request to check the plan again before implementing. Files:
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Aligned PPO's headline conclusion with its boundary rules.** The main
  return-target summary in `Docs/PPO.md` now names every non-terminal boundary
  (interior forced cutoffs and a final rollout tail), rather than only the
  final tail, and its test plan now proves the clean-graph value-only network
  path returns values without policy logits. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Tightened the PPO return-target plan's remaining critic details.**
  `Docs/PPO.md` now requires bootstrap evaluation to use clean state graphs
  only (no legal-action expansion), bounds that batch by actual boundaries,
  renames the planned GAE helper to avoid misleading terminology, and removes
  the risk of `PPO_GAE_LAMBDA` becoming a silent inactive setting. Files:
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Removed a "seed G=0" ambiguity from the PPO GAE bug-fix plan before
  coding it.** `Docs/PPO.md`'s corrected `_gae(...)` steps said to "seed `G =
  0`" for a terminal transition, which read as if the terminal transition's
  own return becomes zero — wrong, since it would drop that transition's own
  reward (including the dominant `+300/-300` terminal reward) from its return
  and value target. Reworded so `G[i] = r[i] + gamma * bootstrap[i]` always
  includes the transition's own reward, and only the `bootstrap[i]`
  continuation term (0 / fresh `V(next_state)` / carried `G[i+1]`) is chosen
  by the done/boundary/contiguous priority. No code changed — plan-only, per
  request to check the plan again before implementing. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Clarified the critic's role under complete-episode PPO returns.**
  `Docs/PPO.md` now distinguishes the required current-state critic forwards
  used to learn `V(state) -> G` from the redundant next-state bootstrap
  forwards that the corrected return design removes. Also repaired the
  boundary-marker contract sentence. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Closed a last-position gap in the PPO GAE bug-fix plan before coding it.**
  `Docs/PPO.md`'s corrected `_gae(...)` plan bootstrapped only on
  `gae_boundary=True`, but that flag is only ever set by a
  `MAX_STEPS_PER_EPISODE` cutoff (`mark_last_boundary()`); an ordinary
  mid-game rollout fill (the common case) leaves the final transition
  unflagged, which would have made the corrected `_gae` silently skip the one
  bootstrap that position needs. The plan now also treats "last transition in
  the passed-in sequence" as an implicit boundary, gives `done` explicit
  priority over `gae_boundary` when both land on the same transition, and adds
  a test case for the unflagged-last-transition scenario. No code changed yet
  — this is a plan correction ahead of implementation. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Made the PPO GAE bug-fix plan complete and closed the 100-turn
  loophole.** `Docs/PPO.md`'s "Superseding conclusion" section named the right
  fix (backward episode returns, one bootstrap per boundary) but left it as
  prose with an ambiguous "final unfinished boundary state" phrase that read
  as if a rollout has at most one boundary; it now spells out that any number
  of interior `MAX_STEPS_PER_EPISODE` cutoffs each need their own bootstrap,
  notes `PPO_GAE_LAMBDA` goes inert under this design, and adds concrete
  implementation/test/run steps (none of this is in `ppo_agent.py` yet — it
  still forwards a value for every non-terminal transition today). Also added
  a comment to the "100 learner turns" proposal clarifying it is not a
  substitute fix: under the current code it would only shrink the redundant
  forward-pass count instead of removing it, and it reverses PPO_200's own
  rollout-diversity rationale. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Proposed a short PPO rollout experiment.** `Docs/PPO.md` now records the
  100-learner-turn hybrid-return design: real returns for completed games and
  one bootstrap only for the unfinished tail, plus the signal-quality risks
  and measurements needed to compare it fairly with the 1,024-turn cadence.
  Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Corrected the PPO return/bootstrapping design.** `Docs/PPO.md` now uses
  actual discounted returns for completed games and a single value bootstrap
  only for an unfinished rollout tail; it supersedes the mistaken plan to
  evaluate every transition's next state. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Documented the PPO full-rollout bootstrap OOM and remediation plan.**
  Recorded that `_next_values(...)`, not the 64-sample optimizer minibatch,
  forwards every non-terminal next-state action graph in one expanded CUDA
  batch, and added a chunked-evaluation, regression-test, and GPU-smoke plan.
  Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Reorganized the graph-attention guide after the node-only score matrix.**
  The guide now presents border masking immediately after `S_node`, keeps one
  compact edge-injection projection, and removes the duplicate sparse
  gather/calculation walkthrough. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Corrected the edge-attention matrix shapes in the graph-attention guide.**
  The guide now separates the dense node-only score matrix from the real
  sparse `TransformerConv` path, which gathers 166 edge-aligned rows before
  adding the `[166 x 64]` projected-edge matrix. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Kept the edge-injection section entirely in matrix form.** Removed the
  per-territory score/message notation and retained the `E W_E` projection and
  its role in the attention calculation. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Added the edge-feature projection calculation to the attention guide.**
  The guide now shows `E [166 x 2]` projected by `W_E` to one 64-value
  embedding per directed border and explains its use in the actual
  `TransformerConv` score and message calculations. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Highlighted attack injection in the attention-score explanation.** The
  graph-attention guide now identifies the edge-feature term as the selected
  attack's action injection and explains why it makes attention
  candidate-action-specific. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Resolved the attention-width symbols in the graph-attention guide.** The
  query/key/value section now defines `d_hidden`, `d_att`, and `d_value` as 64
  and shows the corresponding concrete matrix dimensions. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Removed the misleading generic `W_O` notation from the encoder guide.**
  The message-update section now describes the actual `TransformerConv` path:
  64-wide value messages, edge projection, internal `W_skip`, and the
  encoder's outer residual. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Kept bias notation out of the graph-attention walkthrough.** The guide's
  equations now show only the weight-matrix operations; exact bias parameters
  remain listed only in the final learned-parameter table. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Defined `W_skip` next to the encoder update.** The graph-attention guide
  now shows its 64-wide root projection and distinguishes the internal
  `TransformerConv` skip path from the encoder's outer residual connection.
  Files: `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Simplified the encoder-parameter table's weight notation.** The guide now
  lists only weight symbols (`W_in`, `W_Q`, and so on) and gives each learned
  bias its own table column. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Named the encoder's raw-feature mapping in the graph-attention guide.**
  The guide now defines `W_in`/`b_in` in the first matrix equation and uses
  those symbols, rather than an unnamed input-projection label, in the learned
  parameter table. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Added exact encoder parameter shapes to the graph-attention guide.** The
  guide now distinguishes calculated `V` from learned weights and records the
  input, query, key, value, edge, and skip projections used by each of the four
  64-wide `TransformerConv` layers. Files: `Docs/GraphAttentionNetwork.md`,
  `Docs/ChangeLog.md`.

- **Recorded the encoder's configured hidden width in the matrix guide.** The
  guide now states `d_hidden = 64` and resolves the input, residual, pooling,
  and final graph-embedding dimensions accordingly. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Added a standalone graph-attention matrix guide.** The new guide defines
  `H`, border masking, the output projection `W_O`, action injection, pooling,
  and the dimensions of each matrix operation for the Risk encoder. Files:
  `Docs/GraphAttentionNetwork.md`, `Docs/ChangeLog.md`.

- **Added the shared GNN and five-head architecture visual to the poster
  pipeline.** `Docs/Poster.md` now embeds a compact, Markdown-rendered version
  of the graph-input, shared-encoder, and phase-specific MLP-head diagram.
  Files: `Docs/Poster.md`, `Assets/RiskMap/network_phase_heads_poster.png`,
  `Docs/ChangeLog.md`.

- **Made the new poster images render reliably in Markdown preview.** Replaced
  the two HTML image elements with normal Markdown image links to 560 px-wide,
  space-free preview PNGs. Files: `Docs/Poster.md`,
  `Assets/RiskMap/start_ui_poster.png`,
  `Assets/RiskMap/partial_graph_attributes_poster.png`, `Docs/ChangeLog.md`.

- **Added the agent-selection UI and action-injected feature diagram to the
  poster brief.** `Docs/Poster.md` now includes the start screen's player
  roster and the compact node/edge/global attribute visual, both constrained
  to the existing 560 px display width. Files: `Docs/Poster.md`,
  `Assets/RiskMap/start UI.png`, `Assets/RiskMap/partial_graph_attributes.png`,
  `Docs/ChangeLog.md`.

- **Added the completed graph-map figure and constrained embedded image sizes.**
  `Docs/Poster.md` now uses the generated 42-node/83-edge map for Figure 2
  and renders both poster images at 560 px wide, preserving a compact,
  readable Markdown brief. Files: `Docs/Poster.md`,
  `Assets/RiskMap/map_graph_nodes_edges.png`, `Docs/ChangeLog.md`.

- **Embedded the poster's two supplied board assets.** `Docs/Poster.md` now
  displays the populated playable-board screenshot for Figure 1 and the
  neutral map as the source for the graph-overlay Figure 2, so the design
  brief is directly usable during poster assembly. Files: `Docs/Poster.md`,
  `Docs/ChangeLog.md`.

- **Configured the launcher for fresh `DQN_303`.**
  `risk/learning/trainer.py` now creates a non-resuming DQN run under
  `Checkpoints/DQN_303`, ready to be started manually against the corrected
  card-trade and fortify action environment. `Docs/Trainer.md` records the
  active launcher. Files: `risk/learning/trainer.py`, `Docs/Trainer.md`,
  `Docs/ChangeLog.md`.

- **Reduced PPO's GPU minibatch size for `PPO_301`.**
  Set `PPO_MINIBATCH_SIZE` to `64` while retaining the 1,024-turn rollout and
  four epochs. PPO now transfers and backpropagates through at most 64
  decisions' action-graph rows at once, reducing CUDA peak memory without
  changing rollout collection or epoch reuse. Updated `Docs/PPO.md`. Files:
  `risk/learning/train_constants.py`, `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Configured the launcher for a parallel fresh `PPO_301` run.**
  `risk/learning/trainer.py` now builds `PPO_Agent` with `RUN_ID = 301` and
  `resume=False`, writing new checkpoints under `Checkpoints/PPO_301` rather
  than colliding with the existing parallel experiment. `Docs/Trainer.md`
  records the active launcher. Files: `risk/learning/trainer.py`,
  `Docs/Trainer.md`, `Docs/ChangeLog.md`.

- **Checked the now-implemented bucketed-fortify change
  (`Environment._legal_fortify`, `Docs/EnvironmentActionPlan.md`'s second
  section) and fixed two doc-staleness bugs left over from it.**
  `Docs/Action.md`'s `FortifyAction` reference had two contradictory
  "Generated by `legal_actions()`" bullets (a correct new one describing the
  `1/middle/maximum` buckets, and a stale leftover describing the old
  max-only enumeration right below it) plus a mojibake-corrupted en dash
  (`sourceâ€“destination`); merged into one accurate bullet. `Docs/Poster.md`'s
  poster-accuracy checklist item 10 said to keep the fortify description
  separate from reinforcement's `1/half/all` bucketing "unless its legal
  action generator is intentionally changed" — it now has, so updated the
  note to describe both consistently. Verified `_legal_fortify`'s
  `1, (1+maximum)//2, maximum` buckets, `Temp/tests/test_environment.py`'s
  fortify coverage (`test_legal_fortify_offers_one_half_and_maximum_transfer`,
  `test_fortify_accepts_an_unenumerated_valid_transfer`), and the trade-in
  fix together — full suite: 411 passed, 1 skipped. Files: `Docs/Action.md`,
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Implemented bounded fortify amounts for learned agents.**
  `Environment._legal_fortify(...)` now offers each reachable owned pair the
  deduplicated `1 / half / maximum` move amounts plus skip, while preserving
  arbitrary valid direct `FortifyAction` counts for the human UI. Added
  environment coverage and updated the action-space plan, action reference,
  and poster accuracy note. This changes the learner action space, so
  `DQN_300` starts fresh. Files: `risk/game/environment.py`,
  `Temp/tests/test_environment.py`, `Docs/EnvironmentActionPlan.md`,
  `Docs/Action.md`, `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Configured the training launcher for fresh `DQN_300`.**
  `risk/learning/trainer.py` now selects `DQN` with `RUN_ID = 300` and
  `resume=False`, so it creates `Checkpoints/DQN_300` without loading an
  incompatible pre-change checkpoint. `Docs/Trainer.md` now describes the
  active launcher. Files: `risk/learning/trainer.py`, `Docs/Trainer.md`,
  `Docs/ChangeLog.md`.

- **Implemented the deferred-trade-in card rule fix from
  `Docs/EnvironmentActionPlan.md`.** `Environment._apply_attack` no longer
  forces `Phase.TRADE_IN` for an ordinary conquest-card draw that brings the
  hand to `MAX_CARDS_IN_HAND` — only an eliminated defender's transferred
  cards still force an immediate mid-attack trade-in; the ordinary case now
  finishes attack/occupy/fortify normally and is forced to trade at the
  start of the player's *next* turn instead (`TRADE_IN` already starts every
  turn). Also fixed a real, independently-reachable bug in the *existing*
  mid-attack forced-trade path: `_apply_trade_in`'s `auto_skipped` branch and
  `_apply_skip_trade` used to jump straight from `TRADE_IN` to `OCCUPY`,
  never giving the accumulated trade-in value a `REINFORCE_PLACE` step —
  that budget was silently discarded when `_begin_turn_for` overwrote
  `reinforcement_budget` at the start of the player's next turn. Both paths
  now route through `REINFORCE_PLACE` first; `_apply_reinforce`'s completion
  now checks `pending_attack` to resume `OCCUPY` (without resetting
  `conquered_this_turn`) instead of always advancing to a fresh `ATTACK`
  phase.
  **Consequential, breaking change:** the fix means a player's own hand can
  now legitimately sit at the full `MAX_CARDS_IN_HAND` (not just
  `MAX_CARDS_IN_HAND - 1`) between an ordinary conquest and their next turn
  — including while sitting as someone else's defender. Combined with an
  eliminated defender's own hand having the same property, the true worst-
  case mid-attack transient hand grows from 9 to 10 cards, so
  `MAX_TRANSIENT_HAND_SIZE` in `risk/constants.py` changed from
  `2 * (MAX_CARDS_IN_HAND - 1) + 1` to `2 * MAX_CARDS_IN_HAND`. This resizes
  `TradeInHead`'s card-slot embedding table
  (`risk/learning/heads.py`) — **any existing DQN/Dueling DQN/PQN/ADQN
  checkpoint using `TradeInHead` will fail `load_state_dict` with a shape
  mismatch** (a loud failure, not silent corruption) and needs a fresh
  training run, same as any other environment/action-space change (see the
  "Experiment rule" pattern in `Docs/EnvironmentActionPlan.md`'s other
  section).
  Updated `Docs/Action.md`'s `TradeInAction`/`AttackAction` sections and
  `risk/constants.py`'s `MAX_TRANSIENT_HAND_SIZE` comment to match, marked
  `Docs/EnvironmentActionPlan.md`'s first section "implemented", and added
  `Temp/tests/test_environment.py` coverage: an ordinary four-to-five
  conquest that defers to the next turn, the corrected mid-attack elimination
  path's `REINFORCE_PLACE` step, a nine-card elimination needing two
  consecutive trade-ins summed into one placement, and the existing +2
  matching-territory-card bonus (previously untested). Full suite: 409
  passed, 1 skipped. Files: `risk/game/environment.py`, `risk/constants.py`,
  `Docs/Action.md`, `Docs/EnvironmentActionPlan.md`,
  `Temp/tests/test_environment.py`, `Docs/ChangeLog.md`.

- **Planned the five-card conquest rule correction without changing code.**
  `Docs/EnvironmentActionPlan.md` now distinguishes a normal four-to-five-card
  conquest (finish the current turn; forced trade-in at the next turn start)
  from an elimination card transfer (force trade-in before the parked occupy
  action resumes). The plan now also requires a dedicated placement step for
  the elimination trade-in set value before occupation resumes, preventing it
  from being overwritten and lost, and explicitly requires repeated trade-ins
  until fewer than five cards remain (including a nine-card regression case).
  Files:
  `Docs/EnvironmentActionPlan.md`, `Docs/ChangeLog.md`.

- **Clarified the learned-seat controls in the New Game screen.** Replaced the
  clipped `Learned Agent` type label with `AI Agent`, renamed the ambiguous
  `Best...` button to `Best DQN`, and documented that each click cycles to the
  next of the five evaluated DQN 103 presets and shows its label. Files:
  `risk/ui/render/init_screen_view.py`, `README.md`,
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Closed the learned-agent play test-coverage gap called out in the
  previous review.** Added real coverage to `Temp/tests/test_learned_agent_play.py`:
  policy-only `.pt` and `epNNNNNN` round-trips through `choose_agent.py`'s
  reused private helpers (built from a real `GNN_DQN_Agent` fixture, not a
  mock), real-seat attachment (`epsilon=0.0`, `train_mode=False`,
  `player_id`/`env` rebinding, deterministic repeated inference),
  `validate_selections` accept/missing-checkpoint/agent-kind-mismatch cases,
  independent agent instances for two seats sharing one checkpoint, and a
  mixed human/heuristic/learned `AppLoop` smoke run driven through the real
  loop under `SDL_VIDEODRIVER=dummy`. The previous `callable(...)`-only
  "coupling" check is replaced by tests that actually call the private
  helpers and would fail on a signature/behavior-breaking refactor. Updated
  `Docs/Testing.md`'s row accordingly (the app-loop coverage lives in this
  file, not `test_game_loop.py`, since `Game`/`Game.tick()` aren't on the
  interactive path). Full suite: 406 passed, 1 skipped. Files:
  `Temp/tests/test_learned_agent_play.py`, `Docs/Testing.md`,
  `Docs/ChangeLog.md`.

- **Fixed a stale-validation bug and cleaned up the learned-agent setup-screen
  implementation.** `refresh_learned_validation()` was only called from
  kind/model-change handlers, not from the `-`/`+` player-count buttons in
  `risk/ui/render/init_screen_view.py`; reducing player count past an erroring
  learned seat left Start Game blocked on a stale message about a seat that no
  longer existed. Also replaced inline `__import__("pathlib")` calls with a
  normal top-level `from pathlib import Path` import, and stopped the
  algo-kind cycle button from being clickable on a preset-sourced selection
  (a preset already supplies its own kind; cycling it desynced the checkpoint
  from the claimed architecture until validation caught it later). Verified
  the adapter end-to-end against a real `Checkpoints/DQN_103/ep006700`
  preset (load, validate, attach, `epsilon=0.0`, `train_mode=False`, act) and
  ran the full suite (396 passed, 1 skipped). Files:
  `risk/ui/render/init_screen_view.py`, `Docs/ChangeLog.md`.

- **Corrected predefined learned policies to use evaluation evidence.**
  Replaced trainer-score DQN/Dueling/PPO manifests with DQN 103's five highest
  evaluated checkpoint directories (episodes 6700, 6000, 5400, 5650, and
  5600); Dueling DQN and PPO remain manual-only until evaluated. Files:
  `Params/play_agents.json`, `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Implemented interactive learned-policy play setup.** Added UI-only learned
  seats with manual file/folder selection and preset cycling, a policy-only
  adapter that reuses the existing loader helpers, actual app-time agent
  replacement, optional sidebar model labels, and focused setup/registry tests
  without changing game-domain or training source files. Files:
  `risk/app/learned_agent_play.py`, `risk/app/main.py`, `risk/app/loop.py`,
  `risk/app/view.py`, `risk/ui/input/init_screen.py`,
  `risk/ui/render/init_screen_view.py`, `risk/ui/render/panels.py`,
  `Params/play_agents.json`, `Temp/tests/test_learned_agent_play.py`, `Temp/tests/test_ui.py`,
  `README.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Preserved direct `GameView` compatibility in the learned-agent plan.** The
  UI label mapping must be an optional empty-default argument through the
  loop/view/panel layers because `risk.learning.self_play.py` already builds a
  `GameView` directly and remains out of scope. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Specified the UI-only learned-seat type cycle.** The plan now records the
  explicit visible cycle, the underlying `ai` placeholder, and clearing the
  learned selection when a seat switches back to an ordinary type or is
  removed by a player-count reduction; the current `AGENT_KIND_ORDER` cannot
  supply this behavior by itself. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Fixed a scope-violating break in the learned-agent play plan's
  `SetupStage.default_settings()` change.** Step 3 had it returning the new
  `InteractiveSetupResult` wrapper for skip-menu/auto-restart/max-ticks paths,
  but `default_settings()` is called directly as
  `GameFactory.build(SetupStage.default_settings(...))` from
  `risk/learning/self_play.py`, `risk/learning/trainer.py`, and several tests
  — files the plan's scope boundary forbids editing. `default_settings()` now
  keeps returning a bare `GameSettings` unchanged; `main.py`'s `run()` wraps
  it locally instead. Files: `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Consolidated learned-agent play into one implementation-ready plan.**
  Replaced the separate review-evidence section with a complete sequence that
  incorporates import safety, lazy validation, setup-result flow, policy-only
  loading, UI labels, multi-seat behavior, tests, and unchanged game/training
  boundaries. Files: `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Flagged a lazy-import requirement in the learned-agent play plan.**
  `risk/app/factory.py` has no `torch`/`risk/learning/` dependency today, and
  `risk/app/setup.py` already defers its `init_screen_view` import to avoid
  loading pygame for headless setup paths. If the new adapter (which reuses
  `choose_agent.py`, which imports `torch`) were imported at module top of
  `risk/ui/render/init_screen_view.py`, every interactive launch — including
  all-human games — would newly pay a `torch` import cost. Step 1 now
  requires a function-local import instead. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Closed the learned-agent setup-result and pre-start-validation gaps.** The
  plan now returns UI-only selections/labels through the setup-to-main path and
  validates them once in a short-lived, unchanged factory context before Start
  Game is enabled. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Corrected the learned-agent play plan's fix for its own circular-import
  risk after reading `risk/app/setup.py`.** The prior fix routed the new
  checkpoint-load-validation adapter through `SetupStage`, but `SetupStage`
  is a one-line static dispatcher with no event loop — it can't gate a live
  "Start Game" button. That button, its `enabled=ok` flag, and the whole
  pygame event loop actually live in `run_init_screen(...)` in
  `risk/ui/render/init_screen_view.py`, a file `choose_agent.py` never
  imports, so it's both the functionally correct call site and still
  cycle-safe. Files: `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Corrected the learned-agent play plan's policy-only `.pt` wording.** The
  reused `choose_agent.py` helper reads raw network state for both artifact
  shapes; it does not call each agent's `load_params()` method. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Re-reviewed the learned-agent play plan after it merged in the earlier
  review notes, and caught a real circular-import risk left by the "reuse
  `choose_agent.py`'s private helpers" decision.** `risk/learning/
  choose_agent.py` already imports `InitScreenState` from
  `risk/ui/input/init_screen.py` (to build evaluation rosters) and
  `GameFactory` from `risk/app/factory.py`; if the new adapter that imports
  `choose_agent.py`'s `_read_policy_state`/`_new_learned_agent` were itself
  reachable from `init_screen.py`, the import graph would close a genuine
  cycle. Moved the adapter and its checkpoint-load validation into
  `risk/app/setup.py` (`SetupStage`, which already imports `InitScreenState`
  and sits above both layers) and restricted `InitScreenState` to
  syntactic-only checks. Also fixed a self-contradictory paragraph left over
  from the previous merge (review item 4 said both "resolved by reuse" and
  "must reimplement" in the same paragraph) and added a coupling-test
  mitigation for depending on `choose_agent.py`'s private functions. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Changed the learned-agent play plan to reuse existing policy-only
  helpers.** The UI/app adapter imports `choose_agent.py`'s established
  loader/constructor helpers without modifying them, rather than duplicating
  saved-policy handling. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Applied the learned-agent play review findings to the implementation
  sequence.** It now specifies UI-only sidebar labels, app-list replacement
  timing, shared-`model.pt` net extraction, explicit class/evaluation-mode
  handling, preset portability, dedicated tests, and the README Killbot fix.
  Files: `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Added a review-notes section to the learned-agent play plan after
  fact-checking it against the current code.** Flags a real gap (the
  in-game sidebar can't show a model label while `Player.agent_kind` stays
  `"ai"`, since `panels.py` reads labels straight from `AGENT_KIND_LABELS`),
  clarifies that only `ctx.agents` — not `ctx.game.agents` — is load-bearing
  for the interactive `AppLoop` path, corrects the checkpoint-file-layout
  description (one shared `model.pt` with a `"net"` key, not a separate
  file), notes the deliberate duplication of `choose_agent.py`'s private
  net-only-loading helpers, and suggests a dedicated test file over folding
  everything into `test_ui.py`/`test_game_loop.py`. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Constrained the learned-agent play plan to the UI/app layer.** Learned
  choices stay outside `GameSettings`; the app swaps validated policies into
  existing `ai` placeholders before play, leaving game rules, the factory, and
  all training files unchanged. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Split planned learned-policy selection into manual and predefined paths.**
  Each seat will either browse for a local checkpoint or select a curated best
  model; `Params/play_agents.json` now records only the maintained predefined
  best-model registry. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Made the planned interactive learned-policy chooser file-picker first.**
  Each learned seat will select a local checkpoint through a native dialog;
  `Params/play_agents.json` is now optional recent-selection persistence, not
  required configuration. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Narrowed the planned interactive learned-policy selector to DQN, Dueling
  DQN, and PPO.** Deferred PQN and ADQN until a later explicit extension of
  the play-mode plan. Files: `Docs/PlayLearnedAgents.md`,
  `Docs/ChangeLog.md`.

- **Added the interactive learned-agent play plan.** Documented the New Game
  selector, supported DQN/PPO/Dueling saved-policy kinds, the planned
  `Params/play_agents.json` catalog, deterministic policy-only loading, and
  validation/test criteria without changing application code. Files:
  `Docs/PlayLearnedAgents.md`, `Docs/ChangeLog.md`.

- **Replaced the stale completed combined-update plan with a current action-space
  plan.** Removed `Docs/Update_Plan.md`, whose reward/history/injection work is
  implemented and recorded in the reference docs and change log. Added
  `Docs/EnvironmentActionPlan.md` with the proposed bounded `1 / half / max`
  fortify candidates, test/measurement requirements, and fresh-run rule.
  Retargeted `Docs/Reward.md`'s historical-plan reference. Files:
  `Docs/Update_Plan.md`, `Docs/EnvironmentActionPlan.md`, `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Added an A0 scientific-poster design brief for the GNN-RL Risk project.**
  It specifies the story, draft poster text, graph/action-injection figures,
  network explanation, W&B-result figure choices, evidence limits, and
  production checklist without creating the poster artwork yet. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Expanded the poster brief with reward and opponent explanations.** Added
  poster-ready descriptions of bounded dense reward shaping, the terminal win/
  loss signal, the varied heuristic-opponent roster, and the planned controlled
  Dueling DQN rerun. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Added poster attribution details.** Recorded Gilad Markman as author and
  the 2026 Practical Deep Learning for Science course team: Prof. Eilam Gross,
  Dmitrii Kobylianskii, Alon Levi, and Etienne Dreyer. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Clarified the poster's core network message.** The title-area summary now
  names the shared graph-attention encoder and separate action-phase heads.
  Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Added the game environment as a poster foundation.** The brief now explains
  the state/legal-action/transition/reward agent-environment loop and reserves
  a compact figure for it. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Connected heuristic opponents to the poster's environment section.** The
  environment narrative now identifies the varied rule-based opponents as
  agents using the same legal-action and transition interface. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Documented quantitative-action discretisation for the poster.** The brief
  now explains reinforcement's `1 / half / all` army buckets, why multi-step
  placement retains flexibility, and that current fortification uses a
  different max-transfer-plus-skip enumeration. Files: `Docs/Poster.md`,
  `Docs/ChangeLog.md`.

- **Added the random baseline agent to the poster environment section.** Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Added a compact Risk-rules panel to the poster brief.** It introduces the
  board, turn phases, cards, continent bonuses, and win objective for readers
  unfamiliar with the game. Files: `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Prioritized attack and conquest rules in the poster game explanation.**
  The brief now explains the attacker/defender dice comparison and occupation
  after eliminating defenders instead of emphasizing cards. Files:
  `Docs/Poster.md`, `Docs/ChangeLog.md`.

- **Configured fresh PPO_202 as the midpoint learning-rate experiment.** Set
  PPO's default `PPO_LR` to `7.5e-5` and switched the PPO launcher to
  `RUN_ID = 202` with `resume=False`, producing a new `PPO_202` W&B run and
  checkpoint namespace. All PPO_200 settings and the shared DQN_105 reward
  remain unchanged. Files: `risk/learning/train_constants.py`,
  `risk/learning/trainer.py`, `Docs/PPO.md`, `Docs/Trainer.md`,
  `Docs/ChangeLog.md`.

## 2026-08-10

- **Configured fresh PPO_201 to test lower PPO learning rate only.** Set the
  PPO-only default `PPO_LR` to `5e-5` and switched the launcher to run id 201
  with its existing fresh W&B configuration. PPO_201 retains PPO_200's shared
  DQN_105 reward and all other PPO settings; it tests whether reduced KL
  saturation permits more useful epochs per rollout. Files:
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Docs/PPO.md`, `Docs/Trainer.md`, `Docs/ChangeLog.md`.

- **Clarified long-line formatting in both repository instruction files.**
  Compact expressions may remain on one line; only genuinely long signatures
  and calls should be split into two or three readable rows. Files:
  `AGENTS.md`, `CLAUDE.md`, `Docs/ChangeLog.md`.

- **Added editor-foldable regions to PPO minibatch diagnostics and metrics.**
  Collapsing these blocks leaves the PPO loss/KL/optimizer path visible; no
  training or logging behavior changed. Files: `risk/learning/ppo_agent.py`,
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Removed PPO's unused Python random generator and unused typing import.**
  The `seed` constructor parameter remains for caller compatibility; PPO
  behavior is unchanged because it never used that generator. Files:
  `risk/learning/ppo_agent.py`, `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Replaced PPO's three read-only progress properties with ordinary public
  counters.** `train_steps`, `optimizer_steps`, and `samples_processed` are
  now initialized and updated directly, while preserving their checkpoint and
  metric behavior. Files: `risk/learning/ppo_agent.py`, `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Added a one-sentence responsibility docstring to every PPO agent method.**
  This is a readability-only change: PPO behavior, metrics, and update flow
  are unchanged. Files: `risk/learning/ppo_agent.py`, `Docs/PPO.md`,
  `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Cleaned up leftover debris from the `PPO_Agent.learn()` readability
  refactor, no behavior change.** Removed `_mean_tensor`/`_weighted_tensor_mean`,
  two helpers left unused after the refactor moved their call sites into
  `_summarize_update`'s `_mean_minibatch_metric`/`_sample_weighted_minibatch_metric`.
  Added the type hints `_prepare_rollout_update`/`_run_update_epochs` were
  missing relative to every other method in the file. Removed four redundant
  `list(indices)` conversions in `_run_minibatch` by converting once (`idx =
  list(indices)`) and reusing it, including in the two counters that still
  read `len(indices)`. No metric, loss, or control-flow change. Files:
  `risk/learning/ppo_agent.py`, `Docs/ChangeLog.md`. Validation: focused
  `test_ppo.py` (11 passed) and full suite (390 passed, 1 skipped).

- **Further shortened PPO's update coordinator without changing behavior or
  the logging contract.** `PPO_Agent.learn()` now contains only the rollout
  gate, one named preparation step, epoch execution, and final summary.
  `_prepare_rollout_update(...)` owns fixed-target/cache construction and
  `_run_update_epochs(...)` owns the KL-guarded loop, while the existing
  minibatch and summary helpers retain the PPO math and metrics. KL early-stop
  timing, loss equations, gradient clipping, counters, and metric names are
  unchanged. Updated the PPO/testing documentation. Files:
  `risk/learning/ppo_agent.py`, `Docs/PPO.md`, `Docs/Testing.md`,
  `Docs/ChangeLog.md`. Validation: focused `test_ppo.py` passed.

## 2026-08-09

- **Promoted PPO_200 from a completed local smoke run to a fresh W&B run.**
  The smoke reached episode 250 with 27 PPO updates, 426 optimizer steps, and
  109,056 processed samples. Preserved its checkpoints under
  `Checkpoints/PPO_200_smoke_ep000250`, then changed the launcher to enable
  W&B with `resume=False` and no cloud run id, ensuring PPO_200 starts with an
  empty checkpoint namespace and a fresh W&B history. Files:
  `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Configured the trainer entry point for PPO_200's local smoke run.**
  `trainer.py` now builds `PPO_Agent` with run id 200, an empty fresh start,
  and W&B disabled; it leaves the DQN_105 resume id and W&B id as comments for
  later restoration. Updated the trainer and PPO launch documentation. Files:
  `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Implemented PPO_200's PPO-only optimization and diagnostic changes without
  touching the active DQN_105 launcher.** Set PPO rollout/minibatch sizes to
  `1024`/`256` and value-loss coefficient to `0.1`; entropy regularization
  remains enabled at `0.01`. `PPO_Agent` now measures policy and weighted-value
  encoder gradients on every executed minibatch, reports their sample-weighted
  update means, and logs a finite
  `ppo_value_to_policy_encoder_grad_ratio`. Added focused PPO coverage for the
  changed loss coefficient and normal/zero-policy-gradient ratio behavior.
  `trainer.py`, reward constants, and non-PPO learners are unchanged. Files:
  `risk/learning/train_constants.py`, `risk/learning/ppo_agent.py`,
  `Temp/tests/test_ppo.py`, `Docs/PPO.md`, `Docs/Testing.md`,
  `Docs/ChangeLog.md`.

- **Completed a final PPO_200 plan-readiness pass, doc only.** Repaired the
  inline `PPO_Agent.unweighted_update_metrics` reference, removed a stale
  reference to the old `0.1` reward regime, and made matched-turn DQN_105 the
  primary comparison target, retaining DQN_103 only as a historical fallback
  until DQN_105 reaches the budget. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Closed four implementation-readiness gaps in the PPO_200 plan, doc only.**
  A re-read of `Docs/PPO.md` after the previous revision found: the new
  `ppo_value_to_policy_encoder_grad_ratio` metric was never added to
  `PPO_Agent.unweighted_update_metrics`, which would have made `Trainer`
  weight it like a raw per-minibatch loss instead of averaging it with equal
  per-update weight like its two input norms; "Trainer notes for PPO" still
  described a "third commented block next to DQN/Dueling" in `main()` that
  the newer PPO_200 wiring comment already contradicts; no test was named for
  the new ratio's `1e-12` epsilon-floor behavior; and the "up to 32
  `autograd.grad` calls" figure was stated as fact rather than specific to
  today's `PPO_ROLLOUT_LENGTH`/`PPO_MINIBATCH_SIZE`/`PPO_EPOCHS`. All four are
  now fixed in the doc: implementation step 2 requires adding the ratio to
  `unweighted_update_metrics`; "Trainer notes for PPO" is marked superseded
  and points at the PPO_200 wiring comment; a new test 15 (and a matching note
  in step 6) requires exercising the epsilon floor; and the call-count
  sentence is qualified as specific to the current constants. No constants,
  agent code, or trainer wiring changed. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Tightened the PPO_200 diagnostic plan, doc only.** The planned
  update-wide encoder-gradient ratio now has an explicit metric name and a
  finite zero-policy-gradient guard, and the plan corrects the full-update
  diagnostic cost to at most 32 `autograd.grad` calls (two per executed
  minibatch), rather than an ambiguous count of 16. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

- **Renamed the planned PPO restart from `PPO_104` to `PPO_200` and added
  reviewer comments to the plan, doc only, no code.** `Docs/PPO.md` now uses a
  fresh `PPO_200` numbering block (separate from the shared DQN/Dueling
  run-id sequence and from the legacy `PPO_041-045` runs) throughout its
  restart section. Added inline `**Comment:**` notes: launching `PPO_200` in
  `trainer.py`'s `main()` replaces `DQN_105`'s current resume configuration
  rather than running alongside it (keep `DQN_105`'s `RUN_ID`/`wandb_run_id`
  as a comment when swapping it out); `train_constants.py` still holds the
  PPO_045 values today, only three constants actually need to change;
  aggregating the actor/critic encoder-gradient diagnostic across every
  minibatch (weighted by minibatch size) instead of only the update's
  first is a real, bounded compute-cost increase and a correctness fix to
  the measurement, not just more logging; a fivefold `PPO_VALUE_LOSS_COEF`
  cut needs explained-variance/value-RMSE monitoring alongside the
  gradient-ratio target to catch an undertrained critic; and episode-counted
  checkpoint/eval cadence should be sanity-checked against the 4x larger
  rollout during the smoke run. No constants, agent code, or trainer wiring
  changed. Files: `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Kept the planned PPO_104 restart on DQN_105's current shared reward
  regime.** The PPO plan no longer proposes changing global reward constants:
  it retains shaping `0.3` and terminal `+300/-300` while DQN_105 remains the
  promising active run. PPO_104 now limits planned changes to PPO rollout and
  optimization settings, update-wide encoder-gradient diagnostics, and its
  fresh launcher; it will not confound PPO tuning with a reward change. Files:
  `Docs/PPO.md`, `Docs/ChangeLog.md`.

- **Corrected and tightened the model-selection plan after the five DQN_103
  best-policy evaluations were appended.** `Docs/ModelSelection.md` now uses
  the current 60-policy/1,080-six-player-game counts, corrects the seat test
  to p=0.126, records sentinel/empire's 60.2% share of learner losses, and
  makes Phase 1 use fresh seed-compatible result files rather than trying to
  append to metadata-incompatible JSON. It also prevents Phase 4 from
  aggregating raw win rates across unequal tournament rosters. Files:
  `Docs/ModelSelection.md`, `Docs/ChangeLog.md`.

- **Added a model-selection plan (`Docs/ModelSelection.md`), doc only, no
  code.** Written after analyzing the DQN_103 checkpoint-eval JSON directly:
  the top 3 non-leading 6-player checkpoints are tied at exactly 55.6%
  (18 games/checkpoint is too few to separate them; seat position is not
  significant, p=0.53, but opponent identity is, p<0.0001 — sentinel/empire
  cause 57% of losses despite being 2 of 5 opponents). The plan notes
  `AgentMatchEvaluator` (`risk/learning/choose_agent.py`) already supports
  learner-vs-learner and cross-run/cross-architecture matches today — no new
  evaluator code needed — and proposes a phased approach: more seeds to
  break the DQN_103 tie, a self-tournament among top DQN_103 checkpoints, an
  anchored tournament (candidates + sentinel/empire heuristics), and a
  stretch cross-run/cross-architecture bracket using each run's
  `best/manifest.json`. Flags legacy `run_0XX` checkpoint dirs as
  unconfirmed `agent_kind` and proposes three small new read-only tools (a
  match report/plot, a cross-match leaderboard aggregator, a candidate-list
  builder) rather than changing `CheckpointEvaluator`/`AgentMatchEvaluator`.
  Files: `Docs/ModelSelection.md`, `Docs/ChangeLog.md`.

- **Made saved-evaluation JSON persistence resilient to transient Windows file
  locks.** Atomic replacement now retries a brief lock held by Dropbox,
  antivirus, or an editor before surfacing a persistent failure, so checkpoint
  evaluation can continue after a momentary `WinError 5`. Added focused
  coverage and documented the bounded retry. Files:
  `risk/learning/choose_agent.py`, `Temp/tests/test_choose_agent.py`,
  `Docs/ChooseAgent.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`.

- **Configured the launcher to resume DQN_105 after episode 400.** Enabled
  local checkpoint restoration and pinned W&B run `5b66yunb`, so restarting
  continues the saved model, optimizer, replay buffer, counters, and existing
  cloud history instead of starting run 105 from scratch. Files:
  `risk/learning/trainer.py`, `Docs/Trainer.md`, `Docs/ChangeLog.md`.

## 2026-08-08

- **Reduced DQN_105's joint reward magnitude to the intermediate `0.3/300`
  setting.** Changed the global dense-shaping coefficient from `0.5` to `0.3`
  and terminal win/loss from `+500/-500` to `+300/-300`. Their relative
  balance remains unchanged while absolute replay targets and gradient pressure
  are reduced by 40%. The marginal reinforcement formula and run ID 105 are
  unchanged; updated its exact-value test and recalculated the documented
  expected actual reinforcement ranges. Files:
  `risk/learning/train_constants.py`, `Temp/tests/test_reward.py`,
  `Docs/Reward.md`, `Docs/Trainer.md`, `Docs/ChangeLog.md`. Validation:
  `377 passed, 1 skipped` in the full test suite.

- **Implemented marginal reinforcement shaping for fresh DQN_105.** Ready and
  total-frontier rewards now use zero-floored before/after potential
  differences, so equivalent partial placements telescope instead of paying
  repeatedly for strength already present. Continent, interior, split, all
  non-reinforcement rewards, global scale `0.5`, and terminal `+500/-500`
  remain unchanged. Added reinforcement/partial-action counts, per-action W&B
  diagnostics, marginal threshold/cap/boundary and split-invariance tests, and
  advanced the non-resuming launcher from DQN_104 to DQN_105. Converted the
  completed plan into current reward documentation. Files:
  `risk/learning/reward.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_reward.py`, `Temp/tests/test_trainer.py`, `Docs/Reward.md`,
  `Docs/Trainer.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`. Validation:
  `377 passed, 1 skipped` in the full test suite.

- **Floored the planned `ready_score` at zero, matching `total_score`.**
  `Docs/Reward.md`'s marginal reinforcement correction plan now wraps
  `ready_score` in the same `max(0, ...)` used by `total_score`. Without it,
  the `- REWARD_REINFORCE_READY_RATIO` offset cancels out of the before/after
  subtraction, so any placement that raises armies on a destination earns a
  small positive `reinforce_ready` no matter how far below 1.5:1 the ratio
  stays. The floor removes that leftover reward for progress that never
  reaches readiness without reintroducing a penalty or breaking the
  telescoping-sum property split sequences rely on. Added a matching test
  requirement and corrected the resulting maximum sequence bounds to `2.75`
  for ready improvement and `5.25` for ready plus total. Still unimplemented;
  no code or reward values changed. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Documented DQN_104's repeated reinforcement-reward finding and correction
  plan.** `Docs/Reward.md` now explains that W&B components are raw episode
  totals, records episode 311's ready/total-dominated breakdown, and identifies
  repeated full-stack scoring across partial reinforcement actions. Recast the
  completed reinforcement section as current behavior, removed its obsolete
  implementation checklist and superseded drafts, and added an unimplemented
  plan to change ready/total shaping to before/after potential differences with
  per-action diagnostics and split-invariance tests. Added the last-50-episode
  component comparison, mathematical corrected-sequence bounds, and explicit
  expected/acceptance ranges for the next run. No code or reward values changed.
  Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

## 2026-08-07

- **Prepared fresh DQN_104 with jointly scaled rewards.** Changed dense
  shaping from `0.1` to `0.5` and terminal win/loss from `+100/-100` to
  `+500/-500`, preserving their relative ratio while increasing reward and
  TD-target magnitudes fivefold. Advanced the fresh classic-DQN launcher from
  run 103 to 104 with `resume=False`; the reinforcement formulas themselves
  are unchanged. Updated the PPO restart plan to distinguish its planned
  restored `0.1`/`+100/-100` scale from DQN_104. Files:
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Temp/tests/test_reward.py`, `Docs/Reward.md`, `Docs/Trainer.md`,
  `Docs/PPO.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`. Validation:
  `371 passed, 1 skipped` in the full test suite.

- **Closed reinforcement reward test gaps and removed dead helper.** Added
  the four cases the reinforcement-revision test checklist called for but the
  initial implementation pass missed: continent term is zero on a fully owned
  continent, zero when the readiness gate hasn't been met, strictly larger
  for a smaller contested continent than a larger one under matching
  per-territory setup (South America vs. Asia), and the case where the
  learner owns only one territory in a contested continent. Factored the
  continent-component formula recomputation shared by these tests into
  `_expected_continent_component` and generalized `_frontier_reinforcement`
  to accept a custom target/enemy territory instead of hardcoding
  Afghanistan/China/India. Also deleted `RewardCalculator
  ._weakest_adjacent_enemy_armies`, left unused after `_reinforce` was
  rewritten to compute weakest/sum enemy armies inline in one pass. Files:
  `risk/learning/reward.py`, `Temp/tests/test_reward.py`. Validation:
  `370 passed, 1 skipped` in the full suite.

- **Implemented the reinforcement-only reward revision.** Reinforcement now
  uses weak-neighbour readiness, whole-frontier strength, gated contested-
  continent priority, interior placement, and one fixed partial-action split
  penalty. Added separate raw diagnostic components and focused formula,
  boundary, cap, split, and aggregation tests. No other action reward was
  changed. Files: `risk/learning/train_constants.py`,
  `risk/learning/reward.py`, `Temp/tests/test_reward.py`, `Docs/Reward.md`,
  `Docs/Trainer.md`, `Docs/Testing.md`, `Docs/ChangeLog.md`. Validation:
  `366 passed, 1 skipped` in the full test suite.

- **Halved the planned reinforcement continent scale for the 1K pilot.**
  `Docs/Reward.md` now starts `REWARD_REINFORCE_CONTINENT_SCALE` at `5.00`
  instead of `10.00`. Documented its board-specific upper bound below `2.19`,
  the combined reinforcement bound below about `7.44`, and the requirement to
  inspect the logged component distribution before considering an increase.
  This remains unimplemented. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Clarified the reinforcement plan's split semantics and diagnostics.**
  `Docs/Reward.md` now states that the fixed `-0.20` partial-action cost does
  not guarantee split-invariant cumulative shaping, scopes split tests to the
  exact per-action formula, and requires separate ready/total/continent/
  interior/split W&B components whose raw sum equals the combined
  reinforcement reward. Reward logic and constants remain unchanged; no code
  is implemented. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Shortened the planned reinforcement constant names.** `Docs/Reward.md`
  now uses compact `READY`, `TOTAL`, `CONTINENT`, `INTERIOR`, and `SPLIT`
  names while retaining the common `REWARD_REINFORCE_` namespace. The new
  `READY_CAP` name remains distinct from the existing, differently defined
  `REWARD_REINFORCE_RATIO_CAP`. Formulas, values, and reward behavior are
  unchanged. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Named and explained every reinforcement-plan tuning value.**
  `Docs/Reward.md` now defines the `1.50` weak-neighbour readiness threshold
  and `2.00` whole-frontier threshold as named constants, moves all planned
  reinforcement constants ahead of the formula table, explains the policy
  meaning of each value, and removes numeric literals from the reward and
  shaping-cap formulas. Reward logic and initial values are unchanged. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Rewrote the planned reinforcement revision as one implementation
  reference without changing its reward logic.** `Docs/Reward.md` now defines
  every input once, gives each reward's complete condition and formula in one
  table, states the exact aggregation/scaling order, and consolidates the
  implementation and test checklist. Removed the worked calibration examples
  and updated the fresh-run instruction to the post-DQN_103 1K pilot. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Simplified the planned reinforcement split penalty to one constant.**
  `Docs/Reward.md` now applies a single raw `-0.20` penalty once to any
  `ReinforcementAction` that places less than the visible remaining budget,
  replacing the proposed `-0.05 x unused_armies` formula. Updated the named
  constant, multi-destination scope, required boundary tests, and worked
  examples; all other reinforcement-plan terms remain unchanged. This remains
  unimplemented. Files: `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Planned a controlled PPO restart under the current DQN_103 task.** Added
  the unimplemented `PPO_104` plan: current 15-column action representation,
  shaping scale `0.1` with terminal `+100/-100`, rollout 1024, minibatch 256,
  and value-loss coefficient `0.1`. The plan records PPO_041--045 evidence,
  targets their measured critic-over-actor gradient dominance, defines
  learner-turn review gates and diagnostic reactions, requires fresh
  namespaces and robust evaluation, and separates a possible DQN-assisted PPO
  fallback from the from-scratch comparison. Files: `Docs/PPO.md`,
  `Docs/ChangeLog.md`.

## 2026-08-02

- **Review-fixed the "simple draft" reinforcement plan and trimmed superseded
  drafts.** `Docs/Reward.md`: fixed an arithmetic error in the worked-example
  table ("dominates the whole local frontier" row: weak term was `+1.250`,
  should be `+1.750`, so the raw/replay total is `+2.375 -> +0.238`, not
  `+1.875 -> +0.188`); renamed the plan's `REWARD_REINFORCE_RATIO_CAP` to
  `REWARD_REINFORCE_FRONTIER_RATIO_CAP` since it reused the name of an
  existing, differently-scaled implemented constant; removed a duplicate
  plain-English table that repeated the formula table right above it; added
  an explicit note that `FortifyAction`'s continent-push is untouched by this
  plan; clarified that the planned favorable-attack stop penalty replaces the
  already-*implemented* unfinished-target mechanism, not a past plan; and
  collapsed the two fully-superseded reinforcement drafts (detailed-draft and
  launch-value) into short pointers, per this file's own stated policy of not
  keeping full superseded formulas here. Still entirely unimplemented. Files:
  `Docs/Reward.md`, `Docs/ChangeLog.md`.

- **Set the planned reinforcement ratio ceiling to 7:1.** `Docs/Reward.md`
  now caps both readiness ratios at 7:1, leaving room to prepare for a
  stronger next target after attacking a weak neighbour while still ending
  reinforcement reward growth for overwhelming stacks. Worked examples and
  formulas now use the 7:1 ceiling; the continent term remains gated on 1.5:1
  direct-neighbour readiness. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

- **Condensed the planned reinforcement reward into an implementation policy.**
  `Docs/Reward.md` now gives one table that ties each planned term to the
  intended behavior: frontier readiness, local-frontier strength,
  contested-continent focus, interior penalty, and per-decision split penalty.
  It explicitly records that the split penalty is zero only when the entire
  remaining budget is placed at once. Files: `Docs/Reward.md`,
  `Docs/ChangeLog.md`.

## 2026-08-01

- **Prepared a fresh low-exploration DQN rerun.** Set the DQN-family epsilon
  floor to `0.01` (from `0.10`) and advanced the fresh launcher to `DQN_103`,
  preserving `DQN_102`'s checkpoints and W&B namespace for comparison. Files:
  `risk/learning/train_constants.py`, `risk/learning/trainer.py`,
  `Docs/Trainer.md`, `Docs/ChangeLog.md`.

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
