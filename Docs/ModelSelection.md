# Model-selection plan: which checkpoint is actually best?

Status: **plan only, not implemented.** This is a starting proposal to review
and adjust before any code is written.

## Background: what motivated this

`Checkpoints/DQN_103/evaluations/checkpoint_eval_ep006200_to_006700.json`
(54 games/checkpoint against fixed heuristic rosters, see `Docs/ChooseAgent.md`)
shows two separate findings once analyzed directly:

- Win rate drops monotonically with player count for essentially every
  checkpoint (3p ~85-100% down to 6p ~55-78% for the current top 5). That
  pattern is consistent and expected — more simultaneous opponents is a
  harder game — and is not the problem this plan addresses.
- Within the 6-player suite, the top 3 non-leading checkpoints
  (`ep006000`, `ep005650`, `ep004700`) are **tied at exactly 55.6% (10/18)**.
  A chi-square test on win rate by seat across all 1,080 six-player games in
  the run gives p=0.53 (seat position is noise), while a chi-square test on
  *which heuristic wins* gives p<0.0001 (sentinel/empire are genuinely
  tougher than raider/killbot, random never wins). In other words: the
  suite has enough games to say something reliable about *opponent*
  identity, but not enough to break ties between close checkpoints — 18
  games per checkpoint per player count is too small a sample for that.

This matches `Docs/PPO.md`'s own caveat: promoting a "best" checkpoint
"requires a larger fixed suite (at least 100 games, balanced across seats,
player counts, and representative rosters)." The current suite is a good
first filter, not a final answer.

## What already exists — reuse before building anything new

`risk/learning/choose_agent.py` (documented in `Docs/ChooseAgent.md`)
already has both pieces this plan needs:

- **`CheckpointEvaluator`** — one learner vs. fixed heuristic rosters, every
  checkpoint in a run, 3/4/5/6-player suites. This is what produced the
  DQN_103 JSON above.
- **`AgentMatchEvaluator`** — **already supports learner-vs-learner.** It
  takes 3-6 participants, each either `"heuristic"` (built-in bot) or
  `"checkpoint"` (any saved policy: any run, any `agent_kind` string
  `build_learner_agent` understands — `DQN`, `Dueling_DQN`, `PPO`, `PQN`,
  `PQN_e`, `PQN_e0`, `ADQN`, ...). Every seed plays one cyclic seat rotation
  per participant. Nothing here needs to be added — a DQN checkpoint, a
  Dueling-DQN checkpoint, and a PPO checkpoint can already sit in the same
  match today via `main()`'s commented-out example block.

So "run learners against each other, possibly from other runs or a
different architecture" is a **methodology/orchestration question, not a
missing-feature question.** The real gaps (below) are: not enough games per
comparison to trust the result, no report/plot for match results (only
`CheckpointEvaluator` has `show_checkpoint_win_rate_chart`), and no script
that automatically assembles a candidate list across runs.

## Candidate inventory (current `Checkpoints/`)

Runs available today, by `agent_kind` (from directory naming and
`build_learner_agent`):

- `DQN`: `DQN_060`, `DQN_102`, `DQN_103`, `DQN_104`, `DQN_105`
- `Dueling_DQN`: `Dueling_DQN_040`, `Dueling_DQN_100`, `Dueling_DQN_101`
- `PPO`: `PPO_041`..`PPO_045`
- `PQN` / `PQN_e`: `PQN_046`, `PQN_e_047`
- `ADQN`: `ADQN_050`..`ADQN_052`
- Legacy `run_013`, `run_014`, `run_020`..`run_023`, `run_030`: predate the
  `<agent_kind>_<run_id>` naming convention. **Unknown agent_kind — do not
  assume compatibility.** Before including any of these, confirm which
  agent class actually produced them (check training-era code via git log,
  or just try loading a `best/*.pt` into each candidate agent class and see
  which one's `net.load_state_dict` accepts it without shape errors). If
  none can be confirmed cheaply, drop them from the tournament rather than
  guessing.

Every non-legacy run already has a `best/manifest.json` (written by the
per-episode `Evaluator` during training, see `Docs/Eval.md`) ranking its own
top checkpoints by training-time eval score. That manifest is the natural
seed list for "best candidate from each run" — no need to re-run
`CheckpointEvaluator` on every run just to pick a contender.

## Phased plan

### Phase 1 — break the current DQN_103 tie (cheap, do first)

Re-run `CheckpointEvaluator` for the tied checkpoints
(`ep006000`, `ep005650`, `ep004700`, plus the two leaders `ep006700`,
`ep005400` as a sanity check) with more seeds — e.g. 3 → 15-20 — so the
6-player suite goes from 18 to 90-120 games per checkpoint. Pure reuse of
existing code (`_DEFAULT_SEEDS` override), no new logic. This alone may
resolve the tie without needing any cross-model matches at all.

### Phase 2 — self-tournament among DQN_103's top candidates

Feed the (now hopefully-distinct) top 3-5 DQN_103 checkpoints into
`AgentMatchEvaluator` with **no heuristics**, many seeds (see "How many
games" below), 6-player games (the suite where they're least
differentiated).

Caveat to watch for: ranking models purely by how they do against each
other risks a rock-paper-scissors artifact (A beats B, B beats C, C beats
A) that says nothing about real strength, especially since these
checkpoints come from the same run and may share correlated blind spots.
Treat Phase 2 as a signal, not a verdict — cross-check against Phase 3.

### Phase 3 — anchored tournament (recommended primary signal)

Same top 3-5 DQN_103 checkpoints, but replace 1-2 seats with fixed
heuristic anchors — `sentinel` and `empire`, since the existing data already
shows they're the two toughest opponents in the 6-player suite (30% and 27%
of all losses across 58 checkpoints, vs. 19% each for raider/killbot, 0%
for random). This keeps a fixed, already-characterized difficulty floor in
every match so the ranking isn't purely self-referential, while still
letting the candidate checkpoints compete directly. `AgentMatchEvaluator`
already supports mixing `"checkpoint"` and `"heuristic"` participants in one
call — this is exactly the commented-out `main()` example, just with more
than one checkpoint participant.

### Phase 4 — cross-run / cross-architecture bracket (stretch goal)

Take the top-1 (or top-2) entry from each other run's `best/manifest.json`
— one from `Dueling_DQN_101`, one from `PPO_045`, one from `PQN_e_047`, one
from `ADQN_052`, plus DQN_103's own best — and compare them directly. Since
a single `AgentMatchEvaluator.evaluate()` call caps at 6 participants, and
this phase alone has 5+ candidates before even adding DQN_103's own
finalists, this needs **grouped matches, not one giant match**:

- Run 2-3 matches of ≤6 participants each (mixing candidates so every pair
  of architectures shares at least one match), then
- combine per-participant win totals across matches into one leaderboard
  (see "New tooling" below — this aggregation doesn't exist yet).

This phase is explicitly lower-confidence: different runs may have used
different reward shaping, training length, or environment settings
(`Docs/Reward.md`/`Docs/Trainer.md` show reward coefficients changed across
DQN_103/104/105 already), so a loss here reflects "which finished training
run is best today," not a controlled architecture comparison. Worth doing,
but label results accordingly rather than treating it as a clean ablation.

## How many games is enough?

`AgentMatchEvaluator` plays `seeds x participants` games (cyclic rotation,
not full permutation — with N participants only N of the N! seat orderings
are sampled per seed). Using the same `_DEFAULT_SEEDS = (0, 1, 2)` as
`CheckpointEvaluator`'s default would give only 3 games per participant per
match — nowhere near enough given Phase 1's finding that 18 games/checkpoint
wasn't enough to separate close candidates. Recommend **20-30 seeds** per
tournament match (matching the order of magnitude that gave the opponent-
identity chi-square test real power in the existing 58-checkpoint dataset:
1,080 six-player games total, ~180-216 games per opponent kind). That's
100-180 games per match for 5-6 participants — a multi-hour run per match at
current step counts, so budget accordingly and consider trimming
`max_steps`/game count only after checking Phase 1's result first.

## New tooling needed (design, not code, in this pass)

1. **Match report/plot function**, analogous to
   `show_checkpoint_win_rate_chart` but for `AgentMatchEvaluator` JSON:
   win rate per participant with a simple confidence interval (or the same
   chi-square-vs-uniform check used above), plus the secondary metrics
   `AgentMatchEvaluator` already records per game (`final_territory_count`,
   `final_army_count`, `territories_conquered`, `agent_turns_survived`) as
   tie-breakers when win-rate differences aren't significant.
2. **Cross-match leaderboard aggregator** for Phase 4: since one match caps
   at 6 seats, sum wins/games per participant *name* across several match
   JSON files into one ranked table. `AgentMatchEvaluator` itself has no
   notion of "this participant also played in that other match file" — this
   has to be a separate, small reducer over multiple result JSONs.
3. **Candidate-list builder**: given a run directory, read `best/manifest.json`
   (or a `CheckpointEvaluator` result JSON) and emit the top-K checkpoint
   paths as ready-to-use `AgentMatchEvaluator` participant dicts, instead of
   hand-editing `main()`'s participant list per run as today.

None of these change `CheckpointEvaluator`, `AgentMatchEvaluator`, or their
result-file formats — they're read-only consumers of existing JSON, in the
same spirit as `show_checkpoint_win_rate_chart`.

## Open decisions before implementing

- Exact seed count per tournament match (20? 30? — trade off runtime vs.
  statistical power; Phase 1's result should inform this).
- Which heuristics anchor Phase 3 — `sentinel` + `empire` is the
  data-backed default, but open to using only one anchor to leave more
  seats for checkpoint candidates.
- Whether to spend time confirming legacy `run_0XX` agent_kind at all, or
  simply exclude them from Phase 4 as untraceable.
- Where match result JSONs live — suggest `Checkpoints/matches/<label>.json`
  per the existing `AgentMatchEvaluator` convention in `Docs/ChooseAgent.md`.

## Suggested implementation order

1. Phase 1 (more seeds on existing tied DQN_103 checkpoints) — run only,
   no code changes.
2. Build the match report/plot tool (#1 under "New tooling").
3. Run Phase 2 and Phase 3 for DQN_103, using that report tool.
4. Build the candidate-list builder and cross-match leaderboard aggregator
   (#2, #3 under "New tooling") only once Phase 4 is actually scheduled.
5. Run Phase 4.
6. Update `Docs/ChangeLog.md` after each implementation step, and update
   this file's "Status" line once any part moves from plan to code.
