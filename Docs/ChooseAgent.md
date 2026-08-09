# Agent evaluation plan

Status: implemented in risk/learning/choose_agent.py. It adds new classes
only; Trainer, Evaluator, agent classes, checkpoint formats, and the training
path remain unchanged.

The evaluators save raw result dictionaries and never choose a winner. A later
Python run decides which checkpoint or agent is best for its intended game
type.

## Private helpers in the new module

The new module owns two private helpers shared by its new classes.

- A pure game rollout builds a fresh game, attaches every agent to its actual
  seat, sets learned policies to epsilon=0.0 and train_mode=False, plays until
  terminal/elimination/max_steps, and never calls remember or learn.
- A net-only checkpoint loader uses the existing load_params for a policy-only
  .pt file. For an epNNNNNN directory it reads trusted local model.pt and
  loads only its net payload into agent.net. It never loads replay, optimizer
  state, target net, epsilon, or train steps.

These are plain module-level functions, not a shared base class: neither new
class specializes the other, so there is no real is-a relationship to model,
and passing/calling shared functions is simpler than a class hierarchy for
pure code reuse. They remain private to the new module and do not touch
`Evaluator`.

The result-file resume/atomic-write pattern (format-version validation,
skip-already-recorded games) described separately under each evaluator below
is also common to both; if writing it twice would be verbatim duplication,
give it a third shared private function instead of repeating it.

### Reward accounting compatibility

`_play_game` applies `env.reward.end_of_turn(...)` when the learner ended its
turn with `FortifyAction`, when the game ends on any later player's step, or
when the learner is eliminated on any later player's step. This matches the
trainer's `FortifyAction or done` rule, so checkpoint `episode_reward_sum`
and `reward_per_agent_turn` remain comparable with training metrics. The
focused tests include an opponent-eliminates-learner case.

## CheckpointEvaluator: evaluate every checkpoint

CheckpointEvaluator takes an explicit learner agent_kind and run id. It finds
every regular epNNNNNN directory in:

    Checkpoints/<agent_kind>_<run_id:03d>/

The caller may restrict the expensive evaluation to an inclusive episode
window:

    evaluate_run(min_episode=2000, max_episode=6700)

Both bounds are optional. A missing minimum includes every earlier checkpoint;
a missing maximum includes every later checkpoint. Reject negative bounds and
min_episode greater than max_episode. Sort checkpoint directories by their
numeric episode and evaluate only entries inside the requested inclusive range.
The range is an execution filter, not result-file identity: a later call with
a wider range may reuse the same compatible JSON and add only its missing
checkpoint/game records without deleting earlier results.

It evaluates each checkpoint against fixed heuristic rosters, testing every
learner seat and three fixed seeds:

    3 players: Raider, Killbot
    4 players: Raider, Sentinel, Killbot
    5 players: Random, Raider, Sentinel, Killbot
    6 players: Random, Raider, Sentinel, Empire, Killbot

The learner replaces each seat in turn, while that game size's same
player_count - 1 heuristic opponents fill the other seats in order.

    (3 + 4 + 5 + 6 seats) x 3 seeds = 54 games per checkpoint

Every game creates a fresh learner through the existing build_learner_agent
factory, then the private rollout helper attaches it to the actual evaluation
environment and seat.

### Results dictionary and resume

evaluate_run returns and atomically writes a JSON-serializable dictionary after
every completed game. By default it saves under the evaluated run:

    Checkpoints/<agent_kind>_<run_id:03d>/evaluations/
        checkpoint_eval_ep<min-or-first>_to_<max-or-last>.json

For example, DQN run 103 evaluated from episode 2000 through 6700 writes:

    Checkpoints/DQN_103/evaluations/checkpoint_eval_ep002000_to_006700.json

The caller may pass output_path to use another location. A later selection
script reads this JSON directly; W&B also shows the separate evaluation run's
per-game rows and completed-checkpoint total-win summaries.

On Windows, a sync client, antivirus scanner, editor, or other process may
briefly lock the existing JSON just as it is being replaced. The atomic replace
is retried up to 10 times with a 0.25-second pause; a lock that outlasts that
short window is still raised so the result is never silently lost.

The dictionary has this shape:

    {
      "format_version": 1,
      "agent_kind": "DQN",
      "run_id": 103,
      "seeds": [0, 1, 2],
      "max_steps": 1000,
      "suites": {"3": ["raider", "killbot"], "...": ["..."]},
      "checkpoints": {
        "ep002450": {
          "path": "Checkpoints/DQN_103/ep002450",
          "episode": 2450,
          "total_wins": 0,
          "scheduled_games": 54,
          "completed_games": 1,
          "total_win_rate": 0.0,
          "games": [
            {
              "player_count": 3,
              "learner_seat": 0,
              "seed": 0,
              "opponent_kinds": ["raider", "killbot"],
              "win": 1,
              "winner_kind": "learner",
              "winner_seat": 0,
              "reached_max_steps": false,
              "final_territory_count": 0,
              "final_army_count": 0,
              "territories_conquered": 0,
              "episode_reward_sum": 0.0,
              "agent_turns_survived": 0,
              "reward_per_agent_turn": 0.0,
              "step_count": 0
            }
          ]
        }
      }
    }

total_wins, completed_games, and total_win_rate are the first-pass overall
comparison, with total_win_rate = total_wins / completed_games.
scheduled_games is always 54; it is separate so a partial resumable result
file is never presented as a finished evaluation. The raw games list remains
the source of truth, allowing later code to select different best checkpoints
for 3-, 4-, 5-, or 6-player games, or inspect seat and seed effects. Timeout
and final-board fields distinguish a loss from a maximum-step game.

If the output JSON already exists, validate its format version, agent kind,
run id, seeds, max steps, and suites. Skip only a game already recorded under the same
(checkpoint, player_count, learner_seat, seed) key. A metadata mismatch raises
an error rather than mixing incomparable runs.

W&B logging is optional and starts a separate evaluation run. Log one row per
game and one completed-checkpoint summary row with the total fields. Do not
log or persist an automatic rank. Evaluation runs use the separate
`Risk-Model-Evaluation` W&B project rather than the training project, because
their charts are checkpoint/game comparisons rather than training curves.
After every fully completed checkpoint, the evaluator also logs the real W&B
image `checkpoint_win_rate_bar_chart`: X is the checkpoint/agent name (for
example `ep006200`) and Y is that checkpoint's final `total_win_rate`. Its
source rows are available separately as `checkpoint_win_rate_table`.
Partially evaluated checkpoints are deliberately omitted so each bar is based
on the same 54-game suite.

## AgentMatchEvaluator: caller-chosen agents

AgentMatchEvaluator lets Python code choose any 3-6 player mixture of saved
learned policies and built-in heuristics.

A checkpoint participant requires its network agent_kind and either a
policy-only .pt path or regular checkpoint directory. A heuristic participant
uses one existing kind: random, raider, sentinel, empire, or killbot.

For every seed, play one cyclic rotation per participant. Each participant
occupies every seat once per seed without the N! cost of all seat permutations.
With N participants and three seeds, this produces 3 x N games.

Each game uses fresh agent instances and the private rollout helper. Persist a
self-describing JSON result after every game with format version, participant
specifications, seeds, max steps, per-game roster/seat assignment, winner,
timeout, final territory/army counts, and total wins/completed games/win rate
for every participant. A compatible existing result file skips already-
recorded roster/seed rotations. This evaluator never declares a winner.

For a mixed-agent match, per-participant metrics must be objective board/game
facts: win, turns survived, territories conquered, final territory count, and
final army count. Do not record per-participant shaped reward totals or reward
per turn: Environment.step returns reward for one designated reward_player at
a time, and calculating every participant's shaped reward would require
changing the environment or replaying reward logic. The checkpoint-versus-
heuristic evaluator may still retain its one learner's reward metrics.

W&B logging is optional and uses a separate AgentMatch_<timestamp> run: one
row per game and one final total per participant.

For a custom match, output_path is required because there is no one natural
training-run directory. The returned dictionary and that JSON file are the
authoritative results; the W&B run is a convenient visual view of the same
games and totals.

## Tests and documentation

Add Temp/tests/test_choose_agent.py and update Docs/Testing.md.

- Verify `Evaluator`'s own tests are untouched by this addition; the new
  module imports nothing private from it.
- Discover all epNNNNNN checkpoints; verify the private loader changes only
  the online network and never loads replay, optimizer, target, or epsilon.
- Verify inclusive min_episode/max_episode filtering, omitted-bound behavior,
  invalid-range rejection, numeric checkpoint ordering, and widening a range
  without repeating recorded games.
- Verify the 54-game checkpoint schedule, raw per-game result shape, atomic
  persistence, individual-game resume, partial-result totals, and rejection
  of incompatible files.
- Verify learner agents are attached to their real evaluation seats and run
  deterministically before playing.
- Verify custom participant validation, cyclic seat rotation, mixed
  heuristic/checkpoint rosters, fresh instances, per-game persistence,
  objective participant metrics, and participant totals.
- Verify W&B-disabled evaluation still writes and returns identical results.

After implementation, update Docs/ChangeLog.md and run the focused tests plus
existing evaluator/checkpoint tests using C:\venvs\ai-rl.

## Current Python interface

Use the fixed checkpoint suite like this (W&B is deliberately opt-in):

    results = CheckpointEvaluator(use_wandb=True).evaluate_run(
        "DQN", 103, min_episode=2000, max_episode=6700
    )

Use a caller-chosen match like this:

    results = AgentMatchEvaluator().evaluate(
        [
            {"name": "dqn_103", "kind": "checkpoint", "agent_kind": "DQN",
             "checkpoint": "Checkpoints/DQN_103/ep006700"},
            {"name": "raider", "kind": "heuristic", "agent_kind": "raider"},
            {"name": "sentinel", "kind": "heuristic", "agent_kind": "sentinel"},
        ],
        output_path="Checkpoints/matches/dqn_103_vs_heuristics.json",
    )

The custom match result has one game per cyclic seat rotation for each seed.
Checkpoint evaluation uses the complete 54-game fixed suite by default.

The module also has an editable `main()` for direct use without a CLI:

    python -m risk.learning.choose_agent

It also supports direct file execution, including the VS Code Run button:

    & "C:\venvs\ai-rl\Scripts\python.exe" "risk\learning\choose_agent.py"

Change `AGENT_KIND`, `RUN_ID`, `MIN_EPISODE`, `MAX_EPISODE`, and `USE_WANDB`
at the top of `main()`. The default block runs checkpoint evaluation. The
commented block immediately below it is the custom `AgentMatchEvaluator`
example; comment out the default call, uncomment that block, and edit its
participants/checkpoint paths to run a chosen match.

`PROGRESS_EVERY_STEPS` controls live console output. The value in `main()`
prints the checkpoint/game when it starts and finishes, then a step count at
that action interval for a long game. JSON results are still written after
every completed game, so the console and saved file both show that an
evaluation is advancing. `main()` also prints one immediate line with the
selected run and episode range before it starts discovery or W&B setup.

`EVALUATION_NAME = None` keeps the default result-file behavior and resumes a
compatible earlier evaluation for the same range. To intentionally start a
separate evaluation, set a new label such as `EVALUATION_NAME = "reward_0_1_v2"`.
Its raw results are saved as:

    Checkpoints/<agent_kind>_<run_id>/evaluations/
        checkpoint_eval_<EVALUATION_NAME>.json

Use a different label whenever game settings, reward policy, comparison goal,
or any other reason makes results non-comparable with an earlier evaluation.

`RESULTS_PATH` is an optional direct JSON path in `main()`. It overrides the
default/named path and is useful when expanding one shared evaluation in
several episode windows: point every compatible window at the same JSON, and
the evaluator skips already recorded games while adding the missing ones.

### Local checkpoint bar chart

`show_checkpoint_win_rate_chart(results_path)` reads a saved checkpoint
evaluation JSON and saves a Matplotlib bar-chart PNG beside it. In `main()`,
set:

    PLOT_RESULTS_PATH = "Checkpoints/DQN_103/evaluations/checkpoint_eval_ep006200_to_006700.json"

Then run the module normally. It saves the chart and exits without starting
W&B or playing/resuming any games. The default PNG name is the JSON file name
with `_win_rate.png` appended. Set it back to `None` for evaluation.

## Checkpoint-weight cache

`CheckpointEvaluator` reads a checkpoint's online `net` state dictionary
once, before its first unfinished game. It keeps those weights in memory only
while that checkpoint is evaluated, then releases them. Each game still builds
a fresh learner and copies the cached online weights into its own `agent.net`,
then attaches that new agent to the real game and seat.

This preserves the important isolation rule: no agent instance, environment,
replay, optimizer, target network, epsilon, or train state is shared between
games. It only removes repeated reads of identical policy weights from disk.
`AgentMatchEvaluator` uses the same cache for repeated checkpoint participants
within one `evaluate(...)` call. Neither evaluator persists the cache in JSON
or loads any additional training state.
