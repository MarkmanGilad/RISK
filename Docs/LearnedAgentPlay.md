# Learned-agent interactive play

`risk/app/learned_agent_play.py` supplies the saved-policy loading contract
for interactive games. It is separate from training and from the offline
checkpoint evaluators in [ChooseAgent.md](ChooseAgent.md).

## Setup-screen selections

`InitScreenState.learned_selections` holds learned-seat choices outside
`GameSettings`. The visible type cycle reaches **Learned Agent** after
Killbot; leaving that type removes the selection. A learned seat is stored in
`GameSettings` as an `ai` placeholder until the game is built. `GameFactory`
creates the normal placeholder agent, then `risk/app/main.py` replaces each
selected seat with its loaded learned agent.

The setup screen supports a manual policy file, a checkpoint folder, and
presets. A manual selection can choose `DQN`, `Dueling_DQN`, or `PPO`; a
preset supplies its configured learner kind and display label.

## Loading and inference contract

Supported learned kinds are `DQN`, `Dueling_DQN`, and `PPO`.

| Selected path | Online weights read |
|---|---|
| Raw `.pt` file | The file itself is an online network state dictionary. |
| Checkpoint folder | `model.pt` is required; the loader reads `payload["net"]`. Standard `epNNNNNN/model.pt` folders work, but the directory name is not required. |

For every selected seat, the loader creates a fresh learner, applies the
online weights directly with `agent.net.load_state_dict(...)`, attaches it to
that seat's real environment, sets `epsilon = 0.0`, and sets evaluation mode
(`train_mode=False`). It deliberately does not restore replay, optimizer,
target-network, epsilon-schedule, or training-progress state. Two seats that
choose the same checkpoint still receive independent agent and network
instances.

## Validation and Start behavior

Before a game can start, `validate_selections()` builds a temporary game
context and attempts every selected load and seat attachment. It returns a
player-specific message for a missing, unsupported, invalid, or incompatible
selection. The Start button remains disabled when ordinary setup validation
fails, any learned seat has no model selected, or learned-policy validation
returns an error.

## Presets and labels

Presets are read from `Params/play_agents.json`. The registry must have
`version: 1`, unique model ids, and non-empty `id`, `label`, `agent_kind`, and
`checkpoint` fields. Relative checkpoint paths are resolved from the
repository root. Interactive labels use the selected label, falling back to
the learner kind.

The current preset list contains the top three fully evaluated compatible
DQN_303 checkpoints from the 4200-5500 selection window: ep004500, ep005000,
and ep004300. Older DQN_103 presets are intentionally not listed because they
pre-date the current `TradeInHead` input shape and fail learned-agent
validation.

## Coverage

`Temp/tests/test_learned_agent_play.py` covers UI cycling, preset validation,
both checkpoint formats, real-seat attachment, deterministic inference,
validation failures, independent instances, labels, and mixed interactive
games.
