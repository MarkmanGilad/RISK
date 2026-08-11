"""Training hyperparameters for reinforcement-learning self-play agents.

Edit this file to tune a run. Every value is a plain module constant —
no classes, importable with ``from risk.learning.train_constants import *``.

The only thing you set *per run* is ``RUN_ID`` in ``risk/learning/trainer.py``
``main()``. All other knobs live here so they're easy to find and change
together without touching the training loop.
"""
from __future__ import annotations

from typing import Final

# region Episode setup

MIN_PLAYERS: Final[int] = 3       # minimum opponents per episode (inclusive)
MAX_PLAYERS: Final[int] = 6       # maximum opponents per episode (inclusive)
MAX_STEPS_PER_EPISODE: Final[int] = 2_000
TRAIN_OPPONENT_AGENT_KINDS: Final[tuple[str, ...]] = (
    "random",
    "raider",
    "sentinel",
    "empire",
    "killbot",
)
# endregion

# region DQN replay and gradient update

REPLAY_BUFFER_CAPACITY: Final[int] = 100_000
BATCH_SIZE: Final[int] = 64
TRAIN_STEPS_PER_CALL: Final[int] = 1  # gradient steps taken each time
TARGET_ALGORITHM: Final[str] = "ddqn"
LOSS_NAME: Final[str] = "smooth_l1"
GRAD_CLIP_MAX_NORM: Final[float] = 10.0
# endregion

# region PPO rollout and gradient update

PPO_ROLLOUT_LENGTH: Final[int] = 1024
PPO_EPOCHS: Final[int] = 4
PPO_MINIBATCH_SIZE: Final[int] = 256
PPO_CLIP_EPS: Final[float] = 0.2
PPO_TARGET_KL: Final[float] = 0.02
PPO_GAE_LAMBDA: Final[float] = 0.95
PPO_VALUE_LOSS_COEF: Final[float] = 0.1
PPO_VALUE_HUBER_BETA: Final[float] = 1.0
PPO_ENTROPY_COEF: Final[float] = 0.01
PPO_LR: Final[float] = 7.5e-5
# endregion

# region PQN replay-based policy improvement

PQN_POLICY_LOSS_COEF: Final[float] = 0.1  # Initial policy-loss weight; Docs/PQN.md §24.C
# endregion

# region ADQN centered-advantage improvement

ADQN_ADVANTAGE_LOSS_COEF: Final[float] = 0.25
ADQN_MAX_ADVANTAGE_LOSS_FRACTION: Final[float] = 0.25
ADQN_LOSS_BALANCE_EPSILON: Final[float] = 1e-8
ADQN_ADVANTAGE_WEIGHT_SCALE: Final[float] = 5.0
ADQN_ADVANTAGE_WEIGHT_SATURATION: Final[float] = 0.95
ADQN_GRAD_DIAGNOSTIC_EVERY: Final[int] = 100
# endregion

# region Exploration

EPSILON_START: Final[float] = 1.0
EPSILON_END: Final[float] = 0.01
EPSILON_DECAY_EPISODES: Final[int] = 100  # episode at which epsilon reaches EPSILON_END
# endregion

# region Run control

# Default episode count when running `python -m risk.learning.trainer`.
TRAIN_EPISODES: Final[int] = 10000

# Print one progress line every N episodes.
PROGRESS_EVERY: Final[int] = 10

# Rolling window (in episodes) for the win_rate_last_n W&B metric.
ROLLING_WIN_RATE_WINDOW: Final[int] = 50
# endregion

# region Checkpointing

CHECKPOINT_DIR: Final[str] = "Checkpoints"
CHECKPOINT_AFTER: Final[int] = 200   # first checkpoint episode
CHECKPOINT_EVERY: Final[int] = 50    # save every N episodes after that
# endregion

# region Evaluation

EVAL_EVERY_EPISODES: Final[int] = 25
EVAL_KEEP_BEST: Final[int] = 5
EVAL_MAX_STEPS: Final[int] = MAX_STEPS_PER_EPISODE
# endregion

# region Reward

REWARD_SHAPING_STEP_CAP: Final[float] = 10.0
# DQN_105 uses an intermediate scale while testing the marginal reinforcement
# correction, preserving the same dense-to-terminal ratio as DQN_103/104.
REWARD_SHAPING_SCALE: Final[float] = 0.3

REWARD_TERMINAL_WIN: Final[float] = 300.0
REWARD_TERMINAL_LOSS: Final[float] = -300.0

REWARD_TRADE_IN_EARLY: Final[float] = 0.30
REWARD_TRADE_IN_TERRITORY_MATCH: Final[float] = 0.60

REWARD_REINFORCE_READY_SCALE: Final[float] = 0.50
REWARD_REINFORCE_READY_RATIO: Final[float] = 1.50
REWARD_REINFORCE_TOTAL_SCALE: Final[float] = 0.50
REWARD_REINFORCE_TOTAL_RATIO: Final[float] = 2.00
REWARD_REINFORCE_READY_CAP: Final[float] = 7.00
REWARD_REINFORCE_CONTINENT_SCALE: Final[float] = 5.00
REWARD_REINFORCE_INTERIOR: Final[float] = -0.80
REWARD_REINFORCE_SPLIT: Final[float] = 0.20

REWARD_ATTACK_FEWER_DICE: Final[float] = -1.25
REWARD_ATTACK_RATIO_SCALE: Final[float] = 2.00
REWARD_ATTACK_RATIO_CAP: Final[float] = 3.00
REWARD_ATTACK_RATIO_THRESHOLD: Final[float] = 1.50
REWARD_ATTACK_CONTINENT_DOMINATION: Final[float] = 0.80
REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN: Final[float] = 0.10
REWARD_ATTACK_CONTINENT_ADVANTAGE: Final[float] = 1.20
REWARD_ATTACK_ARMY_TRADE: Final[float] = 0.60
REWARD_ATTACK_ELIMINATE_OPPONENT_BASE: Final[float] = 4.00
REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD: Final[float] = 1.50
REWARD_ATTACK_CONTINENT_CAPTURED: Final[float] = 4.00
REWARD_ATTACK_CONQUER_TERRITORY: Final[float] = 1.20
REWARD_ATTACK_CONQUER_WITH_CARD: Final[float] = 1.00
REWARD_ATTACK_CARD_TERRITORY_MATCH: Final[float] = 0.60
REWARD_ATTACK_STOP_WITHOUT_CARD: Final[float] = -2.00
REWARD_ATTACK_UNFINISHED_TARGET: Final[float] = -0.50

REWARD_OCCUPY_FORWARD_MOMENTUM: Final[float] = 1.00

REWARD_FORTIFY_TOWARD_FRONTIER: Final[float] = 1.00
REWARD_FORTIFY_BALANCE_SCALE: Final[float] = 2.00
REWARD_FORTIFY_CONTINENT_PUSH: Final[float] = 0.80

REWARD_TERRITORY_DELTA: Final[float] = 20.00
REWARD_ARMY_DELTA_RELATIVE_SCALE: Final[float] = 0.10
REWARD_CONTINENT_DELTA_RELATIVE: Final[float] = 2.50
REWARD_TERRITORY_HOLD: Final[float] = 0.00
REWARD_CONTINENT_LOST: Final[float] = 5.00
REWARD_TERMINAL_TIMEOUT: Final[float] = 0.0
# endregion


# region Public constants
__all__ = [
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "MAX_STEPS_PER_EPISODE",
    "TRAIN_OPPONENT_AGENT_KINDS",
    "REPLAY_BUFFER_CAPACITY",
    "BATCH_SIZE",
    "TRAIN_STEPS_PER_CALL",
    "TARGET_ALGORITHM",
    "LOSS_NAME",
    "GRAD_CLIP_MAX_NORM",
    "PPO_ROLLOUT_LENGTH",
    "PPO_EPOCHS",
    "PPO_MINIBATCH_SIZE",
    "PPO_CLIP_EPS",
    "PPO_TARGET_KL",
    "PPO_GAE_LAMBDA",
    "PPO_VALUE_LOSS_COEF",
    "PPO_VALUE_HUBER_BETA",
    "PPO_ENTROPY_COEF",
    "PPO_LR",
    "PQN_POLICY_LOSS_COEF",
    "ADQN_ADVANTAGE_LOSS_COEF",
    "ADQN_MAX_ADVANTAGE_LOSS_FRACTION",
    "ADQN_LOSS_BALANCE_EPSILON",
    "ADQN_ADVANTAGE_WEIGHT_SCALE",
    "ADQN_ADVANTAGE_WEIGHT_SATURATION",
    "ADQN_GRAD_DIAGNOSTIC_EVERY",
    "EPSILON_START",
    "EPSILON_END",
    "EPSILON_DECAY_EPISODES",
    "TRAIN_EPISODES",
    "PROGRESS_EVERY",
    "ROLLING_WIN_RATE_WINDOW",
    "CHECKPOINT_DIR",
    "CHECKPOINT_AFTER",
    "CHECKPOINT_EVERY",
    "EVAL_EVERY_EPISODES",
    "EVAL_KEEP_BEST",
    "EVAL_MAX_STEPS",
    "REWARD_SHAPING_STEP_CAP",
    "REWARD_SHAPING_SCALE",
    "REWARD_TERMINAL_WIN",
    "REWARD_TERMINAL_LOSS",
    "REWARD_TRADE_IN_EARLY",
    "REWARD_TRADE_IN_TERRITORY_MATCH",
    "REWARD_REINFORCE_READY_SCALE",
    "REWARD_REINFORCE_READY_RATIO",
    "REWARD_REINFORCE_TOTAL_SCALE",
    "REWARD_REINFORCE_TOTAL_RATIO",
    "REWARD_REINFORCE_READY_CAP",
    "REWARD_REINFORCE_CONTINENT_SCALE",
    "REWARD_REINFORCE_INTERIOR",
    "REWARD_REINFORCE_SPLIT",
    "REWARD_ATTACK_FEWER_DICE",
    "REWARD_ATTACK_RATIO_SCALE",
    "REWARD_ATTACK_RATIO_CAP",
    "REWARD_ATTACK_RATIO_THRESHOLD",
    "REWARD_ATTACK_CONTINENT_DOMINATION",
    "REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN",
    "REWARD_ATTACK_CONTINENT_ADVANTAGE",
    "REWARD_ATTACK_ARMY_TRADE",
    "REWARD_ATTACK_ELIMINATE_OPPONENT_BASE",
    "REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD",
    "REWARD_ATTACK_CONTINENT_CAPTURED",
    "REWARD_ATTACK_CONQUER_TERRITORY",
    "REWARD_ATTACK_CONQUER_WITH_CARD",
    "REWARD_ATTACK_CARD_TERRITORY_MATCH",
    "REWARD_ATTACK_STOP_WITHOUT_CARD",
    "REWARD_ATTACK_UNFINISHED_TARGET",
    "REWARD_OCCUPY_FORWARD_MOMENTUM",
    "REWARD_FORTIFY_TOWARD_FRONTIER",
    "REWARD_FORTIFY_BALANCE_SCALE",
    "REWARD_FORTIFY_CONTINENT_PUSH",
    "REWARD_TERRITORY_DELTA",
    "REWARD_ARMY_DELTA_RELATIVE_SCALE",
    "REWARD_CONTINENT_DELTA_RELATIVE",
    "REWARD_TERRITORY_HOLD",
    "REWARD_CONTINENT_LOST",
]
# endregion
