"""Training hyperparameters for `GNN_DQN_Agent` self-play.

Edit this file to tune a run. Every value is a plain module constant —
no classes, importable with ``from risk.learning.train_constants import *``.

The only thing you set *per run* is ``RUN_ID`` in ``risk/learning/trainer.py``
``main()``. All other knobs live here so they're easy to find and change
together without touching the training loop.
"""
from __future__ import annotations

from typing import Final

# --- Episode setup ----------------------------------------------------------

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

# --- Replay / gradient update -----------------------------------------------

BATCH_SIZE: Final[int] = 64
TRAIN_STEPS_PER_CALL: Final[int] = 1  # gradient steps taken each time
TARGET_ALGORITHM: Final[str] = "ddqn"
LOSS_NAME: Final[str] = "smooth_l1"
GRAD_CLIP_MAX_NORM: Final[float] = 10.0

# --- Exploration (epsilon-greedy, linear decay by episode) ------------------

EPSILON_START: Final[float] = 1.0
EPSILON_END: Final[float] = 0.05
EPSILON_DECAY_EPISODES: Final[int] = 200  # episode at which epsilon reaches EPSILON_END

# --- Run control -------------------------------------------------------------

# Default episode count when running `python -m risk.learning.trainer`.
TRAIN_EPISODES: Final[int] = 10000

# Print one progress line every N episodes.
PROGRESS_EVERY: Final[int] = 10

# Rolling window (in episodes) for the win_rate_last_n W&B metric.
ROLLING_WIN_RATE_WINDOW: Final[int] = 50

# --- Checkpointing ----------------------------------------------------------

CHECKPOINT_DIR: Final[str] = "Checkpoints"
CHECKPOINT_AFTER: Final[int] = 200   # don't checkpoint before this episode
CHECKPOINT_EVERY: Final[int] = 200   # save every N episodes after that

# --- Evaluation (see Docs/Eval.md) -------------------------------------------

EVAL_EVERY_EPISODES: Final[int] = 50
EVAL_KEEP_BEST: Final[int] = 5
EVAL_MAX_STEPS: Final[int] = MAX_STEPS_PER_EPISODE

# --- Reward (see Docs/Reward.md) ---------------------------------------------

REWARD_SHAPING_STEP_CAP: Final[float] = 10.0

REWARD_TERMINAL_WIN: Final[float] = 100.0
REWARD_TERMINAL_LOSS: Final[float] = -100.0

REWARD_TRADE_IN_EARLY: Final[float] = 0.30
REWARD_TRADE_IN_TERRITORY_MATCH: Final[float] = 0.60

REWARD_REINFORCE_CONCENTRATION: Final[float] = 1.20
REWARD_REINFORCE_ATTACK_READINESS_SCALE: Final[float] = 1.50
REWARD_REINFORCE_RATIO_CAP: Final[float] = 2.50
REWARD_REINFORCE_NO_ENEMY_NEIGHBOR: Final[float] = -0.80
REWARD_REINFORCE_CONTINENT_PUSH: Final[float] = 0.90

REWARD_ATTACK_FEWER_DICE: Final[float] = -1.25
REWARD_ATTACK_RATIO_SCALE: Final[float] = 2.00
REWARD_ATTACK_RATIO_CAP: Final[float] = 3.00
REWARD_ATTACK_RATIO_THRESHOLD: Final[float] = 1.50
REWARD_ATTACK_CONTINENT_DOMINATION: Final[float] = 0.80
REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN: Final[float] = 0.10
REWARD_ATTACK_CONTINENT_ADVANTAGE: Final[float] = 1.20
REWARD_ATTACK_ARMY_TRADE: Final[float] = 0.60
REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD: Final[float] = 1.25
REWARD_ATTACK_CONTINENT_CAPTURED: Final[float] = 4.00
REWARD_ATTACK_CONQUER_TERRITORY: Final[float] = 1.20
REWARD_ATTACK_CONQUER_WITH_CARD: Final[float] = 1.00
REWARD_ATTACK_CARD_TERRITORY_MATCH: Final[float] = 0.60
REWARD_ATTACK_STOP_WITHOUT_CARD: Final[float] = -2.00

REWARD_OCCUPY_FORWARD_MOMENTUM: Final[float] = 1.00

REWARD_FORTIFY_TOWARD_FRONTIER: Final[float] = 1.00
REWARD_FORTIFY_BALANCE_SCALE: Final[float] = 2.00
REWARD_FORTIFY_CONTINENT_PUSH: Final[float] = 0.80

REWARD_TERRITORY_DELTA: Final[float] = 1.00
REWARD_ARMY_DELTA_RELATIVE_SCALE: Final[float] = 0.10
REWARD_CONTINENT_DELTA_RELATIVE: Final[float] = 2.50
REWARD_TERMINAL_TIMEOUT: Final[float] = 0.0


__all__ = [
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "MAX_STEPS_PER_EPISODE",
    "TRAIN_OPPONENT_AGENT_KINDS",
    "BATCH_SIZE",
    "TRAIN_STEPS_PER_CALL",
    "TARGET_ALGORITHM",
    "LOSS_NAME",
    "GRAD_CLIP_MAX_NORM",
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
    "REWARD_TERMINAL_WIN",
    "REWARD_TERMINAL_LOSS",
    "REWARD_TRADE_IN_EARLY",
    "REWARD_TRADE_IN_TERRITORY_MATCH",
    "REWARD_REINFORCE_CONCENTRATION",
    "REWARD_REINFORCE_ATTACK_READINESS_SCALE",
    "REWARD_REINFORCE_RATIO_CAP",
    "REWARD_REINFORCE_NO_ENEMY_NEIGHBOR",
    "REWARD_REINFORCE_CONTINENT_PUSH",
    "REWARD_ATTACK_FEWER_DICE",
    "REWARD_ATTACK_RATIO_SCALE",
    "REWARD_ATTACK_RATIO_CAP",
    "REWARD_ATTACK_RATIO_THRESHOLD",
    "REWARD_ATTACK_CONTINENT_DOMINATION",
    "REWARD_ATTACK_CONTINENT_DOMINATION_MARGIN",
    "REWARD_ATTACK_CONTINENT_ADVANTAGE",
    "REWARD_ATTACK_ARMY_TRADE",
    "REWARD_ATTACK_ELIMINATE_OPPONENT_PER_CARD",
    "REWARD_ATTACK_CONTINENT_CAPTURED",
    "REWARD_ATTACK_CONQUER_TERRITORY",
    "REWARD_ATTACK_CONQUER_WITH_CARD",
    "REWARD_ATTACK_CARD_TERRITORY_MATCH",
    "REWARD_ATTACK_STOP_WITHOUT_CARD",
    "REWARD_OCCUPY_FORWARD_MOMENTUM",
    "REWARD_FORTIFY_TOWARD_FRONTIER",
    "REWARD_FORTIFY_BALANCE_SCALE",
    "REWARD_FORTIFY_CONTINENT_PUSH",
    "REWARD_TERRITORY_DELTA",
    "REWARD_ARMY_DELTA_RELATIVE_SCALE",
    "REWARD_CONTINENT_DELTA_RELATIVE",
]
