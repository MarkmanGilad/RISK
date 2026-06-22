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
MAX_STEPS_PER_EPISODE: Final[int] = 20_000

# --- Replay / gradient update -----------------------------------------------

BATCH_SIZE: Final[int] = 32
TRAIN_STEPS_PER_CALL: Final[int] = 1  # gradient steps taken each time

# --- Run control -------------------------------------------------------------

# Default episode count when running `python -m risk.learning.trainer`.
TRAIN_EPISODES: Final[int] = 200

# Print one progress line every N episodes.
PROGRESS_EVERY: Final[int] = 10

# --- Checkpointing ----------------------------------------------------------

CHECKPOINT_DIR: Final[str] = "Checkpoints"
CHECKPOINT_AFTER: Final[int] = 200   # don't checkpoint before this episode
CHECKPOINT_EVERY: Final[int] = 200   # save every N episodes after that


__all__ = [
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "MAX_STEPS_PER_EPISODE",
    "BATCH_SIZE",
    "TRAIN_STEPS_PER_CALL",
    "TRAIN_EPISODES",
    "PROGRESS_EVERY",
    "CHECKPOINT_DIR",
    "CHECKPOINT_AFTER",
    "CHECKPOINT_EVERY",
]
