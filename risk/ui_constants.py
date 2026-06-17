"""All UI-only constants: colors, layout, fonts, pacing.

No game rules here — see `risk.constants` for those. Pure data, no pygame
import (so this module stays importable in headless/test contexts).
Import with `from risk.ui_constants import *`.
"""
from __future__ import annotations

from typing import Final

# --- shared player palette (risk/ui/input/init_screen.py, risk_map.py) ---

DEFAULT_PLAYER_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 255, 228),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 204, 204),
)


# --- board / HUD split layout (risk/app/view.py) -------------------------

HUD_WIDTH: Final[int] = 360
BOARD_BG_COLOR: Final[tuple[int, int, int]] = (10, 10, 14)


# --- AI pacing (risk/app/main.py) -----------------------------------------

DEFAULT_PLAY_AI_DELAY_MS: Final[int] = 600
DEFAULT_PLAY_MARKER_MS: Final[int] = 900


# --- last-action marker overlay + win screen (risk/app/loop.py) ----------

AI_MARKER_RGB: Final[tuple[int, int, int]] = (255, 230, 80)
WIN_SCREEN_BG: Final[tuple[int, int, int]] = (10, 10, 14)
WIN_SCREEN_FG: Final[tuple[int, int, int]] = (255, 215, 0)
WIN_SCREEN_SUB_FG: Final[tuple[int, int, int]] = (200, 200, 200)


# --- HUD side panel (risk/ui/render/panels.py) ----------------------------

HUD_BG: Final[tuple[int, int, int]] = (28, 28, 36)
HUD_FG: Final[tuple[int, int, int]] = (235, 235, 235)
HUD_ACCENT: Final[tuple[int, int, int]] = (255, 215, 0)

HUD_PANEL_BG: Final[tuple[int, int, int]] = (40, 44, 60)
HUD_BTN_BG: Final[tuple[int, int, int]] = (60, 70, 95)
HUD_BTN_BG_DISABLED: Final[tuple[int, int, int]] = (50, 52, 60)
HUD_BTN_PRIMARY: Final[tuple[int, int, int]] = (60, 130, 80)
HUD_BTN_PRIMARY_DISABLED: Final[tuple[int, int, int]] = (50, 70, 55)
HUD_BTN_FG: Final[tuple[int, int, int]] = (235, 235, 240)
HUD_BTN_FG_DISABLED: Final[tuple[int, int, int]] = (140, 145, 160)
HUD_LINK_FG: Final[tuple[int, int, int]] = (180, 195, 230)


# --- init / setup screen (risk/ui/render/init_screen_view.py) ------------

SETUP_BG: Final[tuple[int, int, int]] = (16, 18, 26)
SETUP_PANEL: Final[tuple[int, int, int]] = (28, 30, 42)
SETUP_PANEL_HI: Final[tuple[int, int, int]] = (40, 44, 60)
SETUP_TEXT: Final[tuple[int, int, int]] = (235, 235, 240)
SETUP_MUTED: Final[tuple[int, int, int]] = (140, 145, 160)
SETUP_ACCENT: Final[tuple[int, int, int]] = (255, 215, 0)
SETUP_OK: Final[tuple[int, int, int]] = (110, 200, 120)
SETUP_OK_DIM: Final[tuple[int, int, int]] = (60, 90, 70)
SETUP_DANGER: Final[tuple[int, int, int]] = (220, 90, 90)
SETUP_EDIT_BG: Final[tuple[int, int, int]] = (50, 55, 75)

SETUP_ROW_H: Final[int] = 56
SETUP_ROW_PAD: Final[int] = 8
SETUP_COLUMN_X: Final[dict[str, int]] = {
    "idx": 24,
    "name": 70,
    "color": 360,
    "kind": 440,
    "remove": 560,
}
SETUP_NAME_W: Final[int] = 270
SETUP_SWATCH_W: Final[int] = 60
SETUP_KIND_W: Final[int] = 130


__all__ = [
    "DEFAULT_PLAYER_COLORS",
    "HUD_WIDTH",
    "BOARD_BG_COLOR",
    "DEFAULT_PLAY_AI_DELAY_MS",
    "DEFAULT_PLAY_MARKER_MS",
    "AI_MARKER_RGB",
    "WIN_SCREEN_BG",
    "WIN_SCREEN_FG",
    "WIN_SCREEN_SUB_FG",
    "HUD_BG",
    "HUD_FG",
    "HUD_ACCENT",
    "HUD_PANEL_BG",
    "HUD_BTN_BG",
    "HUD_BTN_BG_DISABLED",
    "HUD_BTN_PRIMARY",
    "HUD_BTN_PRIMARY_DISABLED",
    "HUD_BTN_FG",
    "HUD_BTN_FG_DISABLED",
    "HUD_LINK_FG",
    "SETUP_BG",
    "SETUP_PANEL",
    "SETUP_PANEL_HI",
    "SETUP_TEXT",
    "SETUP_MUTED",
    "SETUP_ACCENT",
    "SETUP_OK",
    "SETUP_OK_DIM",
    "SETUP_DANGER",
    "SETUP_EDIT_BG",
    "SETUP_ROW_H",
    "SETUP_ROW_PAD",
    "SETUP_COLUMN_X",
    "SETUP_NAME_W",
    "SETUP_SWATCH_W",
    "SETUP_KIND_W",
]
