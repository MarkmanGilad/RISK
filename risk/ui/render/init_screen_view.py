"""Pygame view + event loop for the init / setup screen.

Visual setup wrapper around the headless `InitScreenState`. Returns a
`GameSettings` when the user clicks **Start Game**, or `None` if they
close the window.

Interaction:
- Click `-` / `+` to change player count (3..6).
- Click a name cell to edit it (type, Enter / Esc to commit, Backspace
  to delete).
- Click a color swatch to cycle through the palette (skipping colors
  already in use by other seats).
- Click the seat-type cell to cycle through Human / Random / Raider / Sentinel / Empire.
- Click **Start Game** to begin; the button is greyed out while the
  config is invalid (duplicate colors, blank names).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pygame

from risk.game.settings import GameSettings
from risk.game.player import AGENT_KIND_LABELS
from risk.ui.input.init_screen import DEFAULT_COLORS, InitScreenState
from risk.ui_constants import (
    SETUP_ACCENT as ACCENT,
    SETUP_BG as BG,
    SETUP_COLUMN_X as COLUMN_X,
    SETUP_DANGER as DANGER,
    SETUP_EDIT_BG as EDIT_BG,
    SETUP_KIND_W as KIND_W,
    SETUP_MUTED as MUTED,
    SETUP_NAME_W as NAME_W,
    SETUP_OK as OK,
    SETUP_OK_DIM as OK_DIM,
    SETUP_PANEL as PANEL,
    SETUP_PANEL_HI as PANEL_HI,
    SETUP_ROW_H as ROW_H,
    SETUP_ROW_PAD as ROW_PAD,
    SETUP_SWATCH_W as SWATCH_W,
    SETUP_TEXT as TEXT,
)


class _Button:
    def __init__(self, rect: pygame.Rect, label: str, enabled: bool = True) -> None:
        self.rect = rect
        self.label = label
        self.enabled = enabled

    def draw(self, surface: pygame.Surface, font: pygame.font.Font,
             bg: tuple[int, int, int], fg: tuple[int, int, int]) -> None:
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, MUTED, self.rect, 1, border_radius=6)
        label = font.render(self.label, True, fg)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


def _next_unused_color(current: tuple[int, int, int],
                       used: set[tuple[int, int, int]]) -> tuple[int, int, int]:
    palette = list(DEFAULT_COLORS)
    try:
        start = palette.index(current)
    except ValueError:
        start = -1
    for k in range(1, len(palette) + 1):
        candidate = palette[(start + k) % len(palette)]
        if candidate == current or candidate not in used:
            return candidate
    return current


def run_init_screen(screen: pygame.Surface) -> Optional[object]:
    """Drive the init screen until the user starts a game or quits.

    Returns the chosen `GameSettings`, or `None` if the user quit.
    """
    pygame.font.init()
    title_font = pygame.font.SysFont("consolas", 32, bold=True)
    head_font = pygame.font.SysFont("consolas", 18, bold=True)
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 16)

    state = InitScreenState()
    editing: Optional[int] = None  # index of name-cell being typed into
    edit_buffer: str = ""
    clock = pygame.time.Clock()
    learned_error = ""

    def setup_result():
        settings = state.build_settings(seed=None)
        selections = {seat: dict(selection) for seat, selection in state.learned_selections.items()}
        return {"settings": settings, "selections": selections,
                "labels": {seat: s["label"] or s["agent_kind"] for seat, s in selections.items()}}

    def refresh_learned_validation() -> None:
        nonlocal learned_error
        ok, _ = state.can_start()
        if not ok or not state.learned_selections:
            learned_error = ""
            return
        from risk.app.learned_agent_play import validate_selections

        learned_error = validate_selections(state.build_settings(seed=None), state.learned_selections)

    def choose_path(directory: bool) -> str:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        try:
            initial = str(Path.cwd() / "Checkpoints")
            if directory:
                return filedialog.askdirectory(initialdir=initial, parent=root)
            return filedialog.askopenfilename(
                initialdir=initial, parent=root, filetypes=[("Policy files", "*.pt"), ("All files", "*.*")]
            )
        finally:
            root.destroy()

    def used_colors(exclude: int = -1) -> set[tuple[int, int, int]]:
        return {s.color for i, s in enumerate(state.seats) if i != exclude}

    while True:
        w, h = screen.get_size()
        screen.fill(BG)

        # --- title -----------------------------------------------------
        title = title_font.render("Risk \u2014 New Game", True, ACCENT)
        screen.blit(title, (40, 30))
        subtitle = small.render(
            "Click cells to edit. Enter commits, Esc cancels.", True, MUTED
        )
        screen.blit(subtitle, (40, 75))

        # --- count controls --------------------------------------------
        count_y = 110
        count_label = head_font.render(f"Players: {state.player_count}", True, TEXT)
        screen.blit(count_label, (40, count_y))
        minus = _Button(
            pygame.Rect(220, count_y - 4, 36, 30),
            "-",
            enabled=state.player_count > 3,
        )
        plus = _Button(
            pygame.Rect(264, count_y - 4, 36, 30),
            "+",
            enabled=state.player_count < 6,
        )
        minus.draw(screen, head_font, PANEL_HI if minus.enabled else PANEL,
                   TEXT if minus.enabled else MUTED)
        plus.draw(screen, head_font, PANEL_HI if plus.enabled else PANEL,
                  TEXT if plus.enabled else MUTED)

        # --- column headers --------------------------------------------
        header_y = 160
        screen.blit(small.render("#", True, MUTED), (COLUMN_X["idx"], header_y))
        screen.blit(small.render("Name (click to edit)", True, MUTED),
                    (COLUMN_X["name"], header_y))
        screen.blit(small.render("Color", True, MUTED), (COLUMN_X["color"], header_y))
        screen.blit(small.render("Type", True, MUTED), (COLUMN_X["kind"], header_y))

        # --- seat rows -------------------------------------------------
        rows_top = 185
        name_rects: list[pygame.Rect] = []
        color_rects: list[pygame.Rect] = []
        kind_rects: list[pygame.Rect] = []
        learned_rects: list[tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect] | None] = []

        for i, seat in enumerate(state.seats):
            y = rows_top + i * (ROW_H + ROW_PAD)
            row_rect = pygame.Rect(20, y, w - 40, ROW_H)
            pygame.draw.rect(screen, PANEL, row_rect, border_radius=8)

            # Index.
            screen.blit(
                head_font.render(str(i + 1), True, ACCENT),
                (COLUMN_X["idx"], y + ROW_H // 2 - head_font.get_height() // 2),
            )

            # Name cell.
            name_rect = pygame.Rect(COLUMN_X["name"], y + 12, NAME_W, ROW_H - 24)
            name_rects.append(name_rect)
            is_editing = editing == i
            pygame.draw.rect(screen, EDIT_BG if is_editing else PANEL_HI, name_rect, border_radius=4)
            border = ACCENT if is_editing else MUTED
            pygame.draw.rect(screen, border, name_rect, 1, border_radius=4)
            display = edit_buffer if is_editing else seat.name
            if is_editing and (pygame.time.get_ticks() // 500) % 2 == 0:
                display = display + "|"
            screen.blit(
                font.render(display, True, TEXT),
                (name_rect.x + 8, name_rect.y + (name_rect.h - font.get_height()) // 2),
            )

            # Color swatch.
            color_rect = pygame.Rect(COLUMN_X["color"], y + 12, SWATCH_W, ROW_H - 24)
            color_rects.append(color_rect)
            pygame.draw.rect(screen, seat.color, color_rect, border_radius=4)
            pygame.draw.rect(screen, MUTED, color_rect, 1, border_radius=4)

            # Kind button.
            kind_rect = pygame.Rect(COLUMN_X["kind"], y + 12, KIND_W, ROW_H - 24)
            kind_rects.append(kind_rect)
            pygame.draw.rect(screen, PANEL_HI, kind_rect, border_radius=4)
            pygame.draw.rect(screen, MUTED, kind_rect, 1, border_radius=4)
            kind_label = font.render(
                "AI Agent" if state.is_learned(i) else AGENT_KIND_LABELS.get(seat.agent_kind, seat.agent_kind.title()),
                True,
                TEXT,
            )
            if state.is_learned(i):
                file_rect = pygame.Rect(COLUMN_X["kind"] + KIND_W + 12, y + 12, 90, ROW_H - 24)
                dir_rect = pygame.Rect(file_rect.right + 6, y + 12, 70, ROW_H - 24)
                preset_rect = pygame.Rect(dir_rect.right + 6, y + 12, 90, ROW_H - 24)
                algo_rect = pygame.Rect(preset_rect.right + 6, y + 12, 110, ROW_H - 24)
                selection = state.learned_selections[i]
                for rect, label in ((file_rect, "File..."), (dir_rect, "Folder..."), (preset_rect, "Best DQN"),
                                    (algo_rect, selection["agent_kind"].replace("_", " "))):
                    pygame.draw.rect(screen, PANEL_HI, rect, border_radius=4)
                    pygame.draw.rect(screen, MUTED, rect, 1, border_radius=4)
                    text = small.render(label, True, TEXT)
                    screen.blit(text, text.get_rect(center=rect.center))
                if selection["label"]:
                    screen.blit(small.render(selection["label"][:35], True, MUTED), (algo_rect.right + 10, y + 18))
                learned_rects.append((file_rect, dir_rect, preset_rect, algo_rect))
            else:
                learned_rects.append(None)
            screen.blit(kind_label, kind_label.get_rect(center=kind_rect.center))

        # --- start button + validation ---------------------------------
        ok, why = state.can_start()
        if learned_error:
            ok, why = False, learned_error
        elif any(not selection["checkpoint"] for selection in state.learned_selections.values()):
            ok, why = False, "Choose a model for every Learned Agent."
        start_rect = pygame.Rect(w - 220, h - 80, 180, 50)
        start_btn = _Button(start_rect, "Start Game", enabled=ok)
        start_btn.draw(
            screen, head_font, OK if ok else OK_DIM, TEXT if ok else MUTED
        )
        if not ok:
            screen.blit(small.render(why, True, DANGER), (40, h - 60))

        pygame.display.flip()
        clock.tick(60)

        # --- events ----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if editing is not None:
                    if event.key == pygame.K_RETURN:
                        try:
                            state.set_name(editing, edit_buffer)
                        except ValueError:
                            pass
                        editing = None
                    elif event.key == pygame.K_ESCAPE:
                        editing = None
                    elif event.key == pygame.K_BACKSPACE:
                        edit_buffer = edit_buffer[:-1]
                    elif event.unicode and event.unicode.isprintable() and len(edit_buffer) < 16:
                        edit_buffer += event.unicode
                else:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_RETURN and ok:
                        return setup_result()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if editing is not None:
                    # Commit current edit on click outside.
                    if not name_rects[editing].collidepoint(pos):
                        try:
                            state.set_name(editing, edit_buffer)
                        except ValueError:
                            pass
                        editing = None

                if minus.hit(pos):
                    state.set_player_count(state.player_count - 1)
                    if editing is not None and editing >= state.player_count:
                        editing = None
                    refresh_learned_validation()
                    continue
                if plus.hit(pos):
                    state.set_player_count(state.player_count + 1)
                    # Avoid color clash on newly added seat.
                    used = used_colors(exclude=state.player_count - 1)
                    if state.seats[-1].color in used:
                        state.seats[-1].color = _next_unused_color(
                            state.seats[-1].color, used
                        )
                    refresh_learned_validation()
                    continue

                if start_btn.hit(pos):
                    return setup_result()

                for i, r in enumerate(name_rects):
                    if r.collidepoint(pos):
                        editing = i
                        edit_buffer = state.seats[i].name
                        break
                else:
                    for i, r in enumerate(color_rects):
                        if r.collidepoint(pos):
                            state.seats[i].color = _next_unused_color(
                                state.seats[i].color, used_colors(exclude=i)
                            )
                            break
                    else:
                        for i, r in enumerate(kind_rects):
                            if r.collidepoint(pos):
                                state.next_visible_agent_kind(i)
                                refresh_learned_validation()
                                break
                        else:
                            for i, controls in enumerate(learned_rects):
                                if controls is None:
                                    continue
                                file_rect, dir_rect, preset_rect, algo_rect = controls
                                selection = state.learned_selections[i]
                                if file_rect.collidepoint(pos) or dir_rect.collidepoint(pos):
                                    checkpoint = choose_path(dir_rect.collidepoint(pos))
                                    if checkpoint:
                                        state.set_learned_selection(
                                            i, source="manual", checkpoint=checkpoint,
                                            agent_kind=selection["agent_kind"],
                                            label=Path(checkpoint).name,
                                        )
                                        refresh_learned_validation()
                                    break
                                if preset_rect.collidepoint(pos):
                                    from risk.app.learned_agent_play import load_presets

                                    try:
                                        presets = load_presets()
                                        if not presets:
                                            learned_error = "No predefined best models are configured."
                                        else:
                                            current = selection.get("preset_id", "")
                                            index = next((j for j, p in enumerate(presets) if p["id"] == current), -1)
                                            preset = presets[(index + 1) % len(presets)]
                                            state.set_learned_selection(
                                                i, source="preset", checkpoint=preset["checkpoint"],
                                                agent_kind=preset["agent_kind"], label=preset["label"],
                                                preset_id=preset["id"],
                                            )
                                            refresh_learned_validation()
                                    except ValueError as exc:
                                        learned_error = str(exc)
                                    break
                                if algo_rect.collidepoint(pos) and selection["source"] != "preset":
                                    kinds = ("DQN", "Dueling_DQN", "PPO")
                                    next_kind = kinds[(kinds.index(selection["agent_kind"]) + 1) % len(kinds)]
                                    state.set_learned_selection(
                                        i, source=selection["source"] or "manual",
                                        checkpoint=selection["checkpoint"], agent_kind=next_kind,
                                        label=selection["label"], preset_id=selection["preset_id"],
                                    )
                                    refresh_learned_validation()
                                    break


__all__ = ["run_init_screen"]
