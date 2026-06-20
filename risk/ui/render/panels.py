"""Side-panel HUD rendering for the live game.

Pure pygame drawing — no game logic. Reads from `State + GameSettings`
and renders text labels into a passed-in surface.
"""
from __future__ import annotations

from typing import Optional

import pygame

from risk.constants import card_set_value
from risk.game.phase import Phase
from risk.game.player import AGENT_KIND_LABELS, Player
from risk.game.settings import GameSettings
from risk.game.state import State
from risk.ui.input.human_input import HudActionPanelModel
from risk.ui_constants import (
    HUD_ACCENT,
    HUD_BG,
    HUD_BTN_BG,
    HUD_BTN_BG_DISABLED,
    HUD_BTN_FG,
    HUD_BTN_FG_DISABLED,
    HUD_BTN_PRIMARY,
    HUD_BTN_PRIMARY_DISABLED,
    HUD_FG,
    HUD_LINK_FG,
    HUD_PANEL_BG,
)


# Encoded hit-test region id format:
#   "button:<button_id>"
#   "field:<field_id>"             (no payload)
#   "field:<field_id>:<payload>"   (payload is a territory name)
# `main.py` parses this back into controller method calls.


class HudPanel:
    BG = HUD_BG
    FG = HUD_FG
    ACCENT = HUD_ACCENT

    def __init__(self, font: Optional[pygame.font.Font] = None) -> None:
        if font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont("consolas", 16)
        self.font = font
        self.title_font = pygame.font.SysFont("consolas", 20, bold=True)

    def render(
        self,
        target: pygame.Surface,
        rect: pygame.Rect,
        state: State,
        settings: GameSettings,
        message: str = "",
        last_action_text: str = "",
        last_action_lines: tuple[str, ...] = (),
    ) -> None:
        target.fill(self.BG, rect)
        x, y = rect.x + 12, rect.y + 12
        line_h = self.font.get_linesize()

        pid = state.current_player_index
        player = settings.players[pid]

        # Title.
        title = self.title_font.render("Risk", True, self.ACCENT)
        target.blit(title, (x, y))
        y += self.title_font.get_linesize() + 4

        # Current player.
        swatch = pygame.Rect(x, y + 3, 14, 14)
        pygame.draw.rect(target, player.color, swatch)
        pygame.draw.rect(target, self.FG, swatch, 1)
        kind_label = AGENT_KIND_LABELS.get(player.agent_kind, player.agent_kind.title())
        label = self.font.render(
            f"{player.name}  ({kind_label})", True, self.FG
        )
        target.blit(label, (x + 22, y))
        y += line_h + 4

        # Phase + budget.
        phase_label = Phase(state.phase).name
        target.blit(self.font.render(f"Phase: {phase_label}", True, self.FG), (x, y))
        y += line_h
        if state.phase is Phase.REINFORCE_PLACE:
            target.blit(
                self.font.render(f"Reinforce: {state.reinforcement_budget}", True, self.ACCENT),
                (x, y),
            )
            y += line_h

        # Per-player status table.
        y += 8
        target.blit(self.font.render("Players:", True, self.FG), (x, y))
        y += line_h
        for p in settings.players:
            n_owned = sum(1 for o in state.owners if o == p.id)
            n_armies = sum(a for i, a in enumerate(state.armies) if state.owners[i] == p.id)
            n_cards = len(state.hands[p.id])
            tag = "X" if p.id in state.eliminated else ("*" if p.id == pid else " ")
            sw = pygame.Rect(x, y + 3, 10, 10)
            pygame.draw.rect(target, p.color, sw)
            p_kind_label = AGENT_KIND_LABELS.get(p.agent_kind, p.agent_kind.title())
            line = self.font.render(
                f"{tag} {p.name[:6]:<6} {p_kind_label:<8} t={n_owned:>2} a={n_armies:>3} c={n_cards}",
                True,
                self.FG,
            )
            target.blit(line, (x + 16, y))
            y += line_h

        # Persistent next-trade value.
        y += 6
        next_trade = card_set_value(state.cards_traded_in_count)
        trade_surf = self.font.render(
            f"Next trade: {next_trade} armies", True, self.ACCENT
        )
        target.blit(trade_surf, (x, y))
        y += line_h

        # Message footer.
        if message:
            y = rect.bottom - line_h - 12
            msg = self.font.render(message, True, self.ACCENT)
            target.blit(msg, (x, y))

    def render_last_move(
        self,
        target: pygame.Surface,
        rect: pygame.Rect,
        last_action_text: str = "",
        last_action_lines: tuple[str, ...] = (),
    ) -> None:
        """Draw the 'Last move:' block, anchored to the bottom strip.

        It stays in the pane until the next action replaces it.
        """
        lines = last_action_lines or ((last_action_text,) if last_action_text else ())
        if not lines:
            return
        x = rect.x + 12
        line_h = self.font.get_linesize()
        target.blit(self.font.render("Last move:", True, self.FG), (x, rect.y))
        y = rect.y + line_h
        for ln in lines:
            target.blit(self.font.render(ln[:48], True, self.LINK_FG), (x, y))
            y += line_h


    # --- action panel for human turns ------------------------------------

    PANEL_BG = HUD_PANEL_BG
    BTN_BG = HUD_BTN_BG
    BTN_BG_DIS = HUD_BTN_BG_DISABLED
    BTN_PRIMARY = HUD_BTN_PRIMARY
    BTN_PRIMARY_DIS = HUD_BTN_PRIMARY_DISABLED
    BTN_FG = HUD_BTN_FG
    BTN_FG_DIS = HUD_BTN_FG_DISABLED
    LINK_FG = HUD_LINK_FG

    def render_action_panel(
        self,
        target: pygame.Surface,
        rect: pygame.Rect,
        model: HudActionPanelModel,
    ) -> list[tuple[pygame.Rect, str]]:
        """Draw the per-turn action panel; return clickable (rect, id_str)."""
        regions: list[tuple[pygame.Rect, str]] = []
        if not model.is_active:
            return regions

        x = rect.x + 10
        width = rect.w - 20
        y = rect.y + 8

        # Background card.
        pygame.draw.rect(target, self.PANEL_BG, rect, border_radius=6)

        title = self.title_font.render(model.header, True, self.ACCENT)
        target.blit(title, (x, y))
        y += self.title_font.get_linesize() + 4

        line_h = self.font.get_linesize()

        for info in model.info_lines:
            target.blit(self.font.render(info, True, self.FG), (x, y))
            y += line_h

        # Reinforce placement rows: "Terr  +N   (-) (+) (Clear)"
        for terr, count in model.placement_rows:
            label = self.font.render(f"{terr[:12]:<12} +{count}", True, self.FG)
            target.blit(label, (x, y))
            btn_y = y - 2
            dec = pygame.Rect(x + width - 96, btn_y, 26, line_h + 4)
            inc = pygame.Rect(x + width - 66, btn_y, 26, line_h + 4)
            clr = pygame.Rect(x + width - 36, btn_y, 32, line_h + 4)
            self._draw_small_btn(target, dec, "-")
            self._draw_small_btn(target, inc, "+")
            self._draw_small_btn(target, clr, "X")
            regions.append((dec, f"field:placements_dec:{terr}"))
            regions.append((inc, f"field:placements_inc:{terr}"))
            regions.append((clr, f"field:placements_clear:{terr}"))
            y += line_h + 6

        # Attack / fortify selection rows.
        if model.from_label is not None or model.to_label is not None:
            y += 4
            y = self._draw_field(
                target, x, y, width, "From",
                model.from_label or "(click own territory)",
                clear_id=self._clear_id_for("from", model.header),
                regions=regions,
                has_clear=model.from_label is not None,
            )
            y = self._draw_field(
                target, x, y, width, "To",
                model.to_label or "(click target)",
                clear_id=self._clear_id_for("to", model.header),
                regions=regions,
                has_clear=model.to_label is not None,
            )

        # Dice spinner (attack).
        if model.dice is not None:
            y += 4
            y = self._draw_spinner(
                target, x, y, width, "Dice",
                model.dice, model.dice_max,
                dec_id="field:dice_dec", inc_id="field:dice_inc",
                regions=regions,
            )

        # Count spinner (fortify).
        if model.count is not None:
            y += 4
            y = self._draw_spinner(
                target, x, y, width, "Count",
                model.count, model.count_max,
                dec_id="field:count_dec", inc_id="field:count_inc",
                regions=regions,
            )

        # Card screen (toggled by the "Cards" button).
        if model.show_cards:
            y += 8
            target.blit(self.title_font.render("Your Cards", True, self.ACCENT), (x, y))
            y += self.title_font.get_linesize() + 2
            if not model.cards:
                target.blit(self.font.render("(no cards)", True, self.FG), (x, y))
                y += line_h
            for i, (label, selected) in enumerate(model.cards):
                row = pygame.Rect(x, y - 2, width, line_h + 4)
                if selected:
                    pygame.draw.rect(target, self.BTN_PRIMARY, row, border_radius=3)
                mark = "[x]" if selected else "[ ]"
                s = self.font.render(f"{mark} {label}", True, self.FG)
                target.blit(s, (x + 4, y))
                regions.append((row, f"field:card_toggle:{i}"))
                y += line_h + 4
            y += 2

        # Footer buttons.
        y += 8
        for btn in model.buttons:
            btn_rect = pygame.Rect(x, y, width, 32)
            bg = (self.BTN_PRIMARY if btn.primary else self.BTN_BG) if btn.enabled \
                else (self.BTN_PRIMARY_DIS if btn.primary else self.BTN_BG_DIS)
            fg = self.BTN_FG if btn.enabled else self.BTN_FG_DIS
            pygame.draw.rect(target, bg, btn_rect, border_radius=4)
            label = self.font.render(btn.label, True, fg)
            target.blit(label, label.get_rect(center=btn_rect.center))
            if btn.enabled:
                regions.append((btn_rect, f"button:{btn.id}"))
            y += 38

        return regions

    # --- internal drawing helpers ---------------------------------------

    def _clear_id_for(self, slot: str, header: str) -> str:
        # Slot is "from" or "to"; header tells us attack vs fortify.
        if "ATTACK" in header:
            return f"field:attack_{slot}_clear"
        return f"field:fortify_{slot}_clear"

    def _draw_small_btn(
        self, target: pygame.Surface, r: pygame.Rect, label: str
    ) -> None:
        pygame.draw.rect(target, self.BTN_BG, r, border_radius=3)
        s = self.font.render(label, True, self.BTN_FG)
        target.blit(s, s.get_rect(center=r.center))

    def _draw_field(
        self,
        target: pygame.Surface,
        x: int,
        y: int,
        width: int,
        label: str,
        value: str,
        clear_id: str,
        regions: list,
        has_clear: bool,
    ) -> int:
        line_h = self.font.get_linesize()
        head = self.font.render(f"{label}:", True, self.LINK_FG)
        target.blit(head, (x, y))
        val_x = x + head.get_width() + 6
        val = self.font.render(value[: max(1, (width - 100) // 8)], True, self.FG)
        target.blit(val, (val_x, y))
        if has_clear:
            clr = pygame.Rect(x + width - 50, y - 2, 50, line_h + 4)
            self._draw_small_btn(target, clr, "Clear")
            regions.append((clr, clear_id))
        return y + line_h + 4

    def _draw_spinner(
        self,
        target: pygame.Surface,
        x: int,
        y: int,
        width: int,
        label: str,
        value: int,
        max_value: int,
        dec_id: str,
        inc_id: str,
        regions: list,
    ) -> int:
        line_h = self.font.get_linesize()
        head = self.font.render(f"{label}:", True, self.LINK_FG)
        target.blit(head, (x, y))
        dec = pygame.Rect(x + width - 80, y - 2, 24, line_h + 4)
        val_rect = pygame.Rect(x + width - 54, y - 2, 30, line_h + 4)
        inc = pygame.Rect(x + width - 22, y - 2, 22, line_h + 4)
        self._draw_small_btn(target, dec, "-")
        pygame.draw.rect(target, self.BG, val_rect, border_radius=3)
        v = self.font.render(f"{value}/{max_value}", True, self.FG)
        target.blit(v, v.get_rect(center=val_rect.center))
        self._draw_small_btn(target, inc, "+")
        regions.append((dec, dec_id))
        regions.append((inc, inc_id))
        return y + line_h + 4


__all__ = ["HudPanel"]
