from __future__ import annotations

import pygame

from demos.graphics_state import build_demo_state
from risk.ui.render.risk_map import RiskMapRenderer


_PLAYER_NAMES = {
    0: "Harry",
    1: "Chihuahua",
    2: "CommandRater",
    3: "Player 3",
    4: "Player 4",
    5: "Player 5",
}

_PLAYER_CARD_COUNTS = {
    0: 4,
    1: 2,
    2: 5,
    3: 1,
    4: 3,
    5: 0,
}


def _fit_size(source_size: tuple[int, int], bounds: tuple[int, int]) -> tuple[int, int]:
    source_width, source_height = source_size
    bound_width, bound_height = bounds
    scale = min(bound_width / source_width, bound_height / source_height)
    return (max(1, int(source_width * scale)), max(1, int(source_height * scale)))


def _draw_text(
    canvas: pygame.Surface,
    text: str,
    position: tuple[int, int],
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> None:
    canvas.blit(font.render(text, True, color), position)


def _render_full_map_demo(renderer: RiskMapRenderer, owners: dict[str, int], armies: dict[str, int], width: int, height: int) -> pygame.Surface:
    return renderer.render(owners=owners, armies=armies, target_size=(width, height))


def _render_gameplay_demo(renderer: RiskMapRenderer, owners: dict[str, int], armies: dict[str, int], width: int, height: int) -> pygame.Surface:
    canvas = pygame.Surface((width, height))
    canvas.fill((245, 244, 240))

    header_height = min(90, max(64, height // 10))
    sidebar_width = min(330, max(260, width // 4))
    gutter = 24
    panel_top = header_height + gutter

    header_rect = pygame.Rect(12, 12, width - 24, header_height - 24)
    sidebar_rect = pygame.Rect(12, panel_top, sidebar_width, height - panel_top - 12)
    map_rect = pygame.Rect(sidebar_rect.right + gutter, panel_top, width - sidebar_rect.right - gutter - 12, height - panel_top - 12)

    pygame.draw.rect(canvas, (233, 233, 233), header_rect, border_radius=6)
    pygame.draw.rect(canvas, (249, 249, 249), sidebar_rect, border_radius=4)

    title_font = pygame.font.SysFont("calibri", 34)
    subtitle_font = pygame.font.SysFont("calibri", 18)
    section_font = pygame.font.SysFont("calibri", 24)
    body_font = pygame.font.SysFont("calibri", 18)
    small_font = pygame.font.SysFont("calibri", 15)
    button_font = pygame.font.SysFont("calibri", 16)

    _draw_text(canvas, "World Domination (Risk)", (header_rect.x + 28, header_rect.y + 10), title_font, (35, 35, 35))
    _draw_text(
        canvas,
        "A World Domination Game based on the board game called Risk.",
        (header_rect.x + 28, header_rect.y + 46),
        subtitle_font,
        (60, 60, 60),
    )

    sidebar_x = sidebar_rect.x + 14
    cursor_y = sidebar_rect.y + 16
    active_player_ids = sorted({owner for owner in owners.values() if isinstance(owner, int)})

    _draw_text(canvas, "You are the host!", (sidebar_x, cursor_y), body_font, (40, 40, 40))
    cursor_y += 42
    _draw_text(canvas, "Players", (sidebar_x, cursor_y), section_font, (35, 35, 35))
    cursor_y += 38

    for player_id in active_player_ids:
        player_name = _PLAYER_NAMES.get(player_id, f"Player {player_id}")
        player_color = renderer.DEFAULT_PLAYER_COLORS.get(player_id, (80, 80, 80))
        card_count = _PLAYER_CARD_COUNTS.get(player_id, 0)
        _draw_text(canvas, f"Player {player_id}:", (sidebar_x, cursor_y), body_font, (100, 100, 100))
        _draw_text(canvas, player_name, (sidebar_x + 82, cursor_y), body_font, player_color)
        _draw_text(canvas, f"Cards: {card_count}", (sidebar_x + 200, cursor_y), small_font, (110, 110, 110))
        if player_id == 0:
            _draw_text(canvas, "(Myself)", (sidebar_x + 148, cursor_y), small_font, (120, 120, 120))
        cursor_y += 28

    cursor_y += 20
    _draw_text(canvas, "Your Turn (STAGE_BATTLES)", (sidebar_x, cursor_y), section_font, (45, 45, 45))
    cursor_y += 46
    _draw_text(canvas, "Attack from Country:", (sidebar_x, cursor_y), body_font, (70, 70, 70))
    _draw_text(canvas, "Southern Europe", (sidebar_x + 160, cursor_y), body_font, (92, 223, 236))
    cursor_y += 28
    _draw_text(canvas, "(Remove)", (sidebar_x, cursor_y), small_font, (120, 160, 180))
    cursor_y += 40
    _draw_text(canvas, "Country to Attack:", (sidebar_x, cursor_y), body_font, (70, 70, 70))
    _draw_text(canvas, "Middle East", (sidebar_x + 140, cursor_y), body_font, (73, 153, 51))
    cursor_y += 28
    _draw_text(canvas, "(Remove)", (sidebar_x + 238, cursor_y - 1), small_font, (120, 160, 180))
    cursor_y += 36
    _draw_text(canvas, "Attack with:", (sidebar_x, cursor_y), body_font, (45, 45, 45))

    select_rect = pygame.Rect(sidebar_x + 150, cursor_y - 6, 84, 40)
    pygame.draw.rect(canvas, (255, 255, 255), select_rect, border_radius=4)
    pygame.draw.rect(canvas, (200, 200, 200), select_rect, 1, border_radius=4)
    _draw_text(canvas, "3", (select_rect.x + 22, select_rect.y + 9), body_font, (70, 70, 70))
    pygame.draw.polygon(
        canvas,
        (90, 90, 90),
        [(select_rect.right - 18, select_rect.y + 16), (select_rect.right - 10, select_rect.y + 16), (select_rect.right - 14, select_rect.y + 24)],
    )

    attack_button_y = min(cursor_y + 64, sidebar_rect.bottom - 44 - 52 - 44)
    attack_button = pygame.Rect(sidebar_x, attack_button_y, 82, 40)
    pygame.draw.rect(canvas, (227, 92, 77), attack_button, border_radius=4)
    _draw_text(canvas, "Attack!", (attack_button.x + 15, attack_button.y + 10), button_font, (255, 255, 255))

    end_button = pygame.Rect(sidebar_x, sidebar_rect.bottom - 44 - 20, 138, 44)
    pygame.draw.rect(canvas, (91, 189, 225), end_button, border_radius=4)
    _draw_text(canvas, "End Attack Phrase", (end_button.x + 14, end_button.y + 12), button_font, (255, 255, 255))

    map_surface = renderer.render(owners=owners, armies=armies)
    fitted_size = _fit_size(map_surface.get_size(), (map_rect.width, map_rect.height))
    fitted_map = pygame.transform.smoothscale(map_surface, fitted_size)
    map_position = (
        map_rect.x + (map_rect.width - fitted_size[0]) // 2,
        map_rect.y + (map_rect.height - fitted_size[1]) // 2,
    )
    canvas.blit(fitted_map, map_position)
    return canvas


def run_map_demo(width: int, height: int, save_path: str = "", layout: str = "full") -> None:
    pygame.init()
    renderer = RiskMapRenderer()
    owners, armies = build_demo_state(renderer)

    if layout == "gameplay":
        rendered = _render_gameplay_demo(renderer, owners, armies, width, height)
    else:
        rendered = _render_full_map_demo(renderer, owners, armies, width, height)

    if save_path:
        pygame.image.save(rendered, save_path)
        pygame.quit()
        return

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Risk map pygame renderer ({layout})")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(rendered, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()