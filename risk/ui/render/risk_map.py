from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pygame
from svg.path import Arc, CubicBezier, Line, QuadraticBezier, parse_path

_CURVED_SEGMENTS = (Arc, CubicBezier, QuadraticBezier)


class RiskMapRenderer:
    DEFAULT_PLAYER_COLORS = {
        0: (0, 255, 228),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
        4: (255, 0, 255),
        5: (255, 204, 204),
    }

    def __init__(self, assets_dir: str | Path | None = None, fill_alpha: int = 102, curve_steps: int = 24) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.assets_dir = Path(assets_dir or self.root_dir / "Assets" / "RiskMap")
        self.fill_alpha = fill_alpha
        self.curve_steps = curve_steps

        self.data = self._load_data()
        self.base_map = pygame.image.load(str(self.assets_dir / "map_grey_new.jpg"))
        self.names_sprite = pygame.image.load(str(self.assets_dir / "names.png"))
        self.label_surfaces = self._build_label_surfaces()
        self.territory_polygons = self._build_territory_polygons()
        self.territory_ids = list(self.data["territory_names"].keys())

    def render(
        self,
        owners: dict[str, int | tuple[int, int, int]] | None = None,
        armies: dict[str, int] | None = None,
        target_size: tuple[int, int] | None = None,
        highlights: dict[str, tuple[int, int, int]] | None = None,
    ) -> pygame.Surface:
        owners = owners or {}
        armies = armies or {}
        highlights = highlights or {}

        canvas = pygame.Surface(self.base_map.get_size(), pygame.SRCALPHA)
        canvas.blit(self.base_map, (0, 0))

        overlay = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        for territory_id, polygons in self.territory_polygons.items():
            owner = owners.get(territory_id)
            if owner is None:
                continue
            fill = (*self._resolve_color(owner), self.fill_alpha)
            for polygon in polygons:
                if len(polygon) >= 3:
                    pygame.draw.polygon(overlay, fill, polygon)
        canvas.blit(overlay, (0, 0))

        # Action markers (drawn over the ownership tint but below labels).
        if highlights:
            marker = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
            for territory_id, color in highlights.items():
                polygons = self.territory_polygons.get(territory_id)
                if not polygons:
                    continue
                rgb = tuple(int(c) for c in color[:3])
                fill = (*rgb, 110)
                for polygon in polygons:
                    if len(polygon) >= 3:
                        pygame.draw.polygon(marker, fill, polygon)
                        pygame.draw.polygon(marker, (*rgb, 255), polygon, width=6)
            canvas.blit(marker, (0, 0))

        for territory_id in self.territory_ids:
            label_surface = self.label_surfaces[territory_id]
            destination = self.data["font_destination_coords"][territory_id]
            canvas.blit(label_surface, (destination["x"], destination["y"]))

        self._draw_continent_labels(canvas)
        self._draw_armies(canvas, armies)

        if target_size and target_size != canvas.get_size():
            return pygame.transform.smoothscale(canvas, target_size)
        return canvas

    def all_territory_ids(self) -> Iterable[str]:
        return tuple(self.territory_ids)

    def _load_data(self) -> dict:
        data_path = self.assets_dir / "risk_map_data.json"
        return json.loads(data_path.read_text(encoding="utf-8"))

    def _build_label_surfaces(self) -> dict[str, pygame.Surface]:
        labels: dict[str, pygame.Surface] = {}
        for territory_id, coords in self.data["font_sprite_coords"].items():
            rect = pygame.Rect(coords["sx"], coords["sy"], coords["sWidth"], coords["sHeight"])
            labels[territory_id] = self.names_sprite.subsurface(rect).copy()
        return labels

    def _build_territory_polygons(self) -> dict[str, list[list[tuple[int, int]]]]:
        polygons: dict[str, list[list[tuple[int, int]]]] = {}
        for territory_id, path_data in self.data["territory_paths"].items():
            polygons[territory_id] = self._path_to_polygons(path_data)
        return polygons

    def _path_to_polygons(self, path_data: str) -> list[list[tuple[int, int]]]:
        parsed = parse_path(path_data)
        subpaths = parsed.continuous_subpaths() if hasattr(parsed, "continuous_subpaths") else [parsed]
        polygons: list[list[tuple[int, int]]] = []

        for subpath in subpaths:
            polygon: list[tuple[int, int]] = []
            for segment in subpath:
                steps = 1 if isinstance(segment, Line) else self.curve_steps
                if not isinstance(segment, (Line, *_CURVED_SEGMENTS)):
                    steps = max(1, self.curve_steps // 2)

                for index in range(steps + 1):
                    point = segment.point(index / steps)
                    xy = (round(point.real), round(point.imag))
                    if not polygon or polygon[-1] != xy:
                        polygon.append(xy)

            if len(polygon) >= 3:
                polygons.append(polygon)

        return polygons

    def _draw_armies(self, canvas: pygame.Surface, armies: dict[str, int]) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("calibri", 30)

        for territory_id, count in armies.items():
            point = self.data["army_points"].get(territory_id)
            if point is None:
                continue

            center = (point["x"] + 13, point["y"] + 10)
            pygame.draw.circle(canvas, (221, 221, 221), center, 20)
            pygame.draw.circle(canvas, (85, 85, 85), center, 20, 1)

            text_surface = font.render(str(count), True, (85, 85, 85))
            text_rect = text_surface.get_rect(center=center)
            canvas.blit(text_surface, text_rect)

    def _draw_continent_labels(self, canvas: pygame.Surface) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        title_font = pygame.font.SysFont("calibri", 24, bold=True)
        bonus_font = pygame.font.SysFont("calibri", 20, bold=True)

        for continent_name, continent_data in self.data.get("continents", {}).items():
            position = continent_data.get("label_position")
            if position is None:
                continue

            title_surface = title_font.render(continent_name, True, (55, 45, 32))
            bonus_surface = bonus_font.render(f"+{continent_data['bonus']}", True, (95, 52, 24))

            width = max(title_surface.get_width(), bonus_surface.get_width()) + 18
            height = title_surface.get_height() + bonus_surface.get_height() + 14
            badge = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(badge, (248, 241, 221, 185), badge.get_rect(), border_radius=8)
            pygame.draw.rect(badge, (120, 95, 62, 210), badge.get_rect(), 1, border_radius=8)
            badge.blit(title_surface, ((width - title_surface.get_width()) // 2, 5))
            badge.blit(bonus_surface, ((width - bonus_surface.get_width()) // 2, title_surface.get_height() + 4))

            top_left = (position["x"] - width // 2, position["y"] - height // 2)
            canvas.blit(badge, top_left)

    def _resolve_color(self, owner: int | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(owner, tuple):
            return owner
        return self.DEFAULT_PLAYER_COLORS.get(owner, list(self.DEFAULT_PLAYER_COLORS.values())[owner % len(self.DEFAULT_PLAYER_COLORS)])
