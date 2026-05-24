"""Map territories from screen-space clicks to territory ids.

The renderer draws into an image-coordinate surface and the demo/game
loop blits it into a window rect (with letterboxing). This helper
inverts that transform and runs point-in-polygon against the renderer's
`territory_polygons` to identify the clicked territory.

Pure geometry: no pygame display required, so it tests headless.
"""
from __future__ import annotations

from typing import Iterable, Optional


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    """Standard even-odd ray-casting test."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


class TerritoryHitTester:
    """Resolve (screen_x, screen_y) -> territory_id.

    Parameters
    ----------
    territory_polygons:
        `{territory_id: [polygon, ...]}` in image (map) coordinates.
    blit_rect:
        The pygame.Rect (or (x, y, w, h) tuple) of the destination
        rectangle the rendered board occupies on screen.
    image_size:
        `(width, height)` of the rendered board image, before scaling
        into `blit_rect`.
    """

    def __init__(
        self,
        territory_polygons: dict[str, list[list[tuple[int, int]]]],
        blit_rect: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> None:
        self.territory_polygons = territory_polygons
        self.blit_rect = tuple(blit_rect)
        self.image_size = tuple(image_size)
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")
        if self.blit_rect[2] <= 0 or self.blit_rect[3] <= 0:
            raise ValueError("blit_rect must have positive size")

    def screen_to_image(self, sx: float, sy: float) -> Optional[tuple[float, float]]:
        x, y, w, h = self.blit_rect
        if not (x <= sx < x + w and y <= sy < y + h):
            return None
        iw, ih = self.image_size
        ix = (sx - x) * iw / w
        iy = (sy - y) * ih / h
        return (ix, iy)

    def territory_at(self, sx: float, sy: float) -> Optional[str]:
        img_pt = self.screen_to_image(sx, sy)
        if img_pt is None:
            return None
        for terr, polygons in self.territory_polygons.items():
            for poly in polygons:
                if _point_in_polygon(img_pt, poly):
                    return terr
        return None


__all__ = ["TerritoryHitTester"]
