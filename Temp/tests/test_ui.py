"""Tests for hit-testing and the init screen (Phase 6)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from risk.ui.input.hit_test import TerritoryHitTester
from risk.ui.input.init_screen import InitScreenState


# --- hit testing ----------------------------------------------------------


def _simple_polys() -> dict[str, list[list[tuple[int, int]]]]:
    return {
        "Alaska": [[(0, 0), (100, 0), (100, 100), (0, 100)]],
        "Alberta": [[(100, 0), (200, 0), (200, 100), (100, 100)]],
    }


def test_hit_test_interior_point_resolves() -> None:
    ht = TerritoryHitTester(_simple_polys(), blit_rect=(0, 0, 200, 100), image_size=(200, 100))
    assert ht.territory_at(50, 50) == "Alaska"
    assert ht.territory_at(150, 50) == "Alberta"


def test_hit_test_outside_returns_none() -> None:
    ht = TerritoryHitTester(_simple_polys(), blit_rect=(0, 0, 200, 100), image_size=(200, 100))
    assert ht.territory_at(-1, 50) is None
    assert ht.territory_at(50, 200) is None


def test_hit_test_with_scaling_and_offset() -> None:
    # Board image is 200x100 but blitted into a 600x300 rect offset by (50, 20).
    ht = TerritoryHitTester(
        _simple_polys(), blit_rect=(50, 20, 600, 300), image_size=(200, 100)
    )
    # Image (50, 50) -> screen (50 + 50*3, 20 + 50*3) = (200, 170).
    assert ht.territory_at(200, 170) == "Alaska"
    # Image (150, 50) -> screen (50 + 150*3, 20 + 50*3) = (500, 170).
    assert ht.territory_at(500, 170) == "Alberta"


def test_hit_test_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        TerritoryHitTester({}, blit_rect=(0, 0, 0, 100), image_size=(10, 10))
    with pytest.raises(ValueError):
        TerritoryHitTester({}, blit_rect=(0, 0, 10, 10), image_size=(0, 10))


def test_hit_test_with_real_board_polygons() -> None:
    """Smoke: use the real renderer's polygons via direct JSON parsing,
    pick the centroid of a known territory, and confirm it resolves."""
    from risk.ui.render.risk_map import RiskMapRenderer

    renderer = RiskMapRenderer()
    polys = renderer.territory_polygons
    w, h = renderer.base_map.get_size()
    ht = TerritoryHitTester(polys, blit_rect=(0, 0, w, h), image_size=(w, h))

    # Pick the centroid of Alaska's first polygon and confirm round-trip.
    alaska_poly = polys["Alaska"][0]
    cx = sum(p[0] for p in alaska_poly) // len(alaska_poly)
    cy = sum(p[1] for p in alaska_poly) // len(alaska_poly)
    # Centroid of an arbitrary polygon may fall outside concave shapes; in
    # that case scan a few interior points until one is found.
    found = ht.territory_at(cx, cy)
    if found != "Alaska":
        # Fallback: scan a coarse grid inside the polygon's bounding box.
        xs = [p[0] for p in alaska_poly]
        ys = [p[1] for p in alaska_poly]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        found = None
        for sx in range(x0 + 1, x1, max(1, (x1 - x0) // 20)):
            for sy in range(y0 + 1, y1, max(1, (y1 - y0) // 20)):
                if ht.territory_at(sx, sy) == "Alaska":
                    found = "Alaska"
                    break
            if found:
                break
    assert found == "Alaska"


# --- init screen ----------------------------------------------------------


def test_init_screen_defaults_to_three_humans() -> None:
    s = InitScreenState()
    assert s.player_count == 3
    assert all(seat.agent_kind == "human" for seat in s.seats)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_init_screen_set_player_count_valid(n: int) -> None:
    s = InitScreenState()
    s.set_player_count(n)
    assert s.player_count == n


@pytest.mark.parametrize("n", [0, 1, 2, 7, 10])
def test_init_screen_rejects_bad_count(n: int) -> None:
    s = InitScreenState()
    with pytest.raises(ValueError):
        s.set_player_count(n)


def test_init_screen_set_name_and_color_and_kind() -> None:
    s = InitScreenState()
    s.set_name(0, "Alice")
    s.set_color(1, (12, 34, 56))
    s.set_agent_kind(2, "raider")
    assert s.seats[0].name == "Alice"
    assert s.seats[1].color == (12, 34, 56)
    assert s.seats[2].agent_kind == "raider"


def test_init_screen_cycles_agent_kinds() -> None:
    s = InitScreenState()
    assert s.seats[0].agent_kind == "human"
    assert s.next_agent_kind(0) == "random"
    assert s.next_agent_kind(0) == "raider"
    assert s.next_agent_kind(0) == "sentinel"
    assert s.next_agent_kind(0) == "empire"
    assert s.next_agent_kind(0) == "human"


def test_init_screen_rejects_invalid_inputs() -> None:
    s = InitScreenState()
    with pytest.raises(ValueError):
        s.set_name(0, "  ")
    with pytest.raises(ValueError):
        s.set_color(0, (300, 0, 0))
    with pytest.raises(ValueError):
        s.set_agent_kind(0, "ROBOT")
    with pytest.raises(IndexError):
        s.set_name(99, "x")


def test_init_screen_build_settings_round_trip() -> None:
    s = InitScreenState()
    s.set_player_count(4)
    s.set_name(0, "Alice")
    s.set_name(1, "Bob")
    s.set_agent_kind(2, "raider")
    s.set_agent_kind(3, "empire")
    settings = s.build_settings(seed=99)
    assert settings.player_count == 4
    assert settings.seed == 99
    assert [p.name for p in settings.players] == ["Alice", "Bob", "Player 3", "Player 4"]
    assert {p.agent_kind for p in settings.players} == {"human", "raider", "empire"}


def test_init_screen_blocks_start_on_duplicate_colors() -> None:
    s = InitScreenState()
    s.set_color(1, s.seats[0].color)  # duplicate
    ok, why = s.can_start()
    assert not ok
    assert "color" in why.lower()
    with pytest.raises(ValueError):
        s.build_settings()


# --- renderer regression --------------------------------------------------


def test_renderer_still_produces_a_surface() -> None:
    """Phase 6 must not break the existing renderer."""
    import pygame

    pygame.init()
    try:
        from risk.ui.render.risk_map import RiskMapRenderer

        r = RiskMapRenderer()
        surface = r.render(owners={"Alaska": 0, "Alberta": 1}, armies={"Alaska": 3})
        assert surface.get_width() > 0
        assert surface.get_height() > 0
    finally:
        pygame.quit()
