from __future__ import annotations

import random

from risk.ui.render.risk_map import RiskMapRenderer


def build_demo_state(renderer: RiskMapRenderer) -> tuple[dict[str, int], dict[str, int]]:
    rng = random.Random(7)
    owners: dict[str, int] = {}
    armies: dict[str, int] = {}
    for territory_id in renderer.all_territory_ids():
        owners[territory_id] = rng.randrange(6)
        armies[territory_id] = rng.randrange(1, 12)
    return owners, armies