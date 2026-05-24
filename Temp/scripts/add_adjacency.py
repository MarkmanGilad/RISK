"""One-shot script that adds the classic Risk adjacency to risk_map_data.json.

Run from the repo root:
    python -m Temp.scripts.add_adjacency

The script is idempotent: running it again overwrites the `adjacency` key with
the same canonical content and validates that every neighbor list is symmetric
and references only known territory ids.
"""
from __future__ import annotations

import json
from pathlib import Path


ADJACENCY: dict[str, list[str]] = {
    # North America
    "Alaska": ["NorthWestTerritory", "Alberta", "Kamchatka"],
    "NorthWestTerritory": ["Alaska", "Alberta", "Ontario", "Greenland"],
    "Alberta": ["Alaska", "NorthWestTerritory", "Ontario", "WesternUnitedStates"],
    "Ontario": [
        "NorthWestTerritory",
        "Alberta",
        "Greenland",
        "Quebec",
        "WesternUnitedStates",
        "EasternUnitedStates",
    ],
    "Greenland": ["NorthWestTerritory", "Ontario", "Quebec", "Iceland"],
    "Quebec": ["Ontario", "Greenland", "EasternUnitedStates"],
    "WesternUnitedStates": ["Alberta", "Ontario", "EasternUnitedStates", "CentralAmerica"],
    "EasternUnitedStates": ["Ontario", "Quebec", "WesternUnitedStates", "CentralAmerica"],
    "CentralAmerica": ["WesternUnitedStates", "EasternUnitedStates", "Venezuela"],
    # South America
    "Venezuela": ["CentralAmerica", "Peru", "Brazil"],
    "Peru": ["Venezuela", "Brazil", "Argentina"],
    "Brazil": ["Venezuela", "Peru", "Argentina", "NorthAfrica"],
    "Argentina": ["Peru", "Brazil"],
    # Europe
    "Iceland": ["Greenland", "GreatBritain", "Scandinavia"],
    "GreatBritain": ["Iceland", "Scandinavia", "NorthernEurope", "WesternEurope"],
    "WesternEurope": ["GreatBritain", "NorthernEurope", "SouthernEurope", "NorthAfrica"],
    "NorthernEurope": [
        "GreatBritain",
        "Scandinavia",
        "Ukraine",
        "SouthernEurope",
        "WesternEurope",
    ],
    "SouthernEurope": [
        "WesternEurope",
        "NorthernEurope",
        "Ukraine",
        "MiddleEast",
        "Egypt",
        "NorthAfrica",
    ],
    "Scandinavia": ["Iceland", "GreatBritain", "NorthernEurope", "Ukraine"],
    "Ukraine": [
        "Scandinavia",
        "NorthernEurope",
        "SouthernEurope",
        "MiddleEast",
        "Afghanistan",
        "Ural",
    ],
    # Africa
    "NorthAfrica": [
        "Brazil",
        "WesternEurope",
        "SouthernEurope",
        "Egypt",
        "EastAfrica",
        "Congo",
    ],
    "Egypt": ["SouthernEurope", "MiddleEast", "EastAfrica", "NorthAfrica"],
    "Congo": ["NorthAfrica", "EastAfrica", "SouthAfrica"],
    "EastAfrica": [
        "Egypt",
        "NorthAfrica",
        "Congo",
        "SouthAfrica",
        "Madagascar",
        "MiddleEast",
    ],
    "SouthAfrica": ["Congo", "EastAfrica", "Madagascar"],
    "Madagascar": ["SouthAfrica", "EastAfrica"],
    # Asia
    "MiddleEast": [
        "SouthernEurope",
        "Ukraine",
        "Afghanistan",
        "India",
        "EastAfrica",
        "Egypt",
    ],
    "Afghanistan": ["Ukraine", "Ural", "China", "India", "MiddleEast"],
    "Ural": ["Ukraine", "Siberia", "China", "Afghanistan"],
    "India": ["MiddleEast", "Afghanistan", "China", "Siam"],
    "Siam": ["India", "China", "Indonesia"],
    "China": ["Afghanistan", "Ural", "Siberia", "Mongolia", "Siam", "India"],
    "Mongolia": ["China", "Siberia", "Irkutsk", "Kamchatka", "Japan"],
    "Irkutsk": ["Siberia", "Yakutsk", "Kamchatka", "Mongolia"],
    "Yakutsk": ["Siberia", "Irkutsk", "Kamchatka"],
    "Siberia": ["Ural", "China", "Mongolia", "Irkutsk", "Yakutsk"],
    "Kamchatka": ["Yakutsk", "Irkutsk", "Mongolia", "Japan", "Alaska"],
    "Japan": ["Kamchatka", "Mongolia"],
    # Australia
    "Indonesia": ["Siam", "NewGuinea", "WesternAustralia"],
    "NewGuinea": ["Indonesia", "WesternAustralia", "EasternAustralia"],
    "WesternAustralia": ["Indonesia", "NewGuinea", "EasternAustralia"],
    "EasternAustralia": ["NewGuinea", "WesternAustralia"],
}


def validate(adjacency: dict[str, list[str]], territories: set[str]) -> None:
    extra = set(adjacency) - territories
    missing = territories - set(adjacency)
    if extra or missing:
        raise ValueError(f"adjacency keys mismatch: extra={extra}, missing={missing}")

    for src, neighbors in adjacency.items():
        if len(neighbors) != len(set(neighbors)):
            raise ValueError(f"{src} has duplicate neighbors: {neighbors}")
        for dst in neighbors:
            if dst not in territories:
                raise ValueError(f"{src} -> unknown territory {dst}")
            if src not in adjacency[dst]:
                raise ValueError(f"adjacency not symmetric: {src} -> {dst} but not back")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "Assets" / "RiskMap" / "risk_map_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))

    territories = set(data["territory_names"].keys())
    validate(ADJACENCY, territories)

    canonical = {src: sorted(neigh) for src, neigh in sorted(ADJACENCY.items())}
    data["adjacency"] = canonical

    data_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote adjacency for {len(canonical)} territories to {data_path}")


if __name__ == "__main__":
    main()
