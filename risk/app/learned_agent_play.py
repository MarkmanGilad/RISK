"""Interactive saved-policy selection and loading helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risk.app.factory import GameFactory
from risk.learning.choose_agent import _new_learned_agent, _read_policy_state


_REPO_ROOT = Path(__file__).resolve().parents[2]
_KINDS = frozenset({"DQN", "Dueling_DQN", "PPO"})


def load_presets() -> list[dict[str, str]]:
    path = _REPO_ROOT / "Params" / "play_agents.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid preset registry: {path}") from exc
    if raw.get("version") != 1 or not isinstance(raw.get("models"), list):
        raise ValueError("Preset registry must contain version 1 and a models list")
    presets: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in raw["models"]:
        if not isinstance(item, dict):
            raise ValueError("Every preset must be an object")
        required = ("id", "label", "agent_kind", "checkpoint")
        if not all(isinstance(item.get(key), str) and item[key] for key in required):
            raise ValueError("Every preset needs id, label, agent_kind, and checkpoint")
        if item["id"] in ids or item["agent_kind"] not in _KINDS:
            raise ValueError("Preset ids must be unique and kinds must be supported")
        ids.add(item["id"])
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = _REPO_ROOT / checkpoint
        presets.append({**item, "checkpoint": str(checkpoint)})
    return presets


def validate_selections(settings, selections: dict[int, dict[str, str]]) -> str:
    if not selections:
        return ""
    ctx = GameFactory.build(settings)
    for seat, selection in selections.items():
        if not selection.get("checkpoint"):
            return f"Player {seat + 1}: choose a learned model."
        kind = selection.get("agent_kind", "")
        if kind not in _KINDS:
            return f"Player {seat + 1}: unsupported learned agent kind."
        try:
            state = _read_policy_state(Path(selection["checkpoint"]))
            _new_learned_agent(kind, state, ctx, seat)
        except Exception as exc:
            return f"Player {seat + 1}: {exc}"
    return ""


def build_agents(ctx, selections: dict[int, dict[str, str]]) -> dict[int, Any]:
    agents: dict[int, Any] = {}
    for seat, selection in selections.items():
        state = _read_policy_state(Path(selection["checkpoint"]))
        agents[seat] = _new_learned_agent(selection["agent_kind"], state, ctx, seat)
    return agents


def labels(selections: dict[int, dict[str, str]]) -> dict[int, str]:
    return {seat: selection["label"] or selection["agent_kind"] for seat, selection in selections.items()}

