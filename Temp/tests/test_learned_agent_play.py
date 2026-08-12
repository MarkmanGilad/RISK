"""Interactive learned-policy setup and adapter coverage (README.md)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest
import torch

from .conftest import make_env, make_settings
from risk.app import learned_agent_play
from risk.app.factory import GameFactory
from risk.game.player import Player
from risk.game.settings import GameSettings
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.ui.input.init_screen import InitScreenState


def _dqn_agent_and_env():
    env = make_env(n=3, seed=5, agent_kind="ai")
    return GNN_DQN_Agent(player_id=0, env=env, train_mode=False), env


def _manual_selection(checkpoint, agent_kind: str = "DQN") -> dict[str, str]:
    return {"source": "manual", "checkpoint": str(checkpoint), "agent_kind": agent_kind,
            "label": "", "preset_id": ""}


# --- UI-only setup state ---------------------------------------------------


def test_visible_cycle_enters_and_leaves_learned_agent() -> None:
    state = InitScreenState()
    for _ in range(6):
        state.next_visible_agent_kind(0)
    assert state.is_learned(0)
    assert state.seats[0].agent_kind == "ai"
    state.set_learned_selection(0, source="manual", checkpoint="model.pt", agent_kind="DQN", label="My DQN")
    assert state.next_visible_agent_kind(0) == "human"
    assert not state.is_learned(0)


def test_player_count_reduction_discards_learned_selection() -> None:
    state = InitScreenState()
    state.set_player_count(4)
    state.set_learned_selection(3, source="manual", checkpoint="model.pt", agent_kind="PPO", label="My PPO")
    state.set_player_count(3)
    assert state.learned_selections == {}


# --- predefined-best-model registry ----------------------------------------


def test_preset_registry_validates_and_resolves_relative_path(tmp_path, monkeypatch) -> None:
    params = tmp_path / "Params"
    params.mkdir()
    (params / "play_agents.json").write_text(json.dumps({"version": 1, "models": [{"id": "best", "label": "Best", "agent_kind": "DQN", "checkpoint": "Checkpoints/a.pt"}]}), encoding="utf-8")
    monkeypatch.setattr(learned_agent_play, "_REPO_ROOT", tmp_path)
    assert learned_agent_play.load_presets() == [{"id": "best", "label": "Best", "agent_kind": "DQN", "checkpoint": str(tmp_path / "Checkpoints" / "a.pt")}]


def test_preset_registry_rejects_duplicate_ids(tmp_path, monkeypatch) -> None:
    params = tmp_path / "Params"
    params.mkdir()
    model = {"id": "same", "label": "Best", "agent_kind": "DQN", "checkpoint": "a.pt"}
    (params / "play_agents.json").write_text(json.dumps({"version": 1, "models": [model, model]}), encoding="utf-8")
    monkeypatch.setattr(learned_agent_play, "_REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="unique"):
        learned_agent_play.load_presets()


# --- direct coupling with risk.learning.choose_agent's private helpers -----


def test_read_policy_state_round_trips_a_policy_only_file(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)

    state = learned_agent_play._read_policy_state(path)

    for key, value in agent.net.state_dict().items():
        assert torch.equal(state[key], value.cpu())


def test_read_policy_state_round_trips_an_episode_checkpoint_directory(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    ckpt_dir = tmp_path / "ep000100"
    agent.save_checkpoint(ckpt_dir)

    state = learned_agent_play._read_policy_state(ckpt_dir)

    for key, value in agent.net.state_dict().items():
        assert torch.equal(state[key], value.cpu())


def test_new_learned_agent_attaches_to_its_real_seat_and_zeroes_epsilon(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)
    state = learned_agent_play._read_policy_state(path)

    ctx = GameFactory.build(make_settings(n=3, seed=0, agent_kind="ai"))
    built = learned_agent_play._new_learned_agent("DQN", state, ctx, 1)

    assert built.player_id == 1
    assert built.env is ctx.env
    assert built.epsilon == 0.0
    assert built.train_mode is False
    for key, value in agent.net.state_dict().items():
        assert torch.equal(built.net.state_dict()[key], value)


def test_new_learned_agent_inference_is_deterministic(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)
    state = learned_agent_play._read_policy_state(path)

    ctx = GameFactory.build(make_settings(n=3, seed=0, agent_kind="ai"))
    built = learned_agent_play._new_learned_agent("DQN", state, ctx, 0)
    game_state = ctx.env.current_state()

    first = built.act([], game_state)
    second = built.act([], game_state)
    assert first.to_dict() == second.to_dict()


# --- pre-start validation ---------------------------------------------------


def test_validate_selections_accepts_a_valid_manual_policy(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)
    settings = make_settings(n=3, seed=0, agent_kind="ai")

    assert learned_agent_play.validate_selections(settings, {0: _manual_selection(path)}) == ""


def test_validate_selections_reports_a_missing_checkpoint(tmp_path) -> None:
    settings = make_settings(n=3, seed=0, agent_kind="ai")
    selections = {0: _manual_selection(tmp_path / "missing.pt")}

    error = learned_agent_play.validate_selections(settings, selections)
    assert "Player 1" in error


def test_validate_selections_reports_an_agent_kind_mismatch(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)
    settings = make_settings(n=3, seed=0, agent_kind="ai")
    selections = {0: _manual_selection(path, agent_kind="PPO")}

    error = learned_agent_play.validate_selections(settings, selections)
    assert "Player 1" in error


def test_validate_selections_ignores_seats_with_no_learned_selection() -> None:
    settings = make_settings(n=3, seed=0, agent_kind="ai")
    assert learned_agent_play.validate_selections(settings, {}) == ""


# --- actual-seat attachment / independence ----------------------------------


def test_build_agents_creates_independent_instances_for_a_shared_checkpoint(tmp_path) -> None:
    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)
    ctx = GameFactory.build(make_settings(n=3, seed=0, agent_kind="ai"))
    selections = {0: _manual_selection(path), 1: _manual_selection(path)}

    built = learned_agent_play.build_agents(ctx, selections)

    assert built[0] is not built[1]
    assert built[0].net is not built[1].net
    built[0].epsilon = 0.5
    assert built[1].epsilon == 0.0


def test_labels_falls_back_to_agent_kind_when_no_display_label(tmp_path) -> None:
    selections = {0: _manual_selection(tmp_path / "policy.pt")}
    assert learned_agent_play.labels(selections) == {0: "DQN"}


# --- mixed roster through the real interactive loop -------------------------


def test_mixed_human_heuristic_learned_apploop_runs_without_crashing(tmp_path) -> None:
    import pygame

    from risk.app.loop import AppLoop

    agent, _ = _dqn_agent_and_env()
    path = tmp_path / "policy.pt"
    agent.save_params(path)

    settings = GameSettings(
        players=(
            Player(id=0, name="Raider", color=(200, 0, 0), agent_kind="raider"),
            Player(id=1, name="Learner", color=(0, 200, 0), agent_kind="ai"),
            Player(id=2, name="Human", color=(0, 0, 200), agent_kind="human"),
        ),
        seed=3,
    )
    ctx = GameFactory.build(settings)
    selections = {1: _manual_selection(path)}
    selections[1]["label"] = "Test DQN"
    for seat, built in learned_agent_play.build_agents(ctx, selections).items():
        ctx.agents[seat] = built

    pygame.init()
    try:
        screen = pygame.display.set_mode((640, 400))
        loop = AppLoop(
            ctx, screen, width=640, height=400, ai_delay_ms=0, marker_ms=0,
            player_labels=learned_agent_play.labels(selections),
        )
        rc = loop.run(max_ticks=150, show_win_screen=False)
        assert rc == 0
        assert loop.last_action_text != ""
        assert isinstance(ctx.agents[1], GNN_DQN_Agent)
    finally:
        pygame.quit()
