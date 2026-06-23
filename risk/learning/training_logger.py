"""Logging/checkpointing orchestrator for RL training runs.

See `Docs/Training-Logging-Plan.md` for the full design. `TrainingLogger`
is the single entry point `Trainer` talks to for both local checkpointing
and W&B metric logging — it owns *when*/*where* a checkpoint happens, not
the actual save/restore logic, which stays on `GNN_DQN_Agent`
(`save_checkpoint`/`load_checkpoint`) — the same split as `Environment`
only ever calling `RewardCalculator`, never computing reward itself.

`wandb` is optional (not in `requirements.txt` by default): if it isn't
installed, or `use_wandb=False`, every W&B-facing method becomes a no-op —
training still runs, just without remote logging.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from risk.learning import train_constants
from risk.learning.gnn_dqn_agent import GNN_DQN_Agent
from risk.learning.train_constants import CHECKPOINT_AFTER, CHECKPOINT_EVERY

try:
    import wandb
except ImportError:
    wandb = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_NAME_RE = re.compile(r"^ep(\d+)$")


class TrainingLogger:
    """Owns W&B init/log/finish and local full-checkpoint save/resume."""

    def __init__(
        self,
        run_id: int,
        checkpoint_dir: str | Path,
        *,
        use_wandb: bool = True,
        project_name: str = "Risk-GNN-DQN",
        resume: bool = True,
        notes: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.project_name = project_name
        self.resume = resume
        self.notes = notes
        self._wandb_enabled = bool(use_wandb and wandb is not None)
        if use_wandb and wandb is None:
            print("wandb not installed — W&B logging disabled (pip install wandb to enable).")

    def start_run(
        self, *, agent: GNN_DQN_Agent, trainer, extra_config: Optional[dict] = None
    ) -> None:
        if not self._wandb_enabled:
            return
        config = self._build_config(agent)
        if extra_config:
            config.update(extra_config)
        wandb.init(
            project=self.project_name,
            name=f"run_{self.run_id:03d}",
            config=config,
            notes=self.notes,
        )

    def log_episode(self, *, episode: int, metrics: dict) -> None:
        if not self._wandb_enabled:
            return
        payload = dict(metrics)
        payload.setdefault("episode", episode)
        wandb.log(payload)

    def format_status_line(
        self,
        *,
        topology,
        episode: int,
        n_episodes: int,
        step_count: int,
        agent_turns: int,
        seat: int,
        current_state,
    ) -> str:
        """Build compact single-line terminal status text."""
        agent_territories = sum(1 for owner in current_state.owners if owner == seat)
        agent_continents = [
            continent
            for continent in topology.continents
            if topology.owns_continent(current_state.owners, continent, seat)
        ]
        opponent_territories_by_player = {
            player_id: sum(1 for owner in current_state.owners if owner == player_id)
            for player_id in range(len(current_state.hands))
            if player_id != seat
        }
        armies_by_player = {
            player_id: sum(
                current_state.armies[idx]
                for idx, owner in enumerate(current_state.owners)
                if owner == player_id
            )
            for player_id in range(len(current_state.hands))
        }
        other_territories = sum(
            1 for owner in current_state.owners if owner is not None and owner != seat
        )
        others_territories_detail = ",".join(
            f"p{player_id}:{count}"
            for player_id, count in sorted(opponent_territories_by_player.items())
        )
        armies_detail = ",".join(
            f"p{player_id}:{count}" for player_id, count in sorted(armies_by_player.items())
        )
        continents_detail = ",".join(agent_continents) if agent_continents else "-"
        return (
            f"run={self.run_id:03d} ep={episode}/{n_episodes} "
            f"steps={step_count} agent_turns={agent_turns} "
            f"agent_terr={agent_territories} "
            f"others_terr_total={other_territories} "
            f"others_terr_by_player=[{others_territories_detail}] "
            f"agent_cont={len(agent_continents)}[{continents_detail}] "
            f"armies=[{armies_detail}]"
        )

    def checkpoint(self, *, episode: int, agent: GNN_DQN_Agent) -> tuple[bool, Optional[Path]]:
        """Check interval thresholds; save full checkpoint if conditions met.
        
        Returns (did_save, path_or_none) where path_or_none is the checkpoint
        path if saved, or None if skipped due to interval logic.
        """
        if episode < CHECKPOINT_AFTER:
            return False, None
        if (episode - CHECKPOINT_AFTER) % CHECKPOINT_EVERY != 0:
            return False, None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_checkpoint(episode=episode, agent=agent)
        return True, path

    def _save_checkpoint(self, *, episode: int, agent: GNN_DQN_Agent) -> Path:
        """Internal: save full checkpoint without interval checks."""
        path = self.checkpoint_dir / f"ep{episode:06d}"
        agent.save_checkpoint(path)
        return path

    def try_resume(self, *, agent: GNN_DQN_Agent) -> Optional[dict]:
        if not self.resume:
            return None
        latest = self._latest_checkpoint()
        if latest is None:
            return None
        episode, path = latest
        try:
            agent.load_checkpoint(path)
        except Exception as exc:
            print(f"Resume from {path} failed ({exc!r}); continuing from a fresh agent state.")
            return None
        return {"episode": episode}

    def finish(self) -> None:
        if not self._wandb_enabled:
            return
        wandb.finish()

    # --- internals -----------------------------------------------------

    def _latest_checkpoint(self) -> Optional[tuple[int, Path]]:
        if not self.checkpoint_dir.exists():
            return None
        best: Optional[tuple[int, Path]] = None
        for child in self.checkpoint_dir.iterdir():
            if not child.is_dir():
                continue
            match = _CHECKPOINT_NAME_RE.match(child.name)
            if match is None:
                continue
            episode = int(match.group(1))
            if best is None or episode > best[0]:
                best = (episode, child)
        return best

    def _build_config(self, agent: GNN_DQN_Agent) -> dict:
        config = {name: getattr(train_constants, name) for name in train_constants.__all__}
        config.update(
            {
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": self._git_commit(),
                "model_class": type(agent.net).__name__,
                "model_str": str(agent.net),
                "target_model_str": str(agent.target_net),
                "param_count": sum(p.numel() for p in agent.net.parameters()),
                "device": str(agent.device),
            }
        )
        return config

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                cwd=_REPO_ROOT,
            )
            return result.stdout.strip()
        except Exception:
            return None


__all__ = ["TrainingLogger"]
