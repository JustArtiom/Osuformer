from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CheckpointState:
    step: int
    best_metric: float | None = None
    history: list[dict] = field(default_factory=list)


class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last_n: int, keep_best: bool, best_metric_mode: str = "min"):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.best_mode = best_metric_mode
        self._state_path = run_dir / "state.json"
        self._state = self._load_state()

    @property
    def step(self) -> int:
        return self._state.step

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR | None,
        metric: float | None = None,
    ) -> Path:
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metric": metric,
        }
        ckpt_path = self.run_dir / f"step_{step:08d}.pt"
        torch.save(ckpt, ckpt_path)

        self._state.step = step
        self._update_latest_pointer(ckpt_path)
        if metric is not None:
            self._state.history.append({"step": step, "metric": float(metric)})
            if self.keep_best and self._is_best(metric):
                self._state.best_metric = float(metric)
                self._write_best_pointer(ckpt_path)
        self._save_state()
        self._prune_old()
        return ckpt_path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LambdaLR | None = None,
        map_location: str = "cpu",
    ) -> int:
        pointer = self.run_dir / "latest.pt"
        if not pointer.exists():
            return 0
        payload = torch.load(pointer, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        return int(payload.get("step", 0))

    def _is_best(self, metric: float) -> bool:
        if self._state.best_metric is None:
            return True
        if self.best_mode == "min":
            return metric < self._state.best_metric
        return metric > self._state.best_metric

    def _update_latest_pointer(self, ckpt_path: Path) -> None:
        pointer = self.run_dir / "latest.pt"
        if pointer.exists() or pointer.is_symlink():
            pointer.unlink()
        try:
            pointer.symlink_to(ckpt_path.name)
        except OSError:
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), pointer)

    def _write_best_pointer(self, ckpt_path: Path) -> None:
        pointer = self.run_dir / "best.pt"
        if pointer.exists() or pointer.is_symlink():
            pointer.unlink()
        try:
            pointer.symlink_to(ckpt_path.name)
        except OSError:
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), pointer)

    def _prune_old(self) -> None:
        if self.keep_last_n <= 0:
            return
        checkpoints = sorted(self.run_dir.glob("step_*.pt"))
        if len(checkpoints) <= self.keep_last_n:
            return
        best_target = None
        best_pointer = self.run_dir / "best.pt"
        if best_pointer.exists():
            try:
                best_target = (best_pointer.resolve()).name
            except Exception:
                best_target = None
        excess = checkpoints[: len(checkpoints) - self.keep_last_n]
        for ckpt in excess:
            if best_target is not None and ckpt.name == best_target:
                continue
            ckpt.unlink()

    def _load_state(self) -> CheckpointState:
        if not self._state_path.exists():
            return CheckpointState(step=0)
        data = json.loads(self._state_path.read_text())
        return CheckpointState(
            step=int(data.get("step", 0)),
            best_metric=data.get("best_metric"),
            history=list(data.get("history", [])),
        )

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(asdict(self._state), indent=2))
