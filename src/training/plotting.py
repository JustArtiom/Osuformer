from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingHistory:
    train_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    def log_train(self, step: int, loss: float) -> None:
        self.train_steps.append(int(step))
        self.train_losses.append(float(loss))

    def log_val(self, step: int, loss: float) -> None:
        self.val_steps.append(int(step))
        self.val_losses.append(float(loss))


def load_history(path: Path) -> TrainingHistory:
    if not path.exists():
        return TrainingHistory()
    data = json.loads(path.read_text())
    return TrainingHistory(
        train_steps=list(data.get("train_steps", [])),
        train_losses=list(data.get("train_losses", [])),
        val_steps=list(data.get("val_steps", [])),
        val_losses=list(data.get("val_losses", [])),
    )


def save_history(path: Path, history: TrainingHistory) -> None:
    path.write_text(json.dumps(asdict(history), indent=2))


def plot_history(path: Path, history: TrainingHistory, run_name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history.train_steps and not history.val_steps:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if history.train_steps:
        ax.plot(history.train_steps, history.train_losses, label="train", alpha=0.6, linewidth=1.2)
    if history.val_steps:
        ax.plot(history.val_steps, history.val_losses, label="val", marker="o", linewidth=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{run_name} — training curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
