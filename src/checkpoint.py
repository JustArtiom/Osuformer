from typing import Optional
from pathlib import Path
from .config import ExperimentConfig
from .tokenizer import Tokenizer
from .data.analytics import CheckpointAnalytics
import torch

class Checkpoint():
  def __init__(
      self, 
      config: ExperimentConfig, 
      tokenizer: Tokenizer, 
      std: float,
      mean: float,
      name: Optional[str] = None,
      ):
    self.parent_path = Path(config.checkpoint.path)
    self.parent_path.mkdir(parents=True, exist_ok=True)
    if name is None:
      self.name = self.get_next_name()
    else: 
      self.name = name
    self.path = self.parent_path / self.name
    self.path.mkdir(parents=True, exist_ok=True)
    self.analytics_path = self.path / "analytics"
    self.analytics_path.mkdir(parents=True, exist_ok=True)
    self.analytics = CheckpointAnalytics(self.analytics_path)
    self.tokenizer = tokenizer
    self.best_val = float("inf")
    self.bad_epochs = 0
    self.mean = mean
    self.std = std
    self.patience = config.training.early_stop.patience
    self.min_delta = config.training.early_stop.delta

  def get_next_name(self) -> str:
    existing = [
      p.name for p in self.parent_path.iterdir()
      if p.is_dir() and p.name.isdigit()
    ]
    existing_nums = [int(n) for n in existing]
    next_num = max(existing_nums, default=0) + 1
    return str(next_num)
  
  def save_model(self, name: str, data):
    model_path = self.path / name
    torch.save(data, str(model_path))

  def save_latest(
    self,
    model,
    optimizer,
    scaler,
    epoch,
    best_val,
  ):
    self.save_model(
      "latest.pt",
      {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
        "mean": self.mean,
        "std": self.std,
        "vocab": self.tokenizer.vocab,
      }
    )

  def save_best(self, model, epoch, val_loss):
    self.save_model(
      "best.pt",
      {
        "epoch": epoch,
        "model": model.state_dict(),
        "val_loss": val_loss,
        "mean": self.mean,
        "std": self.std,
        "vocab": self.tokenizer.vocab,
      }
    )

  def step(
    self,
    *,
    model,
    optimizer,
    scaler,
    epoch,
    train_loss,
    val_loss,
    current_lr,
    token_accuracies=None,
    loss_decomposition=None,
  ) -> bool:
    self.analytics.collect_epoch_metrics(
      epoch=epoch,
      val_loss=val_loss,
      train_loss=train_loss,
      current_lr=current_lr,
      token_accuracies=token_accuracies,
      loss_decomposition=loss_decomposition,
    )
    self.analytics.save()
    improved = val_loss < (self.best_val - self.min_delta)

    if improved:
      self.best_val = val_loss
      self.bad_epochs = 0
      self.save_best(model, epoch, val_loss)
    else:
      self.bad_epochs += 1

    self.save_latest(
      model,
      optimizer,
      scaler,
      epoch,
      self.best_val,
    )

    return self.bad_epochs >= self.patience