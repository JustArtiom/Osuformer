from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.schemas.app import AppConfig
from src.training.checkpoint import CheckpointManager
from src.training.plotting import TrainingHistory, load_history, plot_history, save_history
from src.training.scheduler import build_scheduler


_PRECISION_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class StepMetrics:
    step: int
    loss: float
    lr: float
    tokens: int


class Trainer:
    def __init__(
        self,
        cfg: AppConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler: LambdaLR | None = (
            build_scheduler(
                optimizer=self.optimizer,
                warmup_steps=cfg.training.warmup_steps,
                max_steps=cfg.training.max_steps,
                min_lr=cfg.training.lr_scheduler.min_lr,
                base_lr=cfg.training.learning_rate,
            )
            if cfg.training.lr_scheduler.enabled
            else None
        )

        run_dir = Path(cfg.paths.checkpoints) / cfg.training.run_name
        self.run_dir = run_dir
        self.ckpt_manager = CheckpointManager(
            run_dir=run_dir,
            keep_last_n=cfg.training.keep_last_n_checkpoints,
            keep_best=cfg.training.keep_best_checkpoint,
            best_metric_mode="min",
        )
        self.step = 0
        if cfg.training.resume:
            self.step = self.ckpt_manager.load_latest(self.model, self.optimizer, self.scheduler, map_location=str(device))
        self._history_path = run_dir / "history.json"
        self._plot_path = run_dir / "loss.png"
        self.history: TrainingHistory = load_history(self._history_path) if cfg.training.resume else TrainingHistory()

        self._amp_dtype = _PRECISION_MAP.get(cfg.training.precision, torch.float32)
        self._amp_enabled = self._amp_dtype != torch.float32 and device.type == "cuda"
        self._grad_accum = max(1, cfg.training.grad_accum_steps)

    def fit(self) -> None:
        self.model.train()
        print(f"starting training at step {self.step} (target {self.cfg.training.max_steps})", flush=True)
        print("warming up data pipeline + compiling model kernels (first step may take 30-120s on MPS)...", flush=True)
        data_iter = _infinite(self.train_loader)
        last_metrics: StepMetrics | None = None
        while self.step < self.cfg.training.max_steps:
            last_metrics = self._train_step(data_iter)
            if last_metrics.tokens > 0:
                self.history.log_train(self.step, last_metrics.loss)
            if self.step % self.cfg.training.log_every_steps == 0:
                self._log(last_metrics)
            if self.step > 0 and self.step % self.cfg.training.save_every_steps == 0:
                self._maybe_save(last_metrics.loss)
            if self.val_loader is not None and self.step > 0 and self.step % self.cfg.training.val_every_steps == 0:
                val_loss = self._validate()
                self.history.log_val(self.step, val_loss)
                self._maybe_save(val_loss)
                self.model.train()
        self._maybe_save(last_metrics.loss if last_metrics is not None else None)

    def _train_step(self, data_iter) -> StepMetrics:
        start = time.time()
        total_loss = 0.0
        total_tokens = 0
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(self._grad_accum):
            batch = next(data_iter)
            loss, n_tokens = self._forward_backward(batch, scale=1.0 / self._grad_accum)
            total_loss += loss * n_tokens
            total_tokens += n_tokens
        if total_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return StepMetrics(step=self.step, loss=float("nan"), lr=self._current_lr(), tokens=0)
        if self.cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        avg_loss = total_loss / max(1, total_tokens)
        _ = time.time() - start
        return StepMetrics(step=self.step, loss=avg_loss, lr=self._current_lr(), tokens=total_tokens)

    def _forward_backward(self, batch, scale: float) -> tuple[float, int]:
        mel = batch.mel.to(self.device, non_blocking=True)
        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)
        loss_mask = batch.loss_mask.to(self.device, non_blocking=True)
        loss_weights = batch.loss_weights.to(self.device, non_blocking=True)
        token_pad_mask = batch.token_pad_mask.to(self.device, non_blocking=True)

        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled)
        with autocast_ctx:
            output = self.model(mel=mel, input_ids=input_ids, token_key_padding_mask=token_pad_mask)
            logits = output.logits
            loss = _masked_cross_entropy(logits, target_ids, loss_mask, loss_weights)
        scaled = loss * scale
        scaled.backward()
        return float(loss.detach().item()), int(loss_mask.sum().item())

    @torch.no_grad()
    def _validate(self) -> float:
        assert self.val_loader is not None
        self.model.eval()
        total = 0.0
        tokens = 0
        for batch in self.val_loader:
            mel = batch.mel.to(self.device, non_blocking=True)
            input_ids = batch.input_ids.to(self.device, non_blocking=True)
            target_ids = batch.target_ids.to(self.device, non_blocking=True)
            loss_mask = batch.loss_mask.to(self.device, non_blocking=True)
            loss_weights = batch.loss_weights.to(self.device, non_blocking=True)
            token_pad_mask = batch.token_pad_mask.to(self.device, non_blocking=True)
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled)
            with autocast_ctx:
                output = self.model(mel=mel, input_ids=input_ids, token_key_padding_mask=token_pad_mask)
                loss = _masked_cross_entropy(output.logits, target_ids, loss_mask, loss_weights)
            n = int(loss_mask.sum().item())
            total += float(loss.item()) * n
            tokens += n
        return total / max(1, tokens)

    def _maybe_save(self, metric: float | None) -> None:
        self.ckpt_manager.save(
            step=self.step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=metric,
        )
        save_history(self._history_path, self.history)
        plot_history(self._plot_path, self.history, self.cfg.training.run_name)

    def _log(self, metrics: StepMetrics) -> None:
        print(
            f"[step {metrics.step:>7d}]  loss={metrics.loss:.4f}  lr={metrics.lr:.2e}  tokens={metrics.tokens}",
            flush=True,
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    log_probs = logits.log_softmax(dim=-1)
    gathered = log_probs.gather(dim=-1, index=targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_weight = mask.float() if weights is None else mask.float() * weights
    loss = -gathered * token_weight
    denom = token_weight.sum().clamp(min=1)
    return loss.sum() / denom


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch
