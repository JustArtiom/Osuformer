from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.schemas.app import AppConfig
from src.model.conditioning import ConditionFeatures
from src.training.checkpoint import CheckpointManager
from src.training.distributed import DistEnv, all_reduce_mean, barrier
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
        dist_env: DistEnv | None = None,
    ):
        self.cfg = cfg
        self.dist_env = dist_env if dist_env is not None else DistEnv(
            world_size=1, global_rank=0, local_rank=0, is_main=True, enabled=False
        )
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        raw_model = model.to(device)

        self.optimizer = AdamW(
            _build_param_groups(raw_model, cfg.training.learning_rate, cfg.training.encoder_lr_scale),
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
        self.ckpt_manager: CheckpointManager | None = None
        if self.dist_env.is_main:
            self.ckpt_manager = CheckpointManager(
                run_dir=run_dir,
                keep_last_n=cfg.training.keep_last_n_checkpoints,
                keep_best=cfg.training.keep_best_checkpoint,
                best_metric_mode="min",
            )
        barrier()

        self.step = 0
        if cfg.training.resume:
            self.step = self._load_latest(raw_model)

        if self.dist_env.enabled and device.type == "cuda":
            self.model: nn.Module = DDP(raw_model, device_ids=[device.index], output_device=device.index)
            self.raw_model: nn.Module = raw_model
        else:
            self.model = raw_model
            self.raw_model = raw_model

        self._history_path = run_dir / "history.json"
        self._plot_path = run_dir / "loss.png"
        self.history: TrainingHistory = load_history(self._history_path) if cfg.training.resume else TrainingHistory()

        self._amp_dtype = _PRECISION_MAP.get(cfg.training.precision, torch.float32)
        self._amp_enabled = self._amp_dtype != torch.float32 and device.type == "cuda"
        self._grad_accum = max(1, cfg.training.grad_accum_steps)

    def _load_latest(self, raw_model: nn.Module) -> int:
        latest_path = self.run_dir / "latest.pt"
        if not latest_path.exists():
            return 0
        payload = torch.load(latest_path, map_location=str(self.device), weights_only=False)
        raw_model.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        return int(payload.get("step", 0))

    def fit(self) -> None:
        self.model.train()
        if self.dist_env.is_main:
            print(f"starting training at step {self.step} (target {self.cfg.training.max_steps})", flush=True)
        if self.step >= self.cfg.training.max_steps:
            if self.dist_env.is_main:
                print(
                    f"nothing to do: current step {self.step} >= max_steps {self.cfg.training.max_steps}. "
                    f"either (a) raise training.max_steps, (b) delete the checkpoints dir at {self.run_dir}, "
                    f"or (c) use a different training.run_name.",
                    flush=True,
                )
            return
        if self.dist_env.is_main:
            print("warming up data pipeline + compiling model kernels (first step may take 30-120s)...", flush=True)
        data_iter = _infinite(self.train_loader)
        last_metrics: StepMetrics | None = None
        saved_at_step = -1
        while self.step < self.cfg.training.max_steps:
            last_metrics = self._train_step(data_iter)
            if last_metrics.tokens > 0:
                self.history.log_train(self.step, last_metrics.loss)
            if self.dist_env.is_main and self.step % self.cfg.training.log_every_steps == 0:
                self._log(last_metrics)
            if self.val_loader is not None and self.step > 0 and self.step % self.cfg.training.val_every_steps == 0:
                val_loss = self._validate()
                self.history.log_val(self.step, val_loss)
                self._maybe_save(val_loss)
                saved_at_step = self.step
                self.model.train()
            if (
                self.step > 0
                and self.step % self.cfg.training.save_every_steps == 0
                and saved_at_step != self.step
            ):
                self._maybe_save(None)
                saved_at_step = self.step
        if saved_at_step != self.step:
            self._maybe_save(None)

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
            if self.scheduler is not None:
                self.scheduler.step()
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
        summary_mel = batch.summary_mel.to(self.device, non_blocking=True)
        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)
        loss_mask = batch.loss_mask.to(self.device, non_blocking=True)
        loss_weights = batch.loss_weights.to(self.device, non_blocking=True)
        token_pad_mask = batch.token_pad_mask.to(self.device, non_blocking=True)
        cond_features = _move_cond_features(batch.cond_features, self.device)
        star_target = batch.star_target.to(self.device, non_blocking=True)
        descriptor_target = batch.descriptor_target.to(self.device, non_blocking=True)

        cond_null_mask = self._sample_cond_null_mask(input_ids.shape[0])

        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled)
        with autocast_ctx:
            output = self.model(
                mel=mel,
                input_ids=input_ids,
                cond_features=cond_features,
                summary_mel=summary_mel,
                cond_null_mask=cond_null_mask,
                token_key_padding_mask=token_pad_mask,
            )
            ce_loss = _masked_cross_entropy(output.logits, target_ids, loss_mask, loss_weights)
            star_loss = nn.functional.smooth_l1_loss(output.aux.star, star_target)
            descriptor_loss = nn.functional.binary_cross_entropy_with_logits(
                output.aux.descriptor_logits, descriptor_target
            )
            z_loss = _z_loss(output.logits, loss_mask)
            total = (
                ce_loss
                + self.cfg.training.aux_star_weight * star_loss
                + self.cfg.training.aux_descriptor_weight * descriptor_loss
                + self.cfg.training.z_loss_weight * z_loss
            )
        scaled = total * scale
        scaled.backward()
        return float(ce_loss.detach().item()), int(loss_mask.sum().item())

    def _sample_cond_null_mask(self, batch_size: int) -> Tensor | None:
        prob = float(self.cfg.training.cfg_dropout_prob)
        if prob <= 0.0:
            return None
        rand = torch.rand(batch_size, device=self.device)
        return rand < prob

    @torch.no_grad()
    def _validate(self) -> float:
        assert self.val_loader is not None
        self.model.eval()
        total = 0.0
        tokens = 0
        for batch in self.val_loader:
            mel = batch.mel.to(self.device, non_blocking=True)
            summary_mel = batch.summary_mel.to(self.device, non_blocking=True)
            input_ids = batch.input_ids.to(self.device, non_blocking=True)
            target_ids = batch.target_ids.to(self.device, non_blocking=True)
            loss_mask = batch.loss_mask.to(self.device, non_blocking=True)
            loss_weights = batch.loss_weights.to(self.device, non_blocking=True)
            token_pad_mask = batch.token_pad_mask.to(self.device, non_blocking=True)
            cond_features = _move_cond_features(batch.cond_features, self.device)
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled)
            with autocast_ctx:
                output = self.model(
                    mel=mel,
                    input_ids=input_ids,
                    cond_features=cond_features,
                    summary_mel=summary_mel,
                    token_key_padding_mask=token_pad_mask,
                )
                loss = _masked_cross_entropy(output.logits, target_ids, loss_mask, loss_weights)
            n = int(loss_mask.sum().item())
            total += float(loss.item()) * n
            tokens += n
        local_mean = total / max(1, tokens)
        if self.dist_env.enabled:
            return all_reduce_mean(local_mean, self.device)
        return local_mean

    def _maybe_save(self, metric: float | None) -> None:
        if not self.dist_env.is_main or self.ckpt_manager is None:
            barrier()
            return
        ckpt_path = self.ckpt_manager.save(
            step=self.step,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=metric,
        )
        save_history(self._history_path, self.history)
        plot_history(self._plot_path, self.history, self.cfg.training.run_name)
        metric_str = f" metric={metric:.4f}" if metric is not None else ""
        print(f"[ckpt]   step={self.step}  -> {ckpt_path.name}{metric_str}", flush=True)
        barrier()

    def _log(self, metrics: StepMetrics) -> None:
        print(
            f"[step {metrics.step:>7d}]  loss={metrics.loss:.4f}  lr={metrics.lr:.2e}  tokens={metrics.tokens}",
            flush=True,
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def _build_param_groups(model: nn.Module, base_lr: float, encoder_lr_scale: float) -> list[dict[str, object]]:
    encoder = getattr(model, "encoder", None)
    encoder_params: list[nn.Parameter] = []
    encoder_ids: set[int] = set()
    if encoder is not None:
        for p in encoder.parameters():
            if p.requires_grad:
                encoder_params.append(p)
                encoder_ids.add(id(p))
    other_params: list[nn.Parameter] = [
        p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids
    ]
    groups: list[dict[str, object]] = [{"params": other_params, "lr": base_lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr * encoder_lr_scale})
    return groups


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    vocab_out = logits.shape[-1]
    in_range = (targets >= 0) & (targets < vocab_out)
    safe_targets = targets.clamp(min=0, max=vocab_out - 1)
    log_probs = logits.log_softmax(dim=-1)
    gathered = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_weight = mask.float() * in_range.float()
    if weights is not None:
        token_weight = token_weight * weights
    loss = -gathered * token_weight
    denom = token_weight.sum().clamp(min=1)
    return loss.sum() / denom


def _z_loss(logits: Tensor, mask: Tensor) -> Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    weight = mask.float()
    sq = (log_z * weight).pow(2)
    denom = weight.sum().clamp(min=1.0)
    return sq.sum() / denom


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _move_cond_features(features: ConditionFeatures, device: torch.device) -> ConditionFeatures:
    return ConditionFeatures(
        scalars=features.scalars.to(device, non_blocking=True),
        year_idx=features.year_idx.to(device, non_blocking=True),
        descriptors=features.descriptors.to(device, non_blocking=True),
    )
