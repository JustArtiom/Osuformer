from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from contextlib import nullcontext

from src.data import OsuBeatmapDataset, osu_collate, split_train_val_files
from src.data.tokenizer import TokenAttr, TokenType
from src.models import ConformerSeq2Seq
from src.training import TrainingSession
from src.utils.config import load_config


def log_dataset_summary(dataset: OsuBeatmapDataset, split: str) -> None:
    sample_count = len(dataset)
    tokenizer = getattr(dataset, "tokenizer", None)
    attr_sizes = getattr(tokenizer, "attribute_sizes", []) if tokenizer else []
    print(
        f"[INFO] Dataset[{split}] samples={sample_count} ticks_per_sample={dataset.seq_len} "
        f"context_beats={dataset.context_beats} target_beats={dataset.target_beats}"
    )
    print(
        f"[INFO] Dataset[{split}] tokenizer attr_sizes={attr_sizes} sequence_len={getattr(tokenizer, 'seq_len', 'n/a')}"
    )
    if dataset.samples:
        descriptor = dataset.samples[0]
        print(
            f"[INFO] Sample[{split}] audio_frames={descriptor.frames_per_chunk} "
            f"token_len={descriptor.token_length} loss_mask_shape={descriptor.loss_mask.shape}"
        )
    else:
        print(f"[INFO] Dataset[{split}] has no cached samples yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the osu! beatmap generator model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from (e.g., checkpoints/8/latest.pt).",
    )
    parser.add_argument(
        "--map-cache",
        type=str,
        default=None,
        help="Optional base path for dataset caches (.train.npz/.val.npz). If provided, caches are reused.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights (no gradient updates). Useful for fine-tuning the decoder only.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DistributedDataParallel. Launch with torchrun and set --distributed to true.",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default=None,
        help="Override the DDP backend (defaults to training.distributed.backend).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size.",
    )
    parser.add_argument(
        "--use-ram-for-spec",
        action="store_true",
        help="Force data.use_ram_for_spec=true (keep spectrograms in RAM).",
    )
    parser.add_argument(
        "--no-use-ram-for-spec",
        action="store_true",
        help="Force data.use_ram_for_spec=false (load spectrograms from disk).",
    )
    return parser.parse_args()


def resolve_device(preferred: str) -> torch.device:
    preferred = preferred.lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        preferred = "cpu"
    return torch.device(preferred)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def run_epoch(
    model: ConformerSeq2Seq,
    dataloader: DataLoader,
    device: torch.device,
    grad_clip: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    amp_enabled: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    is_main_process: bool = True,
    distributed: bool = False,
    tick_strict_supervision: bool = False,
    tick_penalty_weight: float = 0.0,
    overlap_penalty: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    total_batch_time = 0.0
    total_data_time = 0.0

    iterator = tqdm(
        dataloader,
        desc="Train" if is_train else "Eval",
        leave=False,
        disable=not is_main_process,
    )
    data_start = time.perf_counter()
    for batch_idx, batch in enumerate(iterator):
        now = time.perf_counter()
        data_elapsed = now - data_start
        total_data_time += data_elapsed
        batch_start = now

        audio = batch["audio"].to(device)
        tokens = batch["tokens"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        token_mask = batch["token_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        if amp_enabled:
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            attr_logits = model(audio, audio_mask, tokens, token_mask)
            valid_mask = (~token_mask) & loss_mask
            type_targets = tokens[..., TokenAttr.TYPE]

            type_logits = attr_logits[TokenAttr.TYPE]
            tick_logits = attr_logits[TokenAttr.TICK]
            tick_targets = tokens[..., TokenAttr.TICK]
            tick_pred = tick_logits.argmax(dim=-1)
            tick_match = tick_pred.eq(tick_targets)
            type_probs = F.softmax(type_logits, dim=-1)
            tick_probs = F.softmax(tick_logits, dim=-1)
            tick_bin_count = tick_probs.size(-1)
            valid_float = valid_mask.float()
            event_prob = (type_probs[..., TokenType.CIRCLE] + type_probs[..., TokenType.SLIDER]) * valid_float

            attr_losses: List[torch.Tensor] = []
            for attr_idx, logits in enumerate(attr_logits):
                target = tokens[..., attr_idx]
                if attr_idx == TokenAttr.TYPE:
                    attr_mask = valid_mask
                elif attr_idx in (TokenAttr.TICK, TokenAttr.X, TokenAttr.Y):
                    attr_mask = valid_mask & (type_targets != TokenType.EOS)
                elif attr_idx in (
                    TokenAttr.DURATION,
                    TokenAttr.SLIDES,
                    TokenAttr.CURVE_TYPE,
                    TokenAttr.SLIDER_SV,
                ):
                    attr_mask = valid_mask & (type_targets == TokenType.SLIDER)
                else:
                    attr_mask = valid_mask

                if tick_strict_supervision and attr_idx != TokenAttr.TICK:
                    attr_mask = attr_mask & tick_match

                flat_mask = attr_mask.reshape(-1).float()
                if flat_mask.sum() == 0:
                    # Keep the graph connected even when a local batch has no
                    # target tokens for this attribute; prevents DDP hangs when
                    # another rank does have targets.
                    attr_losses.append(logits.float().sum() * 0.0)
                    continue

                flat_logits = logits.view(-1, logits.size(-1))
                flat_target = target.view(-1)
                kwargs = {}
                loss_attr = F.cross_entropy(
                    flat_logits,
                    flat_target,
                    reduction="none",
                    ignore_index=0,
                    **kwargs,
                )
                loss_attr = loss_attr * flat_mask
                loss_attr = loss_attr.sum() / flat_mask.sum().clamp_min(1.0)
                attr_losses.append(loss_attr)

            loss = torch.stack(attr_losses).mean() if attr_losses else torch.tensor(0.0, device=device, requires_grad=True)
            total_valid = valid_mask.float().sum().clamp_min(1.0)
            if tick_penalty_weight > 0.0:
                target_tick_prob = tick_probs.gather(-1, tick_targets.unsqueeze(-1)).squeeze(-1)
                tick_penalty = ((1.0 - target_tick_prob) * valid_mask.float()).sum() / total_valid
                loss = loss + tick_penalty_weight * tick_penalty

            if overlap_penalty > 0.0:
                batch_size = audio.size(0)
                pred_tick_usage = torch.zeros(batch_size, tick_bin_count, device=device, dtype=tick_probs.dtype)
                # Circles
                circle_prob = type_probs[..., TokenType.CIRCLE] * valid_float
                circle_usage = circle_prob.unsqueeze(-1) * tick_probs
                pred_tick_usage += circle_usage.sum(dim=1)

                # Sliders coverage using ground-truth durations
                slider_gt_mask = (type_targets == TokenType.SLIDER) & valid_mask
                if slider_gt_mask.any():
                    slider_prob = type_probs[..., TokenType.SLIDER] * valid_float
                    slider_tick_mass = slider_prob.unsqueeze(-1) * tick_probs
                    slider_tick_mass = slider_tick_mass[slider_gt_mask]
                    slider_durations = torch.clamp(tokens[..., TokenAttr.DURATION] - 1, min=0)[slider_gt_mask]
                    slider_batch = slider_gt_mask.nonzero(as_tuple=False)[:, 0]
                    if slider_tick_mass.numel() > 0:
                        unique_durations = slider_durations.unique()
                        for duration_value in unique_durations:
                            dur_mask = slider_durations == duration_value
                            if not dur_mask.any():
                                continue
                            mass_group = slider_tick_mass[dur_mask]  # [G, T]
                            batch_group = slider_batch[dur_mask]
                            kernel_size = int(duration_value.item()) + 1
                            kernel = torch.ones(1, 1, kernel_size, device=device, dtype=mass_group.dtype)
                            cover = F.conv1d(mass_group.unsqueeze(1), kernel, padding=kernel_size - 1)[:, 0, :tick_bin_count]
                            pred_tick_usage.index_add_(0, batch_group, cover)

                target_tick_usage = torch.zeros(batch_size, tick_bin_count, device=device, dtype=tick_probs.dtype)
                tick_ids = torch.clamp(tick_targets.clamp_min(1) - 1, min=0, max=tick_bin_count - 1)
                circle_gt_mask = (type_targets == TokenType.CIRCLE) & valid_mask
                if circle_gt_mask.any():
                    circle_batch, circle_pos = circle_gt_mask.nonzero(as_tuple=True)
                    circle_ticks = tick_ids[circle_batch, circle_pos]
                    target_tick_usage.index_put_(
                        (circle_batch, circle_ticks),
                        torch.ones_like(circle_ticks, dtype=tick_probs.dtype),
                        accumulate=True,
                    )

                if slider_gt_mask.any():
                    slider_batch_gt, slider_pos_gt = slider_gt_mask.nonzero(as_tuple=True)
                    slider_ticks_gt = tick_ids[slider_batch_gt, slider_pos_gt]
                    slider_dur_gt = torch.clamp(tokens[..., TokenAttr.DURATION] - 1, min=0)[slider_gt_mask]
                    for idx in range(slider_batch_gt.size(0)):
                        b_idx = slider_batch_gt[idx].item()
                        start_tick = slider_ticks_gt[idx].item()
                        duration = slider_dur_gt[idx].item()
                        end_tick = min(start_tick + duration, tick_bin_count - 1)
                        target_tick_usage[b_idx, start_tick : end_tick + 1] += 1.0

                overlap = torch.relu(pred_tick_usage - target_tick_usage)
                overlap_penalty_value = overlap.sum(dim=1).mean()
                loss = loss + overlap_penalty * overlap_penalty_value

        if is_train:
            optimizer.zero_grad()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = audio.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_count += 1

        batch_elapsed = time.perf_counter() - batch_start
        total_batch_time += batch_elapsed

        iterator.set_postfix(loss=loss.item(), data=data_elapsed, batch=batch_elapsed)
        data_start = time.perf_counter()

    if distributed:
        stats = torch.tensor(
            [
                total_loss,
                float(total_samples),
                total_batch_time,
                float(batch_count),
                total_data_time,
            ],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples, total_batch_time, batch_count, total_data_time = stats.tolist()

    metrics = {
        "loss": (total_loss / total_samples) if total_samples else 0.0,
        "avg_batch_time": total_batch_time / batch_count if batch_count else 0.0,
        "avg_data_time": total_data_time / batch_count if batch_count else 0.0,
        "samples_per_sec": (total_samples / total_batch_time) if total_batch_time > 0 else 0.0,
    }
    return metrics


def save_checkpoint(
    path: Path,
    model: ConformerSeq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    *,
    tokenizer_meta: Dict[str, object],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "tokenizer_meta": tokenizer_meta,
        "data_cfg": model.data_cfg_snapshot,
    }
    torch.save(payload, path)


def plot_curves(history: List[Dict[str, float]], val_history: List[Dict[str, float]], output_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping loss plot.")
        return
    epochs = range(1, len(history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [m["loss"] for m in history], label="Train Loss")
    plt.plot(epochs, [m["loss"] for m in val_history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_cfg = config["training"]
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.use_ram_for_spec and args.no_use_ram_for_spec:
        raise ValueError("Cannot set both --use-ram-for-spec and --no-use-ram-for-spec.")
    if args.use_ram_for_spec:
        config["data"]["use_ram_for_spec"] = True
        config["data"]["use_spec_cache"] = True  # backward compatibility with older configs
    if args.no_use_ram_for_spec:
        config["data"]["use_ram_for_spec"] = False
        config["data"]["use_spec_cache"] = False  # backward compatibility with older configs
    checkpoint_root = Path(training_cfg["path"]).expanduser().resolve()

    dist_cfg = training_cfg.get("distributed", {})
    if isinstance(dist_cfg, bool):
        dist_cfg = {"enabled": dist_cfg}
    distributed_enabled = bool(args.distributed or dist_cfg.get("enabled", False))
    ddp_backend = args.ddp_backend or dist_cfg.get("backend", "nccl")

    rank = 0
    world_size = 1
    local_rank = 0

    if distributed_enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        dist.init_process_group(backend=ddp_backend)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if training_cfg.get("device", "cuda") != "cuda":
            raise ValueError("Distributed training requires training.device to be 'cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but distributed training was requested.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(training_cfg.get("device", "cpu"))

    is_main_process = rank == 0

    resume_checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    resume_dir = resume_checkpoint_path.parent if resume_checkpoint_path else None
    session = TrainingSession(config, resume_path=str(resume_dir) if resume_dir else None)
    if is_main_process:
        print(f"[INFO] Session directory: {session.path}")
        if distributed_enabled:
            print(f"[INFO] Using DistributedDataParallel with world size {world_size} (backend={ddp_backend}).")

    batch_size = training_cfg["batch_size"]
    num_epochs = args.epochs or training_cfg["num_epochs"]
    num_workers = training_cfg.get("num_workers", 0)
    grad_clip = training_cfg.get("grad_clip", 1.0)
    tick_strict = bool(training_cfg.get("tick_strict_supervision", False))
    tick_penalty_weight = float(training_cfg.get("tick_miss_penalty", 0.0) or 0.0)
    overlap_penalty_weight = float(
        training_cfg.get("overlap_penalty", training_cfg.get("duplicate_tick_penalty", 0.0)) or 0.0
    )

    train_files, val_files = split_train_val_files(config)
    cache_base = Path(args.map_cache).expanduser().resolve() if args.map_cache else None
    train_cache = Path(str(cache_base) + ".train.npz") if cache_base else None
    val_cache = Path(str(cache_base) + ".val.npz") if cache_base else None

    train_dataset = OsuBeatmapDataset(config, split="train", osu_files=train_files, cache_path=train_cache)
    val_dataset = OsuBeatmapDataset(config, split="val", osu_files=val_files, cache_path=val_cache)
    if is_main_process:
        log_dataset_summary(train_dataset, "train")
        log_dataset_summary(val_dataset, "val")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed_enabled
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed_enabled
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not distributed_enabled,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=osu_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=osu_collate,
    )

    model = ConformerSeq2Seq(config).to(device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if is_main_process:
            print("[INFO] Freezing encoder parameters (decoder-only training).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after freezing. Check --freeze-encoder usage.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if distributed_enabled:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    metrics_path = Path(session.path) / "metrics.json"
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []
    start_epoch = 1

    if resume_checkpoint_path:
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{resume_checkpoint_path}' not found.")
        if is_main_process:
            print(f"[INFO] Resuming from checkpoint: {resume_checkpoint_path}")
        payload = torch.load(resume_checkpoint_path, map_location=device)
        unwrap_model(model).load_state_dict(payload["model_state"])
        if args.freeze_encoder:
            if is_main_process:
                print("[INFO] Skipping optimizer state load because encoder is frozen.")
        else:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                train_history = metrics_data.get("train", [])
                val_history = metrics_data.get("val", [])
    best_val = (
        min((entry.get("loss", float("inf")) for entry in val_history), default=float("inf"))
        if val_history
        else float("inf")
    )

    if start_epoch > num_epochs:
        if is_main_process:
            print(f"[INFO] Checkpoint epoch ({start_epoch - 1}) >= target epochs ({num_epochs}). Nothing to do.")
        if distributed_enabled:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            print(f"[INFO] Epoch {epoch}/{num_epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            grad_clip,
            optimizer=optimizer,
            amp_enabled=use_amp,
            scaler=scaler,
            is_main_process=is_main_process,
            distributed=distributed_enabled,
            tick_strict_supervision=tick_strict,
            tick_penalty_weight=tick_penalty_weight,
            overlap_penalty=overlap_penalty_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            grad_clip,
            optimizer=None,
            amp_enabled=use_amp,
            scaler=None,
            is_main_process=is_main_process,
            distributed=distributed_enabled,
            tick_strict_supervision=tick_strict,
            tick_penalty_weight=tick_penalty_weight,
            overlap_penalty=overlap_penalty_weight,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"       Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.6f}"
            )
            if train_metrics.get("avg_batch_time", 0.0) > 0:
                print(
                    "       Throughput: "
                    f"{train_metrics['samples_per_sec']:.2f} samples/s "
                    f"(batch {train_metrics['avg_batch_time']:.3f}s | data {train_metrics['avg_data_time']:.3f}s)"
                )

        checkpoint_metrics = val_metrics.copy()
        if is_main_process:
            session.ensure_created()
            latest_path = Path(session.path) / "latest.pt"
            base_model = unwrap_model(model)
            save_checkpoint(
                latest_path,
                base_model,
                optimizer,
                epoch,
                checkpoint_metrics,
                tokenizer_meta=base_model.tokenizer_meta(),
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    Path(session.path) / "best.pt",
                    base_model,
                    optimizer,
                    epoch,
                    checkpoint_metrics,
                    tokenizer_meta=base_model.tokenizer_meta(),
                )

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({"train": train_history, "val": val_history}, f, indent=2)

            plot_curves(train_history, val_history, Path(session.path) / "loss_curve.png")

    if is_main_process:
        print("[INFO] Training complete.")

    if distributed_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
