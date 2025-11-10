from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import OsuBeatmapDataset, osu_collate, split_train_val_files
from src.models import ConformerSeq2Seq
from src.training import TrainingSession
from src.utils.config import load_config
from torch.cuda.amp import GradScaler, autocast


def build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: dict):
    schedule_cfg = training_cfg.get("lr_schedule")
    if not schedule_cfg:
        return None
    schedule_type = schedule_cfg.get("type", "").lower()
    if schedule_type == "multistep":
        milestones = schedule_cfg.get("milestones", [])
        gamma = schedule_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if schedule_type == "step":
        step_size = schedule_cfg.get("step_size", 10)
        gamma = schedule_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if schedule_type == "cosine":
        t_max = schedule_cfg.get("t_max", training_cfg.get("num_epochs", 100))
        eta_min = schedule_cfg.get("eta_min", schedule_cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    print(f"[WARN] Unknown lr_schedule.type '{schedule_type}'. Skipping scheduler.")
    return None


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
    return parser.parse_args()


def resolve_device(preferred: str) -> torch.device:
    preferred = preferred.lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        preferred = "cpu"
    return torch.device(preferred)


def run_epoch(
    model: ConformerSeq2Seq,
    dataloader: DataLoader,
    device: torch.device,
    eos_loss_weight: float,
    grad_clip: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    amp_enabled: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_coord = 0.0
    total_eos = 0.0
    total_samples = 0
    batch_count = 0
    total_batch_time = 0.0
    total_data_time = 0.0

    iterator = tqdm(dataloader, desc="Train" if is_train else "Eval", leave=False)
    data_start = time.perf_counter()
    for batch in iterator:
        now = time.perf_counter()
        data_elapsed = now - data_start
        total_data_time += data_elapsed
        batch_start = now

        audio = batch["audio"].to(device)
        tokens = batch["tokens"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        token_mask = batch["token_mask"].to(device)

        with autocast(enabled=amp_enabled):
            coords_pred, eos_logits = model(audio, audio_mask, tokens, token_mask)

            target_coords = tokens[..., :3]
            target_eos = tokens[..., 3:]
            valid_mask = (~token_mask).unsqueeze(-1).float()
            mask_total = valid_mask.sum().clamp_min(1.0)

            coord_loss = F.smooth_l1_loss(coords_pred, target_coords, reduction="none")
            coord_loss = (coord_loss * valid_mask).sum() / (mask_total * target_coords.size(-1))

            with torch.no_grad():
                pos_eos = (target_eos * valid_mask).sum().clamp_min(1.0)
                neg_eos = ((1.0 - target_eos) * valid_mask).sum().clamp_min(1.0)
                eos_pos_weight = (neg_eos / pos_eos).clamp_min(1.0)

            eos_loss = F.binary_cross_entropy_with_logits(
                eos_logits,
                target_eos,
                reduction="none",
                pos_weight=eos_pos_weight,
            )
            eos_loss = (eos_loss * valid_mask).sum() / mask_total

            loss = coord_loss + eos_loss_weight * eos_loss

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
        total_coord += coord_loss.item() * batch_size
        total_eos += eos_loss.item() * batch_size
        total_samples += batch_size
        batch_count += 1

        batch_elapsed = time.perf_counter() - batch_start
        total_batch_time += batch_elapsed

        iterator.set_postfix(loss=loss.item(), data=data_elapsed, batch=batch_elapsed)
        data_start = time.perf_counter()

    if total_samples == 0:
        return {"loss": 0.0, "coord_loss": 0.0, "eos_loss": 0.0}

    return {
        "loss": total_loss / total_samples,
        "coord_loss": total_coord / total_samples,
        "eos_loss": total_eos / total_samples,
        "avg_batch_time": total_batch_time / batch_count if batch_count else 0.0,
        "avg_data_time": total_data_time / batch_count if batch_count else 0.0,
        "samples_per_sec": (total_samples / total_batch_time) if total_batch_time > 0 else 0.0,
    }


def save_checkpoint(path: Path, model: ConformerSeq2Seq, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    scheduler_state = metrics.get("_scheduler_state")
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
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

    resume_checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    resume_dir = resume_checkpoint_path.parent if resume_checkpoint_path else None
    session = TrainingSession(config, resume_path=str(resume_dir) if resume_dir else None)
    print(f"[INFO] Session directory: {session.path}")

    device = resolve_device(training_cfg.get("device", "cpu"))
    batch_size = training_cfg["batch_size"]
    num_epochs = args.epochs or training_cfg["num_epochs"]
    num_workers = training_cfg.get("num_workers", 0)
    eos_loss_weight = training_cfg.get("eos_loss_weight", 1.0)
    grad_clip = training_cfg.get("grad_clip", 1.0)

    train_files, val_files = split_train_val_files(config)
    train_dataset = OsuBeatmapDataset(config, split="train", osu_files=train_files)
    val_dataset = OsuBeatmapDataset(config, split="val", osu_files=val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=osu_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=osu_collate,
    )

    model = ConformerSeq2Seq(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    scheduler = build_scheduler(optimizer, training_cfg)

    metrics_path = Path(session.path) / "metrics.json"
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []
    start_epoch = 1

    if resume_checkpoint_path:
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{resume_checkpoint_path}' not found.")
        print(f"[INFO] Resuming from checkpoint: {resume_checkpoint_path}")
        payload = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                train_history = metrics_data.get("train", [])
                val_history = metrics_data.get("val", [])
        if scheduler and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
    best_val = (
        min((entry.get("loss", float("inf")) for entry in val_history), default=float("inf"))
        if val_history
        else float("inf")
    )

    if start_epoch > num_epochs:
        print(f"[INFO] Checkpoint epoch ({start_epoch - 1}) >= target epochs ({num_epochs}). Nothing to do.")
        return

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"[INFO] Epoch {epoch}/{num_epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            eos_loss_weight,
            grad_clip,
            optimizer=optimizer,
            amp_enabled=use_amp,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            eos_loss_weight,
            grad_clip,
            optimizer=None,
            amp_enabled=use_amp,
            scaler=None,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"       Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.6f}")
        if train_metrics.get("avg_batch_time", 0.0) > 0:
            print(
                "       Throughput: "
                f"{train_metrics['samples_per_sec']:.2f} samples/s "
                f"(batch {train_metrics['avg_batch_time']:.3f}s | data {train_metrics['avg_data_time']:.3f}s)"
            )

        checkpoint_metrics = val_metrics.copy()
        if scheduler:
            checkpoint_metrics["_scheduler_state"] = scheduler.state_dict()

        latest_path = Path(session.path) / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, checkpoint_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(Path(session.path) / "best.pt", model, optimizer, epoch, checkpoint_metrics)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"train": train_history, "val": val_history}, f, indent=2)

        if scheduler:
            scheduler.step()
        plot_curves(train_history, val_history, Path(session.path) / "loss_curve.png")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
