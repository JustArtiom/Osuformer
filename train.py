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
from src.data.tokenizer import TokenAttr, TokenType
from src.models import ConformerSeq2Seq
from src.training import TrainingSession
from src.utils.config import load_config
from torch.cuda.amp import autocast


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
    grad_clip: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    amp_enabled: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    class_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
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
            attr_logits = model(audio, audio_mask, tokens, token_mask)
            valid_mask = ~token_mask
            type_targets = tokens[..., TokenAttr.TYPE]

            attr_losses: List[torch.Tensor] = []
            for attr_idx, logits in enumerate(attr_logits):
                target = tokens[..., attr_idx]
                if attr_idx == TokenAttr.TYPE:
                    attr_mask = valid_mask
                elif attr_idx == TokenAttr.DELTA:
                    attr_mask = valid_mask & (type_targets != TokenType.EOS)
                elif attr_idx in (TokenAttr.START_X, TokenAttr.START_Y):
                    attr_mask = valid_mask & (type_targets != TokenType.EOS)
                elif attr_idx in (
                    TokenAttr.END_X,
                    TokenAttr.END_Y,
                    TokenAttr.CTRL1_X,
                    TokenAttr.CTRL1_Y,
                    TokenAttr.CTRL2_X,
                    TokenAttr.CTRL2_Y,
                    TokenAttr.DURATION,
                    TokenAttr.SLIDES,
                    TokenAttr.CURVE_TYPE,
                    TokenAttr.SLIDER_SV,
                ):
                    attr_mask = valid_mask & (type_targets == TokenType.SLIDER)
                else:
                    attr_mask = valid_mask

                flat_mask = attr_mask.reshape(-1).float()
                if flat_mask.sum() == 0:
                    continue

                flat_logits = logits.view(-1, logits.size(-1))
                flat_target = target.view(-1)
                kwargs = {}
                if attr_idx == TokenAttr.TYPE:
                    counts = torch.bincount(flat_target, minlength=3).float()
                    total = counts.sum().clamp_min(1.0)
                    freq = counts / total
                    empty_freq = freq[TokenType.EOS].clamp_min(1e-6)
                    other_freq = (1 - empty_freq).clamp_min(1e-6)
                    weight_empty = float(class_weights.get("empty_type", 2.0)) if class_weights else 2.0
                    weight_circle = float(class_weights.get("circle_type", 1.0)) if class_weights else 1.0
                    weight_slider = float(class_weights.get("slider_type", 1.0)) if class_weights else 1.0
                    ratio = (empty_freq / other_freq).item() if other_freq > 0 else 1.0
                    weight_circle *= ratio
                    weight_slider *= ratio
                    weights = torch.tensor(
                        [weight_empty, weight_circle, weight_slider],
                        device=device,
                        dtype=torch.float32,
                    )
                    kwargs["weight"] = weights

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

            if not attr_losses:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = torch.stack(attr_losses).mean()

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

    return {
        "loss": (total_loss / total_samples) if total_samples else 0.0,
        "avg_batch_time": total_batch_time / batch_count if batch_count else 0.0,
        "avg_data_time": total_data_time / batch_count if batch_count else 0.0,
        "samples_per_sec": (total_samples / total_batch_time) if total_batch_time > 0 else 0.0,
    }


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

    resume_checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    resume_dir = resume_checkpoint_path.parent if resume_checkpoint_path else None
    session = TrainingSession(config, resume_path=str(resume_dir) if resume_dir else None)
    print(f"[INFO] Session directory: {session.path}")

    device = resolve_device(training_cfg.get("device", "cpu"))
    batch_size = training_cfg["batch_size"]
    num_epochs = args.epochs or training_cfg["num_epochs"]
    num_workers = training_cfg.get("num_workers", 0)
    grad_clip = training_cfg.get("grad_clip", 1.0)

    train_files, val_files = split_train_val_files(config)
    cache_base = Path(args.map_cache).expanduser().resolve() if args.map_cache else None
    train_cache = cache_base.with_suffix(".train.npz") if cache_base else None
    val_cache = cache_base.with_suffix(".val.npz") if cache_base else None

    train_dataset = OsuBeatmapDataset(config, split="train", osu_files=train_files, cache_path=train_cache)
    val_dataset = OsuBeatmapDataset(config, split="val", osu_files=val_files, cache_path=val_cache)

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

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
            grad_clip,
            optimizer=optimizer,
            amp_enabled=use_amp,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
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

        latest_path = Path(session.path) / "latest.pt"
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
            tokenizer_meta=model.tokenizer_meta(),
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                Path(session.path) / "best.pt",
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                tokenizer_meta=model.tokenizer_meta(),
            )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"train": train_history, "val": val_history}, f, indent=2)

        plot_curves(train_history, val_history, Path(session.path) / "loss_curve.png")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
