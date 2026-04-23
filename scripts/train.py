from __future__ import annotations

from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.cache import CacheReader
from src.config import with_config
from src.config.schemas.app import AppConfig
from src.model import Osuformer
from src.osu_tokenizer import Vocab
from src.training.data import Collator, OsuDataset, split_beatmap_ids
from src.training.distributed import destroy_distributed, setup_distributed
from src.training.trainer import Trainer


@click.command()
@click.option("--epoch-length", default=10000, type=int)
@click.option("--train-ratio", default=0.95, type=float)
@with_config
def main(cfg: AppConfig, epoch_length: int, train_ratio: float) -> None:
    load_dotenv()
    dist_env = setup_distributed()
    torch.manual_seed(cfg.training.seed + dist_env.global_rank)

    if dist_env.enabled and torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_env.local_rank}")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    if dist_env.is_main:
        print(f"device: {device}  world_size: {dist_env.world_size}")
        print(f"run: {cfg.training.run_name}  cache: {cfg.training.cache_name}")

    vocab = Vocab(cfg.tokenizer)
    if dist_env.is_main:
        print(f"loading cache (preload={cfg.training.cache_preload})...", flush=True)
    reader = CacheReader(
        cache_root=Path(cfg.paths.cache),
        name=cfg.training.cache_name,
        preload=cfg.training.cache_preload,
    )
    all_ids = reader.map_ids()
    if not all_ids:
        raise SystemExit(f"no maps in cache {cfg.paths.cache}/{cfg.training.cache_name}")
    splits = split_beatmap_ids(all_ids, train_ratio=train_ratio, seed=cfg.training.seed)
    if dist_env.is_main:
        print(f"train: {len(splits.train)} maps, val: {len(splits.val)} maps")

    train_ds = OsuDataset(
        cache_root=Path(cfg.paths.cache),
        cache_name=cfg.training.cache_name,
        beatmap_ids=splits.train,
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_cfg=cfg.audio,
        max_decoder_len=cfg.training.max_decoder_len,
        history_event_count=cfg.training.history_event_count,
        epoch_length=epoch_length,
        seed=cfg.training.seed,
        reader=reader,
        timing_jitter_bins=cfg.training.timing_jitter_bins,
    )
    val_ds = OsuDataset(
        cache_root=Path(cfg.paths.cache),
        cache_name=cfg.training.cache_name,
        beatmap_ids=splits.val or splits.train[: max(1, len(splits.train) // 20)],
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_cfg=cfg.audio,
        max_decoder_len=cfg.training.max_decoder_len,
        history_event_count=cfg.training.history_event_count,
        epoch_length=max(512, epoch_length // 20),
        seed=cfg.training.seed + 1,
        reader=reader,
        timing_jitter_bins=0,
    )
    collator = Collator(
        vocab=vocab,
        rhythm_weight=cfg.training.rhythm_loss_weight,
        slider_weight=cfg.training.slider_loss_weight,
    )

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=dist_env.world_size,
            rank=dist_env.global_rank,
            shuffle=True,
            seed=cfg.training.seed,
        )
        if dist_env.enabled
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=dist_env.world_size,
            rank=dist_env.global_rank,
            shuffle=False,
            seed=cfg.training.seed,
        )
        if dist_env.enabled
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collator,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=min(2, cfg.training.num_workers),
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collator,
    )

    model = Osuformer(
        model_cfg=cfg.model,
        audio_cfg=cfg.audio,
        vocab_size_in=vocab.vocab_size_in,
        vocab_size_out=vocab.vocab_size_out,
        max_decoder_len=cfg.training.max_decoder_len,
    )
    if dist_env.is_main:
        print(f"model params: {model.num_parameters()/1e6:.1f}M")

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        dist_env=dist_env,
    )
    try:
        trainer.fit()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
