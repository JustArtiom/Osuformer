from __future__ import annotations

from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.cache import CacheReader
from src.config import with_config
from src.config.schemas.app import AppConfig
from src.model import Osuformer
from src.osu_tokenizer import Vocab
from src.training.data import Collator, OsuDataset, split_beatmap_ids
from src.training.trainer import Trainer


@click.command()
@click.option("--epoch-length", default=10000, type=int)
@click.option("--train-ratio", default=0.95, type=float)
@with_config
def main(cfg: AppConfig, epoch_length: int, train_ratio: float) -> None:
    load_dotenv()
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")
    print(f"run: {cfg.training.run_name}  cache: {cfg.training.cache_name}")

    vocab = Vocab(cfg.tokenizer)
    reader = CacheReader(cache_root=Path(cfg.paths.cache), name=cfg.training.cache_name)
    all_ids = reader.map_ids()
    if not all_ids:
        raise SystemExit(f"no maps in cache {cfg.paths.cache}/{cfg.training.cache_name}")
    splits = split_beatmap_ids(all_ids, train_ratio=train_ratio, seed=cfg.training.seed)
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
        preload=cfg.training.cache_preload,
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
        epoch_length=max(32, epoch_length // 100),
        seed=cfg.training.seed + 1,
        preload=cfg.training.cache_preload,
    )
    collator = Collator(vocab=vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
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
    print(f"model params: {model.num_parameters()/1e6:.1f}M")

    trainer = Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()
