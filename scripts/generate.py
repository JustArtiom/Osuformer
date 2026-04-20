from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch

from src.cache.audio import compute_mel
from src.config import with_config
from src.config.schemas.app import AppConfig
from src.inference import (
    GenerationPrompt,
    SamplingConfig,
    WindowGenerator,
    events_to_beatmap,
)
from src.model import Osuformer
from src.osu_tokenizer import Vocab


@click.command()
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--stars", default=None, type=float)
@click.option("--year", default=None, type=int)
@click.option("--descriptors", default="", type=str, help="Comma-separated descriptor tags.")
@click.option("--cs", default=4.0, type=float)
@click.option("--ar", default=9.0, type=float)
@click.option("--od", default=8.0, type=float)
@click.option("--hp", default=6.0, type=float)
@click.option("--slider-multiplier", default=1.4, type=float)
@click.option("--temperature", default=1.0, type=float)
@click.option("--time-temperature", default=0.6, type=float)
@click.option("--top-p", default=0.95, type=float)
@click.option("--top-k", default=0, type=int)
@click.option("--title", default="Generated", type=str)
@click.option("--artist", default="osuformer", type=str)
@click.option("--creator", default="osuformer", type=str)
@click.option("--version", "diff_version", default="Generated", type=str)
@with_config
def main(
    cfg: AppConfig,
    audio: Path,
    checkpoint: Path,
    out_path: Path,
    stars: float | None,
    year: int | None,
    descriptors: str,
    cs: float,
    ar: float,
    od: float,
    hp: float,
    slider_multiplier: float,
    temperature: float,
    time_temperature: float,
    top_p: float,
    top_k: int,
    title: str,
    artist: str,
    creator: str,
    diff_version: str,
) -> None:
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")

    print(f"computing mel for {audio.name}...")
    mel = compute_mel(audio, cfg.audio).astype(np.float32)
    song_duration_ms = _song_duration_ms(audio)
    print(f"audio: {mel.shape[0]} frames ({song_duration_ms/1000:.1f}s)")

    vocab = Vocab(cfg.tokenizer)
    model = Osuformer(
        model_cfg=cfg.model,
        audio_cfg=cfg.audio,
        vocab_size_in=vocab.vocab_size_in,
        vocab_size_out=vocab.vocab_size_out,
        max_decoder_len=cfg.training.max_decoder_len,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"loaded checkpoint from step {ckpt.get('step', '?')}  ({model.num_parameters()/1e6:.1f}M params)")

    sampling = SamplingConfig(
        temperature=temperature,
        time_temperature=time_temperature,
        top_p=top_p,
        top_k=top_k,
    )
    generator = WindowGenerator(
        model=model,
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_cfg=cfg.audio,
        device=device,
        max_decoder_len=cfg.training.max_decoder_len,
        history_event_count=cfg.training.history_event_count,
        sampling=sampling,
    )

    prompt = GenerationPrompt(
        star_rating=stars,
        descriptors=[d.strip() for d in descriptors.split(",") if d.strip()],
        year=year,
        hitsounded=True,
        cs=cs,
        ar=ar,
        od=od,
        hp=hp,
        slider_multiplier=slider_multiplier,
        song_length_ms=song_duration_ms,
    )

    print("generating...")
    result = generator.generate(mel=mel, prompt=prompt, song_duration_ms=song_duration_ms)
    print(f"generated {sum(1 for e in result.events if e.type.value == 'abs_time')} hit-object groups across {len(result.window_starts_ms)} windows")

    beatmap = events_to_beatmap(
        result.events,
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_filename=audio.name,
        title=title,
        artist=artist,
        creator=creator,
        version=diff_version,
        circle_size=cs,
        approach_rate=ar,
        overall_difficulty=od,
        hp_drain_rate=hp,
        slider_multiplier=slider_multiplier,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(beatmap))
    print(f"wrote {out_path}  ({len(beatmap.hit_objects)} hit objects)")


def _song_duration_ms(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate * 1000.0


if __name__ == "__main__":
    main()
