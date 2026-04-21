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
from src.osu_tokenizer import EventType, Vocab


@click.command()
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--stars", default=None, type=float)
@click.option("--year", default=None, type=int)
@click.option("--bpm", default=180.0, type=float, help="Song BPM used for timing points + slider length math.")
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
@click.option("--circle-bias", default=0.0, type=float, help="Additive logit bias for CIRCLE marker; positive = more circles.")
@click.option("--slider-bias", default=0.0, type=float, help="Additive logit bias for SLIDER_HEAD marker; negative = fewer sliders.")
@click.option("--spinner-bias", default=0.0, type=float, help="Additive logit bias for SPINNER marker.")
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
    bpm: float,
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
    circle_bias: float,
    slider_bias: float,
    spinner_bias: float,
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

    event_bias: dict[EventType, float] = {}
    if circle_bias != 0:
        event_bias[EventType.CIRCLE] = circle_bias
    if slider_bias != 0:
        event_bias[EventType.SLIDER_HEAD] = slider_bias
    if spinner_bias != 0:
        event_bias[EventType.SPINNER] = spinner_bias

    sampling = SamplingConfig(
        temperature=temperature,
        time_temperature=time_temperature,
        top_p=top_p,
        top_k=top_k,
        event_bias=event_bias,
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
    _print_event_breakdown(result.events, len(result.window_starts_ms))

    beatmap = events_to_beatmap(
        result.events,
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_filename=audio.name,
        bpm=bpm,
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
    _print_beatmap_breakdown(beatmap, out_path)


def _print_event_breakdown(events: list, window_count: int) -> None:
    from src.osu_tokenizer import EventType

    total_groups = sum(1 for e in events if e.type == EventType.ABS_TIME)
    circles = sum(1 for e in events if e.type == EventType.CIRCLE)
    slider_heads = sum(1 for e in events if e.type == EventType.SLIDER_HEAD)
    slider_ends = sum(1 for e in events if e.type == EventType.SLIDER_END)
    last_anchors = sum(1 for e in events if e.type == EventType.LAST_ANCHOR)
    spinners = sum(1 for e in events if e.type == EventType.SPINNER)
    spinner_ends = sum(1 for e in events if e.type == EventType.SPINNER_END)
    timing_points = sum(1 for e in events if e.type == EventType.TIMING_POINT)
    scroll_speeds = sum(1 for e in events if e.type == EventType.SCROLL_SPEED)
    anchors = sum(
        1
        for e in events
        if e.type
        in {
            EventType.BEZIER_ANCHOR,
            EventType.PERFECT_ANCHOR,
            EventType.CATMULL_ANCHOR,
            EventType.LINEAR_ANCHOR,
            EventType.RED_ANCHOR,
        }
    )
    print(f"generated across {window_count} windows ({len(events)} events, {total_groups} ABS_TIME groups)")
    print(f"  markers: {circles} circles  {slider_heads} slider_heads  {spinners} spinners")
    print(f"  closes:  {last_anchors} last_anchors  {slider_ends} slider_ends  {spinner_ends} spinner_ends")
    print(f"  inside:  {anchors} anchor/red events")
    print(f"  timing:  {timing_points} timing_points  {scroll_speeds} scroll_speeds")
    open_sliders = slider_heads - slider_ends
    open_spinners = spinners - spinner_ends
    if open_sliders or open_spinners:
        print(f"  WARN:    {open_sliders} unclosed sliders, {open_spinners} unclosed spinners")


def _print_beatmap_breakdown(beatmap, out_path) -> None:
    from src.osu.hit_object import Circle, Slider, Spinner

    n_circles = sum(1 for h in beatmap.hit_objects if isinstance(h, Circle))
    n_sliders = sum(1 for h in beatmap.hit_objects if isinstance(h, Slider))
    n_spinners = sum(1 for h in beatmap.hit_objects if isinstance(h, Spinner))
    total = len(beatmap.hit_objects)
    print(f"wrote {out_path}  ({total} hit objects: {n_circles} circles, {n_sliders} sliders, {n_spinners} spinners, {len(beatmap.timing_points)} timing_points)")


def _song_duration_ms(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate * 1000.0


if __name__ == "__main__":
    main()
