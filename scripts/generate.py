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
@click.option("--bpm", default=0.0, type=float, help="Song BPM. Use 0 to auto-infer from model's BEAT tokens (requires new-tokenizer model).")
@click.option("--offset", "offset_ms", default=None, type=float, help="Timing point offset in ms (the song's first downbeat). When set, overrides auto-timing's offset derivation and the model's TIMING_POINT placements. Use with --bpm for full manual override of the rhythm scaffold.")
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
@click.option("--min-spacing-ms", default=30.0, type=float, help="Minimum gap between consecutive hit-object times in ms; prevents same-bin stacking.")
@click.option("--eos-bias", default=0.0, type=float, help="Additive logit bias for EOS; negative = less likely to terminate windows early.")
@click.option("--guidance", "guidance_scale", default=1.0, type=float, help="Classifier-free guidance scale. 1.0 = off (cond only). >1.0 amplifies conditioning (e.g. 2.0–4.0). <1.0 dampens it. Requires a model trained with cfg_dropout_prob > 0.")
@click.option("--no-grammar", "no_grammar", is_flag=True, default=False, help="Disable grammar-based logit masking and state tracking; lets the model emit any token freely. Debug only.")
@click.option("--auto-timing", "auto_timing", is_flag=True, default=False, help="Derive BPM offset and beat_length from emitted BEAT/MEASURE events instead of trusting the model's TIMING_POINT placements. First MEASURE anchors the downbeat.")
@click.option("--snap-subdivision", default=0, type=int, help="Post-process: snap event times to nearest 1/N beat subdivision (1/4=4, 1/8=8, 1/16=16). 0 disables.")
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
    offset_ms: float | None,
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
    min_spacing_ms: float,
    eos_bias: float,
    guidance_scale: float,
    no_grammar: bool,
    auto_timing: bool,
    snap_subdivision: int,
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
        tokenizer_cfg=cfg.tokenizer,
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

    min_spacing_bins = max(0, int(round(min_spacing_ms / cfg.tokenizer.dt_bin_ms)))
    sampling = SamplingConfig(
        temperature=temperature,
        time_temperature=time_temperature,
        top_p=top_p,
        top_k=top_k,
        event_bias=event_bias,
        min_abs_time_spacing_bins=min_spacing_bins,
        eos_bias=eos_bias,
        disable_grammar=no_grammar,
        guidance_scale=guidance_scale,
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
    effective_bpm = _resolve_effective_bpm(result.events, cfg, bpm)
    if effective_bpm is None:
        raise SystemExit("Could not infer BPM from BEAT tokens and --bpm not set; rerun with --bpm <value>.")
    if bpm <= 0:
        print(f"inferred BPM: {effective_bpm:.4f}")
    if offset_ms is not None:
        snap_offset_ms = float(offset_ms)
    elif auto_timing:
        snap_offset_ms = _resolve_offset_ms(result.events, cfg, effective_bpm)
    else:
        snap_offset_ms = 0.0
    if snap_subdivision > 0:
        result.events[:] = _snap_events_to_beat(
            result.events,
            bpm=effective_bpm,
            subdivision=snap_subdivision,
            dt_bin_ms=cfg.tokenizer.dt_bin_ms,
            offset_ms=snap_offset_ms,
        )
    _print_event_breakdown(result.events, len(result.window_starts_ms))

    beatmap = events_to_beatmap(
        result.events,
        vocab=vocab,
        tokenizer_cfg=cfg.tokenizer,
        audio_filename=audio.name,
        bpm=effective_bpm,
        title=title,
        artist=artist,
        creator=creator,
        version=diff_version,
        circle_size=cs,
        approach_rate=ar,
        overall_difficulty=od,
        hp_drain_rate=hp,
        slider_multiplier=slider_multiplier,
        auto_timing=auto_timing,
        offset_ms=offset_ms,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(beatmap))
    _print_beatmap_breakdown(beatmap, out_path)


def _resolve_effective_bpm(events: list, cfg, explicit_bpm: float) -> float | None:
    from src.inference.detokenizer import _group_by_abs_time, _infer_beat_length_ms

    if explicit_bpm > 0:
        return explicit_bpm
    groups = _group_by_abs_time(events)
    beat_length = _infer_beat_length_ms(groups, cfg.tokenizer)
    if beat_length is None:
        return None
    return 60000.0 / beat_length


def _snap_events_to_beat(
    events: list, bpm: float, subdivision: int, dt_bin_ms: int, offset_ms: float = 0.0
) -> list:
    from src.osu_tokenizer import Event, EventType

    beat_ms = 60000.0 / max(1.0, bpm)
    tick_ms = beat_ms / max(1, subdivision)
    out: list = []
    for ev in events:
        if ev.type == EventType.ABS_TIME:
            t_ms = float(ev.value) * dt_bin_ms
            snapped_ms = round((t_ms - offset_ms) / tick_ms) * tick_ms + offset_ms
            snapped_bin = int(round(snapped_ms / dt_bin_ms))
            out.append(Event(EventType.ABS_TIME, snapped_bin))
        else:
            out.append(ev)
    return out


def _resolve_offset_ms(events: list, cfg, bpm: float) -> float:
    from src.osu_tokenizer import EventType
    from src.inference.detokenizer import _group_by_abs_time

    groups = _group_by_abs_time(events)
    measure_times: list[float] = []
    beat_times: list[float] = []
    for abs_bin, group in groups:
        time_ms = float(abs_bin * cfg.tokenizer.dt_bin_ms)
        if any(ev.type == EventType.MEASURE for ev in group):
            measure_times.append(time_ms)
        if any(ev.type == EventType.BEAT for ev in group):
            beat_times.append(time_ms)
    anchor = measure_times[0] if measure_times else (beat_times[0] if beat_times else 0.0)
    if bpm <= 0:
        return anchor
    beat_ms = 60000.0 / bpm
    offset = anchor
    while offset - beat_ms * 4 >= 0:
        offset -= beat_ms * 4
    return offset


def _print_event_breakdown(events: list, window_count: int) -> None:
    from src.osu_tokenizer import EventType

    total_groups = sum(1 for e in events if e.type == EventType.ABS_TIME)
    circles = sum(1 for e in events if e.type == EventType.CIRCLE)
    slider_heads = sum(1 for e in events if e.type == EventType.SLIDER_HEAD)
    slider_ends = sum(1 for e in events if e.type == EventType.SLIDER_END)
    last_anchors = 0
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
