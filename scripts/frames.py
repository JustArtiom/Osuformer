import shutil
from pathlib import Path
from typing import Optional, Tuple

import click
import librosa
import matplotlib
import numpy as np
try:
  import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
  sf = None
from tqdm.auto import tqdm

from src.config import config_options, ExperimentConfig
from src.data.audio import audio_to_mel, ms_to_samples
from src.data.dataset import TokenWindowBuilder
from src.osu import Beatmap, TimingPoint, Circle, Slider, Spinner
from src.tokenizer import Tokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIT_ASSET_NAMES = ("hit.ogg", "hit.mp3", "hit.wav")
HIT_GAIN = 0.6


@click.command()
@click.argument("path")
@config_options
def main(path: str, config: ExperimentConfig):
  osu_path = Path(path)
  if not osu_path.exists():
    raise FileNotFoundError(f"Map file not found: {osu_path}")

  beatmap = Beatmap(file_path=str(osu_path))
  tokenizer = Tokenizer(config.tokenizer)

  tokens, times = tokenizer.encode(beatmap)
  tokens_np = np.asarray(tokens, dtype=np.int64)
  times_np = np.asarray(times, dtype=np.int64)

  audio_path = osu_path.parent / beatmap.general.audio_filename
  if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

  mel, _ = audio_to_mel(
    path=audio_path,
    sample_rate=config.audio.sample_rate,
    hop_ms=config.audio.hop_ms,
    win_ms=config.audio.win_ms,
    n_mels=config.audio.n_mels,
    n_fft=config.audio.n_fft,
  )
  mel = mel.T  # (time, n_mels)

  audio, sr = librosa.load(str(audio_path), sr=config.audio.sample_rate, mono=True)
  hit_sound, hit_note = _load_hit_sound(Path(__file__).resolve().parent / "assets", int(sr))

  segment_frames = config.dataset.window_ms // config.audio.hop_ms
  hop_frames = int(segment_frames * (1 - config.dataset.overlap))

  window_starts = _build_window_starts(mel.shape[0], segment_frames, hop_frames)
  if not window_starts:
    window_starts = [0]

  out_root = Path(__file__).resolve().parent / "frames_test"
  if out_root.exists():
    shutil.rmtree(out_root)
  out_root.mkdir(parents=True, exist_ok=True)

  builder = TokenWindowBuilder(
    tokenizer=tokenizer,
    max_tokens=1024,
    overlap_ratio=config.dataset.overlap,
  )

  hop_samples = ms_to_samples(sr, config.audio.hop_ms)
  hop_ms_actual = hop_samples * 1000.0 / sr
  segment_samples = segment_frames * hop_samples

  for frame_idx, start in enumerate(tqdm(window_starts, desc="frames", dynamic_ncols=True)):
    frame_dir = out_root / f"{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    segment = mel[start : start + segment_frames]
    if segment.shape[0] < segment_frames:
      segment = _pad_segment(segment, segment_frames - segment.shape[0])

    window_start_ms = int(round(start * hop_ms_actual))
    window_end_ms = int(round(window_start_ms + (segment_frames * hop_ms_actual)))

    audio_start = start * hop_samples
    audio_end = audio_start + segment_samples
    audio_segment = audio[audio_start:audio_end]
    if audio_segment.shape[0] < segment_samples:
      audio_segment = np.pad(audio_segment, (0, segment_samples - audio_segment.shape[0]), mode="constant")

    audio_ext, audio_log = _write_audio_segment(frame_dir, audio_segment, int(sr), name="audio")

    window_tokens, loss_mask = builder.build(
      tokens=tokens_np,
      times=times_np,
      audio_start_ms=window_start_ms,
      audio_end_ms=window_end_ms,
    )

    decoded = tokenizer.decode(window_tokens.tolist())

    raw_timing_points = _filter_timing_points(beatmap.timing_points, window_start_ms, window_end_ms)
    raw_hit_objects = _filter_hit_objects(beatmap.hit_objects, window_start_ms, window_end_ms)
    hit_events = _collect_hit_events(raw_hit_objects, window_start_ms, window_end_ms)

    _save_spectrogram(
      frame_dir / "spectrogram.png",
      segment,
      hit_events=hit_events,
      window_start_ms=window_start_ms,
      window_end_ms=window_end_ms,
    )

    mixed_ext = None
    mixed_log = None
    if hit_sound is not None:
      mixed_audio = _mix_hits(audio_segment, hit_sound, hit_events, window_start_ms, int(sr))
      mixed_ext, mixed_log = _write_audio_segment(frame_dir, mixed_audio, int(sr), name="audio-mixed")

    log_path = frame_dir / "frame.log"
    with open(log_path, "w", encoding="utf-8") as f:
      f.write(f"Map: {osu_path}\n")
      f.write(f"Audio: {audio_path}\n")
      f.write(f"Frame: {frame_idx}\n")
      f.write(f"Window ms: {window_start_ms} -> {window_end_ms}\n")
      f.write(f"Audio segment: audio.{audio_ext}\n")
      if audio_log:
        f.write(f"Audio note: {audio_log}\n")
      if hit_note:
        f.write(f"Hit sound: {hit_note}\n")
      if mixed_ext:
        f.write(f"Audio mixed: audio-mixed.{mixed_ext}\n")
        if mixed_log:
          f.write(f"Audio mixed note: {mixed_log}\n")
      f.write("\n")

      f.write("Window tokens (ids):\n")
      f.write(" ".join(str(t) for t in window_tokens.tolist()) + "\n\n")

      f.write("Window tokens (readable):\n")
      f.write(" ".join(tokenizer.id_to_token[t] for t in window_tokens.tolist()) + "\n\n")

      f.write("Loss mask indices:\n")
      f.write(" ".join(str(i) for i, v in enumerate(loss_mask.tolist()) if v) + "\n\n")

      f.write("Decoded timing points:\n")
      for tp in decoded.timing_points:
        f.write(str(tp) + "\n")
      f.write("\n")

      f.write("Decoded hit objects:\n")
      for ho in decoded.hit_objects:
        f.write(str(ho) + "\n")
      f.write("\n")

      f.write("Raw timing points in window:\n")
      for tp in raw_timing_points:
        f.write(str(tp) + "\n")
      f.write("\n")

      f.write("Raw hit objects in window:\n")
      for ho in raw_hit_objects:
        f.write(str(ho) + "\n")
      f.write("\n")

      f.write("Hit events in window:\n")
      for kind, t in hit_events:
        f.write(f"{kind}@{int(t)}ms\n")
      f.write("\n")




def _build_window_starts(total_frames: int, segment_frames: int, hop_frames: int) -> list[int]:
  if total_frames <= 0:
    return []
  starts = list(range(0, max(total_frames - segment_frames + 1, 0), max(hop_frames, 1)))
  final_start = max(0, total_frames - segment_frames)
  if not starts or starts[-1] != final_start:
    starts.append(final_start)
  return starts


def _pad_segment(segment: np.ndarray, pad_len: int) -> np.ndarray:
  if pad_len <= 0:
    return segment
  return np.pad(
    segment,
    ((0, pad_len), (0, 0)),
    mode="constant",
    constant_values=0.0,
  )


def _save_spectrogram(
  path: Path,
  segment: np.ndarray,
  *,
  hit_events: list[Tuple[str, float]] | None = None,
  window_start_ms: int | None = None,
  window_end_ms: int | None = None,
) -> None:
  plt.figure(figsize=(6, 3))
  plt.imshow(segment.T, origin="lower", aspect="auto", interpolation="nearest")
  plt.colorbar(format="%+2.0f dB")
  plt.title("Mel spectrogram")
  if hit_events and window_start_ms is not None and window_end_ms is not None:
    duration_ms = max(window_end_ms - window_start_ms, 1)
    for kind, t in hit_events:
      rel_ms = t - window_start_ms
      if rel_ms < 0 or rel_ms > duration_ms:
        continue
      x = (rel_ms / duration_ms) * max(segment.shape[0] - 1, 1)
      plt.axvline(x=x, color="white", alpha=0.6, linewidth=0.8)
      plt.text(
        x,
        -1,
        kind,
        rotation=90,
        va="top",
        ha="center",
        fontsize=6,
        color="white",
        clip_on=False,
      )
  plt.tight_layout()
  plt.savefig(path, dpi=150)
  plt.close()


def _write_audio_segment(frame_dir: Path, audio_segment: np.ndarray, sr: int, *, name: str) -> tuple[str, Optional[str]]:
  if sf is not None:
    mp3_path = frame_dir / f"{name}.mp3"
    try:
      sf.write(mp3_path, audio_segment, sr, format="MP3")
      return "mp3", None
    except Exception as exc:
      wav_path = frame_dir / f"{name}.wav"
      sf.write(wav_path, audio_segment, sr)
      return "wav", f"mp3 write failed ({exc}); wrote wav instead"

  wav_path = frame_dir / f"{name}.wav"
  _write_wav_basic(wav_path, audio_segment, sr)
  return "wav", "soundfile not available; wrote wav instead"


def _write_wav_basic(path: Path, audio_segment: np.ndarray, sr: int) -> None:
  import wave

  audio = np.clip(audio_segment, -1.0, 1.0)
  pcm16 = (audio * 32767.0).astype(np.int16)
  with wave.open(str(path), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm16.tobytes())


def _filter_timing_points(points: list[TimingPoint], start_ms: int, end_ms: int) -> list[TimingPoint]:
  return [tp for tp in points if start_ms <= tp.time < end_ms]


def _object_end_time(obj) -> float:
  if isinstance(obj, Spinner):
    return float(obj.object_params.end_time)
  if isinstance(obj, Slider):
    return float(obj.time + obj.object_params.duration)
  return float(obj.time)


def _filter_hit_objects(objects: list[Circle | Slider | Spinner], start_ms: int, end_ms: int):
  filtered = []
  for obj in objects:
    end_time = _object_end_time(obj)
    if obj.time < end_ms and end_time >= start_ms:
      filtered.append(obj)
  return filtered


def _collect_hit_events(
  objects: list[Circle | Slider | Spinner],
  start_ms: int,
  end_ms: int,
) -> list[Tuple[str, float]]:
  events: list[Tuple[str, float]] = []
  for obj in objects:
    if isinstance(obj, Circle):
      if start_ms <= obj.time < end_ms:
        events.append(("circle", float(obj.time)))
      continue
    if isinstance(obj, Slider):
      start_time = float(obj.time)
      end_time = float(obj.time + obj.object_params.duration)
      if start_ms <= start_time < end_ms:
        events.append(("slider_start", start_time))
      if start_ms <= end_time < end_ms:
        events.append(("slider_end", end_time))
      continue
    if isinstance(obj, Spinner):
      start_time = float(obj.time)
      end_time = float(obj.object_params.end_time)
      if start_ms <= start_time < end_ms:
        events.append(("spinner_start", start_time))
      if start_ms <= end_time < end_ms:
        events.append(("spinner_end", end_time))
  events.sort(key=lambda x: x[1])
  return events


def _mix_hits(
  audio_segment: np.ndarray,
  hit_sound: np.ndarray,
  events: list[Tuple[str, float]],
  window_start_ms: int,
  sr: int,
) -> np.ndarray:
  if len(events) == 0:
    return audio_segment
  mixed = audio_segment.astype(np.float32, copy=True)
  hit = (hit_sound * HIT_GAIN).astype(np.float32, copy=False)
  for _, t in events:
    offset = int(round((t - window_start_ms) / 1000.0 * sr))
    if offset < 0 or offset >= mixed.shape[0]:
      continue
    end = min(mixed.shape[0], offset + hit.shape[0])
    mixed[offset:end] += hit[: end - offset]
  return np.clip(mixed, -1.0, 1.0)


def _load_hit_sound(asset_dir: Path, sr: int) -> tuple[Optional[np.ndarray], Optional[str]]:
  for name in HIT_ASSET_NAMES:
    candidate = asset_dir / name
    if candidate.exists():
      audio, _ = librosa.load(str(candidate), sr=sr, mono=True)
      return audio.astype(np.float32, copy=False), str(candidate)
  return None, "no hit asset found"


if __name__ == "__main__":
  main()
