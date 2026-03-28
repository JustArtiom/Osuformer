from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import Beatmap, HitObject, TimingPoint


class MapStyle(Enum):
  DEDICATED = "DEDICATED"
  STREAM = "STREAM"
  DEATHSTREAM = "DEATHSTREAM"
  BURST = "BURST"
  SHORT_JUMPS = "SHORT_JUMPS"
  MID_JUMPS = "MID_JUMPS"
  LONG_JUMPS = "LONG_JUMPS"
  DOUBLES = "DOUBLES"
  TRIPLES = "TRIPLES"
  QUADS = "QUADS"
  FINGER_CONTROL = "FINGER_CONTROL"
  STREAM_INDEX = "STREAM_INDEX"
  JUMP_INDEX = "JUMP_INDEX"
  STREAMS_100 = "STREAMS_100"


@dataclass
class MapFeatures:
  avg_spacing: float
  max_spacing: float
  spacing_std: float
  bpm: float
  star_rating: float
  song: "SongMetrics"


@dataclass
class SongMetrics:
  playable_length: float
  bpm: float
  doubles: float
  triples: float
  bursts: float
  streams: float
  deathstreams: float
  short_jumps: float
  mid_jumps: float
  long_jumps: float
  quads: float
  fcdbi: float
  si: float
  ji: float
  longest_stream: int
  streams100: int
  avg_jump_distance: float
  avg_jump_speed: float
  objects_per_second: float
  timing_alignment_error: float


@dataclass
class MapsetFeatures:
  difficulty_count: int
  star_rating_variance: float
  mean_star_rating: float
  timing_alignment_mean: float
  ops_variance: float


class Thresholds:
  # Stream/jump distances (edge-to-edge) taken from SimpleBeatmapAnalyzer defaults.
  STREAM_DISTANCE: float = 16.0
  JUMP_DISTANCE: float = 110.0
  STREAM_DISTANCE_1_3: float = 16.0
  JUMP_DISTANCE_1_3: float = 110.0

  # Classification cutoffs (percentages of objects unless noted).
  STREAM_PCT: float = 1.0
  DEATHSTREAM_PCT: float = 1.0
  BURST_PCT: float = 2.0
  SHORT_JUMP_PCT: float = 1.5
  MID_JUMP_PCT: float = 1.5
  LONG_JUMP_PCT: float = 1.0
  DOUBLES_PCT: float = 0.8
  TRIPLES_PCT: float = 0.8
  QUADS_PCT: float = 0.4
  STREAM_INDEX: float = 0.02
  JUMP_INDEX: float = 0.02
  FCDBI: float = 0.02
  STREAMS100: int = 1

  DEDICATED_TIMING_ERR_MS: float = 12.0  # Mean absolute timing deviation to beat.
  DEDICATED_OPS_VAR: float = 0.4  # Low variance of objects/sec across diffs.
  DEDICATED_STAR_VAR: float = 0.3


class FeatureExtractor:
  @staticmethod
  def extract(beatmap: Beatmap, star_rating: float) -> MapFeatures:
    hit_objects: List[HitObject] = getattr(beatmap, "hit_objects", [])
    positions = np.array([(ho.x, ho.y, ho.time) for ho in hit_objects], dtype=np.float64)
    positions = positions[positions[:, 2].argsort()] if positions.size else positions
    spacing = (
      np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
      if positions.shape[0] > 1
      else np.array([], dtype=np.float64)
    )

    avg_spacing = float(np.mean(spacing)) if spacing.size else 0.0
    max_spacing = float(np.max(spacing)) if spacing.size else 0.0
    spacing_std = float(np.std(spacing)) if spacing.size else 0.0

    bpm = _primary_bpm(getattr(beatmap, "timing_points", []))
    song_metrics = _compute_song_metrics(
      beatmap,
      stream_distance=Thresholds.STREAM_DISTANCE,
      jump_distance=Thresholds.JUMP_DISTANCE,
    )

    return MapFeatures(
      avg_spacing=avg_spacing,
      max_spacing=max_spacing,
      spacing_std=spacing_std,
      bpm=bpm,
      star_rating=star_rating,
      song=song_metrics,
    )


class StyleClassifier:
  @staticmethod
  def classify_map(features: MapFeatures, mapset: MapsetFeatures) -> Tuple[List[MapStyle], bool]:
    styles: List[MapStyle] = []
    t = Thresholds

    song = features.song
    if song.streams > t.STREAM_PCT:
      styles.append(MapStyle.STREAM)
    if song.deathstreams > t.DEATHSTREAM_PCT or song.longest_stream >= 33:
      styles.append(MapStyle.DEATHSTREAM)
    if song.bursts > t.BURST_PCT:
      styles.append(MapStyle.BURST)
    if song.short_jumps > t.SHORT_JUMP_PCT:
      styles.append(MapStyle.SHORT_JUMPS)
    if song.mid_jumps > t.MID_JUMP_PCT:
      styles.append(MapStyle.MID_JUMPS)
    if song.long_jumps > t.LONG_JUMP_PCT:
      styles.append(MapStyle.LONG_JUMPS)
    if song.doubles > t.DOUBLES_PCT:
      styles.append(MapStyle.DOUBLES)
    if song.triples > t.TRIPLES_PCT:
      styles.append(MapStyle.TRIPLES)
    if song.quads > t.QUADS_PCT:
      styles.append(MapStyle.QUADS)
    if song.fcdbi > t.FCDBI:
      styles.append(MapStyle.FINGER_CONTROL)
    if song.si > t.STREAM_INDEX:
      styles.append(MapStyle.STREAM_INDEX)
    if song.ji > t.JUMP_INDEX:
      styles.append(MapStyle.JUMP_INDEX)
    if song.streams100 >= t.STREAMS100:\
      styles.append(MapStyle.STREAMS_100)

    is_dedicated = _is_dedicated(mapset)
    if is_dedicated:
      styles.append(MapStyle.DEDICATED)

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[MapStyle] = []
    for style in styles:
      if style not in seen:
        ordered.append(style)
        seen.add(style)

    return ordered, is_dedicated


def classify_beatmap(
  beatmap: Beatmap,
  *,
  parent_path: str | Path | None = None,
  mapset_features: MapsetFeatures | None = None,
):
  sr = float(beatmap.get_difficulty().star_rating)
  features = FeatureExtractor.extract(beatmap, sr)
  if parent_path:
    mapset_features = _mapset_features_from_parent(Path(parent_path))
  if mapset_features is None:
    mapset_features = MapsetFeatures(
      difficulty_count=1,
      star_rating_variance=0.0,
      mean_star_rating=sr,
      timing_alignment_mean=features.song.timing_alignment_error,
      ops_variance=0.0,
    )

  styles, is_dedicated = StyleClassifier.classify_map(features, mapset_features)
  # raw_scores = _raw_scores(features, mapset_features, is_dedicated)
  return styles


# def _raw_scores(
#   features: MapFeatures,
#   mapset: MapsetFeatures,
#   is_dedicated: bool,
# ) -> Dict[str, float | int | bool]:
#   song = features.song
#   return {
#     "star_rating": features.star_rating,
#     "bpm": features.bpm,
#     "objects_per_second": song.objects_per_second,
#     "timing_alignment_error": song.timing_alignment_error,
#     "streams_pct": song.streams,
#     "deathstreams_pct": song.deathstreams,
#     "bursts_pct": song.bursts,
#     "short_jumps_pct": song.short_jumps,
#     "mid_jumps_pct": song.mid_jumps,
#     "long_jumps_pct": song.long_jumps,
#     "doubles_pct": song.doubles,
#     "triples_pct": song.triples,
#     "quads_pct": song.quads,
#     "fcdbi": song.fcdbi,
#     "stream_index": song.si,
#     "jump_index": song.ji,
#     "streams100": song.streams100,
#     "mapset_difficulty_count": mapset.difficulty_count,
#     "mapset_star_variance": mapset.star_rating_variance,
#     "mapset_ops_variance": mapset.ops_variance,
#     "mapset_timing_alignment_mean": mapset.timing_alignment_mean,
#     "dedicated": is_dedicated,
#   }


def _is_dedicated(mapset: MapsetFeatures) -> bool:
  t = Thresholds
  if mapset.difficulty_count == 1:
    return True
  difficulty_condition = mapset.star_rating_variance < t.DEDICATED_STAR_VAR
  timing_condition = mapset.timing_alignment_mean < t.DEDICATED_TIMING_ERR_MS
  ops_condition = mapset.ops_variance < t.DEDICATED_OPS_VAR
  return difficulty_condition and timing_condition and ops_condition


def _mapset_features_from_parent(parent_path: Path) -> MapsetFeatures:
  if not parent_path.exists():
    raise FileNotFoundError(f"Mapset folder not found: {parent_path}")

  osu_paths = sorted(parent_path.glob("*.osu"))
  if not osu_paths:
    raise FileNotFoundError(f"No .osu files found in {parent_path}")

  features: List[MapFeatures] = []
  for osu_path in osu_paths:
    try:
      if Beatmap.get_mode(str(osu_path)) != 0:
        continue
      beatmap = Beatmap(file_path=str(osu_path), with_styles=False)
      sr = float(beatmap.get_difficulty().star_rating)
      features.append(FeatureExtractor.extract(beatmap, sr))
    except Exception:
      continue

  if not features:
    raise ValueError(f"No valid osu!standard files in {parent_path}")

  return MapsetFeatures(
    difficulty_count=len(features),
    star_rating_variance=float(np.var([f.star_rating for f in features])),
    mean_star_rating=float(np.mean([f.star_rating for f in features])),
    timing_alignment_mean=float(np.mean([f.song.timing_alignment_error for f in features])),
    ops_variance=float(np.var([f.song.objects_per_second for f in features])),
  )


def _compute_song_metrics(
  beatmap: Beatmap, stream_distance: float, jump_distance: float
) -> SongMetrics:
  hit_objects: List[HitObject] = getattr(beatmap, "hit_objects", [])
  timing_points = [
    tp for tp in getattr(beatmap, "timing_points", []) if getattr(tp, "beat_length", 0) != 0
  ]
  base_tps = sorted([tp for tp in timing_points if tp.beat_length > 0], key=lambda tp: tp.time)
  if len(hit_objects) <= 1 or not base_tps:
    return SongMetrics(
      playable_length=0.0,
      bpm=0.0,
      doubles=0.0,
      triples=0.0,
      bursts=0.0,
      streams=0.0,
      deathstreams=0.0,
      short_jumps=0.0,
      mid_jumps=0.0,
      long_jumps=0.0,
      quads=0.0,
      fcdbi=0.0,
      si=0.0,
      ji=0.0,
      longest_stream=0,
      streams100=0,
      avg_jump_distance=0.0,
      avg_jump_speed=0.0,
      objects_per_second=0.0,
      timing_alignment_error=0.0,
    )

  objs = sorted(hit_objects, key=lambda ho: ho.time)
  r2 = (
    (54.4 - 4.48 * float(getattr(beatmap, "difficulty").circle_size if hasattr(beatmap, "difficulty") else 5.0))
    * 2.0
  )

  doubles = triples = quads = 0
  bursts = streams = deathstreams = 0
  short_jumps = mid_jumps = long_jumps = 0
  counter_one_fourth = 0
  counter_one_half = 0
  longest_stream = 0
  streams100 = 0
  jumps_counter = 0
  avg_jump_distance_sum = 0.0
  avg_jump_speed_sum = 0.0

  time_last = objs[0].time
  x_last = objs[0].x
  y_last = objs[0].y
  start_time = time_last

  bpm_idx = 0
  bpm_max = len(base_tps) - 1

  for ho in objs[1:]:
    x = ho.x
    y = ho.y
    t = ho.time
    while bpm_idx < bpm_max and base_tps[bpm_idx + 1].time <= t:
      bpm_idx += 1
    beat_len = base_tps[bpm_idx].beat_length
    diff_time = t - time_last
    quarter_div = diff_time / (0.25 * beat_len)
    third_div = diff_time / (beat_len / 3.0)
    half_div = diff_time / (0.5 * beat_len)
    d_r = ((x_last - x) ** 2 + (y_last - y) ** 2) ** 0.5 - r2

    if d_r - stream_distance <= 0.0 and 0.9 < quarter_div < 1.1:
      counter_one_fourth += 1
    elif d_r - Thresholds.STREAM_DISTANCE_1_3 <= 0.0 and 0.9 < third_div < 1.1:
      counter_one_fourth += 1
    else:
      if counter_one_fourth > 0:
        counter_one_fourth += 1
        longest_stream = max(longest_stream, counter_one_fourth)
        if counter_one_fourth == 2:
          doubles += 2
        elif counter_one_fourth == 3:
          triples += 3
          bursts += 3
        elif counter_one_fourth == 4:
          quads += 4
          bursts += 4
        elif counter_one_fourth <= 11:
          bursts += counter_one_fourth
        elif counter_one_fourth <= 32:
          streams += counter_one_fourth
        else:
          deathstreams += counter_one_fourth
          if counter_one_fourth >= 100:
            streams100 += 1
      counter_one_fourth = 0

    if d_r - jump_distance > 0.0 and 1.9 < half_div < 2.1:
      counter_one_half += 1
      jumps_counter += 1
      avg_jump_distance_sum += d_r
      avg_jump_speed_sum += (1000.0 * d_r) / max(t - time_last, 1e-6)
    else:
      if counter_one_half >= 2:
        if counter_one_half <= 10:
          short_jumps += counter_one_half + 1
        elif counter_one_half <= 31:
          mid_jumps += counter_one_half + 1
        else:
          long_jumps += counter_one_half + 1
      counter_one_half = 0

    time_last = t
    x_last = x
    y_last = y

  if counter_one_fourth > 0:
    counter_one_fourth += 1
    longest_stream = max(longest_stream, counter_one_fourth)
    if counter_one_fourth == 2:
      doubles += 2
    elif counter_one_fourth == 3:
      triples += 3
      bursts += 3
    elif counter_one_fourth == 4:
      quads += 4
      bursts += 4
    elif counter_one_fourth <= 11:
      bursts += counter_one_fourth
    elif counter_one_fourth <= 32:
      streams += counter_one_fourth
    else:
      deathstreams += counter_one_fourth
      if counter_one_fourth >= 100:
        streams100 += 1

  if counter_one_half >= 2:
    if counter_one_half <= 10:
      short_jumps += counter_one_half + 1
    elif counter_one_half <= 31:
      mid_jumps += counter_one_half + 1
    else:
      long_jumps += counter_one_half + 1

  length = float(len(objs))
  avg_jump_distance = avg_jump_distance_sum / max(jumps_counter, 1)
  avg_jump_speed = avg_jump_speed_sum / max(jumps_counter, 1)

  jump_value = short_jumps + mid_jumps * 1.5 + long_jumps * 2.0
  stream_value = bursts + streams * 1.5 + deathstreams * 2.0
  fcdbi = (
    (bursts - triples - quads)
    + doubles * 1.75
    + triples * 1.5
    + quads * 1.75
    - jump_value
    - streams * 1.35
    - deathstreams * 1.7
  )
  fcdbi /= length
  si = (stream_value - jump_value - doubles * 0.5) / length
  ji = (jump_value - stream_value - doubles * 0.5) / length

  beat_len = base_tps[bpm_idx].beat_length if base_tps else 0.0
  bpm_val = 60000.0 / beat_len if beat_len > 0 else 0.0
  playable_length = (objs[-1].time - start_time) / 1000.0
  ops = length / max(playable_length, 1e-6)
  timing_err = _timing_alignment_error(np.array([ho.time for ho in objs]), timing_points, bpm_val)

  return SongMetrics(
    playable_length=playable_length,
    bpm=bpm_val,
    doubles=doubles / length * 100.0,
    triples=triples / length * 100.0,
    bursts=bursts / length * 100.0,
    streams=streams / length * 100.0,
    deathstreams=deathstreams / length * 100.0,
    short_jumps=short_jumps / length * 100.0,
    mid_jumps=mid_jumps / length * 100.0,
    long_jumps=long_jumps / length * 100.0,
    quads=quads / length * 100.0,
    fcdbi=fcdbi,
    si=si,
    ji=ji,
    longest_stream=longest_stream,
    streams100=streams100,
    avg_jump_distance=avg_jump_distance,
    avg_jump_speed=avg_jump_speed,
    objects_per_second=ops,
    timing_alignment_error=timing_err,
  )


def _primary_bpm(timing_points: List[TimingPoint]) -> float:
  bpms = [tp.get_bpm() for tp in timing_points if tp.uninherited == 1 and tp.get_bpm() > 0]
  if not bpms:
    return 0.0
  rounded = [round(bpm, 2) for bpm in bpms]
  most_common = Counter(rounded).most_common(1)[0][0]
  return float(most_common)


def _timing_alignment_error(
  times_ms: np.ndarray, timing_points: Sequence[TimingPoint], bpm: float
) -> float:
  if len(timing_points) == 0 or bpm <= 0 or times_ms.size == 0:
    return 0.0
  beat_length = 60000.0 / bpm
  errors = np.abs((times_ms % beat_length) - beat_length * np.round(times_ms / beat_length) % beat_length)
  errors = np.minimum(errors, beat_length - errors)
  return float(np.mean(errors))
