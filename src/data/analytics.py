import random
from pathlib import Path
from typing import Optional, Sequence, Union, List, Mapping
from dataclasses import dataclass, field
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")

from ..osu import Beatmap, Slider, Circle, Spinner, MapStyle

class Analytics():
  def __init__(self, parent_path: Union[str, Path]):
    self.parent_path: Path = Path(parent_path)
    if not self.parent_path.exists():
      os.makedirs(self.parent_path)

  def create_metric_curves(
    self,
    *,
    file_name: str,
    title: str,
    metrics: dict[str, list[float | int]],
    y_label: str = "Value",
    x_label: str = "Epoch",
  ):
    import matplotlib.pyplot as plt

    if not metrics:
      return

    plt.figure(figsize=(8, 4))

    for name, values in metrics.items():
      if not values:
        continue
      arr = np.asarray(values, dtype=float)
      if arr.size == 0 or not np.any(np.isfinite(arr)):
        continue
      plt.plot(arr, label=name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()


  def create_histogram(
      self, *, 
      file_name: str, 
      title: str,
      x_label: str, 
      y_label: str = "Count",
      color: Optional[Union[str, tuple[float, float, float], List[str]]] = None,
      data: Union[dict[str, float], dict[str, dict[str, float]]]
  ):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    first_val = next(iter(data.values()), None)

    if isinstance(first_val, dict):
      categories = list(data.keys())
      sub_keys = sorted({
        sk for v in data.values()
        if isinstance(v, dict)
        for sk in v.keys()
      })
      bottom = [0.0] * len(categories)
      for sk in sub_keys:
        values: List[float] = []
        for c in categories:
          v = data[c]
          if isinstance(v, dict):
            values.append(float(v.get(sk, 0.0)))
          else:
            values.append(0.0)

        plt.bar(categories, values, bottom=bottom, label=sk)
        bottom = [b + v for b, v in zip(bottom, values)]
      plt.legend(fontsize=8, frameon=False)

    else:
      values: List[float] = []
      for v in data.values():
        if isinstance(v, dict):
          values.append(0.0)
        else:
          values.append(float(v))

      plt.bar(list(data.keys()), values, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()

  def create_numeric_histogram(
      self, *,
      file_name: str,
      title: str,
      x_label: str,
      data: Sequence[float | int],
      bins: int = 50,
      color: str = "#0c7bdc",
  ):
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
      return

    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins=bins, color=color, alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()


  def create_style_frequency_plot(
    self, *,
    file_name: str,
    title: str,
    data: dict[str, int],
  ):
    import matplotlib.pyplot as plt

    if not data:
      return

    labels, vals = zip(*sorted(data.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color="#0c7bdc")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()


  def create_style_cooccurrence_heatmap(
      self, *,
      file_name: str,
      title: str,
      styles: List[List[str]],
  ):
    import matplotlib.pyplot as plt
    import numpy as np

    all_styles = sorted({s for tags in styles for s in tags})
    if not all_styles:
      return

    idx = {s: i for i, s in enumerate(all_styles)}
    matrix = np.zeros((len(all_styles), len(all_styles)), dtype=int)

    for tags in styles:
      for i, a in enumerate(tags):
        for b in tags[i:]:
          ai, bi = idx[a], idx[b]
          matrix[ai, bi] += 1
          if ai != bi:
            matrix[bi, ai] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(len(all_styles)), all_styles, rotation=45, ha="right")
    plt.yticks(range(len(all_styles)), all_styles)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()

  def create_ratio_pie(
    self, *,
    file_name: str,
    title: str,
    data: Mapping[str, float],
  ):
    import matplotlib.pyplot as plt
    if not data:
      return

    clean = {
      k: float(v)
      for k, v in data.items()
      if v is not None and np.isfinite(v) and v > 0
    }
    if not clean:
      return
    labels = list(clean.keys())
    values = list(clean.values())

    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()

  def create_correlation_heatmap(
      self, *,
      file_name: str,
      title: str,
      data: Union[List[List[float]], np.ndarray],
      x_labels: List[str],
      y_labels: List[str],
      show_values: bool = False,
  ):
    import matplotlib.pyplot as plt

    matrix = np.asarray(data, dtype=float)
    if matrix.size == 0:
      return

    # --- normalize to [-1, 1] ---
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
      return

    vmin = finite.min()
    vmax = finite.max()

    if vmin == vmax:
      norm = np.zeros_like(matrix)
    else:
      norm = 2 * (matrix - vmin) / (vmax - vmin) - 1

    # --- plot ---
    plt.figure(figsize=(10, 8))
    plt.imshow(norm, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Normalized value [-1, 1]")

    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)

    if show_values:
      for i in range(len(y_labels)):
        for j in range(len(x_labels)):
          if np.isfinite(norm[i, j]):
            plt.text(
              j, i,
              f"{norm[i, j]:.2f}",
              ha="center",
              va="center",
              color="black",
              fontsize=8
            )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()

  def create_categorical_scatter_plot(
      self, *,
      file_name: str,
      title: str,
      y_label: str,
      data: dict[str, list[float]],
  ):
    import matplotlib.pyplot as plt

    x_labels = list(data.keys())

    all_y = [v for vals in data.values() for v in vals]
    if not all_y:
      return

    plt.figure(figsize=(14, 6))

    for i, label in enumerate(x_labels):
      vals = data.get(label, [])
      if not vals:
        continue

      x = [i] * len(vals)
      y = vals

      plt.scatter(x, y, alpha=0.6, label=label)

    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    # I have no idea what this does. but it works. 
    plt.ylim(min(all_y) - 0.2, max(all_y) + 0.2)
    plt.tight_layout()
    plt.savefig(self.parent_path / file_name, dpi=150)
    plt.close()

  def _clean_json(self, obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, list):
        return [self._clean_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: self._clean_json(v) for k, v in obj.items()}
    return obj

  def create_json(self, file_name: str, data):
    import json
    file_path = Path(self.parent_path) / file_name
    with file_path.open('w') as f:
      json.dump(self._clean_json(data), f, indent=4)

  

@dataclass
class DatasetAnalyticsData:
  total_maps: int = 0
  total_map_length: float = 0.0
  map_sr: List[float] = field(default_factory=list)
  map_bpm: List[float] = field(default_factory=list)
  map_cs: List[float] = field(default_factory=list)
  map_ar: List[float] = field(default_factory=list)
  map_od: List[float] = field(default_factory=list)
  map_hp: List[float] = field(default_factory=list)
  hit_object_count: List[int] = field(default_factory=list)
  duration_ms: List[float] = field(default_factory=list)
  obj_per_sec: List[float] = field(default_factory=list)
  circle_ratio: List[float] = field(default_factory=list)
  slider_ratio: List[float] = field(default_factory=list)
  spinner_ratio: List[float] = field(default_factory=list)
  spinner_time_distribution: List[float] = field(default_factory=list)
  avg_sv_multiplier: List[float] = field(default_factory=list)
  min_sv_multiplier: List[float] = field(default_factory=list)
  max_sv_multiplier: List[float] = field(default_factory=list)
  avg_control_point: List[int] = field(default_factory=list)
  min_control_point: List[int] = field(default_factory=list)
  max_control_point: List[int] = field(default_factory=list)
  dedicated_map_count: int = 0
  map_styles: List[List[str]] = field(default_factory=list)

  total_audios: int = 0
  total_audio_length: float = 0.0
  audio_durations_ms: List[float] = field(default_factory=list)

  # New: token-level analytics
  token_family_counts: dict = field(default_factory=dict)
  tokens_per_map: List[int] = field(default_factory=list)
  snap_values: List[int] = field(default_factory=list)
  delta_times_ms: List[float] = field(default_factory=list)
  object_song_positions: List[float] = field(default_factory=list)



class DatasetAnalytics(Analytics):
  def __init__(self, parent_path: Union[str, Path]):
    super().__init__(parent_path)
    self.data = DatasetAnalyticsData()

  def collect_beatmap(self, beatmap: Beatmap):
    self.data.total_maps += 1
    bm_diff = beatmap.get_difficulty()

    sr = bm_diff.star_rating
    bpm = beatmap.get_bpm_at()
    cs = bm_diff.circle_size
    ar = bm_diff.approach_rate
    od = bm_diff.overall_difficulty
    hp = bm_diff.drain_rate

    duration = self.get_beatmap_duration(beatmap)
    self.data.total_map_length += duration if duration > 0 else 0.0
    total_object = len(beatmap.hit_objects)

    obj_per_sec = None
    if duration > 0:
        obj_per_sec = total_object / (duration / 1000.0)

    circle_ratio = None
    if total_object > 0 and bm_diff.hit_circle_count > 0:
        circle_ratio = bm_diff.hit_circle_count / total_object

    slider_ratio = None
    if total_object > 0 and bm_diff.slider_count > 0:
        slider_ratio = bm_diff.slider_count / total_object

    spinner_ratio = None
    if total_object > 0 and bm_diff.spinner_count > 0:
        spinner_ratio = bm_diff.spinner_count / total_object

    svms: List[float] = []
    for tp in beatmap.timing_points:
        if tp.uninherited == 0 and tp.beat_length < 0:
            svms.append(tp.get_slider_velocity_multiplier() * beatmap.difficulty.slider_multiplier)

    avg_sv = min_sv = max_sv = None
    if svms:
        avg_sv = sum(svms) / len(svms)
        min_sv = min(svms)
        max_sv = max(svms)

    cps: List[int] = []
    for ho in beatmap.hit_objects:
        if isinstance(ho, Slider):
            for curve in ho.object_params.curves:
                cps.append(len(curve.curve_points))

        if isinstance(ho, Spinner):
            self.data.spinner_time_distribution.append(ho.object_params.end_time - ho.time)

    avg_cp = min_cp = max_cp = None
    if cps:
        avg_cp = int(sum(cps) / len(cps))
        min_cp = min(cps)
        max_cp = max(cps)

    if MapStyle.DEDICATED in beatmap.styles:
        self.data.dedicated_map_count += 1

    styles = [s.name for s in beatmap.styles]

    self.data.map_sr.append(sr)
    self.data.map_bpm.append(bpm if bpm is not None else float("nan"))
    self.data.map_cs.append(cs)
    self.data.map_ar.append(ar)
    self.data.map_od.append(od)
    self.data.map_hp.append(hp)

    self.data.hit_object_count.append(total_object)
    self.data.duration_ms.append(duration if duration > 0 else float("nan"))
    self.data.obj_per_sec.append(obj_per_sec if obj_per_sec is not None else float("nan"))

    self.data.circle_ratio.append(circle_ratio if circle_ratio is not None else float("nan"))
    self.data.slider_ratio.append(slider_ratio if slider_ratio is not None else float("nan"))
    self.data.spinner_ratio.append(spinner_ratio if spinner_ratio is not None else float("nan"))

    self.data.avg_sv_multiplier.append(avg_sv if avg_sv is not None else float("nan"))
    self.data.min_sv_multiplier.append(min_sv if min_sv is not None else float("nan"))
    self.data.max_sv_multiplier.append(max_sv if max_sv is not None else float("nan"))

    self.data.avg_control_point.append(avg_cp if avg_cp is not None else 0)
    self.data.min_control_point.append(min_cp if min_cp is not None else 0)
    self.data.max_control_point.append(max_cp if max_cp is not None else 0)

    self.data.map_styles.append(styles)

  def collect_tokens(self, tokens: list, times: list, snaps: list, id_to_token: list):
    """Collect token-level analytics from encoded beatmap data."""
    self.data.tokens_per_map.append(len(tokens))

    # Token family counts
    for tok_id in tokens:
      tok_str = id_to_token[tok_id]
      prefix = tok_str.split("_")[0] + "_" if "_" in tok_str else tok_str
      self.data.token_family_counts[prefix] = self.data.token_family_counts.get(prefix, 0) + 1

    # Snap values (from snaps array where tokens are OBJ_START)
    for s in snaps:
      if s > 0:
        self.data.snap_values.append(int(s))

    # Delta times between consecutive objects and object song positions
    obj_times = []
    duration = max(float(times[-1]), 1.0) if len(times) > 0 else 1.0
    for i, tok_id in enumerate(tokens):
      tok_str = id_to_token[tok_id]
      if tok_str == "OBJ_START":
        t = float(times[i])
        obj_times.append(t)
        if duration > 0:
          self.data.object_song_positions.append(t / duration)

    for i in range(1, len(obj_times)):
      dt = obj_times[i] - obj_times[i-1]
      if dt >= 0:
        self.data.delta_times_ms.append(dt)

  def collect_audio(self, *, duration_ms: float):
    self.data.total_audios += 1
    self.data.total_audio_length += duration_ms if duration_ms > 0 else 0.0
    self.data.audio_durations_ms.append(duration_ms if duration_ms > 0 else float("nan"))

  def get_beatmap_duration(self, beatmap: Beatmap) -> float:
    first_time = beatmap.hit_objects[0].time if beatmap.hit_objects else 0
    last_object = beatmap.hit_objects[-1] if beatmap.hit_objects else None
    last_time = 0
    if last_object:
        if isinstance(last_object, Circle):
            last_time = last_object.time
        elif isinstance(last_object, Slider):
            last_time = last_object.time + last_object.object_params.duration
        elif isinstance(last_object, Spinner):
            last_time = last_object.object_params.end_time
        else:
            last_time = last_object.time
    return last_time - first_time

  def save(self):
    """
    Save all analytics artifacts for the cached dataset.
    Produces:
    - JSON summary
    - Histograms
    - Ratios
    - Style statistics
    - Correlation heatmaps
    """
    import numpy as np
    from collections import Counter
    # -------------------------
    # BASIC COUNTS
    # -------------------------
    self.create_json("summary.json", {
      "total_maps": self.data.total_maps,
      "total_audios": self.data.total_audios,
      "dedicated_map_ratio": (
          self.data.dedicated_map_count / self.data.total_maps
          if self.data.total_maps > 0 else 0
      ),
      "total_map_length": {
        "hours": int(self.data.total_map_length // 3600000),
        "minutes": int((self.data.total_map_length % 3600000) // 60000),
        "seconds": int((self.data.total_map_length % 60000) // 1000),
        "milliseconds": int(self.data.total_map_length % 1000),
        "original": int(self.data.total_map_length)
      },
      "total_audio_length": {
        "hours": int(self.data.total_audio_length // 3600000),
        "minutes": int((self.data.total_audio_length % 3600000) // 60000),
        "seconds": int((self.data.total_audio_length % 60000) // 1000),
        "milliseconds": int(self.data.total_audio_length % 1000),
        "original": int(self.data.total_audio_length)
      }
    })
    # -------------------------
    # NUMERIC HISTOGRAMS
    # -------------------------
    self.create_numeric_histogram(
      file_name="sr_distribution.png",
      title="Star Rating Distribution",
      x_label="Star Rating",
      data=self.data.map_sr,
      bins=40
    )
    self.create_numeric_histogram(
      file_name="bpm_distribution.png",
      title="BPM Distribution",
      x_label="BPM",
      data=self.data.map_bpm,
      bins=40
    )
    self.create_numeric_histogram(
      file_name="map_duration_distribution.png",
      title="Map Duration (ms)",
      x_label="Duration (ms)",
      data=self.data.duration_ms,
      bins=50
    )
    self.create_numeric_histogram(
      file_name="objects_per_second.png",
      title="Objects per Second",
      x_label="Objects / sec",
      data=self.data.obj_per_sec,
      bins=50
    )
    # -------------------------
    # HIT OBJECT RATIOS
    # -------------------------
    self.create_numeric_histogram(
      file_name="circle_ratio.png",
      title="Circle Ratio",
      x_label="Circle / Total",
      data=self.data.circle_ratio,
      bins=30
    )
    self.create_numeric_histogram(
      file_name="slider_ratio.png",
      title="Slider Ratio",
      x_label="Slider / Total",
      data=self.data.slider_ratio,
      bins=30
    )
    self.create_ratio_pie(
      file_name="circle_vs_slider_pie.png",
      title="Circle vs Slider Ratio (Mean)",
      data={
          "Circles": float(np.nanmean(self.data.circle_ratio)),
          "Sliders": float(np.nanmean(self.data.slider_ratio)),
          "Spinners": float(np.nanmean(self.data.spinner_ratio)),
      }
    )
    # -------------------------
    # SLIDER VELOCITY STATS
    # -------------------------
    self.create_numeric_histogram(
      file_name="avg_sv_distribution.png",
      title="Average Slider Velocity Multiplier",
      x_label="SV Multiplier",
      data=self.data.avg_sv_multiplier,
      bins=40
    )
    self.create_numeric_histogram(
      file_name="max_sv_distribution.png",
      title="Max Slider Velocity Multiplier",
      x_label="SV Multiplier",
      data=self.data.max_sv_multiplier,
      bins=40
    )
    # -------------------------
    # SPINNER TIME DISTRIBUTION
    # -------------------------
    self.create_numeric_histogram(
      file_name="spinner_time_distribution.png",
      title="Spinner Time Distribution",
      x_label="Time (ms)",
      data=self.data.spinner_time_distribution,
      bins=40
    )
    # -------------------------
    # CONTROL POINT STATS
    # -------------------------
    self.create_numeric_histogram(
      file_name="avg_cp_distribution.png",
      title="Average Control Points per Slider",
      x_label="Control Points",
      data=self.data.avg_control_point,
      bins=20
    )

    self.create_numeric_histogram(
      file_name="max_cp_distribution.png",
      title="Max Control Points per Slider",
      x_label="Control Points",
      data=self.data.max_control_point,
      bins=20
    )

    # -------------------------
    # STYLE FREQUENCY
    # -------------------------
    style_counts = Counter(
      style
      for styles in self.data.map_styles
      for style in styles
    )

    self.create_style_frequency_plot(
      file_name="style_frequency.png",
      title="Map Style Frequency",
      data=dict(style_counts)
    )

    # -------------------------
    # STYLE CO-OCCURRENCE
    # -------------------------
    self.create_style_cooccurrence_heatmap(
      file_name="style_cooccurrence.png",
      title="Style Co-occurrence Heatmap",
      styles=self.data.map_styles
    )

    # -------------------------
    # CORRELATION HEATMAP
    # -------------------------
    corr_fields = [
      ("Star Rating", self.data.map_sr),
      ("BPM", self.data.map_bpm),
      ("Objects/sec", self.data.obj_per_sec),
      ("Circle Ratio", self.data.circle_ratio),
      ("Slider Ratio", self.data.slider_ratio),
      ("Avg SV", self.data.avg_sv_multiplier),
      ("Max SV", self.data.max_sv_multiplier),
      ("Avg CP", self.data.avg_control_point),
    ]

    labels = [name for name, _ in corr_fields]
    values = np.array([vals for _, vals in corr_fields], dtype=float)

    # pairwise correlation (nan-safe)
    corr_matrix = np.corrcoef(np.nan_to_num(values), rowvar=True)

    self.create_correlation_heatmap(
      file_name="correlation_heatmap.png",
      title="Dataset Feature Correlation",
      data=corr_matrix,
      x_labels=labels,
      y_labels=labels,
      show_values=True
    )

    # -------------------------
    # AUDIO STATS
    # -------------------------
    self.create_numeric_histogram(
      file_name="audio_duration_distribution.png",
      title="Audio Duration Distribution",
      x_label="Duration (ms)",
      data=self.data.audio_durations_ms,
      bins=40
    )

    # -------------------------
    # TOKEN TYPE DISTRIBUTION
    # -------------------------
    if self.data.token_family_counts:
      sorted_families = dict(sorted(
        self.data.token_family_counts.items(),
        key=lambda x: x[1], reverse=True
      )[:20])  # top 20 families
      self.create_histogram(
        file_name="token_family_distribution.png",
        title="Token Family Distribution (Top 20)",
        x_label="Token Family",
        y_label="Count",
        data={k: float(v) for k, v in sorted_families.items()},
        color="#0c7bdc",
      )

    # -------------------------
    # TOKENS PER MAP
    # -------------------------
    self.create_numeric_histogram(
      file_name="tokens_per_map.png",
      title="Tokens per Map",
      x_label="Token Count",
      data=self.data.tokens_per_map,
      bins=50,
    )

    # -------------------------
    # SNAP DISTRIBUTION
    # -------------------------
    if self.data.snap_values:
      snap_counter: dict[str, float] = {}
      for s in self.data.snap_values:
        key = f"1/{s}" if s > 0 else "off-grid"
        snap_counter[key] = snap_counter.get(key, 0) + 1
      self.create_histogram(
        file_name="snap_distribution.png",
        title="Beat Snap Distribution",
        x_label="Snap Subdivision",
        y_label="Count",
        data=snap_counter,
        color="#ffc107",
      )

    # -------------------------
    # DELTA TIME HISTOGRAM
    # -------------------------
    self.create_numeric_histogram(
      file_name="delta_time_distribution.png",
      title="Delta Time Between Objects",
      x_label="Delta Time (ms)",
      data=self.data.delta_times_ms,
      bins=100,
      color="#28a745",
    )

    # -------------------------
    # OBJECT DENSITY OVER SONG POSITION
    # -------------------------
    self.create_numeric_histogram(
      file_name="object_song_position.png",
      title="Object Density Over Song Position",
      x_label="Song Position (0=start, 1=end)",
      data=self.data.object_song_positions,
      bins=50,
      color="#dc3545",
    )

    # -------------------------
    # FINAL FULL JSON DUMP
    # -------------------------
    self.create_json("analytics_full.json", self.data.__dict__)


class CheckpointAnalytics(Analytics):
  def __init__(self, parent_path: Union[str, Path]):
    super().__init__(parent_path)
    self.epochs: List[dict[str, Union[float, int]]] = []

  def collect_epoch_metrics(
    self,
    *,
    epoch: int,
    val_loss: float,
    train_loss: float,
    current_lr: float,
    token_accuracies: Optional[dict] = None,
    loss_decomposition: Optional[dict] = None,
  ):
    entry: dict[str, Union[float, int]] = {
      "epoch": epoch,
      "val_loss": val_loss,
      "train_loss": train_loss,
      "current_lr": current_lr,
    }
    if token_accuracies is not None:
      entry.update(token_accuracies)
    if loss_decomposition is not None:
      entry.update(loss_decomposition)
    self.epochs.append(entry)

  def save(self):
    self.create_metric_curves(
      file_name="loss_curve.png",
      title="Training and Validation Loss Curve",
      metrics={
        "Train Loss": [e["train_loss"] for e in self.epochs],
        "Validation Loss": [e["val_loss"] for e in self.epochs],
      },
      y_label="Loss",
      x_label="Epoch"
    )

    # Learning rate curve
    self.create_metric_curves(
      file_name="lr_curve.png",
      title="Learning Rate Schedule",
      metrics={
        "Learning Rate": [e["current_lr"] for e in self.epochs],
      },
      y_label="LR",
      x_label="Epoch",
    )

    # Per-token-type accuracy curves (if available)
    accuracy_keys = [
      ("ts_accuracy", "TS Exact"),
      ("ts_fuzzy_1_accuracy", "TS Fuzzy \u00b11"),
      ("ts_fuzzy_2_accuracy", "TS Fuzzy \u00b12"),
      ("snap_accuracy", "SNAP"),
      ("position_accuracy", "Position (X+Y)"),
      ("type_accuracy", "Object Type"),
      ("overall_accuracy", "Overall"),
    ]
    acc_metrics = {}
    for key, label in accuracy_keys:
      vals = [e.get(key) for e in self.epochs if key in e]
      if vals:
        acc_metrics[label] = vals

    if acc_metrics:
      self.create_metric_curves(
        file_name="token_accuracy_curves.png",
        title="Per-Token-Type Accuracy",
        metrics=acc_metrics,
        y_label="Accuracy",
        x_label="Epoch",
      )

    # Loss decomposition curves (if available)
    loss_keys = [
      ("timing_loss", "Timing (TS+SNAP)"),
      ("position_loss", "Position (X+Y)"),
      ("structure_loss", "Structure"),
    ]
    loss_metrics = {}
    for key, label in loss_keys:
      vals = [e.get(key) for e in self.epochs if key in e]
      if vals:
        loss_metrics[label] = vals

    if loss_metrics:
      self.create_metric_curves(
        file_name="loss_decomposition.png",
        title="Loss Decomposition by Token Type",
        metrics=loss_metrics,
        y_label="Loss",
        x_label="Epoch",
      )

    self.create_json("model_analytics.json", {
      "epochs": self.epochs
    })


class GenerationAnalytics(Analytics):
  """Analytics for generated beatmaps."""

  def __init__(self, parent_path: Union[str, Path]):
    super().__init__(parent_path)

  def save_generation_report(
    self,
    *,
    beatmap: Beatmap,
    star_rating: Optional[float] = None,
    mel_rms: Optional[np.ndarray] = None,
    hop_ms: float = 10.0,
  ):
    """Create all generation analytics plots for a generated beatmap."""
    import matplotlib.pyplot as plt

    obj_times = []
    obj_xs = []
    obj_ys = []
    for ho in beatmap.hit_objects:
      obj_times.append(ho.time)
      obj_xs.append(ho.x)
      obj_ys.append(ho.y)

    if not obj_times:
      return

    # -------------------------
    # POSITION HEATMAP
    # -------------------------
    plt.figure(figsize=(6, 5))
    plt.hist2d(
      obj_xs, obj_ys,
      bins=[32, 24],
      range=[[0, 512], [0, 384]],
      cmap="hot",
    )
    plt.colorbar(label="Object Count")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Generated Object Position Heatmap")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(self.parent_path / "position_heatmap.png", dpi=150)
    plt.close()

    # -------------------------
    # OBJECT DENSITY HISTOGRAM
    # -------------------------
    total_duration_s = (max(obj_times) - min(obj_times)) / 1000.0 if len(obj_times) > 1 else 1.0
    n_bins = max(1, int(total_duration_s / 2))  # 2-second bins
    self.create_numeric_histogram(
      file_name="object_density.png",
      title="Object Density Over Time",
      x_label="Time (ms)",
      data=obj_times,
      bins=n_bins,
      color="#17a2b8",
    )

    # -------------------------
    # OBJECT TIMELINE vs AUDIO ENERGY
    # -------------------------
    if mel_rms is not None and len(mel_rms) > 0:
      fig, ax1 = plt.subplots(figsize=(12, 4))

      # Audio energy
      rms_times = np.arange(len(mel_rms)) * hop_ms
      ax1.fill_between(rms_times, mel_rms, alpha=0.3, color="blue", label="Audio RMS")
      ax1.set_xlabel("Time (ms)")
      ax1.set_ylabel("Audio Energy", color="blue")
      ax1.tick_params(axis="y", labelcolor="blue")

      # Objects as scatter
      ax2 = ax1.twinx()
      y_positions = [1] * len(obj_times)
      ax2.scatter(obj_times, y_positions, alpha=0.5, s=10, color="red", label="Objects")
      ax2.set_ylabel("Objects", color="red")
      ax2.set_yticks([])

      plt.title("Object Timeline vs Audio Energy")
      fig.tight_layout()
      plt.savefig(self.parent_path / "object_timeline.png", dpi=150)
      plt.close()

    # -------------------------
    # SUMMARY JSON
    # -------------------------
    summary = {
      "total_objects": len(beatmap.hit_objects),
      "circles": sum(1 for ho in beatmap.hit_objects if isinstance(ho, Circle)),
      "sliders": sum(1 for ho in beatmap.hit_objects if isinstance(ho, Slider)),
      "spinners": sum(1 for ho in beatmap.hit_objects if isinstance(ho, Spinner)),
      "duration_ms": max(obj_times) - min(obj_times) if obj_times else 0,
      "objects_per_second": len(obj_times) / total_duration_s if total_duration_s > 0 else 0,
    }
    if star_rating is not None:
      summary["star_rating"] = star_rating

    self.create_json("generation_summary.json", summary)