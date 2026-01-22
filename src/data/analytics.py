import random
from pathlib import Path
from typing import Optional, Union, List, Mapping
from dataclasses import dataclass, field
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")

from ..osu import Beatmap, Slider, Circle, MapStyle

@dataclass
class AnalyticsData:
  total_maps: int = 0
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
  avg_sv_multiplier: List[float] = field(default_factory=list)
  min_sv_multiplier: List[float] = field(default_factory=list)
  max_sv_multiplier: List[float] = field(default_factory=list)
  avg_control_point: List[int] = field(default_factory=list)
  min_control_point: List[int] = field(default_factory=list)
  max_control_point: List[int] = field(default_factory=list)
  dedicated_map_count: int = 0
  map_styles: List[List[str]] = field(default_factory=list)

  total_audios: int = 0
  audio_durations_ms: List[float] = field(default_factory=list)
  audio_mel_means: List[float] = field(default_factory=list)
  audio_mel_stds: List[float] = field(default_factory=list)



class Analytics():
  def __init__(self, parent_path: Union[str, Path]):
    self.data = AnalyticsData()
    self.parent_path: Path = Path(parent_path)
    if not self.parent_path.exists():
      os.makedirs(self.parent_path)

  def collect_beatmap(self, beatmap: Beatmap):
    self.data.total_maps += 1
    bm_diff = beatmap.get_difficulty()

    sr = bm_diff.star_rating
    bpm = beatmap.get_bpm_at(10**6)
    cs = bm_diff.circle_size
    ar = bm_diff.approach_rate
    od = bm_diff.overall_difficulty
    hp = bm_diff.drain_rate

    duration = self.get_beatmap_duration(beatmap)
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

    self.data.avg_sv_multiplier.append(avg_sv if avg_sv is not None else float("nan"))
    self.data.min_sv_multiplier.append(min_sv if min_sv is not None else float("nan"))
    self.data.max_sv_multiplier.append(max_sv if max_sv is not None else float("nan"))

    self.data.avg_control_point.append(avg_cp if avg_cp is not None else 0)
    self.data.min_control_point.append(min_cp if min_cp is not None else 0)
    self.data.max_control_point.append(max_cp if max_cp is not None else 0)

    self.data.map_styles.append(styles)

  def collect_audio( self, *, duration_ms: float, mel: np.ndarray): 
    self.data.total_audios += 1
    self.data.audio_durations_ms.append(duration_ms if duration_ms > 0 else float("nan"))
    self.data.audio_mel_means.append(float(np.mean(mel)))
    self.data.audio_mel_stds.append(float(np.std(mel)))

  def get_beatmap_duration(self, beatmap: Beatmap) -> float:
    first_time = beatmap.hit_objects[0].time if beatmap.hit_objects else 0
    last_time = beatmap.hit_objects[-1].time if beatmap.hit_objects else 0
    return last_time - first_time

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
      data: List[float],
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

    labels = list(data.keys())
    values = [float(v) for v in data.values()]

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

    matrix = np.array(data, dtype=float)
    if matrix.size == 0:
      return

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)

    if show_values:
      for i in range(len(y_labels)):
        for j in range(len(x_labels)):
          plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

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


  def save(self):
    self.create_json("analytics.json", self.data.__dict__)