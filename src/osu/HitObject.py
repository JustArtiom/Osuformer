from __future__ import annotations
from typing import Union, List
from .HitSample import HitSample  

class HitObject:
  def __init__(
      self, *, 
      raw: str = "",
      x: int = 0,
      y: int = 0,
      time: int = 0,
      type: int = 0,
      hit_sound: int = 0,
      object_params: List[Union[SpinnerObjectParams, SliderObjectParams]] = None,
      hit_sample: HitSample = HitSample()
  ):
    self.x = x
    self.y = y
    self.time = time
    self.type = type
    self.hit_sound = hit_sound
    self.object_params = object_params
    self.hit_sample = hit_sample

    if raw:
      self.load_raw(raw)

  def load_raw(self, raw: str):
    segments = [segment.strip() for segment in raw.split(",")]
    if len(segments) != 7:
      raise ValueError(f"Invalid HitObject format, expected 6 segments but got {len(segments)}")

    self.x = int(segments[0])
    self.y = int(segments[1])
    self.time = int(segments[2])
    self.type = int(segments[3])
    self.hit_sound = int(segments[4])
    self.object_params = segments[5]
    self.hit_sample = HitSample(raw=segments[6])

  def __str__(self) -> str:
    return f"{self.x},{self.y},{self.time},{self.type},{self.hit_sound},{self.object_params},{self.hit_sample}"

class Circle(HitObject):
  def __init__(
    self, *, 
    raw: str = "",
    x: int = 0,
    y: int = 0,
    time: int = 0,
    type: int = 0,
    hit_sound: int = 0,
    hit_sample: HitSample = HitSample()
  ):
    super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=None, hit_sample=hit_sample)

  def load_raw(self, raw: str):
    segments = [segment.strip() for segment in raw.split(",")]
    self.x = int(segments[0])
    self.y = int(segments[1])
    self.time = int(segments[2])
    self.type = int(segments[3])
    self.hit_sound = int(segments[4])
    self.hit_sample = HitSample(raw=segments[5]) if len(segments) > 5 else HitSample()


  def __str__(self) -> str:
    return  f"{self.x},{self.y},{self.time},{self.type},{self.hit_sound},{self.hit_sample}"
  

class SliderObjectParams:
  def __init__(
    self, 
    *,
    raw: str = "",
    curve_type: str = "",
    curve_points: list[tuple[int, int]] = [],
    slides: int = 1,
    length: float = 0.0,
    edge_sounds: list[int] = [],
    edge_sets: list[tuple[int, int]] = []
  ):
    self.curve_type = curve_type
    self.curve_points = curve_points
    self.slides = slides
    self.length = length
    self.edge_sounds = edge_sounds
    self.edge_sets = edge_sets

    if raw:
      self.load_raw(raw)

  def load_raw(self, raw: str):
    segments = [segment.strip() for segment in raw.split(",")]
    [curve_type, *curve_points_str] = segments[0].split("|")

    self.curve_type = curve_type
    self.curve_points = [tuple(map(int, point.split(":"))) for point in curve_points_str]
    self.slides = int(segments[1])
    self.length = float(segments[2])

    if len(segments) >= 4 and segments[3]:
      self.edge_sounds = list(map(int, segments[3].split("|")))
    else:
      self.edge_sounds = [0] * (self.slides + 1)

    if len(segments) >= 5 and segments[4]:
      self.edge_sets = [tuple(map(int, s.split(":"))) for s in segments[4].split("|")]
    else:
      self.edge_sets = [(0, 0)] * (self.slides + 1)

  def __str__(self) -> str:
    curve_points_str = "|".join([f"{x}:{y}" for x, y in self.curve_points])
    edge_sounds_str = "|".join(map(str, self.edge_sounds))
    edge_sets_str = "|".join([f"{s1}:{s2}" for s1, s2 in self.edge_sets])
    return f"{self.curve_type}|{curve_points_str},{self.slides},{self.length},{edge_sounds_str},{edge_sets_str}"

class Slider(HitObject):
  def __init__(
    self, 
    *, 
    raw: str = "",
    x: int = 0,
    y: int = 0,
    time: int = 0,
    type: int = 0,
    hit_sound: int = 0,
    object_params: SliderObjectParams = SliderObjectParams(),
    hit_sample: HitSample = HitSample()
  ):
    super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=object_params, hit_sample=hit_sample)

  def load_raw(self, raw: str):
    print(raw)
    segments = [segment.strip() for segment in raw.split(",")]
    [x, y, time, type, hit_sound, *object_params_str, hit_sample] = segments

    if(":" not in hit_sample):
      object_params_str.append(hit_sample)
      hit_sample = "0:0:0:0:"

    self.x = int(x)
    self.y = int(y)
    self.time = int(time)
    self.type = int(type)
    self.hit_sound = int(hit_sound)
    self.object_params = SliderObjectParams(raw=",".join(object_params_str))
    self.hit_sample = HitSample(raw=hit_sample)

  def __str__(self) -> str:
    return f"{self.x},{self.y},{self.time},{self.type},{self.hit_sound},{self.object_params},{self.hit_sample}"
  

class SpinnerObjectParams:
  def __init__(
    self, 
    *,
    raw: str = "",
    end_time: int = 0
  ):
    self.end_time = end_time

    if raw:
      self.load_raw(raw)

  def load_raw(self, raw: str):
    self.end_time = int(raw)

  def __str__(self) -> str:
    return f"{self.end_time}"


class Spinner(HitObject):
  def __init__(
    self, 
    *, 
    raw: str = "",
    x: int = 0,
    y: int = 0,
    time: int = 0,
    type: int = 0,
    hit_sound: int = 0,
    object_params: SpinnerObjectParams = SpinnerObjectParams(),
    hit_sample: HitSample = HitSample()
  ):
    super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=object_params, hit_sample=hit_sample)

  def load_raw(self, raw: str):
    segments = [segment.strip() for segment in raw.split(",")]
    [x, y, time, type, hit_sound, *object_params_str, hit_sample] = segments
    if(":" not in hit_sample):
      object_params_str.append(hit_sample)
      hit_sample = "0:0:0:0:"
      
    self.x = int(x)
    self.y = int(y)
    self.time = int(time)
    self.type = int(type)
    self.hit_sound = int(hit_sound)
    self.object_params = SpinnerObjectParams(raw=",".join(object_params_str))
    self.hit_sample = HitSample(raw=hit_sample) if hit_sample else HitSample()