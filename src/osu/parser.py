from .general import General
from .timing_point import TimingPoint
from .hit_object import Circle, Slider, Spinner, HitObject
from typing import Union, List

class Beatmap():
  def __init__(
    self, *, 
    raw: str = "", 
    file_path: str = "",
    general: General = None,
    timing_points: List[TimingPoint] = None,
    hit_objects: List[Union[Circle, Slider, Spinner]] = None
  ):
    if file_path:
      with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if general:
      self.general = general
    if timing_points:
      self.timing_points = timing_points
    if hit_objects:
      self.hit_objects = hit_objects

    if raw:
      for section, content in self.split_sections(raw).items():
        if section == "General":
          self.general = General(raw=content)
        elif section == "TimingPoints":
          self.timing_points = [TimingPoint(raw=line) for line in content.splitlines()]
        elif section == "HitObjects":
          self.hit_objects: List[Union[Circle, Slider, Spinner]] = []
          for line in content.splitlines():
            hit_object_class = self.hit_object_type(line, 0)
            self.hit_objects.append(hit_object_class(raw=line))


  def load_raw(self, raw: str):
    self.sections = self.split_sections(raw)
    pass

  def split_sections(self, raw: str) -> dict[str, str]:
    sections = {}
    current_section = None
    section_lines = []

    for line in raw.splitlines():
      line = line.strip()
      if line.startswith("[") and line.endswith("]"):
        if current_section:
          sections[current_section] = "\n".join(section_lines).strip()
        current_section = line[1:-1].strip()
        section_lines = []
      else:
        if current_section:
          section_lines.append(line)

    if current_section:
      sections[current_section] = "\n".join(section_lines).strip()

    return sections

  def hit_object_type(self, raw: str, type_id: int) -> Union[type[Circle], type[Slider], type[Spinner], type[HitObject]]:
    if not type_id:
      segments = [segment.strip() for segment in raw.split(",")]
      type_id = int(segments[3])

    if type_id & 1:
      return Circle
    elif type_id & 2:
      return Slider
    elif type_id & 8:
      return Spinner
    else:
      raise ValueError(f"Unknown hit object type id: {type_id}")
    
  def __str__(self) -> str:
    result = ["osu file format v14\n"]
    if hasattr(self, "general"):
      result.append(f"[General]\n{str(self.general)}")
    if hasattr(self, "timing_points"):
      result.append("[TimingPoints]")
      for tp in self.timing_points:
        result.append(str(tp))
      result.append("")
    if hasattr(self, "hit_objects"):
      result.append("[HitObjects]")
      for ho in self.hit_objects:
        result.append(str(ho))
      result.append("")
    return "\n".join(result)