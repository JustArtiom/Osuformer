from .sections import General, Difficulty, Editor, Metadata, Colours, Events
from .timing_point import TimingPoint
from .hit_object import Circle, Slider, Spinner, HitObject
from typing import Callable, Optional, Union, List

class Beatmap():
  def __init__(
    self, *, 
    raw: str = "", 
    file_path: str = "",
    general: Optional[General] = None,
    difficulty: Optional[Difficulty] = None,
    timing_points: Optional[List[TimingPoint]] = None,
    colours: Optional[Colours] = None,
    hit_objects: Optional[List[Union[Circle, Slider, Spinner]]] = None
  ):
    if file_path:
      with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    self.general = general if general is not None else General()
    self.difficulty = difficulty if difficulty is not None else Difficulty()
    self.timing_points = list(timing_points) if timing_points is not None else []
    self.colours = colours if colours is not None else Colours()
    self.hit_objects = list(hit_objects) if hit_objects is not None else []

    if raw:
      for section, content in self.split_sections(raw).items():
        if section == "General":
          self.general = General(raw=content)
        if section == "Editor":
          self.editor = Editor(raw=content)
        if section == "Metadata":
          self.metadata = Metadata(raw=content)  
        if section == "Difficulty":
          self.difficulty = Difficulty(raw=content)
        if section == "Events":
          self.events = Events(raw=content)
        if section == "Colours":
          self.colours = Colours(raw=content)
        elif section == "TimingPoints":
          self.timing_points = [TimingPoint(raw=line) for line in content.splitlines()]
        elif section == "HitObjects":
          self.hit_objects: List[Union[Circle, Slider, Spinner]] = []
          for line in content.splitlines():
            hit_object_class = self.hit_object_type(line, 0)
            if hit_object_class:
              self.hit_objects.append(hit_object_class(raw=line))

      self._recalculate_slider_durations()

  def _load_raw(self, raw: str):
    self.sections = self.split_sections(raw)
    pass

  def split_sections(self, raw: str) -> dict[str, str]:
    sections = {}
    current_section = None
    section_lines = []

    for line in raw.splitlines():
      if line.startswith("[") and line.endswith("]"):
        if current_section:
          sections[current_section] = "\n".join(section_lines).strip()
        current_section = line[1:-1]
        section_lines = []
      else:
        if current_section:
          section_lines.append(line)

    if current_section:
      sections[current_section] = "\n".join(section_lines).strip()

    return sections

  def get_previous_timing_point(self, time: float, filter: Optional[Callable[[TimingPoint], bool]] = None) -> Union[TimingPoint, None]:
    previous_tp = None
    for tp in self.timing_points:
      if tp.time > time:
        break
      if filter and not filter(tp):
        continue
      previous_tp = tp

    return previous_tp

  def get_next_timing_point(self, time: int, filter: Optional[Callable[[TimingPoint], bool]] = None) -> Union[TimingPoint, None]:
    for tp in self.timing_points:
      if tp.time <= time:
        continue
      if filter and not filter(tp):
        continue
      return tp
    return None

  def get_bpm_at(self, time: int) -> float | None:
    tp = self.get_previous_timing_point(time, filter=lambda t: t.uninherited == 1)
    if not tp:
      tp = self.get_next_timing_point(time, filter=lambda t: t.uninherited == 1)
    
    if tp:
      return tp.get_bpm()
    return None
  
  def get_slider_velocity_multiplier_at(self, time: float) -> float:
    tp = self.get_previous_timing_point(time, filter=lambda t: t.uninherited == 0)
    if tp:
      return tp.get_slider_velocity_multiplier()
    return 1.0
  
  def _recalculate_slider_durations(self):
    for ho in self.hit_objects:
      if isinstance(ho, Slider):
        sv_multiplier = self.get_slider_velocity_multiplier_at(ho.time)
        inherited_tp = self.get_previous_timing_point(ho.time, filter=lambda t: t.uninherited == 1)
        if(not inherited_tp):
          inherited_tp = TimingPoint(time=0, beat_length=500, uninherited=1)
        ho.object_params._load_duration(sv_multiplier * self.difficulty.slider_multiplier, inherited_tp.beat_length)

  def hit_object_type(self, raw: str, type_id: int | None = None) -> Union[type[Circle], type[Slider], type[Spinner]] | None:
    if not type_id:
      segments = [segment.strip() for segment in raw.split(",")]
      type_id = int(segments[3])

    if type_id & 1: # Circle
      return Circle
    elif type_id & 2: # Slider
      return Slider
    elif type_id & 8: # Spinner
      return Spinner
    elif type_id & 4: # New Combo
      return None
    elif type_id & 128: # Hold (mania)
      return None
    else:
      raise ValueError(f"Unknown hit object type id: {type_id}")
    
  def get_difficulty(self):
    from .difficulty import calculate_difficulty
    return calculate_difficulty(self)

  def get_performance(self):
    from .difficulty import calculate_performance
    return calculate_performance(self.get_difficulty())

  def __str__(self) -> str:
    result = ["osu file format v14"]
    if hasattr(self, "general"):
      result.append(f"[General]\n{str(self.general)}")
    if hasattr(self, "editor"):
      result.append(f"[Editor]\n{str(self.editor)}")
    if hasattr(self, "metadata"):
      result.append(f"[Metadata]\n{str(self.metadata)}")
    if hasattr(self, "difficulty"):
      result.append(f"[Difficulty]\n{str(self.difficulty)}")
    if hasattr(self, "events"):
      result.append(f"[Events]\n{str(self.events)}")
    if hasattr(self, "timing_points"):
      result.append("[TimingPoints]\n"+"\n".join([str(tp) for tp in self.timing_points]))
    if hasattr(self, "colours"):
      result.append(f"\n[Colours]\n{str(self.colours)}")
    if hasattr(self, "hit_objects"):
      result.append("[HitObjects]\n"+"\n".join([str(ho) for ho in self.hit_objects]))
    return "\n\n".join(result) + "\n"
