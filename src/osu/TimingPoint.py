class TimingPoint:
  def __init__(
    self, *,
    raw: str = "",
    time: int = 0,
    beat_length: float = 0,
    meter: int = 4,
    sample_set: int = 1,
    sample_index: int = 0,
    volume: int = 100,
    uninherited: int = 1,
    effects: int = 0
  ):
    self.time = time
    self.beat_length = beat_length
    self.meter = meter
    self.sample_set = sample_set
    self.sample_index = sample_index
    self.volume = volume
    self.uninherited = uninherited
    self.effects = effects

    if(raw):
      self.load_raw(raw)

  def load_raw(self, raw: str):
    segments = [segment.strip() for segment in raw.split(",")]

    self.time=int(segments[0])
    self.beat_length=float(segments[1])
    self.meter=int(segments[2] if len(segments) > 2 else 4)
    self.sample_set=int(segments[3] if len(segments) > 3 else 1)
    self.sample_index=int(segments[4] if len(segments) > 4 else 0)
    self.volume=int(segments[5] if len(segments) > 5 else 100)
    self.uninherited=int(segments[6] if len(segments) > 6 else 1)
    self.effects=int(segments[7] if len(segments) > 7 else 0)

  def _format_float(self, value: float) -> str:
    formatted = format(value, ".15g")
    if formatted.endswith(".0"):
      formatted = formatted[:-2]
    if formatted == "-0":
      return "0"
    return formatted

  def __str__(self) -> str:
    beat_length = self._format_float(self.beat_length)
    return f"{self.time},{beat_length},{self.meter},{self.sample_set},{self.sample_index},{self.volume},{self.uninherited},{self.effects}"

