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
    if len(segments) < 8:
      raise ValueError("Invalid TimingPoint format")

    self.time=int(segments[0])
    self.beat_length=float(segments[1])
    self.meter=int(segments[2])
    self.sample_set=int(segments[3])
    self.sample_index=int(segments[4])
    self.volume=int(segments[5])
    self.uninherited=int(segments[6])
    self.effects=int(segments[7])
    

  def __str__(self) -> str:
    return f"{self.time},{self.beat_length},{self.meter},{self.sample_set},{self.sample_index},{self.volume},{self.uninherited},{self.effects}"