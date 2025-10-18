class HitSample:
  def __init__(self, raw: str = "", normal_set: int = 0, addition_set: int = 0, index: int = 0, volume: int = 0, custom: str = ""):
    self.normal_set = normal_set
    self.addition_set = addition_set
    self.index = index
    self.volume = volume
    self.custom = custom

    if raw:
      self._load_raw(raw)

  def _load_raw(self, raw: str):
    sample = [segment.strip() for segment in raw.split(":")]
    self.normal_set = int(sample[0])
    self.addition_set = int(sample[1])
    self.index = int(sample[2])
    self.volume = int(sample[3])
    self.custom = sample[4] if len(sample) > 4 else ""

  def __str__(self) -> str:
    return f"{self.normal_set}:{self.addition_set}:{self.index}:{self.volume}:{self.custom}"