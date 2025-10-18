import re

class Difficulty:
  def __init__(
    self, *,
    raw: str = "",
    hp_drain_rate: float = 5.0,
    circle_size: float = 5.0,
    overall_difficulty: float = 5.0,
    approach_rate: float = 5.0,
    slider_multiplier: float = 1.4,
    slider_tick_rate: float = 1.0
  ):
    self.hp_drain_rate = hp_drain_rate
    self.circle_size = circle_size
    self.overall_difficulty = overall_difficulty
    self.approach_rate = approach_rate
    self.slider_multiplier = slider_multiplier
    self.slider_tick_rate = slider_tick_rate

    if raw:
      kv = self.key_value(raw)
      for k, v in kv.items():
        attr = self._normalize_key(k)
        if hasattr(self, attr):
          current_value = getattr(self, attr)
          try:
            setattr(self, attr, type(current_value)(v))
          except (TypeError, ValueError):
            setattr(self, attr, v)

  def key_value(self, raw: str) -> dict[str, str]:
    rows = raw.splitlines()
    key_value_pairs = []
    for row in rows:
      if ":" not in row:
        continue
      key, value = (segment.strip() for segment in row.split(":", 1))
      key_value_pairs.append((key, value))
    return dict(key_value_pairs)
  
  def _normalize_key(self, key: str) -> str:
    key = key.strip().replace(" ", "_")
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return key.lower()

  
  def __str__(self) -> str:
    return (
      f"HPDrainRate: {self.hp_drain_rate}\n"
      f"CircleSize: {self.circle_size}\n"
      f"OverallDifficulty: {self.overall_difficulty}\n"
      f"ApproachRate: {self.approach_rate}\n"
      f"SliderMultiplier: {self.slider_multiplier}\n"
      f"SliderTickRate: {self.slider_tick_rate}"
    )