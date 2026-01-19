import re

class Colours:
  def __init__(self, *, raw: str = "", colours=None):
    self.colours: dict[int, tuple[int, int, int]] = colours or {}

    if raw:
      kv = self.key_value(raw)
      for k, v in kv.items():
        match = re.match(r"Combo(\d+)", k.strip())
        if match:
          index = int(match.group(1))
          rgb = tuple(int(c) for c in v.split(",")[:3])
          if len(rgb) == 3:
            self.colours[index] = rgb

  def key_value(self, raw: str) -> dict[str, str]:
    rows = raw.splitlines()
    key_value_pairs = []
    for row in rows:
      if ":" not in row:
        continue
      key, value = (segment.strip() for segment in row.split(":", 1))
      key_value_pairs.append((key, value))
    return dict(key_value_pairs)
  
  def __str__(self) -> str:
    result = []
    for index in sorted(self.colours.keys()):
      rgb = self.colours[index]
      result.append(f"Combo{index} : {rgb[0]},{rgb[1]},{rgb[2]}")
    return "\n".join(result)