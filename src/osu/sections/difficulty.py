import re

from ..utils import fmt


class Difficulty:
    def __init__(
        self,
        *,
        raw: str = "",
        hp_drain_rate: float = 5.0,
        circle_size: float = 5.0,
        overall_difficulty: float = 5.0,
        approach_rate: float = 5.0,
        slider_multiplier: float = 1.4,
        slider_tick_rate: float = 1.0,
    ):
        self.hp_drain_rate = hp_drain_rate
        self.circle_size = circle_size
        self.overall_difficulty = overall_difficulty
        self.approach_rate = approach_rate
        self.slider_multiplier = slider_multiplier
        self.slider_tick_rate = slider_tick_rate
        self._has_approach_rate = False

        if raw:
            kv = self._key_value(raw)
            for k, v in kv.items():
                attr = self._normalize_key(k)
                if attr == "approach_rate":
                    self._has_approach_rate = True
                if hasattr(self, attr) and not attr.startswith("_"):
                    current_value = getattr(self, attr)
                    try:
                        setattr(self, attr, type(current_value)(v))
                    except (TypeError, ValueError):
                        setattr(self, attr, v)

    def _key_value(self, raw: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in raw.splitlines():
            if ":" not in row:
                continue
            key, value = (s.strip() for s in row.split(":", 1))
            result[key] = value
        return result

    def _normalize_key(self, key: str) -> str:
        key = key.strip().replace(" ", "_")
        key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
        return key.lower()

    def __str__(self) -> str:
        return (
            f"HPDrainRate:{fmt(self.hp_drain_rate)}\n"
            f"CircleSize:{fmt(self.circle_size)}\n"
            f"OverallDifficulty:{fmt(self.overall_difficulty)}\n"
            f"ApproachRate:{fmt(self.approach_rate)}\n"
            f"SliderMultiplier:{fmt(self.slider_multiplier)}\n"
            f"SliderTickRate:{fmt(self.slider_tick_rate)}"
        )
