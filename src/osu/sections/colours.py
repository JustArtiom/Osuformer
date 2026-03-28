import re
from typing import Optional


RGBColour = tuple[int, int, int]
RGBAColour = tuple[int, int, int, int]
AnyColour = RGBColour | RGBAColour


def _parse_colour(value: str) -> AnyColour:
    components = [int(c.strip()) for c in value.split(",")]
    if len(components) == 4:
        return (components[0], components[1], components[2], components[3])
    return (components[0], components[1], components[2])


def _format_colour(colour: AnyColour) -> str:
    return ",".join(str(c) for c in colour)


class Colours:
    def __init__(
        self,
        *,
        raw: str = "",
        combo_colours: Optional[dict[int, AnyColour]] = None,
        slider_track_override: Optional[AnyColour] = None,
        slider_border: Optional[AnyColour] = None,
    ):
        self.combo_colours: dict[int, AnyColour] = dict(combo_colours) if combo_colours is not None else {}
        self.slider_track_override: Optional[AnyColour] = slider_track_override
        self.slider_border: Optional[AnyColour] = slider_border

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        for row in raw.splitlines():
            if ":" not in row:
                continue
            key, value = (s.strip() for s in row.split(":", 1))
            combo_match = re.match(r"^Combo(\d+)$", key)
            if combo_match:
                self.combo_colours[int(combo_match.group(1))] = _parse_colour(value)
            elif key == "SliderTrackOverride":
                self.slider_track_override = _parse_colour(value)
            elif key == "SliderBorder":
                self.slider_border = _parse_colour(value)

    def __str__(self) -> str:
        lines: list[str] = []
        for index in sorted(self.combo_colours.keys()):
            lines.append(f"Combo{index} : {_format_colour(self.combo_colours[index])}")
        if self.slider_track_override is not None:
            lines.append(f"SliderTrackOverride : {_format_colour(self.slider_track_override)}")
        if self.slider_border is not None:
            lines.append(f"SliderBorder : {_format_colour(self.slider_border)}")
        return "\n".join(lines)
