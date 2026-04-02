import re
from pathlib import Path
from typing import Callable, Optional, Union

from .sections import General, Difficulty, Editor, Metadata, Colours, Events
from .timing_point import TimingPoint
from .hit_object import Circle, Slider, Spinner, HoldNote


HitObjectType = Union[Circle, Slider, Spinner, HoldNote]


class Beatmap:
    def __init__(
        self,
        *,
        raw: str = "",
        file_path: str = "",
        format_version: int = 14,
        general: Optional[General] = None,
        editor: Optional[Editor] = None,
        metadata: Optional[Metadata] = None,
        difficulty: Optional[Difficulty] = None,
        events: Optional[Events] = None,
        timing_points: Optional[list[TimingPoint]] = None,
        colours: Optional[Colours] = None,
        hit_objects: Optional[list[HitObjectType]] = None,
    ):
        self.file_path = file_path
        self.format_version = format_version

        if file_path and not raw:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()

        self.general = general if general is not None else General()
        self.editor = editor if editor is not None else Editor()
        self.metadata = metadata if metadata is not None else Metadata()
        self.difficulty = difficulty if difficulty is not None else Difficulty()
        self.events = events if events is not None else Events()
        self.timing_points: list[TimingPoint] = list(timing_points) if timing_points is not None else []
        self.colours = colours if colours is not None else Colours()
        self.hit_objects: list[HitObjectType] = list(hit_objects) if hit_objects is not None else []

        if raw:
            self._parse_raw(raw)

        self._apply_format_defaults()
        self._recalculate_slider_durations()

    def _parse_raw(self, raw: str) -> None:
        lines = raw.splitlines()

        # Parse format version from first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                m = re.match(r"osu file format v(\d+)", line)
                if m:
                    self.format_version = int(m.group(1))
                break

        for section, content in self._split_sections(raw).items():
            if section == "General":
                self.general = General(raw=content)
            elif section == "Editor":
                self.editor = Editor(raw=content)
            elif section == "Metadata":
                self.metadata = Metadata(raw=content)
            elif section == "Difficulty":
                self.difficulty = Difficulty(raw=content)
            elif section == "Events":
                self.events = Events(raw=content)
            elif section == "Colours":
                self.colours = Colours(raw=content)
            elif section == "TimingPoints":
                self.timing_points = [
                    TimingPoint(raw=line)
                    for line in content.splitlines()
                    if line.strip()
                ]
            elif section == "HitObjects":
                self.hit_objects = []
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    cls = self._resolve_hit_object_class(line)
                    if cls is not None:
                        self.hit_objects.append(cls(raw=line))

    def _apply_format_defaults(self) -> None:
        # Pre-v8 files had no ApproachRate field — AR defaults to OD
        if self.format_version < 8 and not self.difficulty._has_approach_rate:
            self.difficulty.approach_rate = self.difficulty.overall_difficulty

    def _split_sections(self, raw: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current_section: Optional[str] = None
        section_lines: list[str] = []

        for line in raw.splitlines():
            if line.startswith("[") and line.endswith("]"):
                if current_section is not None:
                    sections[current_section] = "\n".join(section_lines).strip()
                current_section = line[1:-1]
                section_lines = []
            elif current_section is not None:
                section_lines.append(line)

        if current_section is not None:
            sections[current_section] = "\n".join(section_lines).strip()

        return sections

    def _resolve_hit_object_class(self, raw: str) -> Optional[type[HitObjectType]]:
        segments = [s.strip() for s in raw.split(",")]
        type_id = int(segments[3])
        if type_id & 1:
            return Circle
        if type_id & 2:
            return Slider
        if type_id & 8:
            return Spinner
        if type_id & 128:
            return HoldNote
        raise ValueError(f"Unknown hit object type flags: {type_id}")

    def get_timing_point_at(
        self,
        time: float,
        predicate: Optional[Callable[[TimingPoint], bool]] = None,
    ) -> Optional[TimingPoint]:
        result: Optional[TimingPoint] = None
        for tp in self.timing_points:
            if tp.time > time:
                break
            if predicate is not None and not predicate(tp):
                continue
            result = tp
        return result

    def get_next_timing_point(
        self,
        time: float,
        predicate: Optional[Callable[[TimingPoint], bool]] = None,
    ) -> Optional[TimingPoint]:
        for tp in self.timing_points:
            if tp.time <= time:
                continue
            if predicate is not None and not predicate(tp):
                continue
            return tp
        return None

    # Keep the old name as an alias for backward compat
    def get_previous_timing_point(
        self,
        time: float,
        predicate: Optional[Callable[[TimingPoint], bool]] = None,
    ) -> Optional[TimingPoint]:
        return self.get_timing_point_at(time, predicate)

    def get_bpm_at(self, time: float = 0) -> Optional[float]:
        tp = self.get_timing_point_at(time, predicate=lambda t: t.is_uninherited)
        if tp is None:
            tp = self.get_next_timing_point(time, predicate=lambda t: t.is_uninherited)
        if tp is not None:
            return tp.get_bpm()
        return None

    def get_slider_velocity_multiplier_at(self, time: float) -> float:
        tp = self.get_timing_point_at(time, predicate=lambda t: not t.is_uninherited)
        if tp is not None:
            return tp.get_slider_velocity_multiplier()
        return 1.0

    def _recalculate_slider_durations(self) -> None:
        for ho in self.hit_objects:
            if not isinstance(ho, Slider):
                continue
            sv_multiplier = self.get_slider_velocity_multiplier_at(ho.time)
            inherited_tp = self.get_timing_point_at(ho.time, predicate=lambda t: t.is_uninherited)
            if inherited_tp is None:
                inherited_tp = TimingPoint(time=0, beat_length=500.0, uninherited=1)
            ho.object_params._load_duration(
                sv_multiplier * self.difficulty.slider_multiplier,
                inherited_tp.beat_length,
            )

    @staticmethod
    def get_mode(path: str) -> int:
        raw = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
        m = re.search(r"^Mode:\s*(\d+)", raw, re.MULTILINE)
        return int(m.group(1)) if m else 0

    def __str__(self) -> str:
        parts = [f"osu file format v{self.format_version}"]
        parts.append(f"[General]\n{self.general}")
        parts.append(f"[Editor]\n{self.editor}")
        parts.append(f"[Metadata]\n{self.metadata}")
        parts.append(f"[Difficulty]\n{self.difficulty}")
        parts.append(f"[Events]\n{self.events}")
        if self.timing_points:
            parts.append("[TimingPoints]\n" + "\n".join(str(tp) for tp in self.timing_points))
        colours_str = str(self.colours)
        if colours_str:
            parts.append(f"[Colours]\n{colours_str}")
        if self.hit_objects:
            parts.append("[HitObjects]\n" + "\n".join(str(ho) for ho in self.hit_objects))
        return "\n\n".join(parts) + "\n"
