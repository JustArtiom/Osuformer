from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackgroundEvent:
    filename: str
    x_offset: int = 0
    y_offset: int = 0


@dataclass
class VideoEvent:
    start_time: int
    filename: str
    x_offset: int = 0
    y_offset: int = 0


@dataclass
class BreakEvent:
    start_time: float
    end_time: float


class Events:
    def __init__(
        self,
        *,
        raw: str = "",
        background: Optional[BackgroundEvent] = None,
        video: Optional[VideoEvent] = None,
        breaks: Optional[list[BreakEvent]] = None,
    ):
        self.background = background
        self.video = video
        self.breaks: list[BreakEvent] = list(breaks) if breaks is not None else []
        self._unparsed_lines: list[str] = []

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            parts = [p.strip() for p in line.split(",")]
            event_type = parts[0]

            if event_type in ("0", "1", "Video"):
                if len(parts) < 3:
                    self._unparsed_lines.append(line)
                    continue
                filename = parts[2].strip('"')
                x_offset = int(parts[3]) if len(parts) > 3 else 0
                y_offset = int(parts[4]) if len(parts) > 4 else 0

                if event_type == "0":
                    self.background = BackgroundEvent(filename=filename, x_offset=x_offset, y_offset=y_offset)
                else:
                    start_time = int(parts[1]) if parts[1] else 0
                    self.video = VideoEvent(start_time=start_time, filename=filename, x_offset=x_offset, y_offset=y_offset)

            elif event_type == "2" or event_type == "Break":
                if len(parts) < 3:
                    self._unparsed_lines.append(line)
                    continue
                self.breaks.append(BreakEvent(start_time=float(parts[1]), end_time=float(parts[2])))

            else:
                self._unparsed_lines.append(line)

    def __str__(self) -> str:
        lines: list[str] = ["//Background and Video events"]
        if self.background:
            bg = self.background
            base = f'0,0,"{bg.filename}"'
            if bg.x_offset != 0 or bg.y_offset != 0:
                base += f",{bg.x_offset},{bg.y_offset}"
            lines.append(base)
        if self.video:
            v = self.video
            base = f'Video,{v.start_time},"{v.filename}"'
            if v.x_offset != 0 or v.y_offset != 0:
                base += f",{v.x_offset},{v.y_offset}"
            lines.append(base)
        if self.breaks:
            lines.append("//Break Periods")
            for b in self.breaks:
                start = int(b.start_time) if b.start_time == int(b.start_time) else b.start_time
                end = int(b.end_time) if b.end_time == int(b.end_time) else b.end_time
                lines.append(f"2,{start},{end}")
        if self._unparsed_lines:
            lines.append("//Storyboard Layer 0 (Background)")
            lines.extend(self._unparsed_lines)
        return "\n".join(lines)
