from __future__ import annotations

from dataclasses import dataclass

from .events import EventType


@dataclass(frozen=True)
class EventRange:
    type: EventType
    min_value: int
    max_value: int

    @property
    def size(self) -> int:
        return self.max_value - self.min_value + 1
