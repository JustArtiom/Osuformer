from __future__ import annotations

from dataclasses import dataclass

from src.config.schemas.tokenizer import TokenizerConfig

from .durations import DURATION_COUNT
from .events import Event, EventType
from .ranges import EventRange
from .special_tokens import SPECIAL_COUNT, SpecialToken


_OUTPUT_ORDER: tuple[EventType, ...] = (
    EventType.ABS_TIME,
    EventType.SNAPPING,
    EventType.DISTANCE,
    EventType.POS,
    EventType.NEW_COMBO,
    EventType.HITSOUND,
    EventType.VOLUME,
    EventType.CIRCLE,
    EventType.SPINNER,
    EventType.SPINNER_END,
    EventType.SLIDER_HEAD,
    EventType.SLIDER_END,
    EventType.BEZIER_ANCHOR,
    EventType.PERFECT_ANCHOR,
    EventType.CATMULL_ANCHOR,
    EventType.LINEAR_ANCHOR,
    EventType.RED_ANCHOR,
    EventType.SLIDER_SLIDES,
    EventType.SLIDER_SUSTAIN,
    EventType.SLIDER_REPEAT_SUSTAIN,
    EventType.DURATION,
    EventType.BEAT,
    EventType.MEASURE,
    EventType.TIMING_POINT,
    EventType.KIAI,
    EventType.SCROLL_SPEED,
    EventType.SAMPLESET,
)

_INPUT_ORDER: tuple[EventType, ...] = (
    EventType.REL_TIME,
)


@dataclass(frozen=True)
class GridLayout:
    x_count: int
    y_count: int
    x_min_px: int
    y_min_px: int
    step_px: int

    @property
    def size(self) -> int:
        return self.x_count * self.y_count

    def encode(self, x_px: float, y_px: float) -> int:
        xi = int(round((x_px - self.x_min_px) / self.step_px))
        yi = int(round((y_px - self.y_min_px) / self.step_px))
        xi = max(0, min(self.x_count - 1, xi))
        yi = max(0, min(self.y_count - 1, yi))
        return yi * self.x_count + xi

    def decode(self, pos_value: int) -> tuple[float, float]:
        yi, xi = divmod(pos_value, self.x_count)
        x_px = self.x_min_px + xi * self.step_px
        y_px = self.y_min_px + yi * self.step_px
        return float(x_px), float(y_px)


class Vocab:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.grid = _build_grid(config)
        self._ranges: dict[EventType, EventRange] = _build_ranges(config, self.grid)
        self._output_ranges: list[EventRange] = [self._ranges[t] for t in _OUTPUT_ORDER]
        self._input_ranges: list[EventRange] = [self._ranges[t] for t in _INPUT_ORDER]
        self._start: dict[EventType, int] = {}
        offset = SPECIAL_COUNT
        for er in self._output_ranges:
            self._start[er.type] = offset
            offset += er.size
        self.vocab_size_out = offset
        for er in self._input_ranges:
            self._start[er.type] = offset
            offset += er.size
        self.vocab_size_in = offset

    @property
    def output_ranges(self) -> list[EventRange]:
        return list(self._output_ranges)

    @property
    def input_ranges(self) -> list[EventRange]:
        return list(self._input_ranges)

    def range_for(self, event_type: EventType) -> EventRange:
        return self._ranges[event_type]

    def token_range(self, event_type: EventType) -> tuple[int, int]:
        er = self._ranges[event_type]
        start = self._start[event_type]
        return start, start + er.size

    def encode_event(self, event: Event) -> int:
        if event.type not in self._ranges:
            raise ValueError(f"unknown event type: {event.type}")
        er = self._ranges[event.type]
        if not (er.min_value <= event.value <= er.max_value):
            raise ValueError(
                f"value {event.value} out of range [{er.min_value}, {er.max_value}] for {event.type}"
            )
        return self._start[event.type] + event.value - er.min_value

    def decode_token(self, token_id: int) -> Event | SpecialToken:
        if 0 <= token_id < SPECIAL_COUNT:
            return SpecialToken(token_id)
        for er in self._output_ranges + self._input_ranges:
            start = self._start[er.type]
            if start <= token_id < start + er.size:
                return Event(type=er.type, value=er.min_value + token_id - start)
        raise ValueError(f"token id {token_id} maps to no event")


def _build_grid(config: TokenizerConfig) -> GridLayout:
    step = config.coordinate_step
    pad = config.coordinate_padding
    x_min = -pad
    x_max = 512 + pad
    y_min = -pad
    y_max = 384 + pad
    x_count = (x_max - x_min) // step + 1
    y_count = (y_max - y_min) // step + 1
    return GridLayout(
        x_count=x_count,
        y_count=y_count,
        x_min_px=x_min,
        y_min_px=y_min,
        step_px=step,
    )


def _build_ranges(config: TokenizerConfig, grid: GridLayout) -> dict[EventType, EventRange]:
    window_ms = config.context_ms + config.generate_ms + config.lookahead_ms
    max_time_bin = window_ms // config.dt_bin_ms

    return {
        EventType.ABS_TIME: EventRange(EventType.ABS_TIME, 0, max_time_bin),
        EventType.REL_TIME: EventRange(EventType.REL_TIME, 0, max_time_bin),
        EventType.SNAPPING: EventRange(EventType.SNAPPING, 0, config.snap_max),
        EventType.DISTANCE: EventRange(EventType.DISTANCE, 0, config.distance_max_px),
        EventType.POS: EventRange(EventType.POS, 0, grid.size - 1),
        EventType.NEW_COMBO: EventRange(EventType.NEW_COMBO, 0, 0),
        EventType.HITSOUND: EventRange(EventType.HITSOUND, 0, config.hitsound_count - 1),
        EventType.VOLUME: EventRange(EventType.VOLUME, 0, config.volume_max),
        EventType.CIRCLE: EventRange(EventType.CIRCLE, 0, 0),
        EventType.SPINNER: EventRange(EventType.SPINNER, 0, 0),
        EventType.SPINNER_END: EventRange(EventType.SPINNER_END, 0, 0),
        EventType.SLIDER_HEAD: EventRange(EventType.SLIDER_HEAD, 0, 0),
        EventType.SLIDER_END: EventRange(EventType.SLIDER_END, 0, 0),
        EventType.BEZIER_ANCHOR: EventRange(EventType.BEZIER_ANCHOR, 0, 0),
        EventType.PERFECT_ANCHOR: EventRange(EventType.PERFECT_ANCHOR, 0, 0),
        EventType.CATMULL_ANCHOR: EventRange(EventType.CATMULL_ANCHOR, 0, 0),
        EventType.LINEAR_ANCHOR: EventRange(EventType.LINEAR_ANCHOR, 0, 0),
        EventType.RED_ANCHOR: EventRange(EventType.RED_ANCHOR, 0, 0),
        EventType.SLIDER_SLIDES: EventRange(EventType.SLIDER_SLIDES, 1, config.slider_slides_max),
        EventType.SLIDER_SUSTAIN: EventRange(EventType.SLIDER_SUSTAIN, 0, 0),
        EventType.SLIDER_REPEAT_SUSTAIN: EventRange(EventType.SLIDER_REPEAT_SUSTAIN, 0, 0),
        EventType.DURATION: EventRange(EventType.DURATION, 0, DURATION_COUNT - 1),
        EventType.BEAT: EventRange(EventType.BEAT, 0, 0),
        EventType.MEASURE: EventRange(EventType.MEASURE, 0, 0),
        EventType.TIMING_POINT: EventRange(EventType.TIMING_POINT, 0, 0),
        EventType.KIAI: EventRange(EventType.KIAI, 0, 1),
        EventType.SCROLL_SPEED: EventRange(EventType.SCROLL_SPEED, 0, config.scroll_speed_max),
        EventType.SAMPLESET: EventRange(EventType.SAMPLESET, 0, 2),
    }
