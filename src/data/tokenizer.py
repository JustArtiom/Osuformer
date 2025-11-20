from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from src.osu import Circle, HitObject, Slider


class TokenType:
    EOS = 0
    CIRCLE = 1
    SLIDER = 2
    SLIDER_PATH = 3


class TokenAttr:
    TYPE = 0
    TICK = 1
    X = 2
    Y = 3
    DURATION = 4
    SLIDES = 5
    CURVE_TYPE = 6
    SLIDER_SV = 7

    COUNT = 8


@dataclass
class TokenSpec:
    type_id: int
    tick: int
    x: int
    y: int
    duration: int
    slides: int
    curve_type: int
    slider_sv: int

    def to_list(self) -> List[int]:
        return [
            self.type_id,
            self.tick,
            self.x,
            self.y,
            self.duration,
            self.slides,
            self.curve_type,
            self.slider_sv,
        ]


class HitObjectTokenizer:
    def __init__(self, data_cfg: dict) -> None:
        self.osu_width = int(data_cfg.get("osu_width", 512))
        self.osu_height = int(data_cfg.get("osu_height", 384))
        self.bin_size = int(data_cfg.get("position_bin_size", 8))
        self.max_duration_ticks = int(data_cfg.get("max_slider_ticks", 64))
        self.max_slides = int(data_cfg.get("max_slides", 4))
        self.curve_types = data_cfg.get("slider_curve_types", ["L", "B", "P", "C"])
        self.curve_type_to_id = {name.upper(): idx + 1 for idx, name in enumerate(self.curve_types)}
        self.sv_precision = int(data_cfg.get("slider_sv_precision", 100))
        self.max_sv = float(data_cfg.get("slider_sv_max", 4.0))
        self.ticks_per_beat = int(data_cfg.get("ticks_per_beat", 4))
        self.context_beats = int(data_cfg.get("context_beats", 8))
        self.target_beats = int(data_cfg.get("target_beats", 16))
        self.total_beats = self.context_beats + self.target_beats
        self.seq_len = self.total_beats * self.ticks_per_beat

        self.max_x_bin = int(math.ceil(self.osu_width / self.bin_size))
        self.max_y_bin = int(math.ceil(self.osu_height / self.bin_size))
        self.x_bins = self.max_x_bin + 2  # sentinel + range
        self.y_bins = self.max_y_bin + 2
        self.tick_bins = self.seq_len + 2  # sentinel + range
        self.duration_bins = self.max_duration_ticks + 2  # sentinel + range
        self.slide_bins = self.max_slides + 2
        self.curve_bins = len(self.curve_type_to_id) + 2
        self.sv_bins = int(self.max_sv * self.sv_precision) + 2

        self.attribute_sizes = [
            4,  # token types
            self.tick_bins,
            self.x_bins,
            self.y_bins,
            self.duration_bins,
            self.slide_bins,
            self.curve_bins,
            self.sv_bins,
        ]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def pad_token(self) -> List[int]:
        return TokenSpec(TokenType.EOS, 0, 0, 0, 0, 0, 0, 0).to_list()

    def eos_token(self) -> List[int]:
        return self.pad_token()

    def _encode_tick(self, tick_index: int) -> int:
        clamped = max(0, min(int(round(tick_index)), self.seq_len))
        return clamped + 1

    def encode_tick(self, tick_index: int) -> int:
        return self._encode_tick(tick_index)

    def tick_from_id(self, tick_id: int) -> int:
        if tick_id <= 0:
            return 0
        return min(self.seq_len, tick_id - 1)

    def _encode_coord(self, value: float, max_bin: int) -> int:
        bin_idx = int(round(float(value) / max(1, self.bin_size)))
        bin_idx = max(0, min(bin_idx, max_bin))
        return bin_idx + 1

    def coord_from_id(self, coord_id: int) -> Optional[float]:
        if coord_id <= 0:
            return None
        return float((coord_id - 1) * self.bin_size)

    def _encode_duration(self, duration_ticks: int) -> int:
        clamped = max(0, min(int(duration_ticks), self.max_duration_ticks))
        return clamped + 1

    def _encode_slides(self, slides: int) -> int:
        clamped = max(1, min(int(slides), self.max_slides))
        return clamped + 1

    def _encode_curve(self, slider: Slider) -> int:
        curve_type = "L"
        if slider.object_params and slider.object_params.curves:
            curve_type = slider.object_params.curves[0].curve_type or "L"
        return self.curve_type_to_id.get(curve_type.upper(), 1)

    def _encode_sv(self, value: float) -> int:
        if value is None or value <= 0:
            return 0
        scaled = int(round(min(self.max_sv, max(1e-4, float(value))) * self.sv_precision))
        return max(1, min(scaled, self.sv_bins - 1))

    def sv_from_id(self, sv_id: int) -> Optional[float]:
        if sv_id <= 0:
            return None
        return min(self.max_sv, sv_id / self.sv_precision)

    def _slider_control_points(self, slider: Slider) -> List[Tuple[float, float]]:
        if not slider.object_params or not slider.object_params.curves:
            return []
        points: List[Tuple[float, float]] = []
        for curve in slider.object_params.curves:
            points.extend([(float(px), float(py)) for px, py in curve.curve_points])
        return points

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def encode_circle(self, circle: Circle, tick_index: int) -> List[int]:
        return TokenSpec(
            type_id=TokenType.CIRCLE,
            tick=self._encode_tick(tick_index),
            x=self._encode_coord(circle.x, self.max_x_bin),
            y=self._encode_coord(circle.y, self.max_y_bin),
            duration=0,
            slides=0,
            curve_type=0,
            slider_sv=0,
        ).to_list()

    def encode_slider(
        self,
        slider: Slider,
        tick_index: int,
        duration_ticks: int,
        sv_multiplier: float,
    ) -> List[List[int]]:
        start_token = TokenSpec(
            type_id=TokenType.SLIDER,
            tick=self._encode_tick(tick_index),
            x=self._encode_coord(slider.x, self.max_x_bin),
            y=self._encode_coord(slider.y, self.max_y_bin),
            duration=self._encode_duration(duration_ticks),
            slides=self._encode_slides(getattr(slider.object_params, "slides", 1) or 1),
            curve_type=self._encode_curve(slider),
            slider_sv=self._encode_sv(sv_multiplier),
        ).to_list()

        path_tokens: List[List[int]] = []
        for px, py in self._slider_control_points(slider):
            path_tokens.append(
                TokenSpec(
                    type_id=TokenType.SLIDER_PATH,
                    tick=self._encode_tick(tick_index),
                    x=self._encode_coord(px, self.max_x_bin),
                    y=self._encode_coord(py, self.max_y_bin),
                    duration=0,
                    slides=0,
                    curve_type=0,
                    slider_sv=0,
                ).to_list()
            )

        return [start_token] + path_tokens

    def tokenize(
        self,
        hit_objects: Sequence[HitObject],
        *,
        chunk_start_ms: float,
        tick_duration_ms: float,
        max_ticks: Optional[int] = None,
        slider_sv_lookup: Optional[Callable[[Slider], float]] = None,
        tick_tolerance_ms: float = 0.0,
    ) -> List[List[int]]:
        if max_ticks is None:
            max_ticks = self.seq_len
        tick_ms = max(1e-6, float(tick_duration_ms))
        tokens: List[List[int]] = []
        events = sorted(hit_objects, key=lambda ho: float(getattr(ho, "time", 0.0)))

        for ho in events:
            time_ms = float(getattr(ho, "time", 0.0))
            rel = time_ms - chunk_start_ms
            tick_index = int(round(rel / tick_ms))
            snapped = chunk_start_ms + tick_index * tick_ms
            if tick_tolerance_ms > 0 and abs(time_ms - snapped) > tick_tolerance_ms:
                continue
            tick_index = max(0, min(tick_index, max_ticks - 1))

            if isinstance(ho, Circle):
                tokens.append(self.encode_circle(ho, tick_index))
            elif isinstance(ho, Slider):
                duration_ms = float(getattr(getattr(ho, "object_params", None), "duration", 0.0) or 0.0)
                duration_ticks = max(1, int(round(duration_ms / tick_ms)))
                if tick_index + duration_ticks > max_ticks:
                    continue
                sv_value = slider_sv_lookup(ho) if slider_sv_lookup else 1.0
                tokens.extend(self.encode_slider(ho, tick_index, duration_ticks, sv_value))

        return tokens

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    def _curve_type_from_id(self, idx: int) -> str:
        if idx <= 0:
            return "L"
        for name, value in self.curve_type_to_id.items():
            if value == idx:
                return name
        return "L"

    def detokenize(
        self,
        tokens: Sequence[Sequence[int]],
        *,
        chunk_start_ms: float,
        tick_duration_ms: float,
        cutoff_tick: Optional[int] = None,
    ) -> List[dict]:
        tick_ms = max(1e-6, float(tick_duration_ms))
        events: List[dict] = []
        idx = 0
        max_tick = cutoff_tick if cutoff_tick is not None else self.seq_len

        while idx < len(tokens):
            token = tokens[idx]
            token_type = token[TokenAttr.TYPE]
            if token_type == TokenType.EOS:
                break
            if token_type == TokenType.SLIDER_PATH:
                idx += 1
                continue

            tick_index = self.tick_from_id(token[TokenAttr.TICK])
            if tick_index >= max_tick:
                break

            time_ms = chunk_start_ms + tick_index * tick_ms
            if token_type == TokenType.CIRCLE:
                x = self.coord_from_id(token[TokenAttr.X])
                y = self.coord_from_id(token[TokenAttr.Y])
                if x is None or y is None:
                    idx += 1
                    continue
                events.append(
                    {
                        "type": "circle",
                        "tick": tick_index,
                        "time": time_ms,
                        "x": x,
                        "y": y,
                    }
                )
            elif token_type == TokenType.SLIDER:
                x = self.coord_from_id(token[TokenAttr.X])
                y = self.coord_from_id(token[TokenAttr.Y])
                if x is None or y is None:
                    idx += 1
                    continue
                duration_ticks = max(0, token[TokenAttr.DURATION] - 1)
                slides = max(1, token[TokenAttr.SLIDES] - 1)
                curve_type = self._curve_type_from_id(token[TokenAttr.CURVE_TYPE])
                sv_value = self.sv_from_id(token[TokenAttr.SLIDER_SV]) or 1.0
                control_points: List[Tuple[float, float]] = []
                look_ahead = idx + 1
                while look_ahead < len(tokens):
                    next_token = tokens[look_ahead]
                    if next_token[TokenAttr.TYPE] != TokenType.SLIDER_PATH:
                        break
                    px = self.coord_from_id(next_token[TokenAttr.X])
                    py = self.coord_from_id(next_token[TokenAttr.Y])
                    if px is not None and py is not None:
                        control_points.append((px, py))
                    look_ahead += 1
                idx = look_ahead - 1
                events.append(
                    {
                        "type": "slider",
                        "tick": tick_index,
                        "time": time_ms,
                        "x": x,
                        "y": y,
                        "duration_ticks": duration_ticks,
                        "slides": slides,
                        "curve_type": curve_type,
                        "sv": sv_value,
                        "points": control_points,
                    }
                )
            idx += 1
        return events

    def decode_token(self, token: Sequence[int]) -> dict:
        token_type = token[TokenAttr.TYPE]
        tick = self.tick_from_id(token[TokenAttr.TICK])
        info = {"type": token_type, "tick_index": tick}
        if token_type == TokenType.EOS:
            return info
        info["x"] = self.coord_from_id(token[TokenAttr.X])
        info["y"] = self.coord_from_id(token[TokenAttr.Y])
        if token_type == TokenType.SLIDER:
            info["duration_ticks"] = max(0, token[TokenAttr.DURATION] - 1)
            info["slides"] = max(1, token[TokenAttr.SLIDES] - 1)
            info["curve_type"] = self._curve_type_from_id(token[TokenAttr.CURVE_TYPE])
            info["sv_factor"] = self.sv_from_id(token[TokenAttr.SLIDER_SV])
        return info
