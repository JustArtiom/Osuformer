from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.osu import Slider


class TokenType:
    EOS = 0
    CIRCLE = 1
    SLIDER = 2


class TokenAttr:
    TYPE = 0
    DELTA = 1
    START_X = 2
    START_Y = 3
    END_X = 4
    END_Y = 5
    CTRL1_X = 6
    CTRL1_Y = 7
    CTRL2_X = 8
    CTRL2_Y = 9
    DURATION = 10
    SLIDES = 11
    CURVE_TYPE = 12
    SLIDER_SV = 13

    COUNT = 14


@dataclass
class TokenSpec:
    type_id: int
    delta: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    ctrl1_x: int
    ctrl1_y: int
    ctrl2_x: int
    ctrl2_y: int
    duration: int
    slides: int
    curve_type: int
    slider_sv: int

    def to_list(self) -> List[int]:
        return [
            self.type_id,
            self.delta,
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.ctrl1_x,
            self.ctrl1_y,
            self.ctrl2_x,
            self.ctrl2_y,
            self.duration,
            self.slides,
            self.curve_type,
            self.slider_sv,
        ]


class HitObjectTokenizer:
    def __init__(self, data_cfg: dict) -> None:
        self.osu_width = data_cfg["osu_width"]
        self.osu_height = data_cfg["osu_height"]
        self.bin_size = int(data_cfg.get("position_bin_size", 8))
        self.max_duration_ticks = int(data_cfg.get("max_slider_ticks", 64))
        self.max_slides = int(data_cfg.get("max_slides", 4))
        self.max_control_points = int(data_cfg.get("max_slider_control_points", 2))
        self.sv_precision = int(data_cfg.get("slider_sv_precision", 100))
        self.max_sv = float(data_cfg.get("slider_sv_max", 4.0))
        default_delta = data_cfg.get("beats_per_sample", 16) * data_cfg.get("ticks_per_beat", 8)
        delta_override = data_cfg.get("max_delta_ticks")
        self.max_delta_ticks = int(delta_override if delta_override is not None else default_delta)
        self.max_delta_ticks = max(1, self.max_delta_ticks)

        self.max_x_bin = int(math.ceil(self.osu_width / self.bin_size))
        self.max_y_bin = int(math.ceil(self.osu_height / self.bin_size))
        self.x_bins = self.max_x_bin + 2  # include sentinel
        self.y_bins = self.max_y_bin + 2
        self.delta_bins = self.max_delta_ticks + 2  # sentinel + range
        self.duration_bins = self.max_duration_ticks + 1  # include sentinel
        self.slide_bins = self.max_slides + 1  # include sentinel
        self.curve_types = data_cfg.get("slider_curve_types", ["L", "B", "P", "C"])
        self.curve_type_to_id = {name.upper(): idx + 1 for idx, name in enumerate(self.curve_types)}
        self.curve_bins = len(self.curve_type_to_id) + 1
        self.sv_bins = int(self.max_sv * self.sv_precision) + 2  # sentinel + range

        self.attribute_sizes = [
            3,  # type (EOS, CIRCLE, SLIDER)
            self.delta_bins,
            self.x_bins,
            self.y_bins,
            self.x_bins,
            self.y_bins,
            self.x_bins,
            self.y_bins,
            self.x_bins,
            self.y_bins,
            self.duration_bins,
            self.slide_bins,
            self.curve_bins,
            self.sv_bins,
        ]

    # ---------------- Quantization helpers ----------------

    def _quantize_coord(self, value: float, max_bin: int) -> int:
        bin_idx = int(round(value / self.bin_size))
        bin_idx = max(0, min(bin_idx, max_bin))
        return bin_idx + 1  # reserve 0 as sentinel

    def _encode_coord_pair(self, x: float, y: float) -> Tuple[int, int]:
        return self._quantize_coord(x, self.max_x_bin), self._quantize_coord(y, self.max_y_bin)

    def _encode_control_points(self, slider: Slider) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        if slider.object_params and slider.object_params.curves:
            for curve in slider.object_params.curves:
                for px, py in curve.curve_points:
                    points.append((float(px), float(py)))

        encoded: List[Tuple[int, int]] = []
        for px, py in points[: self.max_control_points]:
            encoded.append(self._encode_coord_pair(px, py))
        while len(encoded) < self.max_control_points:
            encoded.append((0, 0))
        return encoded

    def _slider_end_point(self, slider: Slider) -> Tuple[float, float]:
        end_point = (slider.x, slider.y)
        if slider.object_params and slider.object_params.curves:
            curve_points: List[Tuple[float, float]] = []
            for curve in slider.object_params.curves:
                curve_points.extend([(float(px), float(py)) for px, py in curve.curve_points])
            if curve_points:
                end_point = curve_points[-1]

        if slider.object_params and slider.object_params.slides % 2 == 0:
            end_point = (slider.x, slider.y)
        return float(end_point[0]), float(end_point[1])

    def _slider_curve_type_id(self, slider: Slider) -> int:
        curve_type = "L"
        if slider.object_params and slider.object_params.curves:
            curve_type = slider.object_params.curves[0].curve_type or "L"
        return self.curve_type_to_id.get(curve_type.upper(), 1)

    # ---------------- Public API ----------------

    def pad_token(self) -> List[int]:
        return TokenSpec(TokenType.EOS, *(0 for _ in range(TokenAttr.COUNT - 1))).to_list()

    def eos_token(self) -> List[int]:
        return self.pad_token()

    def empty_token(self) -> List[int]:
        return self.pad_token()

    def _encode_delta(self, delta_ticks: int) -> int:
        clamped = max(0, min(int(round(delta_ticks)), self.max_delta_ticks))
        return clamped + 1

    def delta_from_id(self, delta_id: int) -> int:
        if delta_id <= 0:
            return 0
        return delta_id - 1

    def encode_circle(self, x: float, y: float, delta_ticks: int) -> List[int]:
        sx, sy = self._encode_coord_pair(x, y)
        spec = TokenSpec(
            type_id=TokenType.CIRCLE,
            delta=self._encode_delta(delta_ticks),
            start_x=sx,
            start_y=sy,
            end_x=0,
            end_y=0,
            ctrl1_x=0,
            ctrl1_y=0,
            ctrl2_x=0,
            ctrl2_y=0,
            duration=0,
            slides=0,
            curve_type=0,
            slider_sv=0,
        )
        return spec.to_list()

    def encode_slider(self, slider: Slider, tick_duration_ms: float, sv_multiplier: float, delta_ticks: int) -> List[int]:
        start_x, start_y = self._encode_coord_pair(slider.x, slider.y)
        end_x_val, end_y_val = self._slider_end_point(slider)
        end_x, end_y = self._encode_coord_pair(end_x_val, end_y_val)
        control_points = self._encode_control_points(slider)
        ctrl1_x, ctrl1_y = control_points[0]
        ctrl2_x, ctrl2_y = control_points[1]

        duration_ms = slider.object_params.duration if slider.object_params else 0.0
        duration_ticks = 0
        if tick_duration_ms > 0:
            duration_ticks = int(round(duration_ms / tick_duration_ms))
        duration_ticks = max(1, min(duration_ticks, self.max_duration_ticks))

        slides = slider.object_params.slides if slider.object_params else 1
        slides = max(1, min(int(slides), self.max_slides))
        sv_id = self._encode_sv(sv_multiplier)

        spec = TokenSpec(
            type_id=TokenType.SLIDER,
            delta=self._encode_delta(delta_ticks),
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            ctrl1_x=ctrl1_x,
            ctrl1_y=ctrl1_y,
            ctrl2_x=ctrl2_x,
            ctrl2_y=ctrl2_y,
            duration=duration_ticks,
            slides=slides,
            curve_type=self._slider_curve_type_id(slider),
            slider_sv=sv_id,
        )
        return spec.to_list()

    # --------- Decoding helpers for generation ---------

    def coord_from_id(self, coord_id: int) -> float | None:
        if coord_id <= 0:
            return None
        bin_idx = coord_id - 1
        return min(float(self.osu_width), bin_idx * self.bin_size)

    def decode_token(self, token: Sequence[int]) -> dict:
        token_type = token[TokenAttr.TYPE]
        result = {"type": token_type}

        result["delta_ticks"] = self.delta_from_id(token[TokenAttr.DELTA])
        if token_type == TokenType.EOS:
            return result
        result["start_x"] = self.coord_from_id(token[TokenAttr.START_X])
        result["start_y"] = self.coord_from_id(token[TokenAttr.START_Y])

        if token_type == TokenType.SLIDER:
            result["end_x"] = self.coord_from_id(token[TokenAttr.END_X])
            result["end_y"] = self.coord_from_id(token[TokenAttr.END_Y])
            result["ctrl1_x"] = self.coord_from_id(token[TokenAttr.CTRL1_X])
            result["ctrl1_y"] = self.coord_from_id(token[TokenAttr.CTRL1_Y])
            result["ctrl2_x"] = self.coord_from_id(token[TokenAttr.CTRL2_X])
            result["ctrl2_y"] = self.coord_from_id(token[TokenAttr.CTRL2_Y])
            result["duration_ticks"] = token[TokenAttr.DURATION]
            result["slides"] = token[TokenAttr.SLIDES]
            result["curve_type"] = self._curve_type_from_id(token[TokenAttr.CURVE_TYPE])
            result["sv_factor"] = self.sv_from_id(token[TokenAttr.SLIDER_SV])

        return result

    def _curve_type_from_id(self, idx: int) -> str:
        if idx <= 0:
            return "L"
        for name, stored_idx in self.curve_type_to_id.items():
            if stored_idx == idx:
                return name
        return "L"

    def _encode_sv(self, sv: float) -> int:
        if sv is None or sv <= 0:
            return 0
        value = int(round(min(self.max_sv, max(1e-3, sv)) * self.sv_precision))
        value = max(1, min(value, self.sv_bins - 1))
        return value

    def sv_from_id(self, sv_id: int) -> float | None:
        if sv_id <= 0:
            return None
        return min(self.max_sv, sv_id / self.sv_precision)
