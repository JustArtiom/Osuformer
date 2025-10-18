# difficulty/annotate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .utils import clock_rate_from_mods, dist_xy, angle_between
from .constants import BONUS_CAP_MS

from osu import Slider, Spinner, SliderObjectParams

@dataclass
class Anno:
    obj: object
    strain_time: float
    jump_dist: float
    angle: float | None
    prev_end_xy: Tuple[float, float]

def _is_slider(o: object) -> bool:
    return isinstance(o, Slider)

def _is_spinner(o: object) -> bool:
    return isinstance(o, Spinner)

def _slider_lazy_end_xy(sl: Slider) -> tuple[float, float]:
    params = sl.object_params
    if isinstance(params, SliderObjectParams) and params.curve_points:
        return tuple(params.curve_points[-1])
    return (sl.x, sl.y)

def _end_time_of(prev: object, clock_rate: float) -> float:
    t = prev.time
    if _is_slider(prev) and isinstance(prev.object_params, SliderObjectParams):
        # TODO: replace with real timing/SV-based duration
        base_px_per_s = 100.0
        repeats = max(1, int(prev.object_params.slides))
        dur_ms = (prev.object_params.length / max(1e-6, base_px_per_s)) * 1000.0 * repeats
        t = prev.time + (dur_ms / max(1e-6, clock_rate))
    return t

def annotate(objects: List[object], mods: Iterable[str]) -> List[Anno]:
    if not objects:
        return []

    objs = sorted(objects, key=lambda o: o.time)
    rate = clock_rate_from_mods(mods)

    out: List[Anno] = []
    prev = None
    prev_prev = None
    prev_end_xy = None

    for curr in objs:
        if _is_spinner(curr):
            prev_prev = prev
            prev = curr
            prev_end_xy = (curr.x, curr.y)
            continue

        if prev is None:
            out.append(Anno(curr, 0.0, 0.0, None, (curr.x, curr.y)))
            prev = curr
            prev_end_xy = (curr.x, curr.y)
            continue

        eff_prev_end_t = _end_time_of(prev, rate)
        dt = (curr.time - eff_prev_end_t) / max(1e-6, rate)
        if dt < 0:
            dt = 0.0

        if _is_slider(prev):
            prev_end_xy = _slider_lazy_end_xy(prev)
        else:
            prev_end_xy = (prev.x, prev.y)

        jd = dist_xy(prev_end_xy[0], prev_end_xy[1], curr.x, curr.y)
        ang = angle_between(
            None if prev_prev is None else ((prev_prev.x, prev_prev.y)),
            (prev.x, prev.y),
            (curr.x, curr.y)
        )

        out.append(Anno(curr, dt, jd, ang, prev_end_xy))

        prev_prev = prev
        prev = curr

    for a in out:
        if 0.0 < a.strain_time < BONUS_CAP_MS:
            pass
    return out
