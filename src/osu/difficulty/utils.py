from __future__ import annotations
from math import acos
from typing import Iterable, Tuple

def clock_rate_from_mods(mods: Iterable[str]) -> float:
    m = {x.upper() for x in mods}
    rate = 1.0
    if "DT" in m or "NC" in m:
        rate *= 1.5
    if "HT" in m:
        rate *= 0.75
    return rate

def dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return (dx*dx + dy*dy) ** 0.5

def angle_between(prev_prev_xy: Tuple[float,float] | None,
                  prev_xy: Tuple[float,float],
                  curr_xy: Tuple[float,float]) -> float | None:
    if prev_prev_xy is None:
        return None
    (x1, y1), (x2, y2), (x3, y3) = prev_prev_xy, prev_xy, curr_xy
    ux, uy = x2 - x1, y2 - y1
    vx, vy = x3 - x2, y3 - y2
    um = (ux*ux + uy*uy) ** 0.5
    vm = (vx*vx + vy*vy) ** 0.5
    if um < 1e-6 or vm < 1e-6:
        return None
    cosang = max(-1.0, min(1.0, (ux*vx + uy*vy) / (um*vm)))
    return acos(cosang)
