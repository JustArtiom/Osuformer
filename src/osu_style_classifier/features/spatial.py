from __future__ import annotations

from src.osu.beatmap import Beatmap
from src.osu.hit_object import Circle, Slider

from .feature_set import SpatialFeatures
from .geometry import distance


PLAYFIELD_WIDTH = 512.0
PLAYFIELD_HEIGHT = 384.0
PLAYFIELD_CENTER_X = PLAYFIELD_WIDTH / 2.0
PLAYFIELD_CENTER_Y = PLAYFIELD_HEIGHT / 2.0


def extract_spatial_features(
    beatmap: Beatmap,
    grid_coord_step: float,
    stack_max_distance_px: float,
) -> SpatialFeatures:
    objs = [o for o in beatmap.hit_objects if isinstance(o, (Circle, Slider))]
    if not objs:
        return SpatialFeatures()

    positions = [(float(o.x), float(o.y)) for o in objs]

    grid_hits = sum(
        1
        for x, y in positions
        if _is_grid_aligned(x, grid_coord_step) and _is_grid_aligned(y, grid_coord_step)
    )
    grid_ratio = grid_hits / len(positions)

    coverage = _coverage_ratio(positions)
    sym_x = _x_symmetry_score(positions)
    sym_y = _y_symmetry_score(positions)

    stack_pairs, perfect_stacks = _count_stacks(objs, stack_max_distance_px)
    overlap_ratio = _overlap_ratio(objs, float(beatmap.difficulty.circle_size))

    return SpatialFeatures(
        coverage_ratio=coverage,
        grid_snap_ratio=grid_ratio,
        symmetry_x_score=sym_x,
        symmetry_y_score=sym_y,
        stack_pairs=stack_pairs,
        perfect_stack_count=perfect_stacks,
        overlap_ratio=overlap_ratio,
    )


def _is_grid_aligned(value: float, step: float) -> bool:
    if step <= 0:
        return False
    remainder = value % step
    return remainder < 0.01 or abs(remainder - step) < 0.01


def _coverage_ratio(positions: list[tuple[float, float]]) -> float:
    cols = 16
    rows = 12
    cell_w = PLAYFIELD_WIDTH / cols
    cell_h = PLAYFIELD_HEIGHT / rows
    cells: set[tuple[int, int]] = set()
    for x, y in positions:
        cx = int(max(0.0, min(cols - 1, x // cell_w)))
        cy = int(max(0.0, min(rows - 1, y // cell_h)))
        cells.add((cx, cy))
    return len(cells) / (cols * rows)


def _x_symmetry_score(positions: list[tuple[float, float]]) -> float:
    if not positions:
        return 0.0
    cell = 16.0
    counts: dict[tuple[float, float], int] = {}
    for x, y in positions:
        counts[(round(x / cell), round(y / cell))] = counts.get((round(x / cell), round(y / cell)), 0) + 1
    matched = 0
    for (cx, cy), n in counts.items():
        mirror_cx = round((PLAYFIELD_WIDTH - cx * cell) / cell)
        if (mirror_cx, cy) in counts:
            matched += min(n, counts[(mirror_cx, cy)])
    return matched / len(positions)


def _y_symmetry_score(positions: list[tuple[float, float]]) -> float:
    if not positions:
        return 0.0
    cell = 16.0
    counts: dict[tuple[float, float], int] = {}
    for x, y in positions:
        counts[(round(x / cell), round(y / cell))] = counts.get((round(x / cell), round(y / cell)), 0) + 1
    matched = 0
    for (cx, cy), n in counts.items():
        mirror_cy = round((PLAYFIELD_HEIGHT - cy * cell) / cell)
        if (cx, mirror_cy) in counts:
            matched += min(n, counts[(cx, mirror_cy)])
    return matched / len(positions)


def _count_stacks(objs: list[Circle | Slider], max_distance_px: float) -> tuple[int, int]:
    stack_pairs = 0
    perfect = 0
    for i in range(len(objs) - 1):
        a = (float(objs[i].x), float(objs[i].y))
        b = (float(objs[i + 1].x), float(objs[i + 1].y))
        d = distance(a, b)
        if d <= max_distance_px:
            stack_pairs += 1
            if d < 0.01:
                perfect += 1
    return stack_pairs, perfect


def _overlap_ratio(objs: list[Circle | Slider], circle_size: float) -> int | float:
    radius_px = 54.4 - 4.48 * circle_size
    threshold = radius_px * 0.9
    count = 0
    pairs = 0
    for i in range(len(objs) - 1):
        a = (float(objs[i].x), float(objs[i].y))
        b = (float(objs[i + 1].x), float(objs[i + 1].y))
        d = distance(a, b)
        pairs += 1
        if 0.01 < d < threshold:
            count += 1
    return count / pairs if pairs > 0 else 0.0
