from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Allow running as a script from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.osu import Beatmap, Slider, Spinner  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float | None:
    # Returns angle in degrees between two vectors; None if a vector is too small.
    x1, y1 = v1
    x2, y2 = v2
    norm1 = math.hypot(x1, y1)
    norm2 = math.hypot(x2, y2)
    if norm1 < 1e-3 or norm2 < 1e-3:
        return None
    dot = x1 * x2 + y1 * y2
    cos = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos))


def analyze_map(
    osu_path: Path,
    *,
    stream_interval_ms: float,
    stream_ratio: float,
    stream_gap_distance: float,
    aim_jump_distance: float,
    aim_ratio: float,
    tech_angle_std: float = 40.0,
    tech_slider_ratio: float = 0.35,
    tech_timing_std: float = 90.0,
) -> Dict[str, object]:
    bm = Beatmap(file_path=str(osu_path))
    objects = [ho for ho in bm.hit_objects if not isinstance(ho, Spinner)]
    if len(objects) < 2:
        return {"path": str(osu_path), "labels": {"stream": False, "jump": False, "tech": False}}

    dts: List[float] = []
    dists: List[float] = []
    angles: List[float] = []
    slider_count = 0

    for prev, cur in zip(objects, objects[1:]):
        dt = float(cur.time - prev.time)
        dx = float(cur.x - prev.x)
        dy = float(cur.y - prev.y)
        dist = math.hypot(dx, dy)
        dts.append(dt)
        dists.append(dist)
    for prev, cur, nxt in zip(objects, objects[1:], objects[2:]):
        v1 = (float(cur.x - prev.x), float(cur.y - prev.y))
        v2 = (float(nxt.x - cur.x), float(nxt.y - cur.y))
        ang = angle_between(v1, v2)
        if ang is not None:
            angles.append(ang)
    for ho in objects:
        if isinstance(ho, Slider):
            slider_count += 1

    intervals = len(dts)
    stream_hits = sum(1 for dt, dist in zip(dts, dists) if dt <= stream_interval_ms and dist <= stream_gap_distance)
    jump_hits = sum(1 for dist in dists if dist >= aim_jump_distance)
    stream_score = stream_hits / max(1, intervals)
    jump_score = jump_hits / max(1, intervals)
    slider_ratio = slider_count / max(1, len(objects))
    angle_std = statistics.pstdev(angles) if angles else 0.0
    timing_std = statistics.pstdev(dts) if dts else 0.0

    labels = {
        "stream": stream_score >= stream_ratio,
        "jump": jump_score >= aim_ratio,
        "tech": angle_std >= tech_angle_std and slider_ratio >= tech_slider_ratio and timing_std >= tech_timing_std,
    }

    return {
        "path": str(osu_path),
        "labels": labels,
        "metrics": {
            "stream_score": stream_score,
            "jump_score": jump_score,
            "angle_std": angle_std,
            "slider_ratio": slider_ratio,
            "timing_std": timing_std,
        },
    }


def iter_osu_files(base_path: Path) -> Iterable[Path]:
    return base_path.rglob("*.osu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic labels for stream/jump/tech maps.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--data-path", type=str, default=None, help="Override paths.data from config.")
    parser.add_argument("--tech-angle-std", type=float, default=40.0, help="Angle stddev threshold for tech.")
    parser.add_argument("--tech-slider-ratio", type=float, default=0.35, help="Min slider fraction for tech.")
    parser.add_argument("--tech-timing-std", type=float, default=90.0, help="Timing stddev (ms) threshold for tech.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files (for quick tests).")
    parser.add_argument("--details", action="store_true", help="Print per-map labels and scores.")
    parser.add_argument("--output", type=str, default=None, help="Optional TSV file to save per-map scores.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    thresholds = cfg.get("map_type_thresholds") or cfg.get("filters", {}).get("map_type_thresholds", {})
    stream_interval_ms = float(thresholds.get("stream_interval_ms", 110.0))
    stream_ratio = float(thresholds.get("stream_ratio", 0.35))
    stream_gap_distance = float(thresholds.get("stream_gap_distance", 80.0))
    aim_jump_distance = float(thresholds.get("aim_jump_distance", 120.0))
    aim_ratio = float(thresholds.get("aim_ratio", 0.3))

    data_path = Path(args.data_path or cfg["paths"]["data"]).expanduser().resolve()
    osu_files = list(iter_osu_files(data_path))
    if args.limit:
        osu_files = osu_files[: args.limit]

    total = 0
    counts = {"stream": 0, "jump": 0, "tech": 0}
    results = []
    for osu_file in osu_files:
        try:
            result = analyze_map(
                osu_file,
                stream_interval_ms=stream_interval_ms,
                stream_ratio=stream_ratio,
                stream_gap_distance=stream_gap_distance,
                aim_jump_distance=aim_jump_distance,
                aim_ratio=aim_ratio,
                tech_angle_std=args.tech_angle_std,
                tech_slider_ratio=args.tech_slider_ratio,
                tech_timing_std=args.tech_timing_std,
            )
        except Exception as exc:  # pragma: no cover - defensive against malformed maps
            print(f"[WARN] Failed to analyze {osu_file}: {exc}", file=sys.stderr)
            continue
        total += 1
        for label, is_on in result["labels"].items():
            if is_on:
                counts[label] += 1
        results.append(result)

    print(f"Scanned {total} maps under {data_path}")
    for label, c in counts.items():
        pct = (c / total * 100) if total else 0.0
        print(f"  {label}: {c} ({pct:.1f}%)")

    def map_and_diff(path_str: str) -> tuple[str, str]:
        name = Path(path_str).name
        stem = Path(name).stem
        if "[" in stem and "]" in stem:
            main, diff = stem.rsplit("[", 1)
            return main.strip(), diff.strip("] ")
        return stem, ""

    def trim(name: str, width: int) -> str:
        return name if len(name) <= width else name[: width - 1] + "…"

    if args.details:
        map_w, diff_w = 40, 24
        header = (
            f"{'map':<{map_w}}  {'diff':<{diff_w}}  stream  jump  tech  "
            f"stream_score  jump_score  angle_std  slider_ratio  timing_std"
        )
        print("\n" + header)
        for r in results:
            lbl = r["labels"]
            m = r["metrics"]
            map_name, diff_name = map_and_diff(r["path"])
            print(
                f"{trim(map_name, map_w):<{map_w}}  {trim(diff_name, diff_w):<{diff_w}}  "
                f"{int(lbl['stream']):>6}  {int(lbl['jump']):>4}  {int(lbl['tech']):>4}  "
                f"{m['stream_score']:.3f}       {m['jump_score']:.3f}      "
                f"{m['angle_std']:.1f}      {m['slider_ratio']:.3f}        {m['timing_std']:.1f}"
            )

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("map\tdiff\tstream\tjump\ttech\tstream_score\tjump_score\tangle_std\tslider_ratio\ttiming_std\n")
            for r in results:
                lbl = r["labels"]
                m = r["metrics"]
                map_name, diff_name = map_and_diff(r["path"])
                f.write(
                    f"{map_name}\t{diff_name}\t"
                    f"{int(lbl['stream'])}\t{int(lbl['jump'])}\t{int(lbl['tech'])}\t"
                    f"{m['stream_score']:.3f}\t{m['jump_score']:.3f}\t"
                    f"{m['angle_std']:.1f}\t{m['slider_ratio']:.3f}\t{m['timing_std']:.1f}\n"
                )
        print(f"Wrote per-map scores to {out_path}")


if __name__ == "__main__":
    main()
