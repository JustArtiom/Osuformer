from __future__ import annotations

from src.osu.beatmap import Beatmap


def count_storyboard_events(beatmap: Beatmap) -> int:
    return len(beatmap.events._unparsed_lines)


def detect_video(beatmap: Beatmap) -> bool:
    return beatmap.events.video is not None


def custom_sample_ratio(beatmap: Beatmap) -> float:
    objs = beatmap.hit_objects
    if not objs:
        return 0.0
    custom = 0
    for o in objs:
        sample = getattr(o, "hit_sample", None)
        if sample is None:
            continue
        filename = getattr(sample, "filename", "")
        if filename:
            custom += 1
    return custom / len(objs)
