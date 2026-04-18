from __future__ import annotations

from src.osu.beatmap import Beatmap

from ..config import ClassifierConfig
from .feature_set import BeatmapFeatures
from .misc import count_storyboard_events, custom_sample_ratio, detect_video
from .objects import extract_object_features
from .sliders import extract_slider_features
from .spatial import extract_spatial_features
from .streams import extract_stream_features
from .timing import extract_timing_features


def extract_features(beatmap: Beatmap, config: ClassifierConfig) -> BeatmapFeatures:
    timing = extract_timing_features(beatmap)
    objects, snap = extract_object_features(beatmap, config.streams.snap_tolerance_ms)
    streams = extract_stream_features(
        beatmap,
        snap_tolerance_ms=config.streams.snap_tolerance_ms,
        min_run_length=config.streams.min_run_length,
        spaced_px_per_ms=config.streams.spaced_px_per_ms,
        cutstream_max_gap_beats=config.streams.cutstream_max_gap_beats,
        max_stream_interval_ms=config.streams.max_stream_interval_ms,
    )
    sliders = extract_slider_features(beatmap)
    spatial = extract_spatial_features(
        beatmap,
        grid_coord_step=config.visual.grid_snap_coord_step,
        stack_max_distance_px=config.visual.perfect_stacks_max_distance_px,
    )

    return BeatmapFeatures(
        circle_size=float(beatmap.difficulty.circle_size),
        approach_rate=float(beatmap.difficulty.approach_rate),
        timing=timing,
        snap=snap,
        objects=objects,
        streams=streams,
        sliders=sliders,
        spatial=spatial,
        storyboard_event_count=count_storyboard_events(beatmap),
        has_video=detect_video(beatmap),
        custom_sample_ratio=custom_sample_ratio(beatmap),
    )
