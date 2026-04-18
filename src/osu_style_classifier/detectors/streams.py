from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above, soft_and
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    s = features.streams
    o = features.objects
    cfg = config.streams
    bpm = features.timing.primary_bpm

    streams_score = soft_and(
        soft_above(bpm, cfg.min_bpm, width=15.0),
        soft_above(s.stream_object_ratio, 0.20, width=0.08),
        soft_above(float(s.max_run_length), float(cfg.min_run_length), width=5.0),
    )

    bursts_score = soft_and(
        soft_above(float(s.burst_count), float(cfg.burst_count_threshold), width=8.0),
        soft_above(bpm, cfg.burst_min_bpm, width=15.0),
    )

    spaced_score = soft_above(float(s.spaced_stream_count), 0.5, width=1.5)

    cutstream_score = soft_above(float(s.cutstream_count), float(cfg.cutstream_count_threshold), width=10.0)

    stamina_score = soft_and(
        soft_above(o.drain_time_seconds, config.aim.stamina_min_drain_seconds, width=40.0),
        soft_above(s.stream_object_ratio, cfg.stamina_min_stream_ratio, width=0.10),
    )

    return [
        TagPrediction(tag="skillset/streams", confidence=streams_score),
        TagPrediction(tag="streams/bursts", confidence=bursts_score),
        TagPrediction(tag="streams/spaced streams", confidence=spaced_score),
        TagPrediction(tag="streams/cutstreams", confidence=cutstream_score),
        TagPrediction(tag="streams/stamina", confidence=stamina_score),
    ]
