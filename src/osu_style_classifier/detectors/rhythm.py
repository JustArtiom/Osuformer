from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above, soft_and, soft_below
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    snap = features.snap
    t = features.timing
    o = features.objects
    cfg = config.rhythm

    simple_score = soft_and(
        soft_below(float(snap.diversity), float(cfg.simple_max_snap_diversity), width=0.7),
        soft_below(snap.entropy_bits, 1.2, width=0.3),
        soft_above(float(o.total), 40.0, width=15.0),
    )

    chaotic_score = soft_and(
        soft_above(float(snap.diversity), float(cfg.chaotic_min_snap_diversity), width=1.0),
        soft_above(snap.entropy_bits, cfg.chaotic_min_entropy_bits, width=0.4),
    )

    finger_control_score = soft_and(
        soft_above(float(snap.diversity), float(cfg.finger_control_min_snap_diversity), width=0.8),
        soft_above(o.object_density_per_second, cfg.finger_control_min_density, width=1.0),
    )

    variable_timing_score = soft_above(
        float(t.uninherited_count), float(cfg.variable_timing_min_uninherited), width=10.0
    )

    time_signatures_score = soft_above(
        t.non_4_4_ratio, cfg.time_signatures_min_non_4_4_ratio, width=0.08
    )

    return [
        TagPrediction(tag="expression/simple", confidence=simple_score),
        TagPrediction(tag="expression/chaotic", confidence=chaotic_score),
        TagPrediction(tag="tech/finger control", confidence=finger_control_score),
        TagPrediction(tag="meta/variable timing", confidence=variable_timing_score),
        TagPrediction(tag="meta/time signatures", confidence=time_signatures_score),
    ]
