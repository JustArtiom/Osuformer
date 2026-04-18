from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    o = features.objects
    cfg = config.meta

    short_spinners = sum(1 for d in o.spinner_durations_ms if d < cfg.ninja_spinner_max_ms)
    ninja_score = soft_above(float(short_spinners), 2.0, width=1.0)

    keysounds_score = soft_above(features.custom_sample_ratio, cfg.keysounds_min_custom_sample_ratio, width=0.10)

    two_b_score = soft_above(
        float(o.simultaneous_object_count), float(cfg.two_b_min_overlap_count), width=8.0
    )

    storyboard_score = soft_above(
        float(features.storyboard_event_count), float(config.visual.storyboard_min_events), width=30.0
    )

    return [
        TagPrediction(tag="gimmick/ninja spinners", confidence=ninja_score),
        TagPrediction(tag="additions/keysounds", confidence=keysounds_score),
        TagPrediction(tag="gimmick/2B", confidence=two_b_score),
        TagPrediction(tag="gimmick/storyboard", confidence=storyboard_score),
    ]
