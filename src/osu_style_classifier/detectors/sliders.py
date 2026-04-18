from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above, soft_and, soft_below, soft_or
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    o = features.objects
    s = features.sliders
    t = features.timing
    cfg = config.sliders

    total = o.circles + o.sliders
    slider_ratio = o.sliders / total if total > 0 else 0.0
    circle_ratio = o.circles / total if total > 0 else 0.0

    slider_only_score = soft_above(slider_ratio, cfg.slider_only_min_ratio, width=0.08)
    circle_only_score = soft_above(circle_ratio, cfg.circle_only_min_ratio, width=0.08)

    slider_tech_score = soft_or(
        soft_above(s.avg_anchors, cfg.slider_tech_min_anchors_per_slider, width=2.0),
        soft_above(s.short_slider_ratio, cfg.slider_tech_min_short_slider_ratio, width=0.10),
    )

    slidershapes_score = soft_or(
        soft_above(s.avg_length_px, cfg.slidershapes_min_avg_length_px, width=40.0),
        soft_above(s.multi_curve_ratio, cfg.slidershapes_min_multi_curve_ratio, width=0.08),
    )

    high_sv_score = soft_and(
        soft_above(t.avg_sv_multiplier, cfg.high_sv_multiplier, width=0.15),
        soft_above(t.max_sv_multiplier, cfg.high_sv_multiplier, width=0.15),
    )
    low_sv_score = soft_and(
        soft_below(t.avg_sv_multiplier, cfg.low_sv_multiplier, width=0.15),
        soft_below(t.min_sv_multiplier, cfg.low_sv_multiplier, width=0.15),
    )

    sv_range = t.max_sv_multiplier - t.min_sv_multiplier
    complex_sv_score = soft_and(
        soft_above(t.sv_changes_per_minute, cfg.complex_sv_changes_per_minute, width=8.0),
        soft_above(sv_range, cfg.complex_sv_min_range, width=0.2),
    )

    return [
        TagPrediction(tag="gimmick/slider only", confidence=slider_only_score),
        TagPrediction(tag="gimmick/circle only", confidence=circle_only_score),
        TagPrediction(tag="tech/slider tech", confidence=slider_tech_score),
        TagPrediction(tag="sliders/complex slidershapes", confidence=slidershapes_score),
        TagPrediction(tag="sliders/high sv", confidence=high_sv_score),
        TagPrediction(tag="sliders/low sv", confidence=low_sv_score),
        TagPrediction(tag="sliders/complex sv", confidence=complex_sv_score),
    ]
