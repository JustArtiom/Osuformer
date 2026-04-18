from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above, soft_and, soft_below, soft_between
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    o = features.objects
    a = config.aim
    snap = features.snap

    jumps_score = soft_above(o.avg_distance_px, a.jump_min_avg_distance_px, width=30.0)

    wide_score = soft_and(
        soft_above(o.avg_distance_px, 150.0, width=25.0),
        soft_above(o.avg_flow_angle_deg, a.wide_min_avg_angle_deg, width=12.0),
    )

    linear_score = soft_and(
        soft_below(o.std_flow_angle_deg, a.linear_max_angle_std_deg, width=8.0),
        soft_below(o.avg_flow_angle_deg, a.linear_max_avg_angle_deg, width=12.0),
        soft_above(float(len(o.flow_angles_deg)), 50.0, width=20.0),
    )

    sharp_score = soft_and(
        soft_above(o.avg_flow_angle_deg, a.sharp_min_avg_angle_deg, width=10.0),
        soft_below(o.std_flow_angle_deg, a.sharp_max_angle_std_deg, width=10.0),
    )

    aim_control_score = soft_and(
        soft_above(o.std_flow_angle_deg, a.aim_control_min_angle_std_deg, width=7.0),
        soft_above(o.avg_distance_px, a.aim_control_min_avg_distance_px, width=25.0),
        soft_above(float(o.total), float(a.aim_control_min_total_objects), width=80.0),
    )

    flow_score = soft_and(
        soft_between(o.avg_flow_angle_deg, a.flow_min_avg_angle_deg, a.flow_max_avg_angle_deg, width=12.0),
        soft_below(o.std_flow_angle_deg, a.flow_max_angle_std_deg, width=10.0),
    )

    precision_score = soft_and(
        soft_above(features.circle_size, a.precision_min_cs, width=0.3),
        soft_below(o.avg_distance_px, a.precision_max_spacing_px, width=20.0),
    )

    jumps_stamina_score = soft_and(
        soft_above(float(o.total), float(a.stamina_min_objects), width=120.0),
        soft_above(o.drain_time_seconds, a.stamina_min_drain_seconds, width=40.0),
        soft_above(o.avg_distance_px, a.stamina_min_avg_distance_px, width=25.0),
    )

    half_beat_ratio = snap.divisor_counts.get(2, 0) / snap.total_intervals if snap.total_intervals > 0 else 0.0
    alt_score = soft_and(
        soft_above(features.timing.primary_bpm, a.alt_min_bpm, width=10.0),
        soft_above(o.object_density_per_second, a.alt_min_density, width=1.0),
        soft_between(o.avg_distance_px, a.alt_min_avg_distance_px, a.alt_max_avg_distance_px, width=25.0),
        soft_above(half_beat_ratio, a.alt_min_half_beat_ratio, width=0.10),
    )

    return [
        TagPrediction(tag="skillset/jumps", confidence=jumps_score),
        TagPrediction(tag="jumps/wide", confidence=wide_score),
        TagPrediction(tag="jumps/linear", confidence=linear_score),
        TagPrediction(tag="jumps/sharp", confidence=sharp_score),
        TagPrediction(tag="tech/aim control", confidence=aim_control_score),
        TagPrediction(tag="streams/flow aim", confidence=flow_score),
        TagPrediction(tag="skillset/precision", confidence=precision_score),
        TagPrediction(tag="jumps/stamina", confidence=jumps_stamina_score),
        TagPrediction(tag="skillset/alt", confidence=alt_score),
    ]
