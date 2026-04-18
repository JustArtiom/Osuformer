from __future__ import annotations

from ..config import ClassifierConfig
from ..features import BeatmapFeatures
from ..scoring import soft_above, soft_and, soft_below
from ..types import TagPrediction


def detect(features: BeatmapFeatures, config: ClassifierConfig) -> list[TagPrediction]:
    sp = features.spatial
    o = features.objects
    cfg = config.visual

    grid_score = soft_and(
        soft_above(sp.grid_snap_ratio, cfg.grid_snap_min_ratio, width=0.03),
        soft_above(float(o.total), 40.0, width=15.0),
    )

    constraint_score = soft_and(
        soft_below(sp.coverage_ratio, cfg.playfield_constraint_max_coverage, width=0.08),
        soft_above(float(o.total), 40.0, width=15.0),
    )

    usage_score = soft_above(sp.coverage_ratio, cfg.playfield_usage_min_coverage, width=0.02)

    symmetry_best = max(sp.symmetry_x_score, sp.symmetry_y_score)
    symmetry_score = soft_above(symmetry_best, cfg.symmetry_min_axis_score, width=0.08)

    perfect_stacks_score = soft_above(
        float(sp.perfect_stack_count), float(cfg.perfect_stacks_min_count), width=40.0
    )

    visually_dense_score = soft_above(
        o.peak_density_per_second, cfg.visually_dense_min_concurrent, width=2.0
    )

    overlap_score = soft_above(sp.overlap_ratio, cfg.overlap_min_ratio, width=0.08)

    return [
        TagPrediction(tag="style/grid snap", confidence=grid_score),
        TagPrediction(tag="gimmick/playfield constraint", confidence=constraint_score),
        TagPrediction(tag="expression/playfield usage", confidence=usage_score),
        TagPrediction(tag="style/symmetrical", confidence=symmetry_score),
        TagPrediction(tag="reading/perfect stacks", confidence=perfect_stacks_score),
        TagPrediction(tag="reading/visually dense", confidence=visually_dense_score),
        TagPrediction(tag="reading/overlaps", confidence=overlap_score),
    ]
