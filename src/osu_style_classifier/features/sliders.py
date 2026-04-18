from __future__ import annotations

from src.osu.beatmap import Beatmap
from src.osu.hit_object import Slider

from .feature_set import SliderFeatures


def extract_slider_features(beatmap: Beatmap) -> SliderFeatures:
    sliders = [o for o in beatmap.hit_objects if isinstance(o, Slider)]
    if not sliders:
        return SliderFeatures()

    lengths = [float(s.object_params.length) for s in sliders]
    anchors_per_slider = [
        sum(len(c.curve_points) for c in s.object_params.curves)
        for s in sliders
    ]
    multi_curve = sum(1 for s in sliders if len(s.object_params.curves) > 1)
    short_thresh_px = 40.0
    short_sliders = sum(1 for l in lengths if l < short_thresh_px)

    return SliderFeatures(
        avg_length_px=sum(lengths) / len(lengths),
        avg_anchors=sum(anchors_per_slider) / len(anchors_per_slider),
        multi_curve_ratio=multi_curve / len(sliders),
        short_slider_ratio=short_sliders / len(sliders),
        max_anchors=max(anchors_per_slider) if anchors_per_slider else 0,
    )
