from __future__ import annotations

from src.osu.beatmap import Beatmap

from .config import ClassifierConfig
from .detectors import aim, meta, rhythm, sliders, streams, visual
from .features import BeatmapFeatures, extract_features
from .types import StyleResult, TagPrediction


_DETECTORS = (
    streams.detect,
    aim.detect,
    sliders.detect,
    rhythm.detect,
    visual.detect,
    meta.detect,
)


def classify(beatmap: Beatmap, config: ClassifierConfig | None = None) -> StyleResult:
    cfg = config or ClassifierConfig()
    features = extract_features(beatmap, cfg)
    return classify_from_features(features, cfg)


def classify_from_features(features: BeatmapFeatures, config: ClassifierConfig | None = None) -> StyleResult:
    cfg = config or ClassifierConfig()
    preds: list[TagPrediction] = []
    for detector in _DETECTORS:
        preds.extend(detector(features, cfg))
    preds = _dedupe_max_confidence(preds)
    result = StyleResult(predictions=preds)
    return result.filter(cfg.min_confidence)


def _dedupe_max_confidence(preds: list[TagPrediction]) -> list[TagPrediction]:
    best: dict[str, TagPrediction] = {}
    for p in preds:
        prev = best.get(p.tag)
        if prev is None or p.confidence > prev.confidence:
            best[p.tag] = p
    return sorted(best.values(), key=lambda p: -p.confidence)
