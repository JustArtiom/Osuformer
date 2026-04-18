from __future__ import annotations

from pathlib import Path

from src.osu.beatmap import Beatmap

from ..classifier import classify
from ..config import ClassifierConfig
from .api_labels import ApiLabelSource
from .dataset import BeatmapsetSample
from .metrics import EvalReport


def evaluate(
    samples: list[BeatmapsetSample],
    labels: ApiLabelSource,
    config: ClassifierConfig,
    verbose: bool = False,
    progress: bool = False,
) -> EvalReport:
    report = EvalReport(sample_count=0)
    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        set_labels = _safe_labels(labels, sample.set_id)
        for osu_path in sample.osu_files:
            report.sample_count += 1
            beatmap = _safe_parse(osu_path)
            if beatmap is None:
                continue
            if beatmap.general.mode.value != 0:
                continue
            bm_id = beatmap.metadata.beatmap_id
            actual = set(set_labels.get(bm_id, []))
            if not actual:
                continue
            report.labeled_count += 1
            result = classify(beatmap, config)
            predicted: set[str] = {str(t) for t in result.tags}
            report.update(predicted, actual)
            if verbose:
                print(f"{osu_path.name} bmid={bm_id}")
                print(f"  actual   : {sorted(actual)}")
                print(f"  predicted: {sorted(predicted)}")
        if progress and idx % 10 == 0:
            print(f"  ... {idx}/{total} sets, {report.labeled_count} labeled maps so far")
    return report


def _safe_parse(path: Path) -> Beatmap | None:
    try:
        return Beatmap(file_path=str(path))
    except Exception:
        return None


def _safe_labels(labels: ApiLabelSource, set_id: int) -> dict[int, list[str]]:
    try:
        return labels.labels_for_set(set_id)
    except Exception:
        return {}
