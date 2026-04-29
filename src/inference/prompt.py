from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.cache.metadata import MetadataRecord
from src.config.schemas.tokenizer import TokenizerConfig
from src.model.conditioning import ConditionFeatures, encode_condition_features
from src.osu_tokenizer import DESCRIPTOR_TO_INDEX


@dataclass
class GenerationPrompt:
    star_rating: float | None = None
    descriptors: list[str] = field(default_factory=list)
    year: int | None = None
    hitsounded: bool = True
    cs: float = 4.0
    ar: float = 9.0
    od: float = 8.0
    hp: float = 6.0
    slider_multiplier: float = 1.4
    song_length_ms: float = 0.0


def build_condition_features(
    prompt: GenerationPrompt,
    tokenizer_cfg: TokenizerConfig,
    descriptor_count: int,
) -> ConditionFeatures:
    descriptor_indices: list[int] = []
    for tag in prompt.descriptors:
        idx = DESCRIPTOR_TO_INDEX.get(tag)
        if idx is not None:
            descriptor_indices.append(idx)
    metadata = MetadataRecord(
        beatmap_id=0,
        star_rating=float(prompt.star_rating) if prompt.star_rating is not None else 0.0,
        ranked_year=prompt.year if prompt.year is not None else tokenizer_cfg.year_min - 1,
        descriptor_indices=descriptor_indices,
    )
    map_record = {
        "hitsounded": prompt.hitsounded,
        "circle_size": prompt.cs,
        "approach_rate": prompt.ar,
        "overall_difficulty": prompt.od,
        "hp_drain_rate": prompt.hp,
        "slider_multiplier": prompt.slider_multiplier,
        "duration_ms": prompt.song_length_ms,
    }
    return encode_condition_features(
        map_record=map_record,
        metadata=metadata,
        tokenizer_cfg=tokenizer_cfg,
        descriptor_count=descriptor_count,
    )


def condition_features_to_device(features: ConditionFeatures, device: torch.device) -> ConditionFeatures:
    return ConditionFeatures(
        scalars=features.scalars.unsqueeze(0).to(device),
        year_idx=features.year_idx.unsqueeze(0).to(device),
        descriptors=features.descriptors.unsqueeze(0).to(device),
    )
