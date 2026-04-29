from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.cache.metadata import MetadataRecord
from src.config.schemas.tokenizer import TokenizerConfig
from src.osu_tokenizer.descriptors import DESCRIPTOR_TAGS

SCALAR_FEATURE_COUNT: int = 8


@dataclass(frozen=True)
class ConditionSpec:
    cond_dim: int
    year_min: int
    year_max: int
    year_embed_dim: int
    descriptor_count: int
    descriptor_embed_dim: int
    mapper_count: int
    mapper_embed_dim: int
    hidden_dim: int


def default_condition_spec(tokenizer_cfg: TokenizerConfig, cond_dim: int, mapper_count: int = 1024) -> ConditionSpec:
    return ConditionSpec(
        cond_dim=cond_dim,
        year_min=tokenizer_cfg.year_min,
        year_max=tokenizer_cfg.year_max,
        year_embed_dim=64,
        descriptor_count=len(DESCRIPTOR_TAGS),
        descriptor_embed_dim=128,
        mapper_count=mapper_count,
        mapper_embed_dim=128,
        hidden_dim=cond_dim * 2,
    )


@dataclass
class ConditionFeatures:
    scalars: Tensor
    year_idx: Tensor
    descriptors: Tensor
    mapper_idx: Tensor


class ConditionEncoder(nn.Module):
    def __init__(self, spec: ConditionSpec):
        super().__init__()
        self.spec = spec
        year_count = spec.year_max - spec.year_min + 1
        self.year_embed = nn.Embedding(year_count + 1, spec.year_embed_dim)
        self.descriptor_proj = nn.Linear(spec.descriptor_count, spec.descriptor_embed_dim)
        self.mapper_embed = nn.Embedding(spec.mapper_count + 1, spec.mapper_embed_dim)
        in_dim = (
            SCALAR_FEATURE_COUNT
            + spec.year_embed_dim
            + spec.descriptor_embed_dim
            + spec.mapper_embed_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, spec.hidden_dim),
            nn.SiLU(),
            nn.Linear(spec.hidden_dim, spec.cond_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(spec.cond_dim))

    @property
    def cond_dim(self) -> int:
        return self.spec.cond_dim

    def forward(self, features: ConditionFeatures, null_mask: Tensor | None = None) -> Tensor:
        year_emb = self.year_embed(features.year_idx)
        desc_emb = self.descriptor_proj(features.descriptors)
        mapper_emb = self.mapper_embed(features.mapper_idx)
        combined = torch.cat([features.scalars, year_emb, desc_emb, mapper_emb], dim=-1)
        encoded = self.mlp(combined)
        if null_mask is not None:
            null = self.null_embedding.unsqueeze(0).expand_as(encoded)
            encoded = torch.where(null_mask.unsqueeze(-1), null, encoded)
        return encoded

    def null_vector(self, batch_size: int, device: torch.device) -> Tensor:
        return self.null_embedding.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()


def encode_condition_features(
    map_record: dict,
    metadata: MetadataRecord | None,
    tokenizer_cfg: TokenizerConfig,
    descriptor_count: int,
    mapper_idx: int = 0,
) -> ConditionFeatures:
    scalars = torch.tensor(
        [
            1.0 if map_record.get("hitsounded", False) else 0.0,
            float(map_record.get("circle_size", 5.0)) / 10.0,
            float(map_record.get("approach_rate", 5.0)) / 10.0,
            float(map_record.get("overall_difficulty", 5.0)) / 10.0,
            float(map_record.get("hp_drain_rate", 5.0)) / 10.0,
            float(map_record.get("slider_multiplier", 1.4)) / 4.0,
            float(map_record.get("duration_ms", 0.0)) / (10.0 * 60.0 * 1000.0),
            (metadata.star_rating if metadata is not None else 0.0) / 12.0,
        ],
        dtype=torch.float32,
    )
    if metadata is not None and tokenizer_cfg.year_min <= metadata.ranked_year <= tokenizer_cfg.year_max:
        year_idx = torch.tensor(metadata.ranked_year - tokenizer_cfg.year_min + 1, dtype=torch.long)
    else:
        year_idx = torch.tensor(0, dtype=torch.long)
    desc = torch.zeros(descriptor_count, dtype=torch.float32)
    if metadata is not None:
        for idx in metadata.descriptor_indices:
            if 0 <= idx < descriptor_count:
                desc[idx] = 1.0
    mapper_tensor = torch.tensor(mapper_idx, dtype=torch.long)
    return ConditionFeatures(
        scalars=scalars,
        year_idx=year_idx,
        descriptors=desc,
        mapper_idx=mapper_tensor,
    )
