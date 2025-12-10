from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType


_PENALTY_KEYS = (
    "object_tick_weight",
    "duplicate_tick_weight",
    "circle_in_slider_weight",
    "coord_distance_weight",
    "coord_angle_weight",
    "slider_path_weight",
    "density_weight",
    "type_mismatch_weight",
    "slider_duration_weight",
    "curve_type_weight",
    "slider_sv_weight",
)


def normalize_penalty_config(cfg: Dict[str, float] | None) -> Dict[str, float]:
    base = {key: 0.0 for key in _PENALTY_KEYS}
    if not cfg:
        return base
    is_dict = isinstance(cfg, dict)
    for key in base:
        value = cfg.get(key, 0.0) if is_dict else 0.0
        try:
            base[key] = float(value or 0.0)
        except (TypeError, ValueError):
            base[key] = 0.0
    # Backward compatibility for older configs that used circle_tick_weight
    if is_dict and "circle_tick_weight" in cfg and base["object_tick_weight"] == 0.0:
        try:
            base["object_tick_weight"] = float(cfg.get("circle_tick_weight") or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            pass
    return base


def _id_lookup(size: int, *, offset: float = -1.0, clamp_min: float = 0.0, scale: float = 1.0, device=None, dtype=None):
    values = torch.arange(size, device=device, dtype=dtype)
    if offset:
        values = values + offset
    if clamp_min is not None:
        values = torch.clamp(values, min=clamp_min)
    if scale != 1.0:
        values = values * scale
    return values


def _collect_tick_targets(tokens: torch.Tensor, mask: torch.Tensor) -> List[List[float]]:
    per_sample: List[List[float]] = [[] for _ in range(tokens.size(0))]
    positions = mask.nonzero(as_tuple=False)
    for sample_idx, pos_idx in positions.tolist():
        tick_id = tokens[sample_idx, pos_idx, TokenAttr.TICK].item()
        per_sample[sample_idx].append(float(max(0, tick_id - 1)))
    return per_sample


def _slider_coverage(tokens: torch.Tensor, mask: torch.Tensor, tick_bins: int, dtype, device) -> torch.Tensor:
    coverage = torch.zeros(tokens.size(0), tick_bins, device=device, dtype=dtype)
    positions = mask.nonzero(as_tuple=False)
    for sample_idx, pos_idx in positions.tolist():
        tick_id = tokens[sample_idx, pos_idx, TokenAttr.TICK].item()
        duration_id = tokens[sample_idx, pos_idx, TokenAttr.DURATION].item()
        start = max(0, tick_id - 1)
        duration = max(0, duration_id - 1)
        end = min(tick_bins - 1, start + duration)
        if end >= start:
            coverage[sample_idx, start : end + 1] = 1.0
    return coverage


def _angle_from_components(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return angles in degrees within [0, 360)."""
    angle = torch.rad2deg(torch.atan2(y, x))
    return torch.remainder(angle + 360.0, 360.0)


def _angle_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest positive difference in degrees."""
    delta = torch.abs(a - b)
    delta = torch.remainder(delta, 360.0)
    return torch.where(delta > 180.0, 360.0 - delta, delta)


def compute_penalty_loss(
    attr_logits: Sequence[torch.Tensor],
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    tokenizer: HitObjectTokenizer,
    penalty_cfg: Dict[str, float] | None,
) -> torch.Tensor:
    cfg = normalize_penalty_config(penalty_cfg)
    if all(weight <= 0.0 for weight in cfg.values()):
        return torch.zeros(1, device=tokens.device, dtype=attr_logits[0].dtype)

    device = tokens.device
    dtype = attr_logits[0].dtype
    zero = torch.zeros(1, device=device, dtype=dtype)

    type_logits = attr_logits[TokenAttr.TYPE]
    tick_logits = attr_logits[TokenAttr.TICK]
    type_probs = F.softmax(type_logits, dim=-1)
    tick_probs = F.softmax(tick_logits, dim=-1)

    tick_bins = tick_probs.size(-1)
    tick_values = _id_lookup(tick_bins, device=device, dtype=dtype)

    mask_float = valid_mask.float()
    type_targets = tokens[..., TokenAttr.TYPE]
    circle_mask = (type_targets == TokenType.CIRCLE) & valid_mask
    slider_mask = (type_targets == TokenType.SLIDER) & valid_mask
    event_mask = ((type_targets == TokenType.CIRCLE) | (type_targets == TokenType.SLIDER)) & valid_mask

    circle_prob = type_probs[..., TokenType.CIRCLE]
    slider_prob = type_probs[..., TokenType.SLIDER]
    event_prob = (circle_prob + slider_prob) * mask_float

    bin_size = float(tokenizer.bin_size)
    seq_len_max = max(1.0, float(tokenizer.seq_len))
    diag = math.hypot(float(tokenizer.osu_width), float(tokenizer.osu_height))
    diag = max(diag, 1.0)
    diag_sq = diag * diag

    needs_coord_logits = (
        cfg["coord_distance_weight"] > 0.0
        or cfg["coord_angle_weight"] > 0.0
        or cfg["slider_path_weight"] > 0.0
    )
    if needs_coord_logits:
        x_logits = attr_logits[TokenAttr.X]
        y_logits = attr_logits[TokenAttr.Y]
        x_probs = F.softmax(x_logits, dim=-1)
        y_probs = F.softmax(y_logits, dim=-1)
        x_values = _id_lookup(x_probs.size(-1), device=device, dtype=dtype) * bin_size
        y_values = _id_lookup(y_probs.size(-1), device=device, dtype=dtype) * bin_size
        pred_coord_x = (x_probs * x_values).sum(dim=-1)
        pred_coord_y = (y_probs * y_values).sum(dim=-1)
    else:
        pred_coord_x = pred_coord_y = None

    total_penalty = zero.clone()

    # Penalty 1: start tick delta for any circle or slider
    if cfg["object_tick_weight"] > 0.0:
        event_tick_targets = _collect_tick_targets(tokens, event_mask)
        pred_tick_expect = (tick_probs * tick_values).sum(dim=-1)
        event_weights = event_prob
        weight_sum = torch.zeros(1, device=device, dtype=dtype)
        penalty_sum = torch.zeros(1, device=device, dtype=dtype)
        for sample_idx in range(tokens.size(0)):
            gt_ticks = event_tick_targets[sample_idx]
            if not gt_ticks:
                continue
            gt_tensor = pred_tick_expect.new_tensor(gt_ticks)
            sample_ticks = pred_tick_expect[sample_idx]
            sample_weights = event_weights[sample_idx]
            for pos in range(sample_ticks.size(0)):
                weight = sample_weights[pos]
                if weight.item() <= 0:
                    continue
                diffs = torch.abs(sample_ticks[pos] - gt_tensor)
                min_diff = diffs.min()
                penalty_sum = penalty_sum + weight * (min_diff ** 2)
                weight_sum = weight_sum + weight
        if weight_sum.item() > 0:
            normalized = (penalty_sum / weight_sum.clamp_min(1.0)) / (seq_len_max * seq_len_max)
            total_penalty = total_penalty + cfg["object_tick_weight"] * normalized

    # Penalty 2: duplicate ticks (two events sharing same tick)
    if cfg["duplicate_tick_weight"] > 0.0 and tokens.size(1) > 1:
        pred_tick_expect = (tick_probs * tick_values).sum(dim=-1)
        diff = pred_tick_expect.unsqueeze(2) - pred_tick_expect.unsqueeze(1)
        abs_diff = torch.abs(diff)
        closeness = F.relu(1.0 - abs_diff)
        weight_matrix = event_prob.unsqueeze(2) * event_prob.unsqueeze(1)
        upper_mask = torch.triu(torch.ones(tokens.size(1), tokens.size(1), device=device, dtype=dtype), diagonal=1)
        closeness = closeness * upper_mask
        weight_matrix = weight_matrix * upper_mask
        denom = weight_matrix.sum().clamp_min(1.0)
        duplicate_penalty = (closeness * weight_matrix).sum() / denom
        total_penalty = total_penalty + cfg["duplicate_tick_weight"] * duplicate_penalty

    # Penalty 3: circles placed during slider coverage
    if cfg["circle_in_slider_weight"] > 0.0:
        coverage = _slider_coverage(tokens, slider_mask, tick_bins, dtype, device)
        if coverage.max().item() > 0:
            slider_mask_map = coverage.unsqueeze(1)
            overlap_prob = (tick_probs * slider_mask_map).sum(dim=-1)
            circle_weights = circle_prob * mask_float
            denom = circle_weights.sum().clamp_min(1.0)
            overlap_penalty = (circle_weights * overlap_prob).sum() / denom
            total_penalty = total_penalty + cfg["circle_in_slider_weight"] * overlap_penalty

    # Coordinate penalties (distance + angle)
    coord_mask = (
        ((type_targets == TokenType.CIRCLE) | (type_targets == TokenType.SLIDER))
        & valid_mask
        & (tokens[..., TokenAttr.X] > 0)
        & (tokens[..., TokenAttr.Y] > 0)
    )
    coord_mask_float = coord_mask.float()
    if (cfg["coord_distance_weight"] > 0.0 or cfg["coord_angle_weight"] > 0.0) and coord_mask.any():
        pred_x = pred_coord_x
        pred_y = pred_coord_y
        tgt_x = torch.clamp(tokens[..., TokenAttr.X].float() - 1.0, min=0.0) * bin_size
        tgt_y = torch.clamp(tokens[..., TokenAttr.Y].float() - 1.0, min=0.0) * bin_size
        center_x = float(tokenizer.osu_width) / 2.0
        center_y = float(tokenizer.osu_height) / 2.0
        pred_dx = pred_x - center_x
        pred_dy = pred_y - center_y
        tgt_dx = tgt_x - center_x
        tgt_dy = tgt_y - center_y
        pred_dist = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + 1e-6)
        tgt_dist = torch.sqrt(tgt_dx ** 2 + tgt_dy ** 2 + 1e-6)
        if cfg["coord_distance_weight"] > 0.0:
            dist_delta = (pred_dist - tgt_dist).pow(2) * coord_mask_float
            denom = coord_mask_float.sum().clamp_min(1.0)
            normalized = (dist_delta.sum() / denom) / diag_sq
            total_penalty = total_penalty + cfg["coord_distance_weight"] * normalized
        if cfg["coord_angle_weight"] > 0.0:
            pred_angle = _angle_from_components(pred_dx, pred_dy)
            tgt_angle = _angle_from_components(tgt_dx, tgt_dy)
            angle_delta = _angle_distance_deg(pred_angle, tgt_angle).pow(2) * coord_mask_float
            denom = coord_mask_float.sum().clamp_min(1.0)
            normalized = (angle_delta.sum() / denom) / (180.0 * 180.0)
            total_penalty = total_penalty + cfg["coord_angle_weight"] * normalized

    # Slider path anchor penalty
    if cfg["slider_path_weight"] > 0.0 and needs_coord_logits:
        path_mask = (
            (type_targets == TokenType.SLIDER_PATH)
            & valid_mask
            & (tokens[..., TokenAttr.X] > 0)
            & (tokens[..., TokenAttr.Y] > 0)
        )
        if path_mask.any():
            path_mask_float = path_mask.float()
            tgt_x = torch.clamp(tokens[..., TokenAttr.X].float() - 1.0, min=0.0) * bin_size
            tgt_y = torch.clamp(tokens[..., TokenAttr.Y].float() - 1.0, min=0.0) * bin_size
            dist_sq = ((pred_coord_x - tgt_x) ** 2 + (pred_coord_y - tgt_y) ** 2) * path_mask_float
            penalty = dist_sq.sum() / path_mask_float.sum().clamp_min(1.0)
            penalty = penalty / diag_sq
            total_penalty = total_penalty + cfg["slider_path_weight"] * penalty

    # Density penalty (overall hit count alignment)
    if cfg["density_weight"] > 0.0:
        gt_counts = event_mask.float().sum(dim=1)
        pred_counts = event_prob.sum(dim=1)
        diff = torch.abs(pred_counts - gt_counts)
        normalized = diff / seq_len_max
        density_penalty = normalized.mean()
        total_penalty = total_penalty + cfg["density_weight"] * density_penalty

    # Type mismatch penalty (circle vs slider)
    if cfg["type_mismatch_weight"] > 0.0:
        cs_mask = ((type_targets == TokenType.CIRCLE) | (type_targets == TokenType.SLIDER)) & valid_mask
        if cs_mask.any():
            cs_mask_float = cs_mask.float()
            type_target_prob = type_probs.gather(-1, type_targets.unsqueeze(-1)).squeeze(-1)
            mismatch_penalty = ((1.0 - type_target_prob) * cs_mask_float).sum() / cs_mask_float.sum().clamp_min(1.0)
            total_penalty = total_penalty + cfg["type_mismatch_weight"] * mismatch_penalty

    # Slider duration penalty
    if cfg["slider_duration_weight"] > 0.0 and slider_mask.any():
        duration_logits = attr_logits[TokenAttr.DURATION]
        duration_probs = F.softmax(duration_logits, dim=-1)
        duration_values = _id_lookup(duration_probs.size(-1), device=device, dtype=dtype)
        pred_duration = (duration_probs * duration_values).sum(dim=-1)
        pred_duration = torch.clamp(pred_duration, min=0.0)
        tgt_duration = torch.clamp(tokens[..., TokenAttr.DURATION].float() - 1.0, min=0.0)
        slider_mask_float = slider_mask.float()
        duration_delta = (pred_duration - tgt_duration).pow(2) * slider_mask_float
        denom = slider_mask_float.sum().clamp_min(1.0)
        max_duration = max(1.0, float(tokenizer.max_duration_ticks))
        normalized = (duration_delta.sum() / denom) / (max_duration * max_duration)
        total_penalty = total_penalty + cfg["slider_duration_weight"] * normalized

    # Curve type penalty (classification style)
    if cfg["curve_type_weight"] > 0.0 and slider_mask.any():
        curve_logits = attr_logits[TokenAttr.CURVE_TYPE]
        curve_probs = F.softmax(curve_logits, dim=-1)
        curve_targets = tokens[..., TokenAttr.CURVE_TYPE]
        curve_mask = slider_mask & (curve_targets > 0)
        if curve_mask.any():
            curve_mask_float = curve_mask.float()
            curve_target_prob = curve_probs.gather(-1, curve_targets.unsqueeze(-1)).squeeze(-1)
            curve_penalty = ((1.0 - curve_target_prob) * curve_mask_float).sum() / curve_mask_float.sum().clamp_min(1.0)
            total_penalty = total_penalty + cfg["curve_type_weight"] * curve_penalty

    # Slider SV penalty (duration-style)
    if cfg["slider_sv_weight"] > 0.0 and slider_mask.any():
        sv_logits = attr_logits[TokenAttr.SLIDER_SV]
        sv_probs = F.softmax(sv_logits, dim=-1)
        sv_idx = torch.arange(sv_probs.size(-1), device=device, dtype=dtype)
        sv_values = torch.clamp(sv_idx / float(tokenizer.sv_precision), min=0.0, max=float(tokenizer.max_sv))
        pred_sv = (sv_probs * sv_values).sum(dim=-1)
        tgt_sv = torch.clamp(
            tokens[..., TokenAttr.SLIDER_SV].float() / float(tokenizer.sv_precision),
            min=0.0,
            max=float(tokenizer.max_sv),
        )
        sv_mask = slider_mask & (tokens[..., TokenAttr.SLIDER_SV] > 0)
        slider_mask_float = sv_mask.float()
        denom = slider_mask_float.sum().clamp_min(1.0)
        sv_penalty = ((pred_sv - tgt_sv).pow(2) * slider_mask_float).sum() / denom
        max_sv = max(1e-6, float(tokenizer.max_sv))
        sv_penalty = sv_penalty / (max_sv * max_sv)
        total_penalty = total_penalty + cfg["slider_sv_weight"] * sv_penalty

    return total_penalty
