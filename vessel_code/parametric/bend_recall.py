# Transferred from methods/parametric_methods/bend_recall.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from collections.abc import Iterable
import torch
import torch.nn.functional as F

def parse_bend_steps(raw_steps: Iterable[int] | str | int) -> tuple[int, ...]:
    """Return unique positive stencil strides in their configured order."""
    if isinstance(raw_steps, str):
        values = [
            value.strip()
            for value in raw_steps.replace(";", ",").split(",")
            if value.strip()
        ]
    elif isinstance(raw_steps, int):
        values = [raw_steps]
    else:
        values = list(raw_steps)
    steps: list[int] = []
    for value in values:
        step = int(value)
        if step < 1:
            raise ValueError(f"bend-recall steps must be positive, got {raw_steps!r}")
        if step not in steps:
            steps.append(step)
    if not steps:
        raise ValueError("bend-recall steps must contain at least one positive integer")
    return tuple(steps)

def _smooth(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel < 1:
        raise ValueError(f"bend-recall smooth kernel must be >= 1, got {kernel_size}")
    if kernel == 1 or int(values.numel()) <= 1:
        return values
    if kernel % 2 == 0:
        kernel += 1
    padded = F.pad(
        values.reshape(1, 1, -1),
        (kernel // 2, kernel // 2),
        mode="replicate",
    )
    return F.avg_pool1d(padded, kernel_size=kernel, stride=1).reshape_as(values)

def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    dilation = int(radius)
    if dilation < 0:
        raise ValueError(
            f"bend-recall high-region dilation must be >= 0, got {radius}"
        )
    if dilation == 0 or int(mask.numel()) <= 1:
        return mask
    return (
        F.max_pool1d(
            mask.to(dtype=torch.float32).reshape(1, 1, -1),
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        ).reshape_as(mask)
        > 0.5
    )

def asymmetric_bend_recall_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: torch.Tensor,
    *,
    steps: Iterable[int] | str | int = (1, 2, 4),
    high_fraction: float = 0.15,
    low_fraction: float = 0.70,
    min_recall_ratio: float = 0.80,
    jitter_tolerance_ratio: float = 0.10,
    direction_weight: float = 0.25,
    smooth_kernel: int = 5,
    high_region_dilation: int = 5,
    reference_floor: float = 0.05,
    min_target_step: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Preserve important GT bends while suppressing prediction-only jitter.

    The input is one ordered 2D or 3D centreline ``[N,D]``. High/low regions are
    selected only from smoothed GT turning per unit local length (a discrete
    curvature score) and are detached from autograd. Residuals use unsmoothed
    turn vectors so the prediction receives a direct gradient for missing bend
    magnitude/direction and excess turning.
    """
    if (
        pred_points.shape != target_points.shape
        or pred_points.dim() != 2
        or pred_points.shape[-1] not in {2, 3}
    ):
        raise ValueError(
            "pred_points and target_points must have matching [N,2] or [N,3] "
            f"shapes, got {tuple(pred_points.shape)} and {tuple(target_points.shape)}"
        )
    if valid.shape != pred_points.shape[:-1]:
        raise ValueError(
            f"valid must have shape {tuple(pred_points.shape[:-1])}, got {tuple(valid.shape)}"
        )

    high_fraction = float(high_fraction)
    low_fraction = float(low_fraction)
    min_recall_ratio = float(min_recall_ratio)
    jitter_tolerance_ratio = float(jitter_tolerance_ratio)
    direction_weight = float(direction_weight)
    reference_floor = float(reference_floor)
    min_target_step = float(min_target_step)
    if not 0.0 < high_fraction <= 1.0:
        raise ValueError("bend-recall high_fraction must be in (0, 1]")
    if not 0.0 < low_fraction <= 1.0:
        raise ValueError("bend-recall low_fraction must be in (0, 1]")
    if not 0.0 <= min_recall_ratio <= 1.0:
        raise ValueError("bend-recall min_recall_ratio must be in [0, 1]")
    if jitter_tolerance_ratio < 0.0 or not math.isfinite(jitter_tolerance_ratio):
        raise ValueError(
            "bend-recall jitter_tolerance_ratio must be finite and >= 0"
        )
    if direction_weight < 0.0 or not math.isfinite(direction_weight):
        raise ValueError("bend-recall direction_weight must be finite and >= 0")
    if reference_floor <= 0.0 or not math.isfinite(reference_floor):
        raise ValueError("bend-recall reference_floor must be finite and > 0")
    if min_target_step < 0.0 or not math.isfinite(min_target_step):
        raise ValueError("bend-recall min_target_step must be finite and >= 0")

    point_valid = (
        valid.to(device=pred_points.device, dtype=torch.bool)
        & torch.isfinite(pred_points).all(dim=-1)
        & torch.isfinite(target_points).all(dim=-1)
    )
    zero = pred_points.new_zeros(())
    underbend_terms: list[torch.Tensor] = []
    excess_terms: list[torch.Tensor] = []
    direction_terms: list[torch.Tensor] = []
    high_fractions: list[torch.Tensor] = []
    valid_fractions: list[torch.Tensor] = []
    point_count = int(pred_points.shape[0])

    for step in parse_bend_steps(steps):
        if point_count <= 2 * step:
            continue

        pred_left_chord = (
            pred_points[step:-step] - pred_points[: -2 * step]
        )
        pred_right_chord = (
            pred_points[2 * step :] - pred_points[step:-step]
        )
        target_left_chord = (
            target_points[step:-step] - target_points[: -2 * step]
        )
        target_right_chord = (
            target_points[2 * step :] - target_points[step:-step]
        )
        target_left_length = torch.linalg.vector_norm(
            target_left_chord, dim=-1
        )
        target_right_length = torch.linalg.vector_norm(
            target_right_chord, dim=-1
        )
        stencil_valid = (
            point_valid[: -2 * step]
            & point_valid[step:-step]
            & point_valid[2 * step :]
            & (target_left_length > min_target_step)
            & (target_right_length > min_target_step)
        )
        valid_fractions.append(stencil_valid.to(dtype=pred_points.dtype).mean())
        if not bool(stencil_valid.any().item()):
            continue

        pred_left = F.normalize(pred_left_chord, p=2, dim=-1, eps=eps)
        pred_right = F.normalize(pred_right_chord, p=2, dim=-1, eps=eps)
        target_left = F.normalize(target_left_chord, p=2, dim=-1, eps=eps)
        target_right = F.normalize(target_right_chord, p=2, dim=-1, eps=eps)
        pred_turn = pred_right - pred_left
        target_turn = target_right - target_left
        pred_magnitude = torch.linalg.vector_norm(pred_turn, dim=-1)
        target_magnitude = torch.linalg.vector_norm(target_turn, dim=-1)

        local_target_step = 0.5 * (
            target_left_length + target_right_length
        )
        target_curvature_score = target_magnitude / local_target_step.clamp_min(
            max(min_target_step, eps)
        )
        score_input = torch.where(
            stencil_valid,
            target_curvature_score,
            torch.zeros_like(target_curvature_score),
        )
        score = _smooth(score_input, int(smooth_kernel)).detach()
        valid_score = score[stencil_valid]
        reference = torch.quantile(
            target_magnitude[stencil_valid].detach(), 0.90
        ).clamp_min(reference_floor)

        high_mask = torch.zeros_like(stencil_valid)
        positive_valid = stencil_valid & (score > eps)
        if bool(positive_valid.any().item()):
            positive_indices = torch.nonzero(
                positive_valid, as_tuple=False
            ).flatten()
            high_count = min(
                int(positive_indices.numel()),
                max(
                    1,
                    int(
                        math.ceil(
                            high_fraction * float(positive_indices.numel())
                        )
                    ),
                ),
            )
            local_high = torch.topk(
                score[positive_indices],
                k=high_count,
                largest=True,
                sorted=False,
            ).indices
            high_mask[positive_indices[local_high]] = True
        high_region = _dilate(high_mask, int(high_region_dilation)) & stencil_valid
        high_fractions.append(
            high_region.to(dtype=pred_points.dtype).sum()
            / stencil_valid.to(dtype=pred_points.dtype).sum().clamp_min(1.0)
        )

        low_threshold = torch.quantile(valid_score, low_fraction)
        low_mask = (
            stencil_valid
            & (score <= low_threshold + eps)
            & ~high_region
        )

        if bool(high_region.any().item()):
            target_unit = target_turn / target_magnitude.clamp_min(eps).unsqueeze(-1)
            aligned_pred_turn = torch.sum(pred_turn * target_unit, dim=-1)
            normalized_shortfall = F.relu(
                min_recall_ratio * target_magnitude - aligned_pred_turn
            ) / reference
            underbend_terms.append(
                normalized_shortfall[high_region].square().mean()
            )

            direction_valid = (
                high_region
                & (target_magnitude > eps)
                & (pred_magnitude > 0.05 * reference)
            )
            if bool(direction_valid.any().item()):
                direction_cosine = F.cosine_similarity(
                    pred_turn,
                    target_turn,
                    dim=-1,
                    eps=eps,
                ).clamp(-1.0, 1.0)
                direction_terms.append(
                    (1.0 - direction_cosine[direction_valid]).mean()
                )

        if bool(low_mask.any().item()):
            tolerance = jitter_tolerance_ratio * reference
            normalized_excess = F.relu(
                pred_magnitude - target_magnitude - tolerance
            ) / reference
            excess_terms.append(normalized_excess[low_mask].square().mean())

    underbend = torch.stack(underbend_terms).mean() if underbend_terms else zero
    excess = torch.stack(excess_terms).mean() if excess_terms else zero
    direction = torch.stack(direction_terms).mean() if direction_terms else zero
    return {
        "loss": underbend + excess + direction_weight * direction,
        "underbend_loss": underbend,
        "excess_loss": excess,
        "direction_loss": direction,
        "high_region_fraction": (
            torch.stack(high_fractions).mean() if high_fractions else zero
        ),
        "valid_fraction": (
            torch.stack(valid_fractions).mean() if valid_fractions else zero
        ),
    }
