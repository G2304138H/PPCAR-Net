# Transferred from methods/parametric_methods/local_progress.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def decoded_local_progress_loss(
    predicted_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float = 0.0,
    min_target_step_fraction: float = 0.25,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Penalize decoded segments that fail to advance along the GT curve.

    Progress is the component of each predicted segment along the corresponding
    normalized GT segment, divided by a robust lower-bounded GT step length.
    A zero margin penalizes only backwards motion; a positive margin also
    penalizes decoded segments that make too little forward progress.
    """
    if (
        predicted_xyz.shape != target_xyz.shape
        or predicted_xyz.dim() != 4
        or predicted_xyz.shape[-1] != 3
    ):
        raise ValueError(
            "predicted_xyz and target_xyz must have matching [B,M,N,3] "
            f"shapes, got {tuple(predicted_xyz.shape)} and "
            f"{tuple(target_xyz.shape)}."
        )
    if valid.shape != predicted_xyz.shape[:-1]:
        raise ValueError(
            f"valid must have shape {tuple(predicted_xyz.shape[:-1])}, "
            f"got {tuple(valid.shape)}."
        )
    margin = float(margin)
    min_target_step_fraction = float(min_target_step_fraction)
    eps = float(eps)
    if not math.isfinite(margin):
        raise ValueError(
            f"decoded_local_progress_margin must be finite, got {margin}."
        )
    if (
        not math.isfinite(min_target_step_fraction)
        or min_target_step_fraction <= 0.0
    ):
        raise ValueError(
            "decoded_local_progress_min_target_step_fraction must be finite "
            f"and > 0, got {min_target_step_fraction}."
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            f"decoded_local_progress_eps must be finite and > 0, got {eps}."
        )

    zero = predicted_xyz.new_zeros(())
    if int(predicted_xyz.shape[-2]) < 2:
        return {
            "loss": zero,
            "valid_fraction": zero,
            "backward_fraction": zero,
            "insufficient_fraction": zero,
            "mean_ratio": zero,
            "p05_ratio": zero,
        }

    point_valid = (
        valid.to(device=predicted_xyz.device, dtype=torch.bool)
        & torch.isfinite(predicted_xyz).all(dim=-1)
        & torch.isfinite(target_xyz).all(dim=-1)
    )
    segment_valid = point_valid[..., :-1] & point_valid[..., 1:]
    predicted_segments = predicted_xyz[..., 1:, :] - predicted_xyz[..., :-1, :]
    target_segments = target_xyz[..., 1:, :] - target_xyz[..., :-1, :]
    predicted_segments = torch.where(
        segment_valid.unsqueeze(-1),
        predicted_segments,
        torch.zeros_like(predicted_segments),
    )
    target_segments = torch.where(
        segment_valid.unsqueeze(-1),
        target_segments,
        torch.zeros_like(target_segments),
    )
    target_step = torch.linalg.vector_norm(target_segments, dim=-1)
    active = segment_valid & (target_step > eps)
    if not bool(active.any().item()):
        return {
            "loss": zero,
            "valid_fraction": active.to(dtype=predicted_xyz.dtype).mean(),
            "backward_fraction": zero,
            "insufficient_fraction": zero,
            "mean_ratio": zero,
            "p05_ratio": zero,
        }

    active_float = active.to(dtype=predicted_xyz.dtype)
    active_count = active_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_target_step = (
        (target_step * active_float).sum(dim=-1, keepdim=True) / active_count
    )
    denominator = torch.maximum(
        target_step,
        min_target_step_fraction * mean_target_step,
    ).clamp_min(eps)
    target_tangent = F.normalize(target_segments, p=2, dim=-1, eps=eps)
    progress_ratio = (
        (predicted_segments * target_tangent).sum(dim=-1) / denominator
    )
    active_ratio = progress_ratio[active]
    shortfall = F.relu(margin - active_ratio)
    return {
        "loss": shortfall.square().mean(),
        "valid_fraction": active_float.mean(),
        "backward_fraction": (active_ratio < 0.0)
        .to(dtype=predicted_xyz.dtype)
        .mean(),
        "insufficient_fraction": (active_ratio < margin)
        .to(dtype=predicted_xyz.dtype)
        .mean(),
        "mean_ratio": active_ratio.mean(),
        "p05_ratio": torch.quantile(active_ratio, 0.05),
    }
