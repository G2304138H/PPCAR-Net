# Transferred from methods/parametric_methods/arc_length.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F

def parse_branch_length_window_sizes(raw_value: Any) -> tuple[int, ...]:
    """Return unique local arc-length windows in configured order."""
    if raw_value is None:
        raw_value = (4, 8, 16)
    if isinstance(raw_value, str):
        values = [
            part.strip()
            for part in raw_value.replace(";", ",").split(",")
            if part.strip()
        ]
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raise ValueError(
            "branch_length_local_window_sizes must be a comma-separated "
            "string or a list of integers >= 2."
        )
    windows: list[int] = []
    for value in values:
        window = int(value)
        if window < 2:
            raise ValueError(
                "branch_length_local_window_sizes must contain integers >= 2, "
                f"got {raw_value!r}."
            )
        if window not in windows:
            windows.append(window)
    if not windows:
        raise ValueError(
            "branch_length_local_window_sizes must contain at least one window."
        )
    return tuple(windows)

def _smooth_masked(
    values: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Smooth the final dimension without allowing invalid turns to contribute."""
    kernel = int(kernel_size)
    if kernel < 1:
        raise ValueError(
            "branch_length_curvature_smooth_kernel must be >= 1, "
            f"got {kernel_size}."
        )
    if kernel == 1 or int(values.shape[-1]) <= 1:
        return torch.where(valid, values, torch.zeros_like(values))
    if kernel % 2 == 0:
        kernel += 1
    padding = kernel // 2
    flat_values = torch.where(valid, values, torch.zeros_like(values)).reshape(
        -1, 1, int(values.shape[-1])
    )
    flat_valid = valid.to(dtype=values.dtype).reshape(
        -1, 1, int(values.shape[-1])
    )
    numerator = F.avg_pool1d(
        F.pad(flat_values, (padding, padding)),
        kernel_size=kernel,
        stride=1,
    )
    denominator = F.avg_pool1d(
        F.pad(flat_valid, (padding, padding)),
        kernel_size=kernel,
        stride=1,
    )
    return (numerator / denominator.clamp_min(1e-8)).reshape_as(values)

def decoded_branch_length_loss(
    predicted_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    global_component_weight: float = 1.0,
    local_deficit_component_weight: float = 1.0,
    smooth_excess_component_weight: float = 1.0,
    local_window_sizes: Any = (4, 8, 16),
    high_curvature_fraction: float = 0.15,
    low_curvature_fraction: float = 0.70,
    deficit_tolerance_ratio: float = 0.02,
    excess_tolerance_ratio: float = 0.02,
    smooth_l1_beta: float = 0.05,
    curvature_smooth_kernel: int = 5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Supervise global and curvature-localized decoded arc length.

    The global component is the absolute log ratio of complete valid branch
    lengths. GT high-curvature windows receive a one-sided squared penalty for
    predicted length deficits. GT smooth windows receive a one-sided SmoothL1
    penalty for predicted length excess. Region selection uses only detached GT
    geometry; every loss remains differentiable with respect to decoded points.
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
            "valid must match decoded point dimensions, got "
            f"{tuple(valid.shape)} and {tuple(predicted_xyz.shape[:-1])}."
        )
    if int(predicted_xyz.shape[-2]) < 2:
        raise ValueError("Branch-length loss requires at least two decoded points.")

    component_weights = {
        "branch_length_global_component_weight": float(global_component_weight),
        "branch_length_local_deficit_component_weight": float(
            local_deficit_component_weight
        ),
        "branch_length_smooth_excess_component_weight": float(
            smooth_excess_component_weight
        ),
    }
    for name, value in component_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}.")

    high_fraction = float(high_curvature_fraction)
    low_fraction = float(low_curvature_fraction)
    deficit_tolerance = float(deficit_tolerance_ratio)
    excess_tolerance = float(excess_tolerance_ratio)
    beta = float(smooth_l1_beta)
    epsilon = float(eps)
    if not 0.0 < high_fraction <= 1.0:
        raise ValueError(
            "branch_length_high_curvature_fraction must be in (0, 1]."
        )
    if not 0.0 < low_fraction <= 1.0:
        raise ValueError("branch_length_low_curvature_fraction must be in (0, 1].")
    if deficit_tolerance < 0.0 or not math.isfinite(deficit_tolerance):
        raise ValueError(
            "branch_length_deficit_tolerance_ratio must be finite and >= 0."
        )
    if excess_tolerance < 0.0 or not math.isfinite(excess_tolerance):
        raise ValueError(
            "branch_length_excess_tolerance_ratio must be finite and >= 0."
        )
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("branch_length_smooth_l1_beta must be finite and > 0.")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("branch_length_eps must be finite and > 0.")
    smooth_kernel = int(curvature_smooth_kernel)
    if smooth_kernel < 1:
        raise ValueError(
            "branch_length_curvature_smooth_kernel must be >= 1."
        )
    windows = parse_branch_length_window_sizes(local_window_sizes)

    point_valid = (
        valid.to(device=predicted_xyz.device, dtype=torch.bool)
        & torch.isfinite(predicted_xyz).all(dim=-1)
        & torch.isfinite(target_xyz).all(dim=-1)
    )
    segment_valid = point_valid[..., :-1] & point_valid[..., 1:]
    predicted_delta = predicted_xyz[..., 1:, :] - predicted_xyz[..., :-1, :]
    target_delta = target_xyz[..., 1:, :] - target_xyz[..., :-1, :]
    predicted_segment_length = torch.linalg.vector_norm(predicted_delta, dim=-1)
    target_segment_length = torch.linalg.vector_norm(target_delta, dim=-1)
    predicted_segment_length = torch.where(
        segment_valid,
        predicted_segment_length,
        torch.zeros_like(predicted_segment_length),
    )
    target_segment_length = torch.where(
        segment_valid,
        target_segment_length,
        torch.zeros_like(target_segment_length),
    )

    predicted_total_length = predicted_segment_length.sum(dim=-1)
    target_total_length = target_segment_length.sum(dim=-1)
    branch_valid = segment_valid.any(dim=-1) & (target_total_length > epsilon)
    zero = predicted_xyz.new_zeros(())
    if bool(branch_valid.any().item()):
        global_log_ratio = torch.log(
            (predicted_total_length + epsilon)
            / (target_total_length + epsilon)
        )
        global_loss = global_log_ratio[branch_valid].abs().mean()
        absolute_error = torch.abs(
            predicted_total_length[branch_valid]
            - target_total_length[branch_valid]
        )
        relative_error = absolute_error / target_total_length[
            branch_valid
        ].clamp_min(epsilon)
        arc_length_abs_error_mm = absolute_error.mean()
        arc_length_rel_error = relative_error.mean()
    else:
        global_loss = zero
        arc_length_abs_error_mm = zero
        arc_length_rel_error = zero

    target_direction = F.normalize(target_delta, p=2, dim=-1, eps=epsilon)
    target_turn = torch.linalg.vector_norm(
        target_direction[..., 1:, :] - target_direction[..., :-1, :],
        dim=-1,
    )
    turn_valid = segment_valid[..., :-1] & segment_valid[..., 1:]
    target_turn = _smooth_masked(
        target_turn,
        turn_valid,
        smooth_kernel,
    ).detach()

    deficit_terms: list[torch.Tensor] = []
    excess_terms: list[torch.Tensor] = []
    high_region_count = zero
    smooth_region_count = zero
    local_region_count = zero
    num_segments = int(predicted_segment_length.shape[-1])
    for window in windows:
        if window > num_segments:
            continue
        predicted_local_length = predicted_segment_length.unfold(
            -1, window, 1
        ).sum(dim=-1)
        target_local_length = target_segment_length.unfold(
            -1, window, 1
        ).sum(dim=-1)
        window_valid = segment_valid.unfold(-1, window, 1).all(dim=-1)
        window_turn = target_turn.unfold(-1, window - 1, 1).mean(dim=-1)

        for batch_index in range(int(predicted_xyz.shape[0])):
            for branch_index in range(int(predicted_xyz.shape[1])):
                valid_window = window_valid[batch_index, branch_index]
                target_length = target_local_length[batch_index, branch_index]
                predicted_length = predicted_local_length[batch_index, branch_index]
                score = window_turn[batch_index, branch_index]
                valid_window = (
                    valid_window
                    & torch.isfinite(score)
                    & torch.isfinite(target_length)
                    & torch.isfinite(predicted_length)
                    & (target_length > epsilon)
                )
                valid_count = int(valid_window.sum().item())
                if valid_count < 1:
                    continue

                valid_score = score[valid_window]
                high_mask = torch.zeros_like(valid_window)
                positive = valid_window & (score > epsilon)
                if bool(positive.any().item()):
                    high_count = min(
                        int(positive.sum().item()),
                        max(1, int(math.ceil(high_fraction * valid_count))),
                    )
                    high_indices = torch.topk(
                        score.masked_fill(~positive, float("-inf")),
                        k=high_count,
                        largest=True,
                        sorted=False,
                    ).indices
                    high_mask[high_indices] = True

                low_threshold = torch.quantile(valid_score, low_fraction)
                smooth_mask = (
                    valid_window
                    & (score <= low_threshold + epsilon)
                    & ~high_mask
                )
                relative_difference = (
                    predicted_length - target_length
                ) / target_length.clamp_min(epsilon)
                if bool(high_mask.any().item()):
                    deficit = F.relu(
                        -relative_difference - deficit_tolerance
                    )
                    deficit_terms.append(deficit[high_mask].square().mean())
                if bool(smooth_mask.any().item()):
                    excess = F.relu(relative_difference - excess_tolerance)
                    excess_terms.append(
                        F.smooth_l1_loss(
                            excess[smooth_mask],
                            torch.zeros_like(excess[smooth_mask]),
                            reduction="mean",
                            beta=beta,
                        )
                    )
                high_region_count = high_region_count + high_mask.to(
                    dtype=predicted_xyz.dtype
                ).sum()
                smooth_region_count = smooth_region_count + smooth_mask.to(
                    dtype=predicted_xyz.dtype
                ).sum()
                local_region_count = local_region_count + valid_window.to(
                    dtype=predicted_xyz.dtype
                ).sum()

    local_deficit_loss = (
        torch.stack(deficit_terms).mean() if deficit_terms else zero
    )
    smooth_excess_loss = (
        torch.stack(excess_terms).mean() if excess_terms else zero
    )
    total = (
        component_weights["branch_length_global_component_weight"] * global_loss
        + component_weights["branch_length_local_deficit_component_weight"]
        * local_deficit_loss
        + component_weights["branch_length_smooth_excess_component_weight"]
        * smooth_excess_loss
    )
    region_denominator = local_region_count.clamp_min(1.0)
    return {
        "branch_length_loss": total,
        "branch_length_global_loss": global_loss,
        "branch_length_local_deficit_loss": local_deficit_loss,
        "branch_length_smooth_excess_loss": smooth_excess_loss,
        "branch_length_high_region_fraction": high_region_count
        / region_denominator,
        "branch_length_smooth_region_fraction": smooth_region_count
        / region_denominator,
        "branch_length_valid_fraction": branch_valid.to(
            dtype=predicted_xyz.dtype
        ).mean(),
        "arc_length_abs_error_mm": arc_length_abs_error_mm,
        "arc_length_rel_error": arc_length_rel_error,
    }
