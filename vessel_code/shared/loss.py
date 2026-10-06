# Transferred from methods/src/loss.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector

VESSEL_LOSS_KEYS = {
    "existence_loss_weight",
    "point_loss_weight",
    "xyz_loss_weight",
    "xyz_loss_type",
    "xyz_loss_p",
    "geometry_loss_coordinate_space",
    "xyz_scale_mm",
    "radius_scale_mm",
    "exclude_fixed_main_from_existence_loss",
    "radius_loss_weight",
    "tangent_loss_weight",
    "curve_loss_weight",
    "branch_length_loss_weight",
    "endpoint_vector_loss_weight",
    "chamfer_loss_weight",
    "pred_to_gt_curve_loss_weight",
    "gt_to_pred_curve_loss_weight",
    "detail_loss_weight",
    "relative_first_difference_loss_weight",
    "relative_second_difference_loss_weight",
    "relative_second_difference_curvature_alpha",
    "relative_second_difference_curvature_max",
    "relative_second_difference_curvature_smooth_kernel",
    "local_progress_loss_weight",
    "local_progress_margin",
    "local_progress_min_target_step_fraction",
    "arc_resample_loss_weight",
    "multiscale_topk_curve_loss_weight",
    "multiscale_topk_curve_steps",
    "multiscale_topk_curve_fraction",
    "curvature_weighted_xyz_loss_weight",
    "curvature_weighted_tangent_loss_weight",
    "curvature_weighted_curve_loss_weight",
    "curvature_weight_alpha",
    "curvature_weight_max",
    "curvature_weight_smooth_kernel",
    "soft_mask_floor",
    "attachment_loss_weight",
    "attachment_softmin_temp",
    "occupancy_3d_ssim_loss_weight",
    "occupancy_3d_voxel_resolution",
    "occupancy_3d_sigma",
    "occupancy_3d_ssim_window_size",
    "occupancy_3d_chunk_voxels",
}

def _parse_asymmetric_curvature_steps(raw_value: Any) -> tuple[int, ...]:
    if raw_value is None:
        raw_value = "1,2,4,8"
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raise ValueError(
            "asymmetric_curvature_steps must be a comma-separated string or a list of positive integers."
        )
    steps: list[int] = []
    for value in values:
        step = int(value)
        if step < 1:
            raise ValueError(f"asymmetric_curvature_steps must be positive, got {raw_value!r}.")
        if step not in steps:
            steps.append(step)
    if not steps:
        raise ValueError("asymmetric_curvature_steps must contain at least one positive integer.")
    return tuple(steps)

_EXPENSIVE_LOSS_WEIGHT_DEFAULTS = {
    "tangent_loss_weight": 0.1,
    "curve_loss_weight": 0.1,
    "branch_length_loss_weight": 0.1,
    "endpoint_vector_loss_weight": 0.0,
    "chamfer_loss_weight": 0.1,
    "pred_to_gt_curve_loss_weight": 0.0,
    "gt_to_pred_curve_loss_weight": 0.0,
    "detail_loss_weight": 0.0,
    "relative_first_difference_loss_weight": 0.0,
    "relative_second_difference_loss_weight": 0.0,
    "local_progress_loss_weight": 0.0,
    "arc_resample_loss_weight": 0.0,
    "multiscale_topk_curve_loss_weight": 0.0,
    "curvature_weighted_xyz_loss_weight": 0.0,
    "curvature_weighted_tangent_loss_weight": 0.0,
    "curvature_weighted_curve_loss_weight": 0.0,
    "attachment_loss_weight": 0.0,
    "occupancy_3d_ssim_loss_weight": 0.0,
}

def _dice_loss_from_probs(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return 1.0 - dice.mean()

def _projected_centerline_derivative_metrics(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    *,
    curvature_steps: Any = (1, 2, 4),
    min_step_px: float = 0.5,
    curvature_tolerance_inv_px: float = 0.02,
    curvature_high_fraction: float = 1.0,
    curvature_high_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compare ordered projected curves using 2D tangents and signed curvature.

    Tangents use normalized chords, while curvature uses the signed Menger
    curvature of each three-point stencil. The latter is independent of point
    parameterization within a local circular arc and has units of inverse pixels.
    Multiple index strides make the comparison less sensitive to pixel-scale
    target noise. Curvature residuals in the configured highest-GT-curvature
    fraction receive ``curvature_high_weight`` while all other valid residuals
    retain unit weight. The GT-derived weights are detached and the reduction is
    normalized by their sum, so weighting changes gradient allocation rather
    than the overall loss scale.
    """
    if pred_xy.shape != target_xy.shape or pred_xy.dim() != 2 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy and target_xy must have matching [N,2] shapes, got "
            f"{tuple(pred_xy.shape)} vs {tuple(target_xy.shape)}."
        )
    if valid.shape != pred_xy.shape[:-1]:
        raise ValueError(f"valid must have shape {tuple(pred_xy.shape[:-1])}, got {tuple(valid.shape)}.")
    min_step = float(min_step_px)
    curvature_tolerance = float(curvature_tolerance_inv_px)
    high_fraction = float(curvature_high_fraction)
    high_weight = float(curvature_high_weight)
    if not math.isfinite(min_step) or min_step <= 0.0:
        raise ValueError(f"proj_derivative_min_step_px must be finite and > 0, got {min_step_px}.")
    if not math.isfinite(curvature_tolerance) or curvature_tolerance <= 0.0:
        raise ValueError(
            "proj_curvature_tolerance_inv_px must be finite and > 0, got "
            f"{curvature_tolerance_inv_px}."
        )
    if not math.isfinite(high_fraction) or not 0.0 < high_fraction <= 1.0:
        raise ValueError(
            "proj_curvature_high_fraction must be finite and in (0,1], got "
            f"{curvature_high_fraction}."
        )
    if not math.isfinite(high_weight) or high_weight < 1.0:
        raise ValueError(
            "proj_curvature_high_weight must be finite and >= 1, got "
            f"{curvature_high_weight}."
        )
    steps = _parse_asymmetric_curvature_steps(curvature_steps)
    zero = pred_xy.new_zeros(())
    tangent_losses: list[torch.Tensor] = []
    tangent_angle_errors: list[torch.Tensor] = []
    tangent_valid_fractions: list[torch.Tensor] = []
    curvature_losses: list[torch.Tensor] = []
    curvature_errors: list[torch.Tensor] = []
    curvature_valid_fractions: list[torch.Tensor] = []
    point_count = int(pred_xy.shape[0])
    eps = 1e-8

    for step in steps:
        if point_count <= 2 * step:
            continue

        pred_chords = pred_xy[step:] - pred_xy[:-step]
        target_chords = target_xy[step:] - target_xy[:-step]
        target_chord_lengths = torch.linalg.norm(target_chords, dim=-1)
        tangent_valid = (
            valid[step:]
            & valid[:-step]
            & torch.isfinite(pred_chords).all(dim=-1)
            & torch.isfinite(target_chords).all(dim=-1)
            & (target_chord_lengths > min_step)
        )
        tangent_valid_fractions.append(tangent_valid.to(dtype=pred_xy.dtype).mean())
        if bool(tangent_valid.any().item()):
            pred_tangent = F.normalize(pred_chords, dim=-1, eps=eps)
            target_tangent = F.normalize(target_chords, dim=-1, eps=eps)
            tangent_cosine = torch.sum(pred_tangent * target_tangent, dim=-1).clamp(-1.0, 1.0)
            tangent_losses.append((1.0 - tangent_cosine[tangent_valid]).mean())
            tangent_angle_errors.append(
                torch.rad2deg(torch.acos(tangent_cosine[tangent_valid]))
            )

        pred_left = pred_xy[step:-step] - pred_xy[:-2 * step]
        pred_right = pred_xy[2 * step:] - pred_xy[step:-step]
        pred_span = pred_xy[2 * step:] - pred_xy[:-2 * step]
        target_left = target_xy[step:-step] - target_xy[:-2 * step]
        target_right = target_xy[2 * step:] - target_xy[step:-step]
        target_span = target_xy[2 * step:] - target_xy[:-2 * step]
        target_left_length = torch.linalg.norm(target_left, dim=-1)
        target_right_length = torch.linalg.norm(target_right, dim=-1)
        target_span_length = torch.linalg.norm(target_span, dim=-1)
        curvature_valid = (
            valid[:-2 * step]
            & valid[step:-step]
            & valid[2 * step:]
            & torch.isfinite(pred_left).all(dim=-1)
            & torch.isfinite(pred_right).all(dim=-1)
            & torch.isfinite(pred_span).all(dim=-1)
            & torch.isfinite(target_left).all(dim=-1)
            & torch.isfinite(target_right).all(dim=-1)
            & torch.isfinite(target_span).all(dim=-1)
            & (target_left_length > min_step)
            & (target_right_length > min_step)
            & (target_span_length > min_step)
        )
        curvature_valid_fractions.append(curvature_valid.to(dtype=pred_xy.dtype).mean())
        if not bool(curvature_valid.any().item()):
            continue

        pred_left_length = torch.linalg.norm(pred_left, dim=-1).clamp_min(min_step)
        pred_right_length = torch.linalg.norm(pred_right, dim=-1).clamp_min(min_step)
        pred_span_length = torch.linalg.norm(pred_span, dim=-1).clamp_min(min_step)
        pred_cross = pred_left[:, 0] * pred_right[:, 1] - pred_left[:, 1] * pred_right[:, 0]
        target_cross = (
            target_left[:, 0] * target_right[:, 1]
            - target_left[:, 1] * target_right[:, 0]
        )
        pred_curvature = 2.0 * pred_cross / (
            pred_left_length * pred_right_length * pred_span_length
        )
        target_curvature = 2.0 * target_cross / (
            target_left_length * target_right_length * target_span_length
        ).clamp_min(eps)
        curvature_delta = pred_curvature[curvature_valid] - target_curvature[curvature_valid]
        normalized_delta = curvature_delta / curvature_tolerance
        curvature_residual = F.smooth_l1_loss(
            normalized_delta,
            torch.zeros_like(normalized_delta),
            reduction="none",
        )
        target_importance = target_curvature[curvature_valid].abs().detach()
        importance_weights = torch.ones_like(target_importance)
        positive_importance = target_importance > eps
        if high_weight > 1.0 and bool(positive_importance.any().item()):
            positive_count = int(positive_importance.sum().item())
            high_count = min(
                positive_count,
                max(
                    1,
                    int(math.ceil(high_fraction * float(target_importance.numel()))),
                ),
            )
            ranked_importance = target_importance.masked_fill(
                ~positive_importance,
                float("-inf"),
            )
            high_indices = torch.topk(
                ranked_importance,
                k=high_count,
                largest=True,
                sorted=False,
            ).indices
            importance_weights[high_indices] = high_weight
        curvature_losses.append(
            (curvature_residual * importance_weights).sum()
            / importance_weights.sum().clamp_min(eps)
        )
        curvature_errors.append(curvature_delta.abs())

    tangent_angle = (
        torch.cat(tangent_angle_errors).mean()
        if tangent_angle_errors
        else zero
    )
    curvature_mae = (
        torch.cat(curvature_errors).mean()
        if curvature_errors
        else zero
    )
    return {
        "tangent_loss": torch.stack(tangent_losses).mean() if tangent_losses else zero,
        "tangent_angle_mae_deg": tangent_angle,
        "tangent_valid_fraction": (
            torch.stack(tangent_valid_fractions).mean()
            if tangent_valid_fractions
            else zero
        ),
        "curvature_loss": torch.stack(curvature_losses).mean() if curvature_losses else zero,
        "curvature_mae_inv_px": curvature_mae,
        "curvature_valid_fraction": (
            torch.stack(curvature_valid_fractions).mean()
            if curvature_valid_fractions
            else zero
        ),
    }

def _projector_main_surface_center(projector: DifferentiableVesselProjector, vessel_code: torch.Tensor) -> torch.Tensor:
    branch = vessel_code[0]
    centerline = branch[:, :3]
    radius = torch.clamp(branch[:, 3], min=1e-5)
    derivatives = projector._estimate_derivatives(centerline)
    normals, convecs = projector._tubeplot_frame(centerline, derivatives)
    main_points = projector._non_diff_centering_surface_points(centerline, radius, normals, convecs)
    if main_points.numel() == 0:
        return vessel_code.new_zeros((3,))
    return torch.mean(main_points, dim=0)

def _project_with_existing_center_mode(
    projector: DifferentiableVesselProjector,
    vessel_batch: torch.Tensor,
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
) -> torch.Tensor:
    old_center_main_branch = bool(projector.center_main_branch)
    projector.center_main_branch = False
    try:
        return projector(vessel_batch, theta_deg, phi_deg)
    finally:
        projector.center_main_branch = old_center_main_branch

def _project_points_uncropped(
    projector: DifferentiableVesselProjector,
    points: torch.Tensor,
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project points to the stored Stage-2 image's ``(column, row)`` coordinates.

    The legacy NumPy renderer flips detector y in ``convert3D_to_pixels`` and then
    flips it again while converting that value to an image row. Those operations
    cancel, so the stored mask row is the unflipped detector-y pixel coordinate.
    Do not apply another y flip here: callers use this output directly for
    ``grid_sample`` and Matplotlib overlays on the stored masks.
    """
    cameras = projector.prepare_cameras(
        theta_deg.reshape(1), phi_deg.reshape(1)
    )
    xy, valid = projector._project_points_batched(
        points, cameras, apply_crop=False
    )
    return xy[0], valid[0]

def _sample_distance_map(
    distance_map: torch.Tensor, xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample a pixel-distance map at continuous projected points.

    Points outside the image receive the sampled boundary value plus their
    Euclidean distance to the image boundary. This is shared with the
    parametric-model projection loss so both training paths use identical
    centreline-to-mask geometry.
    """
    height, width = int(distance_map.shape[-2]), int(distance_map.shape[-1])
    x = xy[:, 0]
    y = xy[:, 1]
    x_clamped = x.clamp(0.0, float(max(width - 1, 0)))
    y_clamped = y.clamp(0.0, float(max(height - 1, 0)))
    boundary_distance = torch.sqrt(
        (x - x_clamped).square() + (y - y_clamped).square() + 1e-8
    )
    grid_x = 2.0 * x_clamped / float(max(width - 1, 1)) - 1.0
    grid_y = 2.0 * y_clamped / float(max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        distance_map.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)
    inside_bounds = (
        (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    return sampled + boundary_distance, inside_bounds

__all__ = [
    "VESSEL_LOSS_KEYS",
    "_asymmetric_curvature_recall_jitter_loss",
    "_bounded_curvature_vector_loss",
    "_bounded_curvature_vector_profile",
    "_parse_bounded_curvature_steps",
    "_sample_distance_map",
    "attach_index_loss",
    "centerline_dt_batch_to_device",
    "compute_centerline_3d_dt_loss",
    "compute_vessel_code_loss",
    "compute_optional_projection_loss",
    "realdata_soft_mask_loss",
    "realdata_vessel_code_loss",
    "root_attachment_loss",
    "slice_centerline_dt_batch",
    "vessel_loss_kwargs_from_config",
    "vessel_occupancy_3d_ssim_loss",
]
