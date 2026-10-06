# Transferred from methods/parametric_methods/projection_loss.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector
from vessel_code.parametric.bend_recall import asymmetric_bend_recall_loss
from vessel_code.parametric.representation import normalize_centerline_prediction_mode
from vessel_code.shared.loss import _dice_loss_from_probs, _project_points_uncropped, _projected_centerline_derivative_metrics, _project_with_existing_center_mode, _sample_distance_map
from vessel_code.shared.mountain_weighting import detached_mountain_weighted_mean
from vessel_code.shared.model import ABSOLUTE_PARALLEL_DECODER, MAIN_FIRST_HIERARCHICAL_DECODER, normalize_decoder_architecture

def _centerline_prediction_mode(config: dict[str, Any]) -> str:
    value: Any = config.get(
        "centerline_prediction_mode", "bspline_control_points"
    )
    model_config = config.get("model")
    if isinstance(model_config, dict) and "centerline_prediction_mode" in model_config:
        value = model_config["centerline_prediction_mode"]
    return normalize_centerline_prediction_mode(value)

def _absolute_parallel_coordinates(config: dict[str, Any]) -> bool:
    model_config = config.get("model")
    value: Any = config.get(
        "decoder_architecture", MAIN_FIRST_HIERARCHICAL_DECODER
    )
    if isinstance(model_config, dict) and "decoder_architecture" in model_config:
        value = model_config["decoder_architecture"]
    return normalize_decoder_architecture(value) == ABSOLUTE_PARALLEL_DECODER

def _global_landmark_coordinates(
    landmarks_mm: torch.Tensor,
    decoded_centerlines_mm: torch.Tensor,
) -> torch.Tensor:
    """Convert main-global/side-relative landmarks to global XYZ coordinates."""
    if landmarks_mm.dim() != 4 or landmarks_mm.shape[-1] != 3:
        raise ValueError(
            "Landmarks must have shape [B,M,K,3], got "
            f"{tuple(landmarks_mm.shape)}"
        )
    if (
        decoded_centerlines_mm.dim() != 4
        or decoded_centerlines_mm.shape[-1] != 3
        or decoded_centerlines_mm.shape[:2] != landmarks_mm.shape[:2]
    ):
        raise ValueError(
            "Decoded centrelines must have shape [B,M,N,3] with the same "
            f"batch/branch dimensions as landmarks; got "
            f"{tuple(decoded_centerlines_mm.shape)} and "
            f"{tuple(landmarks_mm.shape)}"
        )
    global_landmarks = landmarks_mm.clone()
    if landmarks_mm.shape[1] > 1:
        local_offsets = (
            landmarks_mm[:, 1:] - landmarks_mm[:, 1:, :1]
        )
        attachment_origins = decoded_centerlines_mm[:, 1:, :1]
        global_landmarks[:, 1:] = local_offsets + attachment_origins
    return global_landmarks

def build_centerline_projector(
    config: dict[str, Any], image_size: int
) -> DifferentiableVesselProjector:
    return DifferentiableVesselProjector(
        image_size=int(image_size),
        sid=float(config.get("proj_loss_sid", config.get("vis_proj_sid", 0.9))),
        source_to_iso=float(config.get("proj_loss_source_to_iso", 0.75)),
        imager_pixel_spacing=float(
            config.get(
                "proj_loss_imager_pixel_spacing",
                config.get("vis_proj_imager_pixel_spacing", 0.55),
            )
        ),
        num_circle_points=int(config.get("proj_render_num_circle_points", 48)),
        radial_subsamples=int(config.get("proj_render_radial_subsamples", 1)),
        axial_subsamples=int(config.get("proj_render_axial_subsamples", 4)),
        center_main_branch=False,
        crop_mode="none",
    )

def build_bspline_refiner_centerline_projector(
    config: dict[str, Any],
) -> DifferentiableVesselProjector:
    """Build the target projector with the refiner's exact camera contract.

    The ordinary projection loss and the B-spline refiner may intentionally use
    different raster sizes or camera settings.  Keeping this projector separate
    prevents centreline-map targets from silently being rendered in the
    ordinary projection-loss coordinate system.
    """

    merged = dict(config)
    model_config = config.get("model")
    if isinstance(model_config, dict):
        merged.update(model_config)
    return DifferentiableVesselProjector(
        image_size=int(merged.get("bspline_refiner_image_size", 256)),
        sid=float(
            merged.get(
                "bspline_refiner_sid",
                merged.get("proj_loss_sid", 0.9),
            )
        ),
        source_to_iso=float(
            merged.get(
                "bspline_refiner_source_to_iso",
                merged.get("proj_loss_source_to_iso", 0.75),
            )
        ),
        imager_pixel_spacing=float(
            merged.get(
                "bspline_refiner_imager_pixel_spacing",
                merged.get("proj_loss_imager_pixel_spacing", 0.55),
            )
        ),
        # These surface-rendering settings are inert for point projection, but
        # explicit valid values keep the projector independently usable.
        num_circle_points=int(config.get("proj_render_num_circle_points", 48)),
        radial_subsamples=int(config.get("proj_render_radial_subsamples", 1)),
        axial_subsamples=int(config.get("proj_render_axial_subsamples", 4)),
        center_main_branch=False,
        crop_mode="none",
    )

def projection_schedule_factor(config: dict[str, Any], epoch: int | None) -> float:
    if epoch is None:
        return 1.0
    start = int(config.get("proj_loss_start_epoch", 1))
    if int(epoch) < start:
        return 0.0
    if not bool(config.get("proj_loss_schedule_enabled", False)):
        return 1.0
    ramp_start = int(config.get("proj_loss_schedule_start_epoch", start))
    ramp_end = int(config.get("proj_loss_schedule_end_epoch", ramp_start))
    start_factor = float(config.get("proj_loss_schedule_start_factor", 0.0))
    end_factor = float(config.get("proj_loss_schedule_end_factor", 1.0))
    if ramp_end <= ramp_start:
        return end_factor
    fraction = min(max((int(epoch) - ramp_start) / float(ramp_end - ramp_start), 0.0), 1.0)
    return start_factor + fraction * (end_factor - start_factor)

def _gradient_routed_surface_vessel(
    vessel: torch.Tensor,
    *,
    geometry_weight: float,
    radius_weight: float,
    joint_weight: float,
) -> tuple[torch.Tensor, float]:
    """Route one rendered Dice gradient as three logical stop-gradient terms.

    Rendering ``(XYZ, stopgrad(radius))``, ``(stopgrad(XYZ), radius)``, and
    ``(XYZ, radius)`` produces the same forward mask three times.  Their
    weighted sum therefore has XYZ gradient ``geometry + joint`` and radius
    gradient ``radius + joint``.  The straight-through scaling below is
    algebraically identical for first-order optimization while requiring only
    one expensive differentiable surface render.
    """
    if vessel.shape[-1] != 4:
        raise ValueError(
            "Split surface Dice expects vessel channels [X,Y,Z,radius], got "
            f"shape {tuple(vessel.shape)}."
        )
    total_weight = float(geometry_weight + radius_weight + joint_weight)
    if total_weight <= 0.0:
        raise ValueError("Split surface Dice requires a positive total weight")
    xyz_gradient_scale = (geometry_weight + joint_weight) / total_weight
    radius_gradient_scale = (radius_weight + joint_weight) / total_weight
    channel_scale = vessel.new_tensor(
        [
            xyz_gradient_scale,
            xyz_gradient_scale,
            xyz_gradient_scale,
            radius_gradient_scale,
        ]
    )
    detached = vessel.detach()
    routed = detached + (vessel - detached) * channel_scale
    return routed, total_weight

def _landmark_margin_tail_loss(
    distance_groups_mm: list[torch.Tensor],
    *,
    margin_mm: float,
    topk_fraction: float,
    scale_mm: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the worst squared landmark-margin violations per case/view.

    Each input tensor contains the valid projected landmark distances for one
    case/view, potentially concatenated across active branches.  Keeping the
    hard-mining operation inside each case/view prevents a well-matched view
    from hiding a local violation in another view.
    """
    if not math.isfinite(float(margin_mm)) or float(margin_mm) < 0.0:
        raise ValueError(
            f"proj_landmark_margin_tail_margin_mm must be finite and >= 0, got {margin_mm}."
        )
    if (
        not math.isfinite(float(topk_fraction))
        or float(topk_fraction) <= 0.0
        or float(topk_fraction) > 1.0
    ):
        raise ValueError(
            "proj_landmark_margin_tail_fraction must be finite and in (0, 1], "
            f"got {topk_fraction}."
        )
    if not math.isfinite(float(scale_mm)) or float(scale_mm) <= 0.0:
        raise ValueError(
            f"proj_landmark_margin_tail_scale_mm must be finite and > 0, got {scale_mm}."
        )

    group_losses: list[torch.Tensor] = []
    exceedance_counts: list[torch.Tensor] = []
    for distances_mm in distance_groups_mm:
        if distances_mm.numel() == 0:
            continue
        violations = F.relu(distances_mm - float(margin_mm)) / float(scale_mm)
        per_landmark_loss = violations.square()
        k = min(
            int(per_landmark_loss.numel()),
            max(1, int(math.ceil(float(topk_fraction) * per_landmark_loss.numel()))),
        )
        group_losses.append(torch.topk(per_landmark_loss, k=k).values.mean())
        exceedance_counts.append((distances_mm > float(margin_mm)).to(zero.dtype))

    if not group_losses:
        return zero, zero
    exceedance_fraction = torch.cat(exceedance_counts).mean()
    return torch.stack(group_losses).mean(), exceedance_fraction

def _aggregate_paired_branch_view_losses(
    branch_view_losses: list[list[list[torch.Tensor]]],
    branch_view_mae_px: list[list[list[torch.Tensor]]],
    *,
    aggregation: str,
    worst_view_weight: float,
    worst_view_temperature_px: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate paired losses without letting easy branches/views dominate.

    ``branch_view_losses[case][branch]`` contains one scalar loss per valid
    selected view.  The branch-view modes first average views within each
    branch, then branches within each case, and finally cases.  This gives each
    valid branch equal weight even when its raw centreline has fewer points.

    ``branch_view_soft_worst`` mixes the ordinary per-branch view mean with a
    differentiable soft worst-view term.  Its detached softmax weights are
    computed from detector-pixel MAE: gradients therefore flow through the
    selected view losses without allowing the model to manipulate the weights.
    The convex mixture preserves the loss scale and is exactly the ordinary
    mean when only one view is available.
    """

    normalized = str(aggregation).strip().lower()
    supported = {"global_mean", "branch_view_mean", "branch_view_soft_worst"}
    if normalized not in supported:
        raise ValueError(
            "proj_paired_aggregation must be one of "
            f"{sorted(supported)}, got {aggregation!r}."
        )
    if not math.isfinite(float(worst_view_weight)) or not (
        0.0 <= float(worst_view_weight) <= 1.0
    ):
        raise ValueError(
            "proj_paired_worst_view_weight must be finite and in [0, 1], "
            f"got {worst_view_weight}."
        )
    if (
        not math.isfinite(float(worst_view_temperature_px))
        or float(worst_view_temperature_px) <= 0.0
    ):
        raise ValueError(
            "proj_paired_worst_view_temperature_px must be finite and > 0, "
            f"got {worst_view_temperature_px}."
        )
    if len(branch_view_losses) != len(branch_view_mae_px):
        raise ValueError("Paired branch-view loss and MAE case counts must match")

    case_losses: list[torch.Tensor] = []
    case_mean_losses: list[torch.Tensor] = []
    case_worst_losses: list[torch.Tensor] = []
    case_worst_mae_px: list[torch.Tensor] = []
    for case_losses_raw, case_mae_raw in zip(
        branch_view_losses, branch_view_mae_px
    ):
        if len(case_losses_raw) != len(case_mae_raw):
            raise ValueError(
                "Paired branch-view loss and MAE branch counts must match"
            )
        per_branch_losses: list[torch.Tensor] = []
        per_branch_means: list[torch.Tensor] = []
        per_branch_worst: list[torch.Tensor] = []
        per_branch_worst_mae: list[torch.Tensor] = []
        for view_losses_raw, view_mae_raw in zip(
            case_losses_raw, case_mae_raw
        ):
            if len(view_losses_raw) != len(view_mae_raw):
                raise ValueError(
                    "Paired branch-view loss and MAE view counts must match"
                )
            if not view_losses_raw:
                continue
            view_losses = torch.stack(view_losses_raw)
            view_mae = torch.stack(view_mae_raw)
            mean_loss = view_losses.mean()
            soft_weights = torch.softmax(
                view_mae.detach() / float(worst_view_temperature_px), dim=0
            )
            soft_worst_loss = (soft_weights * view_losses).sum()
            soft_worst_mae = (soft_weights * view_mae).sum()
            if normalized == "branch_view_soft_worst":
                branch_loss = (
                    (1.0 - float(worst_view_weight)) * mean_loss
                    + float(worst_view_weight) * soft_worst_loss
                )
            else:
                branch_loss = mean_loss
            per_branch_losses.append(branch_loss)
            per_branch_means.append(mean_loss)
            per_branch_worst.append(soft_worst_loss)
            per_branch_worst_mae.append(soft_worst_mae)
        if not per_branch_losses:
            continue
        case_losses.append(torch.stack(per_branch_losses).mean())
        case_mean_losses.append(torch.stack(per_branch_means).mean())
        case_worst_losses.append(torch.stack(per_branch_worst).mean())
        case_worst_mae_px.append(torch.stack(per_branch_worst_mae).mean())

    if not case_losses:
        return zero, zero, zero, zero
    return (
        torch.stack(case_losses).mean(),
        torch.stack(case_mean_losses).mean(),
        torch.stack(case_worst_losses).mean(),
        torch.stack(case_worst_mae_px).mean(),
    )

def compute_centerline_projection_losses(
    *,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    projector: DifferentiableVesselProjector,
    epoch: int | None,
    return_per_case_metrics: bool = False,
) -> dict[str, torch.Tensor]:
    predicted = output["decoded_vessel_mm"]
    device, dtype = predicted.device, predicted.dtype
    target_mode = str(config.get("projection_centerline_target", "raw")).strip().lower()
    if target_mode not in {"raw", "parametric"}:
        raise ValueError("projection_centerline_target must be 'raw' or 'parametric'")
    target = batch[
        "target_raw_vessel_mm"
        if target_mode == "raw"
        else "target_reconstructed_vessel_mm"
    ].to(device=device, dtype=dtype)
    branch_exists = batch["target_branch_exist"].to(device=device) > 0.5
    branch_exists = branch_exists.clone()
    branch_exists[:, 0] = True
    point_valid = batch["target_point_valid_mask"].to(device=device, dtype=torch.bool)
    view_mask = batch["view_mask"].to(device=device) > 0.5
    theta = batch.get("theta")
    phi = batch.get("phi")
    images = batch.get("images")
    distance_maps = batch.get("projection_mask_distance_px")
    mask_weight = float(config.get("proj_mask_distance_weight", 1.0))
    if theta is None or phi is None:
        raise ValueError("2D projection loss requires theta/phi view angles")
    if images is None:
        raise ValueError("2D projection loss requires input images")
    if mask_weight > 0.0 and distance_maps is None:
        raise ValueError(
            "A positive projection mask-distance weight requires precomputed "
            "mask distances"
        )
    theta = theta.to(device=device, dtype=dtype)
    phi = phi.to(device=device, dtype=dtype)
    images = images.to(device=device, dtype=dtype)
    if distance_maps is not None:
        distance_maps = distance_maps.to(device=device, dtype=dtype)

    scale_to_meter = float(config.get("projection_coord_scale_to_meter", 0.001))
    predicted_m = predicted[..., :3] * scale_to_meter
    target_m = target[..., :3] * scale_to_meter
    if str(config.get("target_coordinate_frame", "projection_centered")) == "absolute_world":
        center = batch["projection_center_offset"].to(device=device, dtype=dtype)
        center = center[:, None, None, :] * scale_to_meter
        predicted_m = predicted_m - center
        target_m = target_m - center

    max_views = int(config.get("proj_loss_num_views", view_mask.shape[1]))
    tolerance = max(float(config.get("proj_centerline_tolerance_px", 3.0)), 1e-6)
    mask_threshold = float(config.get("projection_mask_threshold", 0.5))
    paired_weight = float(config.get("proj_paired_centerline_weight", 1.0))
    paired_aggregation = str(
        config.get("proj_paired_aggregation", "global_mean")
    ).strip().lower()
    paired_worst_view_weight = float(
        config.get("proj_paired_worst_view_weight", 0.0)
    )
    paired_worst_view_temperature_px = float(
        config.get("proj_paired_worst_view_temperature_px", 2.0)
    )
    if paired_aggregation not in {
        "global_mean",
        "branch_view_mean",
        "branch_view_soft_worst",
    }:
        raise ValueError(
            "proj_paired_aggregation must be one of ['branch_view_mean', "
            "'branch_view_soft_worst', 'global_mean'], got "
            f"{paired_aggregation!r}."
        )
    if not math.isfinite(paired_worst_view_weight) or not (
        0.0 <= paired_worst_view_weight <= 1.0
    ):
        raise ValueError(
            "proj_paired_worst_view_weight must be finite and in [0, 1], "
            f"got {paired_worst_view_weight}."
        )
    if (
        not math.isfinite(paired_worst_view_temperature_px)
        or paired_worst_view_temperature_px <= 0.0
    ):
        raise ValueError(
            "proj_paired_worst_view_temperature_px must be finite and > 0, "
            f"got {paired_worst_view_temperature_px}."
        )
    if (
        paired_aggregation != "branch_view_soft_worst"
        and paired_worst_view_weight != 0.0
    ):
        raise ValueError(
            "proj_paired_worst_view_weight must be 0 unless "
            "proj_paired_aggregation='branch_view_soft_worst'."
        )
    dice_weight = float(config.get("proj_dice_weight", 1.0))
    dice_gradient_split_raw = config.get(
        "proj_dice_gradient_split_enabled", False
    )
    if not isinstance(dice_gradient_split_raw, bool):
        raise ValueError(
            "proj_dice_gradient_split_enabled must be a JSON boolean "
            f"(true or false), got {dice_gradient_split_raw!r}."
        )
    dice_gradient_split_enabled = dice_gradient_split_raw
    geometry_dice_weight = float(config.get("proj_geometry_dice_weight", 1.0))
    radius_dice_weight = float(config.get("proj_radius_dice_weight", 1.0))
    joint_dice_weight = float(config.get("proj_joint_dice_weight", 0.1))
    split_dice_total_weight = 0.0
    if dice_gradient_split_enabled:
        for weight_name, weight_value in (
            ("proj_geometry_dice_weight", geometry_dice_weight),
            ("proj_radius_dice_weight", radius_dice_weight),
            ("proj_joint_dice_weight", joint_dice_weight),
        ):
            if not math.isfinite(weight_value):
                raise ValueError(
                    f"{weight_name} must be finite, got {weight_value}."
                )
            if weight_value < 0.0:
                raise ValueError(
                    f"{weight_name} must be non-negative, got {weight_value}."
                )
        if dice_weight != 0.0:
            raise ValueError(
                "proj_dice_weight must be 0 when split Dice gradients are "
                "enabled; configure proj_geometry_dice_weight, "
                "proj_radius_dice_weight, and proj_joint_dice_weight instead."
            )
        split_dice_total_weight = (
            geometry_dice_weight + radius_dice_weight + joint_dice_weight
        )
        if split_dice_total_weight <= 0.0:
            raise ValueError(
                "at least one split Dice weight must be positive when "
                "proj_dice_gradient_split_enabled=true."
            )
    tangent_weight = float(config.get("proj_tangent_weight", 0.0))
    curvature_weight = float(config.get("proj_curvature_weight", 0.0))
    bend_recall_weight = float(config.get("proj_bend_recall_weight", 0.0))
    mountain_weight_boost = float(
        config.get("proj_centerline_mountain_weight_boost", 4.0)
    )
    mountain_detector = str(
        config.get(
            "proj_centerline_mountain_detector",
            "local_prominence",
        )
    )
    mountain_width_pairs = config.get(
        "proj_centerline_mountain_width_pairs",
        ((8, 8), (8, 12), (12, 8), (16, 16), (24, 24)),
    )
    mountain_scale_fractions = config.get(
        "proj_centerline_mountain_scale_fractions",
        (0.02, 0.04, 0.06, 0.08, 0.12, 0.16),
    )
    mountain_min_prominence_px = float(
        config.get("proj_centerline_mountain_min_prominence_px", 2.0)
    )
    mountain_temperature_px = float(
        config.get("proj_centerline_mountain_temperature_px", 0.5)
    )
    mountain_smooth_kernel = int(
        config.get("proj_centerline_mountain_smooth_kernel", 5)
    )
    mountain_edge_weight = float(
        config.get("proj_centerline_mountain_edge_weight", 0.25)
    )
    mountain_min_slope_fraction = float(
        config.get("proj_centerline_mountain_min_slope_fraction", 0.65)
    )
    mountain_min_area_fraction = float(
        config.get("proj_centerline_mountain_min_area_fraction", 0.25)
    )
    mountain_min_region_width = int(
        config.get("proj_centerline_mountain_min_region_width", 7)
    )
    mountain_min_scale_persistence = int(
        config.get("proj_centerline_mountain_min_scale_persistence", 2)
    )
    landmark_reprojection_enabled = bool(
        config.get("proj_landmark_reprojection_enabled", False)
    )
    landmark_reprojection_active = (
        landmark_reprojection_enabled
        and _centerline_prediction_mode(config) == "adaptive_landmarks"
    )
    landmark_reprojection_weight = float(
        config.get("proj_landmark_reprojection_weight", 1.0)
    )
    landmark_margin_tail_weight = float(
        config.get("proj_landmark_margin_tail_weight", 0.0)
    )
    landmark_margin_tail_margin_mm = float(
        config.get("proj_landmark_margin_tail_margin_mm", 5.0)
    )
    landmark_margin_tail_fraction = float(
        config.get("proj_landmark_margin_tail_fraction", 0.10)
    )
    landmark_margin_tail_scale_mm = float(
        config.get(
            "proj_landmark_margin_tail_scale_mm",
            max(landmark_margin_tail_margin_mm, 1.0),
        )
    )
    paired_geometry_active = any(
        weight > 0.0
        for weight in (
            paired_weight,
            tangent_weight,
            curvature_weight,
            bend_recall_weight,
        )
    )
    landmark_terms_active = landmark_reprojection_active and (
        landmark_reprojection_weight > 0.0
        or landmark_margin_tail_weight > 0.0
    )
    centerline_terms_active = (
        paired_geometry_active or mask_weight > 0.0 or landmark_terms_active
    )
    detector_pixel_spacing_mm = float(
        config.get(
            "proj_loss_imager_pixel_spacing",
            config.get("vis_proj_imager_pixel_spacing", 0.55),
        )
    )
    if (
        not math.isfinite(detector_pixel_spacing_mm)
        or detector_pixel_spacing_mm <= 0.0
    ):
        raise ValueError(
            "proj_loss_imager_pixel_spacing must be finite and > 0, "
            f"got {detector_pixel_spacing_mm}."
        )
    for weight_name, weight_value in (
        ("proj_paired_centerline_weight", paired_weight),
        ("proj_mask_distance_weight", mask_weight),
        ("proj_dice_weight", dice_weight),
        ("proj_tangent_weight", tangent_weight),
        ("proj_curvature_weight", curvature_weight),
        ("proj_bend_recall_weight", bend_recall_weight),
        (
            "proj_centerline_mountain_weight_boost",
            mountain_weight_boost,
        ),
        (
            "proj_landmark_reprojection_weight",
            landmark_reprojection_weight,
        ),
        (
            "proj_landmark_margin_tail_weight",
            landmark_margin_tail_weight,
        ),
    ):
        if weight_value < 0.0:
            raise ValueError(f"{weight_name} must be non-negative, got {weight_value}")
    paired_distances: list[torch.Tensor] = []
    paired_point_losses: list[torch.Tensor] = []
    paired_mountain_region_values: list[torch.Tensor] = []
    paired_mountain_weight_values: list[torch.Tensor] = []
    paired_mountain_peak_activations: list[torch.Tensor] = []
    paired_branch_view_losses: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(int(predicted.shape[1]))]
        for _ in range(int(predicted.shape[0]))
    ]
    paired_branch_view_mae_px: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(int(predicted.shape[1]))]
        for _ in range(int(predicted.shape[0]))
    ]
    mask_distances: list[torch.Tensor] = []
    inside_values: list[torch.Tensor] = []
    surface_dice_losses: list[torch.Tensor] = []
    tangent_losses: list[torch.Tensor] = []
    tangent_angle_mae_values: list[torch.Tensor] = []
    tangent_valid_fractions: list[torch.Tensor] = []
    curvature_losses: list[torch.Tensor] = []
    curvature_mae_values: list[torch.Tensor] = []
    curvature_valid_fractions: list[torch.Tensor] = []
    bend_recall_losses: list[torch.Tensor] = []
    bend_underbend_losses: list[torch.Tensor] = []
    bend_excess_losses: list[torch.Tensor] = []
    bend_direction_losses: list[torch.Tensor] = []
    bend_high_region_fractions: list[torch.Tensor] = []
    bend_valid_fractions: list[torch.Tensor] = []
    landmark_reprojection_distances: list[torch.Tensor] = []
    landmark_reprojection_distance_groups: list[torch.Tensor] = []
    landmark_reprojection_valid_count = 0
    landmark_reprojection_total_count = 0

    predicted_vessel_m = predicted * scale_to_meter
    if str(config.get("target_coordinate_frame", "projection_centered")) == "absolute_world":
        predicted_vessel_m = predicted_vessel_m.clone()
        predicted_vessel_m[..., :3] = predicted_vessel_m[..., :3] - center

    predicted_landmarks_m: torch.Tensor | None = None
    target_landmarks_m: torch.Tensor | None = None
    if landmark_terms_active:
        predicted_landmarks = output.get(
            "centerline_landmarks_mm",
            output.get("centerline_parameters_mm"),
        )
        target_landmarks_raw = batch.get(
            "target_centerline_landmarks_mm",
            batch.get("target_centerline_parameters_mm"),
        )
        if predicted_landmarks is None or target_landmarks_raw is None:
            raise ValueError(
                "Landmark reprojection requires predicted and target landmark "
                "tensors"
            )
        predicted_landmarks = predicted_landmarks.to(device=device, dtype=dtype)
        target_landmarks = target_landmarks_raw.to(device=device, dtype=dtype)
        if predicted_landmarks.shape != target_landmarks.shape:
            raise ValueError(
                "Predicted and target landmark tensors must have the same shape, "
                f"got {tuple(predicted_landmarks.shape)} and "
                f"{tuple(target_landmarks.shape)}"
            )
        target_reconstructed_xyz = batch["target_reconstructed_vessel_mm"][
            ..., :3
        ].to(device=device, dtype=dtype)
        if _absolute_parallel_coordinates(config):
            predicted_landmarks_m = predicted_landmarks * scale_to_meter
            target_landmarks_m = target_landmarks * scale_to_meter
        else:
            predicted_landmarks_m = _global_landmark_coordinates(
                predicted_landmarks,
                predicted[..., :3],
            ) * scale_to_meter
            target_landmarks_m = _global_landmark_coordinates(
                target_landmarks,
                target_reconstructed_xyz,
            ) * scale_to_meter
        if (
            str(config.get("target_coordinate_frame", "projection_centered"))
            == "absolute_world"
        ):
            predicted_landmarks_m = predicted_landmarks_m - center
            target_landmarks_m = target_landmarks_m - center

    for batch_index in range(predicted.shape[0]):
        selected_views = torch.nonzero(view_mask[batch_index], as_tuple=False).flatten()[:max_views]
        active_surface_dice_weight = (
            split_dice_total_weight
            if dice_gradient_split_enabled
            else dice_weight
        )
        if active_surface_dice_weight > 0.0 and bool(selected_views.numel()):
            active_branches = torch.nonzero(
                branch_exists[batch_index], as_tuple=False
            ).flatten()
            if bool(active_branches.numel()):
                case_vessel = predicted_vessel_m[batch_index, active_branches]
                if dice_gradient_split_enabled:
                    case_vessel, routed_total_weight = (
                        _gradient_routed_surface_vessel(
                            case_vessel,
                            geometry_weight=geometry_dice_weight,
                            radius_weight=radius_dice_weight,
                            joint_weight=joint_dice_weight,
                        )
                    )
                    if not math.isclose(
                        routed_total_weight,
                        split_dice_total_weight,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError("Inconsistent split Dice total weight")
                surface_batch = case_vessel.unsqueeze(0).expand(
                    int(selected_views.numel()), -1, -1, -1
                )
                projected_surface = _project_with_existing_center_mode(
                    projector,
                    surface_batch,
                    theta[batch_index, selected_views],
                    phi[batch_index, selected_views],
                )[:, 0]
                gt_masks = images[batch_index, selected_views]
                if gt_masks.dim() == 4:
                    gt_masks = gt_masks[:, 0]
                gt_masks = (gt_masks > mask_threshold).to(dtype=dtype)
                surface_dice_losses.append(
                    _dice_loss_from_probs(projected_surface, gt_masks)
                )
        centerline_views = (
            selected_views if centerline_terms_active else selected_views[:0]
        )
        for view_index_t in centerline_views:
            view_index = int(view_index_t.item())
            view_landmark_reprojection_distances: list[torch.Tensor] = []
            for branch_index in range(predicted.shape[1]):
                if not bool(branch_exists[batch_index, branch_index].item()):
                    continue
                valid_points = point_valid[batch_index, branch_index]
                if not bool(valid_points.any().item()):
                    continue
                pred_xy, pred_geometry_valid = _project_points_uncropped(
                    projector,
                    predicted_m[batch_index, branch_index, valid_points],
                    theta[batch_index, view_index],
                    phi[batch_index, view_index],
                )
                if paired_geometry_active:
                    target_xy, target_geometry_valid = (
                        _project_points_uncropped(
                            projector,
                            target_m[
                                batch_index, branch_index, valid_points
                            ],
                            theta[batch_index, view_index],
                            phi[batch_index, view_index],
                        )
                    )
                    paired_valid = (
                        pred_geometry_valid & target_geometry_valid
                    )
                else:
                    target_xy = pred_xy
                    paired_valid = pred_geometry_valid
                if paired_weight > 0.0 and bool(paired_valid.any().item()):
                    paired_distance_profile = torch.linalg.vector_norm(
                        pred_xy - target_xy,
                        dim=-1,
                    )
                    paired_point_loss = F.smooth_l1_loss(
                        paired_distance_profile / tolerance,
                        torch.zeros_like(paired_distance_profile),
                        reduction="none",
                    )
                    mountain_parts = detached_mountain_weighted_mean(
                        paired_point_loss,
                        error_profile=paired_distance_profile,
                        valid=paired_valid,
                        weight_boost=mountain_weight_boost,
                        detector=mountain_detector,
                        width_pairs=mountain_width_pairs,
                        scale_fractions=mountain_scale_fractions,
                        min_prominence=mountain_min_prominence_px,
                        temperature=mountain_temperature_px,
                        smooth_kernel=mountain_smooth_kernel,
                        edge_weight=mountain_edge_weight,
                        min_slope_fraction=mountain_min_slope_fraction,
                        min_area_fraction=mountain_min_area_fraction,
                        min_region_width=mountain_min_region_width,
                        min_scale_persistence=(
                            mountain_min_scale_persistence
                        ),
                    )
                    paired_region = mountain_parts["region_mask"]
                    paired_point_weight = (
                        1.0 + mountain_weight_boost * paired_region
                    )
                    paired_distances.append(
                        paired_distance_profile[paired_valid]
                    )
                    paired_point_losses.append(
                        paired_point_loss[paired_valid]
                    )
                    paired_mountain_region_values.append(
                        paired_region[paired_valid]
                    )
                    paired_mountain_weight_values.append(
                        paired_point_weight[paired_valid]
                    )
                    paired_mountain_peak_activations.append(
                        mountain_parts["peak_activation"]
                    )
                    valid_group_losses = paired_point_loss[paired_valid]
                    valid_group_weights = paired_point_weight[paired_valid]
                    paired_branch_view_losses[batch_index][branch_index].append(
                        (valid_group_losses * valid_group_weights).sum()
                        / valid_group_weights.sum().clamp_min(1.0)
                    )
                    paired_branch_view_mae_px[batch_index][branch_index].append(
                        paired_distance_profile[paired_valid].mean()
                    )
                if (
                    landmark_terms_active
                    and predicted_landmarks_m is not None
                    and target_landmarks_m is not None
                ):
                    predicted_landmark_xy, predicted_landmark_valid = (
                        _project_points_uncropped(
                            projector,
                            predicted_landmarks_m[
                                batch_index, branch_index
                            ],
                            theta[batch_index, view_index],
                            phi[batch_index, view_index],
                        )
                    )
                    target_landmark_xy, target_landmark_valid = (
                        _project_points_uncropped(
                            projector,
                            target_landmarks_m[batch_index, branch_index],
                            theta[batch_index, view_index],
                            phi[batch_index, view_index],
                        )
                    )
                    landmark_valid = (
                        predicted_landmark_valid & target_landmark_valid
                    )
                    landmark_reprojection_total_count += int(
                        landmark_valid.numel()
                    )
                    landmark_reprojection_valid_count += int(
                        landmark_valid.sum().item()
                    )
                    if bool(landmark_valid.any().item()):
                        landmark_distances = torch.linalg.vector_norm(
                            predicted_landmark_xy[landmark_valid]
                            - target_landmark_xy[landmark_valid],
                            dim=-1,
                        )
                        landmark_reprojection_distances.append(landmark_distances)
                        view_landmark_reprojection_distances.append(
                            landmark_distances
                        )
                if tangent_weight > 0.0 or curvature_weight > 0.0:
                    derivative_metrics = _projected_centerline_derivative_metrics(
                        pred_xy=pred_xy,
                        target_xy=target_xy,
                        valid=paired_valid,
                        curvature_steps=config.get(
                            "proj_curvature_steps", (1, 2, 4)
                        ),
                        min_step_px=float(
                            config.get("proj_derivative_min_step_px", 0.5)
                        ),
                        curvature_tolerance_inv_px=float(
                            config.get(
                                "proj_curvature_tolerance_inv_px", 0.02
                            )
                        ),
                        curvature_high_fraction=float(
                            config.get("proj_curvature_high_fraction", 1.0)
                        ),
                        curvature_high_weight=float(
                            config.get("proj_curvature_high_weight", 1.0)
                        ),
                    )
                    tangent_losses.append(derivative_metrics["tangent_loss"])
                    tangent_angle_mae_values.append(
                        derivative_metrics["tangent_angle_mae_deg"]
                    )
                    tangent_valid_fractions.append(
                        derivative_metrics["tangent_valid_fraction"]
                    )
                    curvature_losses.append(
                        derivative_metrics["curvature_loss"]
                    )
                    curvature_mae_values.append(
                        derivative_metrics["curvature_mae_inv_px"]
                    )
                    curvature_valid_fractions.append(
                        derivative_metrics["curvature_valid_fraction"]
                    )
                if bend_recall_weight > 0.0:
                    bend_parts = asymmetric_bend_recall_loss(
                        pred_points=pred_xy,
                        target_points=target_xy,
                        valid=paired_valid,
                        steps=config.get(
                            "proj_bend_recall_steps",
                            config.get("proj_curvature_steps", (1, 2, 4)),
                        ),
                        high_fraction=float(
                            config.get(
                                "proj_bend_recall_high_fraction", 0.15
                            )
                        ),
                        low_fraction=float(
                            config.get(
                                "proj_bend_recall_low_fraction", 0.70
                            )
                        ),
                        min_recall_ratio=float(
                            config.get(
                                "proj_bend_recall_min_recall_ratio", 0.80
                            )
                        ),
                        jitter_tolerance_ratio=float(
                            config.get(
                                "proj_bend_recall_jitter_tolerance_ratio",
                                0.10,
                            )
                        ),
                        direction_weight=float(
                            config.get(
                                "proj_bend_recall_direction_weight", 0.25
                            )
                        ),
                        smooth_kernel=int(
                            config.get(
                                "proj_bend_recall_smooth_kernel", 5
                            )
                        ),
                        high_region_dilation=int(
                            config.get(
                                "proj_bend_recall_high_region_dilation", 5
                            )
                        ),
                        reference_floor=float(
                            config.get(
                                "proj_bend_recall_reference_floor", 0.05
                            )
                        ),
                        min_target_step=float(
                            config.get(
                                "proj_bend_recall_min_step_px",
                                config.get(
                                    "proj_derivative_min_step_px", 0.5
                                ),
                            )
                        ),
                    )
                    bend_recall_losses.append(bend_parts["loss"])
                    bend_underbend_losses.append(
                        bend_parts["underbend_loss"]
                    )
                    bend_excess_losses.append(bend_parts["excess_loss"])
                    bend_direction_losses.append(
                        bend_parts["direction_loss"]
                    )
                    bend_high_region_fractions.append(
                        bend_parts["high_region_fraction"]
                    )
                    bend_valid_fractions.append(
                        bend_parts["valid_fraction"]
                    )
                if mask_weight > 0.0:
                    mask_distance, in_bounds = _sample_distance_map(
                        distance_maps[batch_index, view_index], pred_xy
                    )
                    mask_valid = pred_geometry_valid
                    if bool(mask_valid.any().item()):
                        mask_distances.append(mask_distance[mask_valid])
                        image = images[batch_index, view_index]
                        if image.dim() == 3:
                            image = image[0]
                        occupancy_distance, _ = _sample_distance_map(
                            (image <= mask_threshold).to(dtype=dtype), pred_xy
                        )
                        inside_values.append(
                            (
                                (occupancy_distance <= 0.5)
                                & in_bounds
                                & mask_valid
                            ).to(dtype=dtype)
                        )
            if view_landmark_reprojection_distances:
                landmark_reprojection_distance_groups.append(
                    torch.cat(view_landmark_reprojection_distances)
                )

    zero = predicted.new_zeros(())
    paired_px = torch.cat(paired_distances) if paired_distances else zero.view(1)
    paired_point_loss_values = (
        torch.cat(paired_point_losses)
        if paired_point_losses
        else zero.view(1)
    )
    paired_mountain_regions = (
        torch.cat(paired_mountain_region_values)
        if paired_mountain_region_values
        else zero.view(1)
    )
    paired_mountain_weights = (
        torch.cat(paired_mountain_weight_values)
        if paired_mountain_weight_values
        else zero.new_ones(1)
    )
    mask_px = torch.cat(mask_distances) if mask_distances else zero.view(1)
    inside = torch.cat(inside_values) if inside_values else zero.view(1)
    landmark_reprojection_px = (
        torch.cat(landmark_reprojection_distances)
        if landmark_reprojection_distances
        else zero.view(1)
    )
    surface_dice_loss = (
        torch.stack(surface_dice_losses).mean()
        if surface_dice_losses
        else zero
    )
    surface_dice_score = 1.0 - surface_dice_loss if surface_dice_losses else zero
    geometry_surface_dice_loss = (
        surface_dice_loss if dice_gradient_split_enabled else zero
    )
    radius_surface_dice_loss = (
        surface_dice_loss if dice_gradient_split_enabled else zero
    )
    joint_surface_dice_loss = (
        surface_dice_loss if dice_gradient_split_enabled else zero
    )
    geometry_surface_dice_score = (
        surface_dice_score if dice_gradient_split_enabled else zero
    )
    radius_surface_dice_score = (
        surface_dice_score if dice_gradient_split_enabled else zero
    )
    joint_surface_dice_score = (
        surface_dice_score if dice_gradient_split_enabled else zero
    )
    geometry_surface_dice_weighted_loss = (
        geometry_dice_weight * geometry_surface_dice_loss
        if dice_gradient_split_enabled
        else zero
    )
    radius_surface_dice_weighted_loss = (
        radius_dice_weight * radius_surface_dice_loss
        if dice_gradient_split_enabled
        else zero
    )
    joint_surface_dice_weighted_loss = (
        joint_dice_weight * joint_surface_dice_loss
        if dice_gradient_split_enabled
        else zero
    )
    surface_dice_weighted_loss = (
        geometry_surface_dice_weighted_loss
        + radius_surface_dice_weighted_loss
        + joint_surface_dice_weighted_loss
        if dice_gradient_split_enabled
        else dice_weight * surface_dice_loss
    )
    tangent_loss = (
        torch.stack(tangent_losses).mean() if tangent_losses else zero
    )
    tangent_angle_mae_deg = (
        torch.stack(tangent_angle_mae_values).mean()
        if tangent_angle_mae_values
        else zero
    )
    tangent_valid_fraction = (
        torch.stack(tangent_valid_fractions).mean()
        if tangent_valid_fractions
        else zero
    )
    curvature_loss = (
        torch.stack(curvature_losses).mean() if curvature_losses else zero
    )
    curvature_mae_inv_px = (
        torch.stack(curvature_mae_values).mean()
        if curvature_mae_values
        else zero
    )
    curvature_valid_fraction = (
        torch.stack(curvature_valid_fractions).mean()
        if curvature_valid_fractions
        else zero
    )
    bend_recall_loss = (
        torch.stack(bend_recall_losses).mean()
        if bend_recall_losses
        else zero
    )
    bend_underbend_loss = (
        torch.stack(bend_underbend_losses).mean()
        if bend_underbend_losses
        else zero
    )
    bend_excess_loss = (
        torch.stack(bend_excess_losses).mean()
        if bend_excess_losses
        else zero
    )
    bend_direction_loss = (
        torch.stack(bend_direction_losses).mean()
        if bend_direction_losses
        else zero
    )
    bend_high_region_fraction = (
        torch.stack(bend_high_region_fractions).mean()
        if bend_high_region_fractions
        else zero
    )
    bend_valid_fraction = (
        torch.stack(bend_valid_fractions).mean()
        if bend_valid_fractions
        else zero
    )
    paired_unweighted_loss = paired_point_loss_values.mean()
    paired_global_loss = (
        paired_point_loss_values * paired_mountain_weights
    ).sum() / paired_mountain_weights.sum().clamp_min(1.0)
    (
        paired_branch_view_loss,
        paired_branch_view_mean_loss,
        paired_soft_worst_view_loss,
        paired_soft_worst_view_mae_px,
    ) = _aggregate_paired_branch_view_losses(
        paired_branch_view_losses,
        paired_branch_view_mae_px,
        aggregation=paired_aggregation,
        worst_view_weight=paired_worst_view_weight,
        worst_view_temperature_px=paired_worst_view_temperature_px,
        zero=zero,
    )
    paired_loss = (
        paired_global_loss
        if paired_aggregation == "global_mean"
        else paired_branch_view_loss
    )
    paired_mountain_region_fraction = paired_mountain_regions.mean()
    paired_mountain_peak_activation = (
        torch.stack(paired_mountain_peak_activations).max()
        if paired_mountain_peak_activations
        else zero
    )
    paired_mountain_mean_weight = paired_mountain_weights.mean()
    mask_loss = F.smooth_l1_loss(
        mask_px / tolerance, torch.zeros_like(mask_px), reduction="mean"
    )
    landmark_reprojection_loss = F.smooth_l1_loss(
        landmark_reprojection_px / tolerance,
        torch.zeros_like(landmark_reprojection_px),
        reduction="mean",
    )
    landmark_reprojection_mm = (
        landmark_reprojection_px * detector_pixel_spacing_mm
    )
    landmark_margin_tail_loss, landmark_margin_exceedance_fraction = (
        _landmark_margin_tail_loss(
            [
                distances_px * detector_pixel_spacing_mm
                for distances_px in landmark_reprojection_distance_groups
            ],
            margin_mm=landmark_margin_tail_margin_mm,
            topk_fraction=landmark_margin_tail_fraction,
            scale_mm=landmark_margin_tail_scale_mm,
            zero=zero,
        )
        if landmark_reprojection_active and landmark_margin_tail_weight > 0.0
        else (zero, zero)
    )
    landmark_reprojection_valid_fraction = predicted.new_tensor(
        landmark_reprojection_valid_count
        / max(landmark_reprojection_total_count, 1)
    )
    schedule = projection_schedule_factor(config, epoch)
    global_weight = float(config.get("proj_loss_weight", 0.2)) * schedule
    combined_unweighted = (
        paired_weight * paired_loss
        + mask_weight * mask_loss
        + surface_dice_weighted_loss
        + tangent_weight * tangent_loss
        + curvature_weight * curvature_loss
        + bend_recall_weight * bend_recall_loss
        + (
            landmark_reprojection_weight * landmark_reprojection_loss
            if landmark_reprojection_active
            else zero
        )
        + (
            landmark_margin_tail_weight * landmark_margin_tail_loss
            if landmark_reprojection_active
            else zero
        )
    )
    metrics = {
        "projection_2d_loss": combined_unweighted * global_weight,
        "projection_2d_unweighted_loss": combined_unweighted,
        "projection_2d_effective_weight": predicted.new_tensor(global_weight),
        "projected_centerline_paired_loss": paired_loss,
        "projected_centerline_paired_unweighted_loss": (
            paired_unweighted_loss
        ),
        "projected_centerline_paired_global_loss": paired_global_loss,
        "projected_centerline_paired_branch_view_mean_loss": (
            paired_branch_view_mean_loss
        ),
        "projected_centerline_paired_soft_worst_view_loss": (
            paired_soft_worst_view_loss
        ),
        "projected_centerline_paired_soft_worst_view_mae_px": (
            paired_soft_worst_view_mae_px
        ),
        "projected_centerline_paired_worst_view_weight": predicted.new_tensor(
            paired_worst_view_weight
        ),
        "projected_centerline_paired_worst_view_temperature_px": (
            predicted.new_tensor(paired_worst_view_temperature_px)
        ),
        "projected_centerline_paired_mae_px": paired_px.mean(),
        "projected_centerline_mountain_region_fraction": (
            paired_mountain_region_fraction
        ),
        "projected_centerline_mountain_peak_activation": (
            paired_mountain_peak_activation
        ),
        "projected_centerline_mountain_mean_weight": (
            paired_mountain_mean_weight
        ),
        "projected_centerline_mountain_weight_boost": predicted.new_tensor(
            mountain_weight_boost
        ),
        "projected_centerline_mask_loss": mask_loss,
        "projected_centerline_mask_distance_px": mask_px.mean(),
        "projected_inside_mask_fraction": inside.mean(),
        "projected_landmark_reprojection_loss": landmark_reprojection_loss,
        "projected_landmark_reprojection_mae_px": (
            landmark_reprojection_px.mean()
        ),
        "projected_landmark_reprojection_mae_detector_mm": (
            landmark_reprojection_mm.mean()
        ),
        "projected_landmark_margin_tail_loss": landmark_margin_tail_loss,
        "projected_landmark_margin_exceedance_fraction": (
            landmark_margin_exceedance_fraction
        ),
        "projected_landmark_margin_tail_margin_mm": predicted.new_tensor(
            landmark_margin_tail_margin_mm
        ),
        "projected_landmark_margin_tail_fraction": predicted.new_tensor(
            landmark_margin_tail_fraction
        ),
        "projected_landmark_margin_tail_weighted_loss": (
            landmark_margin_tail_weight * landmark_margin_tail_loss
            if landmark_reprojection_active
            else zero
        ),
        "projected_landmark_margin_tail_effective_weight": (
            predicted.new_tensor(landmark_margin_tail_weight * global_weight)
            if landmark_reprojection_active
            else zero
        ),
        "projected_landmark_reprojection_valid_fraction": (
            landmark_reprojection_valid_fraction
        ),
        "projected_landmark_reprojection_weighted_loss": (
            landmark_reprojection_weight * landmark_reprojection_loss
            if landmark_reprojection_active
            else zero
        ),
        "projected_landmark_reprojection_effective_weight": (
            predicted.new_tensor(landmark_reprojection_weight * global_weight)
            if landmark_reprojection_active
            else zero
        ),
        "projected_surface_dice_loss": surface_dice_loss,
        "projected_surface_dice_score": surface_dice_score,
        "projected_surface_dice_weighted_loss": surface_dice_weighted_loss,
        "projected_surface_dice_effective_weight": predicted.new_tensor(
            global_weight
            * (
                split_dice_total_weight
                if dice_gradient_split_enabled
                else dice_weight
            )
        ),
        "projected_geometry_surface_dice_loss": geometry_surface_dice_loss,
        "projected_geometry_surface_dice_score": geometry_surface_dice_score,
        "projected_geometry_surface_dice_weighted_loss": (
            geometry_surface_dice_weighted_loss
        ),
        "projected_geometry_surface_dice_effective_weight": (
            predicted.new_tensor(
                global_weight * geometry_dice_weight
                if dice_gradient_split_enabled
                else 0.0
            )
        ),
        "projected_radius_surface_dice_loss": radius_surface_dice_loss,
        "projected_radius_surface_dice_score": radius_surface_dice_score,
        "projected_radius_surface_dice_weighted_loss": (
            radius_surface_dice_weighted_loss
        ),
        "projected_radius_surface_dice_effective_weight": (
            predicted.new_tensor(
                global_weight * radius_dice_weight
                if dice_gradient_split_enabled
                else 0.0
            )
        ),
        "projected_joint_surface_dice_loss": joint_surface_dice_loss,
        "projected_joint_surface_dice_score": joint_surface_dice_score,
        "projected_joint_surface_dice_weighted_loss": (
            joint_surface_dice_weighted_loss
        ),
        "projected_joint_surface_dice_effective_weight": (
            predicted.new_tensor(
                global_weight * joint_dice_weight
                if dice_gradient_split_enabled
                else 0.0
            )
        ),
        "projected_dice_gradient_split_enabled": predicted.new_tensor(
            float(dice_gradient_split_enabled)
        ),
        "projected_centerline_tangent_loss": tangent_loss,
        "projected_centerline_tangent_angle_mae_deg": tangent_angle_mae_deg,
        "projected_centerline_tangent_valid_fraction": tangent_valid_fraction,
        "projected_centerline_curvature_loss": curvature_loss,
        "projected_centerline_curvature_mae_inv_px": curvature_mae_inv_px,
        "projected_centerline_curvature_valid_fraction": curvature_valid_fraction,
        "projected_bend_recall_loss": bend_recall_loss,
        "projected_bend_underbend_loss": bend_underbend_loss,
        "projected_bend_excess_loss": bend_excess_loss,
        "projected_bend_direction_loss": bend_direction_loss,
        "projected_bend_recall_weighted_loss": (
            bend_recall_weight * bend_recall_loss
        ),
        "projected_bend_recall_effective_weight": predicted.new_tensor(
            bend_recall_weight * global_weight
        ),
        "projected_bend_high_region_fraction": bend_high_region_fraction,
        "projected_bend_valid_fraction": bend_valid_fraction,
    }
    if return_per_case_metrics:
        if len(surface_dice_losses) != int(predicted.shape[0]):
            raise RuntimeError(
                "Per-case projected surface Dice requires every case to have "
                "at least one selected view and one active branch."
            )
        metrics["projected_surface_dice_score_per_case"] = 1.0 - torch.stack(
            surface_dice_losses
        )
    return metrics
