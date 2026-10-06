# Transferred from methods/parametric_methods/loss.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F
from vessel_code.losses.error_thresholding import configured_error_thresholded_mean
from vessel_code.losses.regression import elementwise_regression_loss
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector
from vessel_code.parametric.arc_length import decoded_branch_length_loss
from vessel_code.parametric.bend_recall import asymmetric_bend_recall_loss
from vessel_code.parametric.centerline_heatmap import render_centerline_target_maps_from_grid
from vessel_code.parametric.local_progress import decoded_local_progress_loss
from vessel_code.parametric.projection_loss import compute_centerline_projection_losses, projection_schedule_factor
from vessel_code.parametric.representation import normalize_centerline_prediction_mode, normalize_radius_prediction_mode
from vessel_code.shared.branch_visibility import required_branch_count, resolve_artery_type
from vessel_code.shared.mountain_weighting import detached_mountain_weighted_mean

RAW_RADIUS_DISABLED_LOSSES = (
    "radius_coefficient_loss_weight",
    "lesion_exist_loss_weight",
    "lesion_geometry_loss_weight",
)

def _branch_existence_supervision(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
    supervision_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Force fixed RCA/LCA main targets and supervise optional queries only."""

    if logits.ndim != 2 or target.shape != logits.shape:
        raise ValueError(
            "Branch existence logits and targets must share shape [B,M], got "
            f"{tuple(logits.shape)} and {tuple(target.shape)}."
        )
    artery_type = resolve_artery_type(config)
    fixed_count = required_branch_count(artery_type)
    if int(logits.shape[1]) < fixed_count:
        raise ValueError(
            f"{artery_type} existence supervision requires at "
            f"least {fixed_count} branch slots, got {int(logits.shape[1])}."
        )
    forced_target = target.to(device=logits.device, dtype=logits.dtype).clone()
    forced_target[:, :fixed_count] = 1.0
    if supervision_mask is None:
        supervised = torch.ones_like(logits, dtype=torch.bool)
    else:
        if supervision_mask.shape != logits.shape:
            raise ValueError(
                "target_branch_existence_supervision_mask must match branch "
                "logits, got "
                f"{tuple(supervision_mask.shape)} and {tuple(logits.shape)}."
            )
        supervised = supervision_mask.to(
            device=logits.device,
            dtype=torch.bool,
        )
    if int(logits.shape[1]) > fixed_count:
        per_slot_loss = F.binary_cross_entropy_with_logits(
            logits[:, fixed_count:],
            forced_target[:, fixed_count:],
            reduction="none",
        )
        optional_supervised = supervised[:, fixed_count:]
        optional_weights = optional_supervised.to(dtype=logits.dtype)
        loss = (per_slot_loss * optional_weights).sum() / (
            optional_weights.sum().clamp_min(1.0)
        )
    else:
        loss = logits.new_zeros(())
    return forced_target, loss, fixed_count

def radius_prediction_mode_from_config(config: dict[str, Any]) -> str:
    """Resolve the radius mode with the same nested-model precedence as model building."""
    value: Any = config.get("radius_prediction_mode", "parametric")
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and "radius_prediction_mode" in model_cfg:
        value = model_cfg["radius_prediction_mode"]
    return normalize_radius_prediction_mode(value)

def centerline_prediction_mode_from_config(config: dict[str, Any]) -> str:
    """Resolve the centreline mode with nested-model configuration precedence."""
    value: Any = config.get(
        "centerline_prediction_mode", "bspline_control_points"
    )
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and "centerline_prediction_mode" in model_cfg:
        value = model_cfg["centerline_prediction_mode"]
    return normalize_centerline_prediction_mode(value)

def configure_radius_mode_losses(
    config: dict[str, Any], *, announce: bool = False
) -> str:
    """Normalize radius mode and disable incompatible parametric-radius losses."""
    mode = radius_prediction_mode_from_config(config)
    config["radius_prediction_mode"] = mode
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and "radius_prediction_mode" in model_cfg:
        model_cfg["radius_prediction_mode"] = mode
    if mode != "raw":
        return mode

    loss_cfg = config.setdefault("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("loss must be a JSON object")
    changed: list[str] = []
    for key in RAW_RADIUS_DISABLED_LOSSES:
        previous = _weight(config, key, 1.0)
        loss_cfg[key] = 0.0
        if key in config:
            config[key] = 0.0
        if previous != 0.0:
            changed.append(key)
    if announce:
        suffix = f" (overrode: {', '.join(changed)})" if changed else ""
        merged = dict(config)
        if isinstance(model_cfg, dict):
            merged.update(model_cfg)
        num_points = int(merged.get("num_points", 200))
        print(
            f"Raw radius prediction enabled: using {num_points} pointwise radii; "
            "coefficient and lesion losses are disabled; "
            "decoded_radius_loss_weight is unchanged"
            f"{suffix}"
        )
    return mode

def _weight(config: dict[str, Any], key: str, default: float) -> float:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, dict) and key in loss_cfg:
        return float(loss_cfg[key])
    return float(config.get(key, default))

def _loss_option(config: dict[str, Any], key: str, default: Any) -> Any:
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, dict) and key in loss_cfg:
        return loss_cfg[key]
    return config.get(key, default)

def _loss_option_alias(
    config: dict[str, Any],
    primary_key: str,
    alias_key: str,
    default: Any,
) -> Any:
    """Resolve a canonical loss option while accepting an SRC-compatible alias."""
    loss_cfg = config.get("loss", {})
    if isinstance(loss_cfg, dict):
        if primary_key in loss_cfg:
            return loss_cfg[primary_key]
        if alias_key in loss_cfg:
            return loss_cfg[alias_key]
    if primary_key in config:
        return config[primary_key]
    return config.get(alias_key, default)

def _model_option(config: dict[str, Any], key: str, default: Any) -> Any:
    """Resolve a model option with the same nested-model precedence as build."""

    value = config.get(key, default)
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and key in model_cfg:
        value = model_cfg[key]
    return value

def _bspline_refined_branch_existence_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    config: dict[str, Any],
    supervision_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise removal decisions only for coarse-positive side slots.

    Coarse-negative slots are deliberately excluded: this first implementation
    cannot construct missing side-branch geometry and therefore must not claim
    to solve false negatives.  Coarse-positive/GT-present slots remain in the
    loss as retention controls so the removal head does not delete real
    branches.
    """

    stage_logits = output["refinement_stage_branch_exist_logits"]
    candidate_mask = output["bspline_refiner_existence_candidate_mask"]
    coarse_logits = output["coarse_branch_exist_logits"]
    if stage_logits.dim() != 3:
        raise ValueError(
            "refinement_stage_branch_exist_logits must have shape [B,R,M], "
            f"got {tuple(stage_logits.shape)}."
        )
    batch_size, num_stages, num_branches = stage_logits.shape
    if num_stages < 1:
        raise ValueError(
            "Branch-existence refinement must contain at least one stage."
        )
    expected_shape = (batch_size, num_branches)
    if target.shape != expected_shape:
        raise ValueError(
            "target_branch_exist must match refined branch logits, got "
            f"{tuple(target.shape)} and expected {expected_shape}."
        )
    if candidate_mask.shape != expected_shape:
        raise ValueError(
            "bspline_refiner_existence_candidate_mask must have shape [B,M], "
            f"got {tuple(candidate_mask.shape)}."
        )
    if coarse_logits.shape != expected_shape:
        raise ValueError(
            "coarse_branch_exist_logits must have shape [B,M], got "
            f"{tuple(coarse_logits.shape)}."
        )

    final_weight = float(
        _loss_option(
            config,
            "bspline_refiner_branch_existence_loss_weight",
            1.0,
        )
    )
    intermediate_weight = float(
        _loss_option(
            config,
            "bspline_refiner_branch_existence_intermediate_loss_weight",
            0.25,
        )
    )
    false_positive_weight = float(
        _loss_option(
            config,
            "bspline_refiner_branch_existence_false_positive_weight",
            2.0,
        )
    )
    true_positive_weight = float(
        _loss_option(
            config,
            "bspline_refiner_branch_existence_true_positive_weight",
            1.0,
        )
    )
    for name, value in (
        ("bspline_refiner_branch_existence_loss_weight", final_weight),
        (
            "bspline_refiner_branch_existence_intermediate_loss_weight",
            intermediate_weight,
        ),
        (
            "bspline_refiner_branch_existence_false_positive_weight",
            false_positive_weight,
        ),
        (
            "bspline_refiner_branch_existence_true_positive_weight",
            true_positive_weight,
        ),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}.")
    if false_positive_weight == 0.0 and true_positive_weight == 0.0:
        raise ValueError(
            "At least one refined branch-existence class weight must be > 0."
        )

    fixed_count = required_branch_count(resolve_artery_type(config))
    if num_branches < fixed_count:
        raise ValueError(
            "Refined branch-existence supervision has fewer slots than the "
            f"fixed main structure: got {num_branches}, need {fixed_count}."
        )
    candidate = candidate_mask.to(
        device=stage_logits.device,
        dtype=torch.bool,
    ).clone()
    candidate[:, :fixed_count] = False
    if supervision_mask is not None:
        if supervision_mask.shape != expected_shape:
            raise ValueError(
                "target_branch_existence_supervision_mask must match refined "
                "branch "
                f"logits, got {tuple(supervision_mask.shape)} and expected "
                f"{expected_shape}."
            )
        candidate &= supervision_mask.to(
            device=stage_logits.device,
            dtype=torch.bool,
        )
    target_float = target.to(device=stage_logits.device, dtype=stage_logits.dtype)
    target_bool = target_float > 0.5
    class_weights = torch.where(
        target_bool,
        stage_logits.new_tensor(true_positive_weight),
        stage_logits.new_tensor(false_positive_weight),
    )
    expanded_target = target_float.unsqueeze(1).expand_as(stage_logits)
    per_slot = F.binary_cross_entropy_with_logits(
        stage_logits,
        expanded_target,
        reduction="none",
    )
    candidate_float = candidate.to(dtype=stage_logits.dtype)
    weighted_candidate = candidate_float * class_weights
    expanded_weights = weighted_candidate.unsqueeze(1).expand_as(stage_logits)
    # Normalize by the unweighted number of candidates.  Consequently the FP
    # and TP class weights control absolute contribution as well as their
    # relative balance; e.g. FP weight 2 genuinely doubles an FP-only batch.
    stage_denominator = candidate_float.sum().clamp_min(1.0)
    stage_losses = (
        (per_slot * expanded_weights).sum(dim=(0, 2))
        / stage_denominator
    )

    final_loss = stage_losses[-1]
    intermediate_loss = (
        stage_losses[:-1].mean()
        if num_stages > 1
        else stage_logits.new_zeros(())
    )
    final_weighted = final_weight * final_loss
    intermediate_weighted = intermediate_weight * intermediate_loss
    total_weighted = final_weighted + intermediate_weighted

    false_positive_candidates = candidate & ~target_bool
    true_positive_candidates = candidate & target_bool
    final_probabilities = torch.sigmoid(stage_logits[:, -1])
    final_positive = final_probabilities >= 0.5
    remaining_false_positives = false_positive_candidates & final_positive
    removed_false_positives = false_positive_candidates & ~final_positive
    result = {
        "loss": total_weighted,
        "final_loss": final_loss,
        "final_weighted_loss": final_weighted,
        "intermediate_loss": intermediate_loss,
        "intermediate_weighted_loss": intermediate_weighted,
        "candidate_count": candidate.sum().to(dtype=stage_logits.dtype),
        "coarse_false_positive_count": false_positive_candidates.sum().to(
            dtype=stage_logits.dtype
        ),
        "coarse_true_positive_count": true_positive_candidates.sum().to(
            dtype=stage_logits.dtype
        ),
        "final_false_positive_count": remaining_false_positives.sum().to(
            dtype=stage_logits.dtype
        ),
        "removed_false_positive_count": removed_false_positives.sum().to(
            dtype=stage_logits.dtype
        ),
    }
    for stage_index, stage_loss in enumerate(stage_losses):
        result[f"stage_{stage_index + 1}_loss"] = stage_loss
    return result

def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(dtype=values.dtype)
    while expanded.dim() < values.dim():
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)

def _radius_refiner_mask_dice_loss(
    predicted_masks: torch.Tensor,
    target_images: torch.Tensor,
    view_mask: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Mean per-view soft Dice loss for the radius refiner's surface render."""

    if predicted_masks.dim() != 4:
        raise ValueError(
            "Radius-refiner rendered masks must have shape [B,V,H,W], got "
            f"{tuple(predicted_masks.shape)}."
        )
    if target_images.dim() == 5:
        target_masks = target_images[:, :, 0]
    elif target_images.dim() == 4:
        target_masks = target_images
    else:
        raise ValueError(
            "Radius-refiner target images must have shape [B,V,H,W] or "
            f"[B,V,C,H,W], got {tuple(target_images.shape)}."
        )
    target_masks = (target_masks > float(threshold)).to(
        device=predicted_masks.device,
        dtype=predicted_masks.dtype,
    )
    if target_masks.shape != predicted_masks.shape:
        raise ValueError(
            "Radius-refiner target and rendered masks must match, got "
            f"{tuple(target_masks.shape)} and {tuple(predicted_masks.shape)}."
        )
    valid_views = view_mask.to(
        device=predicted_masks.device,
        dtype=torch.bool,
    )
    predicted_flat = predicted_masks.flatten(start_dim=2)
    target_flat = target_masks.flatten(start_dim=2)
    intersection = (predicted_flat * target_flat).sum(dim=-1)
    denominator = predicted_flat.sum(dim=-1) + target_flat.sum(dim=-1)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    valid = valid_views.to(dtype=dice.dtype)
    return ((1.0 - dice) * valid).sum() / valid.sum().clamp_min(1.0)

def _project_target_centerline_grid(
    *,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    projector: DifferentiableVesselProjector,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project masked raw GT points to an align-corners raster grid.

    The differentiable projector stores its second detector coordinate in the
    Stage-2 image-row convention used by its rasterizer, so it is normalized
    directly here. No additional detector-y flip is applied.
    """

    target = batch.get("target_raw_vessel_mm")
    point_valid = batch.get("target_point_valid_mask")
    branch_exist = batch.get("target_branch_exist")
    view_mask = batch.get("view_mask")
    theta = batch.get("theta")
    phi = batch.get("phi")
    required = {
        "target_raw_vessel_mm": target,
        "target_point_valid_mask": point_valid,
        "target_branch_exist": branch_exist,
        "view_mask": view_mask,
        "theta": theta,
        "phi": phi,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "B-spline centerline-map supervision requires batch tensors: "
            + ", ".join(missing)
            + "."
        )
    assert target is not None
    assert point_valid is not None
    assert branch_exist is not None
    assert view_mask is not None
    assert theta is not None
    assert phi is not None

    device, dtype = reference.device, reference.dtype
    target = target.to(device=device, dtype=dtype)
    if target.ndim != 4 or int(target.shape[-1]) < 3:
        raise ValueError(
            "target_raw_vessel_mm must have shape [B,M,N,>=3], got "
            f"{tuple(target.shape)}."
        )
    batch_size, num_branches, num_points = target.shape[:3]
    point_valid = point_valid.to(device=device, dtype=torch.bool)
    branch_exist = branch_exist.to(device=device) > 0.5
    if point_valid.shape != (batch_size, num_branches, num_points):
        raise ValueError(
            "target_point_valid_mask must match target_raw_vessel_mm [B,M,N], "
            f"got {tuple(point_valid.shape)} and "
            f"{(batch_size, num_branches, num_points)}."
        )
    if branch_exist.shape != (batch_size, num_branches):
        raise ValueError(
            "target_branch_exist must match target_raw_vessel_mm [B,M], got "
            f"{tuple(branch_exist.shape)} and {(batch_size, num_branches)}."
        )

    theta = theta.to(device=device, dtype=dtype)
    phi = phi.to(device=device, dtype=dtype)
    view_mask = view_mask.to(device=device, dtype=torch.bool)
    if theta.ndim == 1 and batch_size == 1:
        theta = theta.unsqueeze(0)
    if phi.ndim == 1 and batch_size == 1:
        phi = phi.unsqueeze(0)
    if theta.ndim != 2 or phi.shape != theta.shape:
        raise ValueError(
            "theta and phi must have matching shape [B,V], got "
            f"{tuple(theta.shape)} and {tuple(phi.shape)}."
        )
    if int(theta.shape[0]) != batch_size:
        raise ValueError(
            "Camera batch size must match target_raw_vessel_mm, got "
            f"{int(theta.shape[0])} and {batch_size}."
        )
    num_views = int(theta.shape[1])
    if view_mask.shape != (batch_size, num_views):
        raise ValueError(
            "view_mask must have shape [B,V] matching theta/phi, got "
            f"{tuple(view_mask.shape)} and {(batch_size, num_views)}."
        )

    target_xyz_mm = target[..., :3]
    coordinate_frame = str(
        _model_option(
            config,
            "target_coordinate_frame",
            "projection_centered",
        )
    ).strip().lower()
    if coordinate_frame == "absolute_world":
        center = batch.get("projection_center_offset")
        if center is None:
            raise ValueError(
                "target_coordinate_frame='absolute_world' requires "
                "projection_center_offset for centerline-map supervision."
            )
        center = center.to(device=device, dtype=dtype)
        if center.shape != (batch_size, 3):
            raise ValueError(
                "projection_center_offset must have shape [B,3], got "
                f"{tuple(center.shape)}."
            )
        offset_valid = batch.get("projection_center_offset_valid")
        if offset_valid is not None:
            offset_valid = offset_valid.to(device=device, dtype=torch.bool)
            if offset_valid.shape not in {(batch_size,), (batch_size, 1)}:
                raise ValueError(
                    "projection_center_offset_valid must have shape [B] or "
                    f"[B,1], got {tuple(offset_valid.shape)}."
                )
            if not bool(offset_valid.reshape(batch_size).all().item()):
                raise ValueError(
                    "absolute-world centerline-map supervision received an "
                    "invalid projection_center_offset."
                )
        target_xyz_mm = target_xyz_mm - center[:, None, None, :]
    elif coordinate_frame != "projection_centered":
        raise ValueError(
            "target_coordinate_frame must be 'projection_centered' or "
            f"'absolute_world', got {coordinate_frame!r}."
        )

    coord_scale = float(
        _model_option(
            config,
            "projection_coord_scale_to_meter",
            0.001,
        )
    )
    if not math.isfinite(coord_scale) or coord_scale <= 0.0:
        raise ValueError(
            "projection_coord_scale_to_meter must be finite and > 0, got "
            f"{coord_scale}."
        )
    target_points_m = target_xyz_mm * coord_scale
    point_mask = point_valid & branch_exist.unsqueeze(-1)
    flat_point_mask = point_mask.reshape(batch_size, num_branches * num_points)
    flat_points = target_points_m.reshape(batch_size, num_branches * num_points, 3)
    safe_points = torch.where(
        flat_point_mask.unsqueeze(-1), flat_points, torch.zeros_like(flat_points)
    )

    # Padded views may carry NaN placeholder angles. They are excluded below,
    # but first need finite dummy cameras so projection itself remains finite.
    safe_theta = torch.where(view_mask, theta, torch.zeros_like(theta))
    safe_phi = torch.where(view_mask, phi, torch.zeros_like(phi))
    cameras = projector.prepare_cameras(
        safe_theta.reshape(-1), safe_phi.reshape(-1)
    )
    points_by_view = (
        safe_points[:, None]
        .expand(batch_size, num_views, num_branches * num_points, 3)
        .reshape(batch_size * num_views, num_branches * num_points, 3)
    )
    projected_px, geometry_valid = projector._project_points_batched(
        points_by_view,
        cameras,
        apply_crop=False,
    )
    projected_px = projected_px.reshape(
        batch_size, num_views, num_branches * num_points, 2
    )
    geometry_valid = geometry_valid.reshape(
        batch_size, num_views, num_branches * num_points
    )
    image_extent = float(projector.image_size - 1)
    if image_extent <= 0.0:
        raise ValueError(
            "Centerline-map projection requires projector.image_size >= 2."
        )
    projected_grid = projected_px * (2.0 / image_extent) - 1.0
    # Match _ProjectionEvidenceRefiner._project_points_to_grid exactly. Its
    # legacy detector-y validity check is performed after ``H - y`` (rather
    # than ``H - 1 - y``), so the accepted edge interval is y in [1, H]. The
    # normalized raster coordinate itself remains the unflipped projector y.
    refiner_raster_valid = (
        (projected_px[..., 0] >= 0.0)
        & (projected_px[..., 0] <= image_extent)
        & ((float(projector.image_size) - projected_px[..., 1]) >= 0.0)
        & ((float(projector.image_size) - projected_px[..., 1]) <= image_extent)
    )
    combined_valid = (
        geometry_valid
        & refiner_raster_valid
        & flat_point_mask[:, None, :]
        & view_mask[:, :, None]
    )
    return projected_grid, combined_valid

def _bspline_refiner_centerline_map_loss(
    *,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    projector: DifferentiableVesselProjector,
) -> dict[str, torch.Tensor]:
    """Supervise the image-derived centreline map once per outer loss call."""

    expected_camera = {
        "image_size": int(
            _model_option(config, "bspline_refiner_image_size", 256)
        ),
        "sid": float(
            _model_option(
                config,
                "bspline_refiner_sid",
                _model_option(config, "proj_loss_sid", 0.9),
            )
        ),
        "source_to_iso": float(
            _model_option(
                config,
                "bspline_refiner_source_to_iso",
                _model_option(config, "proj_loss_source_to_iso", 0.75),
            )
        ),
        "imager_pixel_spacing": float(
            _model_option(
                config,
                "bspline_refiner_imager_pixel_spacing",
                _model_option(
                    config,
                    "proj_loss_imager_pixel_spacing",
                    0.55,
                ),
            )
        ),
    }
    actual_camera = {
        "image_size": int(projector.image_size),
        "sid": float(projector.sid),
        "source_to_iso": float(projector.source_to_iso),
        "imager_pixel_spacing": float(projector.imager_pixel_spacing),
    }
    mismatches = []
    for name, expected in expected_camera.items():
        actual = actual_camera[name]
        matches = (
            actual == expected
            if name == "image_size"
            else math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
        )
        if not matches:
            mismatches.append(f"{name}: projector={actual}, refiner={expected}")
    if mismatches:
        raise ValueError(
            "Centerline-map target projection must use the exact B-spline "
            "refiner camera geometry; mismatches: "
            + "; ".join(mismatches)
            + "."
        )

    logits_raw = output.get("bspline_refinement_input_centerline_logits")
    if logits_raw is None:
        raise KeyError(
            "bspline_refiner_use_unexplained_centerline_evidence=true requires "
            "model output 'bspline_refinement_input_centerline_logits'."
        )
    if logits_raw.ndim == 5 and int(logits_raw.shape[2]) == 1:
        logits = logits_raw[:, :, 0]
    elif logits_raw.ndim == 4:
        logits = logits_raw
    else:
        raise ValueError(
            "bspline_refinement_input_centerline_logits must have shape "
            "[B,V,H,W] or [B,V,1,H,W], got "
            f"{tuple(logits_raw.shape)}."
        )
    batch_size, num_views, map_height, map_width = logits.shape
    configured_map_size = int(
        _model_option(
            config,
            "bspline_refiner_centerline_map_size",
            map_height,
        )
    )
    if map_height != configured_map_size or map_width != configured_map_size:
        raise ValueError(
            "Predicted centerline-map shape does not match "
            "bspline_refiner_centerline_map_size: got "
            f"{(map_height, map_width)}, expected "
            f"{(configured_map_size, configured_map_size)}."
        )

    projected_grid, target_valid = _project_target_centerline_grid(
        batch=batch,
        config=config,
        projector=projector,
        reference=logits,
    )
    if projected_grid.shape[:2] != (batch_size, num_views):
        raise ValueError(
            "Predicted centerline-map batch/view shape does not match target "
            f"cameras: got {(batch_size, num_views)} and "
            f"{tuple(projected_grid.shape[:2])}."
        )
    sigma_px = float(
        _model_option(
            config,
            "bspline_refiner_centerline_map_sigma_px",
            1.25,
        )
    )
    radius_px = int(
        _model_option(
            config,
            "bspline_refiner_centerline_map_radius_px",
            2,
        )
    )
    target_mode = str(
        _model_option(
            config,
            "bspline_refiner_centerline_probability_target_mode",
            "gaussian",
        )
    )
    input_images = batch.get("images")
    if input_images is None:
        raise KeyError(
            "Centreline-map supervision requires batch input images."
        )
    input_images = input_images.to(device=logits.device, dtype=logits.dtype)
    if input_images.ndim == 5:
        input_masks = input_images[:, :, 0]
    elif input_images.ndim == 4:
        input_masks = input_images
    else:
        raise ValueError(
            "Centreline-map supervision images must have shape [B,V,H,W] or "
            f"[B,V,C,H,W], got {tuple(input_images.shape)}."
        )
    with torch.no_grad():
        target_maps = render_centerline_target_maps_from_grid(
            projected_grid,
            target_valid,
            input_masks,
            map_size=(map_height, map_width),
            target_mode=target_mode,
            sigma_px=sigma_px,
            radius_px=radius_px,
            mask_threshold=float(
                _model_option(
                    config,
                    "bspline_refiner_centerline_probability_mask_threshold",
                    0.5,
                )
            ),
            affinity_gamma=float(
                _model_option(
                    config,
                    "bspline_refiner_centerline_probability_target_gamma",
                    2.0,
                )
            ),
        )
        target = target_maps.target

    view_mask = batch["view_mask"].to(device=logits.device, dtype=torch.bool)
    if view_mask.shape != (batch_size, num_views):
        raise ValueError(
            "view_mask must match predicted centerline-map [B,V], got "
            f"{tuple(view_mask.shape)} and {(batch_size, num_views)}."
        )
    valid_view_weights = view_mask.to(dtype=logits.dtype)
    valid_view_count = valid_view_weights.sum().clamp_min(1.0)
    per_view_bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    ).mean(dim=(-2, -1))
    per_view_bce = torch.where(
        view_mask, per_view_bce, torch.zeros_like(per_view_bce)
    )
    bce_loss = per_view_bce.sum() / valid_view_count

    probabilities = torch.sigmoid(logits)
    probabilities_flat = probabilities.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)
    intersection = (probabilities_flat * target_flat).sum(dim=-1)
    denominator = probabilities_flat.sum(dim=-1) + target_flat.sum(dim=-1)
    per_view_dice_loss = 1.0 - (
        (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    )
    per_view_dice_loss = torch.where(
        view_mask,
        per_view_dice_loss,
        torch.zeros_like(per_view_dice_loss),
    )
    dice_loss = per_view_dice_loss.sum() / valid_view_count

    bce_weight = float(
        _loss_option(config, "bspline_refiner_centerline_map_bce_weight", 0.25)
    )
    dice_weight = float(
        _loss_option(config, "bspline_refiner_centerline_map_dice_weight", 1.0)
    )
    total_weight = float(
        _loss_option(config, "bspline_refiner_centerline_map_loss_weight", 0.1)
    )
    for name, value in (
        ("bspline_refiner_centerline_map_bce_weight", bce_weight),
        ("bspline_refiner_centerline_map_dice_weight", dice_weight),
        ("bspline_refiner_centerline_map_loss_weight", total_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}.")
    if total_weight > 0.0 and bce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError(
            "A positive bspline_refiner_centerline_map_loss_weight requires "
            "a positive BCE or Dice component weight."
        )
    combined_loss = bce_weight * bce_loss + dice_weight * dice_loss
    weighted_loss = total_weight * combined_loss
    target_per_view_fraction = target.mean(dim=(-2, -1))
    target_per_view_fraction = torch.where(
        view_mask,
        target_per_view_fraction,
        torch.zeros_like(target_per_view_fraction),
    )
    target_positive_fraction = (
        target_per_view_fraction.sum() / valid_view_count
    )
    return {
        "loss": combined_loss,
        "weighted_loss": weighted_loss,
        "bce_loss": bce_loss,
        "dice_loss": dice_loss,
        "target_positive_fraction": target_positive_fraction.detach(),
        "effective_weight": logits.new_tensor(total_weight),
    }

def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Return a mask-normalized mean with non-negative sample weights."""
    if mask.shape != sample_weights.shape:
        raise ValueError(
            "mask and sample_weights must match, got "
            f"{tuple(mask.shape)} and {tuple(sample_weights.shape)}."
        )
    expanded_mask = mask.to(device=values.device, dtype=values.dtype)
    expanded_weights = sample_weights.to(device=values.device, dtype=values.dtype)
    while expanded_mask.dim() < values.dim():
        expanded_mask = expanded_mask.unsqueeze(-1)
        expanded_weights = expanded_weights.unsqueeze(-1)
    combined = (expanded_mask * expanded_weights).expand_as(values)
    safe_values = torch.where(
        torch.isfinite(values), values, torch.zeros_like(values)
    )
    return (safe_values * combined).sum() / combined.sum().clamp_min(1.0)

def _normalized_masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return a true weighted mean, including when all weights are below one."""
    if values.shape != mask.shape or values.shape != weights.shape:
        raise ValueError(
            "values, mask, and weights must have matching shapes, got "
            f"{tuple(values.shape)}, {tuple(mask.shape)}, and "
            f"{tuple(weights.shape)}."
        )
    valid = mask.to(device=values.device, dtype=torch.bool)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    safe_weights = torch.where(
        valid,
        weights.to(device=values.device, dtype=values.dtype),
        torch.zeros_like(values),
    )
    denominator = safe_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (safe_values * safe_weights).sum() / denominator

def _decoded_radius_profile_parts(
    predicted_radius_mm: torch.Tensor,
    target_radius_mm: torch.Tensor,
    point_mask: torch.Tensor,
    config: dict[str, Any],
    *,
    radius_scale_mm: float,
) -> dict[str, Any]:
    """Build the configurable point, change, and branch-calibration terms."""
    if predicted_radius_mm.shape != target_radius_mm.shape:
        raise ValueError(
            "Predicted and target radii must match, got "
            f"{tuple(predicted_radius_mm.shape)} and "
            f"{tuple(target_radius_mm.shape)}."
        )
    if point_mask.shape != predicted_radius_mm.shape:
        raise ValueError(
            "Radius point mask must match the radius profiles, got "
            f"{tuple(point_mask.shape)} and "
            f"{tuple(predicted_radius_mm.shape)}."
        )
    if predicted_radius_mm.dim() != 3:
        raise ValueError(
            "Decoded radius profiles must have shape [B,M,N], got "
            f"{tuple(predicted_radius_mm.shape)}."
        )

    scale = float(radius_scale_mm)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"radius_scale_mm must be finite and > 0, got {radius_scale_mm}."
        )
    zero = predicted_radius_mm.new_zeros(())
    valid = point_mask.to(device=predicted_radius_mm.device, dtype=torch.bool)
    absolute_error_mm = (predicted_radius_mm - target_radius_mm).abs()
    point_values = F.smooth_l1_loss(
        predicted_radius_mm / scale,
        target_radius_mm / scale,
        reduction="none",
    )
    unweighted_point_loss = _masked_mean(point_values, valid)

    raw_hard_weighting = _loss_option(
        config,
        "decoded_radius_hard_point_weighting",
        {},
    )
    if raw_hard_weighting is None:
        raw_hard_weighting = {}
    if not isinstance(raw_hard_weighting, dict):
        raise ValueError(
            "decoded_radius_hard_point_weighting must be a JSON object."
        )
    hard_weighting_enabled = bool(raw_hard_weighting.get("enabled", False))
    hard_threshold_mm = float(raw_hard_weighting.get("threshold_mm", 0.2))
    hard_temperature_mm = float(
        raw_hard_weighting.get("temperature_mm", 0.05)
    )
    minimum_weight = float(raw_hard_weighting.get("minimum_weight", 0.2))
    detach_weights = bool(raw_hard_weighting.get("detach_weights", True))
    if not math.isfinite(hard_threshold_mm) or hard_threshold_mm < 0.0:
        raise ValueError(
            "decoded_radius_hard_point_weighting.threshold_mm must be finite "
            f"and >= 0, got {hard_threshold_mm}."
        )
    if not math.isfinite(hard_temperature_mm) or hard_temperature_mm <= 0.0:
        raise ValueError(
            "decoded_radius_hard_point_weighting.temperature_mm must be finite "
            f"and > 0, got {hard_temperature_mm}."
        )
    if (
        not math.isfinite(minimum_weight)
        or minimum_weight < 0.0
        or minimum_weight > 1.0
    ):
        raise ValueError(
            "decoded_radius_hard_point_weighting.minimum_weight must be finite "
            f"and in [0,1], got {minimum_weight}."
        )
    if hard_weighting_enabled:
        hard_weights = minimum_weight + (1.0 - minimum_weight) * torch.sigmoid(
            (absolute_error_mm - hard_threshold_mm) / hard_temperature_mm
        )
        if detach_weights:
            hard_weights = hard_weights.detach()
        point_loss = _normalized_masked_weighted_mean(
            point_values,
            valid,
            hard_weights,
        )
    else:
        hard_weights = torch.ones_like(point_values)
        point_loss = unweighted_point_loss
    valid_float = valid.to(dtype=point_values.dtype)
    valid_count = valid_float.sum().clamp_min(1.0)
    hard_mean_weight = (
        (hard_weights.detach() * valid_float).sum() / valid_count
    )
    hard_exceedance_fraction = (
        (
            (absolute_error_mm.detach() > hard_threshold_mm).to(
                dtype=point_values.dtype
            )
            * valid_float
        ).sum()
        / valid_count
    )

    point_weight = _weight(config, "decoded_radius_point_loss_weight", 1.0)
    slope_weight = _weight(
        config,
        "decoded_radius_multiscale_slope_loss_weight",
        0.0,
    )
    branch_mean_weight = _weight(
        config,
        "decoded_radius_branch_mean_loss_weight",
        0.0,
    )
    branch_q90_weight = _weight(
        config,
        "decoded_radius_branch_q90_loss_weight",
        0.0,
    )
    for name, value in (
        ("decoded_radius_point_loss_weight", point_weight),
        ("decoded_radius_multiscale_slope_loss_weight", slope_weight),
        ("decoded_radius_branch_mean_loss_weight", branch_mean_weight),
        ("decoded_radius_branch_q90_loss_weight", branch_q90_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}.")

    slope_offset_losses: dict[int, torch.Tensor] = {}
    slope_loss = zero
    if slope_weight > 0.0:
        raw_offsets = _loss_option(
            config,
            "decoded_radius_slope_offsets",
            (1, 4, 8),
        )
        if not isinstance(raw_offsets, (list, tuple)) or not raw_offsets:
            raise ValueError(
                "decoded_radius_slope_offsets must be a non-empty list of "
                "positive integers."
            )
        offsets: list[int] = []
        for raw_offset in raw_offsets:
            if isinstance(raw_offset, bool) or int(raw_offset) != raw_offset:
                raise ValueError(
                    "decoded_radius_slope_offsets must contain only integers."
                )
            offset = int(raw_offset)
            if offset <= 0 or offset >= predicted_radius_mm.shape[-1]:
                raise ValueError(
                    "Each decoded_radius_slope_offsets value must be in "
                    f"[1,{predicted_radius_mm.shape[-1] - 1}], got {offset}."
                )
            offsets.append(offset)
        if len(set(offsets)) != len(offsets):
            raise ValueError("decoded_radius_slope_offsets must be unique.")
        for offset in offsets:
            pair_valid = valid[..., :-offset] & valid[..., offset:]
            # Do not compare across an internal padding hole.
            for interior in range(1, offset):
                pair_valid = pair_valid & valid[
                    ..., interior : interior + predicted_radius_mm.shape[-1] - offset
                ]
            predicted_change = (
                predicted_radius_mm[..., offset:]
                - predicted_radius_mm[..., :-offset]
            ) / scale
            target_change = (
                target_radius_mm[..., offset:]
                - target_radius_mm[..., :-offset]
            ) / scale
            slope_offset_losses[offset] = _masked_mean(
                F.smooth_l1_loss(
                    predicted_change,
                    target_change,
                    reduction="none",
                ),
                pair_valid,
            )
        slope_loss = torch.stack(tuple(slope_offset_losses.values())).mean()

    branch_count = valid.sum(dim=-1)
    valid_branches = branch_count > 0
    safe_count = branch_count.clamp_min(1).to(dtype=predicted_radius_mm.dtype)
    predicted_branch_mean = (
        torch.where(valid, predicted_radius_mm, torch.zeros_like(predicted_radius_mm))
        .sum(dim=-1)
        / safe_count
    )
    target_branch_mean = (
        torch.where(valid, target_radius_mm, torch.zeros_like(target_radius_mm))
        .sum(dim=-1)
        / safe_count
    )
    branch_mean_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_branch_mean / scale,
            target_branch_mean / scale,
            reduction="none",
        ),
        valid_branches,
    )

    quantile = float(
        _loss_option(config, "decoded_radius_branch_quantile", 0.9)
    )
    rank_temperature = float(
        _loss_option(
            config,
            "decoded_radius_soft_quantile_rank_temperature",
            0.02,
        )
    )
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError(
            "decoded_radius_branch_quantile must be finite and in (0,1), "
            f"got {quantile}."
        )
    if not math.isfinite(rank_temperature) or rank_temperature <= 0.0:
        raise ValueError(
            "decoded_radius_soft_quantile_rank_temperature must be finite and "
            f"> 0, got {rank_temperature}."
        )
    branch_q90_loss = zero
    if branch_q90_weight > 0.0 and bool(valid_branches.any().item()):
        quantile_losses: list[torch.Tensor] = []
        for batch_index, branch_index in valid_branches.nonzero(as_tuple=False):
            branch_valid = valid[batch_index, branch_index]
            predicted_sorted = torch.sort(
                predicted_radius_mm[batch_index, branch_index][branch_valid]
            ).values
            target_sorted = torch.sort(
                target_radius_mm[batch_index, branch_index][branch_valid]
            ).values
            ranks = (
                torch.arange(
                    predicted_sorted.numel(),
                    device=predicted_sorted.device,
                    dtype=predicted_sorted.dtype,
                )
                + 0.5
            ) / float(predicted_sorted.numel())
            rank_weights = torch.softmax(
                -0.5 * ((ranks - quantile) / rank_temperature).square(),
                dim=0,
            )
            predicted_quantile = (predicted_sorted * rank_weights).sum()
            target_quantile = (target_sorted * rank_weights).sum()
            quantile_losses.append(
                F.smooth_l1_loss(
                    predicted_quantile / scale,
                    target_quantile / scale,
                    reduction="none",
                )
            )
        branch_q90_loss = torch.stack(quantile_losses).mean()

    return {
        "point_loss": point_loss,
        "unweighted_point_loss": unweighted_point_loss,
        "point_weight": point_weight,
        "hard_weighting_enabled": hard_weighting_enabled,
        "hard_mean_weight": hard_mean_weight,
        "hard_exceedance_fraction": hard_exceedance_fraction,
        "hard_threshold_mm": point_values.new_tensor(hard_threshold_mm),
        "slope_loss": slope_loss,
        "slope_weight": slope_weight,
        "slope_offset_losses": slope_offset_losses,
        "branch_mean_loss": branch_mean_loss,
        "branch_mean_weight": branch_mean_weight,
        "branch_q90_loss": branch_q90_loss,
        "branch_q90_weight": branch_q90_weight,
    }

def ordered_terminal_masks(
    valid: torch.Tensor,
    *,
    region_fraction: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Return first, last, and terminal-region masks for ordered curves."""
    if valid.dim() < 1:
        raise ValueError("valid must have at least one dimension")
    fraction = float(region_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 0.5:
        raise ValueError(
            "decoded_xyz_terminal_region_fraction must be finite and in "
            f"(0,0.5], got {region_fraction}."
        )
    valid_mask = valid.to(dtype=torch.bool)
    forward_rank = torch.cumsum(valid_mask.to(dtype=torch.long), dim=-1)
    reverse_rank = torch.flip(
        torch.cumsum(
            torch.flip(valid_mask.to(dtype=torch.long), dims=(-1,)),
            dim=-1,
        ),
        dims=(-1,),
    )
    valid_count = valid_mask.sum(dim=-1)
    region_count = torch.ceil(
        valid_count.to(dtype=torch.float32) * fraction
    ).to(dtype=torch.long)
    region_count = region_count.clamp_min(1).clamp_max(valid.shape[-1])
    first = valid_mask & (forward_rank == 1)
    last = valid_mask & (reverse_rank == 1)
    terminal_region = valid_mask & (
        (forward_rank <= region_count.unsqueeze(-1))
        | (reverse_rank <= region_count.unsqueeze(-1))
    )
    return {
        "first": first,
        "last": last,
        "region": terminal_region,
    }

def _decoded_terminal_tangent_losses(
    predicted_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    offset_points: int,
    min_target_length_mm: float,
) -> dict[str, torch.Tensor]:
    """Compare ordered start/end tangents over a stable multi-point chord."""
    if predicted_xyz.shape != target_xyz.shape or predicted_xyz.shape[-1] != 3:
        raise ValueError(
            "predicted_xyz and target_xyz must have matching [...,N,3] shapes"
        )
    if valid.shape != predicted_xyz.shape[:-1]:
        raise ValueError(
            "valid must match the curve dimensions, got "
            f"{tuple(valid.shape)} and {tuple(predicted_xyz.shape[:-1])}."
        )
    offset = int(offset_points)
    minimum_length = float(min_target_length_mm)
    if offset < 1:
        raise ValueError(
            "decoded_terminal_tangent_offset_points must be >= 1, got "
            f"{offset_points}."
        )
    if not math.isfinite(minimum_length) or minimum_length <= 0.0:
        raise ValueError(
            "decoded_terminal_tangent_min_length_mm must be finite and > 0, "
            f"got {min_target_length_mm}."
        )

    valid_mask = valid.to(device=predicted_xyz.device, dtype=torch.bool)
    forward_rank = torch.cumsum(valid_mask.to(dtype=torch.long), dim=-1)
    reverse_rank = torch.flip(
        torch.cumsum(
            torch.flip(valid_mask.to(dtype=torch.long), dims=(-1,)),
            dim=-1,
        ),
        dims=(-1,),
    )

    def selected_point(selection: torch.Tensor) -> torch.Tensor:
        return torch.where(
            selection.unsqueeze(-1),
            predicted_xyz,
            torch.zeros_like(predicted_xyz),
        ).sum(dim=-2)

    def selected_target(selection: torch.Tensor) -> torch.Tensor:
        return torch.where(
            selection.unsqueeze(-1),
            target_xyz,
            torch.zeros_like(target_xyz),
        ).sum(dim=-2)

    start_anchor_mask = valid_mask & (forward_rank == 1)
    start_neighbor_mask = valid_mask & (forward_rank == offset + 1)
    end_anchor_mask = valid_mask & (reverse_rank == 1)
    end_neighbor_mask = valid_mask & (reverse_rank == offset + 1)

    predicted_start_chord = selected_point(start_neighbor_mask) - selected_point(
        start_anchor_mask
    )
    target_start_chord = selected_target(start_neighbor_mask) - selected_target(
        start_anchor_mask
    )
    predicted_end_chord = selected_point(end_anchor_mask) - selected_point(
        end_neighbor_mask
    )
    target_end_chord = selected_target(end_anchor_mask) - selected_target(
        end_neighbor_mask
    )

    def tangent_parts(
        predicted_chord: torch.Tensor,
        target_chord: torch.Tensor,
        anchor_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_length = torch.linalg.vector_norm(target_chord, dim=-1)
        endpoint_valid = (
            anchor_mask.any(dim=-1)
            & neighbor_mask.any(dim=-1)
            & torch.isfinite(predicted_chord).all(dim=-1)
            & torch.isfinite(target_chord).all(dim=-1)
            & (target_length > minimum_length)
        )
        if not bool(endpoint_valid.any().item()):
            zero = predicted_xyz.new_zeros(())
            return zero, zero, endpoint_valid
        predicted_tangent = F.normalize(predicted_chord, dim=-1, eps=1e-8)
        target_tangent = F.normalize(target_chord, dim=-1, eps=1e-8)
        cosine = torch.sum(
            predicted_tangent * target_tangent, dim=-1
        ).clamp(-1.0, 1.0)
        valid_cosine = cosine[endpoint_valid]
        return (
            (1.0 - valid_cosine).mean(),
            torch.rad2deg(torch.acos(valid_cosine)).mean(),
            endpoint_valid,
        )

    start_loss, start_angle, start_valid = tangent_parts(
        predicted_start_chord,
        target_start_chord,
        start_anchor_mask,
        start_neighbor_mask,
    )
    end_loss, end_angle, end_valid = tangent_parts(
        predicted_end_chord,
        target_end_chord,
        end_anchor_mask,
        end_neighbor_mask,
    )
    valid_losses = []
    if bool(start_valid.any().item()):
        valid_losses.append(start_loss)
    if bool(end_valid.any().item()):
        valid_losses.append(end_loss)
    combined_loss = (
        torch.stack(valid_losses).mean()
        if valid_losses
        else predicted_xyz.new_zeros(())
    )
    return {
        "loss": combined_loss,
        "start_loss": start_loss,
        "end_loss": end_loss,
        "start_angle_mae_deg": start_angle,
        "end_angle_mae_deg": end_angle,
        "start_valid_fraction": start_valid.to(
            dtype=predicted_xyz.dtype
        ).mean(),
        "end_valid_fraction": end_valid.to(dtype=predicted_xyz.dtype).mean(),
    }

def bend_recall_3d_schedule_factor(
    config: dict[str, Any], epoch: int | None
) -> float:
    """Return the independent 3D bend-recall ramp factor for a 1-based epoch."""
    if epoch is None or not bool(
        _loss_option(config, "bend_recall_3d_schedule_enabled", False)
    ):
        return 1.0
    start_epoch = int(
        _loss_option(config, "bend_recall_3d_schedule_start_epoch", 20)
    )
    end_epoch = int(
        _loss_option(config, "bend_recall_3d_schedule_end_epoch", 60)
    )
    start_factor = float(
        _loss_option(config, "bend_recall_3d_schedule_start_factor", 0.0)
    )
    end_factor = float(
        _loss_option(config, "bend_recall_3d_schedule_end_factor", 1.0)
    )
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError("3D bend-recall schedule epochs must be >= 0")
    if end_epoch <= start_epoch:
        raise ValueError(
            "bend_recall_3d_schedule_end_epoch must be greater than "
            "bend_recall_3d_schedule_start_epoch"
        )
    if (
        not math.isfinite(start_factor)
        or not math.isfinite(end_factor)
        or start_factor < 0.0
        or end_factor < 0.0
    ):
        raise ValueError(
            "3D bend-recall schedule factors must be finite and >= 0"
        )
    if int(epoch) <= start_epoch:
        return start_factor
    if int(epoch) >= end_epoch:
        return end_factor
    fraction = float(int(epoch) - start_epoch) / float(end_epoch - start_epoch)
    return (1.0 - fraction) * start_factor + fraction * end_factor

def bend_recall_3d_final_weight(config: dict[str, Any]) -> float:
    """Return the final scheduled weight used by all-loss checkpoint selection."""
    weight = _weight(config, "bend_recall_3d_loss_weight", 0.0)
    if weight < 0.0:
        raise ValueError(
            f"bend_recall_3d_loss_weight must be non-negative, got {weight}"
        )
    final_factor = (
        float(_loss_option(config, "bend_recall_3d_schedule_end_factor", 1.0))
        if bool(_loss_option(config, "bend_recall_3d_schedule_enabled", False))
        else 1.0
    )
    return weight * final_factor

def decoded_local_progress_schedule_factor(
    config: dict[str, Any], epoch: int | None
) -> float:
    """Return the decoded-centreline local-progress ramp factor."""
    if epoch is None or not bool(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_schedule_enabled",
            "local_progress_loss_schedule_enabled",
            False,
        )
    ):
        return 1.0
    start_epoch = int(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_schedule_start_epoch",
            "local_progress_loss_schedule_start_epoch",
            20,
        )
    )
    end_epoch = int(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_schedule_end_epoch",
            "local_progress_loss_schedule_end_epoch",
            60,
        )
    )
    start_factor = float(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_schedule_start_factor",
            "local_progress_loss_schedule_start_factor",
            0.0,
        )
    )
    end_factor = float(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_schedule_end_factor",
            "local_progress_loss_schedule_end_factor",
            1.0,
        )
    )
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError("Decoded local-progress schedule epochs must be >= 0")
    if end_epoch <= start_epoch:
        raise ValueError(
            "decoded_local_progress_loss_schedule_end_epoch must be greater "
            "than decoded_local_progress_loss_schedule_start_epoch"
        )
    if (
        not math.isfinite(start_factor)
        or not math.isfinite(end_factor)
        or start_factor < 0.0
        or end_factor < 0.0
    ):
        raise ValueError(
            "Decoded local-progress schedule factors must be finite and >= 0"
        )
    if int(epoch) <= start_epoch:
        return start_factor
    if int(epoch) >= end_epoch:
        return end_factor
    fraction = float(int(epoch) - start_epoch) / float(end_epoch - start_epoch)
    return (1.0 - fraction) * start_factor + fraction * end_factor

def decoded_local_progress_final_weight(config: dict[str, Any]) -> float:
    """Return the final decoded local-progress checkpoint-scoring weight."""
    weight = float(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_weight",
            "local_progress_loss_weight",
            0.0,
        )
    )
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "decoded_local_progress_loss_weight must be finite and >= 0, "
            f"got {weight}."
        )
    final_factor = (
        float(
            _loss_option_alias(
                config,
                "decoded_local_progress_loss_schedule_end_factor",
                "local_progress_loss_schedule_end_factor",
                1.0,
            )
        )
        if bool(
            _loss_option_alias(
                config,
                "decoded_local_progress_loss_schedule_enabled",
                "local_progress_loss_schedule_enabled",
                False,
            )
        )
        else 1.0
    )
    if not math.isfinite(final_factor) or final_factor < 0.0:
        raise ValueError(
            "decoded_local_progress_loss_schedule_end_factor must be finite "
            "and >= 0"
        )
    return weight * final_factor

def branch_length_schedule_factor(
    config: dict[str, Any], epoch: int | None
) -> float:
    """Return the decoded branch-length ramp factor for a one-based epoch."""
    if epoch is None or not bool(
        _loss_option(config, "branch_length_loss_schedule_enabled", False)
    ):
        return 1.0
    start_epoch = int(
        _loss_option(config, "branch_length_loss_schedule_start_epoch", 40)
    )
    end_epoch = int(
        _loss_option(config, "branch_length_loss_schedule_end_epoch", 100)
    )
    start_factor = float(
        _loss_option(config, "branch_length_loss_schedule_start_factor", 0.0)
    )
    end_factor = float(
        _loss_option(config, "branch_length_loss_schedule_end_factor", 1.0)
    )
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError("Branch-length schedule epochs must be >= 0")
    if end_epoch <= start_epoch:
        raise ValueError(
            "branch_length_loss_schedule_end_epoch must be greater than "
            "branch_length_loss_schedule_start_epoch"
        )
    if (
        not math.isfinite(start_factor)
        or not math.isfinite(end_factor)
        or start_factor < 0.0
        or end_factor < 0.0
    ):
        raise ValueError(
            "Branch-length schedule factors must be finite and >= 0"
        )
    if int(epoch) <= start_epoch:
        return start_factor
    if int(epoch) >= end_epoch:
        return end_factor
    fraction = float(int(epoch) - start_epoch) / float(end_epoch - start_epoch)
    return (1.0 - fraction) * start_factor + fraction * end_factor

def branch_length_final_weight(config: dict[str, Any]) -> float:
    """Return the final branch-length weight for all-loss checkpoint scoring."""
    weight = _weight(config, "branch_length_loss_weight", 0.0)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"branch_length_loss_weight must be finite and >= 0, got {weight}"
        )
    final_factor = (
        float(
            _loss_option(
                config,
                "branch_length_loss_schedule_end_factor",
                1.0,
            )
        )
        if bool(
            _loss_option(config, "branch_length_loss_schedule_enabled", False)
        )
        else 1.0
    )
    if not math.isfinite(final_factor) or final_factor < 0.0:
        raise ValueError(
            "branch_length_loss_schedule_end_factor must be finite and >= 0"
        )
    return weight * final_factor

def _decoded_bend_recall_3d(
    predicted_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    point_mask: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    zero = predicted_xyz.new_zeros(())
    values: dict[str, list[torch.Tensor]] = {
        "loss": [],
        "underbend_loss": [],
        "excess_loss": [],
        "direction_loss": [],
        "high_region_fraction": [],
        "valid_fraction": [],
    }
    for batch_index in range(int(predicted_xyz.shape[0])):
        for branch_index in range(int(predicted_xyz.shape[1])):
            valid = point_mask[batch_index, branch_index]
            if int(valid.sum().item()) < 3:
                continue
            parts = asymmetric_bend_recall_loss(
                pred_points=predicted_xyz[batch_index, branch_index],
                target_points=target_xyz[batch_index, branch_index],
                valid=valid,
                steps=_loss_option(
                    config, "bend_recall_3d_steps", (1, 2, 4, 8)
                ),
                high_fraction=float(
                    _loss_option(
                        config, "bend_recall_3d_high_fraction", 0.15
                    )
                ),
                low_fraction=float(
                    _loss_option(
                        config, "bend_recall_3d_low_fraction", 0.70
                    )
                ),
                min_recall_ratio=float(
                    _loss_option(
                        config, "bend_recall_3d_min_recall_ratio", 0.80
                    )
                ),
                jitter_tolerance_ratio=float(
                    _loss_option(
                        config,
                        "bend_recall_3d_jitter_tolerance_ratio",
                        0.10,
                    )
                ),
                direction_weight=float(
                    _loss_option(
                        config, "bend_recall_3d_direction_weight", 0.25
                    )
                ),
                smooth_kernel=int(
                    _loss_option(
                        config, "bend_recall_3d_smooth_kernel", 5
                    )
                ),
                high_region_dilation=int(
                    _loss_option(
                        config,
                        "bend_recall_3d_high_region_dilation",
                        5,
                    )
                ),
                reference_floor=float(
                    _loss_option(
                        config, "bend_recall_3d_reference_floor", 0.05
                    )
                ),
                min_target_step=float(
                    _loss_option(
                        config, "bend_recall_3d_min_step_mm", 0.05
                    )
                ),
            )
            for key, value in parts.items():
                values[key].append(value)
    return {
        key: torch.stack(items).mean() if items else zero
        for key, items in values.items()
    }

def compute_parametric_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    projector: DifferentiableVesselProjector | None = None,
    epoch: int | None = None,
    *,
    bspline_refiner_projector: DifferentiableVesselProjector | None = None,
    _include_bspline_refinement_supervision: bool = True,
) -> dict[str, torch.Tensor]:
    radius_prediction_mode = radius_prediction_mode_from_config(config)
    centerline_prediction_mode = centerline_prediction_mode_from_config(config)
    # Once optional existence refinement is enabled, the ordinary branch loss
    # retains its historical coarse-head meaning.  The removal-only refiner has
    # a separate candidate-restricted objective below.
    coarse_branch_exist_logits = output.get(
        "coarse_branch_exist_logits",
        output["branch_exist_logits"],
    )
    branch_exist, branch_exist_loss, _ = (
        _branch_existence_supervision(
            coarse_branch_exist_logits,
            batch["target_branch_exist"],
            config,
            batch.get("target_branch_existence_supervision_mask"),
        )
    )
    branch_mask = branch_exist > 0.5
    side_mask = branch_mask.clone()
    side_mask[:, 0] = False

    output_centerline_parameters = output.get(
        "centerline_parameters_mm", output.get("centerline_control_points_mm")
    )
    target_centerline_parameters_raw = batch.get(
        "target_centerline_parameters_mm",
        batch.get("target_centerline_control_points_mm"),
    )
    if output_centerline_parameters is None or target_centerline_parameters_raw is None:
        raise KeyError(
            "Centreline parameter supervision requires generic centerline parameter "
            "keys or the legacy B-spline control-point keys"
        )
    target_centerline_parameters = target_centerline_parameters_raw.to(
        output_centerline_parameters.device
    )
    centerline_parameter_scale = max(
        _weight(
            config,
            (
                "landmark_scale_mm"
                if centerline_prediction_mode == "adaptive_landmarks"
                else "control_point_scale_mm"
            ),
            100.0,
        ),
        1e-6,
    )
    centerline_loss_type_key = (
        "centerline_landmark_loss_type"
        if centerline_prediction_mode == "adaptive_landmarks"
        else "centerline_control_loss_type"
    )
    centerline_parameter_error = elementwise_regression_loss(
        output_centerline_parameters / centerline_parameter_scale,
        target_centerline_parameters / centerline_parameter_scale,
        _loss_option(config, centerline_loss_type_key, "mse"),
        option_name=centerline_loss_type_key,
    )
    terminal_parameter_weight = _weight(
        config, "centerline_terminal_parameter_weight", 1.0
    )
    if (
        not math.isfinite(terminal_parameter_weight)
        or terminal_parameter_weight < 1.0
    ):
        raise ValueError(
            "centerline_terminal_parameter_weight must be finite and >= 1, "
            f"got {terminal_parameter_weight}."
        )
    parameter_mask = branch_mask.unsqueeze(-1).expand(
        centerline_parameter_error.shape[:-1]
    )
    parameter_weights = torch.ones_like(
        centerline_parameter_error[..., 0]
    )
    parameter_weights[..., 0] = terminal_parameter_weight
    parameter_weights[..., -1] = terminal_parameter_weight
    centerline_parameter_loss = _masked_weighted_mean(
        centerline_parameter_error,
        parameter_mask,
        parameter_weights,
    )
    centerline_terminal_parameter_mean_weight = (
        (parameter_weights * parameter_mask.to(parameter_weights.dtype)).sum()
        / parameter_mask.to(parameter_weights.dtype).sum().clamp_min(1.0)
    ).detach()

    if radius_prediction_mode == "parametric":
        target_radius_coefficients = batch[
            "target_radius_baseline_coefficients_log_mm"
        ].to(output["radius_baseline_coefficients_log_mm"].device)
        radius_coefficient_loss = _masked_mean(
            F.smooth_l1_loss(
                output["radius_baseline_coefficients_log_mm"],
                target_radius_coefficients,
                reduction="none",
            ),
            branch_mask,
        )

        target_lesion_exist = batch["target_lesion_exist"].to(
            output["lesion_exist_logits"].device
        )
        if output["lesion_exist_logits"].shape[-1] > 0:
            lesion_exist_raw = F.binary_cross_entropy_with_logits(
                output["lesion_exist_logits"], target_lesion_exist, reduction="none"
            )
            lesion_exist_loss = _masked_mean(lesion_exist_raw, branch_mask)
            lesion_mask = branch_mask.unsqueeze(-1) & (target_lesion_exist > 0.5)
            target_lesion_geometry = batch["target_lesion_geometry"].to(
                output["lesion_geometry"].device
            )
            lesion_geometry_loss = _masked_mean(
                F.smooth_l1_loss(
                    output["lesion_geometry"],
                    target_lesion_geometry,
                    reduction="none",
                ),
                lesion_mask,
            )
        else:
            lesion_exist_loss = branch_exist_loss.new_zeros(())
            lesion_geometry_loss = branch_exist_loss.new_zeros(())
    else:
        radius_coefficient_loss = branch_exist_loss.new_zeros(())
        lesion_exist_loss = branch_exist_loss.new_zeros(())
        lesion_geometry_loss = branch_exist_loss.new_zeros(())

    attachment_weight = _weight(config, "attachment_loss_weight", 1.0)
    if "attachment_logits" in output:
        attachment_target = batch["target_attachment_index"].to(
            output["attachment_logits"].device, dtype=torch.long
        )
        valid_attachment = side_mask & (attachment_target >= 0)
        if bool(valid_attachment.any()):
            attachment_loss = F.cross_entropy(
                output["attachment_logits"][valid_attachment],
                attachment_target[valid_attachment],
            )
        else:
            attachment_loss = branch_exist_loss.new_zeros(())
    else:
        attachment_loss = branch_exist_loss.new_zeros(())
        if attachment_weight != 0.0:
            raise ValueError(
                "attachment_loss_weight must be 0 when the decoder does not "
                "predict attachment_logits (for example absolute_parallel)."
            )

    target_key = str(config.get("reconstruction_target", "raw")).strip().lower()
    if target_key not in {"raw", "parametric"}:
        raise ValueError("reconstruction_target must be 'raw' or 'parametric'")
    xyz_target_vessel = batch[
        "target_raw_vessel_mm" if target_key == "raw" else "target_reconstructed_vessel_mm"
    ].to(output["decoded_vessel_mm"].device)
    radius_target_vessel = batch[
        "target_raw_vessel_mm"
        if radius_prediction_mode == "raw"
        else "target_reconstructed_vessel_mm"
    ].to(output["decoded_vessel_mm"].device)
    point_valid = batch["target_point_valid_mask"].to(
        output["decoded_vessel_mm"].device, dtype=torch.bool
    )
    point_mask = branch_mask.unsqueeze(-1) & point_valid
    terminal_region_fraction = float(
        _loss_option(config, "decoded_xyz_terminal_region_fraction", 0.05)
    )
    terminal_region_weight = _weight(
        config, "decoded_xyz_terminal_region_weight", 1.0
    )
    terminal_position_weight = _weight(
        config, "decoded_terminal_position_loss_weight", 0.0
    )
    terminal_tangent_weight = _weight(
        config, "decoded_terminal_tangent_loss_weight", 0.0
    )
    for name, value, minimum in (
        (
            "decoded_xyz_terminal_region_weight",
            terminal_region_weight,
            1.0,
        ),
        (
            "decoded_terminal_position_loss_weight",
            terminal_position_weight,
            0.0,
        ),
        (
            "decoded_terminal_tangent_loss_weight",
            terminal_tangent_weight,
            0.0,
        ),
    ):
        if not math.isfinite(value) or value < minimum:
            raise ValueError(
                f"{name} must be finite and >= {minimum:g}, got {value}."
            )
    terminal_masks = ordered_terminal_masks(
        point_mask,
        region_fraction=terminal_region_fraction,
    )
    terminal_region_point_weights = torch.ones_like(
        point_mask, dtype=output["decoded_vessel_mm"].dtype
    )
    terminal_region_point_weights = torch.where(
        terminal_masks["region"],
        terminal_region_point_weights.new_full(
            (), terminal_region_weight
        ),
        terminal_region_point_weights,
    )
    terminal_valid_float = point_mask.to(
        dtype=terminal_region_point_weights.dtype
    )
    terminal_region_mean_weight = (
        (terminal_region_point_weights * terminal_valid_float).sum()
        / terminal_valid_float.sum().clamp_min(1.0)
    ).detach()
    xyz_scale = max(_weight(config, "xyz_scale_mm", 100.0), 1e-6)
    radius_scale = max(_weight(config, "radius_scale_mm", 2.0), 1e-6)
    side_relative_weight = _weight(
        config, "side_relative_xyz_loss_weight", 0.0
    )
    if branch_exist.shape[1] <= 1:
        side_relative_xyz_loss = output["decoded_vessel_mm"].new_zeros(())
    elif "side_branch_relative_code_mm" not in output:
        if side_relative_weight != 0.0:
            raise KeyError(
                "side_relative_xyz_loss_weight is non-zero, but the model output "
                "does not contain side_branch_relative_code_mm."
            )
        side_relative_xyz_loss = output["decoded_vessel_mm"].new_zeros(())
    else:
        predicted_side_relative_xyz = output[
            "side_branch_relative_code_mm"
        ][..., :3]
        target_side_relative_xyz = (
            xyz_target_vessel[:, 1:, :, :3]
            - xyz_target_vessel[:, 1:, :1, :3]
        )
        if predicted_side_relative_xyz.shape != target_side_relative_xyz.shape:
            raise ValueError(
                "side_branch_relative_code_mm XYZ must match target side offsets, got "
                f"{tuple(predicted_side_relative_xyz.shape)} and "
                f"{tuple(target_side_relative_xyz.shape)}."
            )
        side_point_mask = side_mask[:, 1:].unsqueeze(-1) & point_valid[:, 1:]
        side_relative_xyz_loss = _masked_mean(
            F.smooth_l1_loss(
                predicted_side_relative_xyz / xyz_scale,
                target_side_relative_xyz / xyz_scale,
                reduction="none",
            ),
            side_point_mask,
        )
    xyz_coordinate_loss = elementwise_regression_loss(
        output["decoded_vessel_mm"][..., :3] / xyz_scale,
        xyz_target_vessel[..., :3] / xyz_scale,
        _loss_option(config, "decoded_xyz_loss_type", "mse"),
        option_name="decoded_xyz_loss_type",
    )
    xyz_mountain_weight_boost = _weight(
        config, "decoded_xyz_mountain_weight_boost", 4.0
    )
    xyz_mountain_parts = detached_mountain_weighted_mean(
        xyz_coordinate_loss.mean(dim=-1),
        error_profile=torch.linalg.vector_norm(
            output["decoded_vessel_mm"][..., :3]
            - xyz_target_vessel[..., :3],
            dim=-1,
        ),
        valid=point_mask,
        weight_boost=xyz_mountain_weight_boost,
        detector=str(
            _loss_option(
                config,
                "decoded_xyz_mountain_detector",
                "local_prominence",
            )
        ),
        width_pairs=_loss_option(
            config,
            "decoded_xyz_mountain_width_pairs",
            ((8, 8), (8, 12), (12, 8), (16, 16), (24, 24)),
        ),
        scale_fractions=_loss_option(
            config,
            "decoded_xyz_mountain_scale_fractions",
            (0.02, 0.04, 0.06, 0.08, 0.12, 0.16),
        ),
        min_prominence=_weight(
            config, "decoded_xyz_mountain_min_prominence_mm", 1.0
        ),
        temperature=_weight(
            config, "decoded_xyz_mountain_temperature_mm", 0.25
        ),
        smooth_kernel=int(
            _loss_option(config, "decoded_xyz_mountain_smooth_kernel", 5)
        ),
        edge_weight=_weight(
            config, "decoded_xyz_mountain_edge_weight", 0.25
        ),
        min_slope_fraction=_weight(
            config, "decoded_xyz_mountain_min_slope_fraction", 0.65
        ),
        min_area_fraction=_weight(
            config, "decoded_xyz_mountain_min_area_fraction", 0.25
        ),
        min_region_width=int(
            _loss_option(config, "decoded_xyz_mountain_min_region_width", 7)
        ),
        min_scale_persistence=int(
            _loss_option(
                config,
                "decoded_xyz_mountain_min_scale_persistence",
                2,
            )
        ),
    )
    xyz_combined_point_weights = terminal_region_point_weights * (
        1.0
        + xyz_mountain_weight_boost
        * xyz_mountain_parts["region_mask"]
    )
    xyz_unthresholded_loss = _masked_weighted_mean(
        xyz_coordinate_loss.mean(dim=-1),
        point_mask,
        xyz_combined_point_weights,
    )
    xyz_loss = xyz_unthresholded_loss
    radius_unthresholded_loss = _masked_mean(
        F.smooth_l1_loss(
            output["decoded_vessel_mm"][..., 3] / radius_scale,
            radius_target_vessel[..., 3] / radius_scale,
            reduction="none",
        ),
        point_mask,
    )
    radius_loss = radius_unthresholded_loss

    xyz_error_mm = torch.linalg.vector_norm(
        output["decoded_vessel_mm"][..., :3]
        - xyz_target_vessel[..., :3],
        dim=-1,
    )
    xyz_threshold_weights = xyz_combined_point_weights
    xyz_threshold_parts = configured_error_thresholded_mean(
        xyz_error_mm,
        valid=point_mask,
        config=config,
        component_name="decoded_xyz_loss",
        aliases=("xyz_loss",),
        epoch=epoch,
        sample_weights=xyz_threshold_weights,
    )
    if xyz_threshold_parts is not None:
        xyz_loss = xyz_threshold_parts["loss"]
        xyz_final_threshold_parts = configured_error_thresholded_mean(
            xyz_error_mm,
            valid=point_mask,
            config=config,
            component_name="decoded_xyz_loss",
            aliases=("xyz_loss",),
            epoch=int(config.get("num_epochs", epoch or 1)),
            sample_weights=xyz_threshold_weights,
        )
        assert xyz_final_threshold_parts is not None
    else:
        xyz_final_threshold_parts = None

    terminal_position_coordinate_mse = (
        (
            output["decoded_vessel_mm"][..., :3]
            - xyz_target_vessel[..., :3]
        )
        / xyz_scale
    ).square()
    terminal_start_position_mse_loss = _masked_mean(
        terminal_position_coordinate_mse,
        terminal_masks["first"],
    )
    terminal_end_position_mse_loss = _masked_mean(
        terminal_position_coordinate_mse,
        terminal_masks["last"],
    )
    terminal_position_mse_loss = _masked_mean(
        terminal_position_coordinate_mse,
        terminal_masks["first"] | terminal_masks["last"],
    )
    terminal_tangent_parts = _decoded_terminal_tangent_losses(
        predicted_xyz=output["decoded_vessel_mm"][..., :3],
        target_xyz=xyz_target_vessel[..., :3],
        valid=point_mask,
        offset_points=int(
            _loss_option(
                config,
                "decoded_terminal_tangent_offset_points",
                4,
            )
        ),
        min_target_length_mm=float(
            _loss_option(
                config,
                "decoded_terminal_tangent_min_length_mm",
                0.5,
            )
        ),
    )
    terminal_position_weighted_loss = (
        terminal_position_weight * terminal_position_mse_loss
    )
    terminal_tangent_weighted_loss = (
        terminal_tangent_weight * terminal_tangent_parts["loss"]
    )

    branch_length_base_weight = _weight(
        config, "branch_length_loss_weight", 0.0
    )
    if (
        not math.isfinite(branch_length_base_weight)
        or branch_length_base_weight < 0.0
    ):
        raise ValueError(
            "branch_length_loss_weight must be finite and >= 0, got "
            f"{branch_length_base_weight}."
        )
    branch_length_effective_weight = (
        branch_length_base_weight
        * branch_length_schedule_factor(config, epoch)
    )
    if branch_length_base_weight > 0.0:
        branch_length_parts = decoded_branch_length_loss(
            predicted_xyz=output["decoded_vessel_mm"][..., :3],
            target_xyz=xyz_target_vessel[..., :3],
            valid=point_mask,
            global_component_weight=_weight(
                config, "branch_length_global_component_weight", 1.0
            ),
            local_deficit_component_weight=_weight(
                config,
                "branch_length_local_deficit_component_weight",
                1.0,
            ),
            smooth_excess_component_weight=_weight(
                config,
                "branch_length_smooth_excess_component_weight",
                1.0,
            ),
            local_window_sizes=_loss_option(
                config,
                "branch_length_local_window_sizes",
                (4, 8, 16),
            ),
            high_curvature_fraction=_weight(
                config, "branch_length_high_curvature_fraction", 0.15
            ),
            low_curvature_fraction=_weight(
                config, "branch_length_low_curvature_fraction", 0.70
            ),
            deficit_tolerance_ratio=_weight(
                config, "branch_length_deficit_tolerance_ratio", 0.02
            ),
            excess_tolerance_ratio=_weight(
                config, "branch_length_excess_tolerance_ratio", 0.02
            ),
            smooth_l1_beta=_weight(
                config, "branch_length_smooth_l1_beta", 0.05
            ),
            curvature_smooth_kernel=int(
                _loss_option(
                    config,
                    "branch_length_curvature_smooth_kernel",
                    5,
                )
            ),
            eps=_weight(config, "branch_length_eps", 1e-6),
        )
    else:
        branch_length_zero = xyz_loss.new_zeros(())
        branch_length_parts = {
            "branch_length_loss": branch_length_zero,
            "branch_length_global_loss": branch_length_zero,
            "branch_length_local_deficit_loss": branch_length_zero,
            "branch_length_smooth_excess_loss": branch_length_zero,
            "branch_length_high_region_fraction": branch_length_zero,
            "branch_length_smooth_region_fraction": branch_length_zero,
            "branch_length_valid_fraction": branch_length_zero,
            "arc_length_abs_error_mm": branch_length_zero,
            "arc_length_rel_error": branch_length_zero,
        }
    branch_length_weighted_loss = (
        branch_length_effective_weight
        * branch_length_parts["branch_length_loss"]
    )

    radius_error_mm = (
        output["decoded_vessel_mm"][..., 3]
        - radius_target_vessel[..., 3]
    ).abs()
    radius_profile_parts = _decoded_radius_profile_parts(
        output["decoded_vessel_mm"][..., 3],
        radius_target_vessel[..., 3],
        point_mask,
        config,
        radius_scale_mm=radius_scale,
    )
    radius_unthresholded_loss = radius_profile_parts[
        "unweighted_point_loss"
    ]
    radius_threshold_parts = configured_error_thresholded_mean(
        radius_error_mm,
        valid=point_mask,
        config=config,
        component_name="decoded_radius_loss",
        aliases=("radius_loss",),
        epoch=epoch,
    )
    if (
        radius_threshold_parts is not None
        and radius_profile_parts["hard_weighting_enabled"]
    ):
        raise ValueError(
            "decoded_radius_hard_point_weighting.enabled and legacy "
            "error_thresholding.decoded_radius_loss.enabled are mutually "
            "exclusive. Disable one radius hard-point scheme."
        )
    if radius_threshold_parts is not None:
        radius_point_loss = radius_threshold_parts["loss"]
        radius_final_threshold_parts = configured_error_thresholded_mean(
            radius_error_mm,
            valid=point_mask,
            config=config,
            component_name="decoded_radius_loss",
            aliases=("radius_loss",),
            epoch=int(config.get("num_epochs", epoch or 1)),
        )
        assert radius_final_threshold_parts is not None
    else:
        radius_point_loss = radius_profile_parts["point_loss"]
        radius_final_threshold_parts = None

    radius_point_weighted_loss = (
        radius_profile_parts["point_weight"] * radius_point_loss
    )
    radius_slope_weighted_loss = (
        radius_profile_parts["slope_weight"]
        * radius_profile_parts["slope_loss"]
    )
    radius_branch_mean_weighted_loss = (
        radius_profile_parts["branch_mean_weight"]
        * radius_profile_parts["branch_mean_loss"]
    )
    radius_branch_q90_weighted_loss = (
        radius_profile_parts["branch_q90_weight"]
        * radius_profile_parts["branch_q90_loss"]
    )
    radius_loss = (
        radius_point_weighted_loss
        + radius_slope_weighted_loss
        + radius_branch_mean_weighted_loss
        + radius_branch_q90_weighted_loss
    )
    radius_unthresholded_composite_loss = (
        radius_profile_parts["point_weight"] * radius_unthresholded_loss
        + radius_slope_weighted_loss
        + radius_branch_mean_weighted_loss
        + radius_branch_q90_weighted_loss
    )
    if radius_final_threshold_parts is not None:
        radius_final_threshold_composite_loss = (
            radius_profile_parts["point_weight"]
            * radius_final_threshold_parts["loss"]
            + radius_slope_weighted_loss
            + radius_branch_mean_weighted_loss
            + radius_branch_q90_weighted_loss
        )
    else:
        radius_final_threshold_composite_loss = None

    bend_recall_3d_weight = _weight(
        config, "bend_recall_3d_loss_weight", 0.0
    )
    if bend_recall_3d_weight < 0.0:
        raise ValueError(
            "bend_recall_3d_loss_weight must be non-negative, got "
            f"{bend_recall_3d_weight}"
        )
    if bend_recall_3d_weight > 0.0:
        bend_recall_3d_parts = _decoded_bend_recall_3d(
            predicted_xyz=output["decoded_vessel_mm"][..., :3],
            target_xyz=xyz_target_vessel[..., :3],
            point_mask=point_mask,
            config=config,
        )
    else:
        zero = xyz_loss.new_zeros(())
        bend_recall_3d_parts = {
            "loss": zero,
            "underbend_loss": zero,
            "excess_loss": zero,
            "direction_loss": zero,
            "high_region_fraction": zero,
            "valid_fraction": zero,
        }
    bend_recall_3d_effective_weight = (
        bend_recall_3d_weight
        * bend_recall_3d_schedule_factor(config, epoch)
    )
    bend_recall_3d_weighted_loss = (
        bend_recall_3d_effective_weight
        * bend_recall_3d_parts["loss"]
    )

    local_progress_weight = float(
        _loss_option_alias(
            config,
            "decoded_local_progress_loss_weight",
            "local_progress_loss_weight",
            0.0,
        )
    )
    if not math.isfinite(local_progress_weight) or local_progress_weight < 0.0:
        raise ValueError(
            "decoded_local_progress_loss_weight must be finite and >= 0, "
            f"got {local_progress_weight}."
        )
    if local_progress_weight > 0.0:
        local_progress_parts = decoded_local_progress_loss(
            predicted_xyz=output["decoded_vessel_mm"][..., :3],
            target_xyz=xyz_target_vessel[..., :3],
            valid=point_mask,
            margin=float(
                _loss_option_alias(
                    config,
                    "decoded_local_progress_margin",
                    "local_progress_margin",
                    0.0,
                )
            ),
            min_target_step_fraction=float(
                _loss_option_alias(
                    config,
                    "decoded_local_progress_min_target_step_fraction",
                    "local_progress_min_target_step_fraction",
                    0.25,
                )
            ),
            eps=float(
                _loss_option_alias(
                    config,
                    "decoded_local_progress_eps",
                    "local_progress_eps",
                    1e-6,
                )
            ),
        )
    else:
        zero = xyz_loss.new_zeros(())
        local_progress_parts = {
            "loss": zero,
            "valid_fraction": zero,
            "backward_fraction": zero,
            "insufficient_fraction": zero,
            "mean_ratio": zero,
            "p05_ratio": zero,
        }
    local_progress_effective_weight = (
        local_progress_weight
        * decoded_local_progress_schedule_factor(config, epoch)
    )
    local_progress_weighted_loss = (
        local_progress_effective_weight * local_progress_parts["loss"]
    )

    centerline_parameter_weight = _weight(
        config,
        (
            "centerline_landmark_loss_weight"
            if centerline_prediction_mode == "adaptive_landmarks"
            else "centerline_control_loss_weight"
        ),
        1.0,
    )
    total = (
        _weight(config, "branch_exist_loss_weight", 1.0) * branch_exist_loss
        + centerline_parameter_weight * centerline_parameter_loss
        + _weight(config, "radius_coefficient_loss_weight", 1.0) * radius_coefficient_loss
        + _weight(config, "lesion_exist_loss_weight", 1.0) * lesion_exist_loss
        + _weight(config, "lesion_geometry_loss_weight", 1.0) * lesion_geometry_loss
        + attachment_weight * attachment_loss
        + side_relative_weight * side_relative_xyz_loss
        + _weight(config, "decoded_xyz_loss_weight", 1.0) * xyz_loss
        + _weight(config, "decoded_radius_loss_weight", 1.0) * radius_loss
        + bend_recall_3d_weighted_loss
        + local_progress_weighted_loss
        + terminal_position_weighted_loss
        + terminal_tangent_weighted_loss
        + branch_length_weighted_loss
    )
    if bool(config.get("train_radius_head_only", False)):
        # This mode is an explicit radius-only objective. Frozen geometry and
        # classification terms remain useful diagnostics, but must not affect
        # checkpoint selection or send indirect gradients into an LCA radius
        # head through the existing parent-branch XYZR conditioner.
        total = _weight(config, "decoded_radius_loss_weight", 1.0) * radius_loss
    projection_losses: dict[str, torch.Tensor] = {}
    projection_enabled = bool(config.get("enable_projection_2d_loss", False))
    if projection_enabled:
        if bool(config.get("train_radius_head_only", False)):
            raise ValueError(
                "train_radius_head_only=true requires "
                "enable_projection_2d_loss=false."
            )
        if projector is None:
            raise ValueError("enable_projection_2d_loss=true requires a projector")
    if projection_enabled and projection_schedule_factor(config, epoch) != 0.0:
        projection_losses = compute_centerline_projection_losses(
            output=output,
            batch=batch,
            config=config,
            projector=projector,
            epoch=epoch,
        )
        total = total + projection_losses["projection_2d_loss"]
    losses = {
        "loss": total,
        "branch_exist_loss": branch_exist_loss,
        "radius_coefficient_loss": radius_coefficient_loss,
        "lesion_exist_loss": lesion_exist_loss,
        "lesion_geometry_loss": lesion_geometry_loss,
        "attachment_loss": attachment_loss,
        "side_relative_xyz_loss": side_relative_xyz_loss,
        "decoded_xyz_loss": xyz_loss,
        "decoded_xyz_unweighted_loss": xyz_mountain_parts[
            "unweighted_loss"
        ],
        "decoded_xyz_mountain_region_fraction": xyz_mountain_parts[
            "region_fraction"
        ],
        "decoded_xyz_mountain_peak_activation": xyz_mountain_parts[
            "peak_activation"
        ],
        "decoded_xyz_mountain_mean_weight": xyz_mountain_parts[
            "mean_weight"
        ],
        "decoded_xyz_mountain_weight_boost": xyz_loss.new_tensor(
            xyz_mountain_weight_boost
        ),
        "centerline_terminal_parameter_mean_weight": (
            centerline_terminal_parameter_mean_weight
        ),
        "decoded_xyz_terminal_region_fraction": (
            terminal_masks["region"].to(dtype=xyz_loss.dtype).sum()
            / point_mask.to(dtype=xyz_loss.dtype).sum().clamp_min(1.0)
        ),
        "decoded_xyz_terminal_region_mean_weight": (
            terminal_region_mean_weight
        ),
        "decoded_terminal_position_mse_loss": terminal_position_mse_loss,
        "decoded_terminal_start_position_mse_loss": (
            terminal_start_position_mse_loss
        ),
        "decoded_terminal_end_position_mse_loss": (
            terminal_end_position_mse_loss
        ),
        "decoded_terminal_position_weighted_loss": (
            terminal_position_weighted_loss
        ),
        "decoded_terminal_position_effective_weight": xyz_loss.new_tensor(
            terminal_position_weight
        ),
        "decoded_terminal_tangent_loss": terminal_tangent_parts["loss"],
        "decoded_terminal_start_tangent_loss": terminal_tangent_parts[
            "start_loss"
        ],
        "decoded_terminal_end_tangent_loss": terminal_tangent_parts[
            "end_loss"
        ],
        "decoded_terminal_tangent_weighted_loss": (
            terminal_tangent_weighted_loss
        ),
        "decoded_terminal_tangent_effective_weight": xyz_loss.new_tensor(
            terminal_tangent_weight
        ),
        "decoded_terminal_start_tangent_angle_mae_deg": (
            terminal_tangent_parts["start_angle_mae_deg"]
        ),
        "decoded_terminal_end_tangent_angle_mae_deg": (
            terminal_tangent_parts["end_angle_mae_deg"]
        ),
        "decoded_terminal_start_tangent_valid_fraction": (
            terminal_tangent_parts["start_valid_fraction"]
        ),
        "decoded_terminal_end_tangent_valid_fraction": (
            terminal_tangent_parts["end_valid_fraction"]
        ),
        "branch_length_loss": branch_length_parts["branch_length_loss"],
        "branch_length_global_loss": branch_length_parts[
            "branch_length_global_loss"
        ],
        "branch_length_local_deficit_loss": branch_length_parts[
            "branch_length_local_deficit_loss"
        ],
        "branch_length_smooth_excess_loss": branch_length_parts[
            "branch_length_smooth_excess_loss"
        ],
        "branch_length_high_region_fraction": branch_length_parts[
            "branch_length_high_region_fraction"
        ],
        "branch_length_smooth_region_fraction": branch_length_parts[
            "branch_length_smooth_region_fraction"
        ],
        "branch_length_valid_fraction": branch_length_parts[
            "branch_length_valid_fraction"
        ],
        "branch_length_weighted_loss": branch_length_weighted_loss,
        "branch_length_effective_weight": xyz_loss.new_tensor(
            branch_length_effective_weight
        ),
        "arc_length_abs_error_mm": branch_length_parts[
            "arc_length_abs_error_mm"
        ],
        "arc_length_rel_error": branch_length_parts["arc_length_rel_error"],
        "decoded_radius_loss": radius_loss,
        "decoded_radius_point_loss": radius_point_loss,
        "decoded_radius_unweighted_point_loss": (
            radius_profile_parts["unweighted_point_loss"]
        ),
        "decoded_radius_point_weighted_loss": (
            radius_point_weighted_loss
        ),
        "decoded_radius_point_effective_weight": radius_loss.new_tensor(
            radius_profile_parts["point_weight"]
        ),
        "decoded_radius_multiscale_slope_loss": radius_profile_parts[
            "slope_loss"
        ],
        "decoded_radius_multiscale_slope_weighted_loss": (
            radius_slope_weighted_loss
        ),
        "decoded_radius_multiscale_slope_effective_weight": (
            radius_loss.new_tensor(radius_profile_parts["slope_weight"])
        ),
        "decoded_radius_branch_mean_loss": radius_profile_parts[
            "branch_mean_loss"
        ],
        "decoded_radius_branch_mean_weighted_loss": (
            radius_branch_mean_weighted_loss
        ),
        "decoded_radius_branch_mean_effective_weight": (
            radius_loss.new_tensor(radius_profile_parts["branch_mean_weight"])
        ),
        "decoded_radius_branch_q90_loss": radius_profile_parts[
            "branch_q90_loss"
        ],
        "decoded_radius_branch_q90_weighted_loss": (
            radius_branch_q90_weighted_loss
        ),
        "decoded_radius_branch_q90_effective_weight": (
            radius_loss.new_tensor(radius_profile_parts["branch_q90_weight"])
        ),
        "decoded_radius_hard_point_mean_weight": radius_profile_parts[
            "hard_mean_weight"
        ],
        "decoded_radius_hard_point_exceedance_fraction": (
            radius_profile_parts["hard_exceedance_fraction"]
        ),
        "decoded_radius_hard_point_threshold_mm": radius_profile_parts[
            "hard_threshold_mm"
        ],
        "bend_recall_3d_loss": bend_recall_3d_parts["loss"],
        "bend_recall_3d_underbend_loss": bend_recall_3d_parts[
            "underbend_loss"
        ],
        "bend_recall_3d_excess_loss": bend_recall_3d_parts["excess_loss"],
        "bend_recall_3d_direction_loss": bend_recall_3d_parts[
            "direction_loss"
        ],
        "bend_recall_3d_weighted_loss": bend_recall_3d_weighted_loss,
        "bend_recall_3d_effective_weight": xyz_loss.new_tensor(
            bend_recall_3d_effective_weight
        ),
        "bend_recall_3d_high_region_fraction": bend_recall_3d_parts[
            "high_region_fraction"
        ],
        "bend_recall_3d_valid_fraction": bend_recall_3d_parts[
            "valid_fraction"
        ],
        "decoded_local_progress_loss": local_progress_parts["loss"],
        "decoded_local_progress_weighted_loss": (
            local_progress_weighted_loss
        ),
        "decoded_local_progress_effective_weight": xyz_loss.new_tensor(
            local_progress_effective_weight
        ),
        "decoded_local_progress_valid_fraction": local_progress_parts[
            "valid_fraction"
        ],
        "decoded_local_progress_backward_fraction": local_progress_parts[
            "backward_fraction"
        ],
        "decoded_local_progress_insufficient_fraction": local_progress_parts[
            "insufficient_fraction"
        ],
        "decoded_local_progress_mean_ratio": local_progress_parts[
            "mean_ratio"
        ],
        "decoded_local_progress_p05_ratio": local_progress_parts[
            "p05_ratio"
        ],
    }
    for offset, offset_loss in radius_profile_parts[
        "slope_offset_losses"
    ].items():
        losses[f"decoded_radius_slope_offset_{offset}_loss"] = offset_loss
    if xyz_threshold_parts is not None:
        losses["decoded_xyz_pre_threshold_loss"] = xyz_unthresholded_loss
        losses["decoded_xyz_final_threshold_loss"] = (
            xyz_final_threshold_parts["loss"]
        )
        losses["decoded_xyz_effective_threshold_mm"] = xyz_threshold_parts[
            "effective_threshold"
        ]
        losses["decoded_xyz_soft_active_fraction"] = xyz_threshold_parts[
            "soft_active_fraction"
        ]
        losses["decoded_xyz_hard_exceedance_fraction"] = xyz_threshold_parts[
            "hard_exceedance_fraction"
        ]
    if radius_threshold_parts is not None:
        losses["decoded_radius_pre_threshold_loss"] = (
            radius_unthresholded_composite_loss
        )
        losses["decoded_radius_final_threshold_loss"] = (
            radius_final_threshold_composite_loss
        )
        losses["decoded_radius_effective_threshold_mm"] = (
            radius_threshold_parts["effective_threshold"]
        )
        losses["decoded_radius_soft_active_fraction"] = radius_threshold_parts[
            "soft_active_fraction"
        ]
        losses["decoded_radius_hard_exceedance_fraction"] = (
            radius_threshold_parts["hard_exceedance_fraction"]
        )
    if centerline_prediction_mode == "adaptive_landmarks":
        losses["centerline_landmark_loss"] = centerline_parameter_loss
    else:
        losses["centerline_control_loss"] = centerline_parameter_loss
    losses.update(projection_losses)

    centerline_map_enabled_raw = _model_option(
        config,
        "bspline_refiner_use_unexplained_centerline_evidence",
        False,
    )
    if not isinstance(centerline_map_enabled_raw, bool):
        raise ValueError(
            "bspline_refiner_use_unexplained_centerline_evidence must be a "
            f"JSON boolean (true or false), got {centerline_map_enabled_raw!r}."
        )
    centerline_map_loss_weight = float(
        _loss_option(config, "bspline_refiner_centerline_map_loss_weight", 0.1)
    )
    if (
        not math.isfinite(centerline_map_loss_weight)
        or centerline_map_loss_weight < 0.0
    ):
        raise ValueError(
            "bspline_refiner_centerline_map_loss_weight must be finite and "
            f">= 0, got {centerline_map_loss_weight}."
        )
    if (
        _include_bspline_refinement_supervision
        and centerline_map_enabled_raw
        and centerline_map_loss_weight > 0.0
        and not bool(config.get("train_radius_refiner_only", False))
        and not bool(config.get("train_radius_head_only", False))
    ):
        centerline_map_projector = (
            bspline_refiner_projector
            if bspline_refiner_projector is not None
            else projector
        )
        if centerline_map_projector is None:
            raise ValueError(
                "bspline_refiner_use_unexplained_centerline_evidence=true "
                "requires a B-spline refiner centerline projector during "
                "loss computation."
            )
        centerline_map_parts = _bspline_refiner_centerline_map_loss(
            output=output,
            batch=batch,
            config=config,
            projector=centerline_map_projector,
        )
        losses["loss"] = (
            losses["loss"] + centerline_map_parts["weighted_loss"]
        )
        losses["bspline_refiner_centerline_map_loss"] = (
            centerline_map_parts["loss"]
        )
        losses["bspline_refiner_centerline_map_weighted_loss"] = (
            centerline_map_parts["weighted_loss"]
        )
        losses["bspline_refiner_centerline_map_bce_loss"] = (
            centerline_map_parts["bce_loss"]
        )
        losses["bspline_refiner_centerline_map_dice_loss"] = (
            centerline_map_parts["dice_loss"]
        )
        losses["bspline_refiner_centerline_map_target_positive_fraction"] = (
            centerline_map_parts["target_positive_fraction"]
        )
        losses["bspline_refiner_centerline_map_effective_weight"] = (
            centerline_map_parts["effective_weight"]
        )

    if (
        _include_bspline_refinement_supervision
        and "refinement_stage_branch_exist_logits" in output
        and not bool(config.get("train_radius_refiner_only", False))
        and not bool(config.get("train_radius_head_only", False))
    ):
        refined_existence = _bspline_refined_branch_existence_loss(
            output,
            batch["target_branch_exist"],
            config,
            batch.get("target_branch_existence_supervision_mask"),
        )
        losses["loss"] = losses["loss"] + refined_existence["loss"]
        losses["bspline_refiner_branch_existence_loss"] = refined_existence[
            "final_loss"
        ]
        losses[
            "bspline_refiner_branch_existence_weighted_loss"
        ] = refined_existence["final_weighted_loss"]
        losses[
            "bspline_refiner_branch_existence_intermediate_loss"
        ] = refined_existence["intermediate_loss"]
        losses[
            "bspline_refiner_branch_existence_intermediate_weighted_loss"
        ] = refined_existence["intermediate_weighted_loss"]
        for diagnostic_name in (
            "candidate_count",
            "coarse_false_positive_count",
            "coarse_true_positive_count",
            "final_false_positive_count",
            "removed_false_positive_count",
        ):
            losses[
                f"bspline_refiner_branch_existence_{diagnostic_name}"
            ] = refined_existence[diagnostic_name]
        num_existence_stages = int(
            output["refinement_stage_branch_exist_logits"].shape[1]
        )
        for stage_index in range(num_existence_stages):
            losses[
                f"bspline_refiner_stage_{stage_index + 1}_branch_existence_loss"
            ] = refined_existence[f"stage_{stage_index + 1}_loss"]

    if (
        _include_bspline_refinement_supervision
        and "radius_refinement_stage_decoded_vessel_mm" in output
    ):
        if radius_prediction_mode != "raw":
            raise ValueError(
                "Radius-refinement supervision currently requires "
                "radius_prediction_mode='raw'."
            )
        stage_vessels = output["radius_refinement_stage_decoded_vessel_mm"]
        coarse_vessel = output["radius_refiner_coarse_decoded_vessel_mm"]
        if stage_vessels.dim() != 5 or stage_vessels.shape[-1] < 4:
            raise ValueError(
                "radius_refinement_stage_decoded_vessel_mm must have shape "
                f"[B,S,M,N,4], got {tuple(stage_vessels.shape)}."
            )
        num_radius_stages = int(stage_vessels.shape[1])
        if num_radius_stages < 1:
            raise ValueError("Radius refinement must contain at least one stage.")

        coarse_weight = float(
            _loss_option(config, "radius_refiner_coarse_loss_weight", 0.0)
        )
        intermediate_weight = float(
            _loss_option(
                config,
                "radius_refiner_intermediate_loss_weight",
                0.5,
            )
        )
        dice_weight = float(
            _loss_option(config, "radius_refiner_dice_loss_weight", 0.2)
        )
        for name, value in (
            ("radius_refiner_coarse_loss_weight", coarse_weight),
            ("radius_refiner_intermediate_loss_weight", intermediate_weight),
            ("radius_refiner_dice_loss_weight", dice_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and >= 0, got {value}."
                )

        def stage_radius_loss(vessel: torch.Tensor) -> torch.Tensor:
            return _masked_mean(
                F.smooth_l1_loss(
                    vessel[..., 3] / radius_scale,
                    radius_target_vessel[..., 3] / radius_scale,
                    reduction="none",
                ),
                point_mask,
            )

        coarse_radius_loss = stage_radius_loss(coarse_vessel)
        stage_radius_losses = [
            stage_radius_loss(stage_vessels[:, stage_index])
            for stage_index in range(num_radius_stages)
        ]
        intermediate_losses = stage_radius_losses[:-1]
        intermediate_radius_loss = (
            torch.stack(intermediate_losses).mean()
            if intermediate_losses
            else losses["loss"].new_zeros(())
        )
        final_radius_loss = stage_radius_losses[-1]

        images = batch.get("images")
        view_mask = batch.get("view_mask")
        if dice_weight > 0.0 and (images is None or view_mask is None):
            raise ValueError(
                "radius_refiner_dice_loss_weight > 0 requires images and "
                "view_mask in the training batch."
            )
        if dice_weight > 0.0:
            assert images is not None and view_mask is not None
            dice_loss = _radius_refiner_mask_dice_loss(
                output["radius_refiner_final_rendered_masks"],
                images.to(output["decoded_vessel_mm"].device),
                view_mask,
                threshold=float(config.get("projection_mask_threshold", 0.5)),
            )
        else:
            dice_loss = losses["loss"].new_zeros(())

        coarse_weighted = coarse_weight * coarse_radius_loss
        intermediate_weighted = (
            intermediate_weight * intermediate_radius_loss
        )
        dice_weighted = dice_weight * dice_loss
        losses["loss"] = (
            losses["loss"]
            + coarse_weighted
            + intermediate_weighted
            + dice_weighted
        )
        losses["radius_refiner_final_radius_loss"] = final_radius_loss
        losses["radius_refiner_coarse_radius_loss"] = coarse_radius_loss
        losses["radius_refiner_coarse_weighted_loss"] = coarse_weighted
        losses["radius_refiner_intermediate_radius_loss"] = (
            intermediate_radius_loss
        )
        losses["radius_refiner_intermediate_weighted_loss"] = (
            intermediate_weighted
        )
        losses["radius_refiner_mask_dice_loss"] = dice_loss
        losses["radius_refiner_mask_dice_weighted_loss"] = dice_weighted
        losses["radius_refiner_coarse_effective_weight"] = losses[
            "loss"
        ].new_tensor(coarse_weight)
        losses["radius_refiner_intermediate_total_weight"] = losses[
            "loss"
        ].new_tensor(intermediate_weight)
        losses["radius_refiner_dice_effective_weight"] = losses[
            "loss"
        ].new_tensor(dice_weight)
        residual_abs_mm = output["radius_refinement_residual_mm"].abs()
        losses["radius_refiner_residual_mean_mm"] = residual_abs_mm.mean()
        losses["radius_refiner_residual_max_mm"] = residual_abs_mm.amax()
        for stage_index, stage_loss in enumerate(stage_radius_losses):
            losses[
                f"radius_refiner_stage_{stage_index + 1}_radius_loss"
            ] = stage_loss
            losses[
                f"radius_refiner_stage_{stage_index + 1}_residual_mean_mm"
            ] = residual_abs_mm[:, stage_index].mean()

    if (
        _include_bspline_refinement_supervision
        and "coarse_centerline_parameters_mm" in output
        and not bool(config.get("train_radius_refiner_only", False))
        and not bool(config.get("train_radius_head_only", False))
    ):
        if centerline_prediction_mode != "bspline_control_points":
            raise ValueError(
                "B-spline refinement outputs are incompatible with "
                f"centerline_prediction_mode={centerline_prediction_mode!r}."
            )
        coarse_weight = float(
            _loss_option(config, "bspline_refiner_coarse_loss_weight", 0.1)
        )
        intermediate_weight = float(
            _loss_option(
                config, "bspline_refiner_intermediate_loss_weight", 0.25
            )
        )
        for name, value in (
            ("bspline_refiner_coarse_loss_weight", coarse_weight),
            ("bspline_refiner_intermediate_loss_weight", intermediate_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and >= 0, got {value}."
                )

        refinement_keys = {
            "coarse_centerline_parameters_mm",
            "coarse_decoded_vessel_mm",
            "coarse_side_centerline_offsets_mm",
            "coarse_side_branch_relative_code_mm",
            "refinement_stage_centerline_parameters_mm",
            "refinement_stage_decoded_vessel_mm",
            "refinement_stage_side_centerline_offsets_mm",
            "refinement_stage_side_branch_relative_code_mm",
            "bspline_refinement_residual_mm",
            "coarse_branch_exist_logits",
            "coarse_branch_exist_probs",
            "refinement_stage_branch_exist_logits",
            "refinement_stage_branch_exist_probs",
            "bspline_refinement_branch_exist_residual_logits",
            "bspline_refiner_existence_candidate_mask",
        }

        auxiliary_config = config
        if bool(config.get("proj_dice_gradient_split_enabled", False)):
            # Coarse/intermediate B-spline supervision is centreline-only.
            # Preserve the geometry and joint Dice gradients that target XYZ,
            # but exclude the radius-only component from both its gradient and
            # checkpoint scalar. The radius tensor itself is detached below.
            auxiliary_geometry_dice_weight = float(
                config.get("proj_geometry_dice_weight", 1.0)
            ) + float(config.get("proj_joint_dice_weight", 0.1))
            auxiliary_config = dict(config)
            auxiliary_config["proj_dice_weight"] = 0.0
            if auxiliary_geometry_dice_weight > 0.0:
                auxiliary_config["proj_geometry_dice_weight"] = (
                    auxiliary_geometry_dice_weight
                )
                auxiliary_config["proj_radius_dice_weight"] = 0.0
                auxiliary_config["proj_joint_dice_weight"] = 0.0
            else:
                auxiliary_config["proj_dice_gradient_split_enabled"] = False

        def variant_output(
            parameters: torch.Tensor,
            vessel: torch.Tensor,
            side_offsets: torch.Tensor,
            side_relative: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            variant = {
                key: value
                for key, value in output.items()
                if key not in refinement_keys
            }
            variant["centerline_parameters_mm"] = parameters
            variant["centerline_control_points_mm"] = parameters
            # Auxiliary refinement supervision is centreline-only. Keep the
            # radius values available to surface projection, but do not replay
            # its gradient once per coarse/intermediate stage.
            variant["decoded_vessel_mm"] = torch.cat(
                [vessel[..., :3], vessel[..., 3:].detach()], dim=-1
            )
            variant["side_centerline_offsets_mm"] = side_offsets
            variant["side_branch_relative_code_mm"] = side_relative
            return variant

        def centerline_auxiliary_objective(
            parts: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            value = (
                _weight(config, "centerline_control_loss_weight", 1.0)
                * parts["centerline_control_loss"]
                + _weight(config, "side_relative_xyz_loss_weight", 0.0)
                * parts["side_relative_xyz_loss"]
                + _weight(config, "decoded_xyz_loss_weight", 1.0)
                * parts["decoded_xyz_loss"]
            )
            if "projection_2d_loss" in parts:
                value = value + parts["projection_2d_loss"]
            return value

        coarse_auxiliary = total.new_zeros(())
        if coarse_weight > 0.0:
            coarse_output = variant_output(
                output["coarse_centerline_parameters_mm"],
                output["coarse_decoded_vessel_mm"],
                output["coarse_side_centerline_offsets_mm"],
                output["coarse_side_branch_relative_code_mm"],
            )
            coarse_parts = compute_parametric_loss(
                coarse_output,
                batch,
                auxiliary_config,
                projector=projector,
                epoch=epoch,
                _include_bspline_refinement_supervision=False,
            )
            coarse_auxiliary = centerline_auxiliary_objective(coarse_parts)
            losses["bspline_refiner_coarse_centerline_control_loss"] = (
                coarse_parts["centerline_control_loss"]
            )
            losses["bspline_refiner_coarse_decoded_xyz_loss"] = (
                coarse_parts["decoded_xyz_loss"]
            )
            if "decoded_xyz_final_threshold_loss" in coarse_parts:
                losses[
                    "bspline_refiner_coarse_decoded_xyz_final_threshold_loss"
                ] = coarse_parts["decoded_xyz_final_threshold_loss"]
            if "projection_2d_loss" in coarse_parts:
                losses["bspline_refiner_coarse_projection_2d_loss"] = (
                    coarse_parts["projection_2d_loss"]
                )
                losses[
                    "bspline_refiner_coarse_projection_2d_unweighted_loss"
                ] = coarse_parts["projection_2d_unweighted_loss"]
        coarse_weighted = coarse_weight * coarse_auxiliary

        stage_parameters = output[
            "refinement_stage_centerline_parameters_mm"
        ]
        stage_vessels = output["refinement_stage_decoded_vessel_mm"]
        stage_side_offsets = output[
            "refinement_stage_side_centerline_offsets_mm"
        ]
        stage_side_relative = output[
            "refinement_stage_side_branch_relative_code_mm"
        ]
        num_stages = int(stage_parameters.shape[1])
        if num_stages < 1:
            raise ValueError("B-spline refinement must contain at least one stage.")
        intermediate_sum = total.new_zeros(())
        num_intermediate_to_supervise = (
            num_stages - 1 if intermediate_weight > 0.0 else 0
        )
        for stage_index in range(num_intermediate_to_supervise):
            stage_output = variant_output(
                stage_parameters[:, stage_index],
                stage_vessels[:, stage_index],
                stage_side_offsets[:, stage_index],
                stage_side_relative[:, stage_index],
            )
            stage_parts = compute_parametric_loss(
                stage_output,
                batch,
                auxiliary_config,
                projector=projector,
                epoch=epoch,
                _include_bspline_refinement_supervision=False,
            )
            stage_auxiliary = centerline_auxiliary_objective(stage_parts)
            intermediate_sum = intermediate_sum + stage_auxiliary
            losses[
                f"bspline_refiner_stage_{stage_index + 1}_auxiliary_loss"
            ] = stage_auxiliary
            losses[
                f"bspline_refiner_stage_{stage_index + 1}_centerline_control_loss"
            ] = stage_parts["centerline_control_loss"]
            losses[
                f"bspline_refiner_stage_{stage_index + 1}_decoded_xyz_loss"
            ] = stage_parts["decoded_xyz_loss"]
            if "decoded_xyz_final_threshold_loss" in stage_parts:
                losses[
                    f"bspline_refiner_stage_{stage_index + 1}_decoded_xyz_final_threshold_loss"
                ] = stage_parts["decoded_xyz_final_threshold_loss"]
            if "projection_2d_loss" in stage_parts:
                losses[
                    f"bspline_refiner_stage_{stage_index + 1}_projection_2d_loss"
                ] = stage_parts["projection_2d_loss"]
                losses[
                    f"bspline_refiner_stage_{stage_index + 1}_projection_2d_unweighted_loss"
                ] = stage_parts["projection_2d_unweighted_loss"]

        final_auxiliary = centerline_auxiliary_objective(losses)
        num_intermediate_stages = max(num_stages - 1, 0)
        if num_intermediate_stages > 0:
            intermediate_mean = intermediate_sum / float(
                num_intermediate_stages
            )
            intermediate_per_stage_weight = (
                intermediate_weight / float(num_intermediate_stages)
            )
        else:
            intermediate_mean = intermediate_sum
            intermediate_per_stage_weight = 0.0
        intermediate_weighted = intermediate_weight * intermediate_mean
        losses["loss"] = losses["loss"] + coarse_weighted + intermediate_weighted
        losses["bspline_refiner_coarse_auxiliary_loss"] = coarse_auxiliary
        losses["bspline_refiner_coarse_weighted_loss"] = coarse_weighted
        losses["bspline_refiner_final_auxiliary_loss"] = final_auxiliary
        losses["bspline_refiner_intermediate_auxiliary_loss"] = intermediate_mean
        losses[
            "bspline_refiner_intermediate_weighted_loss"
        ] = intermediate_weighted
        losses["bspline_refiner_coarse_effective_weight"] = total.new_tensor(
            coarse_weight
        )
        losses[
            "bspline_refiner_intermediate_effective_weight"
        ] = total.new_tensor(intermediate_per_stage_weight)
        losses[
            "bspline_refiner_intermediate_total_weight"
        ] = total.new_tensor(intermediate_weight)
        residual_norm_mm = torch.linalg.vector_norm(
            output["bspline_refinement_residual_mm"], dim=-1
        )
        losses["bspline_refiner_residual_mean_mm"] = residual_norm_mm.mean()
        losses["bspline_refiner_residual_max_mm"] = residual_norm_mm.amax()
        for stage_index in range(num_stages):
            losses[
                f"bspline_refiner_stage_{stage_index + 1}_residual_mean_mm"
            ] = residual_norm_mm[:, stage_index].mean()
    return losses
