# Transferred from methods/parametric_methods/radius_refiner_loss.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any, Sequence
import torch
import torch.nn.functional as F

NATURAL_NEGATIVE = "natural_negative"

VISIBLE_AUGMENTED = "visible_augmented"

INVISIBLE_STENOSIS = "invisible_stenosis"

SUPPORTED_GROUP_TYPES = frozenset(
    {NATURAL_NEGATIVE, VISIBLE_AUGMENTED, INVISIBLE_STENOSIS}
)

ORIGINAL = "original"

REMOVED = "stenosis_removed"

STRENGTHENED = "stenosis_strengthened"

COUNTERFACTUAL_VARIANTS = (ORIGINAL, REMOVED, STRENGTHENED)

def _option(config: dict[str, Any], key: str, default: Any) -> Any:
    loss_config = config.get("loss", {})
    if isinstance(loss_config, dict) and key in loss_config:
        return loss_config[key]
    return config.get(key, default)

def _finite_nonnegative(config: dict[str, Any], key: str, default: float) -> float:
    value = float(_option(config, key, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and >= 0, got {value}.")
    return value

def _strict_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = _option(config, key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON boolean, got {value!r}.")
    return value

def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.to(device=values.device, dtype=values.dtype)
    if weights is not None:
        effective = effective * weights.to(
            device=values.device, dtype=values.dtype
        )
    while effective.dim() < values.dim():
        effective = effective.unsqueeze(-1)
    effective = effective.expand_as(values)
    return (values * effective).sum() / effective.sum().clamp_min(1.0)

def _target_masks(images: torch.Tensor, *, threshold: float) -> torch.Tensor:
    if images.dim() == 5:
        images = images[:, :, 0]
    if images.dim() != 4:
        raise ValueError(
            "Radius-refiner images must have shape [B,V,H,W] or "
            f"[B,V,C,H,W], got {tuple(images.shape)}."
        )
    return images > float(threshold)

def whole_artery_dice_loss(
    *,
    predicted_masks: torch.Tensor,
    target_images: torch.Tensor,
    view_mask: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute soft Dice over the complete artery in every valid view."""

    if predicted_masks.dim() != 4:
        raise ValueError(
            "radius_refiner_final_rendered_masks must have shape [B,V,H,W], "
            f"got {tuple(predicted_masks.shape)}."
        )
    targets = _target_masks(target_images, threshold=threshold).to(
        device=predicted_masks.device, dtype=predicted_masks.dtype
    )
    if targets.shape != predicted_masks.shape:
        raise ValueError(
            "Rendered and target mask shapes differ: "
            f"{tuple(predicted_masks.shape)} versus {tuple(targets.shape)}."
        )
    valid_views = view_mask.to(
        device=predicted_masks.device, dtype=torch.bool
    )
    if valid_views.shape != predicted_masks.shape[:2]:
        raise ValueError(
            f"view_mask has shape {tuple(valid_views.shape)}, expected "
            f"{tuple(predicted_masks.shape[:2])}."
        )
    if not bool(valid_views.any().item()):
        raise ValueError("Whole-artery Dice received no valid input views.")

    intersection = (predicted_masks * targets).flatten(start_dim=2).sum(-1)
    denominator = predicted_masks.flatten(start_dim=2).sum(-1) + (
        targets.flatten(start_dim=2).sum(-1)
    )
    score = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    mean_score = score[valid_views].mean()
    return 1.0 - mean_score, mean_score

def _variant_index(
    variants: Sequence[str], variant: str, *, required: bool = True
) -> int | None:
    matches = [index for index, value in enumerate(variants) if value == variant]
    if len(matches) > 1:
        raise ValueError(f"Counterfactual group repeats variant {variant!r}.")
    if not matches:
        if required:
            raise ValueError(f"Counterfactual group is missing variant {variant!r}.")
        return None
    return matches[0]

def build_counterfactual_changed_mask_roi(
    *,
    target_images: torch.Tensor,
    variants: Sequence[str],
    dilation_px: int,
    threshold: float,
) -> torch.Tensor:
    """Return a common [V,H,W] ROI from removed/strengthened mask change.

    The mask difference is the same observable signal used by Stage 3.1 to
    decide whether a counterfactual stenosis is visible.  It is deliberately
    independent of the current prediction and therefore cannot move to make a
    poor prediction look better.
    """

    dilation = int(dilation_px)
    if dilation < 0:
        raise ValueError(
            "radius_refiner_local_stenosis_dice_roi_dilation_px must be >= 0, "
            f"got {dilation}."
        )
    masks = _target_masks(target_images, threshold=threshold)
    removed_index = _variant_index(variants, REMOVED)
    strengthened_index = _variant_index(variants, STRENGTHENED)
    assert removed_index is not None and strengthened_index is not None
    changed = torch.logical_xor(
        masks[removed_index], masks[strengthened_index]
    ).to(dtype=target_images.dtype)
    if dilation:
        kernel_size = 2 * dilation + 1
        changed = F.max_pool2d(
            changed.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
        ).squeeze(1)
    return changed > 0.5

def local_stenosis_dice_loss(
    *,
    predicted_masks: torch.Tensor,
    target_images: torch.Tensor,
    view_mask: torch.Tensor,
    group_visible_view_mask: torch.Tensor,
    variants: Sequence[str],
    dilation_px: int,
    threshold: float,
    visible_views_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute triplet-local soft Dice using one common counterfactual ROI."""

    if predicted_masks.dim() != 4:
        raise ValueError(
            "radius_refiner_final_rendered_masks must have shape [B,V,H,W], "
            f"got {tuple(predicted_masks.shape)}."
        )
    targets_bool = _target_masks(target_images, threshold=threshold)
    targets = targets_bool.to(
        device=predicted_masks.device, dtype=predicted_masks.dtype
    )
    if targets.shape != predicted_masks.shape:
        raise ValueError(
            "Rendered and target mask shapes differ: "
            f"{tuple(predicted_masks.shape)} versus {tuple(targets.shape)}."
        )
    roi = build_counterfactual_changed_mask_roi(
        target_images=target_images,
        variants=variants,
        dilation_px=dilation_px,
        threshold=threshold,
    ).to(device=predicted_masks.device)
    if roi.shape != predicted_masks.shape[1:]:
        raise ValueError(
            f"Local stenosis ROI has shape {tuple(roi.shape)}, expected "
            f"{tuple(predicted_masks.shape[1:])}."
        )

    valid_views = view_mask.to(
        device=predicted_masks.device, dtype=torch.bool
    )
    if valid_views.shape != predicted_masks.shape[:2]:
        raise ValueError(
            f"view_mask has shape {tuple(valid_views.shape)}, expected "
            f"{tuple(predicted_masks.shape[:2])}."
        )
    if visible_views_only:
        visible = group_visible_view_mask.to(
            device=predicted_masks.device, dtype=torch.bool
        )
        if visible.shape == (predicted_masks.shape[1],):
            visible = visible.unsqueeze(0).expand(predicted_masks.shape[0], -1)
        if visible.shape != valid_views.shape:
            raise ValueError(
                "stenosis_view_visible_mask must have shape [V] or [B,V], "
                f"got {tuple(visible.shape)}."
            )
        valid_views = valid_views & visible
    roi_has_pixels = roi.flatten(start_dim=1).any(dim=-1)
    valid_views = valid_views & roi_has_pixels.unsqueeze(0)
    valid_count = valid_views.sum()

    roi_float = roi.to(dtype=predicted_masks.dtype).unsqueeze(0)
    predicted_local = predicted_masks * roi_float
    target_local = targets * roi_float
    intersection = (predicted_local * target_local).flatten(start_dim=2).sum(-1)
    denominator = predicted_local.flatten(start_dim=2).sum(-1) + (
        target_local.flatten(start_dim=2).sum(-1)
    )
    score = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    valid = valid_views.to(dtype=score.dtype)
    mean_score = (score * valid).sum() / valid.sum().clamp_min(1.0)
    mean_score = torch.where(
        valid_count > 0,
        mean_score,
        predicted_masks.new_ones(()),
    )
    return 1.0 - mean_score, mean_score, valid_count

def validate_counterfactual_radius_refiner_loss_config(
    config: dict[str, Any],
) -> None:
    for key, default in (
        ("radius_refiner_final_radius_loss_weight", 1.0),
        ("radius_refiner_intermediate_radius_loss_weight", 0.25),
        ("radius_refiner_paired_difference_loss_weight", 1.0),
        ("radius_refiner_outside_consistency_loss_weight", 0.1),
        ("radius_refiner_conservative_residual_loss_weight", 0.05),
        ("radius_refiner_whole_artery_dice_loss_weight", 0.1),
        ("radius_refiner_local_stenosis_dice_loss_weight", 0.25),
    ):
        _finite_nonnegative(config, key, default)
    interval_weight = float(
        _option(config, "radius_refiner_stenosis_interval_point_weight", 5.0)
    )
    if not math.isfinite(interval_weight) or interval_weight < 1.0:
        raise ValueError(
            "radius_refiner_stenosis_interval_point_weight must be finite and "
            f">= 1, got {interval_weight}."
        )
    radius_scale = float(_option(config, "radius_scale_mm", 1.0))
    if not math.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError(
            f"radius_scale_mm must be finite and > 0, got {radius_scale}."
        )
    whole_dice_enabled = _strict_bool(
        config, "radius_refiner_whole_artery_dice_enabled", False
    )
    if whole_dice_enabled and _finite_nonnegative(
        config, "radius_refiner_whole_artery_dice_loss_weight", 0.1
    ) <= 0.0:
        raise ValueError(
            "Enabling whole-artery Dice requires a positive "
            "radius_refiner_whole_artery_dice_loss_weight."
        )
    local_dice_enabled = _strict_bool(
        config, "radius_refiner_local_stenosis_dice_enabled", True
    )
    _strict_bool(
        config,
        "radius_refiner_local_stenosis_dice_visible_views_only",
        True,
    )
    dilation = int(
        _option(config, "radius_refiner_local_stenosis_dice_roi_dilation_px", 8)
    )
    if dilation < 0:
        raise ValueError(
            "radius_refiner_local_stenosis_dice_roi_dilation_px must be >= 0, "
            f"got {dilation}."
        )
    if local_dice_enabled and _finite_nonnegative(
        config, "radius_refiner_local_stenosis_dice_loss_weight", 0.25
    ) <= 0.0:
        raise ValueError(
            "Enabling local stenosis Dice requires a positive "
            "radius_refiner_local_stenosis_dice_loss_weight."
        )

def compute_counterfactual_radius_refiner_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return the clean group-normalized Stage-3 radius-refiner objective."""

    validate_counterfactual_radius_refiner_loss_config(config)
    group_type = str(batch.get("group_type", ""))
    if group_type not in SUPPORTED_GROUP_TYPES:
        raise ValueError(
            f"Unsupported radius-refiner group_type={group_type!r}; expected "
            f"one of {sorted(SUPPORTED_GROUP_TYPES)}."
        )
    variants = [str(value) for value in batch.get("counterfactual_variant", [])]
    final_vessel = output.get("decoded_vessel_mm")
    coarse_vessel = output.get("radius_refiner_coarse_decoded_vessel_mm")
    stage_vessels = output.get("radius_refinement_stage_decoded_vessel_mm")
    residuals = output.get("radius_refinement_residual_mm")
    if final_vessel is None or coarse_vessel is None or stage_vessels is None:
        raise KeyError(
            "Counterfactual radius-refiner loss requires decoded_vessel_mm, "
            "radius_refiner_coarse_decoded_vessel_mm and "
            "radius_refinement_stage_decoded_vessel_mm."
        )
    device = final_vessel.device
    target = batch["target_raw_vessel_mm"].to(device=device)
    point_valid = batch["target_point_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    branch_exists = batch["target_branch_exist"].to(device=device) > 0.5
    point_mask = point_valid & branch_exists.unsqueeze(-1)
    region = batch["stenosis_region_point_mask"].to(
        device=device, dtype=torch.bool
    )
    if region.shape != point_mask.shape:
        raise ValueError(
            f"stenosis_region_point_mask has shape {tuple(region.shape)}, "
            f"expected {tuple(point_mask.shape)}."
        )
    region = region & point_mask
    batch_size = int(final_vessel.shape[0])
    if len(variants) != batch_size:
        raise ValueError(
            f"Expected {batch_size} counterfactual variant names, got {variants}."
        )
    if group_type == VISIBLE_AUGMENTED:
        if set(variants) != set(COUNTERFACTUAL_VARIANTS) or batch_size != 3:
            raise ValueError(
                "A visible_augmented group must contain exactly original, "
                "stenosis_removed and stenosis_strengthened."
            )
        if not bool(region.any().item()):
            raise ValueError("A visible_augmented group has an empty stenosis interval.")
    elif batch_size != 1 or variants != [ORIGINAL]:
        raise ValueError(
            f"A {group_type} group must contain original only, got {variants}."
        )

    radius_scale = float(_option(config, "radius_scale_mm", 1.0))
    interval_point_weight = float(
        _option(config, "radius_refiner_stenosis_interval_point_weight", 5.0)
    )
    direct_mask = point_mask
    if group_type == INVISIBLE_STENOSIS:
        direct_mask = direct_mask & ~region
    point_weights = torch.ones_like(target[..., 3])
    if group_type == VISIBLE_AUGMENTED:
        point_weights = torch.where(
            region,
            point_weights.new_full((), interval_point_weight),
            point_weights,
        )

    def radius_profile_loss(vessel: torch.Tensor) -> torch.Tensor:
        values = F.smooth_l1_loss(
            vessel[..., 3] / radius_scale,
            target[..., 3] / radius_scale,
            reduction="none",
        )
        return _masked_weighted_mean(values, direct_mask, point_weights)

    final_radius_loss = radius_profile_loss(final_vessel)
    if stage_vessels.dim() != 5 or stage_vessels.shape[0] != batch_size:
        raise ValueError(
            "radius_refinement_stage_decoded_vessel_mm must have shape "
            f"[B,S,M,N,4], got {tuple(stage_vessels.shape)}."
        )
    intermediate_parts = [
        radius_profile_loss(stage_vessels[:, index])
        for index in range(max(int(stage_vessels.shape[1]) - 1, 0))
    ]
    intermediate_loss = (
        torch.stack(intermediate_parts).mean()
        if intermediate_parts
        else final_radius_loss.new_zeros(())
    )

    paired_loss = final_radius_loss.new_zeros(())
    outside_loss = final_radius_loss.new_zeros(())
    paired_mae_mm = final_radius_loss.new_zeros(())
    outside_mae_mm = final_radius_loss.new_zeros(())
    if group_type == VISIBLE_AUGMENTED:
        mapping = {variant: index for index, variant in enumerate(variants)}
        pair_names = (
            (STRENGTHENED, REMOVED),
            (STRENGTHENED, ORIGINAL),
            (ORIGINAL, REMOVED),
        )
        paired_parts = []
        outside_parts = []
        paired_mae_parts = []
        outside_mae_parts = []
        prediction = final_vessel[..., 3]
        target_radius = target[..., 3]
        for first_name, second_name in pair_names:
            first = mapping[first_name]
            second = mapping[second_name]
            pair_valid = point_mask[first] & point_mask[second]
            pair_region = region[first] | region[second]
            predicted_delta = prediction[first] - prediction[second]
            target_delta = target_radius[first] - target_radius[second]
            delta_error = predicted_delta - target_delta
            inside_mask = pair_valid & pair_region
            outside_mask = pair_valid & ~pair_region
            paired_parts.append(
                _masked_weighted_mean(
                    F.smooth_l1_loss(
                        predicted_delta / radius_scale,
                        target_delta / radius_scale,
                        reduction="none",
                    ),
                    inside_mask,
                )
            )
            outside_parts.append(
                _masked_weighted_mean(
                    F.smooth_l1_loss(
                        predicted_delta / radius_scale,
                        target_delta / radius_scale,
                        reduction="none",
                    ),
                    outside_mask,
                )
            )
            paired_mae_parts.append(
                _masked_weighted_mean(delta_error.abs(), inside_mask)
            )
            outside_mae_parts.append(
                _masked_weighted_mean(delta_error.abs(), outside_mask)
            )
        paired_loss = torch.stack(paired_parts).mean()
        outside_loss = torch.stack(outside_parts).mean()
        paired_mae_mm = torch.stack(paired_mae_parts).mean()
        outside_mae_mm = torch.stack(outside_mae_parts).mean()

    conservative_loss = final_radius_loss.new_zeros(())
    conservative_mask = torch.zeros_like(point_mask)
    if group_type == NATURAL_NEGATIVE:
        conservative_mask = point_mask
    elif group_type == INVISIBLE_STENOSIS:
        conservative_mask = point_mask & region
    if bool(conservative_mask.any().item()):
        conservative_loss = _masked_weighted_mean(
            F.smooth_l1_loss(
                final_vessel[..., 3] / radius_scale,
                coarse_vessel[..., 3].detach() / radius_scale,
                reduction="none",
            ),
            conservative_mask,
        )

    whole_dice_enabled = _strict_bool(
        config, "radius_refiner_whole_artery_dice_enabled", False
    )
    whole_dice = final_radius_loss.new_zeros(())
    whole_dice_score = final_radius_loss.new_ones(())
    if whole_dice_enabled:
        for required_key in (
            "radius_refiner_final_rendered_masks",
            "images",
            "view_mask",
        ):
            source = output if required_key in output else batch
            if required_key not in source or source[required_key] is None:
                raise KeyError(f"Whole-artery Dice requires {required_key!r}.")
        whole_dice, whole_dice_score = whole_artery_dice_loss(
            predicted_masks=output["radius_refiner_final_rendered_masks"],
            target_images=batch["images"].to(device=device),
            view_mask=batch["view_mask"].to(device=device),
            threshold=float(config.get("projection_mask_threshold", 0.5)),
        )

    local_dice_enabled = _strict_bool(
        config, "radius_refiner_local_stenosis_dice_enabled", True
    )
    local_dice = final_radius_loss.new_zeros(())
    local_dice_score = final_radius_loss.new_ones(())
    local_dice_view_count = final_radius_loss.new_zeros(())
    if local_dice_enabled and group_type == VISIBLE_AUGMENTED:
        for required_key in (
            "radius_refiner_final_rendered_masks",
            "images",
            "view_mask",
            "stenosis_view_visible_mask",
        ):
            source = output if required_key in output else batch
            if required_key not in source or source[required_key] is None:
                raise KeyError(
                    f"Local stenosis Dice requires {required_key!r}."
                )
        local_dice, local_dice_score, local_dice_view_count = (
            local_stenosis_dice_loss(
                predicted_masks=output["radius_refiner_final_rendered_masks"],
                target_images=batch["images"].to(device=device),
                view_mask=batch["view_mask"].to(device=device),
                group_visible_view_mask=batch["stenosis_view_visible_mask"].to(
                    device=device
                ),
                variants=variants,
                dilation_px=int(
                    _option(
                        config,
                        "radius_refiner_local_stenosis_dice_roi_dilation_px",
                        8,
                    )
                ),
                threshold=float(config.get("projection_mask_threshold", 0.5)),
                visible_views_only=_strict_bool(
                    config,
                    "radius_refiner_local_stenosis_dice_visible_views_only",
                    True,
                ),
            )
        )
        selected_visible = batch["stenosis_view_visible_mask"].to(
            device=device, dtype=torch.bool
        ) & batch["view_mask"].to(device=device, dtype=torch.bool)
        if bool(selected_visible.any().item()) and int(
            local_dice_view_count.detach().item()
        ) == 0:
            raise RuntimeError(
                "A Stage-3.1-visible selected view produced an empty local Dice "
                "ROI. Check that Stage-3.1 and training use the same mask "
                "threshold and that Stage-3.5 preserved the projection masks."
            )

    weights = {
        "final": _finite_nonnegative(
            config, "radius_refiner_final_radius_loss_weight", 1.0
        ),
        "intermediate": _finite_nonnegative(
            config, "radius_refiner_intermediate_radius_loss_weight", 0.25
        ),
        "paired": _finite_nonnegative(
            config, "radius_refiner_paired_difference_loss_weight", 1.0
        ),
        "outside": _finite_nonnegative(
            config, "radius_refiner_outside_consistency_loss_weight", 0.1
        ),
        "conservative": _finite_nonnegative(
            config, "radius_refiner_conservative_residual_loss_weight", 0.05
        ),
        "whole_dice": _finite_nonnegative(
            config, "radius_refiner_whole_artery_dice_loss_weight", 0.1
        ),
        "local_dice": _finite_nonnegative(
            config, "radius_refiner_local_stenosis_dice_loss_weight", 0.25
        ),
    }
    weighted = {
        "final": weights["final"] * final_radius_loss,
        "intermediate": weights["intermediate"] * intermediate_loss,
        "paired": weights["paired"] * paired_loss,
        "outside": weights["outside"] * outside_loss,
        "conservative": weights["conservative"] * conservative_loss,
        "whole_dice": (
            weights["whole_dice"] * whole_dice
            if whole_dice_enabled
            else whole_dice.new_zeros(())
        ),
        "local_dice": (
            weights["local_dice"] * local_dice
            if local_dice_enabled
            else local_dice.new_zeros(())
        ),
    }
    total = sum(weighted.values(), final_radius_loss.new_zeros(()))

    absolute_error = (final_vessel[..., 3] - target[..., 3]).abs()
    coarse_absolute_error = (coarse_vessel[..., 3] - target[..., 3]).abs()
    global_mae = _masked_weighted_mean(absolute_error, point_mask)
    coarse_global_mae = _masked_weighted_mean(coarse_absolute_error, point_mask)
    stenosis_mae = (
        _masked_weighted_mean(absolute_error, region)
        if bool(region.any().item())
        else global_mae.new_zeros(())
    )
    coarse_stenosis_mae = (
        _masked_weighted_mean(coarse_absolute_error, region)
        if bool(region.any().item())
        else global_mae.new_zeros(())
    )
    outside_mae = _masked_weighted_mean(absolute_error, point_mask & ~region)
    residual_abs = (
        residuals.abs().mean()
        if residuals is not None
        else global_mae.new_zeros(())
    )
    return {
        "loss": total,
        "radius_refiner_final_radius_loss": final_radius_loss,
        "radius_refiner_final_radius_weighted_loss": weighted["final"],
        "radius_refiner_intermediate_radius_loss": intermediate_loss,
        "radius_refiner_intermediate_radius_weighted_loss": weighted[
            "intermediate"
        ],
        "radius_refiner_paired_difference_loss": paired_loss,
        "radius_refiner_paired_difference_weighted_loss": weighted["paired"],
        "radius_refiner_outside_consistency_loss": outside_loss,
        "radius_refiner_outside_consistency_weighted_loss": weighted["outside"],
        "radius_refiner_conservative_residual_loss": conservative_loss,
        "radius_refiner_conservative_residual_weighted_loss": weighted[
            "conservative"
        ],
        "radius_refiner_whole_artery_dice_loss": whole_dice,
        "radius_refiner_whole_artery_dice_weighted_loss": weighted[
            "whole_dice"
        ],
        "radius_refiner_whole_artery_dice_score": whole_dice_score,
        "radius_refiner_local_stenosis_dice_loss": local_dice,
        "radius_refiner_local_stenosis_dice_weighted_loss": weighted[
            "local_dice"
        ],
        "radius_refiner_local_stenosis_dice_score": local_dice_score,
        "radius_refiner_local_stenosis_dice_view_count": local_dice_view_count.to(
            dtype=total.dtype
        ),
        "radius_refiner_radius_mae_mm": global_mae,
        "radius_refiner_coarse_radius_mae_mm": coarse_global_mae,
        "radius_refiner_radius_mae_improvement_mm": coarse_global_mae
        - global_mae,
        "radius_refiner_stenosis_interval_mae_mm": stenosis_mae,
        "radius_refiner_coarse_stenosis_interval_mae_mm": coarse_stenosis_mae,
        "radius_refiner_stenosis_interval_mae_improvement_mm": (
            coarse_stenosis_mae - stenosis_mae
        ),
        "radius_refiner_outside_interval_mae_mm": outside_mae,
        "radius_refiner_paired_difference_mae_mm": paired_mae_mm,
        "radius_refiner_outside_difference_mae_mm": outside_mae_mm,
        "radius_refiner_residual_mean_mm": residual_abs,
    }
