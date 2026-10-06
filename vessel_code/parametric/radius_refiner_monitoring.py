# Transferred from methods/parametric_methods/radius_refiner_monitoring.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from vessel_code.parametric.radius_refiner_loss import VISIBLE_AUGMENTED, build_counterfactual_changed_mask_roi
from vessel_code.shared.visualization import save_3d_radius_colored_comparison_gif

matplotlib.use("Agg")

def _as_target_masks(images: torch.Tensor) -> np.ndarray:
    value = images.detach().cpu().numpy()
    if value.ndim == 5:
        value = value[:, :, 0]
    if value.ndim != 4:
        raise ValueError(f"Expected monitor images [B,V,H,W], got {value.shape}.")
    return value

def _ordered_counterfactual_member_indices(variants: list[str]) -> list[int]:
    """Return triplet members in the human-readable monitor row order."""

    preferred_order = (
        "original",
        "stenosis_removed",
        "stenosis_strengthened",
    )
    mapping = {str(variant): index for index, variant in enumerate(variants)}
    ordered = [mapping[name] for name in preferred_order if name in mapping]
    ordered.extend(index for index in range(len(variants)) if index not in ordered)
    return ordered

def _save_counterfactual_2d_overlay_grid(
    *,
    target_masks: np.ndarray,
    rendered_masks: np.ndarray,
    variants: list[str],
    selected_view_indices: np.ndarray,
    visible_views: np.ndarray,
    roi: np.ndarray,
    path: Path,
    threshold: float,
) -> None:
    """Plot final rendered-vs-input masks with variants as rows and views as columns."""

    if target_masks.shape != rendered_masks.shape:
        raise ValueError(
            "Target and rendered masks must have identical [B,V,H,W] shapes, "
            f"got {target_masks.shape} and {rendered_masks.shape}."
        )
    if target_masks.ndim != 4:
        raise ValueError(
            f"Counterfactual overlays require [B,V,H,W], got {target_masks.shape}."
        )
    num_members, num_views = target_masks.shape[:2]
    if len(variants) != num_members:
        raise ValueError(
            f"Expected {num_members} counterfactual variant names, got {variants}."
        )
    view_indices = np.asarray(selected_view_indices, dtype=np.int64).reshape(-1)
    visible = np.asarray(visible_views, dtype=bool).reshape(-1)
    if view_indices.size != num_views or visible.size != num_views:
        raise ValueError(
            "Selected view indices and visibility must match the selected mask "
            f"count {num_views}, got {view_indices.size} and {visible.size}."
        )
    ordered_members = _ordered_counterfactual_member_indices(variants)
    if not ordered_members or num_views == 0:
        return

    figure, axes = plt.subplots(
        len(ordered_members),
        num_views,
        figsize=(3.2 * num_views, 3.0 * len(ordered_members)),
        squeeze=False,
        dpi=150,
    )
    for row, member_index in enumerate(ordered_members):
        variant = variants[member_index]
        for column in range(num_views):
            target = target_masks[member_index, column] > threshold
            prediction = rendered_masks[member_index, column] > threshold
            overlap = target & prediction
            target_only = target & ~prediction
            prediction_only = prediction & ~target
            rgb = np.zeros((*target.shape, 3), dtype=np.float32)
            rgb[target_only] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            rgb[overlap] = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            rgb[prediction_only] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            axis = axes[row, column]
            axis.imshow(rgb, vmin=0.0, vmax=1.0)
            local_roi = np.asarray(roi[column], dtype=bool)
            if local_roi.any() and not local_roi.all():
                axis.contour(
                    local_roi.astype(np.float32),
                    levels=[0.5],
                    colors=["#ffd54f"],
                    linewidths=0.7,
                )
            visibility = "stenosis visible" if visible[column] else "not visible"
            if row == 0:
                axis.set_title(
                    f"View {int(view_indices[column])}\n{visibility}", fontsize=8
                )
            if column == 0:
                axis.set_ylabel(variant.replace("stenosis_", ""), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Counterfactual 2D input/refined-surface overlays\n"
        "white = target only, green = overlap, red = prediction only, "
        "yellow = stenosis ROI",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def _highlight_stenosis_interval(axis: Any, region: np.ndarray) -> None:
    region_indices = np.flatnonzero(np.asarray(region, dtype=bool))
    if region_indices.size:
        axis.axvspan(
            float(region_indices.min()) - 0.5,
            float(region_indices.max()) + 0.5,
            color="#ffd54f",
            alpha=0.24,
            label="stenosis interval",
        )

def _save_variant_radius_profiles_and_errors(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    branch_exists: np.ndarray,
    point_valid: np.ndarray,
    region: np.ndarray,
    variants: list[str],
    output_dir: Path,
    epoch: int,
) -> None:
    """Save three uncluttered GT/prediction profiles and one error comparison."""

    active_branches = np.flatnonzero(branch_exists.any(axis=0))
    if not active_branches.size:
        return
    mapping = {variant: index for index, variant in enumerate(variants)}
    ordered_variants = (
        "original",
        "stenosis_removed",
        "stenosis_strengthened",
    )
    colors = {
        "original": "#ff7f0e",
        "stenosis_removed": "#1f77b4",
        "stenosis_strengthened": "#d62728",
    }
    for variant in ordered_variants:
        if variant not in mapping:
            continue
        member_index = mapping[variant]
        figure, axes = plt.subplots(
            len(active_branches),
            1,
            figsize=(13, max(3.8, 3.5 * len(active_branches))),
            squeeze=False,
            dpi=150,
        )
        for row, branch_index in enumerate(active_branches.tolist()):
            axis = axes[row, 0]
            valid = (
                point_valid[member_index, branch_index]
                & branch_exists[member_index, branch_index]
            )
            indices = np.flatnonzero(valid)
            if indices.size:
                target_radius = target[member_index, branch_index, indices, 3]
                predicted_radius = prediction[
                    member_index, branch_index, indices, 3
                ]
                mae = float(np.mean(np.abs(predicted_radius - target_radius)))
                axis.plot(
                    indices,
                    target_radius,
                    color="#2ca02c",
                    linewidth=2.0,
                    label="Ground truth",
                )
                axis.plot(
                    indices,
                    predicted_radius,
                    color="#d62728",
                    linewidth=1.6,
                    linestyle="--",
                    label=f"Prediction (MAE {mae:.3f} mm)",
                )
            _highlight_stenosis_interval(
                axis, region[member_index, branch_index]
            )
            axis.set_title(f"{variant} — Branch {branch_index}")
            axis.set_xlabel("Ordered centreline point index")
            axis.set_ylabel("Radius (mm)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.suptitle(
            f"Epoch {epoch}: {variant} ground-truth versus refined radius"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"radius_profile_{variant}_gt_vs_prediction.png",
            bbox_inches="tight",
        )
        plt.close(figure)

    figure, axes = plt.subplots(
        len(active_branches),
        1,
        figsize=(13, max(3.8, 3.5 * len(active_branches))),
        squeeze=False,
        dpi=150,
    )
    for row, branch_index in enumerate(active_branches.tolist()):
        axis = axes[row, 0]
        for variant in ordered_variants:
            if variant not in mapping:
                continue
            member_index = mapping[variant]
            valid = (
                point_valid[member_index, branch_index]
                & branch_exists[member_index, branch_index]
            )
            indices = np.flatnonzero(valid)
            if not indices.size:
                continue
            absolute_error = np.abs(
                prediction[member_index, branch_index, indices, 3]
                - target[member_index, branch_index, indices, 3]
            )
            axis.plot(
                indices,
                absolute_error,
                color=colors[variant],
                linewidth=1.7,
                label=variant,
            )
        _highlight_stenosis_interval(axis, region[:, branch_index].any(axis=0))
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(f"Branch {branch_index}: absolute radius error")
        axis.set_xlabel("Ordered centreline point index")
        axis.set_ylabel("Absolute error (mm)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Epoch {epoch}: radius prediction error for all counterfactual variants"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "radius_prediction_absolute_errors.png",
        bbox_inches="tight",
    )
    plt.close(figure)

def _save_local_dice_panels(
    *,
    target_masks: np.ndarray,
    rendered_masks: np.ndarray,
    roi: np.ndarray,
    visible_views: np.ndarray,
    variants: list[str],
    path: Path,
    max_views: int,
) -> None:
    selected_views = np.flatnonzero(visible_views & roi.reshape(roi.shape[0], -1).any(1))
    selected_views = selected_views[: max(1, int(max_views))]
    if not selected_views.size:
        return
    rows = len(variants) * len(selected_views)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(12, max(3.0, rows * 2.5)),
        squeeze=False,
        dpi=140,
    )
    row = 0
    for view_index in selected_views.tolist():
        for member_index, variant in enumerate(variants):
            target = np.clip(target_masks[member_index, view_index], 0.0, 1.0)
            prediction = np.clip(
                rendered_masks[member_index, view_index], 0.0, 1.0
            )
            local = roi[view_index]
            panels = (
                (target, "Target mask"),
                (prediction, "Rendered refined mask"),
                (np.abs(target - prediction) * local, "Local absolute error"),
                (local.astype(np.float32), "Counterfactual Dice ROI"),
            )
            for column, (panel, title) in enumerate(panels):
                axes[row, column].imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
                axes[row, column].set_title(
                    f"View {view_index}, {variant}\n{title}", fontsize=7
                )
                axes[row, column].set_axis_off()
            row += 1
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def save_counterfactual_radius_refiner_monitor(
    *,
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    losses: dict[str, torch.Tensor],
    path: Path,
    epoch: int,
    config: dict[str, Any],
) -> None:
    """Save a fixed visible-triplet monitor for the dedicated trainer."""

    if str(batch.get("group_type")) != VISIBLE_AUGMENTED:
        return
    path.mkdir(parents=True, exist_ok=True)
    variants = [str(value) for value in batch["counterfactual_variant"]]
    target = batch["target_raw_vessel_mm"].detach().cpu().numpy()
    prediction = output["decoded_vessel_mm"].detach().cpu().numpy()
    branch_exists = (
        batch["target_branch_exist"].detach().cpu().numpy() > 0.5
    )
    point_valid = (
        batch["target_point_valid_mask"].detach().cpu().numpy().astype(bool)
    )
    region = (
        batch["stenosis_region_point_mask"].detach().cpu().numpy().astype(bool)
    )
    _save_variant_radius_profiles_and_errors(
        target=target,
        prediction=prediction,
        branch_exists=branch_exists,
        point_valid=point_valid,
        region=region,
        variants=variants,
        output_dir=path,
        epoch=epoch,
    )

    target_images = batch["images"]
    roi = build_counterfactual_changed_mask_roi(
        target_images=target_images,
        variants=variants,
        dilation_px=int(
            config.get("loss", {}).get(
                "radius_refiner_local_stenosis_dice_roi_dilation_px",
                config.get(
                    "radius_refiner_local_stenosis_dice_roi_dilation_px", 8
                ),
            )
        ),
        threshold=float(config.get("projection_mask_threshold", 0.5)),
    ).detach().cpu().numpy()
    visible = (
        batch["stenosis_view_visible_mask"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    target_masks = _as_target_masks(target_images)
    rendered_masks = (
        output["radius_refiner_final_rendered_masks"]
        .detach()
        .cpu()
        .numpy()
    )
    _save_counterfactual_2d_overlay_grid(
        target_masks=target_masks,
        rendered_masks=rendered_masks,
        variants=variants,
        selected_view_indices=batch["selected_group_view_indices"]
        .detach()
        .cpu()
        .numpy(),
        visible_views=visible,
        roi=roi,
        path=path / "counterfactual_2d_overlay.png",
        threshold=float(config.get("projection_mask_threshold", 0.5)),
    )
    _save_local_dice_panels(
        target_masks=target_masks,
        rendered_masks=rendered_masks,
        roi=roi,
        visible_views=visible,
        variants=variants,
        path=path / "local_stenosis_dice_views.png",
        max_views=int(config.get("monitor_max_visible_views", 2)),
    )

    if bool(config.get("monitor_generate_gifs", False)):
        frames = int(config.get("monitor_gif_frames", 18))
        fps = int(config.get("monitor_radius_colored_gif_fps", 2))
        for member_index, variant in enumerate(variants):
            save_3d_radius_colored_comparison_gif(
                target[member_index],
                prediction[member_index],
                branch_exists[member_index],
                path / f"{variant}_3d_radius_colored_surfaces.gif",
                num_frames=frames,
                fps=fps,
                point_valid=point_valid[member_index],
                pred_exist=branch_exists[member_index],
                title=f"{variant}: ground truth vs refined radius",
            )

    metrics = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if value.numel() == 1
    }
    metrics.update(
        epoch=int(epoch),
        case_id=str(batch.get("group_case_id", batch.get("case_id", ""))),
        group_type=str(batch.get("group_type", "")),
        counterfactual_variants=variants,
        selected_view_indices=batch["selected_view_indices"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .tolist(),
        selected_visible_view_mask=visible.tolist(),
    )
    if "radius_refiner_precanonical_geometry_p95_mm" in output:
        metrics["precanonical_geometry_p95_mm"] = (
            output["radius_refiner_precanonical_geometry_p95_mm"]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )
        metrics["geometry_canonicalization_applied"] = bool(
            output["radius_refiner_geometry_canonicalization_applied"]
            .detach()
            .max()
            .item()
            > 0.5
        )
    with (path / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
