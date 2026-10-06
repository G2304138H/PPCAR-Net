# Transferred from methods/parametric_methods/monitoring.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio.v2 as imageio
from matplotlib.patches import Patch
from vessel_code.support.centerline_only import only_centerline_from_config
from vessel_code.preprocessing.vessel_code_transform_utils import resolve_save_prediction_npz_files
from vessel_code.shared.branch_visibility import required_branch_count, resolve_paired_visualization_branch_masks
from vessel_code.shared.visualization import _plot_surface_collection, _sanitize_vessel_for_surface, _set_equal_3d_axes, _surface_coords_for_valid_branches, save_2d_centerline_overlay, save_2d_overlay, save_3d_centerline_overlay_gif, save_3d_ground_truth_gif, save_3d_overlay_gif, save_3d_radius_colored_comparison_gif, save_3d_radius_colored_prediction_gif, save_3d_prediction_gif, save_centerline_xyz_error_profile, save_projected_control_point_monitor, save_projection_loss_monitor, save_radius_prediction_profiles
from vessel_code.parametric.projection_loss import build_centerline_projector
from vessel_code.parametric.loss import centerline_prediction_mode_from_config, radius_prediction_mode_from_config

matplotlib.use("Agg")

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return (
        float(numerator / denominator)
        if int(denominator) > 0
        else None
    )

def compute_branch_existence_metrics(
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    *,
    fixed_main_branch_count: int,
) -> dict[str, float | int | None]:
    """Measure optional side-branch classification without fixed main slots."""

    target = np.asarray(target_exists, dtype=bool).reshape(-1)
    predicted = np.asarray(predicted_exists, dtype=bool).reshape(-1)
    if target.shape != predicted.shape:
        raise ValueError(
            "Target and predicted branch-existence arrays must have the same "
            f"shape, got {target.shape} and {predicted.shape}."
        )
    fixed_count = min(max(int(fixed_main_branch_count), 0), target.size)
    side_target = target[fixed_count:]
    side_predicted = predicted[fixed_count:]
    true_positive = int(np.count_nonzero(side_target & side_predicted))
    true_negative = int(np.count_nonzero(~side_target & ~side_predicted))
    false_positive = int(np.count_nonzero(~side_target & side_predicted))
    false_negative = int(np.count_nonzero(side_target & ~side_predicted))
    num_slots = int(side_target.size)
    precision = _optional_ratio(
        true_positive, true_positive + false_positive
    )
    recall = _optional_ratio(true_positive, true_positive + false_negative)
    specificity = _optional_ratio(
        true_negative, true_negative + false_positive
    )
    f1 = _optional_ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    balanced_values = [
        value for value in (recall, specificity) if value is not None
    ]
    return {
        # Preserve the historical all-slot metric for compatibility. Its main
        # slots are fixed and therefore make it optimistic for side detection.
        "branch_exist_accuracy": (
            float((predicted == target).mean()) if target.size else None
        ),
        "side_branch_exist_accuracy": _optional_ratio(
            true_positive + true_negative, num_slots
        ),
        "side_branch_exist_precision": precision,
        "side_branch_exist_recall": recall,
        "side_branch_absence_recall": specificity,
        "side_branch_exist_specificity": specificity,
        "side_branch_exist_f1": f1,
        "side_branch_exist_balanced_accuracy": (
            float(np.mean(balanced_values)) if balanced_values else None
        ),
        "side_branch_exist_num_slots": num_slots,
        "side_branch_exist_true_positive_count": true_positive,
        "side_branch_exist_true_negative_count": true_negative,
        "side_branch_exist_false_positive_count": false_positive,
        "side_branch_exist_false_negative_count": false_negative,
    }

def _save_side_branch_existence_logits_plot(
    *,
    logits: np.ndarray,
    probabilities: np.ndarray,
    target_exists: np.ndarray | None,
    fixed_main_branch_count: int,
    path: Path,
    title: str,
    decision_threshold: float = 0.5,
) -> None:
    """Plot the existence-head decision for every optional side slot.

    When ground-truth labels are unavailable, bars are colored and annotated
    by the predicted decision instead of a TP/TN/FP/FN classification.
    """

    logit_values = np.asarray(logits, dtype=np.float64).reshape(-1)
    probability_values = np.asarray(
        probabilities, dtype=np.float64
    ).reshape(-1)
    if logit_values.shape != probability_values.shape:
        raise ValueError(
            "Branch-existence logits and probabilities must have the same "
            f"shape, got {logit_values.shape} and "
            f"{probability_values.shape}."
        )
    target_values = (
        None
        if target_exists is None
        else np.asarray(target_exists, dtype=bool).reshape(-1)
    )
    if target_values is not None and target_values.shape != logit_values.shape:
        raise ValueError(
            "Branch-existence targets must match the logits/probabilities "
            f"shape, got {target_values.shape} and {logit_values.shape}."
        )
    if not np.isfinite(logit_values).all():
        raise ValueError("Branch-existence logits must be finite.")
    if (
        not np.isfinite(probability_values).all()
        or np.any(probability_values < 0.0)
        or np.any(probability_values > 1.0)
    ):
        raise ValueError(
            "Branch-existence probabilities must be finite and in [0,1]."
        )
    threshold = float(decision_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Branch-existence decision threshold must be finite and in "
            f"[0,1], got {decision_threshold}."
        )

    fixed_count = min(
        max(int(fixed_main_branch_count), 0), int(logit_values.size)
    )
    branch_indices = np.arange(
        fixed_count, int(logit_values.size), dtype=np.int64
    )
    side_logits = logit_values[fixed_count:]
    side_probabilities = probability_values[fixed_count:]
    side_targets = (
        None if target_values is None else target_values[fixed_count:]
    )
    side_predictions = side_probabilities >= threshold

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(
        figsize=(max(6.5, 1.15 * len(branch_indices) + 2.5), 5.2),
        dpi=150,
    )
    if branch_indices.size == 0:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No optional side-branch slots in this checkpoint",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return

    if side_targets is None:
        colors = [
            "#2ca02c" if bool(prediction) else "#4c78a8"
            for prediction in side_predictions
        ]
    else:
        classification_colors = {
            (True, True): "#2ca02c",  # true positive
            (False, False): "#4c78a8",  # true negative
            (False, True): "#ff9f1c",  # false positive
            (True, False): "#d62728",  # false negative
        }
        colors = [
            classification_colors[(bool(target), bool(prediction))]
            for target, prediction in zip(side_targets, side_predictions)
        ]
    x_values = np.arange(len(branch_indices), dtype=np.int64)
    bars = axis.bar(x_values, side_logits, color=colors, width=0.7)
    threshold_handle: plt.Line2D | Patch
    if 0.0 < threshold < 1.0:
        threshold_logit = math.log(threshold / (1.0 - threshold))
        axis.axhline(
            threshold_logit,
            color="black",
            linewidth=1.2,
            linestyle="--",
        )
        threshold_handle = plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label=(
                f"Prediction threshold: logit {threshold_logit:.3g} "
                f"(p={threshold:.3g})"
            ),
        )
    else:
        threshold_handle = Patch(
            facecolor="none",
            edgecolor="none",
            label=f"Prediction threshold: p={threshold:.3g}",
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [
            f"Side {slot_index + 1}\n(B{branch_index})"
            for slot_index, branch_index in enumerate(branch_indices)
        ]
    )
    axis.set_ylabel("Predicted existence logit")
    axis.set_xlabel("Optional side-branch decoder slot")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.margins(y=0.18)
    maximum_magnitude = max(float(np.max(np.abs(side_logits))), 1.0)
    label_offset = 0.04 * maximum_magnitude
    annotation_targets: list[bool | None] = (
        [None] * len(side_logits)
        if side_targets is None
        else [bool(value) for value in side_targets]
    )
    for bar, logit, probability, target, prediction in zip(
        bars,
        side_logits,
        side_probabilities,
        annotation_targets,
        side_predictions,
    ):
        label_y = (
            float(logit) + label_offset
            if logit >= 0.0
            else float(logit) - label_offset
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            (
                f"p={probability:.3f}\n"
                + (
                    f"Pred={'present' if prediction else 'absent'}"
                    if target is None
                    else f"GT={'present' if target else 'absent'}"
                )
            ),
            ha="center",
            va="bottom" if logit >= 0.0 else "top",
            fontsize=8,
        )
    classification_handles = (
        [
            Patch(facecolor="#2ca02c", label="Predicted present"),
            Patch(facecolor="#4c78a8", label="Predicted absent"),
        ]
        if side_targets is None
        else [
            Patch(facecolor="#2ca02c", label="True positive"),
            Patch(facecolor="#4c78a8", label="True negative"),
            Patch(facecolor="#ff9f1c", label="False positive"),
            Patch(facecolor="#d62728", label="False negative"),
        ]
    )
    axis.legend(
        handles=[threshold_handle, *classification_handles],
        loc="best",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def compute_branch_arc_length_metrics(
    predicted_vessel: np.ndarray,
    target_vessel: np.ndarray,
    branch_exists: np.ndarray,
    point_valid: np.ndarray,
    *,
    fixed_main_branch_count: int,
) -> dict[str, Any]:
    """Record overall, main-structure, side, and per-branch arc lengths."""

    predicted = np.asarray(predicted_vessel)
    target = np.asarray(target_vessel)
    exists = np.asarray(branch_exists, dtype=bool).reshape(-1)
    valid_points = np.asarray(point_valid, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            "Predicted and target vessels must share shape [M,N,C], got "
            f"{predicted.shape} and {target.shape}."
        )
    if predicted.shape[-1] < 3:
        raise ValueError("Arc-length metrics require XYZ vessel coordinates.")
    if exists.shape != (predicted.shape[0],):
        raise ValueError(
            "branch_exists must have one value per vessel branch, got "
            f"{exists.shape} for {predicted.shape}."
        )
    if valid_points.shape != predicted.shape[:2]:
        raise ValueError(
            "point_valid must have shape [M,N], got "
            f"{valid_points.shape} for {predicted.shape}."
        )

    segment_valid = (
        exists[:, None]
        & valid_points[..., :-1]
        & valid_points[..., 1:]
    )
    predicted_segments = np.linalg.norm(
        predicted[:, 1:, :3] - predicted[:, :-1, :3], axis=-1
    )
    target_segments = np.linalg.norm(
        target[:, 1:, :3] - target[:, :-1, :3], axis=-1
    )
    predicted_length = np.where(
        segment_valid, predicted_segments, 0.0
    ).sum(axis=-1)
    target_length = np.where(segment_valid, target_segments, 0.0).sum(
        axis=-1
    )
    valid_length = (
        segment_valid.any(axis=-1)
        & np.isfinite(predicted_length)
        & np.isfinite(target_length)
        & (target_length > 1e-6)
    )
    signed_error = predicted_length - target_length
    absolute_error = np.abs(signed_error)
    relative_error = np.divide(
        absolute_error,
        target_length,
        out=np.full_like(absolute_error, np.nan, dtype=np.float64),
        where=target_length > 1e-6,
    )
    fixed_count = min(
        max(int(fixed_main_branch_count), 0), predicted.shape[0]
    )
    main_mask = valid_length.copy()
    main_mask[fixed_count:] = False
    side_mask = valid_length.copy()
    side_mask[:fixed_count] = False

    def grouped(mask: np.ndarray, prefix: str) -> dict[str, float | int | None]:
        count = int(np.count_nonzero(mask))
        key_prefix = f"{prefix}_" if prefix else ""
        return {
            f"{key_prefix}arc_length_num_branches": count,
            f"{key_prefix}arc_length_abs_error_mm": (
                float(absolute_error[mask].mean()) if count else None
            ),
            f"{key_prefix}arc_length_signed_error_mm": (
                float(signed_error[mask].mean()) if count else None
            ),
            f"{key_prefix}arc_length_rel_error": (
                float(relative_error[mask].mean()) if count else None
            ),
            f"{key_prefix}predicted_arc_length_mean_mm": (
                float(predicted_length[mask].mean()) if count else None
            ),
            f"{key_prefix}target_arc_length_mean_mm": (
                float(target_length[mask].mean()) if count else None
            ),
        }

    per_branch = {
        str(branch_index): {
            "branch_index": int(branch_index),
            "branch_role": (
                "main_structure"
                if branch_index < fixed_count
                else "side_branch"
            ),
            "predicted_arc_length_mm": float(predicted_length[branch_index]),
            "target_arc_length_mm": float(target_length[branch_index]),
            "arc_length_signed_error_mm": float(signed_error[branch_index]),
            "arc_length_abs_error_mm": float(absolute_error[branch_index]),
            "arc_length_rel_error": float(relative_error[branch_index]),
        }
        for branch_index in np.flatnonzero(valid_length)
    }
    return {
        **grouped(valid_length, ""),
        **grouped(main_mask, "main_branch"),
        **grouped(side_mask, "side_branch"),
        "arc_length_by_branch": per_branch,
    }

def _save_metric_matrix(metrics: dict[str, Any], path: Path) -> None:
    """Save a compact Tree Monitor table of the main case-level metrics."""

    labels = (
        ("Centreline mean (mm)", "centerline_mean_mm"),
        ("Centreline p95 (mm)", "centerline_p95_mm"),
        ("Centreline max (mm)", "centerline_max_mm"),
        ("Start error (mm)", "centerline_start_mae_mm"),
        ("End error (mm)", "centerline_end_mae_mm"),
        ("Arc-length error (mm)", "arc_length_abs_error_mm"),
        ("Main arc-length error (mm)", "main_branch_arc_length_abs_error_mm"),
        ("Side arc-length error (mm)", "side_branch_arc_length_abs_error_mm"),
        ("Side arc-length relative error", "side_branch_arc_length_rel_error"),
        ("Side existence accuracy", "side_branch_exist_accuracy"),
        ("Side existence precision", "side_branch_exist_precision"),
        ("Side existence recall", "side_branch_exist_recall"),
        ("Side absence recall", "side_branch_absence_recall"),
        ("Side existence F1", "side_branch_exist_f1"),
        ("Side existence TP", "side_branch_exist_true_positive_count"),
        ("Side existence TN", "side_branch_exist_true_negative_count"),
        ("Side existence FP", "side_branch_exist_false_positive_count"),
        ("Side existence FN", "side_branch_exist_false_negative_count"),
        ("Radius MAE (mm)", "radius_mae_mm"),
        ("Radius p95 (mm)", "radius_p95_mm"),
        ("Control-point mean (mm)", "control_point_error_mean_mm"),
        ("Control-point p95 (mm)", "control_point_error_p95_mm"),
        ("Landmark mean (mm)", "landmark_error_mean_mm"),
        ("Landmark p95 (mm)", "landmark_error_p95_mm"),
    )
    rows = []
    for label, key in labels:
        value = metrics.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue
        rows.append((label, f"{float(value):.4f}"))
    if not rows:
        rows.append(("No valid geometry metrics", "N/A"))
    figure, axis = plt.subplots(
        figsize=(7.5, max(2.8, 0.38 * len(rows) + 1.2)), dpi=150
    )
    axis.set_axis_off()
    table = axis.table(
        cellText=rows,
        colLabels=("Metric", "Value"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.68, 0.24),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    axis.set_title("Case evaluation metrics", pad=10)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

_REFINER_STAGE_MONITOR_METRIC_TITLES = {
    "centerline_mean_mm": "Centreline mean error (mm)",
    "centerline_p95_mm": "Centreline p95 error (mm)",
    "centerline_max_mm": "Centreline maximum error (mm)",
    "centerline_start_mae_mm": "Start-point error (mm)",
    "centerline_end_mae_mm": "End-point error (mm)",
    "control_point_mean_mm": "Control-point mean error (mm)",
    "control_point_p95_mm": "Control-point p95 error (mm)",
    "main_branch_centerline_mae_mm": "Main-branch mean error (mm)",
    "side_branch_centerline_mae_mm": "Side-branch mean error (mm)",
    "arc_length_abs_error_mm": "Arc-length absolute error (mm)",
    "arc_length_rel_error": "Arc-length relative error",
    "main_branch_arc_length_abs_error_mm": (
        "Main-structure arc-length absolute error (mm)"
    ),
    "main_branch_arc_length_rel_error": (
        "Main-structure arc-length relative error"
    ),
    "side_branch_arc_length_abs_error_mm": (
        "Side-branch arc-length absolute error (mm)"
    ),
    "side_branch_arc_length_rel_error": (
        "Side-branch arc-length relative error"
    ),
}

def _refiner_stage_geometry_rows(
    *,
    coarse_vessel: np.ndarray,
    stage_vessels: np.ndarray,
    coarse_parameters: np.ndarray,
    stage_parameters: np.ndarray,
    target_vessel: np.ndarray,
    target_parameters: np.ndarray,
    branch_exists: np.ndarray,
    point_valid: np.ndarray,
    fixed_main_branch_count: int = 1,
) -> list[dict[str, Any]]:
    """Compute detached 3D metrics for coarse and learned refiner stages."""

    if stage_vessels.ndim != 4 or stage_parameters.ndim != 4:
        raise ValueError(
            "Per-case refiner stage vessels and parameters must have shapes "
            "[R,M,N,4] and [R,M,K,3]."
        )
    if stage_vessels.shape[0] != stage_parameters.shape[0]:
        raise ValueError(
            "Refiner stage vessels and parameters must have the same number "
            "of stages."
        )
    branch_exists = np.asarray(branch_exists, dtype=bool)
    point_valid = np.asarray(point_valid, dtype=bool)
    valid_points = branch_exists[:, None] & point_valid
    fixed_main_count = min(
        max(int(fixed_main_branch_count), 0), valid_points.shape[0]
    )
    main_valid = np.zeros_like(valid_points, dtype=bool)
    if fixed_main_count > 0:
        main_valid[:fixed_main_count] = valid_points[:fixed_main_count]
    side_valid = valid_points.copy()
    side_valid[:fixed_main_count] = False

    def stage_metrics(
        prediction: np.ndarray,
        parameters: np.ndarray,
    ) -> dict[str, Any]:
        xyz_by_point = np.linalg.norm(
            prediction[..., :3] - target_vessel[..., :3], axis=-1
        )
        xyz = xyz_by_point[valid_points]
        main_xyz = xyz_by_point[main_valid]
        side_xyz = xyz_by_point[side_valid]
        parameter_error = np.linalg.norm(
            parameters - target_parameters, axis=-1
        )[branch_exists]
        start_errors: list[float] = []
        end_errors: list[float] = []
        for branch_index in np.flatnonzero(branch_exists):
            valid_indices = np.flatnonzero(point_valid[branch_index])
            if valid_indices.size:
                start_errors.append(
                    float(xyz_by_point[branch_index, valid_indices[0]])
                )
                end_errors.append(
                    float(xyz_by_point[branch_index, valid_indices[-1]])
                )

        arc_length_metrics = compute_branch_arc_length_metrics(
            prediction,
            target_vessel,
            branch_exists,
            point_valid,
            fixed_main_branch_count=fixed_main_count,
        )
        return {
            "centerline_mean_mm": float(xyz.mean()) if xyz.size else None,
            "centerline_p95_mm": (
                float(np.percentile(xyz, 95)) if xyz.size else None
            ),
            "centerline_max_mm": float(xyz.max()) if xyz.size else None,
            "centerline_start_mae_mm": (
                float(np.mean(start_errors)) if start_errors else None
            ),
            "centerline_end_mae_mm": (
                float(np.mean(end_errors)) if end_errors else None
            ),
            "control_point_mean_mm": (
                float(parameter_error.mean())
                if parameter_error.size
                else None
            ),
            "control_point_p95_mm": (
                float(np.percentile(parameter_error, 95))
                if parameter_error.size
                else None
            ),
            "main_branch_centerline_mae_mm": (
                float(main_xyz.mean()) if main_xyz.size else None
            ),
            "side_branch_centerline_mae_mm": (
                float(side_xyz.mean()) if side_xyz.size else None
            ),
            **arc_length_metrics,
        }

    vessels = [coarse_vessel, *list(stage_vessels)]
    parameters = [coarse_parameters, *list(stage_parameters)]
    return [
        {
            "stage": stage_index,
            "stage_label": (
                "coarse" if stage_index == 0 else f"stage_{stage_index}"
            ),
            **stage_metrics(vessel, parameter),
        }
        for stage_index, (vessel, parameter) in enumerate(
            zip(vessels, parameters)
        )
    ]

def _save_refiner_stage_geometry_plot(
    stage_rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    """Plot coarse-to-final 3D performance for a monitored training case."""

    metric_keys = [
        key
        for key in _REFINER_STAGE_MONITOR_METRIC_TITLES
        if any(
            row.get(key) is not None
            and np.isfinite(float(row[key]))
            for row in stage_rows
        )
    ]
    if not stage_rows or not metric_keys:
        return
    stage_indices = np.asarray(
        [int(row["stage"]) for row in stage_rows], dtype=np.int64
    )
    stage_labels = [
        "Coarse" if stage_index == 0 else str(stage_index)
        for stage_index in stage_indices
    ]
    num_columns = min(3, len(metric_keys))
    num_rows = int(math.ceil(len(metric_keys) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5.2 * num_columns, 3.8 * num_rows),
        dpi=150,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis, metric_key in zip(axes_flat, metric_keys):
        values = np.asarray(
            [
                np.nan if row.get(metric_key) is None else row[metric_key]
                for row in stage_rows
            ],
            dtype=np.float64,
        )
        axis.plot(stage_indices, values, marker="o", linewidth=2.0)
        axis.set_title(
            _REFINER_STAGE_MONITOR_METRIC_TITLES[metric_key], fontsize=9
        )
        axis.set_xticks(stage_indices)
        axis.set_xticklabels(stage_labels)
        axis.set_xlabel("Learned refinement stage")
        axis.set_ylabel("Metric value (lower is better)")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(metric_keys) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)

def _split_history_loss_key(key: str) -> tuple[str, str] | None:
    """Split an epoch-history key into its dataset prefix and loss metric."""

    if key.startswith("train_"):
        prefix, metric = "train", key.removeprefix("train_")
    else:
        validation_match = re.match(r"^(val_k\d+)_(.+)$", key)
        if validation_match is not None:
            prefix, metric = validation_match.groups()
        elif key.startswith("val_mean_"):
            prefix, metric = "val_mean", key.removeprefix("val_mean_")
        elif key == "val_all_losses_loss":
            # Single-validation histories use ``val`` + ``all_losses_loss``.
            # This spelling would otherwise be mistaken for the ``val_all``
            # split used by multi-view validation.
            prefix, metric = "val", "all_losses_loss"
        elif key.startswith("val_all_"):
            prefix, metric = "val_all", key.removeprefix("val_all_")
        elif key.startswith("val_"):
            prefix, metric = "val", key.removeprefix("val_")
        else:
            return None
    if not metric.endswith("_loss") and metric != "loss":
        return None
    return prefix, metric

def _history_loss_plot_spec(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return stable split and panel ordering for the loss-history plot."""

    prefixes: set[str] = set()
    metrics: set[str] = set()
    for row in rows:
        for key, value in row.items():
            split = _split_history_loss_key(key)
            if split is None:
                continue
            try:
                finite = bool(np.isfinite(float(value)))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                continue
            prefix, metric = split
            prefixes.add(prefix)
            metrics.add(metric)

    def prefix_order(prefix: str) -> tuple[int, int, str]:
        if prefix == "train":
            return 0, 0, prefix
        if prefix == "val":
            return 1, 0, prefix
        if prefix == "val_mean":
            return 2, 0, prefix
        match = re.fullmatch(r"val_k(\d+)", prefix)
        if match is not None:
            return 3, int(match.group(1)), prefix
        if prefix == "val_all":
            return 4, 0, prefix
        return 5, 0, prefix

    preferred_metrics = (
        "loss",
        "all_losses_loss",
        "centerline_control_loss",
        "decoded_xyz_loss",
        "decoded_radius_loss",
        "projection_2d_loss",
        "branch_exist_loss",
        "attachment_loss",
        "side_relative_xyz_loss",
    )
    ordered_metrics = [metric for metric in preferred_metrics if metric in metrics]
    ordered_metrics.extend(sorted(metrics.difference(ordered_metrics)))
    return sorted(prefixes, key=prefix_order), ordered_metrics

def _history_prefix_label(prefix: str) -> str:
    if prefix == "val_all":
        return "val (all views)"
    match = re.fullmatch(r"val_k(\d+)", prefix)
    if match is not None:
        return f"val ({match.group(1)} views)"
    return prefix

def plot_history(rows: list[dict[str, Any]], path: Path) -> None:
    """Render SRC-style loss panels instead of overlaying every component."""

    if not rows:
        return
    prefixes, metric_keys = _history_loss_plot_spec(rows)
    if not metric_keys:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(
        [int(row.get("epoch", index + 1)) for index, row in enumerate(rows)]
    )
    title_by_metric = {
        "loss": "Total loss",
        "all_losses_loss": "All-losses objective",
        "centerline_control_loss": "Control-point loss",
        "decoded_xyz_loss": "Decoded XYZ loss",
        "decoded_radius_loss": "Decoded radius loss",
        "projection_2d_loss": "2D projection loss",
        "branch_exist_loss": "Branch-existence loss",
        "attachment_loss": "Attachment loss",
        "side_relative_xyz_loss": "Side-relative XYZ loss",
    }
    num_columns = min(5, len(metric_keys))
    num_rows = int(math.ceil(len(metric_keys) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(22, 4 * num_rows),
        dpi=150,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis, metric in zip(axes_flat, metric_keys):
        for prefix in prefixes:
            history_key = f"{prefix}_{metric}"
            values = np.asarray(
                [row.get(history_key, np.nan) for row in rows],
                dtype=np.float64,
            )
            if not np.isfinite(values).any():
                continue
            axis.plot(
                epochs,
                values,
                label=_history_prefix_label(prefix),
            )
        title = title_by_metric.get(
            metric,
            metric.removesuffix("_loss").replace("_", " ").title()
            + " loss",
        )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
    for axis in axes_flat[len(metric_keys) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

def _save_input_views(images: np.ndarray, path: Path) -> None:
    values = np.asarray(images)
    if values.ndim == 4:
        values = values[:, 0]
    figure, axes = plt.subplots(1, len(values), figsize=(3 * len(values), 3), dpi=140, squeeze=False)
    for index, image in enumerate(values):
        axes[0, index].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, index].set_title(f"view {index}")
        axes[0, index].axis("off")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

def _save_centreline_comparison(
    target: np.ndarray,
    predicted: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    path: Path,
) -> None:
    """Compare target and prediction branches using their selected masks."""
    figure = plt.figure(figsize=(12, 5), dpi=150)
    for panel, (vessel, exists, title) in enumerate(
        (
            (target, target_exists, "target"),
            (predicted, predicted_exists, "prediction"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        for branch_index in range(vessel.shape[0]):
            if branch_index != 0 and not bool(exists[branch_index]):
                continue
            points = vessel[branch_index, :, :3]
            axis.plot(*points.T, linewidth=2.0 if branch_index == 0 else 1.2)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_zlabel("z (mm)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

def _save_centreline_comparison_gif(
    target: np.ndarray,
    predicted: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    point_valid: np.ndarray,
    path: Path,
    num_frames: int,
    fps: int,
    title: str,
) -> None:
    """Save synchronized rotating target/prediction centreline panels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    target_exists = np.asarray(target_exists, dtype=bool)
    predicted_exists = np.asarray(predicted_exists, dtype=bool)
    point_valid = np.asarray(point_valid, dtype=bool)
    branch_count = min(target.shape[0], predicted.shape[0])
    target_active = target_exists[:branch_count].copy()
    prediction_active = predicted_exists[:branch_count].copy()
    if branch_count:
        target_active[0] = True

    target_lines: list[tuple[int, np.ndarray]] = []
    predicted_lines: list[tuple[int, np.ndarray]] = []
    axis_parts: list[np.ndarray] = []
    for branch_index in range(branch_count):
        valid = (
            point_valid[branch_index]
            if branch_index < point_valid.shape[0]
            else np.ones(target.shape[1], dtype=bool)
        )
        if target_active[branch_index]:
            target_line = target[branch_index, valid, :3]
            target_lines.append((branch_index, target_line))
            axis_parts.append(target_line)
        if prediction_active[branch_index]:
            prediction_valid = (
                valid
                if target_active[branch_index] and bool(valid.any())
                else np.ones(predicted.shape[1], dtype=bool)
            )
            predicted_line = predicted[branch_index, prediction_valid, :3]
            predicted_lines.append((branch_index, predicted_line))
            axis_parts.append(predicted_line)

    axis_points = (
        np.concatenate(axis_parts, axis=0)
        if axis_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(branch_count, 1)))
    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_index in range(frame_count):
        figure = plt.figure(figsize=(10.4, 5.2), dpi=120)
        target_axis = figure.add_subplot(1, 2, 1, projection="3d")
        prediction_axis = figure.add_subplot(1, 2, 2, projection="3d")
        for branch_index, line in target_lines:
            target_axis.plot(
                *line.T,
                color=colors[branch_index],
                linewidth=2.2 if branch_index == 0 else 1.6,
            )
        for branch_index, line in predicted_lines:
            prediction_axis.plot(
                *line.T,
                color=colors[branch_index],
                linewidth=2.2 if branch_index == 0 else 1.6,
            )
        azimuth = 360.0 * frame_index / frame_count
        angle = 2.0 * np.pi * frame_index / frame_count
        elevation = 22.0 + 8.0 * np.sin(angle)
        for axis, panel_title in (
            (target_axis, "Target centreline"),
            (prediction_axis, "Predicted centreline"),
        ):
            _set_equal_3d_axes(axis, axis_points)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(panel_title)
            axis.set_axis_off()
        figure.suptitle(title)
        figure.tight_layout(pad=0.4)
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(
            figure.canvas.buffer_rgba(), dtype=np.uint8
        ).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())
        plt.close(figure)
    imageio.mimsave(path, frames, fps=max(1, int(fps)), loop=0)

def _save_whole_centreline_comparison(
    raw: np.ndarray,
    parametric: np.ndarray,
    predicted: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    path: Path,
) -> None:
    """Compare both GT representations with the selected predictions."""
    figure = plt.figure(figsize=(17, 5), dpi=150)
    panels = (
        (raw, target_exists, "raw vessel code"),
        (parametric, target_exists, "GT parametric decoded"),
        (predicted, predicted_exists, "prediction decoded"),
    )
    visible_parts = [
        vessel[branch_index, :, :3]
        for vessel, exists, _ in panels
        for branch_index in range(vessel.shape[0])
        if branch_index == 0 or bool(exists[branch_index])
    ]
    all_points = (
        np.concatenate(visible_parts, axis=0)
        if visible_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    finite = all_points[np.all(np.isfinite(all_points), axis=1)]
    for panel, (vessel, exists, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        for branch_index in range(vessel.shape[0]):
            if branch_index != 0 and not bool(exists[branch_index]):
                continue
            axis.plot(*vessel[branch_index, :, :3].T, linewidth=2.0 if branch_index == 0 else 1.2)
        _set_equal_3d_axes(axis, finite)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_zlabel("z (mm)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

def _save_control_points_2d(
    target: np.ndarray,
    predicted: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    path: Path,
    parameter_label: str = "control point",
    absolute_parameters: bool = False,
) -> None:
    target_exists = np.asarray(target_exists, dtype=bool)
    predicted_exists = np.asarray(predicted_exists, dtype=bool)
    active = np.flatnonzero(target_exists | predicted_exists)
    if active.size == 0:
        return
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(
        len(active),
        len(planes),
        figsize=(5.0 * len(planes), 4.5 * len(active)),
        dpi=150,
        squeeze=False,
    )
    for row, branch_index in enumerate(active):
        frame = (
            "global"
            if absolute_parameters or branch_index == 0
            else "attachment-relative"
        )
        for column, (horizontal, vertical, horizontal_name, vertical_name) in enumerate(planes):
            axis = axes[row, column]
            gt = target[branch_index]
            pred = predicted[branch_index]
            if target_exists[branch_index]:
                axis.plot(
                    gt[:, horizontal],
                    gt[:, vertical],
                    "o-",
                    color="#2ca02c",
                    linewidth=1.4,
                    markersize=4,
                    label=f"GT {parameter_label}s",
                )
                for control_index, point in enumerate(gt):
                    axis.annotate(
                        str(control_index),
                        (point[horizontal], point[vertical]),
                        color="#187a18",
                        fontsize=6,
                        xytext=(2, 2),
                        textcoords="offset points",
                    )
            if predicted_exists[branch_index]:
                axis.plot(
                    pred[:, horizontal],
                    pred[:, vertical],
                    "x--",
                    color="#d62728",
                    linewidth=1.2,
                    markersize=5,
                    label=f"Pred {parameter_label}s",
                )
                for control_index, point in enumerate(pred):
                    axis.annotate(
                        str(control_index),
                        (point[horizontal], point[vertical]),
                        color="#9d1b1b",
                        fontsize=6,
                        xytext=(2, -7),
                        textcoords="offset points",
                    )
            axis.set_title(
                f"Branch {branch_index} ({frame}): {horizontal_name.upper()}{vertical_name.upper()}"
            )
            axis.set_xlabel(f"{horizontal_name} (mm)")
            axis.set_ylabel(f"{vertical_name} (mm)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.legend(fontsize=8)
    figure.suptitle(f"GT versus predicted centreline {parameter_label}s")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def _save_centreline_control_points_3d_gif(
    target_centerline: np.ndarray,
    predicted_centerline: np.ndarray,
    target_controls_global: np.ndarray,
    predicted_controls_global: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    point_valid: np.ndarray,
    path: Path,
    num_frames: int,
    fps: int,
    parameter_label: str = "control point",
) -> None:
    target_exists = np.asarray(target_exists, dtype=bool)
    predicted_exists = np.asarray(predicted_exists, dtype=bool)
    active = np.flatnonzero(target_exists | predicted_exists)
    if active.size == 0:
        return
    target_active = np.flatnonzero(target_exists)
    predicted_active = np.flatnonzero(predicted_exists)
    first_target_branch = (
        int(target_active[0]) if target_active.size else None
    )
    first_predicted_branch = (
        int(predicted_active[0]) if predicted_active.size else None
    )
    axis_parts: list[np.ndarray] = []
    for branch_index in active:
        valid = np.asarray(point_valid[branch_index], dtype=bool)
        if target_exists[branch_index]:
            axis_parts.append(target_controls_global[branch_index])
            axis_parts.append(target_centerline[branch_index, valid, :3])
        if predicted_exists[branch_index]:
            prediction_valid = (
                valid
                if target_exists[branch_index] and bool(valid.any())
                else np.ones(predicted_centerline.shape[1], dtype=bool)
            )
            axis_parts.append(predicted_controls_global[branch_index])
            axis_parts.append(
                predicted_centerline[branch_index, prediction_valid, :3]
            )
    all_points = np.concatenate(axis_parts, axis=0)
    frames: list[np.ndarray] = []
    for frame_index in range(max(1, int(num_frames))):
        figure = plt.figure(figsize=(7.4, 6.6), dpi=120)
        axis = figure.add_subplot(111, projection="3d")
        for branch_index in active:
            valid = np.asarray(point_valid[branch_index], dtype=bool)
            gt_line = target_centerline[branch_index, valid, :3]
            prediction_valid = (
                valid
                if target_exists[branch_index] and bool(valid.any())
                else np.ones(predicted_centerline.shape[1], dtype=bool)
            )
            pred_line = predicted_centerline[
                branch_index, prediction_valid, :3
            ]
            gt_controls = target_controls_global[branch_index]
            pred_controls = predicted_controls_global[branch_index]
            if target_exists[branch_index]:
                axis.plot(
                    *gt_line.T,
                    color="#187a18",
                    linewidth=2.1,
                    label=(
                        "GT parametric centreline"
                        if branch_index == first_target_branch
                        else None
                    ),
                )
                axis.scatter(
                    *gt_controls.T,
                    color="#39d353",
                    edgecolors="#0c5f18",
                    marker="o",
                    s=32,
                    depthshade=False,
                    label=(
                        f"GT {parameter_label}s"
                        if branch_index == first_target_branch
                        else None
                    ),
                )
            if predicted_exists[branch_index]:
                axis.plot(
                    *pred_line.T,
                    color="#b22222",
                    linewidth=1.8,
                    linestyle="--",
                    label=(
                        "Pred centreline"
                        if branch_index == first_predicted_branch
                        else None
                    ),
                )
                axis.scatter(
                    *pred_controls.T,
                    color="#ff3b30",
                    marker="x",
                    s=35,
                    depthshade=False,
                    label=(
                        f"Pred {parameter_label}s"
                        if branch_index == first_predicted_branch
                        else None
                    ),
                )
            if target_exists[branch_index] and predicted_exists[branch_index]:
                for control_index, (gt_point, pred_point) in enumerate(
                    zip(gt_controls, pred_controls)
                ):
                    connector = np.stack((gt_point, pred_point), axis=0)
                    axis.plot(
                        *connector.T,
                        color="#777777",
                        linewidth=0.6,
                        alpha=0.45,
                    )
                    midpoint = 0.5 * (gt_point + pred_point)
                    label = (
                        f"{'L' if parameter_label == 'landmark' else 'C'}"
                        f"{control_index}"
                        if len(active) == 1
                        else (
                            f"B{branch_index}:"
                            f"{'L' if parameter_label == 'landmark' else 'C'}"
                            f"{control_index}"
                        )
                    )
                    axis.text(*midpoint, label, fontsize=6, color="#222222")
        _set_equal_3d_axes(axis, all_points)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_zlabel("z (mm)")
        axis.set_title(f"3D centrelines and {parameter_label}s")
        axis.legend(loc="upper right", fontsize=7)
        angle = 2.0 * np.pi * frame_index / max(1, int(num_frames))
        axis.view_init(
            elev=22.0 + 8.0 * np.sin(angle),
            azim=360.0 * frame_index / max(1, int(num_frames)),
        )
        figure.tight_layout()
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height, width, 4
        )[..., :3]
        frames.append(frame.copy())
        plt.close(figure)
    imageio.mimsave(path, frames, fps=max(1, int(fps)), loop=0)

def _save_control_point_error_by_index(
    target: np.ndarray,
    predicted: np.ndarray,
    branch_exists: np.ndarray,
    path: Path,
    parameter_label: str = "control point",
    absolute_parameters: bool = False,
) -> np.ndarray:
    errors = np.linalg.norm(predicted - target, axis=-1)
    active = np.flatnonzero(np.asarray(branch_exists, dtype=bool))
    if active.size == 0:
        return errors
    figure, axes = plt.subplots(
        len(active),
        1,
        figsize=(max(9.0, 0.48 * errors.shape[1]), 3.8 * len(active)),
        dpi=150,
        squeeze=False,
    )
    control_indices = np.arange(errors.shape[1])
    for row, branch_index in enumerate(active):
        axis = axes[row, 0]
        values = errors[branch_index]
        bars = axis.bar(control_indices, values, color="#4c78a8", alpha=0.85)
        axis.axhline(
            float(values.mean()),
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"mean {values.mean():.2f} mm",
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
            )
        frame = (
            "global"
            if absolute_parameters or branch_index == 0
            else "attachment-relative"
        )
        axis.set_title(
            f"Branch {branch_index} ({frame}) {parameter_label} distance error"
        )
        axis.set_xlabel(f"{parameter_label.title()} index")
        axis.set_ylabel("Euclidean error (mm)")
        axis.set_xticks(control_indices)
        index_prefix = "L" if parameter_label == "landmark" else "C"
        axis.set_xticklabels(
            [f"{index_prefix}{index}" for index in control_indices], rotation=45
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return errors

_CONTROL_METRIC_PATTERN = re.compile(
    r"^(?P<kind>control_point|landmark)_b(?P<branch>\d+)_"
    r"(?:c|l)(?P<control>\d+)_(?P<stat>mae|p95)_mm$"
)

def save_control_point_error_summary(
    *,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    path: Path,
    epoch: int,
    absolute_parameters: bool = False,
) -> None:
    parsed: dict[str, dict[int, dict[int, float]]] = {
        "train_mae": {},
        "train_p95": {},
        "val_mae": {},
        "val_p95": {},
    }
    parameter_kind = "control_point"
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            match = _CONTROL_METRIC_PATTERN.match(key)
            if match is None:
                continue
            parameter_kind = str(match.group("kind"))
            branch = int(match.group("branch"))
            control = int(match.group("control"))
            series = parsed[f"{split}_{match.group('stat')}"]
            series.setdefault(branch, {})[control] = float(value)
    branches = sorted(set(parsed["train_mae"]) | set(parsed["val_mae"]))
    if not branches:
        return
    figure, axes = plt.subplots(
        len(branches),
        1,
        figsize=(11.0, 4.0 * len(branches)),
        dpi=150,
        squeeze=False,
    )
    for row, branch in enumerate(branches):
        axis = axes[row, 0]
        controls = sorted(
            set(parsed["train_mae"].get(branch, {}))
            | set(parsed["val_mae"].get(branch, {}))
        )
        x = np.asarray(controls, dtype=np.int64)
        for split, color in (("train", "#1f77b4"), ("val", "#ff7f0e")):
            means = np.asarray(
                [parsed[f"{split}_mae"].get(branch, {}).get(index, np.nan) for index in controls]
            )
            p95 = np.asarray(
                [parsed[f"{split}_p95"].get(branch, {}).get(index, np.nan) for index in controls]
            )
            axis.plot(x, means, "o-", color=color, linewidth=1.5, label=f"{split} mean")
            axis.plot(x, p95, "--", color=color, linewidth=1.0, alpha=0.75, label=f"{split} p95")
        frame = (
            "global"
            if absolute_parameters or branch == 0
            else "attachment-relative"
        )
        parameter_label = (
            "landmark" if parameter_kind == "landmark" else "control point"
        )
        axis.set_title(
            f"Epoch {epoch}: branch {branch} ({frame}) {parameter_label} errors"
        )
        axis.set_xlabel(f"{parameter_label.title()} index")
        axis.set_ylabel("Euclidean error (mm)")
        axis.set_xticks(x)
        index_prefix = "L" if parameter_kind == "landmark" else "C"
        axis.set_xticklabels(
            [f"{index_prefix}{index}" for index in controls], rotation=45
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def _save_three_way_surface_gif(
    raw: np.ndarray,
    parametric: np.ndarray,
    predicted: np.ndarray,
    target_exists: np.ndarray,
    predicted_exists: np.ndarray,
    path: Path,
    num_frames: int,
    fps: int,
    title: str,
) -> None:
    vessels = [
        _sanitize_vessel_for_surface(raw),
        _sanitize_vessel_for_surface(parametric, fallback=raw),
        _sanitize_vessel_for_surface(predicted, fallback=raw),
    ]
    target_valid_branches = [
        index for index in range(raw.shape[0]) if index == 0 or bool(target_exists[index])
    ]
    prediction_valid_branches = [
        index
        for index in range(predicted.shape[0])
        if index == 0 or bool(predicted_exists[index])
    ]
    surface_sets = [
        _surface_coords_for_valid_branches(
            vessels[0], target_valid_branches, num_circle_points=24
        ),
        _surface_coords_for_valid_branches(
            vessels[1], target_valid_branches, num_circle_points=24
        ),
        _surface_coords_for_valid_branches(
            vessels[2], prediction_valid_branches, num_circle_points=24
        ),
    ]
    all_surfaces = [surface for surfaces in surface_sets for surface in surfaces]
    if all_surfaces:
        reference_points = np.concatenate(
            [surface.reshape(-1, 3) for surface in surface_sets[0] or all_surfaces], axis=0
        )
        center = reference_points[np.all(np.isfinite(reference_points), axis=1)].mean(axis=0)
        surface_sets = [
            [np.asarray(surface, dtype=np.float32) - center for surface in surfaces]
            for surfaces in surface_sets
        ]
        axis_points = np.concatenate(
            [surface.reshape(-1, 3) for surfaces in surface_sets for surface in surfaces], axis=0
        )
    else:
        axis_points = np.zeros((0, 3), dtype=np.float32)
    colors = ("#2ca02c", "#1f77b4", "#d62728")
    labels = ("Raw surface", "GT parametric surface", "Prediction surface")
    frames: list[np.ndarray] = []
    for frame_index in range(max(1, int(num_frames))):
        figure = plt.figure(figsize=(5.4, 5.4), dpi=120)
        axis = figure.add_subplot(111, projection="3d")
        for surfaces, color, zorder in zip(surface_sets, colors, (3, 2, 1)):
            _plot_surface_collection(axis, surfaces, color=color, alpha=0.36, zorder=zorder)
        _set_equal_3d_axes(axis, axis_points)
        axis.set_title(title)
        axis.view_init(elev=22.0, azim=360.0 * frame_index / max(1, int(num_frames)))
        axis.set_axis_off()
        axis.legend(
            handles=[Patch(facecolor=color, alpha=0.36, label=label) for color, label in zip(colors, labels)],
            loc="upper right",
            fontsize=7,
        )
        figure.tight_layout(pad=0.0)
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())
        plt.close(figure)
    imageio.mimsave(path, frames, fps=max(1, int(fps)), loop=0)

def _save_radius_refinement_profiles(
    *,
    target: np.ndarray,
    coarse: np.ndarray,
    stages: np.ndarray,
    branch_exists: np.ndarray,
    point_valid: np.ndarray,
    path: Path,
) -> None:
    active = np.flatnonzero(np.asarray(branch_exists, dtype=bool))
    if active.size == 0:
        active = np.asarray([0], dtype=np.int64)
    figure, axes = plt.subplots(
        len(active),
        1,
        figsize=(12.0, 3.6 * len(active)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    stage_colors = plt.cm.viridis(
        np.linspace(0.20, 0.85, max(int(stages.shape[0]), 1))
    )
    indices = np.arange(target.shape[1])
    for row, branch_index in enumerate(active):
        axis = axes[row, 0]
        valid = point_valid[branch_index]
        axis.plot(
            indices,
            np.where(valid, target[branch_index, :, 3], np.nan),
            color="#2ca02c",
            linewidth=2.0,
            label="Ground truth",
        )
        axis.plot(
            indices,
            np.where(valid, coarse[branch_index, :, 3], np.nan),
            color="#7f7f7f",
            linewidth=1.4,
            linestyle="--",
            label="Coarse radius",
        )
        for stage_index in range(int(stages.shape[0])):
            is_final = stage_index == int(stages.shape[0]) - 1
            axis.plot(
                indices,
                np.where(
                    valid,
                    stages[stage_index, branch_index, :, 3],
                    np.nan,
                ),
                color="#d62728" if is_final else stage_colors[stage_index],
                linewidth=1.8 if is_final else 1.0,
                alpha=1.0 if is_final else 0.75,
                label=(
                    "Final refined radius"
                    if is_final
                    else f"Intermediate stage {stage_index + 1}"
                ),
            )
        axis.set_title(f"Branch {branch_index}")
        axis.set_ylabel("Radius (mm)")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("Ordered centreline point index")
    figure.suptitle("Radius refinement by stage")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def _save_radius_rendered_mask_comparison(
    *,
    images: np.ndarray,
    view_mask: np.ndarray,
    coarse_masks: np.ndarray,
    final_masks: np.ndarray,
    path: Path,
) -> None:
    input_masks = images[:, 0] if images.ndim == 4 else images
    active_views = np.flatnonzero(np.asarray(view_mask, dtype=bool))
    if active_views.size == 0:
        return
    figure, axes = plt.subplots(
        len(active_views),
        4,
        figsize=(12.0, 3.0 * len(active_views)),
        dpi=150,
        squeeze=False,
    )
    for row, view_index in enumerate(active_views):
        target = np.clip(input_masks[view_index], 0.0, 1.0)
        coarse = np.clip(coarse_masks[view_index], 0.0, 1.0)
        final = np.clip(final_masks[view_index], 0.0, 1.0)
        panels = (
            (target, "Input information mask"),
            (coarse, "Coarse rendered surface"),
            (final, "Final rendered surface"),
            (np.abs(target - final), "Absolute final mask difference"),
        )
        for column, (panel, title) in enumerate(panels):
            axes[row, column].imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(
                f"View {view_index}: {title}",
                fontsize=8,
            )
            axes[row, column].set_axis_off()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

def _save_minimal_refiner_stage_visualization(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    supervision_target: np.ndarray,
    target_vis_exists: np.ndarray,
    predicted_vis_exists: np.ndarray,
    point_valid: np.ndarray,
    batch: dict[str, Any],
    item_index: int,
    path: Path,
    view_label: str,
    config: dict[str, Any],
    only_centerline: bool,
) -> None:
    """Write only the three requested intermediate-refiner artifacts."""

    _save_centreline_comparison(
        target,
        predicted,
        target_vis_exists,
        predicted_vis_exists,
        path / "centreline_comparison.png",
    )
    overlay_3d_function = (
        save_3d_centerline_overlay_gif
        if only_centerline
        else save_3d_overlay_gif
    )
    overlay_3d_kwargs: dict[str, Any] = {
        "pred_exist": predicted_vis_exists,
    }
    if only_centerline:
        overlay_3d_kwargs["point_valid"] = point_valid
    overlay_3d_function(
        supervision_target,
        predicted,
        target_vis_exists,
        path / "3d_overlay.gif",
        num_frames=int(
            config.get(
                "monitor_gif_frames",
                config.get("spy_gif_frames", 24),
            )
        ),
        fps=int(
            config.get(
                "monitor_gif_fps",
                config.get("spy_gif_fps", 5),
            )
        ),
        title=f"{view_label}: 3D overlay",
        **overlay_3d_kwargs,
    )

    images = batch.get("images")
    if images is None:
        raise RuntimeError(
            "Intermediate refiner-stage 2D overlays require input images"
        )
    images_np = images[item_index].detach().cpu().numpy()
    theta = batch.get("theta")
    phi = batch.get("phi")
    center_offset_world = None
    if (
        str(config.get("target_coordinate_frame", "projection_centered"))
        == "projection_centered"
    ):
        center_offset_world = torch.zeros(3, dtype=torch.float32)
    elif bool(batch["projection_center_offset_valid"][item_index]):
        center_offset_world = (
            batch["projection_center_offset"][item_index].detach().cpu()
        )
    coord_scale_to_meter = float(
        config.get("projection_coord_scale_to_meter", 0.001)
    )
    offset_m = (
        None
        if center_offset_world is None
        else center_offset_world.numpy() * coord_scale_to_meter
    )
    overlay_2d_kwargs = {
        "input_images": images_np,
        "gt_vessel": target,
        "pred_vessel": predicted,
        "target_exist": target_vis_exists,
        "pred_exist": predicted_vis_exists,
        "theta_deg": (
            None
            if theta is None
            else theta[item_index].detach().cpu().numpy()
        ),
        "phi_deg": (
            None
            if phi is None
            else phi[item_index].detach().cpu().numpy()
        ),
        "out_path": path / "2d_overlay.png",
        "sid": float(config.get("vis_proj_sid", 0.9)),
        "imager_pixel_spacing": float(
            config.get("vis_proj_imager_pixel_spacing", 0.55)
        ),
        "coord_scale_to_meter": coord_scale_to_meter,
        "center_offset_m": offset_m,
        "title": f"{view_label}: 2D overlay",
    }
    if only_centerline:
        save_2d_centerline_overlay(
            **overlay_2d_kwargs,
            point_valid=point_valid,
            mask_threshold=float(
                config.get("projection_mask_threshold", 0.5)
            ),
        )
    else:
        save_2d_overlay(**overlay_2d_kwargs)

def save_vessel_monitor(
    *,
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    item_index: int,
    path: Path,
    epoch: int,
    view_label: str,
    config: dict[str, Any],
    artifact_profile: str = "full",
    include_refined_surface_gifs: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if artifact_profile not in {
        "full",
        "static",
        "summary",
        "refiner_stage_minimal",
    }:
        raise ValueError(
            "artifact_profile must be 'full', 'static', 'summary', or "
            "'refiner_stage_minimal'"
        )
    if include_refined_surface_gifs and artifact_profile != "full":
        raise ValueError(
            "Refined single-surface GIFs require artifact_profile='full'"
        )
    render_gifs = artifact_profile not in {"static", "summary"}
    summary_only = artifact_profile == "summary"
    only_centerline = only_centerline_from_config(config)
    predicted = output["decoded_vessel_mm"][item_index].detach().cpu().numpy()
    coarse_predicted = (
        output["coarse_decoded_vessel_mm"][item_index].detach().cpu().numpy()
        if "coarse_decoded_vessel_mm" in output
        else None
    )
    refinement_stage_predictions = (
        output["refinement_stage_decoded_vessel_mm"][item_index]
        .detach()
        .cpu()
        .numpy()
        if "refinement_stage_decoded_vessel_mm" in output
        else None
    )
    raw_target = batch["target_raw_vessel_mm"][item_index].detach().cpu().numpy()
    parametric_target = batch["target_reconstructed_vessel_mm"][item_index].detach().cpu().numpy()
    reference_mode = str(config.get("monitor_reference", "loss_target")).strip().lower()
    if reference_mode == "loss_target":
        reference_mode = str(config.get("reconstruction_target", "raw")).strip().lower()
    if reference_mode not in {"raw", "parametric"}:
        raise ValueError("monitor_reference must be 'loss_target', 'raw', or 'parametric'")
    target = raw_target if reference_mode == "raw" else parametric_target
    radius_prediction_mode = radius_prediction_mode_from_config(config)
    centerline_prediction_mode = centerline_prediction_mode_from_config(config)
    parameter_label = (
        "landmark"
        if centerline_prediction_mode == "adaptive_landmarks"
        else "control point"
    )
    parameter_file_label = (
        "landmark"
        if centerline_prediction_mode == "adaptive_landmarks"
        else "control_point"
    )
    radius_target = (
        raw_target if radius_prediction_mode == "raw" else parametric_target
    )
    radius_reference_mode = (
        "raw" if radius_prediction_mode == "raw" else "parametric"
    )
    supervision_target = target.copy()
    supervision_target[..., 3] = radius_target[..., 3]
    valid = batch["target_point_valid_mask"][item_index].detach().cpu().numpy().astype(bool)
    exists = batch["target_branch_exist"][item_index].detach().cpu().numpy() > 0.5
    predicted_exist_probabilities = (
        output["branch_exist_probs"][item_index].detach().cpu().numpy()
    )
    predicted_exists = predicted_exist_probabilities >= 0.5
    fixed_main_count = min(
        len(exists), required_branch_count(str(config["artery_type"]))
    )
    target_vis_exists, predicted_vis_exists = resolve_paired_visualization_branch_masks(
        config=config,
        target_exist=exists,
        predicted_exist_probabilities=predicted_exist_probabilities,
        num_branches=predicted.shape[0],
    )
    case_id = str(batch["case_id"][item_index])
    if artifact_profile == "refiner_stage_minimal":
        _save_minimal_refiner_stage_visualization(
            predicted=predicted,
            target=target,
            supervision_target=supervision_target,
            target_vis_exists=target_vis_exists,
            predicted_vis_exists=predicted_vis_exists,
            point_valid=valid,
            batch=batch,
            item_index=item_index,
            path=path,
            view_label=f"Case {case_id}: {view_label}",
            config=config,
            only_centerline=only_centerline,
        )
        return
    if "branch_exist_logits" not in output:
        raise KeyError(
            "A full Tree Monitor requires output['branch_exist_logits'] to "
            "plot optional side-branch existence predictions."
        )
    _save_side_branch_existence_logits_plot(
        logits=output["branch_exist_logits"][item_index]
        .detach()
        .cpu()
        .numpy(),
        probabilities=predicted_exist_probabilities,
        target_exists=exists,
        fixed_main_branch_count=fixed_main_count,
        path=path / "side_branch_existence_logits.png",
        title=(
            "Side-branch existence head: "
            f"case {case_id}, epoch {epoch}, {view_label}"
        ),
    )
    if "radius_refinement_stage_decoded_vessel_mm" in output:
        coarse_radius_prediction = output[
            "radius_refiner_coarse_decoded_vessel_mm"
        ][item_index].detach().cpu().numpy()
        radius_stage_predictions = output[
            "radius_refinement_stage_decoded_vessel_mm"
        ][item_index].detach().cpu().numpy()
        _save_radius_refinement_profiles(
            target=radius_target,
            coarse=coarse_radius_prediction,
            stages=radius_stage_predictions,
            branch_exists=exists,
            point_valid=valid,
            path=path / "radius_refinement_by_stage.png",
        )
        if batch.get("images") is not None:
            images_np = batch["images"][item_index].detach().cpu().numpy()
            _save_input_views(images_np, path / "input_views.png")
            _save_radius_rendered_mask_comparison(
                images=images_np,
                view_mask=(
                    batch["view_mask"][item_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                ),
                coarse_masks=output[
                    "radius_refiner_coarse_rendered_masks"
                ][item_index].detach().cpu().numpy(),
                final_masks=output[
                    "radius_refiner_final_rendered_masks"
                ][item_index].detach().cpu().numpy(),
                path=path / "radius_rendered_mask_comparison.png",
            )
    target_controls = (
        batch["target_centerline_parameters_mm"][item_index]
        .detach()
        .cpu()
        .numpy()
    )
    predicted_controls = (
        output["centerline_parameters_mm"][item_index].detach().cpu().numpy()
    )
    target_controls_global = target_controls.copy()
    predicted_controls_global = predicted_controls.copy()
    if target_controls.shape[0] > 1 and "attachment_probabilities" in output:
        target_attachment = (
            batch["target_attachment_index"][item_index].detach().cpu().numpy()
        )
        predicted_attachment_probabilities = (
            output["attachment_probabilities"][item_index].detach().cpu().numpy()
        )
        for branch_index in range(1, target_controls.shape[0]):
            if target_vis_exists[branch_index]:
                attachment_index = int(target_attachment[branch_index])
                if 0 <= attachment_index < parametric_target.shape[1]:
                    target_controls_global[branch_index] += parametric_target[
                        0, attachment_index, :3
                    ]
            if not predicted_vis_exists[branch_index]:
                continue
            predicted_origin = np.sum(
                predicted_attachment_probabilities[branch_index, :, None]
                * predicted[0, :, :3],
                axis=0,
            )
            predicted_controls_global[branch_index] += predicted_origin

    figure = plt.figure(figsize=(14, 6))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_radius = figure.add_subplot(1, 2, 2)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(exists), 1)))
    visible_branches = np.flatnonzero(target_vis_exists | predicted_vis_exists)
    for branch_index in visible_branches:
        branch_valid = valid[branch_index]
        raw_branch = supervision_target[branch_index, branch_valid]
        radius_target_branch = radius_target[branch_index, branch_valid]
        prediction_valid = (
            branch_valid
            if target_vis_exists[branch_index] and bool(branch_valid.any())
            else np.ones(predicted.shape[1], dtype=bool)
        )
        predicted_branch = predicted[branch_index, prediction_valid]
        color = colors[branch_index]
        if target_vis_exists[branch_index] and raw_branch.size:
            axis_3d.plot(
                *raw_branch[:, :3].T,
                color=color,
                linewidth=2.0,
                label=f"{reference_mode} b{branch_index}",
            )
            axis_radius.plot(
                np.linspace(0.0, 1.0, len(raw_branch)),
                radius_target_branch[:, 3],
                color=color,
                linewidth=2.0,
            )
        if predicted_vis_exists[branch_index] and predicted_branch.size:
            axis_3d.plot(
                *predicted_branch[:, :3].T,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"pred b{branch_index}",
            )
            axis_radius.plot(
                np.linspace(0.0, 1.0, len(predicted_branch)),
                predicted_branch[:, 3],
                color=color,
                linewidth=1.5,
                linestyle="--",
            )
    axis_3d.set_title(f"Case {case_id}: centreline")
    axis_3d.set_xlabel("x (mm)")
    axis_3d.set_ylabel("y (mm)")
    axis_3d.set_zlabel("z (mm)")
    axis_3d.legend(fontsize=6, ncol=2)
    axis_radius.set_title(
        f"Radius profile: {radius_reference_mode} solid, prediction dashed"
    )
    axis_radius.set_xlabel("Normalized branch position")
    axis_radius.set_ylabel("Radius (mm)")
    axis_radius.grid(alpha=0.25)
    figure.suptitle(f"Epoch {epoch}, {view_label}")
    figure.tight_layout()
    figure.savefig(path / "vessel_overlay.png", dpi=160)
    plt.close(figure)

    _save_control_points_2d(
        target_controls,
        predicted_controls,
        target_vis_exists,
        predicted_vis_exists,
        path / f"{parameter_file_label}s_2d.png",
        parameter_label=parameter_label,
        absolute_parameters="attachment_logits" not in output,
    )
    if render_gifs:
        _save_centreline_control_points_3d_gif(
            parametric_target,
            predicted,
            target_controls_global,
            predicted_controls_global,
            target_vis_exists,
            predicted_vis_exists,
            valid,
            path / f"centreline_{parameter_file_label}s_3d.gif",
            num_frames=int(
                config.get(
                    "monitor_control_point_gif_frames",
                    config.get(
                        "monitor_gif_frames", config.get("spy_gif_frames", 24)
                    ),
                )
            ),
            fps=int(
                config.get(
                    "monitor_control_point_gif_fps",
                    config.get(
                        "monitor_gif_fps", config.get("spy_gif_fps", 5)
                    ),
                )
            ),
            parameter_label=parameter_label,
        )
    if summary_only:
        control_errors = np.linalg.norm(
            predicted_controls - target_controls,
            axis=-1,
        )
    else:
        control_errors = _save_control_point_error_by_index(
            target_controls,
            predicted_controls,
            exists,
            path / f"{parameter_file_label}_error_by_index.png",
            parameter_label=parameter_label,
            absolute_parameters="attachment_logits" not in output,
        )
        save_centerline_xyz_error_profile(
            target_vessel=raw_target,
            predicted_vessel=predicted,
            target_exist=exists,
            point_valid=valid,
            out_path=path / "dense_centerline_xyz_error_by_index.png",
            title="Decoded prediction vs raw dense vessel-code XYZ error",
        )
    if not only_centerline and not summary_only:
        save_radius_prediction_profiles(
            target_vessel=radius_target,
            predicted_vessel=predicted,
            target_exist=exists,
            point_valid=valid,
            profile_out_path=(
                path / "radius_ground_truth_vs_prediction_by_index.png"
            ),
            error_out_path=path / "radius_absolute_error_by_index.png",
            target_label=f"Ground truth ({radius_reference_mode})",
            profile_title=(
                f"{radius_reference_mode.title()} ground-truth and predicted "
                "radius by ordered point index"
            ),
        )
        if render_gifs:
            save_3d_radius_colored_comparison_gif(
                supervision_target,
                predicted,
                target_vis_exists,
                path / "3d_radius_colored_surfaces.gif",
                num_frames=int(
                    config.get(
                        "monitor_gif_frames",
                        config.get("spy_gif_frames", 24),
                    )
                ),
                fps=int(
                    config.get(
                        "monitor_radius_colored_gif_fps",
                        2,
                    )
                ),
                point_valid=valid,
                pred_exist=predicted_vis_exists,
                title=(
                    f"Case {case_id}: {radius_reference_mode} ground truth vs "
                    "prediction"
                ),
            )
        if include_refined_surface_gifs:
            gif_frames = int(
                config.get(
                    "monitor_gif_frames",
                    config.get("spy_gif_frames", 24),
                )
            )
            gif_fps = int(
                config.get(
                    "monitor_gif_fps",
                    config.get("spy_gif_fps", 5),
                )
            )
            prediction_point_valid = valid.copy()
            prediction_only_branches = (
                predicted_vis_exists & ~target_vis_exists
            )
            prediction_point_valid[prediction_only_branches] = True
            save_3d_ground_truth_gif(
                supervision_target,
                target_vis_exists,
                path / "3d_ground_truth_surface.gif",
                num_frames=gif_frames,
                fps=gif_fps,
                point_valid=valid,
                title=f"Case {case_id}: ground-truth surface",
            )
            save_3d_prediction_gif(
                predicted,
                predicted_vis_exists,
                path / "3d_prediction_surface.gif",
                num_frames=gif_frames,
                fps=gif_fps,
                point_valid=prediction_point_valid,
                title=f"Case {case_id}: refined prediction surface",
            )
            save_3d_radius_colored_prediction_gif(
                predicted,
                predicted_vis_exists,
                path / "3d_prediction_radius_colored_surface.gif",
                num_frames=gif_frames,
                fps=int(config.get("monitor_radius_colored_gif_fps", 2)),
                point_valid=prediction_point_valid,
                title=(
                    f"Case {case_id}: refined prediction radius-coloured "
                    "surface"
                ),
            )

    if not summary_only:
        _save_centreline_comparison(
            target,
            predicted,
            target_vis_exists,
            predicted_vis_exists,
            path / "centreline_comparison.png",
        )
    if render_gifs:
        _save_centreline_comparison_gif(
            target,
            predicted,
            target_vis_exists,
            predicted_vis_exists,
            valid,
            path / "centreline_comparison.gif",
            num_frames=int(
                config.get(
                    "monitor_gif_frames", config.get("spy_gif_frames", 24)
                )
            ),
            fps=int(
                config.get("monitor_gif_fps", config.get("spy_gif_fps", 5))
            ),
            title=f"Case {case_id}: epoch {epoch}, {view_label}",
        )
    if not summary_only:
        _save_whole_centreline_comparison(
            raw_target,
            parametric_target,
            predicted,
            target_vis_exists,
            predicted_vis_exists,
            path / "centreline_comparison_whole.png",
        )
    overlay_3d_function = (
        save_3d_centerline_overlay_gif
        if only_centerline
        else save_3d_overlay_gif
    )
    overlay_3d_kwargs: dict[str, Any] = {}
    if only_centerline:
        overlay_3d_kwargs["point_valid"] = valid
    overlay_3d_kwargs["pred_exist"] = predicted_vis_exists
    if render_gifs:
        overlay_3d_function(
            supervision_target,
            predicted,
            target_vis_exists,
            path / "3d_overlay.gif",
            num_frames=int(
                config.get(
                    "monitor_gif_frames", config.get("spy_gif_frames", 24)
                )
            ),
            fps=int(
                config.get("monitor_gif_fps", config.get("spy_gif_fps", 5))
            ),
            title=(
                f"Case {case_id}: 3D centreline overlay"
                if only_centerline
                else f"Case {case_id}"
            ),
            **overlay_3d_kwargs,
        )
        _save_three_way_surface_gif(
            raw_target,
            parametric_target,
            predicted,
            target_vis_exists,
            predicted_vis_exists,
            path / "3d_overlay_whole.gif",
            num_frames=int(
                config.get(
                    "monitor_gif_frames", config.get("spy_gif_frames", 24)
                )
            ),
            fps=int(
                config.get("monitor_gif_fps", config.get("spy_gif_fps", 5))
            ),
            title=f"Case {case_id}: raw, GT parametric, prediction",
        )
    images = batch.get("images")
    if images is not None and not summary_only:
        images_np = images[item_index].detach().cpu().numpy()
        _save_input_views(images_np, path / "input_views.png")
        theta = batch.get("theta")
        phi = batch.get("phi")
        center_offset_world = None
        if (
            str(config.get("target_coordinate_frame", "projection_centered"))
            == "projection_centered"
        ):
            center_offset_world = torch.zeros(3, dtype=torch.float32)
        elif bool(batch["projection_center_offset_valid"][item_index]):
            center_offset_world = (
                batch["projection_center_offset"][item_index].detach().cpu()
            )
        coord_scale_to_meter = float(
            config.get("projection_coord_scale_to_meter", 0.001)
        )
        offset_m = (
            None
            if center_offset_world is None
            else center_offset_world.numpy() * coord_scale_to_meter
        )
        overlay_2d_kwargs = {
            "input_images": images_np,
            "gt_vessel": target,
            "pred_vessel": predicted,
            "target_exist": target_vis_exists,
            "pred_exist": predicted_vis_exists,
            "theta_deg": (
                None if theta is None else theta[item_index].detach().cpu().numpy()
            ),
            "phi_deg": (
                None if phi is None else phi[item_index].detach().cpu().numpy()
            ),
            "out_path": path / "2d_overlay.png",
            "sid": float(config.get("vis_proj_sid", 0.9)),
            "imager_pixel_spacing": float(
                config.get("vis_proj_imager_pixel_spacing", 0.55)
            ),
            "coord_scale_to_meter": coord_scale_to_meter,
            "center_offset_m": offset_m,
            "title": (
                f"Case {case_id}: 2D centreline overlay"
                if only_centerline
                else f"Case {case_id}"
            ),
        }
        if only_centerline:
            save_2d_centerline_overlay(
                **overlay_2d_kwargs,
                point_valid=valid,
                mask_threshold=float(config.get("projection_mask_threshold", 0.5)),
            )
        else:
            save_2d_overlay(**overlay_2d_kwargs)
        if theta is not None and phi is not None:
            num_input_views = min(
                int(batch["view_mask"][item_index].sum().item()),
                int(images_np.shape[0]),
                int(theta.shape[1]),
                int(phi.shape[1]),
            )
            projection_target_mode = str(
                config.get("projection_centerline_target", "raw")
            ).strip().lower()
            if projection_target_mode not in {"raw", "parametric"}:
                raise ValueError(
                    "projection_centerline_target must be 'raw' or 'parametric'"
                )
            projection_target = (
                batch["target_raw_vessel_mm"]
                if projection_target_mode == "raw"
                else batch["target_reconstructed_vessel_mm"]
            )[item_index].detach().cpu()
            predicted_tensor = output["decoded_vessel_mm"][item_index].detach().cpu()
            monitor_projector = build_centerline_projector(
                config, int(images_np.shape[-1])
            )
            if num_input_views > 0:
                save_projected_control_point_monitor(
                    input_images=images_np[:num_input_views],
                    gt_vessel=projection_target,
                    pred_vessel=predicted_tensor,
                    gt_control_points_global=target_controls_global,
                    pred_control_points_global=predicted_controls_global,
                    target_exist=target_vis_exists,
                    pred_exist=predicted_vis_exists,
                    point_valid=valid,
                    theta_deg=theta[item_index, :num_input_views].detach().cpu(),
                    phi_deg=phi[item_index, :num_input_views].detach().cpu(),
                    projector=monitor_projector,
                    coord_scale_to_meter=coord_scale_to_meter,
                    clean_out_path=path
                    / f"{parameter_file_label}s_2d_input_views.png",
                    mask_out_path=path
                    / f"{parameter_file_label}s_2d_input_views_mask_overlay.png",
                    center_offset_world=center_offset_world,
                    title=(
                        f"Case {case_id}: {projection_target_mode} GT centreline"
                    ),
                    point_label=parameter_label,
                )
            if bool(config.get("enable_projection_2d_loss", False)):
                num_loss_views = min(
                    num_input_views,
                    int(config.get("proj_loss_num_views", images_np.shape[0])),
                )
            else:
                num_loss_views = 0
            if num_loss_views > 0:
                save_projection_loss_monitor(
                    input_images=images_np[:num_loss_views],
                    gt_vessel=projection_target,
                    pred_vessel=predicted_tensor,
                    theta_deg=theta[item_index, :num_loss_views].detach().cpu(),
                    phi_deg=phi[item_index, :num_loss_views].detach().cpu(),
                    projector=monitor_projector,
                    coord_scale_to_meter=coord_scale_to_meter,
                    out_path=path / "projection_loss_monitor.png",
                    center_offset_world=center_offset_world,
                    target_exist=(
                        batch["target_branch_exist"][item_index].detach().cpu() > 0.5
                    ),
                    mask_threshold=float(config.get("projection_mask_threshold", 0.5)),
                )

    valid_points = exists[:, None] & valid
    xyz_error_by_point = np.linalg.norm(
        predicted[..., :3] - target[..., :3], axis=-1
    )
    xyz_error = xyz_error_by_point[valid_points]
    start_errors: list[float] = []
    end_errors: list[float] = []
    for branch_index in np.flatnonzero(exists):
        valid_indices = np.flatnonzero(valid[branch_index])
        if valid_indices.size == 0:
            continue
        start_errors.append(
            float(xyz_error_by_point[branch_index, valid_indices[0]])
        )
        end_errors.append(
            float(xyz_error_by_point[branch_index, valid_indices[-1]])
        )
    arc_length_metrics = compute_branch_arc_length_metrics(
        predicted,
        target,
        exists,
        valid,
        fixed_main_branch_count=fixed_main_count,
    )
    existence_metrics = compute_branch_existence_metrics(
        exists,
        predicted_exists,
        fixed_main_branch_count=fixed_main_count,
    )
    radius_error = np.abs(predicted[..., 3] - radius_target[..., 3])[valid_points]
    metrics = {
        "case_id": case_id,
        "epoch": int(epoch),
        "view_label": view_label,
        "reference": reference_mode,
        "centerline_prediction_mode": centerline_prediction_mode,
        "radius_prediction_mode": radius_prediction_mode,
        "radius_reference": radius_reference_mode,
        "num_views": int(batch["view_mask"][item_index].sum().item()),
        "selected_view_indices": batch["selected_view_indices"][item_index, : int(batch["view_mask"][item_index].sum().item())].tolist(),
        "centerline_mean_mm": float(xyz_error.mean()) if xyz_error.size else None,
        "centerline_p95_mm": float(np.percentile(xyz_error, 95)) if xyz_error.size else None,
        "centerline_max_mm": float(xyz_error.max()) if xyz_error.size else None,
        "centerline_start_mae_mm": (
            float(np.mean(start_errors)) if start_errors else None
        ),
        "centerline_end_mae_mm": (
            float(np.mean(end_errors)) if end_errors else None
        ),
        **arc_length_metrics,
        "radius_mae_mm": float(radius_error.mean()) if radius_error.size else None,
        "radius_p95_mm": float(np.percentile(radius_error, 95)) if radius_error.size else None,
        "predicted_radius_min_mm": float(predicted[..., 3][valid_points].min()) if valid_points.any() else None,
        "predicted_radius_p95_mm": float(np.percentile(predicted[..., 3][valid_points], 95)) if valid_points.any() else None,
        "predicted_radius_max_mm": float(predicted[..., 3][valid_points].max()) if valid_points.any() else None,
        **existence_metrics,
        "visualization_branch_existence_source": str(
            config["visualization_branch_existence_source"]
        ),
        "visualization_artery_type": str(config["artery_type"]),
        "visualization_snap_predicted_side_branches_to_main": bool(
            config.get(
                "visualization_snap_predicted_side_branches_to_main_effective",
                False,
            )
        ),
        "visualization_side_branch_translation_mm": (
            output["visualization_side_branch_translation_mm"][item_index]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
            if "visualization_side_branch_translation_mm" in output
            else None
        ),
        "visualization_target_branch_exists": target_vis_exists.astype(bool).tolist(),
        "visualization_predicted_branch_exists": predicted_vis_exists.astype(bool).tolist(),
        f"{parameter_file_label}_error_mean_mm": (
            float(control_errors[exists].mean()) if bool(exists.any()) else None
        ),
        f"{parameter_file_label}_error_p95_mm": (
            float(np.percentile(control_errors[exists], 95))
            if bool(exists.any())
            else None
        ),
        f"{parameter_file_label}_error_by_branch_mm": {
            str(branch_index): control_errors[branch_index].astype(float).tolist()
            for branch_index in np.flatnonzero(exists)
        },
    }
    if coarse_predicted is not None:
        coarse_xyz_error = np.linalg.norm(
            coarse_predicted[..., :3] - target[..., :3], axis=-1
        )[valid_points]
        metrics["coarse_centerline_mean_mm"] = (
            float(coarse_xyz_error.mean()) if coarse_xyz_error.size else None
        )
        metrics["coarse_centerline_p95_mm"] = (
            float(np.percentile(coarse_xyz_error, 95))
            if coarse_xyz_error.size
            else None
        )
    if refinement_stage_predictions is not None:
        for stage_index, stage_prediction in enumerate(
            refinement_stage_predictions, start=1
        ):
            stage_xyz_error = np.linalg.norm(
                stage_prediction[..., :3] - target[..., :3], axis=-1
            )[valid_points]
            metrics[f"refinement_stage_{stage_index}_centerline_mean_mm"] = (
                float(stage_xyz_error.mean()) if stage_xyz_error.size else None
            )
            metrics[f"refinement_stage_{stage_index}_centerline_p95_mm"] = (
                float(np.percentile(stage_xyz_error, 95))
                if stage_xyz_error.size
                else None
            )
    monitor_refiner_stage_metrics = config.get(
        "monitor_refiner_stage_metrics",
        True,
    )
    if not isinstance(monitor_refiner_stage_metrics, bool):
        raise ValueError("monitor_refiner_stage_metrics must be boolean")
    if (
        monitor_refiner_stage_metrics
        and coarse_predicted is not None
        and refinement_stage_predictions is not None
        and "coarse_centerline_parameters_mm" in output
        and "refinement_stage_centerline_parameters_mm" in output
    ):
        stage_rows = _refiner_stage_geometry_rows(
            coarse_vessel=coarse_predicted,
            stage_vessels=refinement_stage_predictions,
            coarse_parameters=output["coarse_centerline_parameters_mm"][
                item_index
            ]
            .detach()
            .cpu()
            .numpy(),
            stage_parameters=output[
                "refinement_stage_centerline_parameters_mm"
            ][item_index]
            .detach()
            .cpu()
            .numpy(),
            target_vessel=target,
            target_parameters=target_controls,
            branch_exists=exists,
            point_valid=valid,
            fixed_main_branch_count=fixed_main_count,
        )
        metrics["refiner_stage_3d_metrics"] = stage_rows
        save_json(path / "refiner_stage_3d_metrics.json", stage_rows)
        _save_refiner_stage_geometry_plot(
            stage_rows,
            path / "refiner_stage_3d_metrics.png",
            title=(
                f"Case {case_id}, epoch {epoch}: 3D performance across "
                "learned refinement stages"
            ),
        )
    save_json(path / "metrics.json", metrics)
    _save_metric_matrix(metrics, path / "metrics_matrix.png")

    prediction_payload: dict[str, Any] = dict(
        case_id=np.asarray(case_id),
        raw_vessel_code_mm=raw_target,
        target_parametric_reconstructed_vessel_code_mm=parametric_target,
        monitor_reference_vessel_code_mm=target,
        monitor_reference=np.asarray(reference_mode),
        radius_target_vessel_code_mm=radius_target,
        supervision_reference_vessel_code_mm=supervision_target,
        centerline_prediction_mode=np.asarray(centerline_prediction_mode),
        radius_prediction_mode=np.asarray(radius_prediction_mode),
        reconstructed_vessel_code_mm=predicted,
        branch_exists=exists,
        point_valid_mask=valid,
        predicted_branch_exists=predicted_exists,
        branch_exist_logits=output["branch_exist_logits"][item_index]
        .detach()
        .cpu()
        .numpy(),
        branch_exist_probabilities=predicted_exist_probabilities,
        visualization_branch_existence_source=np.asarray(
            config["visualization_branch_existence_source"]
        ),
        visualization_artery_type=np.asarray(config["artery_type"]),
        visualization_snap_predicted_side_branches_to_main=np.asarray(
            bool(
                config.get(
                    "visualization_snap_predicted_side_branches_to_main_effective",
                    False,
                )
            )
        ),
        visualization_side_branch_translation_mm=(
            output["visualization_side_branch_translation_mm"][item_index]
            .detach()
            .cpu()
            .numpy()
            if "visualization_side_branch_translation_mm" in output
            else np.zeros((predicted.shape[0], 3), dtype=np.float32)
        ),
        visualization_target_branch_exists=target_vis_exists,
        visualization_predicted_branch_exists=predicted_vis_exists,
        target_centerline_parameters_mm=target_controls,
        centerline_parameters_mm=predicted_controls,
        centerline_parameter_error_mm=control_errors,
        side_branch_attachment_index=(
            output["attachment_logits"][item_index]
            .argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            if "attachment_logits" in output
            else np.full((predicted.shape[0],), -1, dtype=np.int64)
        ),
        centerline_parameter_frame=np.asarray(
            "main_global_side_attachment_relative_mm"
            if "attachment_logits" in output
            else "global_mm"
        ),
        decoder_architecture=np.asarray(
            "absolute_parallel"
            if "attachment_logits" not in output
            else str(
                dict(config.get("model", {}) or {}).get(
                    "decoder_architecture",
                    config.get(
                        "decoder_architecture", "main_first_hierarchical"
                    ),
                )
            )
        ),
    )
    if coarse_predicted is not None:
        prediction_payload["coarse_reconstructed_vessel_code_mm"] = (
            coarse_predicted
        )
        prediction_payload["coarse_centerline_control_points_mm"] = output[
            "coarse_centerline_parameters_mm"
        ][item_index].detach().cpu().numpy()
        if "coarse_branch_exist_logits" in output:
            prediction_payload["coarse_branch_exist_logits"] = output[
                "coarse_branch_exist_logits"
            ][item_index].detach().cpu().numpy()
            prediction_payload["coarse_branch_exist_probabilities"] = output[
                "coarse_branch_exist_probs"
            ][item_index].detach().cpu().numpy()
    if refinement_stage_predictions is not None:
        prediction_payload["refinement_stage_reconstructed_vessel_code_mm"] = (
            refinement_stage_predictions
        )
        prediction_payload["refinement_stage_centerline_control_points_mm"] = (
            output["refinement_stage_centerline_parameters_mm"][item_index]
            .detach()
            .cpu()
            .numpy()
        )
        prediction_payload["bspline_refinement_residual_mm"] = output[
            "bspline_refinement_residual_mm"
        ][item_index].detach().cpu().numpy()
        if "refinement_stage_branch_exist_logits" in output:
            prediction_payload[
                "refinement_stage_branch_exist_logits"
            ] = output["refinement_stage_branch_exist_logits"][item_index]
            prediction_payload[
                "refinement_stage_branch_exist_probabilities"
            ] = output["refinement_stage_branch_exist_probs"][item_index]
            prediction_payload[
                "bspline_refinement_branch_exist_residual_logits"
            ] = output[
                "bspline_refinement_branch_exist_residual_logits"
            ][item_index]
            for key in (
                "refinement_stage_branch_exist_logits",
                "refinement_stage_branch_exist_probabilities",
                "bspline_refinement_branch_exist_residual_logits",
            ):
                prediction_payload[key] = (
                    prediction_payload[key].detach().cpu().numpy()
                )
    if centerline_prediction_mode == "adaptive_landmarks":
        prediction_payload.update(
            target_centerline_landmarks_mm=target_controls,
            centerline_landmarks_mm=predicted_controls,
            centerline_landmark_error_mm=control_errors,
        )
    else:
        prediction_payload.update(
            target_centerline_control_points_mm=target_controls,
            centerline_control_points_mm=predicted_controls,
            centerline_control_point_error_mm=control_errors,
        )
    if radius_prediction_mode == "raw":
        prediction_payload.update(
            raw_radius_log_mm=output["raw_radius_log_mm"][item_index]
            .detach()
            .cpu()
            .numpy(),
            raw_radius_mm=output["raw_radius_mm"][item_index]
            .detach()
            .cpu()
            .numpy(),
        )
    else:
        prediction_payload.update(
            radius_baseline_coefficients_log_mm=output[
                "radius_baseline_coefficients_log_mm"
            ][item_index]
            .detach()
            .cpu()
            .numpy(),
            radius_lesion_exist_probability=output["lesion_exist_logits"][
                item_index
            ]
            .sigmoid()
            .detach()
            .cpu()
            .numpy(),
            radius_lesion_geometry=output["lesion_geometry"][item_index]
            .detach()
            .cpu()
            .numpy(),
        )
    if resolve_save_prediction_npz_files(config):
        np.savez_compressed(path / "prediction.npz", **prediction_payload)
