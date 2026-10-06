# Transferred from methods/parametric_methods/train_radius_refiner.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np
import torch
from torch.utils.data import DataLoader
from vessel_code.parametric.data import resolve_feature_dataset_dir, resolve_raw_image_dataset_dir, uses_online_vggt_features
from vessel_code.parametric.loss import centerline_prediction_mode_from_config, radius_prediction_mode_from_config
from vessel_code.parametric.model import build_model_for_checkpoint, infer_checkpoint_decoder_architecture
from vessel_code.parametric.monitoring import plot_history, save_json
from vessel_code.parametric.online_vggt import OnlineVGGTFeatureProvider
from vessel_code.parametric.radius_refiner_data import INVISIBLE_STENOSIS_GROUP, ORIGINAL_VARIANT, VISIBLE_AUGMENTED_GROUP, Stage35RadiusRefinerGroupDataset, build_radius_refiner_group_datasets, collate_radius_refiner_groups, summarize_radius_refiner_groups
from vessel_code.parametric.radius_refiner_loss import compute_counterfactual_radius_refiner_loss, validate_counterfactual_radius_refiner_loss_config
from vessel_code.parametric.radius_refiner_monitoring import save_counterfactual_radius_refiner_monitor
from vessel_code.parametric.train import _deep_merge_config, configure_bspline_refiner_only_training, device_from_config, load_config, refiner_branch_mask_for_batch, resolve_batch_image_features, set_parametric_model_training_mode
from vessel_code.shared.branch_visibility import refiner_branch_existence_source
from vessel_code.shared.data import normalize_feature_backbone
from vessel_code.shared.paths import resolve_train_output_config

"""Train only the post-geometry radius refiner on Stage-3.5 groups.

This entry point intentionally does not call the ordinary parametric training
loss or ordinary file-level data loader.  A counterfactual case is one
optimiser unit (one original sample or one three-member triplet), its variants
share the same selected views, and checkpoint selection uses only the explicit
radius-refinement objective implemented in ``radius_refiner_loss.py``.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_LOSS_SEMANTIC_KEYS = (
    "radius_scale_mm",
    "radius_refiner_final_radius_loss_weight",
    "radius_refiner_stenosis_interval_point_weight",
    "radius_refiner_intermediate_radius_loss_weight",
    "radius_refiner_paired_difference_loss_weight",
    "radius_refiner_outside_consistency_loss_weight",
    "radius_refiner_conservative_residual_loss_weight",
    "radius_refiner_whole_artery_dice_enabled",
    "radius_refiner_whole_artery_dice_loss_weight",
    "radius_refiner_local_stenosis_dice_enabled",
    "radius_refiner_local_stenosis_dice_loss_weight",
    "radius_refiner_local_stenosis_dice_roi_dilation_px",
    "radius_refiner_local_stenosis_dice_visible_views_only",
)

_FROZEN_CHECKPOINT_MODEL_KEYS = (
    "num_branches",
    "num_points",
    "num_control_points",
    "num_landmarks",
    "num_radius_coefficients",
    "num_lesions",
    "lesion_profile",
    "centerline_prediction_mode",
    "radius_prediction_mode",
    "model_dim",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_attention_heads",
    "mlp_hidden_dim",
    "dropout",
    "use_encoded_view_dir",
    "use_post_backbone_transformer_encoder",
    "use_ray_camera_encoding",
    "ray_camera_encoding_dim",
    "use_prope_attention",
    "prope_frequency_base",
    "camera_encoding_image_size",
    "camera_encoding_sid",
    "camera_encoding_source_to_iso",
    "camera_encoding_imager_pixel_spacing",
    "resnet_pre_fpn_channels",
    "image_fpn_pool_size",
    "use_learned_side_parent_projection",
    "use_bspline_control_refiner",
    "bspline_refiner_num_stages",
    "bspline_refiner_evidence_hidden_dim",
    "bspline_refiner_patch_size",
    "bspline_refiner_use_learned_image_features",
    "bspline_refiner_learned_feature_dim",
    "bspline_refiner_use_distance_transform",
    "bspline_refiner_distance_transform_num_iters",
    "bspline_refiner_image_size",
    "bspline_refiner_sid",
    "bspline_refiner_source_to_iso",
    "bspline_refiner_imager_pixel_spacing",
    "bspline_refiner_residual_scale_mm",
    "bspline_refiner_control_position_scale_mm",
)

def _format_progress_duration(seconds: float) -> str:
    """Format an elapsed time or ETA without hiding long-running hours."""

    if not math.isfinite(seconds) or seconds < 0.0:
        return "--:--"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"

def _progress_report_due(
    *,
    completed: int,
    total: int,
    every_groups: int,
    seconds_since_report: float,
    minimum_interval_seconds: float,
) -> bool:
    return (
        completed == 1
        or completed >= total
        or completed % every_groups == 0
        or seconds_since_report >= minimum_interval_seconds
    )

def _model_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    model_config = config.get("model", {})
    if isinstance(model_config, Mapping) and key in model_config:
        return model_config[key]
    return config.get(key, default)

def _loss_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    loss_config = config.get("loss", {})
    if isinstance(loss_config, Mapping) and key in loss_config:
        return loss_config[key]
    return config.get(key, default)

def _checkpoint_path_for_run(
    raw_config: dict[str, Any],
    *,
    cli_resume: str | None,
    cli_output_dir: str | None,
) -> tuple[Path, bool]:
    resume_value: Any = (
        cli_resume
        if cli_resume is not None
        else raw_config.get("resume_checkpoint")
    )
    if resume_value not in {None, ""}:
        if str(resume_value).strip().lower() == "latest":
            preview = dict(raw_config)
            if cli_output_dir:
                preview["output_dir"] = cli_output_dir
                for key in ("output_root_dir", "experiment_dir", "out_dir"):
                    preview.pop(key, None)
            backbone = normalize_feature_backbone(
                preview.get("feature_backbone", "vggt")
            )
            return resolve_train_output_config(preview, backbone) / "latest.pt", True
        return Path(str(resume_value)).expanduser().resolve(), True
    initial = raw_config.get("initial_checkpoint")
    if initial in {None, ""}:
        raise ValueError(
            "Dedicated radius-refiner training requires initial_checkpoint for "
            "a new run, or resume_checkpoint/--resume for an existing run."
        )
    return Path(str(initial)).expanduser().resolve(), False

def _load_merged_training_config(
    config_path: str | Path,
    *,
    cli_resume: str | None,
    cli_output_dir: str | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], Path, bool]:
    raw_config = load_config(config_path)
    checkpoint_path, resuming = _checkpoint_path_for_run(
        raw_config,
        cli_resume=cli_resume,
        cli_output_dir=cli_output_dir,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict.")
    source_config = checkpoint.get("config", {})
    if not isinstance(source_config, dict):
        raise ValueError(f"{checkpoint_path} config must be a JSON object.")
    # The compact Stage-3 config changes data, loss, and the new radius module;
    # every unchanged predictor/refiner architecture setting comes from the
    # checkpoint that produced the frozen geometry model.
    config = _deep_merge_config(source_config, raw_config)
    # Prefer the weight mapping for new runs, while allowing an explicitly
    # supplied legacy two-array schedule to replace a mapping inherited from
    # the geometry-refiner checkpoint.
    legacy_view_schedule_keys = (
        "train_view_count_choices",
        "train_view_count_probabilities",
    )
    if raw_config.get("train_view_count_weights") is not None:
        for key in legacy_view_schedule_keys:
            config.pop(key, None)
    elif any(key in raw_config for key in legacy_view_schedule_keys):
        config.pop("train_view_count_weights", None)
    if not resuming:
        # A normal training checkpoint stores resolved output aliases. They
        # identify the *source geometry experiment* and must never survive a
        # fresh Stage-3 run, otherwise resolve_train_output_config() would
        # prioritize the inherited experiment_dir and overwrite that source.
        output_keys = (
            "output_dir",
            "output_root_dir",
            "experiment_dir",
            "out_dir",
            "experiment_name",
        )
        for key in output_keys:
            config.pop(key, None)
        for key in output_keys:
            if key in raw_config:
                config[key] = raw_config[key]
    if cli_output_dir:
        config["output_dir"] = cli_output_dir
        for key in ("output_root_dir", "experiment_dir", "out_dir"):
            config.pop(key, None)
    config["radius_refiner_source_checkpoint"] = str(checkpoint_path)
    config["radius_refiner_source_checkpoint_epoch"] = int(
        checkpoint.get("epoch", -1)
    )
    config["radius_refiner_checkpoint_config_merged"] = True
    config["radius_refiner_resume"] = bool(resuming)
    if resuming:
        config["resume_checkpoint"] = str(checkpoint_path)
    else:
        config["initial_checkpoint"] = str(checkpoint_path)
        config["resume_checkpoint"] = None
    return config, checkpoint, checkpoint_path, resuming

def _validate_architecture_against_checkpoint(
    config: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    source = checkpoint.get("config", {})
    if not isinstance(source, dict):
        source = {}
    for key in _FROZEN_CHECKPOINT_MODEL_KEYS:
        source_value = _model_value(source, key)
        current_value = _model_value(config, key)
        if source_value is not None and current_value != source_value:
            raise ValueError(
                "Stage-3 radius-refiner config changes frozen checkpoint "
                "architecture "
                f"{key}: checkpoint={source_value!r}, config={current_value!r}."
            )
    source_backbone = normalize_feature_backbone(
        source.get("feature_backbone", config.get("feature_backbone", "vggt"))
    )
    current_backbone = normalize_feature_backbone(
        config.get("feature_backbone", "vggt")
    )
    if source_backbone != current_backbone:
        raise ValueError(
            "Stage-3 cached feature backbone must match the geometry checkpoint: "
            f"checkpoint={source_backbone}, config={current_backbone}."
        )
    state_dict = checkpoint["model_state_dict"]
    inferred_architecture = infer_checkpoint_decoder_architecture(state_dict)
    configured_architecture = _model_value(config, "decoder_architecture")
    if configured_architecture is not None and str(configured_architecture) != str(
        inferred_architecture
    ):
        raise ValueError(
            "Configured decoder_architecture disagrees with checkpoint state: "
            f"config={configured_architecture!r}, inferred={inferred_architecture!r}."
        )
    config.setdefault("model", {})["decoder_architecture"] = inferred_architecture

def validate_counterfactual_radius_refiner_training_config(
    config: dict[str, Any]
) -> None:
    refiner_branch_existence_source(config)
    radius_dataset = config.get("radius_refiner_feature_dataset_dir")
    generic_dataset = config.get("feature_dataset_dir")
    if radius_dataset is not None and generic_dataset is not None:
        radius_path = Path(str(radius_dataset)).expanduser().resolve()
        generic_path = Path(str(generic_dataset)).expanduser().resolve()
        if radius_path != generic_path:
            raise ValueError(
                "radius_refiner_feature_dataset_dir and feature_dataset_dir "
                "must identify the same Stage-3.5 anatomy root; the first feeds "
                "the grouped loader and the second is saved for standard eval."
            )
    if not bool(config.get("train_radius_refiner_only", False)):
        raise ValueError(
            "The dedicated entry point requires train_radius_refiner_only=true."
        )
    for incompatible_scope in (
        "train_bspline_refiner_only",
        "train_radius_head_only",
        "train_bspline_refiner_and_radius_head_only",
    ):
        if bool(config.get(incompatible_scope, False)):
            raise ValueError(
                f"{incompatible_scope}=true is incompatible with the dedicated "
                "radius-refiner trainer."
            )
    if bool(config.get("enable_projection_2d_loss", False)):
        raise ValueError(
            "The dedicated radius-refiner trainer requires "
            "enable_projection_2d_loss=false. Its dedicated 2D terms are the "
            "whole-artery Dice and additional stenosis-local Dice objectives."
        )
    if bool(config.get("only_centerline", False)):
        raise ValueError("Radius-refiner training is incompatible with only_centerline.")
    if centerline_prediction_mode_from_config(config) != "bspline_control_points":
        raise ValueError(
            "Radius-refiner training requires B-spline control-point centreline "
            "prediction."
        )
    if radius_prediction_mode_from_config(config) != "raw":
        raise ValueError("Radius-refiner training requires radius_prediction_mode='raw'.")
    if not bool(_model_value(config, "use_bspline_control_refiner", False)):
        raise ValueError(
            "Radius-refiner training requires the trained B-spline geometry refiner."
        )
    if not bool(_model_value(config, "use_radius_evidence_refiner", False)):
        raise ValueError("model.use_radius_evidence_refiner must be true.")
    if int(config.get("groups_per_batch", 1)) != 1:
        raise ValueError("groups_per_batch must be 1.")
    if str(config.get("group_sampling_strategy", "exhaustive_shuffled")) != (
        "exhaustive_shuffled"
    ):
        raise ValueError("group_sampling_strategy must be 'exhaustive_shuffled'.")
    if float(_loss_value(config, "radius_refiner_dice_loss_weight", 0.0)) != 0.0:
        raise ValueError(
            "Set legacy global radius_refiner_dice_loss_weight=0. The dedicated "
            "trainer uses radius_refiner_whole_artery_dice_loss_weight plus "
            "the additional local stenosis Dice weight instead."
        )
    if float(
        _loss_value(config, "radius_refiner_final_radius_loss_weight", 1.0)
    ) <= 0.0:
        raise ValueError(
            "The dedicated trainer requires a positive "
            "radius_refiner_final_radius_loss_weight so every semantic group "
            "has direct radius supervision."
        )
    vessel_type = str(
        config.get("radius_refiner_vessel_type", "")
    ).strip().lower()
    if vessel_type not in {"rca", "lca"}:
        raise ValueError("radius_refiner_vessel_type must be 'rca' or 'lca'.")

    configured_artery_type = config.get("artery_type")
    if configured_artery_type is not None:
        artery_type = str(configured_artery_type).strip().lower()
        if artery_type != vessel_type:
            raise ValueError(
                "artery_type must match radius_refiner_vessel_type, got "
                f"{configured_artery_type!r} and {vessel_type!r}."
            )

    raw_num_branches = _model_value(config, "num_branches")
    if raw_num_branches is None or isinstance(raw_num_branches, bool):
        raise ValueError("num_branches must be a non-null positive integer.")
    try:
        actual_branches = int(raw_num_branches)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "num_branches must be a non-null positive integer, got "
            f"{raw_num_branches!r}."
        ) from error
    if actual_branches < 1 or (
        isinstance(raw_num_branches, float)
        and not raw_num_branches.is_integer()
    ):
        raise ValueError(
            "num_branches must be a non-null positive integer, got "
            f"{raw_num_branches!r}."
        )

    # Both anatomies may retain any positive number of configured branches.
    # The frozen-checkpoint and grouped-data validators separately enforce
    # that this value matches the actual architecture and Stage-3.5 NPZ schema.
    branch_probability_threshold = float(
        _model_value(
            config,
            "radius_refiner_branch_probability_threshold",
            0.5,
        )
    )
    if not math.isfinite(branch_probability_threshold) or not (
        0.0 <= branch_probability_threshold <= 1.0
    ):
        raise ValueError(
            "model.radius_refiner_branch_probability_threshold must be finite "
            f"and in [0, 1], got {branch_probability_threshold}."
        )
    expected_spacing = 0.55 if vessel_type == "rca" else 0.65
    actual_spacing = float(
        _model_value(config, "radius_refiner_imager_pixel_spacing", float("nan"))
    )
    if not math.isclose(actual_spacing, expected_spacing, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"{vessel_type.upper()} radius refinement requires pixel spacing "
            f"{expected_spacing} mm, got {actual_spacing}."
        )
    for mode_key, default_mode in (
        ("radius_refiner_canonical_geometry_mode", "auto"),
        ("radius_refiner_canonical_geometry_eval_mode", "audit"),
    ):
        mode = str(config.get(mode_key, default_mode)).lower()
        if mode not in {"disabled", "audit", "auto", "always"}:
            raise ValueError(
                f"{mode_key} must be disabled, audit, auto, or always."
            )
    threshold = float(
        config.get("radius_refiner_canonical_geometry_p95_threshold_mm", 0.05)
    )
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "radius_refiner_canonical_geometry_p95_threshold_mm must be finite "
            f"and >= 0, got {threshold}."
        )
    backbone = normalize_feature_backbone(config.get("feature_backbone", "vggt"))
    if backbone in {"vggt", "vggt_omega"}:
        context_mode = str(config.get("expected_vggt_context_mode", "")).lower()
        if uses_online_vggt_features(dict(config)):
            if bool(config.get("lazy_load_image_features", False)):
                raise ValueError(
                    "lazy_load_image_features=true is only for cached features "
                    "and cannot be combined with on-the-fly VGGT."
                )
            if context_mode not in {"all_views", "per_view"}:
                raise ValueError(
                    "On-the-fly VGGT radius-refiner training requires "
                    "expected_vggt_context_mode='all_views' or 'per_view'."
                )
        elif context_mode != "per_view":
            raise ValueError(
                "Variable 1-7-view cached VGGT training requires "
                "expected_vggt_context_mode='per_view'."
            )
    validate_counterfactual_radius_refiner_loss_config(config)

def _loader(
    dataset: Stage35RadiusRefinerGroupDataset,
    config: Mapping[str, Any],
    *,
    train: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 0)))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(train),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_radius_refiner_groups,
        generator=generator,
        # The dataset's epoch controls the deterministic view draw. Recreate
        # workers so each epoch receives the value set by set_epoch().
        persistent_workers=False,
    )

def _fixed_view_dataset(
    dataset: Stage35RadiusRefinerGroupDataset,
    config: Mapping[str, Any],
    *,
    view_count: int,
) -> Stage35RadiusRefinerGroupDataset:
    fixed = dict(config)
    prefix = "validation" if dataset.split == "val" else "test"
    fixed[f"{prefix}_view_count_choices"] = [int(view_count)]
    fixed[f"{prefix}_view_count_probabilities"] = [1.0]
    fixed[f"radius_refiner_{dataset.split}_resample_views_each_epoch"] = False
    return Stage35RadiusRefinerGroupDataset(
        dataset.groups, fixed, split=dataset.split
    )

def _validation_view_counts(config: Mapping[str, Any]) -> list[int]:
    return _configured_view_counts(
        config, key="val_num_views", default=[1, 2, 4, 7]
    )

def _configured_view_counts(
    config: Mapping[str, Any], *, key: str, default: list[int]
) -> list[int]:
    raw = config.get(key, default)
    values = [int(raw)] if isinstance(raw, int) else [int(value) for value in raw]
    if not values or any(value < 1 or value > 7 for value in values):
        raise ValueError(f"{key} must contain values from 1 through 7.")
    return list(dict.fromkeys(values))

def _forward_model(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    training: bool,
    online_vggt: OnlineVGGTFeatureProvider | None = None,
) -> dict[str, torch.Tensor]:
    canonical_index: int | None = None
    canonical_threshold: float | None = None
    if str(batch.get("group_type")) == VISIBLE_AUGMENTED_GROUP:
        mode_key = (
            "radius_refiner_canonical_geometry_mode"
            if training
            else "radius_refiner_canonical_geometry_eval_mode"
        )
        default_mode = "auto" if training else "audit"
        mode = str(config.get(mode_key, default_mode)).lower()
        if mode != "disabled":
            variants = [str(value) for value in batch["counterfactual_variant"]]
            canonical_index = variants.index(ORIGINAL_VARIANT)
            configured = float(
                config.get("radius_refiner_canonical_geometry_p95_threshold_mm", 0.05)
            )
            canonical_threshold = {
                "audit": 1.0e12,
                "auto": configured,
                "always": 0.0,
            }[mode]
    return model(
        views=batch["view_features"].to(device, non_blocking=True),
        view_mask=batch["view_mask"].to(device, non_blocking=True),
        image_features=resolve_batch_image_features(
            batch, device, online_vggt
        ),
        images=batch["images"].to(device, non_blocking=True),
        projection_center_offset=batch["projection_center_offset"].to(
            device, non_blocking=True
        ),
        refiner_branch_mask=(
            refiner_branch_mask_for_batch(batch, config, device)
            if training
            else None
        ),
        radius_refiner_canonical_geometry_member_index=canonical_index,
        radius_refiner_canonical_geometry_p95_threshold_mm=canonical_threshold,
    )

def _finite_percentile(values: list[np.ndarray], percentile: float) -> float:
    if not values:
        return float("nan")
    merged = np.concatenate(values)
    merged = merged[np.isfinite(merged)]
    return float(np.percentile(merged, percentile)) if merged.size else float("nan")

def run_radius_refiner_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str | None = None,
    per_group_records: list[dict[str, Any]] | None = None,
    online_vggt: OnlineVGGTFeatureProvider | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    phase_label = phase or ("train" if training else "evaluation")
    progress_enabled = config.get("progress_enabled", True)
    if not isinstance(progress_enabled, bool):
        raise TypeError("progress_enabled must be a JSON boolean.")
    progress_every = int(config.get("progress_every_groups", 10))
    progress_min_interval = float(
        config.get("progress_min_interval_seconds", 30.0)
    )
    if progress_every < 1:
        raise ValueError("progress_every_groups must be >= 1.")
    if not math.isfinite(progress_min_interval) or progress_min_interval <= 0.0:
        raise ValueError("progress_min_interval_seconds must be finite and > 0.")
    total_groups = len(loader)
    progress_start = time.perf_counter()
    last_progress_time = progress_start
    if progress_enabled:
        print(
            f"[Epoch {epoch:04d}][{phase_label}] starting "
            f"{total_groups} group(s)...",
            flush=True,
        )
    set_parametric_model_training_mode(model, training=training, config=config)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(int(epoch))
    sums: dict[str, float] = {}
    group_type_sums: dict[str, float] = {}
    group_type_counts: dict[str, int] = {}
    radius_errors: list[np.ndarray] = []
    coarse_radius_errors: list[np.ndarray] = []
    stenosis_errors: list[np.ndarray] = []
    coarse_stenosis_errors: list[np.ndarray] = []
    visible_stenosis_errors: list[np.ndarray] = []
    coarse_visible_stenosis_errors: list[np.ndarray] = []
    invisible_stenosis_errors: list[np.ndarray] = []
    coarse_invisible_stenosis_errors: list[np.ndarray] = []
    geometry_p95_values: list[np.ndarray] = []
    local_dice_score_sum = 0.0
    local_dice_view_count = 0.0
    group_count = member_count = 0
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = _forward_model(
                model,
                batch,
                config,
                device,
                training=training,
                online_vggt=online_vggt,
            )
            losses = compute_counterfactual_radius_refiner_loss(
                output, batch, config
            )
            if training:
                losses["loss"].backward()
                parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config.get("grad_clip_norm", 1.0))
                )
                optimizer.step()
        group_count += 1
        current_members = int(batch["view_features"].shape[0])
        member_count += current_members
        scalar_losses = [
            (key, value)
            for key, value in losses.items()
            if value.numel() == 1
        ]
        scalar_values = torch.stack(
            [value.detach().reshape(()) for _, value in scalar_losses]
        ).cpu().tolist()
        loss_values = {
            key: float(value)
            for (key, _), value in zip(scalar_losses, scalar_values)
        }
        for key, value in loss_values.items():
            if key in {
                "radius_refiner_local_stenosis_dice_score",
                "radius_refiner_local_stenosis_dice_view_count",
            }:
                continue
            sums[key] = sums.get(key, 0.0) + value
        group_local_dice_views = loss_values[
            "radius_refiner_local_stenosis_dice_view_count"
        ]
        if group_local_dice_views > 0.0:
            local_dice_score_sum += group_local_dice_views * loss_values[
                "radius_refiner_local_stenosis_dice_score"
            ]
            local_dice_view_count += group_local_dice_views
        group_type = str(batch["group_type"])
        group_type_sums[group_type] = (
            group_type_sums.get(group_type, 0.0) + loss_values["loss"]
        )
        group_type_counts[group_type] = group_type_counts.get(group_type, 0) + 1

        predicted = output["decoded_vessel_mm"][..., 3].detach().cpu()
        coarse = (
            output["radius_refiner_coarse_decoded_vessel_mm"][..., 3]
            .detach()
            .cpu()
        )
        target = batch["target_raw_vessel_mm"][..., 3].detach().cpu()
        valid = (
            batch["target_point_valid_mask"].detach().cpu().bool()
            & (batch["target_branch_exist"].detach().cpu() > 0.5).unsqueeze(-1)
        )
        error = (predicted - target).abs()
        coarse_error = (coarse - target).abs()
        group_radius_error = error[valid].numpy()
        group_coarse_radius_error = coarse_error[valid].numpy()
        radius_errors.append(group_radius_error)
        coarse_radius_errors.append(group_coarse_radius_error)
        region = batch["stenosis_region_point_mask"].detach().cpu().bool() & valid
        group_region_error = error[region].numpy()
        group_coarse_region_error = coarse_error[region].numpy()
        if bool(region.any()):
            stenosis_errors.append(group_region_error)
            coarse_stenosis_errors.append(group_coarse_region_error)
            if group_type == VISIBLE_AUGMENTED_GROUP:
                visible_stenosis_errors.append(group_region_error)
                coarse_visible_stenosis_errors.append(group_coarse_region_error)
            elif group_type == INVISIBLE_STENOSIS_GROUP:
                invisible_stenosis_errors.append(group_region_error)
                coarse_invisible_stenosis_errors.append(group_coarse_region_error)
        if per_group_records is not None:
            record: dict[str, Any] = {
                "case_id": str(batch.get("group_case_id", "")),
                "group_id": str(batch.get("group_id", "")),
                "group_type": group_type,
                "counterfactual_variants": [
                    str(value) for value in batch["counterfactual_variant"]
                ],
                "selected_view_indices": batch[
                    "selected_group_view_indices"
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .tolist(),
                "selected_has_visible_stenosis": bool(
                    batch["selected_group_has_visible_stenosis"].item()
                ),
                "coarse_radius_mae_mm": float(
                    group_coarse_radius_error.mean()
                ),
                "refined_radius_mae_mm": float(group_radius_error.mean()),
                "radius_mae_improvement_mm": float(
                    group_coarse_radius_error.mean() - group_radius_error.mean()
                ),
                "coarse_radius_p95_mm": _finite_percentile(
                    [group_coarse_radius_error], 95.0
                ),
                "refined_radius_p95_mm": _finite_percentile(
                    [group_radius_error], 95.0
                ),
            }
            if group_region_error.size:
                record.update(
                    coarse_stenosis_interval_mae_mm=float(
                        group_coarse_region_error.mean()
                    ),
                    refined_stenosis_interval_mae_mm=float(
                        group_region_error.mean()
                    ),
                    stenosis_interval_mae_improvement_mm=float(
                        group_coarse_region_error.mean()
                        - group_region_error.mean()
                    ),
                    coarse_stenosis_interval_p95_mm=_finite_percentile(
                        [group_coarse_region_error], 95.0
                    ),
                    refined_stenosis_interval_p95_mm=_finite_percentile(
                        [group_region_error], 95.0
                    ),
                )
            if "radius_refiner_geometry_canonicalization_applied" in output:
                record["geometry_canonicalization_applied"] = bool(
                    output["radius_refiner_geometry_canonicalization_applied"]
                    .detach()
                    .max()
                    .item()
                    > 0.5
                )
            record.update(loss_values)
            record["local_stenosis_dice_applicable"] = (
                group_local_dice_views > 0.0
            )
            per_group_records.append(record)
        if "radius_refiner_precanonical_geometry_p95_mm" in output:
            geometry_p95_values.append(
                output["radius_refiner_precanonical_geometry_p95_mm"]
                .detach()
                .cpu()
                .numpy()
            )
        now = time.perf_counter()
        if progress_enabled and _progress_report_due(
            completed=group_count,
            total=total_groups,
            every_groups=progress_every,
            seconds_since_report=now - last_progress_time,
            minimum_interval_seconds=progress_min_interval,
        ):
            elapsed = max(now - progress_start, 1e-9)
            groups_per_second = group_count / elapsed
            remaining = max(total_groups - group_count, 0)
            eta_seconds = remaining / max(groups_per_second, 1e-12)
            current_loss = float(losses["loss"].detach())
            average_loss = sums.get("loss", 0.0) / group_count
            selected_views = int(
                batch["selected_group_view_indices"].numel()
            )
            case_id = str(batch.get("group_case_id", "?"))
            learning_rate = ""
            if training:
                learning_rate = (
                    f" | lr={float(optimizer.param_groups[0]['lr']):.3g}"
                )
            print(
                f"[Epoch {epoch:04d}][{phase_label}] "
                f"{group_count}/{total_groups} groups "
                f"({100.0 * group_count / max(total_groups, 1):5.1f}%) "
                f"| loss={current_loss:.5f} avg={average_loss:.5f} "
                f"| elapsed={_format_progress_duration(elapsed)} "
                f"ETA={_format_progress_duration(eta_seconds)} "
                f"| {groups_per_second:.3f} groups/s "
                f"| case={case_id} type={group_type} "
                f"members={current_members} views={selected_views}"
                f"{learning_rate}",
                flush=True,
            )
            last_progress_time = now
    if group_count == 0:
        raise ValueError("Radius-refiner epoch received no groups.")
    metrics = {key: value / group_count for key, value in sums.items()}
    metrics["group_count"] = float(group_count)
    metrics["member_count"] = float(member_count)
    metrics["radius_refiner_local_stenosis_dice_view_count"] = (
        local_dice_view_count
    )
    metrics["radius_refiner_local_stenosis_dice_score"] = (
        local_dice_score_sum / local_dice_view_count
        if local_dice_view_count > 0.0
        else float("nan")
    )
    metrics["radius_p95_mm"] = _finite_percentile(radius_errors, 95.0)
    metrics["coarse_radius_p95_mm"] = _finite_percentile(
        coarse_radius_errors, 95.0
    )
    metrics["radius_p95_improvement_mm"] = (
        metrics["coarse_radius_p95_mm"] - metrics["radius_p95_mm"]
    )
    metrics["stenosis_interval_radius_p95_mm"] = _finite_percentile(
        stenosis_errors, 95.0
    )
    metrics["coarse_stenosis_interval_radius_p95_mm"] = _finite_percentile(
        coarse_stenosis_errors, 95.0
    )
    metrics["stenosis_interval_radius_p95_improvement_mm"] = (
        metrics["coarse_stenosis_interval_radius_p95_mm"]
        - metrics["stenosis_interval_radius_p95_mm"]
    )
    for label, final_values, coarse_values in (
        (
            "visible_stenosis_interval",
            visible_stenosis_errors,
            coarse_visible_stenosis_errors,
        ),
        (
            "invisible_stenosis_interval",
            invisible_stenosis_errors,
            coarse_invisible_stenosis_errors,
        ),
    ):
        final_p95 = _finite_percentile(final_values, 95.0)
        coarse_p95 = _finite_percentile(coarse_values, 95.0)
        metrics[f"{label}_radius_p95_mm"] = final_p95
        metrics[f"coarse_{label}_radius_p95_mm"] = coarse_p95
        metrics[f"{label}_radius_p95_improvement_mm"] = coarse_p95 - final_p95
    metrics["precanonical_geometry_member_p95_mm"] = _finite_percentile(
        geometry_p95_values, 95.0
    )
    for group_type, count in group_type_counts.items():
        metrics[f"{group_type}_group_count"] = float(count)
        metrics[f"{group_type}_loss"] = group_type_sums[group_type] / count
    return metrics

def _mean_metrics(metric_sets: Iterable[dict[str, float]]) -> dict[str, float]:
    metric_sets = list(metric_sets)
    keys = set().union(*(metrics for metrics in metric_sets)) if metric_sets else set()
    result: dict[str, float] = {}
    for key in keys:
        values = [
            float(metrics[key])
            for metrics in metric_sets
            if key in metrics and math.isfinite(float(metrics[key]))
        ]
        if values:
            result[key] = float(np.mean(values))
    return result

def _write_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    keys = sorted({key for row in history for key in row})
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    save_json(output_dir / "training_monitor" / "history.json", history)
    plot_history(history, output_dir / "training_monitor" / "loss_plot_latest.png")

def _group_manifest(
    datasets: Mapping[str, Stage35RadiusRefinerGroupDataset],
) -> dict[str, list[dict[str, Any]]]:
    manifest: dict[str, list[dict[str, Any]]] = {}
    for split_name, dataset in datasets.items():
        records: list[dict[str, Any]] = []
        for group in dataset.groups:
            region_indices = np.argwhere(
                np.asarray(group.stenosis_region_point_mask, dtype=bool)
            )
            records.append(
                {
                    "case_id": group.case_id,
                    "group_id": group.group_id,
                    "group_type": group.group_type,
                    "variants": list(group.variants),
                    "paths": [str(path) for path in group.paths],
                    "file_signatures": [
                        {
                            "path": str(path),
                            "size_bytes": int(path.stat().st_size),
                            "mtime_ns": int(path.stat().st_mtime_ns),
                        }
                        for path in group.paths
                    ],
                    "visible_view_indices": np.flatnonzero(
                        np.asarray(group.stenosis_view_visible_mask, dtype=bool)
                    )
                    .astype(int)
                    .tolist(),
                    "variant_has_stenosis": {
                        variant: bool(value)
                        for variant, value in zip(
                            group.variants, group.variant_has_stenosis
                        )
                    },
                    "variant_visible_view_indices": {
                        variant: np.flatnonzero(
                            np.asarray(mask, dtype=bool)
                        )
                        .astype(int)
                        .tolist()
                        for variant, mask in zip(
                            group.variants,
                            group.variant_stenosis_view_visible_masks,
                        )
                    },
                    "stenosis_region_branch_point_indices": region_indices.astype(
                        int
                    ).tolist(),
                }
            )
        manifest[split_name] = records
    return manifest

def _capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Keep this weights-only-checkpoint compatible: NumPy ndarrays require
        # unsafe pickle globals under newer PyTorch, whereas tensors/scalars do
        # not.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(
                np.asarray(numpy_state[1], dtype=np.uint32).astype(np.int64)
            ),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        state["train_loader_generator"] = generator.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(state: Any, train_loader: DataLoader) -> None:
    if not isinstance(state, Mapping):
        raise ValueError(
            "Resume checkpoint is missing rng_state; exact grouped/view/dropout "
            "continuation is not possible. Start a new run instead."
        )
    required = {"python", "numpy", "torch_cpu", "train_loader_generator"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Resume checkpoint rng_state is missing {missing}.")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("Resume checkpoint NumPy RNG state is invalid.")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].detach().cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    generator = getattr(train_loader, "generator", None)
    if generator is None:
        raise ValueError("Training DataLoader has no reproducible generator.")
    generator.set_state(state["train_loader_generator"].detach().cpu())
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise ValueError(
                "CUDA resume checkpoint is missing torch_cuda RNG state."
            )
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in cuda_state])

def _monitor_datasets(
    datasets: Mapping[str, Stage35RadiusRefinerGroupDataset],
    config: Mapping[str, Any],
) -> list[tuple[str, str, Stage35RadiusRefinerGroupDataset]]:
    raw_counts = config.get(
        "monitor_num_groups_per_split", {"train": 2, "val": 2, "test": 0}
    )
    counts = (
        {name: int(raw_counts) for name in ("train", "val", "test")}
        if isinstance(raw_counts, int)
        else dict(raw_counts)
    )
    monitor_count = int(config.get("monitor_num_views", 2))
    selected: list[tuple[str, str, Stage35RadiusRefinerGroupDataset]] = []
    for split_name in ("train", "val", "test"):
        dataset = datasets.get(split_name)
        if dataset is None:
            continue
        groups = [
            group
            for group in dataset.groups
            if group.group_type == VISIBLE_AUGMENTED_GROUP
        ][: int(counts.get(split_name, 0))]
        for group in groups:
            monitor_config = dict(config)
            prefix = "validation" if split_name == "val" else (
                "test" if split_name == "test" else "train"
            )
            if split_name == "train":
                monitor_config["train_view_count_weights"] = {
                    str(monitor_count): 1.0
                }
                monitor_config["min_train_views"] = monitor_count
                monitor_config["max_train_views"] = monitor_count
            else:
                monitor_config[f"{prefix}_view_count_choices"] = [monitor_count]
                monitor_config[f"{prefix}_view_count_probabilities"] = [1.0]
            monitor_config[
                f"radius_refiner_{split_name}_require_visible_view"
            ] = True
            monitor_config[
                f"radius_refiner_{split_name}_resample_views_each_epoch"
            ] = False
            single = Stage35RadiusRefinerGroupDataset(
                [group], monitor_config, split=split_name
            )
            selected.append((split_name, group.case_id, single))
    return selected

def _save_monitors(
    *,
    model: torch.nn.Module,
    monitor_datasets: list[tuple[str, str, Stage35RadiusRefinerGroupDataset]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    epoch: int,
    online_vggt: OnlineVGGTFeatureProvider | None = None,
) -> None:
    set_parametric_model_training_mode(model, training=False, config=config)
    with torch.inference_mode():
        for split_name, case_id, dataset in monitor_datasets:
            batch = next(iter(_loader(dataset, config, train=False)))
            output = _forward_model(
                model,
                batch,
                config,
                device,
                training=False,
                online_vggt=online_vggt,
            )
            losses = compute_counterfactual_radius_refiner_loss(
                output, batch, config
            )
            path = (
                output_dir
                / "training_monitor"
                / f"epoch_{epoch:04d}"
                / split_name
                / f"case_{case_id}"
            )
            save_counterfactual_radius_refiner_monitor(
                output=output,
                batch=batch,
                losses=losses,
                path=path,
                epoch=epoch,
                config=config,
            )

def _load_model_state(
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    resuming: bool,
) -> None:
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if resuming:
        if missing or unexpected:
            raise RuntimeError(
                "Resume checkpoint is not exact: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        return
    allowed_missing = {
        key for key in missing if key.startswith("radius_evidence_refiner.")
    }
    disallowed_missing = sorted(missing - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Geometry checkpoint is not architecture-compatible: "
            f"missing={disallowed_missing}, unexpected={sorted(unexpected)}."
        )
    state_keys = set(checkpoint["model_state_dict"])
    if not any(key.startswith("bspline_control_refiner.") for key in state_keys):
        raise RuntimeError(
            "The initial checkpoint has no trained bspline_control_refiner "
            "weights. Supply the geometry-refiner checkpoint, not the baseline."
        )
    print(
        "Initialized frozen coarse/geometry model; newly initialized radius "
        f"refiner tensors: {len(allowed_missing)}"
    )

def _validate_resume_config(
    current: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    previous = checkpoint.get("config", {})
    if not isinstance(previous, Mapping):
        raise ValueError("Resume checkpoint config is invalid.")
    keys = (
        "train_view_count_weights",
        "train_view_count_choices",
        "train_view_count_probabilities",
        "val_num_views",
        "radius_refiner_train_require_visible_view",
        "radius_refiner_val_require_visible_view",
        "radius_refiner_feature_dataset_dir",
        "feature_dataset_dir",
        "raw_image_dataset_dir",
        "radius_refiner_stage3_1_visibility_report",
        "radius_refiner_split_json_paths",
        "radius_refiner_split_missing_case_policy",
        "radius_refiner_vessel_type",
        "radius_refiner_require_all_configured_branches",
        "group_sampling_strategy",
        "feature_key",
        "expected_vggt_context_mode",
        "vggt_context_mode",
        "image_key",
        "view_feature_key",
        "input_scale_to_mm",
        "target_coordinate_frame",
        "projection_coord_scale_to_meter",
        "projection_mask_threshold",
        "radius_refiner_random_view_order",
        "radius_refiner_resample_views_each_epoch",
        "radius_refiner_train_resample_views_each_epoch",
        "seed",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "num_epochs",
        "radius_refiner_canonical_geometry_mode",
        "radius_refiner_canonical_geometry_eval_mode",
        "radius_refiner_canonical_geometry_p95_threshold_mm",
    )
    for key in keys:
        if previous.get(key) != current.get(key):
            raise ValueError(
                f"Cannot resume after changing {key}: checkpoint="
                f"{previous.get(key)!r}, config={current.get(key)!r}. Start a "
                "new run from initial_checkpoint instead."
            )
    for key in (
        "num_branches",
        "num_points",
        "radius_refiner_num_stages",
        "radius_refiner_evidence_hidden_dim",
        "radius_refiner_profile_samples",
        "radius_refiner_profile_half_width_px",
        "radius_refiner_image_size",
        "radius_refiner_sid",
        "radius_refiner_source_to_iso",
        "radius_refiner_imager_pixel_spacing",
        "radius_refiner_residual_scale_mm",
        "radius_refiner_radius_value_scale_mm",
        "radius_refiner_min_radius_mm",
        "radius_refiner_render_num_circle_points",
        "radius_refiner_render_radial_subsamples",
        "radius_refiner_render_axial_subsamples",
        "radius_refiner_branch_probability_threshold",
    ):
        default_value = (
            0.5
            if key == "radius_refiner_branch_probability_threshold"
            else None
        )
        previous_value = _model_value(previous, key, default_value)
        current_value = _model_value(current, key, default_value)
        if previous_value != current_value:
            raise ValueError(
                f"Cannot resume after changing model key {key}: checkpoint="
                f"{previous_value!r}, config={current_value!r}."
            )
    for key in _LOSS_SEMANTIC_KEYS:
        if _loss_value(previous, key) != _loss_value(current, key):
            raise ValueError(
                f"Cannot resume after changing loss key {key}. Start a new run."
            )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Stage-3 counterfactual parametric radius refiner."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume a dedicated checkpoint path, or latest.pt.",
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    provisional_device = device_from_config(raw_config)
    config, source_checkpoint, checkpoint_path, resuming = (
        _load_merged_training_config(
            args.config,
            cli_resume=args.resume,
            cli_output_dir=args.output_dir,
            device=provisional_device,
        )
    )
    backbone = normalize_feature_backbone(config.get("feature_backbone", "vggt"))
    config["feature_backbone"] = backbone
    online_vggt_enabled = uses_online_vggt_features(config)
    if online_vggt_enabled:
        raw_image_dataset_dir = resolve_raw_image_dataset_dir(config)
        config["raw_image_dataset_dir"] = str(raw_image_dataset_dir)
        config["resolved_raw_image_dataset_dir"] = str(raw_image_dataset_dir)
        config["resolved_feature_dataset_dir"] = None
        config["model_input_mode"] = "online_frozen_vggt"
        config["load_images"] = True
    elif config.get("feature_dataset_dir") is not None:
        config["resolved_feature_dataset_dir"] = str(
            resolve_feature_dataset_dir(config, backbone)
        )
        config["model_input_mode"] = "precomputed_features"
    output_dir = resolve_train_output_config(config, backbone)
    if not resuming:
        existing_checkpoints = [
            path
            for path in (output_dir / "latest.pt", output_dir / "best.pt")
            if path.is_file()
        ]
        if existing_checkpoints:
            raise FileExistsError(
                "Fresh radius-refiner training would overwrite an existing "
                f"checkpoint ({existing_checkpoints[0]}). Use --resume or a "
                "new experiment_name/output directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_architecture_against_checkpoint(config, source_checkpoint)
    validate_counterfactual_radius_refiner_training_config(config)
    if resuming:
        _validate_resume_config(config, source_checkpoint)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    report_path = config.get("radius_refiner_stage3_1_visibility_report")
    if report_path is None:
        report_path = config.get("stage3_1_visibility_report")
    print(
        "[Radius refiner] Discovering grouped counterfactual cases from "
        + (
            "raw images with a separate Stage-3.1 visibility report..."
            if report_path is not None
            else "the enriched dataset..."
        ),
        flush=True,
    )
    datasets = build_radius_refiner_group_datasets(config)
    if "val" not in datasets:
        raise ValueError(
            "The inherited split contains no validation groups. Radius-refiner "
            "checkpoint selection must not reuse the training groups."
        )
    split_summary = {
        split_name: summarize_radius_refiner_groups(dataset.groups)
        for split_name, dataset in datasets.items()
    }
    group_manifest = _group_manifest(datasets)
    if resuming:
        previous_manifest = source_checkpoint.get("config", {}).get(
            "radius_refiner_group_manifest"
        )
        if previous_manifest != group_manifest:
            raise ValueError(
                "The Stage-3.5 split/group manifest changed since the resume "
                "checkpoint. Start a new run so variants, visibility labels, "
                "and case membership cannot change mid-optimization."
            )
    config["radius_refiner_group_manifest"] = group_manifest
    config["radius_refiner_group_summary"] = split_summary
    train_view_count_probabilities = {
        str(int(count)): float(probability)
        for count, probability in zip(
            datasets["train"].choices,
            datasets["train"].probabilities,
        )
    }
    config["resolved_train_view_count_probabilities"] = (
        train_view_count_probabilities
    )
    save_json(output_dir / "radius_refiner_group_manifest.json", group_manifest)
    save_json(output_dir / "radius_refiner_group_summary.json", split_summary)
    print(
        f"[Radius refiner groups] {json.dumps(split_summary, sort_keys=True)}",
        flush=True,
    )
    print(
        "[Radius refiner view counts] "
        + ", ".join(
            f"{count}={probability:.1%}"
            for count, probability in train_view_count_probabilities.items()
        ),
        flush=True,
    )
    if split_summary["train"].get(VISIBLE_AUGMENTED_GROUP, 0) == 0:
        raise ValueError(
            "The inherited training intersection contains no visible augmented "
            "triplet. There would be no counterfactual stenosis-refinement "
            "signal; inspect the Stage-3.1 visibility report and split paths."
        )

    print(
        "[Radius refiner] Loading a sample and constructing the model...",
        flush=True,
    )
    device = device_from_config(config)
    online_vggt: OnlineVGGTFeatureProvider | None = None
    if online_vggt_enabled:
        online_vggt = OnlineVGGTFeatureProvider(
            config,
            device=device,
            raw_image_dataset_dir=config["raw_image_dataset_dir"],
        )
        config.update(online_vggt.resolved_config())
        print(
            "[Radius refiner] Extracting frozen VGGT features on the fly "
            f"with context_mode={online_vggt.context_mode!r}.",
            flush=True,
        )
    sample_batch = datasets["train"][0]
    view_feat_dim = int(sample_batch["view_features"].shape[-1])
    inferred_feature_dim: int | None = None
    image_features = sample_batch["image_features"]
    if online_vggt is not None:
        inferred_feature_dim = int(online_vggt.token_dim)
    elif isinstance(image_features, dict):
        channels = [
            int(image_features[key].shape[2])
            for key in ("c2", "c3", "c4", "c5")
        ]
        config.setdefault("model", {})["resnet_pre_fpn_channels"] = channels
    else:
        if image_features is None:
            raise RuntimeError(
                "Radius-refiner sample has no cached features, but online VGGT "
                "was not configured."
            )
        inferred_feature_dim = int(image_features.shape[-1])
    stored_view_dim = source_checkpoint.get("view_feat_dim")
    if stored_view_dim is not None and int(stored_view_dim) != view_feat_dim:
        raise ValueError(
            f"Stage-3 view feature dimension {view_feat_dim} does not match "
            f"checkpoint dimension {stored_view_dim}."
        )

    model = build_model_for_checkpoint(
        config,
        view_feat_dim,
        inferred_feature_dim,
        source_checkpoint["model_state_dict"],
    ).to(device)
    _load_model_state(
        model=model, checkpoint=source_checkpoint, resuming=resuming
    )
    trainable_parameters = configure_bspline_refiner_only_training(model, config)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("radius_evidence_refiner.") for name in trainable_names
    ):
        raise RuntimeError(
            "Dedicated scope must train only radius_evidence_refiner parameters; "
            f"got {trainable_names[:20]}."
        )
    print(
        f"Training {config['num_trainable_parameters']:,} radius-refiner "
        f"parameters; frozen parameters: {config['num_frozen_parameters']:,}",
        flush=True,
    )

    num_epochs = int(config.get("num_epochs", 100))
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs, 1),
        eta_min=float(config.get("min_learning_rate", 1e-6)),
    )
    train_loader = _loader(datasets["train"], config, train=True)
    val_counts = _validation_view_counts(config)
    val_loaders = {
        view_count: _loader(
            _fixed_view_dataset(
                datasets["val"], config, view_count=view_count
            ),
            config,
            train=False,
        )
        for view_count in val_counts
    }
    monitors = _monitor_datasets(datasets, config)

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_loss = float("inf")
    if resuming:
        model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" not in source_checkpoint:
            raise KeyError("Resume checkpoint is missing optimizer_state_dict.")
        optimizer.load_state_dict(source_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in source_checkpoint:
            scheduler.load_state_dict(source_checkpoint["scheduler_state_dict"])
        _restore_rng_state(source_checkpoint.get("rng_state"), train_loader)
        history = list(source_checkpoint.get("history", []))
        start_epoch = int(source_checkpoint["epoch"]) + 1
        best_loss = float(source_checkpoint.get("best_val_loss", best_loss))
        print(f"Resuming {checkpoint_path} at epoch {start_epoch}")

    config["checkpoint_selection_metric"] = "mean_validation_group_loss"
    config["checkpoint_selection_view_counts"] = val_counts
    config["num_epochs"] = num_epochs
    monitor_generate_gifs = config.get("monitor_generate_gifs", False)
    if not isinstance(monitor_generate_gifs, bool):
        raise ValueError("monitor_generate_gifs must be a JSON boolean.")
    config["monitor_generate_gifs"] = monitor_generate_gifs
    save_json(output_dir / "resolved_config.json", config)
    monitor_every = int(config.get("monitor_every_epochs", 10))
    if monitor_every < 1:
        raise ValueError("monitor_every_epochs must be >= 1.")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            f"[Epoch {epoch:04d}/{num_epochs:04d}] beginning one training "
            f"pass and {len(val_counts)} validation pass(es) at view counts "
            f"{val_counts}.",
            flush=True,
        )
        train_metrics = run_radius_refiner_epoch(
            model=model,
            loader=train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
            online_vggt=online_vggt,
        )
        validation = {
            count: run_radius_refiner_epoch(
                model=model,
                loader=loader,
                config=config,
                device=device,
                optimizer=None,
                epoch=epoch,
                phase=f"val@{count}-views",
                online_vggt=online_vggt,
            )
            for count, loader in val_loaders.items()
        }
        mean_val = _mean_metrics(validation.values())
        current_val_loss = float(mean_val["loss"])
        if not math.isfinite(float(train_metrics["loss"])) or not math.isfinite(
            current_val_loss
        ):
            raise FloatingPointError(
                f"Non-finite radius-refiner objective at epoch {epoch}: "
                f"train={train_metrics['loss']}, val={current_val_loss}."
            )
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        for count, metrics in validation.items():
            row.update({f"val_k{count}_{key}": value for key, value in metrics.items()})
        row.update({f"val_mean_{key}": value for key, value in mean_val.items()})
        history.append(row)
        print(
            f"[Epoch {epoch:04d}] train loss={train_metrics['loss']:.5f}, "
            f"radius={train_metrics['radius_refiner_radius_mae_mm']:.4f} mm | "
            f"val mean loss={current_val_loss:.5f}, "
            f"radius={mean_val['radius_refiner_radius_mae_mm']:.4f} mm",
            flush=True,
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "view_feat_dim": view_feat_dim,
            "inferred_feature_dim": inferred_feature_dim,
            "best_val_loss": min(best_loss, current_val_loss),
            "history": history,
            "rng_state": _capture_rng_state(train_loader),
        }
        if online_vggt is not None:
            config["online_vggt_cache_stats"] = online_vggt.stats()
        torch.save(checkpoint, output_dir / "latest.pt")
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            torch.save(checkpoint, output_dir / "best.pt")
        _write_history(history, output_dir)
        if epoch % monitor_every == 0 or epoch == num_epochs:
            print(
                f"[Epoch {epoch:04d}] saving configured training monitors...",
                flush=True,
            )
            _save_monitors(
                model=model,
                monitor_datasets=monitors,
                config=config,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                online_vggt=online_vggt,
            )

    if "test" in datasets and bool(config.get("evaluate_test_after_training", True)):
        best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        test_counts = _configured_view_counts(
            config,
            key="test_num_views",
            default=[1, 2, 3, 4, 5, 6, 7],
        )
        test_metrics: dict[str, dict[str, float]] = {}
        test_case_metrics: dict[str, list[dict[str, Any]]] = {}
        for count in test_counts:
            test_dataset = _fixed_view_dataset(
                datasets["test"], config, view_count=count
            )
            case_records: list[dict[str, Any]] = []
            test_metrics[f"k{count}"] = run_radius_refiner_epoch(
                model=model,
                loader=_loader(test_dataset, config, train=False),
                config=config,
                device=device,
                optimizer=None,
                epoch=num_epochs,
                phase=f"test@{count}-views",
                per_group_records=case_records,
                online_vggt=online_vggt,
            )
            test_case_metrics[f"k{count}"] = case_records
        save_json(output_dir / "test_metrics.json", test_metrics)
        save_json(output_dir / "test_case_metrics.json", test_case_metrics)
        print(f"Saved counterfactual test metrics to {output_dir / 'test_metrics.json'}")
        if bool(
            config.get("evaluate_visible_conditioned_test_after_training", True)
        ):
            conditioned_config = dict(config)
            conditioned_config[
                "radius_refiner_test_require_visible_view"
            ] = True
            conditioned_metrics: dict[str, dict[str, float]] = {}
            conditioned_case_metrics: dict[str, list[dict[str, Any]]] = {}
            for count in test_counts:
                conditioned_dataset = _fixed_view_dataset(
                    datasets["test"],
                    conditioned_config,
                    view_count=count,
                )
                conditioned_case_records: list[dict[str, Any]] = []
                conditioned_metrics[f"k{count}"] = run_radius_refiner_epoch(
                    model=model,
                    loader=_loader(
                        conditioned_dataset, conditioned_config, train=False
                    ),
                    config=conditioned_config,
                    device=device,
                    optimizer=None,
                    epoch=num_epochs,
                    phase=f"test-visible@{count}-views",
                    per_group_records=conditioned_case_records,
                    online_vggt=online_vggt,
                )
                conditioned_case_metrics[f"k{count}"] = (
                    conditioned_case_records
                )
            conditioned_path = output_dir / "test_visible_conditioned_metrics.json"
            save_json(conditioned_path, conditioned_metrics)
            save_json(
                output_dir / "test_visible_conditioned_case_metrics.json",
                conditioned_case_metrics,
            )
            print(
                "Saved oracle-conditioned diagnostic test metrics to "
                f"{conditioned_path}"
            )

if __name__ == "__main__":
    main()
