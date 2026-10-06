# Transferred from methods/parametric_methods/train.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch.utils.data import DataLoader
from vessel_code.support.centerline_only import configure_only_centerline_losses
from vessel_code.preprocessing.split_subdirectories import discover_npz_split_subdirectories
from vessel_code.parametric.data import BranchVariantFileGroup, ParametricBranchVariantGroupDataset, ParametricFeatureDataset, case_identifier, collate_parametric_batches, discover_branch_variant_groups, filter_original_parametric_variant_files, load_branch_variant_metadata, load_stage4_1_visibility_metadata, load_paired_items, resolve_branch_variant_group_training, resolve_branch_visibility_sampling, resolve_parametric_original_variant_only, resolve_parametric_num_branches, resolve_parametric_target_num_branches, resolve_parametric_target_source, resolve_feature_dataset_dir, resolve_model_input_dataset_dir, resolve_raw_image_dataset_dir, uses_online_vggt_features
from vessel_code.parametric.centerline_probability import enforce_frozen_centerline_predictor, load_pretrained_centerline_predictor_into_refiner
from vessel_code.parametric.loss import bend_recall_3d_final_weight, branch_length_final_weight, centerline_prediction_mode_from_config, compute_parametric_loss, configure_radius_mode_losses, decoded_local_progress_final_weight, ordered_terminal_masks, radius_prediction_mode_from_config
from vessel_code.parametric.model import build_model_from_config
from vessel_code.parametric.monitoring import plot_history, save_control_point_error_summary, save_json, save_vessel_monitor
from vessel_code.parametric.projection_loss import build_bspline_refiner_centerline_projector, build_centerline_projector
from vessel_code.parametric.online_vggt import OnlineVGGTFeatureProvider
from vessel_code.parametric.sampling_schedule import ReplayTrainingBatchDataset, TrainingSamplingSchedule, resolve_training_sampling_schedule, sampling_policy_from_config
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector
from vessel_code.losses.error_thresholding import configured_error_threshold_names, resolve_error_threshold_settings
from vessel_code.shared.branch_visibility import refiner_branch_existence_source, required_branch_count, resolve_artery_type, validate_visualization_branch_existence_config
from vessel_code.shared.train import _load_explicit_split
from vessel_code.shared.paths import resolve_train_output_config
from vessel_code.shared.data import SplitFiles, discover_npz_files, ensure_item_images, filter_excluded_case_files, infer_feature_shapes, infer_resnet_pre_fpn_channels, load_split_record, normalize_feature_backbone, resolve_clinical_rca_view_indices, resolve_excluded_case_numbers, resolve_excluded_dataset_dir_names, resolve_lazy_load_image_features, resolve_zero_cached_image_features, save_split_record, select_case_files, split_case_files

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _deep_merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key != "train_view_count_weights"
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(
    path: str | Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, config_path))
        raise ValueError(f"Circular config inheritance: {chain}")
    with open(config_path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Config must contain a JSON object")
    extends = value.pop("extends", None)
    if extends is None:
        return value
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("Config 'extends' must be a non-empty path string")
    base_path = Path(extends).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_config(base_path, _stack=(*_stack, config_path))
    merged = _deep_merge_config(base, value)
    merged["config_extends"] = str(base_path.resolve())
    return merged

def device_from_config(config: dict[str, Any]) -> torch.device:
    value = str(config.get("device", "auto"))
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

_GEOMETRY_REFINER_COARSE_CONFIG_KEYS = (
    "feature_backbone",
    "feature_key",
    "backbone_feature_source",
    "vggt_context_mode",
    "expected_vggt_context_mode",
    "vggt_backbone",
    "vggt_pretrained",
    "vggt_model_name",
    "vggt_load_mode",
    "vggt_torchhub_repo",
    "vggt_omega_checkpoint_path",
    "vggt_image_size_mode",
    "vggt_patch_size",
    "vggt_target_image_size",
    "vggt_token_dim",
    "zero_cached_image_features",
    "check_feature_finite",
    "parametric_original_variant_only",
    "artery_type",
    "num_branches",
    "num_points",
    "centerline_prediction_mode",
    "num_control_points",
    "num_landmarks",
    "num_radius_coefficients",
    "num_lesions",
    "lesion_profile",
    "radius_prediction_mode",
    "only_centerline",
    "reconstruction_target",
    "projection_centerline_target",
    "target_coordinate_frame",
    "input_scale_to_mm",
    "parametric_centering_source",
    "projection_coord_scale_to_meter",
    "max_views",
    "min_train_views",
    "max_train_views",
    "train_view_count_weights",
    "train_random_view_order",
    "clinical_rca_views",
    "clinical_rca_view_indices",
    "decoder_architecture",
    "model_dim",
    "num_encoder_layers",
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
    "num_decoder_layers",
    "num_attention_heads",
    "mlp_hidden_dim",
    "dropout",
    "use_learned_side_parent_projection",
)

_GEOMETRY_REFINER_COARSE_TRAINING_PROTOCOL_KEYS = (
    "branch_variant_group_training",
    "branch_variant_train_sampling",
    "branch_variant_resample_each_epoch",
)

def _effective_model_config_value(
    config: dict[str, Any], key: str
) -> tuple[bool, Any]:
    model_config = config.get("model")
    if isinstance(model_config, dict) and key in model_config:
        return True, model_config[key]
    if key in config:
        return True, config[key]
    return False, None

def validate_geometry_refiner_coarse_config(
    config: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> tuple[str, ...]:
    """Require a frozen geometry refiner to preserve its coarse-model contract."""

    if not bool(config.get("train_bspline_refiner_only", False)):
        return ()
    enforce = config.get(
        "enforce_initial_checkpoint_coarse_config_match", True
    )
    if not isinstance(enforce, bool):
        raise ValueError(
            "enforce_initial_checkpoint_coarse_config_match must be a JSON "
            f"boolean, got {enforce!r}."
        )
    if not enforce:
        return ()

    compared: list[str] = []
    mismatches: list[str] = []
    for key in _GEOMETRY_REFINER_COARSE_CONFIG_KEYS:
        current_present, current_value = _effective_model_config_value(
            config, key
        )
        checkpoint_present, checkpoint_value = _effective_model_config_value(
            checkpoint_config, key
        )
        # Older checkpoints may not record every modern option. State-dict
        # compatibility remains authoritative for fields absent from either
        # resolved configuration.
        if not (current_present and checkpoint_present):
            continue
        compared.append(key)
        if current_value != checkpoint_value:
            mismatches.append(
                f"{key}: checkpoint={checkpoint_value!r}, "
                f"refiner={current_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Geometry-refiner configuration does not preserve the frozen "
            "coarse model's input/architecture contract. Only refiner-specific "
            "settings should differ from the initial checkpoint. Mismatches: "
            + "; ".join(mismatches)
        )

    training_protocol_mismatches: list[str] = []
    for key in _GEOMETRY_REFINER_COARSE_TRAINING_PROTOCOL_KEYS:
        current_present, current_value = _effective_model_config_value(
            config, key
        )
        checkpoint_present, checkpoint_value = _effective_model_config_value(
            checkpoint_config, key
        )
        if not (current_present and checkpoint_present):
            continue
        if current_value != checkpoint_value:
            training_protocol_mismatches.append(
                f"{key}: checkpoint={checkpoint_value!r}, "
                f"refiner={current_value!r}"
            )
    if training_protocol_mismatches:
        warnings.warn(
            "Geometry-refiner training protocol differs from the coarse "
            "checkpoint. This is allowed because these settings control data "
            "sampling/grouping rather than the frozen coarse-model "
            "architecture. Mismatches: "
            + "; ".join(training_protocol_mismatches),
            RuntimeWarning,
            stacklevel=2,
        )
    config["initial_checkpoint_coarse_config_match"] = True
    config["initial_checkpoint_coarse_config_compared_fields"] = compared
    config["initial_checkpoint_training_protocol_mismatches"] = (
        training_protocol_mismatches
    )
    return tuple(compared)

def configure_bspline_refiner_only_training(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> list[torch.nn.Parameter]:
    """Freeze the predictor and select one optional fine-tuning scope."""
    bspline_only = bool(config.get("train_bspline_refiner_only", False))
    radius_only = bool(config.get("train_radius_refiner_only", False))
    radius_head_only = bool(config.get("train_radius_head_only", False))
    bspline_and_radius_head_only = bool(
        config.get("train_bspline_refiner_and_radius_head_only", False)
    )
    enabled_scopes = [
        bspline_only,
        radius_only,
        radius_head_only,
        bspline_and_radius_head_only,
    ]
    if sum(enabled_scopes) > 1:
        raise ValueError(
            "train_bspline_refiner_only, train_radius_refiner_only, "
            "train_radius_head_only, and "
            "train_bspline_refiner_and_radius_head_only are mutually "
            "exclusive."
        )
    selected_modules: list[tuple[str, torch.nn.Module]] = []
    selected_refiner_name = ""
    if bspline_only:
        module = getattr(model, "bspline_control_refiner", None)
        if module is None:
            raise ValueError(
                "train_bspline_refiner_only=true requires "
                "model.use_bspline_control_refiner=true."
            )
        selected_modules.append(("bspline_control_refiner", module))
        selected_refiner_name = "bspline_control_refiner"
    elif radius_only:
        module = getattr(model, "radius_evidence_refiner", None)
        if module is None:
            raise ValueError(
                "train_radius_refiner_only=true requires "
                "model.use_radius_evidence_refiner=true."
            )
        selected_modules.append(("radius_evidence_refiner", module))
        selected_refiner_name = "radius_evidence_refiner"
    elif radius_head_only or bspline_and_radius_head_only:
        scope_name = (
            "train_bspline_refiner_and_radius_head_only"
            if bspline_and_radius_head_only
            else "train_radius_head_only"
        )
        if getattr(model, "bspline_control_refiner", None) is None:
            raise ValueError(
                f"{scope_name}=true requires "
                "model.use_bspline_control_refiner=true."
            )
        if getattr(model, "radius_evidence_refiner", None) is not None:
            raise ValueError(
                f"{scope_name}=true requires "
                "model.use_radius_evidence_refiner=false."
            )
        raw_radius_head = getattr(model, "raw_radius_head", None)
        if raw_radius_head is None:
            raise ValueError(
                f"{scope_name}=true requires "
                "radius_prediction_mode='raw'."
            )
        if bspline_and_radius_head_only:
            selected_modules.append(
                ("bspline_control_refiner", model.bspline_control_refiner)
            )
            selected_refiner_name = "bspline_control_refiner"
        selected_modules.append(("raw_radius_head", raw_radius_head))
        if int(getattr(model, "num_branches", 1)) > 1:
            side_raw_radius_head = getattr(model, "side_raw_radius_head", None)
            # Parallel decoders use the shared raw_radius_head for every branch.
            if side_raw_radius_head is None:
                if getattr(model, "decoder_architecture", "") not in {
                    "legacy_parallel",
                    "absolute_parallel",
                }:
                    raise ValueError(
                        f"{scope_name}=true requires side_raw_radius_head "
                        "for a multi-branch hierarchical model."
                    )
            else:
                selected_modules.append(
                    ("side_raw_radius_head", side_raw_radius_head)
                )

    if not selected_modules:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameters = []
        for _name, module in selected_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        bspline_refiner = getattr(model, "bspline_control_refiner", None)
        if bspline_refiner is not None:
            enforce_frozen_centerline_predictor(bspline_refiner)
        parameters = [
            parameter
            for _name, module in selected_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
    if not parameters:
        raise ValueError("Training configuration produced no trainable parameters.")
    config["train_bspline_refiner_only_effective"] = bspline_only
    config["train_radius_refiner_only_effective"] = radius_only
    config["train_radius_head_only_effective"] = radius_head_only
    config["train_bspline_refiner_and_radius_head_only_effective"] = (
        bspline_and_radius_head_only
    )
    config["trainable_refiner_module"] = selected_refiner_name or None
    config["trainable_modules"] = [
        name for name, _module in selected_modules
    ]
    config["num_trainable_parameters"] = sum(
        parameter.numel() for parameter in parameters
    )
    config["num_frozen_parameters"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return parameters

def set_parametric_model_training_mode(
    model: torch.nn.Module,
    *,
    training: bool,
    config: dict[str, Any],
) -> None:
    """Set train/eval state while keeping a frozen coarse model deterministic."""
    model.train(training)
    bspline_refiner = getattr(model, "bspline_control_refiner", None)
    if bspline_refiner is not None:
        enforce_frozen_centerline_predictor(bspline_refiner)
    if training and (
        bool(config.get("train_bspline_refiner_only", False))
        or bool(config.get("train_radius_refiner_only", False))
        or bool(config.get("train_radius_head_only", False))
        or bool(
            config.get("train_bspline_refiner_and_radius_head_only", False)
        )
    ):
        radius_only = bool(config.get("train_radius_refiner_only", False))
        radius_head_only = bool(config.get("train_radius_head_only", False))
        bspline_and_radius_head_only = bool(
            config.get("train_bspline_refiner_and_radius_head_only", False)
        )
        if radius_head_only or bspline_and_radius_head_only:
            selected_names = ["raw_radius_head"]
            if (
                int(getattr(model, "num_branches", 1)) > 1
                and getattr(model, "side_raw_radius_head", None) is not None
            ):
                selected_names.append("side_raw_radius_head")
            if bspline_and_radius_head_only:
                selected_names.insert(0, "bspline_control_refiner")
        else:
            selected_names = [
                "radius_evidence_refiner"
                if radius_only
                else "bspline_control_refiner"
            ]
        # requires_grad=False does not disable dropout. Keep every coarse module
        # in inference mode so its output remains the checkpoint prediction,
        # then enable training behavior only inside the selected module(s).
        model.eval()
        for module_name in selected_names:
            module = getattr(model, module_name, None)
            if module is None:
                raise ValueError(
                    "Specialized training requires the selected module "
                    f"{module_name!r}."
                )
            module.train(True)
        if bspline_refiner is not None:
            enforce_frozen_centerline_predictor(bspline_refiner)

def configure_pretrained_centerline_predictor(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    load_weights: bool,
) -> dict[str, Any] | None:
    """Optionally initialize and freeze the refiner's 2D centreline network."""

    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model must be a JSON object.")
    checkpoint_value = model_config.get(
        "bspline_refiner_pretrained_centerline_checkpoint",
        config.get("bspline_refiner_pretrained_centerline_checkpoint"),
    )
    freeze_value = model_config.get(
        "bspline_refiner_freeze_pretrained_centerline_predictor",
        config.get(
            "bspline_refiner_freeze_pretrained_centerline_predictor", True
        ),
    )
    if not isinstance(freeze_value, bool):
        raise ValueError(
            "bspline_refiner_freeze_pretrained_centerline_predictor must be "
            "a JSON boolean."
        )
    if checkpoint_value is None or not str(checkpoint_value).strip():
        return None
    refiner = getattr(model, "bspline_control_refiner", None)
    if refiner is None:
        raise ValueError(
            "bspline_refiner_pretrained_centerline_checkpoint requires "
            "model.use_bspline_control_refiner=true."
        )
    use_unexplained = bool(
        model_config.get(
            "bspline_refiner_use_unexplained_centerline_evidence",
            config.get(
                "bspline_refiner_use_unexplained_centerline_evidence", False
            ),
        )
    )
    use_probability_patch = bool(
        model_config.get(
            "bspline_refiner_use_centerline_probability_patch_evidence",
            config.get(
                "bspline_refiner_use_centerline_probability_patch_evidence",
                False,
            ),
        )
    )
    if not (use_unexplained or use_probability_patch):
        raise ValueError(
            "A pretrained centreline checkpoint requires "
            "bspline_refiner_use_unexplained_centerline_evidence=true or "
            "bspline_refiner_use_centerline_probability_patch_evidence=true."
        )
    if load_weights:
        info = load_pretrained_centerline_predictor_into_refiner(
            refiner,
            checkpoint_value,
            freeze=freeze_value,
        )
    else:
        setattr(
            refiner,
            "pretrained_centerline_predictor_frozen",
            freeze_value,
        )
        setattr(
            refiner,
            "pretrained_centerline_predictor_checkpoint",
            str(Path(str(checkpoint_value)).expanduser().resolve()),
        )
        enforce_frozen_centerline_predictor(refiner)
        info = {
            "checkpoint": str(
                Path(str(checkpoint_value)).expanduser().resolve()
            ),
            "source_epoch": None,
            "separate_encoder": getattr(
                refiner, "centerline_image_feature_encoder", None
            )
            is not None,
            "frozen": freeze_value,
            "weights_source": "resume_checkpoint",
        }
    config["pretrained_centerline_predictor"] = info
    if freeze_value and not info["separate_encoder"]:
        warnings.warn(
            "The frozen centreline predictor shares image_feature_encoder with "
            "ordinary refiner evidence, so that local CNN is frozen too. Set "
            "model.bspline_refiner_use_separate_centerline_encoder=true to "
            "keep the ordinary local feature path trainable.",
            stacklevel=2,
        )
    return info

def move_features(value: torch.Tensor | dict[str, torch.Tensor], device: torch.device):
    if isinstance(value, dict):
        return {key: tensor.to(device, non_blocking=True) for key, tensor in value.items()}
    return value.to(device, non_blocking=True)

def refiner_branch_mask_for_batch(
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    """Build a teacher-forced geometry/radius mask, or use predicted gating."""

    source = refiner_branch_existence_source(config)
    if source == "predicted":
        return None
    target = batch.get("target_branch_exist")
    if target is None:
        raise KeyError(
            "refiner_branch_existence_source='ground_truth' requires "
            "target_branch_exist in every training batch."
        )
    target_on_device = target.to(device=device, non_blocking=True)
    mask = (target_on_device > 0.5).clone()
    if mask.dim() != 2:
        raise ValueError(
            "target_branch_exist must have shape [B,M] for refiner gating, "
            f"got {tuple(mask.shape)}."
        )
    fixed_count = min(
        int(mask.shape[1]),
        required_branch_count(resolve_artery_type(config)),
    )
    mask[:, :fixed_count] = True
    return mask

def resolve_batch_image_features(
    batch: dict[str, Any],
    device: torch.device,
    online_vggt: Any | None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    features = batch.get("image_features")
    if features is not None:
        return move_features(features, device)
    if online_vggt is None:
        raise ValueError(
            "The batch has no precomputed image_features and no online frozen "
            "backbone feature provider is configured."
        )
    return online_vggt.features_for_batch(batch)

def _validate_variable_view_context(
    config: dict[str, Any],
    *,
    min_views: int | None,
    max_views: int | None,
) -> None:
    feature_backbone = normalize_feature_backbone(
        config.get("feature_backbone", "vggt")
    )
    if (
        feature_backbone not in ("vggt", "vggt_omega")
        or (min_views is None and max_views is None)
    ):
        return
    if uses_online_vggt_features(config):
        # The selected subset is sent through VGGT together at this step, so
        # its contextual tokens cannot contain evidence from omitted views.
        return
    context_mode = str(
        config.get("expected_vggt_context_mode") or ""
    ).strip().lower()
    if context_mode != "per_view":
        raise ValueError(
            "Variable-view VGGT training requires "
            "expected_vggt_context_mode='per_view'. Tokens cached with "
            "all-view context already contain evidence from views that a "
            "later subset would appear to exclude."
        )

def validate_branch_variant_training_config(config: dict[str, Any]) -> bool:
    """Validate the opt-in grouped Stage-4/5 training contract."""

    enabled = resolve_branch_variant_group_training(config)
    if not enabled:
        return False
    if resolve_parametric_original_variant_only(config):
        raise ValueError(
            "branch_variant_group_training=true is incompatible with "
            "parametric_original_variant_only=true; all Stage-4/5 variants "
            "must remain discoverable."
        )
    # Both target sources are supported. Embedded feature-file targets are
    # already variant-masked; one shared directory target per physical case is
    # masked during pairing from each feature NPZ's training_branch_mask.
    resolve_parametric_target_source(config)
    strategy = str(
        config.get("branch_variant_train_sampling", "all")
    ).strip().lower()
    if strategy not in {"all", "random_one", "random_n"}:
        raise ValueError(
            "branch_variant_train_sampling must be 'all', 'random_one', or "
            f"'random_n', got {strategy!r}."
        )
    variants_per_group = config.get(
        "branch_variant_train_variants_per_group", 1
    )
    try:
        resolved_variants_per_group = int(variants_per_group)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "branch_variant_train_variants_per_group must be an integer >= 1."
        ) from error
    if (
        isinstance(variants_per_group, bool)
        or resolved_variants_per_group < 1
    ):
        raise ValueError(
            "branch_variant_train_variants_per_group must be an integer >= 1."
        )
    resample = config.get("branch_variant_resample_each_epoch", True)
    if not isinstance(resample, bool):
        raise ValueError(
            "branch_variant_resample_each_epoch must be a JSON boolean, got "
            f"{resample!r}."
        )
    groups_per_batch = config.get("branch_variant_groups_per_batch", 1)
    try:
        resolved_groups_per_batch = int(groups_per_batch)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "branch_variant_groups_per_batch must be 1; each variable-size "
            "physical case group is one optimiser step."
        ) from error
    if isinstance(groups_per_batch, bool) or resolved_groups_per_batch != 1:
        raise ValueError(
            "branch_variant_groups_per_batch must be 1; each variable-size "
            "physical case group is one optimiser step."
        )
    if config.get("min_steps_per_epoch") is not None:
        raise ValueError(
            "min_steps_per_epoch is incompatible with "
            "branch_variant_group_training because replacement sampling "
            "would destroy one-step-per-case semantics."
        )
    visibility = resolve_branch_visibility_sampling(config)
    if visibility.enabled:
        assert visibility.metadata_json is not None
        load_stage4_1_visibility_metadata(
            visibility.metadata_json,
            num_branches=resolve_parametric_num_branches(config),
        )
    return True

def make_loader(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    train: bool,
    num_views: int | None = None,
) -> DataLoader:
    if not items:
        raise ValueError("Cannot create a DataLoader for an empty split")
    if train:
        min_views = config.get("min_train_views", config.get("max_views"))
        max_views = config.get("max_train_views", config.get("max_views"))
    else:
        min_views = max_views = num_views
    _validate_variable_view_context(
        config,
        min_views=None if min_views is None else int(min_views),
        max_views=None if max_views is None else int(max_views),
    )
    grouped_variants = resolve_branch_variant_group_training(config)
    dataset_kwargs = {
        "min_views": None if min_views is None else int(min_views),
        "max_views": None if max_views is None else int(max_views),
        "random_view_order": bool(
            config.get(
                "train_random_view_order" if train else "val_random_view_order",
                False,
            )
        ),
        "view_count_weights": (
            config.get("train_view_count_weights") if train else None
        ),
    }
    dataset: ParametricFeatureDataset | ParametricBranchVariantGroupDataset
    if grouped_variants:
        validate_branch_variant_training_config(config)
        dataset = ParametricBranchVariantGroupDataset(
            items,
            config,
            train=train,
            **dataset_kwargs,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=train,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
    dataset = ParametricFeatureDataset(items, **dataset_kwargs)
    batch_size = int(config.get("batch_size", 4) if train else config.get("val_batch_size", config.get("batch_size", 4)))
    sampler = None
    min_steps = config.get("min_steps_per_epoch") if train else None
    if min_steps is not None and int(min_steps) > math.ceil(len(dataset) / batch_size):
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(min_steps) * batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_parametric_batches,
    )

def validation_view_counts(config: dict[str, Any]) -> list[int | None]:
    raw = config.get("val_num_views")
    if raw is None:
        lower = config.get("min_train_views")
        upper = config.get("max_train_views")
        if lower is not None and upper is not None:
            return list(range(int(lower), int(upper) + 1))
        fallback = config.get("max_views")
        return [None if fallback is None else int(fallback)]
    if isinstance(raw, int):
        return [int(raw)]
    values = [int(value) for value in raw]
    if not values or any(value < 1 for value in values):
        raise ValueError("val_num_views must contain at least one positive integer")
    return list(dict.fromkeys(values))

def _mean_validation_metrics(
    metric_sets: list[dict[str, float]],
) -> dict[str, float]:
    """Average each finite validation metric equally across view counts."""
    keys = (
        set().union(*(metrics.keys() for metrics in metric_sets))
        if metric_sets
        else set()
    )
    averaged: dict[str, float] = {}
    for key in sorted(keys):
        values = [
            float(metrics[key])
            for metrics in metric_sets
            if key in metrics and math.isfinite(float(metrics[key]))
        ]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged

def validate_target_scale(items: list[dict[str, Any]], config: dict[str, Any], split: str) -> None:
    coordinate_values: list[np.ndarray] = []
    radius_values: list[np.ndarray] = []
    for item in items:
        vessel = np.asarray(item["target_raw_vessel_mm"], dtype=np.float32)
        valid = (
            np.asarray(item["target_branch_exist"])[:, None] > 0.5
        ) & np.asarray(item["target_point_valid_mask"], dtype=bool)
        if valid.any():
            coordinate_values.append(np.abs(vessel[..., :3][valid]).reshape(-1))
            radius_values.append(vessel[..., 3][valid].reshape(-1))
    if not coordinate_values:
        raise ValueError(f"The {split} split contains no valid parametric vessel points")
    coordinate_p99 = float(np.percentile(np.concatenate(coordinate_values), 99))
    radius_p99 = float(np.percentile(np.concatenate(radius_values), 99))
    coordinate_limit = float(config.get("max_expected_coordinate_magnitude_mm", 1000.0))
    radius_limit = float(config.get("max_expected_radius_mm", 20.0))
    print(
        f"{split} target scale: |xyz| p99={coordinate_p99:.3f} mm, "
        f"radius p99={radius_p99:.3f} mm"
    )
    if coordinate_p99 > coordinate_limit or radius_p99 > radius_limit:
        raise ValueError(
            f"Implausible {split} target scale: |xyz| p99={coordinate_p99:.3f} mm, "
            f"radius p99={radius_p99:.3f} mm. Check input_scale_to_mm and the units "
            "used to generate the parametric transform before training."
        )

def _accumulate_geometry(
    sums: dict[str, Any],
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
    defer_synchronization: bool = False,
) -> int | tuple[tuple[str, ...], torch.Tensor]:
    """Accumulate geometry, optionally deferring the device-to-host sync."""
    prediction = output["decoded_vessel_mm"].detach()
    xyz_target = batch["target_raw_vessel_mm"].to(prediction.device)
    radius_target = batch[
        "target_raw_vessel_mm"
        if radius_prediction_mode_from_config(config) == "raw"
        else "target_reconstructed_vessel_mm"
    ].to(prediction.device)
    branch_exists = batch["target_branch_exist"].to(prediction.device) > 0.5
    point_valid = batch["target_point_valid_mask"].to(prediction.device, dtype=torch.bool)
    mask = branch_exists.unsqueeze(-1) & point_valid
    xyz_error_all = torch.linalg.vector_norm(
        prediction[..., :3] - xyz_target[..., :3], dim=-1
    )
    radius_error_all = torch.abs(
        prediction[..., 3] - radius_target[..., 3]
    )
    branch_indices = torch.arange(
        mask.shape[1], device=mask.device
    ).view(1, -1, 1)
    fixed_main_count = min(
        int(mask.shape[1]),
        required_branch_count(resolve_artery_type(config)),
    )
    group_masks = {
        "main_branch": mask & (branch_indices < fixed_main_count),
        "side_branch": mask & (branch_indices >= fixed_main_count),
    }
    terminal_masks = ordered_terminal_masks(mask, region_fraction=0.05)
    segment_valid = mask[..., :-1] & mask[..., 1:]
    predicted_segment_length = torch.linalg.vector_norm(
        prediction[..., 1:, :3] - prediction[..., :-1, :3], dim=-1
    )
    target_segment_length = torch.linalg.vector_norm(
        xyz_target[..., 1:, :3] - xyz_target[..., :-1, :3], dim=-1
    )
    predicted_length = torch.where(
        segment_valid,
        predicted_segment_length,
        torch.zeros_like(predicted_segment_length),
    ).sum(dim=-1)
    target_length = torch.where(
        segment_valid,
        target_segment_length,
        torch.zeros_like(target_segment_length),
    ).sum(dim=-1)
    valid_length = (
        segment_valid.any(dim=-1)
        & torch.isfinite(predicted_length)
        & torch.isfinite(target_length)
        & (target_length > 1e-6)
    )
    signed_length_error = predicted_length - target_length
    absolute_length_error = signed_length_error.abs()
    relative_length_error = absolute_length_error / target_length.clamp_min(
        1e-6
    )
    zero_length = torch.zeros_like(predicted_length)
    signed_length_error = torch.where(
        valid_length, signed_length_error, zero_length
    )
    absolute_length_error = torch.where(
        valid_length, absolute_length_error, zero_length
    )
    relative_length_error = torch.where(
        valid_length, relative_length_error, zero_length
    )

    predicted_exists = output["branch_exist_probs"].detach() >= 0.5
    branch_supervision = batch.get(
        "target_branch_existence_supervision_mask"
    )
    if branch_supervision is None:
        supervised_branches = torch.ones_like(branch_exists, dtype=torch.bool)
    else:
        supervised_branches = branch_supervision.to(
            device=prediction.device,
            dtype=torch.bool,
        )
        if supervised_branches.shape != branch_exists.shape:
            raise ValueError(
                "target_branch_existence_supervision_mask must match "
                f"target_branch_exist, got {tuple(supervised_branches.shape)} "
                f"and {tuple(branch_exists.shape)}."
            )
    metric_tensors = {
        "point_count": mask.sum(),
        "centerline_error_sum_mm": xyz_error_all[mask].sum(),
        "radius_error_sum_mm": radius_error_all[mask].sum(),
        "centerline_start_error_sum_mm": xyz_error_all[
            terminal_masks["first"]
        ].sum(),
        "centerline_start_count": terminal_masks["first"].sum(),
        "centerline_end_error_sum_mm": xyz_error_all[
            terminal_masks["last"]
        ].sum(),
        "centerline_end_count": terminal_masks["last"].sum(),
        "arc_length_abs_error_sum_mm": absolute_length_error.sum(),
        "arc_length_signed_error_sum_mm": signed_length_error.sum(),
        "arc_length_rel_error_sum": relative_length_error.sum(),
        "predicted_arc_length_sum_mm": torch.where(
            valid_length, predicted_length, zero_length
        ).sum(),
        "target_arc_length_sum_mm": torch.where(
            valid_length, target_length, zero_length
        ).sum(),
        "arc_length_count": valid_length.sum(),
        "branch_correct": (
            (predicted_exists == branch_exists) & supervised_branches
        ).sum(),
        "branch_count": supervised_branches.sum(),
    }
    for prefix, group_mask in group_masks.items():
        metric_tensors[f"{prefix}_point_count"] = group_mask.sum()
        metric_tensors[f"{prefix}_centerline_error_sum_mm"] = (
            xyz_error_all[group_mask].sum()
        )
        metric_tensors[f"{prefix}_radius_error_sum_mm"] = (
            radius_error_all[group_mask].sum()
        )
    metric_names = tuple(metric_tensors)
    detached_metrics = {
        name: metric_tensors[name].detach().to(dtype=prediction.dtype)
        for name in metric_names
    }
    if defer_synchronization:
        return metric_names, torch.stack(
            [detached_metrics[name] for name in metric_names]
        )

    metric_values = torch.stack(
        [detached_metrics[name] for name in metric_names]
    ).cpu().tolist()
    for name, value in zip(metric_names, metric_values):
        if name != "point_count":
            sums[name] = sums.get(name, 0.0) + float(value)
    return int(metric_values[metric_names.index("point_count")])

def _synchronized_named_scalars(
    names: tuple[str, ...] | None,
    values: torch.Tensor | None,
) -> dict[str, float]:
    """Copy a named scalar vector to CPU with one device synchronization."""

    if names is None or values is None:
        return {}
    return {
        name: float(value)
        for name, value in zip(names, values.cpu().tolist())
    }

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    projector: DifferentiableVesselProjector | None = None,
    epoch: int | None = None,
    online_vggt: OnlineVGGTFeatureProvider | None = None,
    training_sampling_schedule: TrainingSamplingSchedule | None = None,
    bspline_refiner_projector: DifferentiableVesselProjector | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if training_sampling_schedule is not None:
        if not training:
            raise ValueError(
                "A training sampling schedule cannot be attached to a "
                "validation/test epoch."
            )
        training_sampling_schedule.start_epoch(
            0 if epoch is None else int(epoch)
        )
    set_dataset_epoch = getattr(loader.dataset, "set_epoch", None)
    if callable(set_dataset_epoch):
        set_dataset_epoch(0 if epoch is None else int(epoch))
    set_parametric_model_training_mode(
        model,
        training=training,
        config=config,
    )
    loss_names: tuple[str, ...] | None = None
    loss_batch_vectors: list[torch.Tensor] = []
    geometry_names: tuple[str, ...] | None = None
    geometry_batch_vectors: list[torch.Tensor] = []
    centerline_parameter_error_batches: list[torch.Tensor] = []
    centerline_mode = centerline_prediction_mode_from_config(config)
    parameter_prefix = (
        "landmark" if centerline_mode == "adaptive_landmarks" else "control_point"
    )
    parameter_index_prefix = "l" if parameter_prefix == "landmark" else "c"
    case_count = point_count = 0
    visibility_group_count = 0
    visibility_required_count = 0
    visibility_covered_count = 0
    visibility_uncovered_count = 0
    visibility_groups_with_uncovered = 0
    visibility_skipped_variant_count = 0
    trainable_parameters = (
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        if training
        else []
    )
    grad_clip_norm = float(
        config.get(
            "grad_clip_norm",
            config.get("gradient_clip_norm", 1.0),
        )
    )
    for sampling_step, batch in enumerate(loader):
        if training_sampling_schedule is not None:
            training_sampling_schedule.observe_batch(batch, sampling_step)
        if bool(batch.get("branch_visibility_sampling_active", False)):
            required_visibility = tuple(
                int(value)
                for value in batch.get(
                    "branch_visibility_required_branch_indices", ()
                )
            )
            covered_visibility = tuple(
                int(value)
                for value in batch.get(
                    "branch_visibility_covered_branch_indices", ()
                )
            )
            uncovered_visibility = tuple(
                int(value)
                for value in batch.get(
                    "branch_visibility_uncovered_branch_indices", ()
                )
            )
            if set(covered_visibility).intersection(uncovered_visibility) or set(
                covered_visibility
            ).union(uncovered_visibility) != set(required_visibility):
                raise RuntimeError(
                    "Stage-4.1 batch visibility diagnostics are inconsistent: "
                    "covered and uncovered branches must partition required "
                    "branches."
                )
            visibility_group_count += 1
            visibility_required_count += len(required_visibility)
            visibility_covered_count += len(covered_visibility)
            visibility_uncovered_count += len(uncovered_visibility)
            visibility_groups_with_uncovered += int(bool(uncovered_visibility))
            visibility_skipped_variant_count += len(
                batch.get("branch_visibility_policy_skipped_variants", ())
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        image_features = resolve_batch_image_features(
            batch, device, online_vggt
        )
        with (torch.enable_grad() if training else torch.inference_mode()):
            output = model(
                views=batch["view_features"].to(device, non_blocking=True),
                view_mask=batch["view_mask"].to(device, non_blocking=True),
                image_features=image_features,
                images=(
                    None
                    if batch.get("images") is None
                    else batch["images"].to(device, non_blocking=True)
                ),
                projection_center_offset=(
                    None
                    if batch.get("projection_center_offset") is None
                    else batch["projection_center_offset"].to(
                        device, non_blocking=True
                    )
                ),
                refiner_branch_mask=(
                    refiner_branch_mask_for_batch(batch, config, device)
                    if training
                    else None
                ),
            )
            losses = compute_parametric_loss(
                output,
                batch,
                config,
                projector=projector,
                bspline_refiner_projector=bspline_refiner_projector,
                epoch=epoch,
            )
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    grad_clip_norm,
                )
                optimizer.step()
        batch_size = int(batch["view_features"].shape[0])
        group_normalized = bool(batch.get("branch_variant_group_mode", False))
        metric_weight = 1 if group_normalized else batch_size
        case_count += metric_weight
        deferred_geometry = _accumulate_geometry(
            {}, output, batch, config, True
        )
        if isinstance(deferred_geometry, tuple):
            batch_geometry_names, batch_geometry_vector = deferred_geometry
            if geometry_names is None:
                geometry_names = batch_geometry_names
            elif batch_geometry_names != geometry_names:
                raise RuntimeError(
                    "Geometry metric names changed within one epoch."
                )
            geometry_batch_vectors.append(batch_geometry_vector)
        else:
            # Preserve compatibility with tests or callers that replace the
            # accumulator with the legacy integer-returning interface.
            point_count += deferred_geometry
        predicted_parameters = output["centerline_parameters_mm"].detach()
        target_parameters = batch["target_centerline_parameters_mm"].to(
            predicted_parameters.device
        )
        parameter_errors = torch.linalg.vector_norm(
            predicted_parameters - target_parameters, dim=-1
        )
        parameter_branch_exists = (
            batch["target_branch_exist"].to(predicted_parameters.device) > 0.5
        )
        parameter_branch_exists = parameter_branch_exists.clone()
        parameter_branch_exists[:, 0] = True
        parameter_errors = parameter_errors.masked_fill(
            ~parameter_branch_exists.unsqueeze(-1), float("nan")
        )
        centerline_parameter_error_batches.append(parameter_errors)
        scalar_losses = tuple(
            (key, value.detach().reshape(()))
            for key, value in losses.items()
            if value.numel() == 1
        )
        batch_loss_names = tuple(key for key, _value in scalar_losses)
        if loss_names is None:
            loss_names = batch_loss_names
        elif batch_loss_names != loss_names:
            raise RuntimeError("Scalar loss names changed within one epoch.")
        batch_loss_vector = torch.stack(
            [value for _key, value in scalar_losses]
        )
        loss_batch_vectors.append(batch_loss_vector * metric_weight)
    if training_sampling_schedule is not None:
        training_sampling_schedule.finish_epoch()
    loss_sum_vector = (
        torch.stack(loss_batch_vectors, dim=0)
        .to(dtype=torch.float64)
        .sum(dim=0)
        if loss_batch_vectors
        else None
    )
    synchronized_loss_sums = _synchronized_named_scalars(
        loss_names,
        loss_sum_vector,
    )
    geometry_sum_vector = (
        torch.stack(geometry_batch_vectors, dim=0)
        .to(dtype=torch.float64)
        .sum(dim=0)
        if geometry_batch_vectors
        else None
    )
    geometry_sums = _synchronized_named_scalars(
        geometry_names,
        geometry_sum_vector,
    )
    if "point_count" in geometry_sums:
        synchronized_point_count = geometry_sums.pop("point_count")
    else:
        synchronized_point_count = float(point_count)
    metrics = {
        key: value / max(case_count, 1)
        for key, value in synchronized_loss_sums.items()
    }
    metrics["centerline_mae_mm"] = geometry_sums.get(
        "centerline_error_sum_mm", 0.0
    ) / max(synchronized_point_count, 1)
    metrics["centerline_start_mae_mm"] = geometry_sums.get(
        "centerline_start_error_sum_mm", 0.0
    ) / max(geometry_sums.get("centerline_start_count", 0.0), 1.0)
    metrics["centerline_end_mae_mm"] = geometry_sums.get(
        "centerline_end_error_sum_mm", 0.0
    ) / max(geometry_sums.get("centerline_end_count", 0.0), 1.0)
    arc_length_count = max(geometry_sums.get("arc_length_count", 0.0), 1.0)
    metrics["arc_length_abs_error_mm"] = geometry_sums.get(
        "arc_length_abs_error_sum_mm", 0.0
    ) / arc_length_count
    metrics["arc_length_signed_error_mm"] = geometry_sums.get(
        "arc_length_signed_error_sum_mm", 0.0
    ) / arc_length_count
    metrics["arc_length_rel_error"] = geometry_sums.get(
        "arc_length_rel_error_sum", 0.0
    ) / arc_length_count
    metrics["predicted_arc_length_mean_mm"] = geometry_sums.get(
        "predicted_arc_length_sum_mm", 0.0
    ) / arc_length_count
    metrics["target_arc_length_mean_mm"] = geometry_sums.get(
        "target_arc_length_sum_mm", 0.0
    ) / arc_length_count
    metrics["radius_mae_mm"] = geometry_sums.get(
        "radius_error_sum_mm", 0.0
    ) / max(synchronized_point_count, 1)
    for prefix in ("main_branch", "side_branch"):
        group_count = geometry_sums.get(f"{prefix}_point_count", 0.0)
        metrics[f"{prefix}_centerline_mae_mm"] = (
            geometry_sums.get(f"{prefix}_centerline_error_sum_mm", 0.0)
            / group_count
            if group_count > 0.0
            else float("nan")
        )
        metrics[f"{prefix}_radius_mae_mm"] = (
            geometry_sums.get(f"{prefix}_radius_error_sum_mm", 0.0)
            / group_count
            if group_count > 0.0
            else float("nan")
        )
    metrics["branch_exist_accuracy"] = geometry_sums.get("branch_correct", 0.0) / max(geometry_sums.get("branch_count", 0.0), 1.0)
    if visibility_group_count:
        metrics["branch_visibility_required_branch_count"] = float(
            visibility_required_count
        )
        metrics["branch_visibility_covered_branch_count"] = float(
            visibility_covered_count
        )
        metrics["branch_visibility_uncovered_branch_count"] = float(
            visibility_uncovered_count
        )
        metrics["branch_visibility_coverage_fraction"] = float(
            1.0
            if visibility_required_count == 0
            else visibility_covered_count / visibility_required_count
        )
        metrics["branch_visibility_groups_with_uncovered_fraction"] = float(
            visibility_groups_with_uncovered / visibility_group_count
        )
        metrics["branch_visibility_skipped_variant_count"] = float(
            visibility_skipped_variant_count
        )
    if centerline_parameter_error_batches:
        all_parameter_errors = (
            torch.cat(centerline_parameter_error_batches, dim=0).cpu().numpy()
        )
        for branch_index in range(all_parameter_errors.shape[1]):
            for parameter_index in range(all_parameter_errors.shape[2]):
                values = all_parameter_errors[:, branch_index, parameter_index]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                prefix = (
                    f"{parameter_prefix}_b{branch_index:02d}_"
                    f"{parameter_index_prefix}{parameter_index:02d}"
                )
                metrics[f"{prefix}_mae_mm"] = float(values.mean())
                metrics[f"{prefix}_p95_mm"] = float(np.percentile(values, 95))
    return metrics

def _all_losses_objective(
    metrics: dict[str, float], config: dict[str, Any]
) -> float:
    """Return a schedule-independent objective containing every configured loss.

    Replace current error thresholds, scheduled 3D bend-recall/branch-length/
    local-progress contributions, and projection contributions with their final
    configured values so scores remain comparable throughout their schedules.
    """
    total = float(metrics["loss"])
    loss_config = config.get("loss", {})
    if not isinstance(loss_config, dict):
        loss_config = {}
    for component_name, weight_name, default in (
        ("decoded_xyz", "decoded_xyz_loss_weight", 1.0),
        ("decoded_radius", "decoded_radius_loss_weight", 1.0),
    ):
        final_key = f"{component_name}_final_threshold_loss"
        current_key = f"{component_name}_loss"
        if final_key not in metrics:
            continue
        weight = float(
            loss_config.get(
                weight_name,
                config.get(weight_name, default),
            )
        )
        total += weight * (
            float(metrics[final_key]) - float(metrics[current_key])
        )
    decoded_xyz_weight = float(
        loss_config.get(
            "decoded_xyz_loss_weight",
            config.get("decoded_xyz_loss_weight", 1.0),
        )
    )
    auxiliary_threshold_suffix = "_decoded_xyz_final_threshold_loss"
    for final_key in tuple(metrics):
        if not (
            final_key.startswith("bspline_refiner_")
            and final_key.endswith(auxiliary_threshold_suffix)
        ):
            continue
        prefix = final_key[: -len(auxiliary_threshold_suffix)]
        current_key = f"{prefix}_decoded_xyz_loss"
        if current_key not in metrics:
            continue
        auxiliary_weight = float(
            metrics[
                "bspline_refiner_coarse_effective_weight"
                if prefix == "bspline_refiner_coarse"
                else "bspline_refiner_intermediate_effective_weight"
            ]
        )
        total += auxiliary_weight * decoded_xyz_weight * (
            float(metrics[final_key]) - float(metrics[current_key])
        )
    if "bend_recall_3d_weighted_loss" in metrics:
        total -= float(metrics["bend_recall_3d_weighted_loss"])
        total += bend_recall_3d_final_weight(config) * float(
            metrics["bend_recall_3d_loss"]
        )
    if "branch_length_weighted_loss" in metrics:
        total -= float(metrics["branch_length_weighted_loss"])
        total += branch_length_final_weight(config) * float(
            metrics["branch_length_loss"]
        )
    if "decoded_local_progress_weighted_loss" in metrics:
        total -= float(metrics["decoded_local_progress_weighted_loss"])
        total += decoded_local_progress_final_weight(config) * float(
            metrics["decoded_local_progress_loss"]
        )
    # Projection work is intentionally skipped while its schedule factor is
    # zero, so an enabled loss can legitimately have no projection metrics in
    # early epochs. Once either metric is present, however, both are required
    # to replace the scheduled contribution with its final-weight value.
    projection_metric_keys = {
        "projection_2d_loss",
        "projection_2d_unweighted_loss",
    }
    available_projection_metric_keys = projection_metric_keys.intersection(
        metrics
    )
    if available_projection_metric_keys and (
        available_projection_metric_keys != projection_metric_keys
    ):
        missing = sorted(
            projection_metric_keys - available_projection_metric_keys
        )
        raise KeyError(
            "Incomplete 2D projection metrics: missing " + ", ".join(missing)
        )
    if (
        bool(config.get("enable_projection_2d_loss", False))
        and available_projection_metric_keys == projection_metric_keys
    ):
        scheduled_projection = float(metrics["projection_2d_loss"])
        unweighted_projection = float(
            metrics["projection_2d_unweighted_loss"]
        )
        final_factor = (
            float(config.get("proj_loss_schedule_end_factor", 1.0))
            if bool(config.get("proj_loss_schedule_enabled", False))
            else 1.0
        )
        final_projection_weight = (
            float(config.get("proj_loss_weight", 0.2)) * final_factor
        )
        total = (
            total
            - scheduled_projection
            + final_projection_weight * unweighted_projection
        )
        auxiliary_projection_suffix = "_projection_2d_unweighted_loss"
        for unweighted_key in tuple(metrics):
            if not (
                unweighted_key.startswith("bspline_refiner_")
                and unweighted_key.endswith(auxiliary_projection_suffix)
            ):
                continue
            prefix = unweighted_key[: -len(auxiliary_projection_suffix)]
            scheduled_key = f"{prefix}_projection_2d_loss"
            if scheduled_key not in metrics:
                continue
            auxiliary_weight = float(
                metrics[
                    "bspline_refiner_coarse_effective_weight"
                    if prefix == "bspline_refiner_coarse"
                    else "bspline_refiner_intermediate_effective_weight"
                ]
            )
            total += auxiliary_weight * (
                final_projection_weight * float(metrics[unweighted_key])
                - float(metrics[scheduled_key])
            )
    return total

def _write_history(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    render_loss_plot: bool,
) -> None:
    keys = sorted({key for row in rows for key in row})
    with open(output_dir / "history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_dir / "training_monitor" / "history.json", rows)
    if render_loss_plot:
        plot_history(
            rows,
            output_dir / "training_monitor" / "loss_plot_latest.png",
        )

def _format_epoch_loss_metrics(metrics: dict[str, float]) -> str:
    """Format the compact loss summary shared with SRC training logs."""

    fields = (
        ("loss", "loss"),
        ("radius_loss", "decoded_radius_loss"),
        ("xyz_loss", "decoded_xyz_loss"),
        (
            "side_exist_refine_loss",
            "bspline_refiner_branch_existence_loss",
        ),
    )
    return ", ".join(
        f"{label}={float(metrics[key]):.4f}"
        for label, key in fields
        if key in metrics
    )

def _monitor_items(
    train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], test_items: list[dict[str, Any]], config: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    counts = config.get("monitor_num_cases_per_split", {"train": 2, "val": 2, "test": 1})
    if isinstance(counts, int):
        counts = {"train": counts, "val": counts, "test": counts}
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[int] = set()
    for name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        case_groups = _monitor_case_groups(items)
        for _, group_items in case_groups[: int(counts.get(name, 0))]:
            for item in group_items:
                selected.append((name, item))
                selected_ids.add(id(item))
    requested = _spy_case_keys(config.get("spy_monitor_cases"))
    found: set[str] = set()
    for name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        for _, group_items in _monitor_case_groups(items):
            matches = _item_case_keys(group_items[0]) & requested
            if not matches:
                continue
            found.update(matches)
            for item in group_items:
                if id(item) not in selected_ids:
                    selected.append((name, item))
                    selected_ids.add(id(item))
    missing = sorted(requested - found)
    if missing:
        print(f"[warn] spy_monitor_cases not found in loaded splits: {missing}")
    return selected

def _normalized_case_key(value: Any) -> str:
    path = Path(str(value))
    if path.parent.name.isdigit():
        return str(int(path.parent.name))
    stem = path.stem.strip().lower()
    groups = re.findall(r"\d+", stem)
    return str(int(groups[-1])) if groups else stem

def _spy_case_keys(raw: Any) -> set[str]:
    if raw is None:
        return set()
    values = [part.strip() for part in raw.split(",")] if isinstance(raw, str) else list(raw)
    return {_normalized_case_key(value) for value in values if str(value).strip()}

def _item_case_keys(item: dict[str, Any]) -> set[str]:
    # In grouped Stage-4/5 data, case_name is a variant name such as
    # ``prefix_04``. Treating its numeric suffix as a physical case ID makes
    # every prefix_04 file match spy case rca_0004. Prefer the explicit case ID
    # and numeric parent directory; use case_name only for legacy flat layouts.
    path = Path(str(item.get("path", "")))
    keys = {
        _normalized_case_key(item.get("case_id", "")),
        _normalized_case_key(path),
    }
    if not path.parent.name.isdigit():
        keys.add(_normalized_case_key(item.get("case_name", "")))
    return {key for key in keys if key}

def _monitor_case_groups(
    items: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group every branch variant under its physical monitor case."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        keys = _item_case_keys(item)
        case_id = _normalized_case_key(item.get("case_id", ""))
        if not case_id:
            path = Path(str(item.get("path", "")))
            case_id = _normalized_case_key(path)
        if not case_id and keys:
            case_id = sorted(keys)[0]
        grouped.setdefault(case_id, []).append(item)

    def variant_key(item: dict[str, Any]) -> tuple[int, str]:
        branch_mask = item.get("training_branch_mask")
        if branch_mask is None:
            branch_mask = item.get("target_branch_exist")
        active_count = (
            int(np.count_nonzero(np.asarray(branch_mask)))
            if branch_mask is not None
            else 0
        )
        variant = str(
            item.get("branch_subset_variant", item.get("case_name", ""))
        )
        return active_count, variant

    return [
        (case_id, sorted(group, key=variant_key))
        for case_id, group in grouped.items()
    ]

def include_spy_files(
    selected_files: list[Path], all_files: list[Path], config: dict[str, Any]
) -> list[Path]:
    requested = _spy_case_keys(config.get("spy_monitor_cases"))
    if not requested:
        return selected_files
    output = list(selected_files)
    selected_paths = {path.resolve() for path in output}
    found: set[str] = set()
    for path in all_files:
        key = _normalized_case_key(path)
        if key in requested:
            found.add(key)
            if path.resolve() not in selected_paths:
                output.append(path)
                selected_paths.add(path.resolve())
    missing = sorted(requested - found)
    if missing:
        print(f"[warn] spy_monitor_cases not found in feature dataset: {missing}")
    return output

def _explicit_branch_variant_group_split(
    split_json_path: str | Path,
    *,
    groups: list[BranchVariantFileGroup],
    config: dict[str, Any],
) -> SplitFiles:
    """Map any case-level split record onto all variants without leakage."""

    split_path = Path(split_json_path).expanduser().resolve()
    source = load_split_record(split_path)
    assignments: dict[str, str] = {}
    for split_name, paths in (
        ("train", source.train),
        ("val", source.val),
        ("test", source.test),
    ):
        for path in paths:
            identifier = case_identifier(path)
            previous = assignments.get(identifier)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Explicit split {split_path} assigns physical case "
                    f"{identifier} to both {previous!r} and {split_name!r}."
                )
            assignments[identifier] = split_name
    missing_policy = str(
        config.get("explicit_split_missing_case_policy", "error")
    ).strip().lower()
    if missing_policy not in {"error", "drop"}:
        raise ValueError(
            "explicit_split_missing_case_policy must be 'error' or 'drop'."
        )
    missing = sorted(
        {group.case_id for group in groups}.difference(assignments), key=int
    )
    if missing and missing_policy == "error":
        raise ValueError(
            f"Explicit split {split_path} has no assignment for branch-variant "
            f"physical cases {missing}."
        )
    representatives: dict[str, list[Path]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for group in groups:
        split_name = assignments.get(group.case_id)
        if split_name is not None:
            representatives[split_name].append(group.paths[0])
    return SplitFiles(**representatives)

def _expand_branch_variant_split(
    representative_split: SplitFiles,
    groups: list[BranchVariantFileGroup],
) -> SplitFiles:
    """Expand one representative path per case to every member of its group."""

    group_by_case = {group.case_id: group for group in groups}

    def expand(paths: list[Path]) -> list[Path]:
        output: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            identifier = case_identifier(path)
            if identifier in seen:
                raise ValueError(
                    f"Physical case {identifier} occurs more than once in a "
                    "representative branch-variant split."
                )
            seen.add(identifier)
            output.extend(group_by_case[identifier].paths)
        return output

    expanded = SplitFiles(
        train=expand(representative_split.train),
        val=expand(representative_split.val),
        test=expand(representative_split.test),
    )
    memberships: dict[str, str] = {}
    for split_name, paths in (
        ("train", expanded.train),
        ("val", expanded.val),
        ("test", expanded.test),
    ):
        for path in paths:
            identifier = case_identifier(path)
            previous = memberships.get(identifier)
            if previous is not None and previous != split_name:
                raise RuntimeError(
                    f"Branch-variant split leaked physical case {identifier} "
                    f"between {previous!r} and {split_name!r}."
                )
            memberships[identifier] = split_name
    return expanded

def _coarse_refiner_monitor_output(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Return a normal model-output view containing only the coarse geometry."""
    if "coarse_decoded_vessel_mm" not in output:
        return None
    refiner_prefixes = (
        "coarse_",
        "refinement_stage_",
        "bspline_refinement_",
    )
    coarse_output = {
        key: value
        for key, value in output.items()
        if not key.startswith(refiner_prefixes)
    }
    coarse_parameters = output["coarse_centerline_parameters_mm"]
    coarse_output["centerline_parameters_mm"] = coarse_parameters
    coarse_output["centerline_control_points_mm"] = coarse_parameters
    coarse_output["decoded_vessel_mm"] = output["coarse_decoded_vessel_mm"]
    coarse_output["side_centerline_offsets_mm"] = output[
        "coarse_side_centerline_offsets_mm"
    ]
    coarse_output["side_branch_relative_code_mm"] = output[
        "coarse_side_branch_relative_code_mm"
    ]
    if "coarse_branch_exist_logits" in output:
        coarse_output["branch_exist_logits"] = output[
            "coarse_branch_exist_logits"
        ]
        coarse_output["branch_exist_probs"] = output[
            "coarse_branch_exist_probs"
        ]
    return coarse_output

_TRAINING_MONITOR_ARTIFACT_PROFILES = frozenset({"summary", "static", "full"})

def _resolve_training_monitor_artifact_profile(
    config: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    profile = str(config.get(key, default)).strip().lower()
    if profile not in _TRAINING_MONITOR_ARTIFACT_PROFILES:
        raise ValueError(
            f"{key} must be one of "
            f"{sorted(_TRAINING_MONITOR_ARTIFACT_PROFILES)}, got {profile!r}."
        )
    return profile

def save_monitors(
    model: torch.nn.Module,
    monitor_items: list[tuple[str, dict[str, Any]]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    epoch: int,
    online_vggt: OnlineVGGTFeatureProvider | None = None,
) -> None:
    raw_views = config.get("monitor_num_views", validation_view_counts(config))
    view_counts = [None] if raw_views is None else [raw_views] if isinstance(raw_views, int) else list(raw_views)
    monitor_loader_config = config
    if resolve_branch_variant_group_training(config):
        # ``monitor_items`` contains individual variants, whereas the grouped
        # loader requires the complete physical-case variant set (and a
        # Stage-4.1 record containing that same set).  Materialise each chosen
        # monitor as an ordinary one-item batch; grouping remains unchanged for
        # the actual train/validation/test loaders.
        monitor_loader_config = dict(config)
        monitor_loader_config["branch_variant_group_training"] = False
        visibility = config.get("branch_visibility_sampling")
        if isinstance(visibility, Mapping):
            monitor_visibility = dict(visibility)
            monitor_visibility["enabled"] = False
            monitor_loader_config["branch_visibility_sampling"] = (
                monitor_visibility
            )
    model.eval()
    artifact_profile = _resolve_training_monitor_artifact_profile(
        config,
        "monitor_artifact_profile",
        "summary",
    )
    coarse_artifact_profile = _resolve_training_monitor_artifact_profile(
        config,
        "monitor_coarse_artifact_profile",
        artifact_profile,
    )
    started_at = time.perf_counter()
    artifact_bundle_count = 0
    with torch.inference_mode():
        for source, item in monitor_items:
            ensure_item_images(item)
            for num_views in view_counts:
                loader = make_loader(
                    [item],
                    monitor_loader_config,
                    train=False,
                    num_views=None if num_views is None else int(num_views),
                )
                batch = next(iter(loader))
                image_features = resolve_batch_image_features(
                    batch, device, online_vggt
                )
                output = model(
                    views=batch["view_features"].to(device),
                    view_mask=batch["view_mask"].to(device),
                    image_features=image_features,
                    images=(
                        None
                        if batch.get("images") is None
                        else batch["images"].to(device)
                    ),
                    projection_center_offset=(
                        None
                        if batch.get("projection_center_offset") is None
                        else batch["projection_center_offset"].to(device)
                    ),
                )
                label = "all_views" if num_views is None else f"k{int(num_views)}"
                case_monitor_path = (
                    output_dir
                    / "training_monitor"
                    / f"epoch_{epoch:04d}"
                    / source
                    / f"case_{item['case_id']}"
                )
                if resolve_branch_variant_group_training(config):
                    variant = str(
                        item.get(
                            "branch_subset_variant",
                            item.get("case_name", "variant"),
                        )
                    ).strip()
                    if not variant or Path(variant).name != variant:
                        raise ValueError(
                            f"Invalid branch monitor variant name {variant!r} "
                            f"for case {item['case_id']}."
                        )
                    case_monitor_path = case_monitor_path / variant
                monitor_path = case_monitor_path / label
                save_vessel_monitor(
                    output=output,
                    batch=batch,
                    item_index=0,
                    path=monitor_path,
                    epoch=epoch,
                    view_label=label,
                    config=config,
                    artifact_profile=artifact_profile,
                )
                artifact_bundle_count += 1
                if bool(
                    config.get(
                        "monitor_bspline_refiner_coarse_prediction", True
                    )
                ):
                    coarse_output = _coarse_refiner_monitor_output(output)
                    if coarse_output is not None:
                        save_vessel_monitor(
                            output=coarse_output,
                            batch=batch,
                            item_index=0,
                            path=monitor_path / "coarse_prediction",
                            epoch=epoch,
                            view_label=f"{label}, coarse prediction",
                            config=config,
                            artifact_profile=coarse_artifact_profile,
                        )
                        artifact_bundle_count += 1
    elapsed_seconds = time.perf_counter() - started_at
    print(
        "[Monitor] saved "
        f"{artifact_bundle_count} artifact bundle(s) in "
        f"{elapsed_seconds:.1f}s (profile={artifact_profile!r}, "
        f"coarse_profile={coarse_artifact_profile!r})",
        flush=True,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the parametric multi-view vessel predictor.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output_dir",
        help=(
            "Output root. The resolved experiment is created below this "
            "directory."
        ),
    )
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume from a checkpoint path, or latest.pt when no path is given.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
        for resolved_key in ("output_root_dir", "experiment_dir", "out_dir"):
            config.pop(resolved_key, None)
    backbone = normalize_feature_backbone(
        config.get("feature_backbone", "vggt")
    )
    output_dir = resolve_train_output_config(config, backbone)
    config["feature_backbone"] = backbone
    print(f"[Experiment] {output_dir.name}")
    configure_radius_mode_losses(config, announce=True)
    configure_only_centerline_losses(
        config,
        pipeline="parametric",
        announce=True,
    )
    epochs = int(config.get("num_epochs", config.get("epochs", 100)))
    if epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {epochs}.")
    config["num_epochs"] = epochs
    monitor_artifact_profile = _resolve_training_monitor_artifact_profile(
        config,
        "monitor_artifact_profile",
        "summary",
    )
    config["monitor_artifact_profile"] = monitor_artifact_profile
    config["monitor_coarse_artifact_profile"] = (
        _resolve_training_monitor_artifact_profile(
            config,
            "monitor_coarse_artifact_profile",
            monitor_artifact_profile,
        )
    )
    threshold_names = configured_error_threshold_names(config)
    loss_config = config.get("loss", {})
    if not isinstance(loss_config, dict):
        raise ValueError("loss must be a JSON object")
    hard_radius_weighting = loss_config.get(
        "decoded_radius_hard_point_weighting",
        config.get("decoded_radius_hard_point_weighting", {}),
    )
    if hard_radius_weighting is None:
        hard_radius_weighting = {}
    if not isinstance(hard_radius_weighting, dict):
        raise ValueError(
            "decoded_radius_hard_point_weighting must be a JSON object."
        )
    if bool(hard_radius_weighting.get("enabled", False)) and (
        "decoded_radius_loss" in threshold_names
        or "radius_loss" in threshold_names
    ):
        raise ValueError(
            "decoded_radius_hard_point_weighting.enabled and legacy radius "
            "error_thresholding are mutually exclusive."
        )
    supported_threshold_names = {
        "decoded_xyz_loss",
        "decoded_radius_loss",
        "xyz_loss",
        "radius_loss",
    }
    unsupported_threshold_names = sorted(
        set(threshold_names) - supported_threshold_names
    )
    if unsupported_threshold_names:
        raise ValueError(
            "Parametric error_thresholding supports per-point "
            "'decoded_xyz_loss' and 'decoded_radius_loss' (with 'xyz_loss' and "
            "'radius_loss' aliases); unsupported enabled entries: "
            f"{unsupported_threshold_names}."
        )
    for canonical_name, alias_name in (
        ("decoded_xyz_loss", "xyz_loss"),
        ("decoded_radius_loss", "radius_loss"),
    ):
        if canonical_name in threshold_names and alias_name in threshold_names:
            raise ValueError(
                "Configure only one error-threshold name per parametric "
                f"component, not both {canonical_name!r} and {alias_name!r}."
            )
    threshold_schedule_summary: dict[str, dict[str, float]] = {}
    for component_name in threshold_names:
        first_settings = resolve_error_threshold_settings(
            config,
            component_name,
            epoch=1,
        )
        final_settings = resolve_error_threshold_settings(
            config,
            component_name,
            epoch=epochs,
        )
        threshold_schedule_summary[component_name] = {
            "epoch_1_mm": first_settings.effective_threshold,
            f"epoch_{epochs}_mm": final_settings.effective_threshold,
        }
    if threshold_schedule_summary:
        config["error_threshold_schedule_epoch_numbering"] = "one_based"
        config["error_threshold_schedule_applies_to"] = (
            "training_and_validation"
        )
        config["effective_error_threshold_schedule_mm"] = (
            threshold_schedule_summary
        )
        print(
            "[Error threshold schedule] physical per-point margins "
            f"{threshold_schedule_summary}"
        )
    centerline_mode = centerline_prediction_mode_from_config(config)
    configured_num_branches = resolve_parametric_num_branches(config)
    configured_target_num_branches = resolve_parametric_target_num_branches(
        config,
        model_num_branches=configured_num_branches,
    )
    config["resolved_model_num_branch_tokens"] = configured_num_branches
    config["resolved_model_num_branch_queries"] = configured_num_branches
    config["resolved_target_num_branches"] = configured_target_num_branches
    if config.get("model") is None:
        config["model"] = {}
    config["model"]["num_branches"] = configured_num_branches
    merged_model_config = dict(config)
    merged_model_config.update(dict(config.get("model", {}) or {}))
    merged_model_config["num_branches"] = configured_num_branches
    artery_type, visualization_existence_source = (
        validate_visualization_branch_existence_config(
            config,
            num_branches=int(merged_model_config.get("num_branches", 7)),
        )
    )
    minimum_target_branches = required_branch_count(artery_type)
    if configured_target_num_branches < minimum_target_branches:
        raise ValueError(
            f"{artery_type} training requires at least "
            f"{minimum_target_branches} supervised target branches, but "
            f"num_branches={configured_target_num_branches}."
        )
    print(
        "Branch contract: "
        f"{configured_num_branches} model branch queries/output slots; "
        f"first {configured_target_num_branches} geometry target slots; "
        f"{configured_num_branches - configured_target_num_branches} later "
        "slots supervised as absent for branch existence"
    )
    print(
        "Paired visualization branch existence: "
        f"{visualization_existence_source} ({artery_type})"
    )
    refiner_existence_source = refiner_branch_existence_source(config)
    print(f"Refiner branch existence during training: {refiner_existence_source}")
    centerline_count = int(
        merged_model_config.get(
            "num_landmarks",
            merged_model_config.get("num_control_points", 20),
        )
        if centerline_mode == "adaptive_landmarks"
        else merged_model_config.get("num_control_points", 20)
    )
    print(
        f"Centreline prediction mode: {centerline_mode}; "
        f"predicting {centerline_count} "
        f"{'on-curve landmarks' if centerline_mode == 'adaptive_landmarks' else 'B-spline control points'}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    online_vggt_enabled = uses_online_vggt_features(config)
    lazy_load_image_features = resolve_lazy_load_image_features(
        config, default=False
    )
    config["lazy_load_image_features"] = lazy_load_image_features
    if online_vggt_enabled and lazy_load_image_features:
        raise ValueError(
            "lazy_load_image_features=true applies only to cached NPZ "
            "features and cannot be combined with online frozen VGGT."
        )
    input_dataset_dir = resolve_model_input_dataset_dir(config, backbone)
    if online_vggt_enabled:
        raw_image_dataset_dir = resolve_raw_image_dataset_dir(config)
        config["raw_image_dataset_dir"] = str(raw_image_dataset_dir)
        config["resolved_raw_image_dataset_dir"] = str(raw_image_dataset_dir)
        config["resolved_feature_dataset_dir"] = None
        config["feature_dataset_root"] = None
        config["model_input_mode"] = "online_frozen_vggt"
        expected_context = config.get("expected_vggt_context_mode")
        if expected_context is not None and str(expected_context).strip().lower() not in {
            "",
            "all_views",
        }:
            raise ValueError(
                "feature_dataset_dir=null requires "
                "expected_vggt_context_mode=null or 'all_views'."
            )
        config["expected_vggt_context_mode"] = "all_views"
        config["vggt_context_mode"] = "all_views"
        config["load_images"] = True
    else:
        feature_dataset_dir = resolve_feature_dataset_dir(config, backbone)
        config["feature_dataset_root"] = str(config["feature_dataset_dir"])
        config["feature_dataset_dir"] = str(feature_dataset_dir)
        config["resolved_feature_dataset_dir"] = str(feature_dataset_dir)
        config["model_input_mode"] = "precomputed_features"
        if lazy_load_image_features:
            print(
                "Lazy cached-feature loading enabled: startup retains feature "
                "descriptors and materialises only each batch's selected "
                "variants/views."
            )
    zero_image_features = resolve_zero_cached_image_features(config, default=False)
    if online_vggt_enabled and zero_image_features:
        raise ValueError(
            "zero_cached_image_features=true is incompatible with "
            "feature_dataset_dir=null online VGGT."
        )
    config["zero_cached_image_features"] = zero_image_features
    if online_vggt_enabled:
        print(
            f"Feature backbone: {backbone}; raw image dataset: "
            f"{input_dataset_dir}; features: frozen VGGT on the selected "
            "all-view context; view directions: original"
        )
    else:
        print(
            f"Feature backbone: {backbone}; feature dataset: {input_dataset_dir}; "
            f"cached image features: {'zeros' if zero_image_features else 'original'}; "
            "view directions: original"
        )
    clinical_indices = resolve_clinical_rca_view_indices(config, backbone)
    config["clinical_rca_views"] = clinical_indices is not None
    if clinical_indices is not None:
        config["view_selection_mode"] = "clinical_rca_views"
        config["clinical_rca_view_indices"] = [int(index) for index in clinical_indices]
    refiner_enabled = bool(
        merged_model_config.get("use_bspline_control_refiner", False)
    )
    refiner_only_training = bool(
        config.get("train_bspline_refiner_only", False)
    )
    radius_refiner_enabled = bool(
        merged_model_config.get("use_radius_evidence_refiner", False)
    )
    radius_refiner_only_training = bool(
        config.get("train_radius_refiner_only", False)
    )
    radius_head_only_training = bool(
        config.get("train_radius_head_only", False)
    )
    bspline_and_radius_head_only_training = bool(
        config.get("train_bspline_refiner_and_radius_head_only", False)
    )
    enabled_training_scopes = [
        refiner_only_training,
        radius_refiner_only_training,
        radius_head_only_training,
        bspline_and_radius_head_only_training,
    ]
    if sum(enabled_training_scopes) > 1:
        raise ValueError(
            "train_bspline_refiner_only, train_radius_refiner_only, "
            "train_radius_head_only, and "
            "train_bspline_refiner_and_radius_head_only are mutually "
            "exclusive."
        )
    if refiner_only_training and not refiner_enabled:
        raise ValueError(
            "train_bspline_refiner_only=true requires "
            "model.use_bspline_control_refiner=true."
        )
    if radius_refiner_only_training and not radius_refiner_enabled:
        raise ValueError(
            "train_radius_refiner_only=true requires "
            "model.use_radius_evidence_refiner=true."
        )
    if radius_head_only_training:
        if not refiner_enabled:
            raise ValueError(
                "train_radius_head_only=true requires "
                "model.use_bspline_control_refiner=true."
            )
        if radius_refiner_enabled:
            raise ValueError(
                "train_radius_head_only=true requires "
                "model.use_radius_evidence_refiner=false so the final radius "
                "comes directly from the raw radius heads."
            )
        if radius_prediction_mode_from_config(config) != "raw":
            raise ValueError(
                "train_radius_head_only=true requires "
                "radius_prediction_mode='raw'."
            )
        if bool(config.get("only_centerline", False)):
            raise ValueError(
                "train_radius_head_only=true is incompatible with "
                "only_centerline=true."
            )
        if bool(config.get("enable_projection_2d_loss", False)):
            raise ValueError(
                "train_radius_head_only=true uses an explicit radius-only "
                "objective and requires enable_projection_2d_loss=false."
            )
    if bspline_and_radius_head_only_training:
        if not refiner_enabled:
            raise ValueError(
                "train_bspline_refiner_and_radius_head_only=true requires "
                "model.use_bspline_control_refiner=true."
            )
        if radius_refiner_enabled:
            raise ValueError(
                "train_bspline_refiner_and_radius_head_only=true requires "
                "model.use_radius_evidence_refiner=false."
            )
        if radius_prediction_mode_from_config(config) != "raw":
            raise ValueError(
                "train_bspline_refiner_and_radius_head_only=true requires "
                "radius_prediction_mode='raw'."
            )
        if bool(config.get("only_centerline", False)):
            raise ValueError(
                "train_bspline_refiner_and_radius_head_only=true is "
                "incompatible with only_centerline=true."
            )
    if radius_refiner_enabled:
        if not refiner_enabled:
            raise ValueError(
                "use_radius_evidence_refiner=true requires "
                "model.use_bspline_control_refiner=true."
            )
        if centerline_mode != "bspline_control_points":
            raise ValueError(
                "use_radius_evidence_refiner=true requires "
                "centerline_prediction_mode='bspline_control_points'."
            )
        if radius_prediction_mode_from_config(config) != "raw":
            raise ValueError(
                "use_radius_evidence_refiner=true requires "
                "radius_prediction_mode='raw'."
            )
        if bool(config.get("only_centerline", False)):
            raise ValueError(
                "use_radius_evidence_refiner=true is incompatible with "
                "only_centerline=true."
            )
        config["load_images"] = True
        print(
            "Iterative radius refinement enabled: rendering the "
            "geometry-refined vessel surface for multi-view mask evidence"
        )
    if refiner_enabled:
        if centerline_mode != "bspline_control_points":
            raise ValueError(
                "use_bspline_control_refiner=true requires "
                "centerline_prediction_mode='bspline_control_points'."
            )
        config["load_images"] = True
        print(
            "B-spline iterative control refinement enabled: loading masks for "
            "local multi-view evidence"
        )
        if bool(
            merged_model_config.get(
                "bspline_refiner_refine_branch_existence",
                False,
            )
        ):
            print(
                "B-spline side-branch existence refinement enabled: "
                "coarse-positive optional slots may be removed; "
                "coarse-negative slots are not added"
            )
    if bool(config.get("enable_projection_2d_loss", False)):
        config["load_images"] = True
        print("2D projection losses enabled: loading masks and preparing distance maps")

    excluded_case_numbers = resolve_excluded_case_numbers(
        config.get("exclude_case_numbers")
    )
    config["exclude_case_numbers"] = [
        f"{case_number:04d}" for case_number in excluded_case_numbers
    ]
    # Persist the complete non-data-dependent configuration before discovery
    # or loading so failed dataset jobs still leave an auditable run record.
    save_json(output_dir / "resolved_config.json", config)
    excluded_dataset_dir_names = resolve_excluded_dataset_dir_names(
        config.get("exclude_dataset_dir_names")
    )
    config["exclude_dataset_dir_names"] = list(excluded_dataset_dir_names)
    discovered_files_unfiltered = discover_npz_files(
        input_dataset_dir,
        recursive=bool(config.get("recursive", True)),
        exclude_dir_names=excluded_dataset_dir_names,
    )
    branch_variant_group_training = resolve_branch_variant_group_training(config)
    config["branch_variant_group_training"] = branch_variant_group_training
    branch_visibility_sampling = resolve_branch_visibility_sampling(config)
    if branch_visibility_sampling.enabled and not branch_variant_group_training:
        raise ValueError(
            "branch_visibility_sampling.enabled=true requires "
            "branch_variant_group_training=true."
        )
    original_variant_only = resolve_parametric_original_variant_only(config)
    parametric_target_source = resolve_parametric_target_source(config)
    config["parametric_target_source"] = parametric_target_source
    if branch_variant_group_training:
        validate_branch_variant_training_config(config)
    if parametric_target_source == "feature_file":
        print(
            "[Data] parametric targets: reading control points and vessel "
            "targets directly from each selected feature NPZ."
        )
    else:
        suffix = (
            " Each Stage-4/5 prefix will apply its own training_branch_mask "
            "to the matched full-case target."
            if branch_variant_group_training
            else ""
        )
        print(
            "[Data] parametric targets: pairing selected features with "
            f"{config['parametric_target_dir']}.{suffix}"
        )
    discovered_files, ignored_variant_files = (
        filter_original_parametric_variant_files(
            discovered_files_unfiltered,
            original_only=original_variant_only,
            source_description=(
                f"Model input dataset directory {input_dataset_dir}"
            ),
        )
    )
    config["parametric_original_variant_only"] = original_variant_only
    config["num_discovered_npz_before_variant_filter"] = len(
        discovered_files_unfiltered
    )
    config["num_ignored_non_original_variant_files"] = len(
        ignored_variant_files
    )
    if original_variant_only:
        print(
            "[Data] original-only counterfactual policy: kept "
            f"{len(discovered_files)} original.npz case files and ignored "
            f"{len(ignored_variant_files)} non-original variants."
        )
    all_files, excluded_case_files = filter_excluded_case_files(
        discovered_files,
        excluded_case_numbers,
    )
    config["num_discovered_cases_before_case_exclusion"] = len(discovered_files)
    config["num_discovered_cases"] = len(all_files)
    config["num_excluded_cases"] = len(excluded_case_files)
    if excluded_case_files:
        print(
            "[Split] ignored configured case number(s): "
            + ", ".join(path.stem for path in excluded_case_files)
        )
    split_json_value = config.get("split_json_path")
    dataset_split_mode = str(
        config.get("dataset_split_mode", "ratio")
    ).strip().lower()
    if dataset_split_mode not in {"ratio", "subdirectories"}:
        raise ValueError(
            "dataset_split_mode must be 'ratio' or 'subdirectories', got "
            f"{dataset_split_mode!r}."
        )
    if split_json_value and dataset_split_mode == "subdirectories":
        raise ValueError(
            "dataset_split_mode='subdirectories' cannot be combined with "
            "split_json_path."
        )
    config["dataset_split_mode"] = dataset_split_mode
    if branch_variant_group_training:
        if dataset_split_mode == "subdirectories":
            raise ValueError(
                "dataset_split_mode='subdirectories' is not supported with "
                "branch_variant_group_training."
            )
        branch_groups = discover_branch_variant_groups(
            all_files,
            num_branches=int(merged_model_config.get("num_branches", 7)),
        )
        grouped_vessel_types = {
            group.group_id.split(":", 1)[0].lower()
            for group in branch_groups
        }
        if grouped_vessel_types != {str(artery_type).strip().lower()}:
            raise ValueError(
                "Configured artery_type does not match the grouped "
                "branch-variant dataset: configured="
                f"{artery_type!r}, dataset={sorted(grouped_vessel_types)}."
            )
        representative_files = [group.paths[0] for group in branch_groups]
        config["num_discovered_variant_files"] = len(all_files)
        config["num_discovered_physical_cases"] = len(branch_groups)
        if split_json_value:
            representative_split = _explicit_branch_variant_group_split(
                split_json_value,
                groups=branch_groups,
                config=config,
            )
            files = [
                *representative_split.train,
                *representative_split.val,
                *representative_split.test,
            ]
            config["split_json_path"] = str(
                Path(str(split_json_value)).expanduser().resolve()
            )
            config["effective_split_mode"] = (
                "explicit_split_json_grouped_by_physical_case"
            )
        else:
            files = select_case_files(
                representative_files,
                num_cases=config.get("num_cases"),
                case_fraction=config.get("case_fraction"),
                seed=seed,
                shuffle=bool(config.get("shuffle_cases", True)),
            )
            files = include_spy_files(files, representative_files, config)
            representative_split = split_case_files(
                files,
                float(config.get("train_ratio", 0.8)),
                float(config.get("val_ratio", 0.1)),
                float(config.get("test_ratio", 0.1)),
                seed=seed,
                shuffle=bool(config.get("shuffle_cases", True)),
            )
            config["effective_split_mode"] = (
                "ratio_split_grouped_by_physical_case"
            )
        split = _expand_branch_variant_split(
            representative_split, branch_groups
        )
        config["num_selected_cases"] = len(files)
        config["num_selected_variant_files"] = sum(
            len(paths) for paths in (split.train, split.val, split.test)
        )
        config["branch_variant_split_variant_counts"] = {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        }
        # split.json remains case-level and therefore reusable by any later
        # stage without encoding an arbitrary number of variant paths.
        save_split_record(output_dir / "split.json", representative_split, config)
        split_by_case = {
            case_identifier(path): split_name
            for split_name, paths in (
                ("train", representative_split.train),
                ("val", representative_split.val),
                ("test", representative_split.test),
            )
            for path in paths
        }
        branch_variant_manifest = {
            "schema_version": 1,
            "grouping_unit": "physical_case",
            "train_sampling": str(
                config.get("branch_variant_train_sampling", "all")
            ),
            "train_variants_per_group": int(
                config.get("branch_variant_train_variants_per_group", 1)
            ),
            "validation_sampling": "all",
            "groups": [],
        }
        if branch_visibility_sampling.enabled:
            assert branch_visibility_sampling.metadata_json is not None
            visibility_metadata = load_stage4_1_visibility_metadata(
                branch_visibility_sampling.metadata_json,
                num_branches=int(merged_model_config.get("num_branches", 7)),
            )
            branch_variant_manifest["branch_visibility_sampling"] = {
                "stage": "stage4.1",
                "metadata_json": str(branch_visibility_sampling.metadata_json),
                "metadata_sha256": str(visibility_metadata["sha256"]),
                "unsatisfied_policy": str(
                    branch_visibility_sampling.unsatisfied_policy
                ),
                "apply_to_validation": bool(
                    branch_visibility_sampling.apply_to_validation
                ),
            }
            config["branch_visibility_metadata_sha256"] = str(
                visibility_metadata["sha256"]
            )
        for group in branch_groups:
            split_name = split_by_case.get(group.case_id)
            if split_name is None:
                continue
            variant_records = []
            for variant, path in zip(group.variants, group.paths):
                metadata = load_branch_variant_metadata(
                    path,
                    num_branches=int(
                        merged_model_config.get("num_branches", 7)
                    ),
                )
                variant_records.append(
                    {
                        "variant": variant,
                        "path": str(path),
                        "training_branch_mask": np.asarray(
                            metadata["training_branch_mask"], dtype=bool
                        ).tolist(),
                        "selected_branch_indices": np.asarray(
                            metadata["selected_branch_indices"], dtype=np.int64
                        ).tolist(),
                        "projected_branch_indices": np.asarray(
                            metadata["projected_branch_indices"], dtype=np.int64
                        ).tolist(),
                    }
                )
            branch_variant_manifest["groups"].append(
                {
                    "case_id": group.case_id,
                    "group_id": group.group_id,
                    "stage": group.stage,
                    "split": split_name,
                    "full_branch_exists": list(group.full_branch_exists),
                    "variants": variant_records,
                }
            )
        save_json(
            output_dir / "branch_variant_group_manifest.json",
            branch_variant_manifest,
        )
        config["branch_variant_group_manifest_sha256"] = hashlib.sha256(
            json.dumps(
                branch_variant_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        print(
            "[Split] grouped branch variants: "
            f"{len(files)} physical cases, "
            f"{config['num_selected_variant_files']} variant files; every "
            "case and all of its variants share one split."
        )
    else:
        if split_json_value:
            split = _load_explicit_split(
                split_json_value,
                discovered_files=discovered_files,
                config=config,
            )
            files = [*split.train, *split.val, *split.test]
            config["split_json_path"] = str(
                Path(str(split_json_value)).expanduser().resolve()
            )
            config["effective_split_mode"] = "explicit_split_json"
        elif dataset_split_mode == "subdirectories":
            directory_paths = discover_npz_split_subdirectories(
                input_dataset_dir,
                exclude_dir_names=excluded_dataset_dir_names,
            )
            eligible_paths = {path.resolve() for path in all_files}
            split = SplitFiles(
                train=[
                    path for path in directory_paths["train"]
                    if path in eligible_paths
                ],
                val=[
                    path for path in directory_paths["val"]
                    if path in eligible_paths
                ],
                test=[
                    path for path in directory_paths["test"]
                    if path in eligible_paths
                ],
            )
            empty_splits = [
                name
                for name, paths in (
                    ("train", split.train),
                    ("validation", split.val),
                    ("test", split.test),
                )
                if not paths
            ]
            if empty_splits:
                raise ValueError(
                    "Configured exclusions removed every case from supplied "
                    f"split(s): {empty_splits}."
                )
            files = [*split.train, *split.val, *split.test]
            config["effective_split_mode"] = "existing_subdirectories"
        else:
            files = select_case_files(
                all_files,
                num_cases=config.get("num_cases"),
                case_fraction=config.get("case_fraction"),
                seed=seed,
                shuffle=bool(config.get("shuffle_cases", True)),
            )
            files = include_spy_files(files, all_files, config)
            split = split_case_files(
                files,
                float(config.get("train_ratio", 0.8)),
                float(config.get("val_ratio", 0.1)),
                float(config.get("test_ratio", 0.1)),
                seed=seed,
                shuffle=bool(config.get("shuffle_cases", True)),
            )
            config["effective_split_mode"] = "ratio_split_on_selected_cases"
        config["num_selected_cases"] = len(files)
        save_split_record(output_dir / "split.json", split, config)
    train_items = load_paired_items(split.train, config, desc="Loading parametric training cases")
    val_items = load_paired_items(split.val or split.train, config, desc="Loading parametric validation cases")
    test_items = load_paired_items(split.test, config, desc="Loading parametric test cases") if split.test else []
    validate_target_scale(train_items, config, "train")
    validate_target_scale(val_items, config, "validation")
    if test_items:
        validate_target_scale(test_items, config, "test")

    shapes = infer_feature_shapes(train_items)
    inferred_dim = (
        int(
            merged_model_config.get(
                "vggt_token_dim",
                config.get("vggt_token_dim", 2048),
            )
        )
        if online_vggt_enabled
        else (
            None
            if backbone == "resnet_pre_fpn"
            else int(next(iter(shapes.values()))[-1])
        )
    )
    channels = infer_resnet_pre_fpn_channels(train_items)
    if channels is not None:
        config.setdefault("model", {})["resnet_pre_fpn_channels"] = list(channels)
    view_feat_dim = int(np.asarray(train_items[0]["view_features"]).shape[-1])
    device = device_from_config(config)
    online_vggt = (
        OnlineVGGTFeatureProvider(
            config,
            device=device,
            raw_image_dataset_dir=input_dataset_dir,
        )
        if online_vggt_enabled
        else None
    )
    if online_vggt is not None:
        config.update(online_vggt.resolved_config())
        config["online_vggt_cache_stats"] = online_vggt.stats()
        print(
            "Online VGGT cache: "
            f"read={online_vggt.cache_read}, write={online_vggt.cache_write}, "
            f"directory={online_vggt.cache_dir}, "
            f"signature={online_vggt.signature_hash}"
        )
    model = build_model_from_config(config, view_feat_dim, inferred_dim).to(device)
    resume_value = (
        args.resume if args.resume is not None else config.get("resume_checkpoint")
    )
    initial_checkpoint_value = config.get("initial_checkpoint")
    if resume_value and initial_checkpoint_value:
        raise ValueError(
            "Configure either resume_checkpoint/--resume or initial_checkpoint, "
            "not both."
        )
    any_specialized_only_training = (
        refiner_only_training
        or radius_refiner_only_training
        or radius_head_only_training
        or bspline_and_radius_head_only_training
    )
    if any_specialized_only_training and not (
        resume_value or initial_checkpoint_value
    ):
        raise ValueError(
            "Specialized refiner/radius-head training requires "
            "initial_checkpoint for a new run, or resume_checkpoint/--resume "
            "for an existing specialized run."
        )
    if initial_checkpoint_value:
        initial_checkpoint_path = Path(initial_checkpoint_value).expanduser()
        initial_checkpoint = torch.load(
            initial_checkpoint_path, map_location=device
        , weights_only=False)
        source_config = dict(initial_checkpoint.get("config", {}))
        compared_coarse_fields = validate_geometry_refiner_coarse_config(
            config, source_config
        )
        if compared_coarse_fields:
            print(
                "Verified frozen coarse-model configuration: "
                f"{len(compared_coarse_fields)} model/input/view fields match "
                "the initial checkpoint."
            )
        source_radius_mode = radius_prediction_mode_from_config(source_config)
        if source_radius_mode != radius_prediction_mode_from_config(config):
            raise ValueError(
                "initial_checkpoint radius mode does not match the new model: "
                f"checkpoint={source_radius_mode!r}."
            )
        source_centerline_mode = centerline_prediction_mode_from_config(
            source_config
        )
        if source_centerline_mode != centerline_mode:
            raise ValueError(
                "initial_checkpoint centreline mode does not match the new model: "
                f"checkpoint={source_centerline_mode!r}, current={centerline_mode!r}."
            )
        incompatible = model.load_state_dict(
            initial_checkpoint["model_state_dict"], strict=False
        )
        new_bspline_refiner_missing = {
            key
            for key in incompatible.missing_keys
            if key == "bspline_refiner_anchor_indices"
            or key.startswith("bspline_control_refiner.")
        }
        new_radius_refiner_missing = {
            key
            for key in incompatible.missing_keys
            if key.startswith("radius_evidence_refiner.")
        }
        # The base architecture historically constructs this optional module
        # even for a one-branch RCA model, although forward() can never use it.
        # Therefore a true/false configuration difference only changes inert
        # state-dict entries and is safe to ignore for num_branches == 1.
        unused_single_branch_prefixes = (
            ("side_parent_projection.",)
            if int(getattr(model, "num_branches", 0)) == 1
            else ()
        )
        unused_side_missing = {
            key
            for key in incompatible.missing_keys
            if key.startswith(unused_single_branch_prefixes)
        }
        ignored_unexpected = {
            key
            for key in incompatible.unexpected_keys
            if key.startswith(unused_single_branch_prefixes)
        }
        allowed_missing = (
            new_bspline_refiner_missing
            | new_radius_refiner_missing
            | unused_side_missing
        )
        disallowed_missing = sorted(
            set(incompatible.missing_keys) - allowed_missing
        )
        disallowed_unexpected = sorted(
            set(incompatible.unexpected_keys) - ignored_unexpected
        )
        if disallowed_unexpected or disallowed_missing:
            raise RuntimeError(
                "initial_checkpoint is not architecture-compatible. "
                f"Missing keys: {disallowed_missing}; unexpected keys: "
                f"{disallowed_unexpected}."
            )
        if new_bspline_refiner_missing and not refiner_enabled:
            raise RuntimeError(
                "initial_checkpoint is missing refiner weights, but the current "
                "model does not enable B-spline refinement."
            )
        if radius_refiner_only_training and new_bspline_refiner_missing:
            raise RuntimeError(
                "train_radius_refiner_only=true requires an initial checkpoint "
                "that already contains trained B-spline geometry-refiner "
                "weights; the supplied checkpoint does not contain them."
            )
        if radius_head_only_training and new_bspline_refiner_missing:
            raise RuntimeError(
                "train_radius_head_only=true requires an initial checkpoint "
                "that already contains trained B-spline geometry-refiner "
                "weights; the supplied checkpoint does not contain them."
            )
        if new_radius_refiner_missing and not radius_refiner_enabled:
            raise RuntimeError(
                "initial_checkpoint is missing radius-refiner weights, but the "
                "current model does not enable radius refinement."
            )
        config["initial_checkpoint_resolved"] = str(initial_checkpoint_path)
        config["initial_checkpoint_source_epoch"] = int(
            initial_checkpoint.get("epoch", -1)
        )
        ignored_side_tensor_count = len(ignored_unexpected) + len(
            unused_side_missing
        )
        print(
            f"Initialized coarse model from {initial_checkpoint_path}; "
            "new B-spline refiner tensors: "
            f"{len(new_bspline_refiner_missing)}; new radius-refiner tensors: "
            f"{len(new_radius_refiner_missing)}; ignored inert "
            f"single-branch side tensors: {ignored_side_tensor_count}"
        )
    pretrained_centerline_info = configure_pretrained_centerline_predictor(
        model,
        config,
        load_weights=not bool(resume_value),
    )
    if pretrained_centerline_info is not None:
        print(
            "Pretrained centreline predictor: "
            f"checkpoint={pretrained_centerline_info['checkpoint']}, "
            f"separate_encoder="
            f"{pretrained_centerline_info['separate_encoder']}, "
            f"frozen={pretrained_centerline_info['frozen']}"
        )
    projector = None
    centerline_loss_config = config.get("loss", {})
    if not isinstance(centerline_loss_config, Mapping):
        raise ValueError("loss must be a JSON object.")
    centerline_map_supervision_enabled = bool(
        merged_model_config.get(
            "bspline_refiner_use_unexplained_centerline_evidence",
            False,
        )
    ) and float(
        centerline_loss_config.get(
            "bspline_refiner_centerline_map_loss_weight",
            config.get("bspline_refiner_centerline_map_loss_weight", 0.1),
        )
    ) > 0.0
    if bool(config.get("enable_projection_2d_loss", False)):
        sample_images = np.asarray(train_items[0]["images"])
        projector = build_centerline_projector(
            config,
            int(sample_images.shape[-1]),
        ).to(device)
    bspline_refiner_projector = None
    if centerline_map_supervision_enabled:
        bspline_refiner_projector = (
            build_bspline_refiner_centerline_projector(config).to(device)
        )
    trainable_parameters = configure_bspline_refiner_only_training(model, config)
    if refiner_only_training:
        print(
            "Refiner-only training enabled: the pretrained coarse predictor is "
            "frozen in evaluation mode; only bspline_control_refiner parameters "
            "will be optimized"
        )
    elif radius_refiner_only_training:
        print(
            "Radius-refiner-only training enabled: the pretrained coarse "
            "predictor and B-spline geometry refiner are frozen in evaluation "
            "mode; only radius_evidence_refiner parameters will be optimized"
        )
    elif radius_head_only_training:
        trainable_radius_heads = ["raw_radius_head"]
        if (
            int(getattr(model, "num_branches", 1)) > 1
            and getattr(model, "side_raw_radius_head", None) is not None
        ):
            trainable_radius_heads.append("side_raw_radius_head")
        print(
            "Radius-head-only training enabled: the pretrained coarse "
            "predictor and B-spline geometry refiner are frozen in evaluation "
            "mode; only "
            + " and ".join(trainable_radius_heads)
            + " parameters will be optimized"
        )
    elif bspline_and_radius_head_only_training:
        trainable_modules = ["bspline_control_refiner", "raw_radius_head"]
        if (
            int(getattr(model, "num_branches", 1)) > 1
            and getattr(model, "side_raw_radius_head", None) is not None
        ):
            trainable_modules.append("side_raw_radius_head")
        print(
            "Joint B-spline-refiner/radius-head training enabled: the "
            "rest of the pretrained predictor is frozen in evaluation mode; "
            "only "
            + ", ".join(trainable_modules)
            + " parameters will be optimized"
        )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config.get("learning_rate", 1e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(config.get("min_learning_rate", 1e-6)))
    train_loader = make_loader(train_items, config, train=True)
    train_dataset = train_loader.dataset
    if branch_variant_group_training:
        print(
            "[Training] grouped branch variants: one physical case per "
            "optimizer step; train sampling="
            f"{config.get('branch_variant_train_sampling', 'all')!r}; "
            "validation/test sampling='all'."
        )
    train_view_count_probabilities = (
        train_loader.dataset.base.view_count_weights
    )
    if train_view_count_probabilities is not None:
        config["resolved_train_view_count_probabilities"] = {
            str(count): probability
            for count, probability in train_view_count_probabilities.items()
        }
        distribution = ", ".join(
            f"{count}={probability:.1%}"
            for count, probability in train_view_count_probabilities.items()
            if probability > 0.0
        )
        print(f"Training view-count probabilities: {distribution}")
    val_counts = validation_view_counts(config)
    val_loaders = {count: make_loader(val_items, config, train=False, num_views=count) for count in val_counts}
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_all_losses = float("inf")
    start_epoch = 1

    if resume_value:
        resume_path = output_dir / "latest.pt" if str(resume_value) == "latest" else Path(resume_value)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_mode = radius_prediction_mode_from_config(
            dict(checkpoint.get("config", {}))
        )
        current_mode = radius_prediction_mode_from_config(config)
        if checkpoint_mode != current_mode:
            raise ValueError(
                "Cannot resume with a different radius_prediction_mode: "
                f"checkpoint={checkpoint_mode!r}, config={current_mode!r}"
            )
        checkpoint_centerline_mode = centerline_prediction_mode_from_config(
            dict(checkpoint.get("config", {}))
        )
        if checkpoint_centerline_mode != centerline_mode:
            raise ValueError(
                "Cannot resume with a different centerline_prediction_mode: "
                f"checkpoint={checkpoint_centerline_mode!r}, "
                f"config={centerline_mode!r}"
            )
        checkpoint_config = dict(checkpoint.get("config", {}))
        checkpoint_grouped = resolve_branch_variant_group_training(
            checkpoint_config
        )
        if checkpoint_grouped != branch_variant_group_training:
            raise ValueError(
                "Cannot resume with a different "
                "branch_variant_group_training setting: checkpoint="
                f"{checkpoint_grouped}, config={branch_variant_group_training}."
            )
        if branch_variant_group_training:
            for setting, default in (
                ("parametric_target_source", "directory"),
                ("branch_variant_train_sampling", "all"),
                ("branch_variant_train_variants_per_group", 1),
                ("branch_variant_resample_each_epoch", True),
            ):
                checkpoint_value = checkpoint_config.get(setting, default)
                current_value = config.get(setting, default)
                if checkpoint_value != current_value:
                    raise ValueError(
                        f"Cannot resume grouped branch-variant training with "
                        f"a different {setting}: checkpoint="
                        f"{checkpoint_value!r}, config={current_value!r}."
                    )
            checkpoint_manifest_hash = checkpoint_config.get(
                "branch_variant_group_manifest_sha256"
            )
            current_manifest_hash = config.get(
                "branch_variant_group_manifest_sha256"
            )
            if checkpoint_manifest_hash != current_manifest_hash:
                raise ValueError(
                    "Cannot resume because the branch-variant group/split "
                    "manifest changed: checkpoint="
                    f"{checkpoint_manifest_hash!r}, current="
                    f"{current_manifest_hash!r}."
                )
        checkpoint_refiner_only = bool(
            dict(checkpoint.get("config", {})).get(
                "train_bspline_refiner_only", False
            )
        )
        if checkpoint_refiner_only != refiner_only_training:
            raise ValueError(
                "Cannot resume with a different train_bspline_refiner_only "
                "setting: checkpoint="
                f"{checkpoint_refiner_only}, config={refiner_only_training}. "
                "Use initial_checkpoint to start a new optimizer instead."
            )
        checkpoint_radius_refiner_only = bool(
            dict(checkpoint.get("config", {})).get(
                "train_radius_refiner_only", False
            )
        )
        if checkpoint_radius_refiner_only != radius_refiner_only_training:
            raise ValueError(
                "Cannot resume with a different train_radius_refiner_only "
                "setting: checkpoint="
                f"{checkpoint_radius_refiner_only}, "
                f"config={radius_refiner_only_training}. Use "
                "initial_checkpoint to start a new optimizer instead."
            )
        checkpoint_radius_head_only = bool(
            dict(checkpoint.get("config", {})).get(
                "train_radius_head_only", False
            )
        )
        if checkpoint_radius_head_only != radius_head_only_training:
            raise ValueError(
                "Cannot resume with a different train_radius_head_only "
                "setting: checkpoint="
                f"{checkpoint_radius_head_only}, "
                f"config={radius_head_only_training}. Use initial_checkpoint "
                "to start a new optimizer instead."
            )
        checkpoint_bspline_and_radius_head_only = bool(
            dict(checkpoint.get("config", {})).get(
                "train_bspline_refiner_and_radius_head_only", False
            )
        )
        if (
            checkpoint_bspline_and_radius_head_only
            != bspline_and_radius_head_only_training
        ):
            raise ValueError(
                "Cannot resume with a different "
                "train_bspline_refiner_and_radius_head_only setting: "
                "checkpoint="
                f"{checkpoint_bspline_and_radius_head_only}, config="
                f"{bspline_and_radius_head_only_training}. Use "
                "initial_checkpoint to start a new optimizer instead."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        checkpoint_selection_metric = str(
            dict(checkpoint.get("config", {})).get(
                "checkpoint_selection_metric", "legacy_last_validation_view"
            )
        )
        if checkpoint_selection_metric == "mean_validation_loss":
            best = float(checkpoint.get("best_val_loss", best))
            best_all_losses = float(
                checkpoint.get("best_all_losses_val_loss", best_all_losses)
            )
        else:
            print(
                "[warn] Resumed checkpoint used a different validation-view "
                "selection rule; resetting best-loss trackers for mean-view "
                "checkpoint selection."
            )
        history = list(checkpoint.get("history", []))
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    schedule_mode, schedule_path = resolve_training_sampling_schedule(
        config,
        output_dir=output_dir,
    )
    config["training_sampling_schedule_mode"] = schedule_mode
    config["resolved_training_sampling_schedule_path"] = (
        None if schedule_path is None else str(schedule_path)
    )
    training_sampling_schedule: TrainingSamplingSchedule | None = None
    if schedule_mode != "off":
        assert schedule_path is not None
        sampling_policy = sampling_policy_from_config(
            config,
            grouped=branch_variant_group_training,
            normalized_view_count_weights=train_view_count_probabilities,
            replacement_sampling=bool(
                getattr(train_loader.sampler, "replacement", False)
            ),
        )
        training_sampling_schedule = TrainingSamplingSchedule(
            mode=schedule_mode,
            path=schedule_path,
            sampling_policy=sampling_policy,
            resume_recording=bool(resume_value),
        )
        config["training_sampling_schedule_policy"] = dict(
            training_sampling_schedule.payload.get(
                "sampling_policy", sampling_policy
            )
        )
        if schedule_mode == "replay":
            replay_dataset = ReplayTrainingBatchDataset(
                train_dataset,
                training_sampling_schedule,
                initial_epoch=start_epoch,
            )
            train_loader = DataLoader(
                replay_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=int(config.get("num_workers", 0)),
                pin_memory=torch.cuda.is_available(),
            )
        print(
            f"Training sampling schedule: mode={schedule_mode}, "
            f"path={schedule_path}"
        )

    config["effective_val_num_views"] = val_counts
    config["effective_val_random_view_order"] = bool(config.get("val_random_view_order", False))
    config["checkpoint_selection_metric"] = "mean_validation_loss"
    config["checkpoint_selection_view_counts"] = [
        "all" if count is None else int(count) for count in val_counts
    ]
    if branch_variant_group_training:
        config["num_train_variants"] = len(train_items)
        config["num_val_variants"] = len(val_items)
        config["num_test_variants"] = len(test_items)
        config["num_train_cases"] = len(train_dataset)
        first_val_loader = next(iter(val_loaders.values()))
        config["num_val_cases"] = len(first_val_loader.dataset)
        config["num_test_cases"] = len(
            {case_identifier(item["path"]) for item in test_items}
        )
        config["branch_variant_training_objective"] = (
            "one optimiser step per physical case group; scalar loss uses "
            "the standard active-element reduction over selected variants"
        )
        config["branch_variant_validation_sampling"] = "all"
    else:
        config["num_train_cases"] = len(train_items)
        config["num_val_cases"] = len(val_items)
        config["num_test_cases"] = len(test_items)
    save_json(output_dir / "resolved_config.json", config)
    monitor_items = _monitor_items(train_items, val_items, test_items, config)
    monitor_every = int(config.get("monitor_every_epochs", 10))
    if monitor_every < 1:
        raise ValueError("monitor_every_epochs must be >= 1.")

    for epoch in range(start_epoch, epochs + 1):
        train_started_at = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer,
            projector=projector,
            bspline_refiner_projector=bspline_refiner_projector,
            epoch=epoch,
            online_vggt=online_vggt,
            training_sampling_schedule=training_sampling_schedule,
        )
        train_elapsed_seconds = time.perf_counter() - train_started_at
        validation_started_at = time.perf_counter()
        validation: dict[int | None, dict[str, float]] = {
            count: run_epoch(
                model,
                loader,
                config,
                device,
                None,
                projector=projector,
                bspline_refiner_projector=bspline_refiner_projector,
                epoch=epoch,
                online_vggt=online_vggt,
            )
            for count, loader in val_loaders.items()
        }
        validation_elapsed_seconds = (
            time.perf_counter() - validation_started_at
        )
        train_metrics["all_losses_loss"] = _all_losses_objective(
            train_metrics, config
        )
        for metrics in validation.values():
            metrics["all_losses_loss"] = _all_losses_objective(metrics, config)
        primary_val = _mean_validation_metrics(list(validation.values()))
        all_losses_val = float(primary_val["all_losses_loss"])
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        for count, metrics in validation.items():
            prefix = "val" if len(validation) == 1 else f"val_k{count}" if count is not None else "val_all"
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        row.update(
            {f"val_mean_{key}": value for key, value in primary_val.items()}
        )
        history.append(row)
        primary_val_label = "val_mean"
        train_message = _format_epoch_loss_metrics(train_metrics)
        validation_message = _format_epoch_loss_metrics(primary_val)
        print(
            f"[Epoch {epoch:04d}] train {train_message} | "
            f"{primary_val_label} {validation_message} | "
            f"timing train={train_elapsed_seconds:.1f}s, "
            f"validation={validation_elapsed_seconds:.1f}s",
            flush=True,
        )
        if online_vggt is not None:
            config["online_vggt_cache_stats"] = online_vggt.stats()
            cache_stats = config["online_vggt_cache_stats"]
            print(
                "[Online VGGT cache] "
                f"requests={cache_stats['requests']}, "
                f"hits={cache_stats['cache_hits']}, "
                f"computed={cache_stats['extracted_cases']}, "
                f"writes={cache_stats['cache_writes']}",
                flush=True,
            )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "view_feat_dim": view_feat_dim,
            "inferred_feature_dim": inferred_dim,
            "best_val_loss": min(best, primary_val["loss"]),
            "all_losses_val_loss": all_losses_val,
            "best_all_losses_val_loss": min(best_all_losses, all_losses_val),
            "history": history,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if primary_val["loss"] < best:
            best = primary_val["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
        if all_losses_val < best_all_losses:
            best_all_losses = all_losses_val
            torch.save(checkpoint, output_dir / "best_all_losses.pt")
        is_monitor_epoch = epoch % monitor_every == 0 or epoch == epochs
        _write_history(
            history,
            output_dir,
            render_loss_plot=is_monitor_epoch,
        )
        parameter_metric_prefix = (
            "landmark_"
            if centerline_mode == "adaptive_landmarks"
            else "control_point_"
        )
        parameter_file_prefix = (
            "landmark" if centerline_mode == "adaptive_landmarks" else "control_point"
        )
        centerline_parameter_summary = {
            "epoch": int(epoch),
            "centerline_prediction_mode": centerline_mode,
            "train": {
                key: value
                for key, value in train_metrics.items()
                if key.startswith(parameter_metric_prefix)
            },
            "val": {
                key: value
                for key, value in primary_val.items()
                if key.startswith(parameter_metric_prefix)
            },
        }
        if not (
            radius_refiner_only_training or radius_head_only_training
        ):
            save_control_point_error_summary(
                train_metrics=train_metrics,
                val_metrics=primary_val,
                path=output_dir
                / "training_monitor"
                / f"{parameter_file_prefix}_error_train_val_latest.png",
                epoch=epoch,
                absolute_parameters=(
                    getattr(model, "decoder_architecture", "")
                    == "absolute_parallel"
                ),
            )
            save_json(
                output_dir
                / "training_monitor"
                / f"{parameter_file_prefix}_error_train_val_latest.json",
                centerline_parameter_summary,
            )
        if is_monitor_epoch:
            save_monitors(
                model,
                monitor_items,
                config,
                device,
                output_dir,
                epoch,
                online_vggt=online_vggt,
            )
            epoch_monitor_dir = (
                output_dir / "training_monitor" / f"epoch_{epoch:04d}"
            )
            if not (
                radius_refiner_only_training or radius_head_only_training
            ):
                save_control_point_error_summary(
                    train_metrics=train_metrics,
                    val_metrics=primary_val,
                    path=epoch_monitor_dir
                    / f"{parameter_file_prefix}_error_train_val.png",
                    epoch=epoch,
                    absolute_parameters=(
                        getattr(model, "decoder_architecture", "")
                        == "absolute_parallel"
                    ),
                )
                save_json(
                    epoch_monitor_dir
                    / f"{parameter_file_prefix}_error_train_val.json",
                    centerline_parameter_summary,
                )
            save_json(
                epoch_monitor_dir / "checkpoint_status.json",
                {
                    "epoch": epoch,
                    "best_val_loss": best,
                    "best_all_losses_val_loss": best_all_losses,
                    "latest_checkpoint": str(output_dir / "latest.pt"),
                    "best_checkpoint": str(output_dir / "best.pt"),
                    "best_all_losses_checkpoint": str(
                        output_dir / "best_all_losses.pt"
                    ),
                },
            )

    if online_vggt is not None:
        config["online_vggt_cache_stats"] = online_vggt.stats()
        save_json(output_dir / "resolved_config.json", config)

    if test_items and bool(config.get("evaluate_test_after_training", True)):
        best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = {}
        for count in val_counts:
            label = "all" if count is None else f"k{count}"
            test_metrics[label] = run_epoch(
                model,
                make_loader(test_items, config, train=False, num_views=count),
                config,
                device,
                None,
                projector=projector,
                bspline_refiner_projector=bspline_refiner_projector,
                epoch=epochs,
                online_vggt=online_vggt,
            )
        if online_vggt is not None:
            config["online_vggt_cache_stats"] = online_vggt.stats()
            save_json(output_dir / "resolved_config.json", config)
        save_json(output_dir / "test_metrics.json", test_metrics)
        print(f"Saved final test metrics to {output_dir / 'test_metrics.json'}")

if __name__ == "__main__":
    main()
