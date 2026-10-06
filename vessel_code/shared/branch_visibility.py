# Transferred from methods/src/branch_visibility.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from typing import Any
import numpy as np

VISUALIZATION_BRANCH_EXISTENCE_SOURCES = {"ground_truth", "predicted"}

REFINER_BRANCH_EXISTENCE_SOURCES = {"ground_truth", "predicted"}

def resolve_artery_type(config: dict[str, Any]) -> str:
    """Resolve RCA/LCA while remaining compatible with older path-only configs."""

    explicit = config.get("artery_type")
    if explicit is not None:
        artery_type = str(explicit).strip().upper()
        if artery_type not in {"RCA", "LCA"}:
            raise ValueError("artery_type must be 'RCA' or 'LCA'.")
        return artery_type

    path_keys = (
        "dataset_dir",
        "feature_dataset_dir",
        "parametric_target_dir",
        "external_dataset_dir",
        "experiment_name",
        "experiment_dir",
        "output_dir",
        "out_dir",
    )
    path_text = " ".join(
        str(config[key]).lower()
        for key in path_keys
        if config.get(key) not in {None, ""}
    )
    if "lca" in path_text:
        return "LCA"
    return "RCA"

def required_branch_count(artery_type: str) -> int:
    normalized = str(artery_type).strip().upper()
    if normalized == "RCA":
        return 1
    if normalized == "LCA":
        return 2
    raise ValueError("artery_type must be 'RCA' or 'LCA'.")

def visualization_branch_existence_source(config: dict[str, Any]) -> str:
    raw_source = config.get("visualization_branch_existence_source")
    source = str(
        "ground_truth" if raw_source is None else raw_source
    ).strip().lower()
    if source not in VISUALIZATION_BRANCH_EXISTENCE_SOURCES:
        raise ValueError(
            "visualization_branch_existence_source must be 'predicted' or "
            "'ground_truth'."
        )
    return source

def refiner_branch_existence_source(config: dict[str, Any]) -> str:
    """Resolve which geometry/radius mask is supplied during refiner training.

    ``predicted`` preserves the historical behaviour when the option is absent.
    Refiner-only training templates explicitly select ``ground_truth`` so their
    geometry stages receive teacher-forced branch activity masks. Optional
    B-spline existence refinement instead uses the same coarse-positive
    optional-branch trajectory in training and inference, ensuring the
    ground-truth existence label is used only as loss supervision.
    """

    raw_source = config.get("refiner_branch_existence_source")
    source = str("predicted" if raw_source is None else raw_source).strip().lower()
    if source not in REFINER_BRANCH_EXISTENCE_SOURCES:
        raise ValueError(
            "refiner_branch_existence_source must be 'predicted' or "
            "'ground_truth'."
        )
    config["refiner_branch_existence_source"] = source
    return source

def validate_visualization_branch_existence_config(
    config: dict[str, Any],
    *,
    num_branches: int,
) -> tuple[str, str]:
    """Validate and return the artery type and paired-visualization policy."""

    count = int(num_branches)
    if count < 1:
        raise ValueError(f"num_branches must be >= 1, got {num_branches}.")
    artery_type = resolve_artery_type(config)
    minimum_count = required_branch_count(artery_type)
    if count < minimum_count:
        raise ValueError(
            f"{artery_type} requires at least {minimum_count} branches, "
            f"got num_branches={count}."
        )
    source = visualization_branch_existence_source(config)
    config["artery_type"] = artery_type
    config["visualization_branch_existence_source"] = source
    return artery_type, source

def resolve_paired_visualization_branch_masks(
    *,
    config: dict[str, Any],
    target_exist: np.ndarray,
    predicted_exist_probabilities: np.ndarray,
    num_branches: int | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate target and prediction masks for paired visualizations.

    RCA branch 0 and LCA branches 0--1 are fixed-topology branches and are
    always visible. Optional branches use either target labels or thresholded
    prediction probabilities according to
    ``visualization_branch_existence_source``.
    """

    target_values = np.asarray(target_exist, dtype=np.float32).reshape(-1)
    predicted_values = np.asarray(
        predicted_exist_probabilities, dtype=np.float32
    ).reshape(-1)
    count = int(
        max(target_values.size, predicted_values.size)
        if num_branches is None
        else num_branches
    )
    artery_type, source = validate_visualization_branch_existence_config(
        config,
        num_branches=count,
    )
    probability_threshold = float(threshold)
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("Visualization existence threshold must be in [0,1].")

    target_mask = np.zeros((count,), dtype=np.bool_)
    predicted_mask = np.zeros((count,), dtype=np.bool_)
    target_count = min(count, int(target_values.size))
    predicted_count = min(count, int(predicted_values.size))
    target_mask[:target_count] = target_values[:target_count] > 0.5
    predicted_mask[:predicted_count] = (
        predicted_values[:predicted_count] >= probability_threshold
    )

    fixed_count = min(count, required_branch_count(artery_type))
    target_mask[:fixed_count] = True
    predicted_mask[:fixed_count] = True
    if source == "ground_truth":
        prediction_visualization_mask = target_mask.copy()
    else:
        prediction_visualization_mask = predicted_mask
    return target_mask, prediction_visualization_mask
