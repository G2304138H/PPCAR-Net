# Transferred from methods/parametric_methods/data.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from vessel_code.parametric.representation import normalize_centerline_prediction_mode, normalize_lesion_profile, normalize_radius_prediction_mode
from vessel_code.shared.data import PrecomputedFeatureDataset, collate_precomputed_batches, discover_npz_files, ensure_item_images, load_precomputed_items, normalize_feature_backbone, resolve_clinical_rca_view_indices, resolve_lazy_load_image_features, resolve_target_coordinate_frame, resolve_zero_cached_image_features
from vessel_code.shared.model import ABSOLUTE_PARALLEL_DECODER, MAIN_FIRST_HIERARCHICAL_DECODER, normalize_decoder_architecture

PARAMETRIC_BATCH_KEYS = (
    "target_centerline_parameters_mm",
    "target_radius_baseline_coefficients_log_mm",
    "target_lesion_exist",
    "target_lesion_geometry",
    "target_attachment_index",
    "target_branch_exist",
    "target_branch_existence_supervision_mask",
    "target_raw_vessel_mm",
    "target_reconstructed_vessel_mm",
    "target_point_valid_mask",
    "parametric_centering_offset_mm",
)

PARAMETRIC_OPTIONAL_BATCH_KEYS = (
    "target_centerline_control_points_mm",
    "target_centerline_landmarks_mm",
)

PARAMETRIC_METADATA_KEYS = (
    "target_centerline_parameter_frame",
    "parametric_centering_source",
    "parametric_target_branch_mask_source",
)

ORIGINAL_COUNTERFACTUAL_FILENAME = "original.npz"

BRANCH_SUBSET_SCHEMA_VERSION = 1

STAGE4_1_VISIBILITY_SCHEMA_VERSION = 1

BRANCH_VISIBILITY_POLICIES = frozenset(
    {"best_effort", "error", "skip_variant"}
)

@dataclass(frozen=True)
class BranchVariantFileGroup:
    """One physical case and all of its Stage-4/5 branch-subset variants."""

    case_id: str
    group_id: str
    stage: str
    variants: tuple[str, ...]
    paths: tuple[Path, ...]
    full_branch_exists: tuple[bool, ...]

@dataclass(frozen=True)
class BranchVisibilitySamplingSettings:
    """Resolved opt-in Stage-4.1 view-sampling configuration."""

    enabled: bool
    metadata_json: Path | None
    unsatisfied_policy: str = "best_effort"
    apply_to_validation: bool = False

def resolve_branch_visibility_sampling(
    config: Mapping[str, Any],
) -> BranchVisibilitySamplingSettings:
    """Resolve the nested external Stage-4.1 visibility configuration."""

    raw = config.get("branch_visibility_sampling")
    if raw is None:
        return BranchVisibilitySamplingSettings(False, None)
    if not isinstance(raw, Mapping):
        raise ValueError(
            "branch_visibility_sampling must be a JSON object containing "
            "enabled, metadata_json, unsatisfied_policy, and optionally "
            "apply_to_validation."
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "branch_visibility_sampling.enabled must be a JSON boolean, got "
            f"{enabled!r}."
        )
    apply_to_validation = raw.get("apply_to_validation", False)
    if not isinstance(apply_to_validation, bool):
        raise ValueError(
            "branch_visibility_sampling.apply_to_validation must be a JSON "
            f"boolean, got {apply_to_validation!r}."
        )
    policy = str(raw.get("unsatisfied_policy", "best_effort")).strip().lower()
    if policy not in BRANCH_VISIBILITY_POLICIES:
        raise ValueError(
            "branch_visibility_sampling.unsatisfied_policy must be "
            "'best_effort', 'error', or 'skip_variant', got "
            f"{policy!r}."
        )
    metadata_value = raw.get("metadata_json")
    metadata_path = (
        None
        if metadata_value is None or not str(metadata_value).strip()
        else Path(str(metadata_value)).expanduser().resolve()
    )
    if enabled and metadata_path is None:
        raise ValueError(
            "branch_visibility_sampling.enabled=true requires a non-empty "
            "branch_visibility_sampling.metadata_json path."
        )
    if enabled and apply_to_validation and policy == "skip_variant":
        raise ValueError(
            "branch_visibility_sampling.apply_to_validation=true cannot be "
            "combined with unsatisfied_policy='skip_variant', because grouped "
            "validation must retain every variant. Use 'best_effort' or "
            "'error'."
        )
    return BranchVisibilitySamplingSettings(
        enabled=enabled,
        metadata_json=metadata_path,
        unsatisfied_policy=policy,
        apply_to_validation=apply_to_validation,
    )

def _json_boolean_list(value: Any, *, field: str) -> np.ndarray:
    if not isinstance(value, list) or not all(
        isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{field} must be a JSON array of booleans.")
    return np.asarray(value, dtype=bool)

def _json_unique_integer_list(value: Any, *, field: str) -> np.ndarray:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{field} must be a JSON array of integers.")
    array = np.asarray(value, dtype=np.int64)
    if np.any(array < 0) or np.unique(array).size != array.size:
        raise ValueError(f"{field} must contain unique non-negative integers.")
    return array

def _stage4_1_added_pixel_counts(
    record: Mapping[str, Any],
    *,
    num_views: int,
    field: str,
) -> np.ndarray:
    """Extract threshold-independent added-pixel evidence for best effort.

    Stage 4.1 keeps per-view audit metrics.  Accept the canonical list of
    metric objects and a direct numeric array for forwards compatibility.
    """

    direct_keys = (
        "added_foreground_pixels_by_view",
        "added_foreground_pixel_counts",
        "added_pixel_counts",
        "new_foreground_pixel_counts",
    )
    for key in direct_keys:
        if key in record:
            value = np.asarray(record[key], dtype=np.float64).reshape(-1)
            if value.shape != (num_views,):
                raise ValueError(
                    f"{field}.{key} must have length {num_views}, got "
                    f"{value.shape}."
                )
            if not np.isfinite(value).all() or np.any(value < 0.0):
                raise ValueError(f"{field}.{key} must be finite and non-negative.")
            return value
    metrics_key = "views" if "views" in record else "per_view_metrics"
    metrics = record.get(metrics_key)
    if metrics is None:
        return np.zeros((num_views,), dtype=np.float64)
    if not isinstance(metrics, list) or len(metrics) != num_views or not all(
        isinstance(metric, Mapping) for metric in metrics
    ):
        raise ValueError(
            f"{field}.{metrics_key} must contain {num_views} JSON objects."
        )
    metric_keys = (
        "added_foreground_pixels",
        "added_pixel_count",
        "new_foreground_pixels",
    )
    values: list[float] = []
    for view_index, metric in enumerate(metrics):
        raw_value = next(
            (metric[key] for key in metric_keys if key in metric),
            0.0,
        )
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field}.{metrics_key}[{view_index}] added-pixel value "
                "must be numeric."
            ) from error
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                f"{field}.{metrics_key}[{view_index}] added-pixel value "
                "must be finite and non-negative."
            )
        values.append(parsed)
    return np.asarray(values, dtype=np.float64)

def load_stage4_1_visibility_metadata(
    path: str | Path,
    *,
    num_branches: int,
) -> dict[str, Any]:
    """Load and strictly validate the external Stage-4.1 JSON contract."""

    source = Path(path).expanduser().resolve()
    return _load_stage4_1_visibility_metadata_cached(
        str(source), int(num_branches)
    )

@lru_cache(maxsize=None)
def _load_stage4_1_visibility_metadata_cached(
    source_text: str,
    num_branches: int,
) -> dict[str, Any]:
    source = Path(source_text)
    if not source.is_file():
        raise FileNotFoundError(
            f"Stage-4.1 branch visibility JSON does not exist: {source}"
        )
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Stage-4.1 visibility JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stage-4.1 visibility JSON must contain an object: {source}")
    schema = payload.get("schema_version")
    if isinstance(schema, bool) or schema != STAGE4_1_VISIBILITY_SCHEMA_VERSION:
        raise ValueError(
            f"{source} schema_version must be "
            f"{STAGE4_1_VISIBILITY_SCHEMA_VERSION}, got {schema!r}."
        )
    if str(payload.get("stage", "")).strip().lower() != "stage4.1":
        raise ValueError(f"{source} stage must be 'stage4.1'.")
    if str(payload.get("source_stage", "")).strip().lower() != "stage4":
        raise ValueError(f"{source} source_stage must be 'stage4'.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping) or not raw_cases:
        raise ValueError(f"{source} cases must be a non-empty JSON object.")

    cases: dict[str, dict[str, Any]] = {}
    for raw_group_id, raw_case in raw_cases.items():
        group_id = str(raw_group_id).strip()
        field = f"{source}:cases[{group_id!r}]"
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"{field} must be a JSON object.")
        stored_group_id = str(raw_case.get("group_id", "")).strip()
        if stored_group_id != group_id:
            raise ValueError(
                f"{field}.group_id={stored_group_id!r} must match its key."
            )
        vessel_type = str(raw_case.get("vessel_type", "")).strip().lower()
        if vessel_type not in {"rca", "lca"}:
            raise ValueError(f"{field}.vessel_type must be 'rca' or 'lca'.")
        case_id = str(raw_case.get("case_id", "")).strip()
        if group_id != f"{vessel_type}:{case_id}":
            raise ValueError(
                f"{field} must use group_id {vessel_type}:{case_id}."
            )
        raw_num_views = raw_case.get("num_views")
        if (
            isinstance(raw_num_views, bool)
            or not isinstance(raw_num_views, int)
            or raw_num_views < 1
        ):
            raise ValueError(f"{field}.num_views must be a positive integer.")
        num_views = int(raw_num_views)
        branch_capacity = raw_case.get("branch_capacity")
        if (
            isinstance(branch_capacity, bool)
            or not isinstance(branch_capacity, int)
            or branch_capacity != int(num_branches)
        ):
            raise ValueError(
                f"{field}.branch_capacity must equal configured num_branches="
                f"{num_branches}."
            )
        full_branch_exists = _json_boolean_list(
            raw_case.get("full_branch_exists"),
            field=f"{field}.full_branch_exists",
        )
        if full_branch_exists.shape != (int(num_branches),):
            raise ValueError(
                f"{field}.full_branch_exists must have length {num_branches}."
            )
        required_main = np.asarray(
            (0,) if vessel_type == "rca" else (0, 1), dtype=np.int64
        )
        stored_required_main = _json_unique_integer_list(
            raw_case.get("required_main_branch_indices"),
            field=f"{field}.required_main_branch_indices",
        )
        if not np.array_equal(stored_required_main, required_main):
            raise ValueError(
                f"{field}.required_main_branch_indices must equal "
                f"{required_main.tolist()} for {vessel_type.upper()}."
            )
        case_view_indices = _json_unique_integer_list(
            raw_case.get("view_indices"),
            field=f"{field}.view_indices",
        )
        if case_view_indices.shape != (num_views,):
            raise ValueError(f"{field}.view_indices must have length {num_views}.")
        raw_variants = raw_case.get("variants")
        if not isinstance(raw_variants, Mapping) or not raw_variants:
            raise ValueError(f"{field}.variants must be a non-empty JSON object.")

        variants: dict[str, dict[str, Any]] = {}
        common_view_indices: np.ndarray | None = None
        for raw_variant, raw_record in raw_variants.items():
            variant = str(raw_variant).strip()
            record_field = f"{field}.variants[{variant!r}]"
            if not variant or not isinstance(raw_record, Mapping):
                raise ValueError(f"{record_field} must be a JSON object.")
            mask = _json_boolean_list(
                raw_record.get("training_branch_mask"),
                field=f"{record_field}.training_branch_mask",
            )
            if mask.shape != (int(num_branches),):
                raise ValueError(
                    f"{record_field}.training_branch_mask must have length "
                    f"{num_branches}, got {mask.shape}."
                )
            if bool(np.any(mask & ~full_branch_exists)) or not bool(
                mask[required_main].all()
            ):
                raise ValueError(
                    f"{record_field}.training_branch_mask must select only "
                    "anatomically existing slots and retain every fixed main "
                    "branch."
                )
            selected = _json_unique_integer_list(
                raw_record.get("selected_branch_indices"),
                field=f"{record_field}.selected_branch_indices",
            )
            if not np.array_equal(selected, np.flatnonzero(mask)):
                raise ValueError(
                    f"{record_field}.selected_branch_indices must equal the "
                    "true positions in training_branch_mask."
                )
            view_indices = _json_unique_integer_list(
                raw_record.get("view_indices"),
                field=f"{record_field}.view_indices",
            )
            if view_indices.shape != (num_views,):
                raise ValueError(
                    f"{record_field}.view_indices must have length {num_views}."
                )
            if common_view_indices is None:
                common_view_indices = view_indices
            elif not np.array_equal(common_view_indices, view_indices):
                raise ValueError(
                    f"{field} variants must use identical view_indices."
                )
            visibility = _json_boolean_list(
                raw_record.get("view_visibility"),
                field=f"{record_field}.view_visibility",
            )
            if visibility.shape != (num_views,):
                raise ValueError(
                    f"{record_field}.view_visibility must have length {num_views}."
                )
            visible_views = _json_unique_integer_list(
                raw_record.get("visible_view_indices"),
                field=f"{record_field}.visible_view_indices",
            )
            expected_visible_views = view_indices[visibility]
            if not np.array_equal(visible_views, expected_visible_views):
                raise ValueError(
                    f"{record_field}.visible_view_indices must equal the "
                    "view_indices positions whose view_visibility is true."
                )
            raw_visible_count = raw_record.get("num_visible_views")
            if (
                isinstance(raw_visible_count, bool)
                or not isinstance(raw_visible_count, int)
                or raw_visible_count != int(visibility.sum())
            ):
                raise ValueError(
                    f"{record_field}.num_visible_views must equal "
                    f"{int(visibility.sum())}."
                )
            raw_added = raw_record.get("added_branch_index")
            if raw_added is None:
                added_branch_index = None
            elif (
                isinstance(raw_added, bool)
                or not isinstance(raw_added, int)
                or raw_added < 0
                or raw_added >= int(num_branches)
            ):
                raise ValueError(
                    f"{record_field}.added_branch_index must be null or a "
                    f"branch index in [0, {num_branches})."
                )
            else:
                added_branch_index = int(raw_added)
            predecessor_value = raw_record.get("predecessor_variant")
            predecessor = (
                None
                if predecessor_value is None
                else str(predecessor_value).strip()
            )
            added_pixels = _stage4_1_added_pixel_counts(
                raw_record,
                num_views=num_views,
                field=record_field,
            )
            audit_rows = raw_record.get(
                "views", raw_record.get("per_view_metrics")
            )
            if audit_rows is not None:
                baseline_without_comparison = (
                    added_branch_index is None and audit_rows == []
                )
                if not baseline_without_comparison and (
                    not isinstance(audit_rows, list)
                    or len(audit_rows) != num_views
                    or not all(isinstance(row, Mapping) for row in audit_rows)
                ):
                    raise ValueError(
                        f"{record_field} per-view audit metadata must contain "
                        f"exactly {num_views} JSON objects."
                    )
                for position, audit_row in enumerate(audit_rows):
                    if "view_index" in audit_row and audit_row["view_index"] != int(
                        view_indices[position]
                    ):
                        raise ValueError(
                            f"{record_field} audit row {position} view_index "
                            "does not match view_indices."
                        )
                    if "visible" in audit_row and audit_row["visible"] is not bool(
                        visibility[position]
                    ):
                        raise ValueError(
                            f"{record_field} audit row {position} visible flag "
                            "does not match view_visibility."
                        )
                    if "added_foreground_pixels" in audit_row:
                        audit_added = audit_row["added_foreground_pixels"]
                        if (
                            isinstance(audit_added, bool)
                            or not isinstance(audit_added, (int, float))
                            or not np.isfinite(float(audit_added))
                            or float(audit_added) < 0.0
                            or not np.isclose(
                                float(audit_added),
                                float(added_pixels[position]),
                                rtol=0.0,
                                atol=0.0,
                            )
                        ):
                            raise ValueError(
                                f"{record_field} audit row {position} "
                                "added_foreground_pixels does not match the "
                                "canonical per-view added-pixel counts."
                            )
            variants[variant] = {
                "training_branch_mask": mask,
                "selected_branch_indices": selected,
                "predecessor_variant": predecessor,
                "added_branch_index": added_branch_index,
                "view_indices": view_indices,
                "view_visibility": visibility,
                "visible_view_indices": visible_views,
                "added_pixel_counts": added_pixels,
            }

        baseline_count = 0
        baseline_variant: str | None = None
        successors: dict[str, list[str]] = {variant: [] for variant in variants}
        branch_records: dict[int, dict[str, Any]] = {}
        for variant, record in variants.items():
            predecessor = record["predecessor_variant"]
            added_branch = record["added_branch_index"]
            if predecessor is None:
                baseline_count += 1
                baseline_variant = variant
                if (
                    added_branch is not None
                    or bool(record["view_visibility"].any())
                    or bool(np.any(record["added_pixel_counts"] != 0.0))
                ):
                    raise ValueError(
                        f"{field}.variants[{variant!r}] baseline must have "
                        "added_branch_index=null, all-false view_visibility, "
                        "and zero added-pixel counts."
                    )
                continue
            if predecessor not in variants or predecessor == variant:
                raise ValueError(
                    f"{field}.variants[{variant!r}].predecessor_variant must "
                    "name another variant in the same case."
                )
            successors[predecessor].append(variant)
            previous_mask = variants[predecessor]["training_branch_mask"]
            current_mask = record["training_branch_mask"]
            added = np.flatnonzero(current_mask & ~previous_mask)
            removed = np.flatnonzero(previous_mask & ~current_mask)
            if removed.size or added.size != 1 or added_branch != int(added[0]):
                raise ValueError(
                    f"{field}.variants[{variant!r}] must add exactly its "
                    "added_branch_index relative to predecessor_variant and "
                    "must not remove a branch."
                )
            if added_branch in branch_records:
                raise ValueError(
                    f"{field} introduces branch {added_branch} more than once."
                )
            branch_records[int(added_branch)] = record
        if baseline_count != 1:
            raise ValueError(f"{field} must contain exactly one baseline variant.")
        assert common_view_indices is not None
        if not np.array_equal(common_view_indices, case_view_indices):
            raise ValueError(
                f"{field}.view_indices must match every variant's view_indices."
            )
        if any(len(values) > 1 for values in successors.values()):
            raise ValueError(
                f"{field} variants must form one progressive prefix chain, "
                "not a fork."
            )
        assert baseline_variant is not None
        expected_baseline = np.zeros((int(num_branches),), dtype=bool)
        expected_baseline[required_main] = True
        if not np.array_equal(
            variants[baseline_variant]["training_branch_mask"], expected_baseline
        ):
            raise ValueError(
                f"{field} baseline variant must contain exactly the fixed main "
                f"slots {required_main.tolist()}."
            )
        visited: set[str] = set()
        cursor: str | None = baseline_variant
        terminal_variant = baseline_variant
        while cursor is not None:
            if cursor in visited:
                raise ValueError(f"{field} predecessor chain contains a cycle.")
            visited.add(cursor)
            terminal_variant = cursor
            next_values = successors[cursor]
            cursor = next_values[0] if next_values else None
        if visited != set(variants):
            raise ValueError(
                f"{field} variants must form one connected progressive prefix chain."
            )
        if not np.array_equal(
            variants[terminal_variant]["training_branch_mask"],
            full_branch_exists,
        ):
            raise ValueError(
                f"{field} final progressive variant must contain every branch "
                "in full_branch_exists."
            )
        cases[group_id] = {
            "case_id": case_id,
            "group_id": group_id,
            "vessel_type": vessel_type,
            "num_views": num_views,
            "view_indices": np.asarray(common_view_indices, dtype=np.int64),
            "variants": variants,
            "branch_records": branch_records,
        }

    return {
        "path": source,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "schema_version": STAGE4_1_VISIBILITY_SCHEMA_VERSION,
        "stage": "stage4.1",
        "cases": cases,
    }

def resolve_branch_variant_group_training(config: Mapping[str, Any]) -> bool:
    value = config.get("branch_variant_group_training", False)
    if not isinstance(value, bool):
        raise ValueError(
            "branch_variant_group_training must be a JSON boolean, got "
            f"{value!r}."
        )
    return value

def _branch_variant_scalar(
    payload: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> Any:
    if key not in payload.files:
        raise KeyError(f"{path} is missing required branch-subset field {key!r}.")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(
            f"{path} field {key!r} must be scalar, got shape {value.shape}."
        )
    return value.reshape(()).item()

def load_branch_variant_metadata(
    path: str | Path,
    *,
    num_branches: int,
) -> dict[str, Any]:
    """Load and strictly validate the Stage-4/5 branch-subset contract."""

    source = Path(path).expanduser().resolve()
    return _load_branch_variant_metadata_cached(str(source), int(num_branches))

@lru_cache(maxsize=None)
def _load_branch_variant_metadata_cached(
    source_text: str,
    num_branches: int,
) -> dict[str, Any]:
    # Stage datasets are immutable during a training process. Caching avoids
    # reopening every (potentially large) NPZ once for each validation view
    # count while retaining the same strict checks.
    source = Path(source_text)
    required = (
        "branch_subset_schema_version",
        "branch_subset_stage",
        "branch_subset_group_id",
        "branch_subset_variant",
        "full_branch_exists",
        "training_branch_mask",
        "branch_exists",
        "selected_branch_indices",
        "projected_branch_indices",
        "num_projected_branches",
    )
    with np.load(source, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(
                f"{source} is missing required Stage-4/5 branch-subset fields: "
                f"{missing}."
            )
        schema = int(
            _branch_variant_scalar(
                payload, "branch_subset_schema_version", source
            )
        )
        if schema != BRANCH_SUBSET_SCHEMA_VERSION:
            raise ValueError(
                f"{source} stores branch_subset_schema_version={schema}; "
                f"expected {BRANCH_SUBSET_SCHEMA_VERSION}."
            )
        stage = str(
            _branch_variant_scalar(payload, "branch_subset_stage", source)
        ).strip().lower()
        if stage not in {"stage4", "stage5"}:
            raise ValueError(
                f"{source} branch_subset_stage must be 'stage4' or 'stage5', "
                f"got {stage!r}."
            )
        group_id = str(
            _branch_variant_scalar(payload, "branch_subset_group_id", source)
        ).strip()
        variant = str(
            _branch_variant_scalar(payload, "branch_subset_variant", source)
        ).strip()
        if not group_id or not variant:
            raise ValueError(
                f"{source} branch_subset_group_id and branch_subset_variant "
                "must be non-empty."
            )
        full = np.asarray(payload["full_branch_exists"], dtype=bool).reshape(-1)
        mask = np.asarray(payload["training_branch_mask"], dtype=bool).reshape(-1)
        exists = np.asarray(payload["branch_exists"], dtype=bool).reshape(-1)
        expected_shape = (int(num_branches),)
        for key, value in (
            ("full_branch_exists", full),
            ("training_branch_mask", mask),
            ("branch_exists", exists),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"{source} {key} has shape {value.shape}; expected "
                    f"{expected_shape}."
                )
        expected_exists = full & mask
        if bool(np.any(mask & ~full)):
            raise ValueError(
                f"{source} training_branch_mask selects anatomically unavailable "
                "slots from full_branch_exists."
            )
        if not np.array_equal(exists, expected_exists):
            raise ValueError(
                f"{source} branch_exists must equal full_branch_exists & "
                "training_branch_mask."
            )
        selected = np.asarray(
            payload["selected_branch_indices"], dtype=np.int64
        ).reshape(-1)
        projected = np.asarray(
            payload["projected_branch_indices"], dtype=np.int64
        ).reshape(-1)
        active_indices = np.flatnonzero(expected_exists).astype(np.int64)
        for key, indices in (
            ("selected_branch_indices", selected),
            ("projected_branch_indices", projected),
        ):
            if (
                np.any(indices < 0)
                or np.any(indices >= int(num_branches))
                or len(set(indices.tolist())) != int(indices.size)
                or set(indices.tolist()) != set(active_indices.tolist())
            ):
                raise ValueError(
                    f"{source} {key}={indices.tolist()} must contain each "
                    "active branch index exactly once."
                )
        projected_count = int(
            _branch_variant_scalar(payload, "num_projected_branches", source)
        )
        if projected_count != int(projected.size):
            raise ValueError(
                f"{source} num_projected_branches={projected_count}, but "
                f"projected_branch_indices contains {projected.size} entries."
            )
        if "point_valid_mask" in payload.files:
            point_valid = np.asarray(payload["point_valid_mask"], dtype=bool)
            if point_valid.shape[0] != int(num_branches):
                raise ValueError(
                    f"{source} point_valid_mask has incompatible shape "
                    f"{point_valid.shape}."
                )
            if bool(point_valid[~expected_exists].any()):
                raise ValueError(
                    f"{source} has valid vessel points on branches excluded by "
                    "training_branch_mask."
                )
        zero_masked_keys = (
            "raw_vessel_code_mm",
            "reconstructed_vessel_code_mm",
            "centerline_control_points_mm",
            "centerline_landmarks_mm",
            "radius_baseline_coefficients_log_mm",
            "radius_lesion_parameters",
            "artery",
        )
        for key in zero_masked_keys:
            if key not in payload.files:
                continue
            value = np.asarray(payload[key])
            if value.ndim < 1 or value.shape[0] != int(num_branches):
                raise ValueError(
                    f"{source} {key} has incompatible branch dimension "
                    f"{value.shape}."
                )
            inactive = value[~expected_exists]
            if inactive.size and (
                not np.isfinite(inactive).all()
                or not np.allclose(inactive, 0.0, rtol=0.0, atol=1e-7)
            ):
                raise ValueError(
                    f"{source} {key} must be finite and zero on branches "
                    "excluded by training_branch_mask."
                )
        if "side_branch_attachment_index" in payload.files:
            attachment = np.asarray(
                payload["side_branch_attachment_index"], dtype=np.int64
            ).reshape(-1)
            if attachment.shape != expected_shape or bool(
                np.any(attachment[~expected_exists] != -1)
            ):
                raise ValueError(
                    f"{source} side_branch_attachment_index must be -1 on "
                    "excluded branches."
                )
    return {
        "path": source,
        "stage": stage,
        "group_id": group_id,
        "variant": variant,
        "full_branch_exists": full,
        "training_branch_mask": mask,
        "branch_exists": exists,
        "selected_branch_indices": selected,
        "projected_branch_indices": projected,
        "num_projected_branches": projected_count,
    }

def discover_branch_variant_groups(
    files: Sequence[Path],
    *,
    num_branches: int,
) -> list[BranchVariantFileGroup]:
    """Group validated variant files by physical numeric case directory."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_path in files:
        path = Path(raw_path).expanduser().resolve()
        if not path.parent.name.isdigit():
            raise ValueError(
                "Grouped branch-variant training requires layout "
                "<root>/<rca|lca>/<numeric_case>/<variant>.npz; got "
                f"{path}."
            )
        case_id = case_identifier(path)
        vessel_type = path.parent.parent.name.strip().lower()
        if vessel_type not in {"rca", "lca"}:
            raise ValueError(
                "Grouped branch-variant training requires the numeric case "
                f"directory to be below rca/ or lca/; got {path}."
            )
        metadata = load_branch_variant_metadata(
            path, num_branches=num_branches
        )
        expected_group_id = f"{vessel_type}:{case_id}"
        if str(metadata["group_id"]) != expected_group_id:
            raise ValueError(
                f"{path} branch_subset_group_id={metadata['group_id']!r}; "
                f"expected {expected_group_id!r} from its directory layout."
            )
        required_main_count = 1 if vessel_type == "rca" else 2
        active = np.asarray(metadata["branch_exists"], dtype=bool)
        if not bool(active[:required_main_count].all()):
            raise ValueError(
                f"{path} must retain the first {required_main_count} fixed "
                f"{vessel_type.upper()} main-branch slot(s)."
            )
        grouped.setdefault(case_id, []).append(
            metadata
        )
    output: list[BranchVariantFileGroup] = []
    for case_id, records in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        group_ids = {str(record["group_id"]) for record in records}
        stages = {str(record["stage"]) for record in records}
        variants = [str(record["variant"]) for record in records]
        if len(group_ids) != 1:
            raise ValueError(
                f"Case {case_id} contains multiple branch_subset_group_id "
                f"values: {sorted(group_ids)}."
            )
        if len(stages) != 1:
            raise ValueError(
                f"Case {case_id} mixes Stage-4 and Stage-5 variants: "
                f"{sorted(stages)}."
            )
        if len(set(variants)) != len(variants):
            raise ValueError(
                f"Case {case_id} repeats branch_subset_variant names: {variants}."
            )
        reference_full = np.asarray(records[0]["full_branch_exists"], dtype=bool)
        if any(
            not np.array_equal(
                np.asarray(record["full_branch_exists"], dtype=bool),
                reference_full,
            )
            for record in records[1:]
        ):
            raise ValueError(
                f"Case {case_id} variants disagree on full_branch_exists."
            )
        ordered = sorted(
            records,
            key=lambda record: (
                str(record["variant"]),
                str(record["path"]),
            ),
        )
        output.append(
            BranchVariantFileGroup(
                case_id=case_id,
                group_id=next(iter(group_ids)),
                stage=next(iter(stages)),
                variants=tuple(str(record["variant"]) for record in ordered),
                paths=tuple(Path(record["path"]) for record in ordered),
                full_branch_exists=tuple(bool(value) for value in reference_full),
            )
        )
    if not output:
        raise ValueError("Grouped branch-variant discovery produced no case groups.")
    return output

def resolve_feature_dataset_dir(
    config: dict[str, Any], feature_backbone: str | None = None
) -> Path:
    """Resolve a shared feature root to its backbone-specific cached-feature directory."""
    backbone = normalize_feature_backbone(
        feature_backbone or config.get("feature_backbone", "vggt")
    )
    raw = config.get("feature_dataset_dir")
    if raw is None:
        raise ValueError("Missing feature_dataset_dir in config")
    root = Path(raw).expanduser()
    if not bool(config.get("auto_backbone_dataset_subdir", True)):
        return root.resolve()
    expected = {
        "resnet_pre_fpn": "resnet",
        "vggt": "vggt",
        "vggt_omega": "vggt_omega",
    }[backbone]
    known = {"resnet", "vggt", "vggt_omega"}
    if root.name == expected:
        return root.resolve()
    if root.name in known:
        return (root.parent / expected).resolve()
    return (root / expected).resolve()

def uses_online_backbone_features(config: dict[str, Any]) -> bool:
    """Return whether raw images, rather than cached features, are configured."""

    # Explicit null selects online mode.  An omitted key remains an error so a
    # misspelled feature-dataset setting cannot silently switch input modes.
    return "feature_dataset_dir" in config and config["feature_dataset_dir"] is None

def uses_online_vggt_features(config: dict[str, Any]) -> bool:
    """Return whether the configured online backbone is a VGGT variant."""

    return uses_online_backbone_features(config) and normalize_feature_backbone(
        config.get("feature_backbone", "vggt")
    ) in {"vggt", "vggt_omega"}

def resolve_raw_image_dataset_dir(config: dict[str, Any]) -> Path:
    """Resolve the required raw-image root for on-the-fly frozen features."""

    raw = config.get("raw_image_dataset_dir")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "feature_dataset_dir=null requires a non-null "
            "raw_image_dataset_dir in config."
        )
    return Path(str(raw)).expanduser().resolve()

def resolve_model_input_dataset_dir(
    config: dict[str, Any], feature_backbone: str | None = None
) -> Path:
    """Resolve either the cached-feature root or the online raw-image root."""

    if uses_online_backbone_features(config):
        backbone = normalize_feature_backbone(
            feature_backbone or config.get("feature_backbone", "vggt")
        )
        if backbone not in ("resnet_pre_fpn", "vggt", "vggt_omega"):
            raise ValueError(
                "feature_dataset_dir=null supports feature_backbone="
                "'resnet_pre_fpn', 'vggt', or 'vggt_omega'."
            )
        return resolve_raw_image_dataset_dir(config)
    return resolve_feature_dataset_dir(config, feature_backbone)

def case_identifier(path: str | Path) -> str:
    value = Path(path)
    if value.parent.name.isdigit():
        return str(int(value.parent.name))
    groups = re.findall(r"\d+", value.stem)
    if not groups:
        raise ValueError(f"Could not infer numeric case ID from {value}")
    return str(int(groups[-1]))

def resolve_parametric_original_variant_only(config: dict[str, Any]) -> bool:
    value = config.get("parametric_original_variant_only", False)
    if not isinstance(value, bool):
        raise ValueError(
            "parametric_original_variant_only must be a JSON boolean, got "
            f"{value!r}."
        )
    return value

def resolve_parametric_target_source(config: dict[str, Any]) -> str:
    value = str(config.get("parametric_target_source", "directory")).strip().lower()
    aliases = {
        "directory": "directory",
        "separate_directory": "directory",
        "feature_file": "feature_file",
        "feature_npz": "feature_file",
    }
    resolved = aliases.get(value)
    if resolved is None:
        raise ValueError(
            "parametric_target_source must be 'directory' or 'feature_file', "
            f"got {value!r}."
        )
    if resolved == "directory" and not config.get("parametric_target_dir"):
        raise ValueError(
            "parametric_target_source='directory' requires "
            "parametric_target_dir. Use 'feature_file' when each precomputed "
            "feature NPZ already stores the parametric targets."
        )
    return resolved

def resolve_parametric_target_branch_mask_source(
    config: Mapping[str, Any],
) -> str:
    """Resolve which source controls parametric branch-existence supervision."""

    raw_value = config.get(
        "parametric_target_branch_mask_source", "parametric_target"
    )
    value = str(raw_value).strip().lower().replace("-", "_")
    aliases = {
        "parametric_target": "parametric_target",
        "target": "parametric_target",
        "directory_target": "parametric_target",
        "feature": "feature",
        "feature_file": "feature",
        "feature_npz": "feature",
        "input": "feature",
    }
    resolved = aliases.get(value)
    if resolved is None:
        raise ValueError(
            "parametric_target_branch_mask_source must be "
            "'parametric_target' or 'feature', got "
            f"{raw_value!r}."
        )
    return resolved

def filter_original_parametric_variant_files(
    files: Sequence[Path],
    *,
    original_only: bool,
    source_description: str,
) -> tuple[list[Path], list[Path]]:
    """Keep one ``original.npz`` per case for ordinary parametric training."""

    paths = [Path(path) for path in files]
    if not original_only:
        return paths, []
    selected = [
        path
        for path in paths
        if path.name.casefold() == ORIGINAL_COUNTERFACTUAL_FILENAME
    ]
    ignored = [path for path in paths if path not in selected]
    if not selected:
        raise FileNotFoundError(
            f"{source_description} contains no files named "
            f"{ORIGINAL_COUNTERFACTUAL_FILENAME!r}, but "
            "parametric_original_variant_only=true."
        )
    by_case: dict[str, Path] = {}
    for path in selected:
        identifier = case_identifier(path)
        previous = by_case.get(identifier)
        if previous is not None:
            raise ValueError(
                f"{source_description} contains more than one original variant "
                f"for case {identifier}: {previous} and {path}."
            )
        by_case[identifier] = path
    return selected, ignored

def discover_parametric_targets(
    target_dir: str | Path,
    *,
    original_only: bool = False,
) -> dict[str, Path]:
    files = discover_npz_files(target_dir, recursive=True)
    if original_only:
        grouped: dict[str, list[Path]] = {}
        for path in files:
            grouped.setdefault(case_identifier(path), []).append(path)
        selected: list[Path] = []
        for identifier, candidates in sorted(
            grouped.items(), key=lambda item: int(item[0])
        ):
            named_original = [
                path
                for path in candidates
                if path.name.casefold() == ORIGINAL_COUNTERFACTUAL_FILENAME
            ]
            if len(named_original) == 1:
                selected.append(named_original[0])
                continue
            if len(named_original) > 1:
                raise ValueError(
                    "Parametric target directory contains more than one "
                    f"original.npz for case {identifier}: {named_original}."
                )
            if len(candidates) == 1:
                # Older B-spline target trees commonly use names such as
                # vessel_code_step6.npz.  A unique target is unambiguous even
                # though its basename does not encode the variant.
                selected.append(candidates[0])
                continue
            raise ValueError(
                "parametric_original_variant_only=true cannot choose a target "
                f"for case {identifier}: no original.npz exists and multiple "
                f"candidate targets were found: {candidates}."
            )
        files = selected
    mapping: dict[str, Path] = {}
    for path in files:
        identifier = case_identifier(path)
        if identifier in mapping:
            raise ValueError(
                f"Duplicate parametric target for case {identifier}: {mapping[identifier]} and {path}"
            )
        mapping[identifier] = path
    return mapping

def _pad_branches(array: np.ndarray, num_branches: int, fill_value: float = 0.0) -> np.ndarray:
    value = np.asarray(array)
    if value.shape[0] > int(num_branches):
        return value[: int(num_branches)].copy()
    if value.shape[0] == int(num_branches):
        return value.copy()
    output = np.full((int(num_branches), *value.shape[1:]), fill_value, dtype=value.dtype)
    output[: value.shape[0]] = value
    return output

def resolve_parametric_num_branches(
    config: Mapping[str, Any],
    *,
    default: int = 7,
) -> int:
    """Resolve the number of model branch queries/output slots.

    ``model.num_branches`` is the explicit architecture setting. The legacy
    top-level ``num_branches`` remains its fallback for existing configs.
    """

    model_config_raw = config.get("model", {})
    if model_config_raw is None:
        model_config_raw = {}
    if not isinstance(model_config_raw, Mapping):
        raise ValueError("model must be a JSON object when provided.")

    model_level_value = (
        int(model_config_raw["num_branches"])
        if "num_branches" in model_config_raw
        else None
    )
    resolved = (
        model_level_value
        if model_level_value is not None
        else int(config["num_branches"])
        if "num_branches" in config
        else int(default)
    )
    if resolved < 1:
        raise ValueError(
            f"model.num_branches must be >= 1, got {resolved}."
        )
    return resolved

def resolve_parametric_target_num_branches(
    config: Mapping[str, Any],
    *,
    model_num_branches: int | None = None,
) -> int:
    """Resolve how many leading branches have geometry targets.

    Model queries after this prefix still receive explicit zero branch-
    existence labels; only their geometry/radius/attachment targets are
    disabled.
    """

    model_capacity = (
        resolve_parametric_num_branches(config)
        if model_num_branches is None
        else int(model_num_branches)
    )
    resolved = int(config.get("num_branches", model_capacity))
    if resolved < 1:
        raise ValueError(f"num_branches must be >= 1, got {resolved}.")
    if resolved > model_capacity:
        raise ValueError(
            "The target branch count cannot exceed the model branch-query "
            f"capacity: num_branches={resolved}, "
            f"model.num_branches={model_capacity}."
        )
    return resolved

def load_parametric_target(
    path: str | Path,
    *,
    num_branches: int,
    num_points: int,
    num_control_points: int,
    num_radius_coefficients: int,
    num_lesions: int,
    lesion_profile: str,
    projection_center_offset_mm: np.ndarray | None = None,
    centerline_prediction_mode: str | None = "bspline_control_points",
    num_landmarks: int | None = None,
    radius_prediction_mode: str | None = "parametric",
    catmull_rom_alpha: float = 0.5,
    catmull_rom_dense_samples: int = 1000,
    absolute_centerline_parameters: bool = False,
) -> dict[str, np.ndarray]:
    target_path = Path(path)
    profile = normalize_lesion_profile(lesion_profile)
    centerline_mode = normalize_centerline_prediction_mode(
        centerline_prediction_mode
    )
    radius_mode = normalize_radius_prediction_mode(radius_prediction_mode)
    expected_centerline_count = int(
        num_control_points if num_landmarks is None else num_landmarks
    )
    if centerline_mode == "bspline_control_points":
        expected_centerline_count = int(num_control_points)
    with np.load(target_path, allow_pickle=True) as payload:
        required = [
            "raw_vessel_code_mm",
            "reconstructed_vessel_code_mm",
            "branch_exists",
            "point_valid_mask",
        ]
        if not absolute_centerline_parameters:
            required.append("side_branch_attachment_index")
        centerline_key = (
            "centerline_landmarks_mm"
            if centerline_mode == "adaptive_landmarks"
            else "centerline_control_points_mm"
        )
        required.append(centerline_key)
        if radius_mode == "parametric":
            required.extend(
                (
                    "radius_baseline_coefficients_log_mm",
                    "radius_lesion_parameters",
                )
            )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{target_path} is missing parametric target arrays: {missing}")
        if centerline_mode == "adaptive_landmarks":
            if "transform_type" in payload.files:
                transform_type = str(
                    np.asarray(payload["transform_type"]).reshape(()).item()
                )
                if transform_type != "hierarchical_centerline_catmull_rom":
                    raise ValueError(
                        f"{target_path} stores transform_type={transform_type!r}; "
                        "adaptive_landmarks requires a Catmull--Rom landmark transform"
                    )
            if "transform_config_json" in payload.files:
                stored_transform_config = json.loads(
                    str(
                        np.asarray(payload["transform_config_json"])
                        .reshape(())
                        .item()
                    )
                )
                stored_alpha = float(
                    stored_transform_config.get("catmull_rom_alpha", 0.5)
                )
                if not np.isclose(
                    stored_alpha, float(catmull_rom_alpha), atol=1e-8
                ):
                    raise ValueError(
                        f"{target_path} was decoded with catmull_rom_alpha="
                        f"{stored_alpha}, but training is configured with "
                        f"{catmull_rom_alpha}"
                    )
                stored_dense_samples = int(
                    stored_transform_config.get(
                        "adaptive_dense_samples", catmull_rom_dense_samples
                    )
                )
                if stored_dense_samples != int(catmull_rom_dense_samples):
                    raise ValueError(
                        f"{target_path} was decoded with adaptive_dense_samples="
                        f"{stored_dense_samples}, but training is configured with "
                        f"catmull_rom_dense_samples={catmull_rom_dense_samples}"
                    )
        if radius_mode == "parametric":
            stored_profile = (
                str(np.asarray(payload["radius_lesion_profile"]).reshape(()).item())
                if "radius_lesion_profile" in payload.files
                else "gaussian"
            )
            if normalize_lesion_profile(stored_profile) != profile:
                raise ValueError(
                    f"{target_path} stores lesion_profile={stored_profile!r}, expected {profile!r}"
                )
        centerline_parameters = np.asarray(payload[centerline_key], dtype=np.float32)
        if radius_mode == "parametric":
            radius_coefficients = np.asarray(
                payload["radius_baseline_coefficients_log_mm"], dtype=np.float32
            )
            lesions = np.asarray(
                payload["radius_lesion_parameters"], dtype=np.float32
            )
        else:
            stored_branches = int(centerline_parameters.shape[0])
            expected_geometry_dim = 3 if profile == "gaussian" else 5
            radius_coefficients = np.zeros(
                (stored_branches, int(num_radius_coefficients)), dtype=np.float32
            )
            lesions = np.zeros(
                (
                    stored_branches,
                    int(num_lesions),
                    expected_geometry_dim + 1,
                ),
                dtype=np.float32,
            )
        attachment_key = (
            "side_branch_attachment_output_index"
            if centerline_mode == "adaptive_landmarks"
            and "side_branch_attachment_output_index" in payload.files
            else "side_branch_attachment_index"
        )
        attachment = (
            np.asarray(payload[attachment_key], dtype=np.int64)
            if attachment_key in payload.files
            else np.full(
                (int(centerline_parameters.shape[0]),), -1, dtype=np.int64
            )
        )
        branch_exists = np.asarray(payload["branch_exists"], dtype=bool)
        point_valid = np.asarray(payload["point_valid_mask"], dtype=bool)
        source_raw_vessel = np.asarray(
            payload["raw_vessel_code_mm"], dtype=np.float32
        )
        raw_reference_key = (
            "uniform_arc_vessel_code_mm"
            if centerline_mode == "adaptive_landmarks"
            and "uniform_arc_vessel_code_mm" in payload.files
            else "raw_vessel_code_mm"
        )
        raw_vessel = np.asarray(payload[raw_reference_key], dtype=np.float32)
        reconstructed = np.asarray(
            payload["reconstructed_vessel_code_mm"], dtype=np.float32
        )
        # Alignment validation should measure the decoded parametric geometry,
        # not the transform's optional topology snap.  A relative side branch
        # is therefore translated back so its decoded first point matches its
        # raw first point.  Its B-spline shape and endpoint fit are preserved;
        # only the artificial attachment-to-branch-0 translation is removed.
        alignment_vessel = source_raw_vessel.copy()
        alignment_frame_key = (
            "centerline_landmark_frame"
            if centerline_mode == "adaptive_landmarks"
            else "centerline_control_point_frame"
        )
        alignment_reference = "source_raw_fallback_missing_parameter_frame"
        if alignment_frame_key in payload.files:
            alignment_frames = np.asarray(payload[alignment_frame_key]).astype(str)
            stored_branches = int(centerline_parameters.shape[0])
            if alignment_frames.shape != (stored_branches,):
                raise ValueError(
                    f"{target_path} {alignment_frame_key} has shape "
                    f"{alignment_frames.shape}; expected ({stored_branches},)"
                )
            has_active_relative_frame = any(
                bool(branch_exists[branch_index])
                and str(alignment_frames[branch_index]).strip().lower()
                == "parent_attachment_relative_mm"
                for branch_index in range(stored_branches)
            )
            alignment_reference = (
                "decoded_without_attachment_translation"
                if has_active_relative_frame
                else "decoded_global"
            )
            alignment_vessel = reconstructed.copy()
            for branch_index in range(stored_branches):
                if not bool(branch_exists[branch_index]):
                    continue
                frame = str(alignment_frames[branch_index]).strip().lower()
                if frame == "global_mm":
                    continue
                if frame != "parent_attachment_relative_mm":
                    raise ValueError(
                        f"{target_path} has unsupported active "
                        f"{alignment_frame_key} value {frame!r} for branch "
                        f"{branch_index}."
                    )
                valid_points = point_valid[branch_index]
                valid_indices = np.flatnonzero(valid_points)
                if not valid_indices.size:
                    continue
                first_index = int(valid_indices[0])
                translation = (
                    source_raw_vessel[branch_index, first_index, :3]
                    - alignment_vessel[branch_index, first_index, :3]
                )
                alignment_vessel[branch_index, valid_points, :3] += (
                    translation.reshape(1, 3)
                )
            if not absolute_centerline_parameters:
                global_side_branches = [
                    branch_index
                    for branch_index in range(1, stored_branches)
                    if bool(branch_exists[branch_index])
                    and str(alignment_frames[branch_index]).strip().lower()
                    == "global_mm"
                ]
                if global_side_branches:
                    raise ValueError(
                        f"{target_path} preserves global side-branch controls "
                        f"for active branches {global_side_branches}. Use "
                        "model.decoder_architecture='absolute_parallel'; an "
                        "attachment-based decoder would move their raw start "
                        "and end coordinates onto branch 0."
                    )
        if absolute_centerline_parameters:
            frame_key = (
                "centerline_landmark_frame"
                if centerline_mode == "adaptive_landmarks"
                else "centerline_control_point_frame"
            )
            if frame_key not in payload.files:
                raise KeyError(
                    f"{target_path} cannot provide absolute centreline "
                    f"parameters; missing array: {frame_key!r}"
                )
            parameter_frames = np.asarray(payload[frame_key]).astype(str)
            stored_branches = int(centerline_parameters.shape[0])
            if parameter_frames.shape != (stored_branches,):
                raise ValueError(
                    f"{target_path} {frame_key} has shape "
                    f"{parameter_frames.shape}; expected ({stored_branches},)"
                )
            active_frames = {
                str(parameter_frames[index]).strip().lower()
                for index in range(stored_branches)
                if bool(branch_exists[index])
            }
            supported_frames = {
                "global_mm",
                "parent_attachment_relative_mm",
            }
            unsupported_frames = sorted(active_frames - supported_frames)
            if unsupported_frames:
                raise ValueError(
                    f"{target_path} has unsupported active {frame_key} values "
                    f"for decoder_architecture='absolute_parallel': "
                    f"{unsupported_frames}"
                )
            has_relative_parameters = (
                "parent_attachment_relative_mm" in active_frames
            )
            if has_relative_parameters:
                origin_keys = (
                    "raw_branch_origin_mm",
                    "decoded_branch_origin_mm",
                )
                missing_origins = [
                    key for key in origin_keys if key not in payload.files
                ]
                if missing_origins:
                    raise KeyError(
                        f"{target_path} cannot globalize relative centreline "
                        f"parameters; missing arrays: {missing_origins}"
                    )
                raw_branch_origins = np.asarray(
                    payload["raw_branch_origin_mm"], dtype=np.float32
                )
                decoded_branch_origins = np.asarray(
                    payload["decoded_branch_origin_mm"], dtype=np.float32
                )
                for origin_name, origins in (
                    ("raw_branch_origin_mm", raw_branch_origins),
                    ("decoded_branch_origin_mm", decoded_branch_origins),
                ):
                    if origins.shape != (stored_branches, 3):
                        raise ValueError(
                            f"{target_path} {origin_name} has shape "
                            f"{origins.shape}; expected ({stored_branches}, 3)"
                        )
            else:
                raw_branch_origins = np.empty((stored_branches, 3))
                decoded_branch_origins = np.empty((stored_branches, 3))
            centerline_parameters = centerline_parameters.copy()
            reconstructed = reconstructed.copy()
            for branch_index in range(stored_branches):
                if not bool(branch_exists[branch_index]):
                    continue
                frame = str(parameter_frames[branch_index]).strip().lower()
                if frame == "global_mm":
                    continue
                assert frame == "parent_attachment_relative_mm"
                raw_origin = raw_branch_origins[branch_index]
                decoded_origin = decoded_branch_origins[branch_index]
                if not (
                    np.isfinite(raw_origin).all()
                    and np.isfinite(decoded_origin).all()
                ):
                    raise ValueError(
                        f"{target_path} branch {branch_index} has non-finite "
                        "raw/decoded origins required for absolute conversion."
                    )
                centerline_parameters[branch_index] += raw_origin.reshape(1, 3)
                valid_points = point_valid[branch_index]
                reconstructed[branch_index, valid_points, :3] += (
                    raw_origin - decoded_origin
                ).reshape(1, 3)

    expected_geometry_dim = 3 if profile == "gaussian" else 5
    if centerline_parameters.shape[1:] != (expected_centerline_count, 3):
        raise ValueError(
            f"{target_path} {centerline_key} has shape "
            f"{centerline_parameters.shape}; expected "
            f"[M,{expected_centerline_count},3]"
        )
    if radius_coefficients.shape[1:] != (int(num_radius_coefficients),):
        raise ValueError(
            f"{target_path} radius coefficients have shape {radius_coefficients.shape}; "
            f"expected [M,{num_radius_coefficients}]"
        )
    if lesions.shape[1:] != (int(num_lesions), expected_geometry_dim + 1):
        raise ValueError(
            f"{target_path} lesions have shape {lesions.shape}; expected "
            f"[M,{num_lesions},{expected_geometry_dim + 1}]"
        )
    if raw_vessel.shape[1:] != (int(num_points), 4):
        raise ValueError(
            f"{target_path} vessel has shape {raw_vessel.shape}; expected [M,{num_points},4]"
        )

    centerline_parameters = _pad_branches(
        centerline_parameters, num_branches
    )
    radius_coefficients = _pad_branches(radius_coefficients, num_branches)
    lesions = _pad_branches(lesions, num_branches)
    attachment = _pad_branches(attachment, num_branches, fill_value=-1)
    branch_exists = _pad_branches(branch_exists, num_branches)
    point_valid = _pad_branches(point_valid, num_branches)
    raw_vessel = _pad_branches(raw_vessel, num_branches)
    source_raw_vessel = _pad_branches(source_raw_vessel, num_branches)
    alignment_vessel = _pad_branches(alignment_vessel, num_branches)
    reconstructed = _pad_branches(reconstructed, num_branches)
    centerline_parameters = np.nan_to_num(
        centerline_parameters, nan=0.0, posinf=0.0, neginf=0.0
    )
    radius_coefficients = np.nan_to_num(
        radius_coefficients, nan=0.0, posinf=0.0, neginf=0.0
    )

    if projection_center_offset_mm is not None:
        offset = np.asarray(projection_center_offset_mm, dtype=np.float32).reshape(3)
        parameter_branch_mask = (
            branch_exists
            if absolute_centerline_parameters
            else np.arange(int(num_branches)) == 0
        )
        centerline_parameters[
            parameter_branch_mask & branch_exists
        ] -= offset.reshape(1, 1, 3)
        raw_vessel[branch_exists, :, :3] -= offset.reshape(1, 1, 3)
        source_raw_vessel[branch_exists, :, :3] -= offset.reshape(1, 1, 3)
        alignment_vessel[branch_exists, :, :3] -= offset.reshape(1, 1, 3)
        reconstructed[branch_exists, :, :3] -= offset.reshape(1, 1, 3)

    attachment = np.clip(attachment, -1, int(num_points) - 1)
    if absolute_centerline_parameters:
        # The absolute-parallel decoder predicts every branch directly in the
        # global frame and has no attachment head.  Retaining source attachment
        # indices would expose targets for an output that does not exist.
        attachment.fill(-1)
    else:
        attachment[0] = -1
    output = {
        "target_centerline_parameters_mm": centerline_parameters.astype(
            np.float32
        ),
        "target_radius_baseline_coefficients_log_mm": radius_coefficients.astype(np.float32),
        "target_lesion_exist": lesions[..., 0].astype(np.float32),
        "target_lesion_geometry": lesions[..., 1:].astype(np.float32),
        "target_attachment_index": attachment.astype(np.int64),
        "target_branch_exist": branch_exists.astype(np.float32),
        "target_raw_vessel_mm": raw_vessel.astype(np.float32),
        "target_alignment_vessel_mm": alignment_vessel.astype(np.float32),
        "target_alignment_reference": np.asarray(
            alignment_reference
        ),
        "target_reconstructed_vessel_mm": reconstructed.astype(np.float32),
        "target_point_valid_mask": point_valid.astype(np.bool_),
        "parametric_target_path": np.asarray(str(target_path)),
        "target_centerline_parameter_frame": np.asarray(
            "global_mm"
            if absolute_centerline_parameters
            else "main_global_side_attachment_relative_mm"
        ),
    }
    if centerline_mode == "adaptive_landmarks":
        output["target_centerline_landmarks_mm"] = centerline_parameters.astype(
            np.float32
        )
    else:
        output["target_centerline_control_points_mm"] = (
            centerline_parameters.astype(np.float32)
        )
    return output

_BRANCH_VARIANT_ZERO_TARGET_KEYS = (
    "target_centerline_parameters_mm",
    "target_centerline_control_points_mm",
    "target_centerline_landmarks_mm",
    "target_radius_baseline_coefficients_log_mm",
    "target_lesion_exist",
    "target_lesion_geometry",
    "target_raw_vessel_mm",
    "target_alignment_vessel_mm",
    "target_reconstructed_vessel_mm",
)

def _mask_parametric_target_branches(
    target: Mapping[str, Any],
    active: np.ndarray,
    *,
    source_description: str,
) -> dict[str, Any]:
    """Zero and invalidate all branch-wise targets outside ``active``."""

    branch_mask = np.asarray(active, dtype=bool).reshape(-1)
    masked = dict(target)
    for key in _BRANCH_VARIANT_ZERO_TARGET_KEYS:
        if key not in target:
            continue
        value = np.asarray(target[key]).copy()
        if value.ndim < 1 or value.shape[0] != branch_mask.size:
            raise ValueError(
                f"{source_description} field {key!r} has shape {value.shape}; "
                f"expected first dimension {branch_mask.size}."
            )
        value[~branch_mask] = 0
        masked[key] = value
    point_valid = np.asarray(
        target["target_point_valid_mask"], dtype=bool
    ).copy()
    if point_valid.ndim < 1 or point_valid.shape[0] != branch_mask.size:
        raise ValueError(
            f"{source_description} field 'target_point_valid_mask' has shape "
            f"{point_valid.shape}; expected first dimension {branch_mask.size}."
        )
    point_valid[~branch_mask] = False
    masked["target_point_valid_mask"] = point_valid
    attachment = np.asarray(
        target["target_attachment_index"], dtype=np.int64
    ).copy()
    if attachment.shape != branch_mask.shape:
        raise ValueError(
            f"{source_description} field 'target_attachment_index' has shape "
            f"{attachment.shape}; expected {branch_mask.shape}."
        )
    attachment[~branch_mask] = -1
    masked["target_attachment_index"] = attachment
    masked["target_branch_exist"] = branch_mask.astype(np.float32)
    return masked

def _mask_directory_target_for_branch_variant(
    target: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    target_path: str | Path,
) -> dict[str, Any]:
    """Apply one Stage-4/5 variant mask to a shared full-case target."""

    source = Path(target_path).expanduser().resolve()
    full_exists = np.asarray(
        metadata["full_branch_exists"], dtype=bool
    ).reshape(-1)
    training_mask = np.asarray(
        metadata["training_branch_mask"], dtype=bool
    ).reshape(-1)
    target_full_exists = np.asarray(
        target["target_branch_exist"], dtype=np.float32
    ).reshape(-1) > 0.5
    if target_full_exists.shape != full_exists.shape:
        raise ValueError(
            f"Directory parametric target {source} has "
            f"target_branch_exist shape {target_full_exists.shape}, but its "
            f"Stage-4/5 feature variant expects {full_exists.shape}."
        )
    if not np.array_equal(target_full_exists, full_exists):
        differing = np.flatnonzero(target_full_exists != full_exists).tolist()
        raise ValueError(
            f"Directory parametric target {source} does not match the full "
            "branch topology stored by its Stage-4/5 feature variant. "
            f"Differing branch slot(s): {differing}; directory="
            f"{target_full_exists.astype(int).tolist()}, feature="
            f"{full_exists.astype(int).tolist()}."
        )
    active = full_exists & training_mask
    masked = _mask_parametric_target_branches(
        target,
        active,
        source_description=f"Directory parametric target {source}",
    )
    masked["parametric_target_full_branch_exist"] = full_exists.copy()
    masked["parametric_target_training_branch_mask"] = training_mask.copy()
    return masked

def load_paired_items(
    feature_files: Sequence[Path],
    config: dict[str, Any],
    *,
    desc: str = "Loading paired parametric cases",
    parametric_target_map: Mapping[str, Path] | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    profile = normalize_lesion_profile(config.get("lesion_profile", "gaussian"))
    model_cfg = dict(config.get("model", {}) or {})
    merged = dict(config)
    merged.update(model_cfg)
    num_branches = resolve_parametric_num_branches(config)
    target_num_branches = resolve_parametric_target_num_branches(
        config,
        model_num_branches=num_branches,
    )
    target_branch_geometry_scope_mask = np.arange(num_branches) < int(
        target_num_branches
    )
    # Every model query has a known existence label. Queries outside the
    # requested geometry target scope are explicit absent-branch negatives,
    # rather than unlabelled slots that should be ignored by BCE.
    target_branch_existence_supervision_mask = np.ones(
        (num_branches,), dtype=np.bool_
    )
    centerline_mode = normalize_centerline_prediction_mode(
        merged.get("centerline_prediction_mode", "bspline_control_points")
    )
    radius_mode = normalize_radius_prediction_mode(
        merged.get("radius_prediction_mode", "parametric")
    )
    decoder_architecture = normalize_decoder_architecture(
        merged.get(
            "decoder_architecture", MAIN_FIRST_HIERARCHICAL_DECODER
        )
    )
    absolute_centerline_parameters = (
        decoder_architecture == ABSOLUTE_PARALLEL_DECODER
    )
    original_only = resolve_parametric_original_variant_only(config)
    target_source = resolve_parametric_target_source(config)
    target_branch_mask_source = resolve_parametric_target_branch_mask_source(
        config
    )
    grouped_variants = resolve_branch_variant_group_training(config)
    target_map = None
    if target_source == "directory":
        target_map = (
            parametric_target_map
            if parametric_target_map is not None
            else discover_parametric_targets(
                config["parametric_target_dir"],
                original_only=original_only,
            )
        )
    elif parametric_target_map is not None:
        raise ValueError(
            "parametric_target_map is only valid when "
            "parametric_target_source='directory'."
        )
    feature_backbone = normalize_feature_backbone(config.get("feature_backbone", "vggt"))
    clinical_view_indices = resolve_clinical_rca_view_indices(config, feature_backbone)
    view_indices = (
        clinical_view_indices
        if clinical_view_indices is not None
        else config.get("view_indices")
    )
    online_backbone = uses_online_backbone_features(config)
    lazy_features = resolve_lazy_load_image_features(config, default=False)
    if online_backbone and lazy_features:
        raise ValueError(
            "lazy_load_image_features=true applies only to cached NPZ "
            "features and is incompatible with an online frozen backbone."
        )
    items = load_precomputed_items(
        files=list(feature_files),
        feature_backbone=feature_backbone,
        feature_key=str(config.get("feature_key", "image_features")),
        image_key=str(config.get("image_key", "images")),
        view_feature_key=str(config.get("view_feature_key", "view_features")),
        num_branches=num_branches,
        num_points=int(merged.get("num_points", 200)),
        input_scale_to_mm=float(config.get("input_scale_to_mm", 1000.0)),
        view_indices=None if view_indices is None else tuple(int(value) for value in view_indices),
        dataset_root=resolve_model_input_dataset_dir(config, feature_backbone),
        load_images=(
            False if online_backbone else bool(config.get("load_images", False))
        ),
        feature_metadata_required=(
            False
            if online_backbone
            else bool(config.get("feature_metadata_required", True))
        ),
        check_feature_finite=bool(config.get("check_feature_finite", True)),
        expected_vggt_context_mode=(
            None if online_backbone else config.get("expected_vggt_context_mode")
        ),
        target_coordinate_frame=str(config.get("target_coordinate_frame", "projection_centered")),
        zero_cached_image_features=resolve_zero_cached_image_features(
            config, default=False
        ),
        desc=desc,
        load_image_features=not online_backbone,
        lazy_load_image_features=lazy_features,
        show_progress=show_progress,
    )
    if online_backbone:
        for item in items:
            item["lazy_load_images"] = True
    # ``load_precomputed_case`` already restricts its model-facing
    # ``target_points`` and ``target_exist`` arrays. Keep the retained raw
    # artery on the same contract as well, so no full-tree ground truth is
    # carried into a two-branch parametric training batch.
    for item in items:
        feature_target_presence = (
            "target_points" in item,
            "target_exist" in item,
        )
        if any(feature_target_presence) and not all(feature_target_presence):
            raise ValueError(
                f"Case {item.get('path')} must provide target_points and "
                "target_exist together."
            )
        if all(feature_target_presence):
            target_points = np.asarray(item["target_points"]).copy()
            target_exist = np.asarray(item["target_exist"]).copy()
            if target_points.shape[0] != num_branches:
                raise ValueError(
                    f"Case {item.get('path')} target_points has branch "
                    f"capacity {target_points.shape[0]}, expected "
                    f"{num_branches}."
                )
            if target_exist.shape != (num_branches,):
                raise ValueError(
                    f"Case {item.get('path')} target_exist has shape "
                    f"{target_exist.shape}, expected ({num_branches},)."
                )
            target_points[~target_branch_geometry_scope_mask] = 0
            target_exist[~target_branch_geometry_scope_mask] = 0
            item["target_points"] = target_points
            item["target_exist"] = target_exist
        if "artery" in item:
            artery = _pad_branches(item["artery"], num_branches)
            artery[~target_branch_geometry_scope_mask] = 0
            item["artery"] = artery
    paired: list[dict[str, Any]] = []
    cached_directory_target_path: Path | None = None
    cached_directory_target: dict[str, np.ndarray] | None = None
    for item in items:
        identifier = case_identifier(item["path"])
        if target_source == "feature_file":
            target_path = Path(item["path"]).expanduser().resolve()
        else:
            assert target_map is not None
            target_path = target_map.get(identifier)
            if target_path is None:
                raise FileNotFoundError(
                    "No parametric target for feature case "
                    f"{identifier}: {item['path']}"
                )
        coordinate_frame = resolve_target_coordinate_frame(config)
        centering_source = str(
            config.get("parametric_centering_source", "paired_vessel")
        ).strip().lower()
        if centering_source not in {"paired_vessel", "projection_offset"}:
            raise ValueError(
                "parametric_centering_source must be 'paired_vessel' or 'projection_offset'"
            )
        reuse_directory_target = (
            grouped_variants
            and target_source == "directory"
            and cached_directory_target_path == target_path
            and cached_directory_target is not None
        )
        target = (
            cached_directory_target
            if reuse_directory_target
            else load_parametric_target(
                target_path,
                num_branches=num_branches,
                num_points=int(merged.get("num_points", 200)),
                num_control_points=int(merged.get("num_control_points", 20)),
                num_landmarks=int(
                    merged.get(
                        "num_landmarks", merged.get("num_control_points", 20)
                    )
                ),
                num_radius_coefficients=int(merged.get("num_radius_coefficients", 6)),
                num_lesions=int(merged.get("num_lesions", 3)),
                lesion_profile=profile,
                projection_center_offset_mm=None,
                centerline_prediction_mode=centerline_mode,
                radius_prediction_mode=radius_mode,
                catmull_rom_alpha=float(merged.get("catmull_rom_alpha", 0.5)),
                catmull_rom_dense_samples=int(
                    merged.get("catmull_rom_dense_samples", 1000)
                ),
                absolute_centerline_parameters=absolute_centerline_parameters,
            )
        )
        if grouped_variants and target_source == "directory":
            if not reuse_directory_target:
                cached_directory_target_path = target_path
                cached_directory_target = target
            assert target is not None
            variant_metadata = load_branch_variant_metadata(
                item["path"],
                num_branches=num_branches,
            )
            target = _mask_directory_target_for_branch_variant(
                target,
                variant_metadata,
                target_path=target_path,
            )
        if target_branch_mask_source == "feature":
            feature_branch_exist = np.asarray(
                item["target_exist"], dtype=np.float32
            ).reshape(-1) > 0.5
            target_branch_exist_before_feature_mask = np.asarray(
                target["target_branch_exist"], dtype=np.float32
            ).reshape(-1) > 0.5
            if (
                feature_branch_exist.shape
                != target_branch_exist_before_feature_mask.shape
            ):
                raise ValueError(
                    f"Case {identifier} feature branch-existence shape "
                    f"{feature_branch_exist.shape} does not match parametric "
                    "target branch-existence shape "
                    f"{target_branch_exist_before_feature_mask.shape}."
                )
            active_from_feature = (
                feature_branch_exist
                & target_branch_exist_before_feature_mask
            )
            target = _mask_parametric_target_branches(
                target,
                active_from_feature,
                source_description=(
                    f"Case {identifier} feature-masked parametric target "
                    f"{Path(target_path).expanduser().resolve()}"
                ),
            )
            target["parametric_target_branch_exist_before_feature_mask"] = (
                target_branch_exist_before_feature_mask.astype(np.bool_)
            )
            target["parametric_target_feature_branch_mask"] = (
                feature_branch_exist.astype(np.bool_)
            )
        branch_exist_before_target_limit = np.asarray(
            target["target_branch_exist"], dtype=np.float32
        ).reshape(-1) > 0.5
        target = _mask_parametric_target_branches(
            target,
            branch_exist_before_target_limit
            & target_branch_geometry_scope_mask,
            source_description=(
                f"Case {identifier} target branch scope from "
                f"{Path(target_path).expanduser().resolve()}"
            ),
        )
        target["target_branch_existence_supervision_mask"] = (
            target_branch_existence_supervision_mask.copy()
        )
        target["parametric_target_num_branches"] = np.asarray(
            target_num_branches,
            dtype=np.int64,
        )
        target["parametric_target_branch_mask_source"] = np.asarray(
            target_branch_mask_source
        )
        if coordinate_frame == "projection_centered":
            if centering_source == "projection_offset":
                offset = np.asarray(item["projection_center_offset"], dtype=np.float32)
                alignment_p95 = float("nan")
            else:
                raw = np.asarray(
                    target["target_alignment_vessel_mm"], dtype=np.float32
                )
                feature_target = np.asarray(item["target_points"], dtype=np.float32)
                if feature_target.shape != raw.shape:
                    raise ValueError(
                        f"Case {identifier} feature vessel shape "
                        f"{feature_target.shape} does not match parametric "
                        f"alignment vessel shape {raw.shape}."
                    )
                feature_branch_exist = np.asarray(
                    item["target_exist"], dtype=np.float32
                ).reshape(-1) > 0.5
                parametric_branch_exist = np.asarray(
                    target["target_branch_exist"], dtype=np.float32
                ).reshape(-1) > 0.5
                if feature_branch_exist.shape != parametric_branch_exist.shape:
                    raise ValueError(
                        f"Case {identifier} feature branch-existence shape "
                        f"{feature_branch_exist.shape} does not match parametric "
                        f"branch-existence shape {parametric_branch_exist.shape}."
                    )
                alignment_branch_mask = (
                    feature_branch_exist & parametric_branch_exist
                )
                valid = alignment_branch_mask[:, None] & np.asarray(
                    target["target_point_valid_mask"], dtype=bool
                )
                deltas = raw[..., :3][valid] - feature_target[..., :3][valid]
                if not deltas.size:
                    raise ValueError(
                        f"No valid paired vessel points for case {identifier}: "
                        "the feature NPZ and parametric target have no existing "
                        "branches in common."
                    )
                offset = np.median(deltas, axis=0).astype(np.float32)
                residual = np.linalg.norm(deltas - offset.reshape(1, 3), axis=-1)
                alignment_p95 = float(np.percentile(residual, 95))
                tolerance = float(config.get("paired_vessel_alignment_tolerance_mm", 5.0))
                if alignment_p95 > tolerance:
                    alignment_branch_indices = np.flatnonzero(
                        alignment_branch_mask
                    ).tolist()
                    raise ValueError(
                        f"Case {identifier} feature/parametric vessel alignment p95 is "
                        f"{alignment_p95:.3f} mm (limit {tolerance:.3f} mm) "
                        f"on common branch indices {alignment_branch_indices}. This usually "
                        "means the attachment-free decoded B-spline is a poor fit, "
                        "input_scale_to_mm is wrong (use 1000 for metre inputs and "
                        "1 for millimetre inputs), or the feature and target cases "
                        "do not match."
                    )
                target["paired_vessel_alignment_branch_mask"] = (
                    alignment_branch_mask.astype(np.bool_)
                )
            active = target["target_branch_exist"] > 0.5
            parameter_active = active.copy()
            if not absolute_centerline_parameters:
                parameter_active[1:] = False
            target["target_centerline_parameters_mm"][
                parameter_active
            ] -= offset.reshape(1, 1, 3)
            mode_key = (
                "target_centerline_landmarks_mm"
                if centerline_mode == "adaptive_landmarks"
                else "target_centerline_control_points_mm"
            )
            target[mode_key][parameter_active] -= offset.reshape(1, 1, 3)
            target["target_raw_vessel_mm"][active, :, :3] -= offset.reshape(1, 1, 3)
            target["target_alignment_vessel_mm"][
                active, :, :3
            ] -= offset.reshape(1, 1, 3)
            target["target_reconstructed_vessel_mm"][active, :, :3] -= offset.reshape(1, 1, 3)
            target["parametric_centering_offset_mm"] = offset
            target["paired_vessel_alignment_p95_mm"] = np.asarray(alignment_p95, dtype=np.float32)
            target["parametric_centering_source"] = np.asarray(
                centering_source
            )
        else:
            target["parametric_centering_offset_mm"] = np.zeros(
                (3,), dtype=np.float32
            )
            target["paired_vessel_alignment_p95_mm"] = np.asarray(
                float("nan"), dtype=np.float32
            )
            target["parametric_centering_source"] = np.asarray(
                "none_absolute_world"
            )
        item.update(target)
        if bool(config.get("enable_projection_2d_loss", False)):
            from scipy.ndimage import distance_transform_edt

            images = ensure_item_images(item)
            masks = images[:, 0] if images.ndim == 4 else images
            threshold = float(config.get("projection_mask_threshold", 0.5))
            item["projection_mask_distance_px"] = np.stack(
                [
                    distance_transform_edt(mask <= threshold).astype(np.float32)
                    for mask in masks
                ],
                axis=0,
            )
        item["case_id"] = identifier
        item["parametric_target_source"] = target_source
        paired.append(item)
    return paired

def load_prediction_items(
    feature_files: Sequence[Path],
    config: dict[str, Any],
    *,
    desc: str = "Loading parametric prediction cases",
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Load only model inputs; no raw or parametric 3D target is required."""
    model_cfg = dict(config.get("model", {}) or {})
    merged = dict(config)
    merged.update(model_cfg)
    num_branches = resolve_parametric_num_branches(config)
    feature_backbone = normalize_feature_backbone(
        config.get("feature_backbone", "vggt")
    )
    clinical_view_indices = resolve_clinical_rca_view_indices(
        config,
        feature_backbone,
    )
    view_indices = (
        clinical_view_indices
        if clinical_view_indices is not None
        else config.get("view_indices")
    )
    items = load_precomputed_items(
        files=list(feature_files),
        feature_backbone=feature_backbone,
        feature_key=str(config.get("feature_key", "image_features")),
        image_key=str(config.get("image_key", "images")),
        view_feature_key=str(config.get("view_feature_key", "view_features")),
        num_branches=num_branches,
        num_points=int(merged.get("num_points", 200)),
        input_scale_to_mm=float(config.get("input_scale_to_mm", 1000.0)),
        view_indices=(
            None
            if view_indices is None
            else tuple(int(value) for value in view_indices)
        ),
        dataset_root=resolve_feature_dataset_dir(config, feature_backbone),
        load_images=True,
        feature_metadata_required=bool(
            config.get("feature_metadata_required", True)
        ),
        check_feature_finite=bool(config.get("check_feature_finite", True)),
        expected_vggt_context_mode=config.get("expected_vggt_context_mode"),
        target_coordinate_frame=str(
            config.get("target_coordinate_frame", "projection_centered")
        ),
        zero_cached_image_features=resolve_zero_cached_image_features(
            config,
            default=False,
        ),
        require_targets=False,
        desc=desc,
        lazy_load_image_features=resolve_lazy_load_image_features(
            config, default=False
        ),
        show_progress=show_progress,
    )
    for item in items:
        view_count = int(np.asarray(item["view_mask"]).shape[0])
        if item.get("images") is None:
            raise KeyError(
                f"{item['path']} is missing input images required for the "
                "prediction overlay."
            )
        for angle_key in ("theta", "phi"):
            angles = item.get(angle_key)
            if angles is None:
                raise KeyError(
                    f"{item['path']} is missing {angle_key} projection angles "
                    "required for the prediction overlay."
                )
            if int(np.asarray(angles).reshape(-1).shape[0]) != view_count:
                raise ValueError(
                    f"{item['path']} has {view_count} cached views but "
                    f"{angle_key} has {np.asarray(angles).reshape(-1).shape[0]} "
                    "values."
                )
        item["case_id"] = str(item["case_name"])
    return items

class ParametricFeatureDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        min_views: int | None = None,
        max_views: int | None = None,
        random_view_order: bool = False,
        random_view_seed: int | None = None,
        view_count_weights: dict[int | str, float] | None = None,
    ) -> None:
        self.items = list(items)
        self.base = PrecomputedFeatureDataset(
            self.items,
            min_views=min_views,
            max_views=max_views,
            random_view_order=random_view_order,
            random_view_seed=random_view_seed,
            view_count_weights=view_count_weights,
        )

    def __len__(self) -> int:
        return len(self.base)

    def _attach_parametric_targets(
        self,
        output: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        item = self.items[int(index)]
        output["case_id"] = item["case_id"]
        if "parametric_target_path" in item:
            output["parametric_target_path"] = str(
                item["parametric_target_path"]
            )
        for key in PARAMETRIC_METADATA_KEYS:
            if key in item:
                output[key] = str(np.asarray(item[key]).reshape(()).item())
        for key in PARAMETRIC_BATCH_KEYS:
            if key in item:
                output[key] = torch.from_numpy(np.asarray(item[key]))
        for key in PARAMETRIC_OPTIONAL_BATCH_KEYS:
            if key in item:
                output[key] = torch.from_numpy(np.asarray(item[key]))
        if "projection_mask_distance_px" in item:
            selected = output["local_view_indices"].numpy()
            output["projection_mask_distance_px"] = torch.from_numpy(
                np.asarray(item["projection_mask_distance_px"], dtype=np.float32)[selected]
            )
        return output

    def item_with_view_indices(
        self,
        index: int,
        view_indices: np.ndarray | Sequence[int],
    ) -> dict[str, Any]:
        return self._attach_parametric_targets(
            self.base.item_with_view_indices(int(index), view_indices),
            int(index),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._attach_parametric_targets(self.base[index], int(index))

class ParametricBranchVariantGroupDataset(Dataset):
    """Present one physical case group per optimiser/evaluation step.

    All selected members share the same view indices and order.  Training may
    use every variant or a deterministic random subset per epoch; validation
    always returns every variant.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        config: Mapping[str, Any],
        *,
        train: bool,
        min_views: int | None = None,
        max_views: int | None = None,
        random_view_order: bool = False,
        view_count_weights: dict[int | str, float] | None = None,
    ) -> None:
        if not items:
            raise ValueError(
                "ParametricBranchVariantGroupDataset requires at least one item."
            )
        self.config = dict(config)
        self.train = bool(train)
        self.seed = int(self.config.get("seed", 0))
        resample_value = self.config.get(
            "branch_variant_resample_each_epoch", True
        )
        if not isinstance(resample_value, bool):
            raise ValueError(
                "branch_variant_resample_each_epoch must be a JSON boolean, "
                f"got {resample_value!r}."
            )
        self.resample_each_epoch = resample_value
        raw_strategy = str(
            self.config.get("branch_variant_train_sampling", "all")
        ).strip().lower()
        self.sampling_strategy = raw_strategy
        if self.sampling_strategy not in {"all", "random_one", "random_n"}:
            raise ValueError(
                "branch_variant_train_sampling must be 'all', 'random_one', "
                f"or 'random_n', got {raw_strategy!r}."
            )
        self.variants_per_group = int(
            self.config.get("branch_variant_train_variants_per_group", 1)
        )
        if self.variants_per_group < 1:
            raise ValueError(
                "branch_variant_train_variants_per_group must be >= 1."
            )
        num_branches = resolve_parametric_num_branches(self.config)
        self.num_branches = num_branches
        self.visibility_settings = resolve_branch_visibility_sampling(self.config)
        self.visibility_sampling_active = bool(
            self.visibility_settings.enabled
            and (self.train or self.visibility_settings.apply_to_validation)
        )
        self.visibility_metadata = (
            load_stage4_1_visibility_metadata(
                self.visibility_settings.metadata_json,
                num_branches=num_branches,
            )
            if self.visibility_sampling_active
            and self.visibility_settings.metadata_json is not None
            else None
        )
        metadata_by_path = {
            Path(item["path"]).expanduser().resolve(): load_branch_variant_metadata(
                item["path"], num_branches=num_branches
            )
            for item in items
        }
        item_by_path = {
            Path(item["path"]).expanduser().resolve(): item for item in items
        }
        file_groups = discover_branch_variant_groups(
            list(item_by_path), num_branches=num_branches
        )
        self.groups: list[tuple[BranchVariantFileGroup, tuple[int, ...]]] = []
        self.group_visibility: dict[str, dict[str, Any]] = {}
        ordered_items: list[dict[str, Any]] = []
        for group in file_groups:
            group_indices: list[int] = []
            for path in group.paths:
                item = item_by_path[path]
                metadata = metadata_by_path[path]
                item["branch_subset_group_id"] = str(metadata["group_id"])
                item["branch_subset_variant"] = str(metadata["variant"])
                item["branch_subset_stage"] = str(metadata["stage"])
                item["training_branch_mask"] = np.asarray(
                    metadata["training_branch_mask"], dtype=bool
                )
                # A shared identity makes the underlying deterministic view
                # selector choose the same count, indices, and order.
                item["view_selection_identity"] = str(group.group_id)
                group_indices.append(len(ordered_items))
                ordered_items.append(item)
            self.groups.append((group, tuple(group_indices)))
        self.items = ordered_items
        self.base = PrecomputedFeatureDataset(
            self.items,
            min_views=min_views,
            max_views=max_views,
            random_view_order=bool(random_view_order),
            random_view_seed=self.seed,
            view_count_weights=view_count_weights,
        )
        self.flat = ParametricFeatureDataset(self.items)
        # Reuse the configured selector while retaining ParametricFeatureDataset's
        # target materialisation logic.
        self.flat.base = self.base
        self.epoch = 0
        for group, indices in self.groups:
            reference = self.items[indices[0]]
            for index in indices[1:]:
                self._validate_common_views(group, reference, self.items[index])
            if self.visibility_metadata is not None:
                self.group_visibility[group.group_id] = (
                    self._validate_and_map_group_visibility(group, indices)
                )

    @staticmethod
    def _canonical_view_indices(item: Mapping[str, Any]) -> np.ndarray:
        total = int(np.asarray(item["view_mask"]).shape[0])
        stored = item.get("selected_view_indices")
        if stored is not None:
            stored_array = np.asarray(stored, dtype=np.int64).reshape(-1)
            if stored_array.shape != (total,):
                raise ValueError(
                    f"Case {item.get('case_name')} has {total} cached views but "
                    "selected_view_indices has shape "
                    f"{stored_array.shape}."
                )
            if np.any(stored_array < 0) or np.unique(stored_array).size != total:
                raise ValueError(
                    f"Case {item.get('case_name')} selected_view_indices must "
                    "contain one unique non-negative original ID per cached "
                    f"view, got {stored_array.tolist()}."
                )
            return stored_array
        return np.arange(total, dtype=np.int64)

    @classmethod
    def _validate_common_views(
        cls,
        group: BranchVariantFileGroup,
        reference: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> None:
        if not np.array_equal(
            np.asarray(reference["view_mask"]), np.asarray(current["view_mask"])
        ):
            raise ValueError(
                f"Branch-variant group {group.group_id} has different view masks."
            )
        if not np.array_equal(
            cls._canonical_view_indices(reference),
            cls._canonical_view_indices(current),
        ):
            raise ValueError(
                f"Branch-variant group {group.group_id} has different cached "
                "view mappings."
            )
        for angle_key in ("theta", "phi"):
            reference_angles = reference.get(angle_key)
            current_angles = current.get(angle_key)
            if (reference_angles is None) != (current_angles is None) or (
                reference_angles is not None
                and not np.allclose(
                    np.asarray(reference_angles),
                    np.asarray(current_angles),
                    rtol=0.0,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    f"Branch-variant group {group.group_id} has different "
                    f"{angle_key} camera angles."
                )
        if not np.allclose(
            np.asarray(reference["view_features"]),
            np.asarray(current["view_features"]),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                f"Branch-variant group {group.group_id} has different "
                "model-facing view direction features."
            )
        if not np.allclose(
            np.asarray(reference["projection_center_offset"]),
            np.asarray(current["projection_center_offset"]),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"Branch-variant group {group.group_id} has different "
                "projection centre offsets."
            )

    def _validate_and_map_group_visibility(
        self,
        group: BranchVariantFileGroup,
        indices: tuple[int, ...],
    ) -> dict[str, Any]:
        if group.stage != "stage4":
            raise ValueError(
                "Stage-4.1 branch visibility sampling only supports Stage-4 "
                f"progressive variants; group {group.group_id} stores "
                f"{group.stage!r}. Stage 5.1 is intentionally unsupported."
            )
        assert self.visibility_metadata is not None
        cases = self.visibility_metadata["cases"]
        if group.group_id not in cases:
            raise KeyError(
                f"Stage-4.1 visibility JSON {self.visibility_metadata['path']} "
                f"has no case {group.group_id!r}."
            )
        case = cases[group.group_id]
        expected_vessel_type = group.group_id.split(":", 1)[0]
        if case["vessel_type"] != expected_vessel_type:
            raise ValueError(
                f"Stage-4.1 case {group.group_id} stores vessel_type="
                f"{case['vessel_type']!r}; expected {expected_vessel_type!r}."
            )
        json_variants = set(case["variants"])
        group_variants = set(group.variants)
        if json_variants != group_variants:
            raise ValueError(
                f"Stage-4.1 case {group.group_id} variants do not match the "
                "Stage-4 feature group: missing_in_json="
                f"{sorted(group_variants - json_variants)}, extra_in_json="
                f"{sorted(json_variants - group_variants)}."
            )
        canonical_indices = self._canonical_view_indices(self.items[indices[0]])
        json_view_indices = np.asarray(case["view_indices"], dtype=np.int64)
        missing_cached_views = sorted(
            set(canonical_indices.tolist()).difference(json_view_indices.tolist())
        )
        if missing_cached_views:
            raise ValueError(
                f"Stage-4.1 case {group.group_id} does not describe cached "
                f"original view indices {missing_cached_views}."
            )
        for item_index in indices:
            item = self.items[item_index]
            variant = str(item["branch_subset_variant"])
            record = case["variants"][variant]
            if not np.array_equal(
                np.asarray(item["training_branch_mask"], dtype=bool),
                np.asarray(record["training_branch_mask"], dtype=bool),
            ):
                raise ValueError(
                    f"Stage-4.1 case {group.group_id} variant {variant!r} "
                    "training_branch_mask does not match its feature NPZ."
                )

        required_main_count = 1 if expected_vessel_type == "rca" else 2
        optional_existing = {
            int(index)
            for index, exists in enumerate(group.full_branch_exists)
            if exists and int(index) >= required_main_count
        }
        branch_records = case["branch_records"]
        if set(branch_records) != optional_existing:
            raise ValueError(
                f"Stage-4.1 case {group.group_id} branch introduction records "
                f"must cover exactly the existing side branches; expected "
                f"{sorted(optional_existing)}, got {sorted(branch_records)}."
            )
        original_to_local = {
            int(original): int(local)
            for local, original in enumerate(canonical_indices.tolist())
        }
        visible_local_by_branch: dict[int, frozenset[int]] = {}
        evidence_local_by_branch: dict[int, np.ndarray] = {}
        for branch_index, record in branch_records.items():
            visible_local_by_branch[int(branch_index)] = frozenset(
                original_to_local[int(original)]
                for original in np.asarray(
                    record["visible_view_indices"], dtype=np.int64
                ).tolist()
                if int(original) in original_to_local
            )
            evidence_by_original = {
                int(original): float(score)
                for original, score in zip(
                    np.asarray(record["view_indices"], dtype=np.int64).tolist(),
                    np.asarray(record["added_pixel_counts"], dtype=np.float64).tolist(),
                )
            }
            evidence_local_by_branch[int(branch_index)] = np.asarray(
                [
                    evidence_by_original.get(int(original), 0.0)
                    for original in canonical_indices.tolist()
                ],
                dtype=np.float64,
            )
        return {
            "required_main_count": required_main_count,
            "canonical_view_indices": canonical_indices,
            "visible_local_by_branch": visible_local_by_branch,
            "evidence_local_by_branch": evidence_local_by_branch,
        }

    @staticmethod
    def _active_side_branches(
        items: Sequence[Mapping[str, Any]],
        item_indices: Sequence[int],
        *,
        required_main_count: int,
    ) -> tuple[int, ...]:
        active: set[int] = set()
        for item_index in item_indices:
            mask = np.asarray(
                items[int(item_index)]["training_branch_mask"], dtype=bool
            )
            active.update(
                int(branch_index)
                for branch_index in np.flatnonzero(mask)
                if int(branch_index) >= int(required_main_count)
            )
        return tuple(sorted(active))

    @staticmethod
    def _visibility_cover_subset(
        required_branches: Sequence[int],
        *,
        max_views: int,
        priority: Sequence[int],
        visible_local_by_branch: Mapping[int, frozenset[int]],
        evidence_local_by_branch: Mapping[int, np.ndarray],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Solve the small branch/view set-cover problem deterministically."""

        branches = tuple(int(value) for value in required_branches)
        if not branches:
            return (), (), ()
        branch_bits = {
            branch: 1 << position for position, branch in enumerate(branches)
        }
        view_masks: dict[int, int] = {}
        for view_index in priority:
            mask = 0
            for branch in branches:
                if int(view_index) in visible_local_by_branch.get(
                    branch, frozenset()
                ):
                    mask |= branch_bits[branch]
            view_masks[int(view_index)] = mask

        dp: dict[int, tuple[int, ...]] = {0: ()}

        def evidence_score(selection: tuple[int, ...]) -> float:
            return float(
                sum(
                    float(evidence_local_by_branch[branch][priority[position]])
                    for branch in branches
                    for position in selection
                )
            )

        def preferable(
            candidate: tuple[int, ...],
            current: tuple[int, ...] | None,
        ) -> bool:
            if current is None or len(candidate) < len(current):
                return True
            if len(candidate) > len(current):
                return False
            candidate_score = evidence_score(candidate)
            current_score = evidence_score(current)
            if not np.isclose(candidate_score, current_score):
                return candidate_score > current_score
            return candidate < current

        for priority_position, view_index in enumerate(priority):
            for covered_mask, chosen_positions in list(dp.items())[::-1]:
                if len(chosen_positions) >= int(max_views):
                    continue
                next_mask = covered_mask | view_masks[int(view_index)]
                candidate = (*chosen_positions, int(priority_position))
                if preferable(candidate, dp.get(next_mask)):
                    dp[next_mask] = candidate

        best_mask = max(
            dp,
            key=lambda mask: (
                bin(int(mask)).count("1"),
                -len(dp[mask]),
                evidence_score(dp[mask]),
                tuple(-position for position in dp[mask]),
            ),
        )
        selected = tuple(int(priority[position]) for position in dp[best_mask])
        covered = tuple(
            branch for branch in branches if best_mask & branch_bits[branch]
        )
        uncovered = tuple(branch for branch in branches if branch not in covered)
        return selected, covered, uncovered

    def _visibility_view_selection(
        self,
        group: BranchVariantFileGroup,
        selected_indices: tuple[int, ...],
        base_view_indices: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        visibility = self.group_visibility[group.group_id]
        required = self._active_side_branches(
            self.items,
            selected_indices,
            required_main_count=int(visibility["required_main_count"]),
        )
        if not required:
            return np.asarray(base_view_indices, dtype=np.int64), (), (), ()

        total_views = int(
            np.asarray(self.items[selected_indices[0]]["view_mask"]).shape[0]
        )
        priority = [int(value) for value in np.asarray(base_view_indices).tolist()]
        remaining = [value for value in range(total_views) if value not in priority]
        if self.base.random_view_order and remaining:
            digest = hashlib.sha256(
                f"{self.base.random_view_seed}:{group.group_id}:stage4.1-fill".encode(
                    "utf-8"
                )
            ).digest()
            rng = np.random.default_rng(
                int.from_bytes(digest[:8], byteorder="big", signed=False)
            )
            rng.shuffle(remaining)
        priority.extend(remaining)
        core, covered, uncovered = self._visibility_cover_subset(
            required,
            max_views=int(len(base_view_indices)),
            priority=priority,
            visible_local_by_branch=visibility["visible_local_by_branch"],
            evidence_local_by_branch=visibility["evidence_local_by_branch"],
        )
        selected = list(core)
        remaining_priority = [view for view in priority if view not in selected]
        priority_rank = {view: position for position, view in enumerate(priority)}

        # Boolean set-cover cannot distinguish branches that have no
        # threshold-visible cached view. Give every such branch its strongest
        # incremental-silhouette evidence view when the remaining capacity can
        # hold the distinct choices. This prevents summed evidence from using
        # all spare views on one strong branch while omitting another entirely.
        strongest_evidence_views: dict[int, list[int]] = {}
        for branch in uncovered:
            scores = np.asarray(
                visibility["evidence_local_by_branch"][branch], dtype=np.float64
            )
            best_score = max(float(scores[view]) for view in priority)
            strongest_evidence_views[branch] = [
                view
                for view in priority
                if np.isclose(float(scores[view]), best_score)
            ]
        unrepresented = {
            branch
            for branch in uncovered
            if not any(
                view in selected for view in strongest_evidence_views[branch]
            )
        }
        while unrepresented and len(selected) < int(len(base_view_indices)):
            candidates = {
                view
                for branch in unrepresented
                for view in strongest_evidence_views[branch]
                if view not in selected
            }
            if not candidates:
                break
            best_view = min(
                candidates,
                key=lambda view: (
                    -sum(
                        view in strongest_evidence_views[branch]
                        for branch in unrepresented
                    ),
                    -sum(
                        float(
                            visibility["evidence_local_by_branch"][branch][view]
                        )
                        for branch in unrepresented
                    ),
                    priority_rank[view],
                ),
            )
            selected.append(best_view)
            unrepresented = {
                branch
                for branch in unrepresented
                if best_view not in strongest_evidence_views[branch]
            }

        remaining_priority = [view for view in priority if view not in selected]
        remaining_priority.sort(
            key=lambda view: (
                -sum(
                    float(visibility["evidence_local_by_branch"][branch][view])
                    for branch in uncovered
                ),
                priority_rank[view],
            )
        )
        selected.extend(
            remaining_priority[: int(len(base_view_indices)) - len(selected)]
        )
        if not self.base.random_view_order:
            selected.sort()
        selected_array = np.asarray(selected, dtype=np.int64)
        covered = tuple(
            branch
            for branch in required
            if any(
                int(view) in visibility["visible_local_by_branch"][branch]
                for view in selected_array.tolist()
            )
        )
        uncovered = tuple(branch for branch in required if branch not in covered)
        return selected_array, required, covered, uncovered

    def _variant_visibility_is_satisfiable(
        self,
        group: BranchVariantFileGroup,
        item_index: int,
        *,
        view_count: int,
    ) -> bool:
        visibility = self.group_visibility[group.group_id]
        required = self._active_side_branches(
            self.items,
            (item_index,),
            required_main_count=int(visibility["required_main_count"]),
        )
        if not required:
            return True
        _, _, uncovered = self._visibility_cover_subset(
            required,
            max_views=int(view_count),
            priority=tuple(
                range(
                    int(np.asarray(self.items[item_index]["view_mask"]).shape[0])
                )
            ),
            visible_local_by_branch=visibility["visible_local_by_branch"],
            evidence_local_by_branch=visibility["evidence_local_by_branch"],
        )
        return not uncovered

    def __len__(self) -> int:
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self.epoch = int(epoch)
        effective_epoch = self.epoch if self.resample_each_epoch and self.train else 0
        # PrecomputedFeatureDataset hashes this seed with the shared group
        # identity, so every member receives identical views.
        self.base.random_view_seed = int(self.seed + 1_000_003 * effective_epoch)

    def _selected_member_indices(
        self,
        group: BranchVariantFileGroup,
        indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not self.train or self.sampling_strategy == "all":
            return indices
        count = (
            1
            if self.sampling_strategy == "random_one"
            else min(self.variants_per_group, len(indices))
        )
        effective_epoch = self.epoch if self.resample_each_epoch else 0
        digest = hashlib.sha256(
            f"{self.seed}:{effective_epoch}:{group.group_id}:members".encode(
                "utf-8"
            )
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], byteorder="big", signed=False)
        )
        selected_positions = np.sort(
            rng.choice(len(indices), size=count, replace=False)
        )
        return tuple(indices[int(position)] for position in selected_positions)

    def _build_group_batch(
        self,
        group: BranchVariantFileGroup,
        all_indices: tuple[int, ...],
        selected_indices: tuple[int, ...],
        selected_view_indices: np.ndarray,
        *,
        required: tuple[int, ...] = (),
        covered: tuple[int, ...] = (),
        uncovered: tuple[int, ...] = (),
        policy_skipped_indices: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        members = [
            self.flat.item_with_view_indices(item_index, selected_view_indices)
            for item_index in selected_indices
        ]
        reference_indices = members[0]["local_view_indices"]
        if any(
            not torch.equal(member["local_view_indices"], reference_indices)
            for member in members[1:]
        ):
            raise RuntimeError(
                f"Branch-variant group {group.group_id} did not receive common "
                "view indices."
            )
        batch = collate_parametric_batches(members)
        batch["branch_variant_group_mode"] = True
        batch["branch_variant_group_id"] = group.group_id
        batch["branch_variant_group_case_id"] = group.case_id
        batch["branch_variant_group_stage"] = group.stage
        batch["branch_subset_variant"] = [
            str(self.items[item_index]["branch_subset_variant"])
            for item_index in selected_indices
        ]
        batch["branch_variant_group_size"] = len(all_indices)
        batch["branch_variant_selected_count"] = len(selected_indices)
        batch["branch_visibility_sampling_enabled"] = bool(
            self.visibility_settings.enabled
        )
        batch["branch_visibility_sampling_active"] = bool(
            self.visibility_sampling_active
        )
        batch["branch_visibility_unsatisfied_policy"] = str(
            self.visibility_settings.unsatisfied_policy
        )
        batch["branch_visibility_required_branch_indices"] = list(required)
        batch["branch_visibility_covered_branch_indices"] = list(covered)
        batch["branch_visibility_uncovered_branch_indices"] = list(uncovered)
        batch["branch_visibility_policy_skipped_variants"] = [
            str(self.items[item_index]["branch_subset_variant"])
            for item_index in policy_skipped_indices
        ]
        batch["branch_visibility_selected_local_view_indices"] = (
            selected_view_indices.tolist()
        )
        batch["branch_visibility_selected_original_view_indices"] = (
            batch["selected_view_indices"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
            .tolist()
        )
        batch["training_branch_mask"] = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        self.items[item_index]["training_branch_mask"],
                        dtype=bool,
                    )
                )
                for item_index in selected_indices
            ],
            dim=0,
        )
        return batch

    def item_from_sampling_schedule(
        self,
        index: int,
        *,
        local_view_indices: Sequence[int],
        variants: Sequence[str],
        policy_skipped_variants: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Materialise one group using an externally recorded exact selection."""

        group, all_indices = self.groups[int(index)]
        requested_variants = tuple(str(value) for value in variants)
        if not requested_variants:
            raise ValueError(
                f"Sampling schedule selected no variants for group {group.group_id}."
            )
        index_by_variant = {
            str(self.items[item_index]["branch_subset_variant"]): item_index
            for item_index in all_indices
        }
        if len(index_by_variant) != len(all_indices):
            raise ValueError(
                f"Branch-variant group {group.group_id} contains duplicate variant names."
            )
        missing = [
            variant for variant in requested_variants if variant not in index_by_variant
        ]
        if missing:
            raise ValueError(
                f"Sampling schedule requests unknown variants {missing} for group "
                f"{group.group_id}; available={sorted(index_by_variant)}."
            )
        if len(set(requested_variants)) != len(requested_variants):
            raise ValueError(
                f"Sampling schedule repeats a variant for group {group.group_id}: "
                f"{list(requested_variants)}."
            )
        selected_indices = tuple(
            index_by_variant[variant] for variant in requested_variants
        )
        requested_skipped_variants = tuple(
            str(value) for value in policy_skipped_variants
        )
        missing_skipped = [
            variant
            for variant in requested_skipped_variants
            if variant not in index_by_variant
        ]
        if missing_skipped:
            raise ValueError(
                f"Sampling schedule reports unknown policy-skipped variants "
                f"{missing_skipped} for group {group.group_id}."
            )
        if (
            len(set(requested_skipped_variants)) != len(requested_skipped_variants)
            or set(requested_variants).intersection(requested_skipped_variants)
        ):
            raise ValueError(
                f"Sampling schedule has duplicate or selected-and-skipped "
                f"variants for group {group.group_id}."
            )
        policy_skipped_indices = tuple(
            index_by_variant[variant] for variant in requested_skipped_variants
        )
        selected_view_indices = np.asarray(
            local_view_indices, dtype=np.int64
        ).reshape(-1)

        required: tuple[int, ...] = ()
        covered: tuple[int, ...] = ()
        uncovered: tuple[int, ...] = ()
        if self.visibility_sampling_active:
            visibility = self.group_visibility[group.group_id]
            required = self._active_side_branches(
                self.items,
                selected_indices,
                required_main_count=int(visibility["required_main_count"]),
            )
            selected_set = set(int(value) for value in selected_view_indices)
            covered = tuple(
                branch
                for branch in required
                if selected_set.intersection(
                    visibility["visible_local_by_branch"][branch]
                )
            )
            uncovered = tuple(
                branch for branch in required if branch not in set(covered)
            )
            if uncovered and self.visibility_settings.unsatisfied_policy in {
                "error",
                "skip_variant",
            }:
                raise RuntimeError(
                    f"Replayed sampling schedule leaves branches {list(uncovered)} "
                    f"uncovered for group {group.group_id} under policy "
                    f"{self.visibility_settings.unsatisfied_policy!r}."
                )

        return self._build_group_batch(
            group,
            all_indices,
            selected_indices,
            selected_view_indices,
            required=required,
            covered=covered,
            uncovered=uncovered,
            policy_skipped_indices=policy_skipped_indices,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        group, all_indices = self.groups[int(index)]
        base_view_indices = self.base._selected_view_indices(
            self.items[all_indices[0]]
        )
        eligible_indices = all_indices
        policy_skipped_indices: tuple[int, ...] = ()
        if (
            self.visibility_sampling_active
            and self.visibility_settings.unsatisfied_policy == "skip_variant"
        ):
            eligible_indices = tuple(
                item_index
                for item_index in all_indices
                if self._variant_visibility_is_satisfiable(
                    group,
                    item_index,
                    view_count=int(len(base_view_indices)),
                )
            )
            policy_skipped_indices = tuple(
                item_index
                for item_index in all_indices
                if item_index not in eligible_indices
            )
            if not eligible_indices:
                raise RuntimeError(
                    f"Stage-4.1 skip_variant removed every member of group "
                    f"{group.group_id}; the baseline main-only variant should "
                    "always remain feasible."
                )
        selected_indices = self._selected_member_indices(group, eligible_indices)
        required: tuple[int, ...] = ()
        covered: tuple[int, ...] = ()
        uncovered: tuple[int, ...] = ()
        selected_view_indices = np.asarray(base_view_indices, dtype=np.int64)
        if self.visibility_sampling_active:
            (
                selected_view_indices,
                required,
                covered,
                uncovered,
            ) = self._visibility_view_selection(
                group,
                selected_indices,
                base_view_indices,
            )
            if uncovered and self.visibility_settings.unsatisfied_policy == "error":
                raise RuntimeError(
                    f"Stage-4.1 visibility requirement is unsatisfied for "
                    f"group {group.group_id}: selected {len(selected_view_indices)} "
                    f"view(s), required side branches {list(required)}, "
                    f"uncovered branches {list(uncovered)}. Increase the view "
                    "count, relax the Stage-4.1 threshold, or choose "
                    "unsatisfied_policy='best_effort'/'skip_variant'."
                )
            if uncovered and self.visibility_settings.unsatisfied_policy == "skip_variant":
                raise RuntimeError(
                    f"Internal Stage-4.1 skip_variant error for group "
                    f"{group.group_id}: eligible progressive variants still "
                    f"left uncovered branches {list(uncovered)}."
                )
        return self._build_group_batch(
            group,
            all_indices,
            selected_indices,
            selected_view_indices,
            required=required,
            covered=covered,
            uncovered=uncovered,
            policy_skipped_indices=policy_skipped_indices,
        )

def collate_parametric_batches(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output = collate_precomputed_batches(batch)
    # Keep the model-facing name available while preserving the vessel_code.support/src batch contract.
    output["views"] = output["view_features"]
    output["case_id"] = [item["case_id"] for item in batch]
    if all("parametric_target_path" in item for item in batch):
        output["parametric_target_path"] = [
            item["parametric_target_path"] for item in batch
        ]
    for key in PARAMETRIC_METADATA_KEYS:
        presence = [key in item for item in batch]
        if any(presence) and not all(presence):
            raise ValueError(
                f"A batch cannot mix cases with and without {key!r}."
            )
        if all(presence):
            output[key] = [str(item[key]) for item in batch]
    for key in PARAMETRIC_BATCH_KEYS:
        presence = [key in item for item in batch]
        if any(presence) and not all(presence):
            raise ValueError(
                f"A batch cannot mix cases with and without {key!r}."
            )
        if all(presence):
            output[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in PARAMETRIC_OPTIONAL_BATCH_KEYS:
        if all(key in item for item in batch):
            output[key] = torch.stack([item[key] for item in batch], dim=0)
    if all("projection_mask_distance_px" in item for item in batch):
        max_views = int(output["view_mask"].shape[1])
        padded = []
        for item in batch:
            value = item["projection_mask_distance_px"]
            if value.shape[0] < max_views:
                result = value.new_zeros((max_views, *value.shape[1:]))
                result[: value.shape[0]] = value
                value = result
            padded.append(value)
        output["projection_mask_distance_px"] = torch.stack(padded, dim=0)
    return output
