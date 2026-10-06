# Transferred from methods/parametric_methods/radius_refiner_data.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from vessel_code.parametric.data import PARAMETRIC_BATCH_KEYS, PARAMETRIC_METADATA_KEYS, PARAMETRIC_OPTIONAL_BATCH_KEYS, collate_parametric_batches, load_parametric_target, resolve_feature_dataset_dir, resolve_raw_image_dataset_dir, uses_online_vggt_features
from vessel_code.parametric.representation import normalize_centerline_prediction_mode, normalize_lesion_profile
from vessel_code.shared.data import load_precomputed_case, load_split_record, normalize_view_count_weights
from vessel_code.shared.model import ABSOLUTE_PARALLEL_DECODER, MAIN_FIRST_HIERARCHICAL_DECODER, normalize_decoder_architecture

"""Group-aware Stage-3.5 input pipeline for radius-refiner training.

The ordinary parametric loader assumes one NPZ per case.  Stage 3.5 instead
stores either an original-only case or a counterfactual triplet below one
numeric case directory.  This module keeps that grouping intact, inherits the
case split from existing experiment split records, and presents one case group
per optimiser step.

The dataset item is already collated on the *member* dimension.  Its leading
batch dimension is therefore one for original-only groups and three for a
visible counterfactual group.  Use ``batch_size=1`` together with
``collate_radius_refiner_groups`` (or ``batch_size=None``) so a DataLoader does
not add another batch dimension.
"""

ORIGINAL_VARIANT = "original"

REMOVED_VARIANT = "stenosis_removed"

STRENGTHENED_VARIANT = "stenosis_strengthened"

COUNTERFACTUAL_VARIANTS = (
    ORIGINAL_VARIANT,
    REMOVED_VARIANT,
    STRENGTHENED_VARIANT,
)

NATURAL_NEGATIVE_GROUP = "natural_negative"

VISIBLE_AUGMENTED_GROUP = "visible_augmented"

INVISIBLE_STENOSIS_GROUP = "invisible_stenosis"

RADIUS_REFINER_GROUP_TYPES = (
    NATURAL_NEGATIVE_GROUP,
    VISIBLE_AUGMENTED_GROUP,
    INVISIBLE_STENOSIS_GROUP,
)

_VISIBILITY_KEYS = (
    "stage3_1_visibility_schema_version",
    "stenosis_visibility_mask_threshold",
    "stenosis_visibility_applicable",
    "stenosis_view_visible_mask",
    "stenosis_view_invisible_mask",
    "stenosis_any_view_visible",
    "stenosis_no_sufficient_view",
    "variant_has_stenosis",
    "variant_stenosis_view_visible_mask",
    "variant_stenosis_view_invisible_mask",
    "stenosis_region_point_mask",
)

@dataclass(frozen=True)
class RadiusRefinerGroup:
    """One case-level training unit before tensor loading."""

    case_id: str
    vessel_type: str
    group_id: str
    group_type: str
    variants: tuple[str, ...]
    paths: tuple[Path, ...]
    stenosis_view_visible_mask: tuple[bool, ...]
    stenosis_region_point_mask: np.ndarray
    split: str | None = None
    variant_has_stenosis: tuple[bool, ...] = ()
    variant_stenosis_view_visible_masks: tuple[tuple[bool, ...], ...] = ()

    def path_for_variant(self, variant: str) -> Path:
        try:
            return self.paths[self.variants.index(str(variant))]
        except ValueError as error:
            raise KeyError(
                f"Group {self.group_id!r} does not load variant {variant!r}."
            ) from error

@dataclass(frozen=True)
class RadiusRefinerGroupSplits:
    train: tuple[RadiusRefinerGroup, ...]
    val: tuple[RadiusRefinerGroup, ...]
    test: tuple[RadiusRefinerGroup, ...]

    def for_name(self, split: str) -> tuple[RadiusRefinerGroup, ...]:
        normalized = _normalize_split_name(split)
        return getattr(self, normalized)

@dataclass(frozen=True)
class Stage31VisibilityReport:
    """Validated Stage-3.1 visibility metadata used when NPZs are not enriched."""

    path: Path
    mask_threshold: float
    cases: dict[tuple[str, str], dict[str, Any]]

def _stage31_visibility_report_path(
    config: Mapping[str, Any],
) -> Path | None:
    raw = config.get("radius_refiner_stage3_1_visibility_report")
    if raw is None:
        raw = config.get("stage3_1_visibility_report")
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage-3.1 visibility report does not exist: {path}"
        )
    return path

def _load_stage31_visibility_report(
    config: Mapping[str, Any],
) -> Stage31VisibilityReport | None:
    path = _stage31_visibility_report_path(config)
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("stage")) != "3.1":
        raise ValueError(
            f"{path} is not a Stage-3.1 visibility report: "
            f"stage={payload.get('stage')!r}."
        )
    if int(payload.get("visibility_schema_version", -1)) != 1:
        raise ValueError(
            f"{path} has unsupported visibility_schema_version="
            f"{payload.get('visibility_schema_version')!r}; expected 1."
        )
    settings = payload.get("settings")
    if not isinstance(settings, Mapping) or "mask_threshold" not in settings:
        raise ValueError(f"{path} is missing settings.mask_threshold.")
    mask_threshold = float(settings["mask_threshold"])
    if not np.isfinite(mask_threshold) or not 0.0 <= mask_threshold <= 1.0:
        raise ValueError(
            f"{path} settings.mask_threshold must be finite and in [0, 1]."
        )
    records = payload.get("cases")
    if not isinstance(records, list):
        raise ValueError(f"{path} field 'cases' must be a list.")
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{path} cases[{index}] must be an object.")
        record = dict(raw_record)
        anatomy = str(record.get("anatomy", "")).strip().lower()
        if anatomy not in {"rca", "lca"}:
            raise ValueError(
                f"{path} cases[{index}].anatomy must be RCA or LCA."
            )
        case_id = _normalize_case_id(record.get("case_id"))
        key = (anatomy, case_id)
        if key in cases:
            raise ValueError(
                f"{path} contains duplicate visibility records for "
                f"{anatomy}:{case_id}."
            )
        cases[key] = record
    return Stage31VisibilityReport(
        path=path,
        mask_threshold=mask_threshold,
        cases=cases,
    )

def _scalar(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> Any:
    if key not in payload.files:
        raise KeyError(f"{path} is missing required field {key!r}.")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(
            f"{path} field {key!r} must be scalar, got shape {value.shape}."
        )
    return value.reshape(()).item()

def _normalize_case_id(value: Any) -> str:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ValueError(f"Case ID must be a non-negative integer, got {value!r}.")
    return str(int(text))

def _case_id_from_split_path(path: Path) -> str:
    if path.parent.name.isdigit():
        return _normalize_case_id(path.parent.name)
    groups = [part for part in path.stem.replace("-", "_").split("_") if part.isdigit()]
    if not groups:
        import re

        groups = re.findall(r"\d+", path.stem)
    if not groups:
        raise ValueError(
            f"Cannot infer a numeric case ID from split-record path {path}."
        )
    return _normalize_case_id(groups[-1])

def _normalize_split_name(value: str) -> str:
    name = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val"}
    name = aliases.get(name, name)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split name {value!r}.")
    return name

def _resolved_anatomy_root(config: Mapping[str, Any]) -> tuple[Path, str]:
    vessel_type = str(
        config.get("radius_refiner_vessel_type", config.get("vessel_type", ""))
    ).strip().lower()
    if vessel_type not in {"rca", "lca"}:
        raise ValueError(
            "radius_refiner_vessel_type (or vessel_type) must be 'rca' or 'lca'."
        )
    online_vggt = uses_online_vggt_features(dict(config))
    raw_root = (
        resolve_raw_image_dataset_dir(dict(config))
        if online_vggt
        else config.get(
            "radius_refiner_feature_dataset_dir", config.get("feature_dataset_dir")
        )
    )
    if raw_root is None:
        raise ValueError(
            "Missing radius_refiner_feature_dataset_dir (or feature_dataset_dir)."
        )
    root = Path(str(raw_root)).expanduser().resolve()
    candidates: list[Path] = []
    if root.name.lower() in {"rca", "lca"}:
        if root.name.lower() != vessel_type:
            raise ValueError(
                f"Dataset path ends in {root.name!r}, but vessel type is {vessel_type!r}."
            )
        candidates.append(root)
    else:
        candidates.append(root / vessel_type)
        if not online_vggt:
            resolver_config = dict(config)
            resolver_config["feature_dataset_dir"] = str(root)
            try:
                backbone_root = resolve_feature_dataset_dir(resolver_config)
            except (KeyError, ValueError):
                backbone_root = root
            candidates.append(backbone_root / vessel_type)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir():
            return candidate, vessel_type
    raise FileNotFoundError(
        "Could not resolve the radius-refiner anatomy directory. Checked: "
        + ", ".join(str(path) for path in dict.fromkeys(candidates))
    )

def _stenosis_region_from_payload(
    payload: np.lib.npyio.NpzFile,
    *,
    raw_shape: tuple[int, int, int],
    path: Path,
) -> np.ndarray:
    region = np.zeros(raw_shape[:2], dtype=bool)
    branch_indices = np.asarray(
        payload["stenosis_segment_branch_idx"]
        if "stenosis_segment_branch_idx" in payload.files
        else np.empty((0,), dtype=np.int32)
    ).reshape(-1)
    point_indices = np.asarray(
        payload["stenosis_segment_centerline_idx"]
        if "stenosis_segment_centerline_idx" in payload.files
        else np.empty((0,), dtype=np.int32)
    ).reshape(-1)
    if branch_indices.shape != point_indices.shape:
        raise ValueError(
            f"{path} stenosis segment branch/point index shapes disagree: "
            f"{branch_indices.shape} versus {point_indices.shape}."
        )
    valid = (
        (branch_indices >= 0)
        & (branch_indices < raw_shape[0])
        & (point_indices >= 0)
        & (point_indices < raw_shape[1])
    )
    region[
        branch_indices[valid].astype(int), point_indices[valid].astype(int)
    ] = True
    return region

def _visibility_from_stage31_record(
    record: Mapping[str, Any],
    *,
    variant: str,
    raw: np.ndarray,
    payload: np.lib.npyio.NpzFile,
    path: Path,
    mask_threshold: float,
) -> dict[str, Any]:
    augmented = bool(record.get("augmented", False))
    variants = tuple(str(value) for value in record.get("variants", []))
    if variant not in variants:
        raise ValueError(
            f"{path} variant {variant!r} is absent from its Stage-3.1 report "
            f"record variants={list(variants)}."
        )

    def view_mask(key: str) -> np.ndarray:
        indices = np.asarray(record.get(key, []), dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= 7) or (
            np.unique(indices).size != indices.size
        ):
            raise ValueError(
                f"{path} Stage-3.1 report field {key!r} must contain unique "
                "view indices from 0 through 6."
            )
        result = np.zeros((7,), dtype=bool)
        result[indices] = True
        return result

    visible = view_mask("visible_view_indices") if augmented else np.zeros(7, dtype=bool)
    original_visible = (
        view_mask("original_visible_view_indices")
        if augmented
        else np.zeros(7, dtype=bool)
    )
    variant_has_stenosis = bool(
        augmented and variant in {ORIGINAL_VARIANT, STRENGTHENED_VARIANT}
    )
    if variant == ORIGINAL_VARIANT and variant_has_stenosis:
        variant_visible = original_visible
    elif variant == STRENGTHENED_VARIANT and variant_has_stenosis:
        variant_visible = visible
    else:
        variant_visible = np.zeros((7,), dtype=bool)
    region = _stenosis_region_from_payload(
        payload, raw_shape=tuple(raw.shape), path=path
    )
    expected_region_count = int(record.get("stenosis_region_point_count", 0))
    if int(region.sum()) != expected_region_count:
        raise ValueError(
            f"{path} derives {int(region.sum())} stenosis-region point(s), but "
            f"the Stage-3.1 report records {expected_region_count}."
        )
    no_sufficient = bool(record.get("no_sufficient_view", False))
    if no_sufficient != bool(augmented and not visible.any()):
        raise ValueError(
            f"{path} has inconsistent no_sufficient_view in the Stage-3.1 report."
        )
    return {
        "group_id": str(record.get("counterfactual_group_id", "")),
        "mask_threshold": float(mask_threshold),
        "applicable": augmented,
        "visible": visible,
        "variant_visible": variant_visible,
        "any_visible": bool(augmented and visible.any()),
        "no_sufficient": no_sufficient,
        "variant_has_stenosis": variant_has_stenosis,
        "region": region,
    }

def _read_group_metadata(
    paths: Mapping[str, Path],
    *,
    case_id: str,
    vessel_type: str,
    expected_num_branches: int,
    expected_num_points: int,
    expected_imager_pixel_spacing: float,
    require_all_configured_branches: bool,
    expected_projection_mask_threshold: float,
    visibility_record: Mapping[str, Any] | None = None,
    visibility_report_mask_threshold: float | None = None,
) -> dict[str, Any]:
    per_variant: dict[str, dict[str, Any]] = {}
    for variant, path in paths.items():
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["raw_vessel_code_mm"], dtype=np.float32)
            if raw.ndim != 3 or raw.shape[-1] != 4:
                raise ValueError(f"{path} raw vessel must be [M,N,4], got {raw.shape}.")
            if visibility_record is not None:
                visibility_threshold = float(visibility_report_mask_threshold)
                visibility = _visibility_from_stage31_record(
                    visibility_record,
                    variant=variant,
                    raw=raw,
                    payload=payload,
                    path=path,
                    mask_threshold=visibility_threshold,
                )
            else:
                missing = [
                    key for key in _VISIBILITY_KEYS if key not in payload.files
                ]
                if missing:
                    raise KeyError(
                        f"{path} is missing Stage-3.1 radius-refiner metadata: "
                        f"{missing}. Configure "
                        "radius_refiner_stage3_1_visibility_report to use the "
                        "separate Stage-3.1 report with raw Stage-3 NPZ files."
                    )
                schema = int(
                    _scalar(payload, "stage3_1_visibility_schema_version", path)
                )
                if schema != 1:
                    raise ValueError(
                        f"{path} has Stage-3.1 visibility schema {schema}; expected 1."
                    )
                visibility_threshold = float(
                    _scalar(payload, "stenosis_visibility_mask_threshold", path)
                )
                visible = np.asarray(
                    payload["stenosis_view_visible_mask"], dtype=bool
                )
                invisible = np.asarray(
                    payload["stenosis_view_invisible_mask"], dtype=bool
                )
                variant_visible = np.asarray(
                    payload["variant_stenosis_view_visible_mask"], dtype=bool
                )
                variant_invisible = np.asarray(
                    payload["variant_stenosis_view_invisible_mask"], dtype=bool
                )
                if not (
                    visible.shape
                    == invisible.shape
                    == variant_visible.shape
                    == variant_invisible.shape
                    == (7,)
                ):
                    raise ValueError(
                        f"{path} Stage-3.1 visibility fields must each have "
                        "shape (7,)."
                    )
                visibility = {
                    "group_id": str(
                        _scalar(payload, "counterfactual_group_id", path)
                    ),
                    "mask_threshold": visibility_threshold,
                    "applicable": bool(
                        _scalar(payload, "stenosis_visibility_applicable", path)
                    ),
                    "visible": visible,
                    "variant_visible": variant_visible,
                    "any_visible": bool(
                        _scalar(payload, "stenosis_any_view_visible", path)
                    ),
                    "no_sufficient": bool(
                        _scalar(payload, "stenosis_no_sufficient_view", path)
                    ),
                    "variant_has_stenosis": bool(
                        _scalar(payload, "variant_has_stenosis", path)
                    ),
                    "region": np.asarray(
                        payload["stenosis_region_point_mask"], dtype=bool
                    ),
                }
            if not np.isfinite(visibility_threshold) or not np.isclose(
                visibility_threshold,
                expected_projection_mask_threshold,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError(
                    f"{path} Stage-3.1 visibility used mask threshold "
                    f"{visibility_threshold}, but radius-refiner local Dice uses "
                    f"projection_mask_threshold={expected_projection_mask_threshold}."
                )
            stored_case = _normalize_case_id(_scalar(payload, "case_id", path))
            stored_vessel = str(_scalar(payload, "vessel_type", path)).lower()
            stored_variant = str(_scalar(payload, "counterfactual_variant", path))
            group_id = str(_scalar(payload, "counterfactual_group_id", path))
            if stored_case != case_id or stored_vessel != vessel_type:
                raise ValueError(
                    f"{path} metadata identifies {stored_vessel}:{stored_case}, "
                    f"expected {vessel_type}:{case_id}."
                )
            if stored_variant != variant:
                raise ValueError(
                    f"{path} stores counterfactual_variant={stored_variant!r}, "
                    f"but its filename identifies {variant!r}."
                )
            if str(visibility["group_id"]) != group_id:
                raise ValueError(
                    f"{path} group ID {group_id!r} disagrees with the Stage-3.1 "
                    f"visibility record {visibility['group_id']!r}."
                )
            visible = np.asarray(visibility["visible"], dtype=bool)
            variant_visible = np.asarray(
                visibility["variant_visible"], dtype=bool
            )
            region = np.asarray(visibility["region"], dtype=bool)
            expected_shape = (
                int(expected_num_branches),
                int(expected_num_points),
                4,
            )
            if raw.shape != expected_shape:
                raise ValueError(
                    f"{path} raw_vessel_code_mm has shape {raw.shape}; the model "
                    f"expects {expected_shape}."
                )
            branch_exists = np.asarray(payload["branch_exists"], dtype=bool)
            point_valid = np.asarray(payload["point_valid_mask"], dtype=bool)
            if branch_exists.shape != (expected_num_branches,):
                raise ValueError(
                    f"{path} branch_exists has shape {branch_exists.shape}; "
                    f"expected {(expected_num_branches,)}."
                )
            if point_valid.shape != (expected_num_branches, expected_num_points):
                raise ValueError(
                    f"{path} point_valid_mask has shape {point_valid.shape}; "
                    f"expected {(expected_num_branches, expected_num_points)}."
                )
            if require_all_configured_branches and (
                not bool(branch_exists.all())
                or not bool(point_valid.any(axis=1).all())
            ):
                raise ValueError(
                    f"{path} does not contain every configured simplified "
                    "branch while "
                    "radius_refiner_require_all_configured_branches=true. "
                    "Set the flag to false to retain padded absent branches; "
                    "the radius refiner will mask them using the coarse model's "
                    "predicted branch-existence probabilities."
                )
            configured_num_branches = int(
                _scalar(payload, "configured_num_branches", path)
            )
            if configured_num_branches != int(expected_num_branches):
                raise ValueError(
                    f"{path} stores configured_num_branches="
                    f"{configured_num_branches}, but the model expects "
                    f"num_branches={expected_num_branches}."
                )
            spacing = float(_scalar(payload, "imager_pixel_spacing", path))
            if not np.isfinite(spacing) or not np.isclose(
                spacing,
                float(expected_imager_pixel_spacing),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"{path} stores imager_pixel_spacing={spacing} mm, but "
                    "model.radius_refiner_imager_pixel_spacing is "
                    f"{expected_imager_pixel_spacing} mm."
                )
            if "imager_pixel_spacing_units" in payload.files:
                spacing_units = str(
                    _scalar(payload, "imager_pixel_spacing_units", path)
                ).strip().lower()
                if spacing_units not in {"mm", "millimetre", "millimeter"}:
                    raise ValueError(
                        f"{path} imager_pixel_spacing_units={spacing_units!r}; "
                        "radius-refiner projection expects millimetres."
                    )
            if region.shape != raw.shape[:2]:
                raise ValueError(
                    f"{path} stenosis_region_point_mask has shape {region.shape}; "
                    f"expected {raw.shape[:2]}."
                )
            projection_offset = np.asarray(
                payload["projection_center_offset"], dtype=np.float32
            ).reshape(3)
            stenosis_xy = (
                None
                if "stenosis_xy" not in payload.files
                else np.asarray(payload["stenosis_xy"], dtype=np.float32)
            )
            stenosis_xy_valid = (
                None
                if "stenosis_xy_valid" not in payload.files
                else np.asarray(payload["stenosis_xy_valid"], dtype=bool)
            )
            per_variant[variant] = {
                "group_id": group_id,
                "mask_threshold": visibility_threshold,
                "applicable": bool(visibility["applicable"]),
                "visible": visible,
                "variant_visible": variant_visible,
                "any_visible": bool(visibility["any_visible"]),
                "no_sufficient": bool(visibility["no_sufficient"]),
                "variant_has_stenosis": bool(
                    visibility["variant_has_stenosis"]
                ),
                "region": region,
                "raw": raw,
                "branch_exists": branch_exists,
                "point_valid": point_valid,
                "projection_offset": projection_offset,
                "stenosis_xy": stenosis_xy,
                "stenosis_xy_valid": stenosis_xy_valid,
            }

    original = per_variant[ORIGINAL_VARIANT]
    for variant, metadata in per_variant.items():
        for scalar_key in (
            "group_id",
            "mask_threshold",
            "applicable",
            "any_visible",
            "no_sufficient",
        ):
            if metadata[scalar_key] != original[scalar_key]:
                raise ValueError(
                    f"Case {case_id} triplet disagrees on {scalar_key!r}: "
                    f"original={original[scalar_key]!r}, {variant}={metadata[scalar_key]!r}."
                )
        for array_key in (
            "visible",
            "region",
            "projection_offset",
            "branch_exists",
            "point_valid",
        ):
            if not np.array_equal(metadata[array_key], original[array_key]):
                raise ValueError(
                    f"Case {case_id} triplet disagrees on {array_key!r} for {variant}."
                )
        if not np.array_equal(metadata["raw"][..., :3], original["raw"][..., :3]):
            raise ValueError(
                f"Case {case_id} counterfactual {variant} changes XYZ; only radius may change."
            )
        outside = ~original["region"]
        if not np.array_equal(
            metadata["raw"][..., 3][outside], original["raw"][..., 3][outside]
        ):
            raise ValueError(
                f"Case {case_id} counterfactual {variant} changes radius outside "
                "stenosis_region_point_mask."
            )
        if (metadata["stenosis_xy"] is None) != (original["stenosis_xy"] is None):
            raise ValueError(
                f"Case {case_id} variants disagree on the presence of stenosis_xy."
            )
        if metadata["stenosis_xy"] is not None:
            if not np.array_equal(metadata["stenosis_xy"], original["stenosis_xy"]):
                raise ValueError(
                    f"Case {case_id} variants disagree on stenosis_xy for {variant}."
                )
            if not np.array_equal(
                metadata["stenosis_xy_valid"], original["stenosis_xy_valid"]
            ):
                raise ValueError(
                    f"Case {case_id} variants disagree on stenosis_xy_valid for {variant}."
                )
    original = dict(original)
    original["variant_has_stenosis_by_variant"] = {
        variant: bool(metadata["variant_has_stenosis"])
        for variant, metadata in per_variant.items()
    }
    original["variant_visible_by_variant"] = {
        variant: np.asarray(metadata["variant_visible"], dtype=bool).copy()
        for variant, metadata in per_variant.items()
    }
    return original

def discover_radius_refiner_groups(
    config: Mapping[str, Any],
) -> list[RadiusRefinerGroup]:
    """Discover and validate numeric Stage-3.5 case directories."""

    anatomy_root, vessel_type = _resolved_anatomy_root(config)
    visibility_report = _load_stage31_visibility_report(config)
    model_config = dict(config.get("model", {}) or {})
    merged = dict(config)
    merged.update(model_config)
    expected_num_branches = int(merged.get("num_branches", 7))
    expected_num_points = int(merged.get("num_points", 200))
    spacing_value = model_config.get(
        "radius_refiner_imager_pixel_spacing",
        config.get("radius_refiner_imager_pixel_spacing"),
    )
    if spacing_value is None:
        raise ValueError(
            "Stage-3.5 radius-refiner loading requires "
            "model.radius_refiner_imager_pixel_spacing (0.55 mm for RCA or "
            "0.65 mm for LCA)."
        )
    expected_imager_pixel_spacing = float(spacing_value)
    if not np.isfinite(expected_imager_pixel_spacing) or expected_imager_pixel_spacing <= 0.0:
        raise ValueError(
            "model.radius_refiner_imager_pixel_spacing must be finite and positive."
        )
    anatomy_spacing = {"rca": 0.55, "lca": 0.65}[vessel_type]
    if not np.isclose(
        expected_imager_pixel_spacing,
        anatomy_spacing,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            f"model.radius_refiner_imager_pixel_spacing="
            f"{expected_imager_pixel_spacing} mm is inconsistent with the "
            f"Stage-3 {vessel_type.upper()} spacing {anatomy_spacing} mm."
        )
    require_all_branches = config.get(
        "radius_refiner_require_all_configured_branches", False
    )
    if not isinstance(require_all_branches, bool):
        raise ValueError(
            "radius_refiner_require_all_configured_branches must be a JSON "
            f"boolean, got {require_all_branches!r}."
        )
    projection_mask_threshold = float(config.get("projection_mask_threshold", 0.5))
    if not np.isfinite(projection_mask_threshold) or not (
        0.0 <= projection_mask_threshold <= 1.0
    ):
        raise ValueError(
            "projection_mask_threshold must be finite and between 0 and 1."
        )
    if visibility_report is not None and not np.isclose(
        visibility_report.mask_threshold,
        projection_mask_threshold,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(
            f"{visibility_report.path} used mask_threshold="
            f"{visibility_report.mask_threshold}, but radius-refiner local Dice "
            f"uses projection_mask_threshold={projection_mask_threshold}."
        )
    case_dirs = sorted(
        (path for path in anatomy_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No numeric Stage-3.5 case directories found under {anatomy_root}."
        )

    groups: list[RadiusRefinerGroup] = []
    for case_dir in case_dirs:
        case_id = _normalize_case_id(case_dir.name)
        visibility_record = (
            None
            if visibility_report is None
            else visibility_report.cases.get((vessel_type, case_id))
        )
        if visibility_report is not None and visibility_record is None:
            raise KeyError(
                f"{visibility_report.path} has no {vessel_type.upper()} "
                f"visibility record for case {case_id}."
            )
        npz_paths = sorted(path for path in case_dir.glob("*.npz") if path.is_file())
        unknown = [path.name for path in npz_paths if path.stem not in COUNTERFACTUAL_VARIANTS]
        if unknown:
            raise ValueError(
                f"Case directory {case_dir} contains unsupported NPZ files: {unknown}."
            )
        paths = {path.stem: path.resolve() for path in npz_paths}
        names = set(paths)
        if names == {ORIGINAL_VARIANT}:
            metadata = _read_group_metadata(
                paths,
                case_id=case_id,
                vessel_type=vessel_type,
                expected_num_branches=expected_num_branches,
                expected_num_points=expected_num_points,
                expected_imager_pixel_spacing=expected_imager_pixel_spacing,
                require_all_configured_branches=require_all_branches,
                expected_projection_mask_threshold=projection_mask_threshold,
                visibility_record=visibility_record,
                visibility_report_mask_threshold=(
                    None
                    if visibility_report is None
                    else visibility_report.mask_threshold
                ),
            )
            if metadata["applicable"]:
                raise ValueError(
                    f"Case {case_id} has applicable stenosis metadata but only original.npz. "
                    "An augmented case must retain the exact counterfactual triplet."
                )
            if bool(np.asarray(metadata["region"], dtype=bool).any()):
                raise ValueError(
                    f"Natural-negative case {case_id} has a non-empty stenosis interval."
                )
            group_type = NATURAL_NEGATIVE_GROUP
            loaded_variants = (ORIGINAL_VARIANT,)
            if metadata["variant_has_stenosis"]:
                raise ValueError(
                    f"Natural-negative case {case_id} is marked variant_has_stenosis."
                )
        elif names == set(COUNTERFACTUAL_VARIANTS):
            metadata = _read_group_metadata(
                paths,
                case_id=case_id,
                vessel_type=vessel_type,
                expected_num_branches=expected_num_branches,
                expected_num_points=expected_num_points,
                expected_imager_pixel_spacing=expected_imager_pixel_spacing,
                require_all_configured_branches=require_all_branches,
                expected_projection_mask_threshold=projection_mask_threshold,
                visibility_record=visibility_record,
                visibility_report_mask_threshold=(
                    None
                    if visibility_report is None
                    else visibility_report.mask_threshold
                ),
            )
            if not metadata["applicable"]:
                raise ValueError(
                    f"Case {case_id} stores a triplet but visibility is not applicable."
                )
            if not bool(np.asarray(metadata["region"], dtype=bool).any()):
                raise ValueError(
                    f"Augmented case {case_id} has an empty stenosis interval."
                )
            expected_variant_status = {
                ORIGINAL_VARIANT: True,
                REMOVED_VARIANT: False,
                STRENGTHENED_VARIANT: True,
            }
            if metadata["variant_has_stenosis_by_variant"] != expected_variant_status:
                raise ValueError(
                    f"Augmented case {case_id} has inconsistent "
                    "variant_has_stenosis labels: "
                    f"{metadata['variant_has_stenosis_by_variant']}."
                )
            expected_no_sufficient = not bool(metadata["visible"].any())
            if bool(metadata["no_sufficient"]) != expected_no_sufficient:
                raise ValueError(
                    f"Case {case_id} has inconsistent no-sufficient-view metadata."
                )
            if bool(metadata["any_visible"]) != (not expected_no_sufficient):
                raise ValueError(
                    f"Case {case_id} has inconsistent any-view-visible metadata."
                )
            if expected_no_sufficient:
                group_type = INVISIBLE_STENOSIS_GROUP
                loaded_variants = (ORIGINAL_VARIANT,)
            else:
                if bool(config.get("radius_refiner_require_stenosis_xy", False)) and (
                    metadata["stenosis_xy"] is None
                    or metadata["stenosis_xy_valid"] is None
                ):
                    raise KeyError(
                        f"Visible augmented case {case_id} is missing stenosis_xy/"
                        "stenosis_xy_valid requested as optional monitoring metadata."
                    )
                group_type = VISIBLE_AUGMENTED_GROUP
                loaded_variants = COUNTERFACTUAL_VARIANTS
        else:
            missing = sorted(set(COUNTERFACTUAL_VARIANTS).difference(names))
            raise ValueError(
                f"Case {case_id} contains an incomplete counterfactual group. "
                f"Found={sorted(names)}, missing={missing}."
            )
        groups.append(
            RadiusRefinerGroup(
                case_id=case_id,
                vessel_type=vessel_type,
                group_id=str(metadata["group_id"]),
                group_type=group_type,
                variants=tuple(loaded_variants),
                paths=tuple(paths[name] for name in loaded_variants),
                stenosis_view_visible_mask=tuple(
                    bool(value) for value in metadata["visible"].tolist()
                ),
                stenosis_region_point_mask=np.asarray(
                    metadata["region"], dtype=bool
                ).copy(),
                variant_has_stenosis=tuple(
                    bool(metadata["variant_has_stenosis_by_variant"][name])
                    for name in loaded_variants
                ),
                variant_stenosis_view_visible_masks=tuple(
                    tuple(
                        bool(value)
                        for value in metadata["variant_visible_by_variant"][
                            name
                        ].tolist()
                    )
                    for name in loaded_variants
                ),
            )
        )
    return groups

def _manifest_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = config.get("radius_refiner_split_json_paths")
    if raw is None:
        raw = config.get(
            "radius_refiner_split_json_path", config.get("split_json_path")
        )
    if raw is None:
        return ()
    values = [raw] if isinstance(raw, (str, Path)) else list(raw)
    paths = tuple(Path(str(value)).expanduser().resolve() for value in values)
    if not paths:
        raise ValueError("radius_refiner_split_json_paths cannot be empty.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Inherited split record(s) do not exist: {missing}")
    return paths

def _add_assignment(
    assignments: dict[str, str],
    *,
    case_id: str,
    split: str,
    source: str,
) -> None:
    previous = assignments.get(case_id)
    if previous is not None and previous != split:
        raise ValueError(
            f"Inherited split sources disagree for case {case_id}: "
            f"{previous!r} versus {split!r} in {source}."
        )
    assignments[case_id] = split

def inherit_radius_refiner_group_splits(
    groups: Sequence[RadiusRefinerGroup], config: Mapping[str, Any]
) -> RadiusRefinerGroupSplits:
    """Assign case groups using one or more existing case-level split records."""

    assignment_sources: list[tuple[str, dict[str, str]]] = []
    for path in _manifest_paths(config):
        local_assignments: dict[str, str] = {}
        split_record = load_split_record(path)
        for split_name in ("train", "val", "test"):
            for case_path in getattr(split_record, split_name):
                _add_assignment(
                    local_assignments,
                    case_id=_case_id_from_split_path(case_path),
                    split=split_name,
                    source=str(path),
                )
        assignment_sources.append((str(path), local_assignments))

    direct = config.get("radius_refiner_split_case_ids")
    if direct is not None:
        if not isinstance(direct, Mapping):
            raise ValueError("radius_refiner_split_case_ids must be a mapping.")
        direct_assignments: dict[str, str] = {}
        for raw_split, case_ids in direct.items():
            split_name = _normalize_split_name(str(raw_split))
            for case_id in case_ids:
                _add_assignment(
                    direct_assignments,
                    case_id=_normalize_case_id(case_id),
                    split=split_name,
                    source="radius_refiner_split_case_ids",
                )
        assignment_sources.append(
            ("radius_refiner_split_case_ids", direct_assignments)
        )

    if not assignment_sources:
        raise ValueError(
            "Radius-refiner training requires an inherited case split. Configure "
            "radius_refiner_split_json_paths, radius_refiner_split_json_path/"
            "split_json_path, or radius_refiner_split_case_ids."
        )

    missing_policy = str(
        config.get("radius_refiner_split_missing_case_policy", "error")
    ).strip().lower()
    if missing_policy not in {"error", "drop"}:
        raise ValueError(
            "radius_refiner_split_missing_case_policy must be 'error' or 'drop'."
        )
    discovered_ids = {group.case_id for group in groups}
    conflicts: list[str] = []
    for case_id in sorted(discovered_ids, key=int):
        present = [
            (source, source_assignments[case_id])
            for source, source_assignments in assignment_sources
            if case_id in source_assignments
        ]
        if len({split for _, split in present}) > 1:
            conflicts.append(
                f"case {case_id}: "
                + ", ".join(f"{source}={split!r}" for source, split in present)
            )
    if conflicts:
        raise ValueError(
            "Inherited split sources disagree for case membership: "
            + "; ".join(conflicts)
        )

    missing_by_source = {
        source: sorted(discovered_ids.difference(assignments), key=int)
        for source, assignments in assignment_sources
        if discovered_ids.difference(assignments)
    }
    if missing_by_source and missing_policy == "error":
        details = "; ".join(
            f"{source}: {case_ids}"
            for source, case_ids in missing_by_source.items()
        )
        raise ValueError(
            "Every retained Stage-3.5 case must appear in every inherited split "
            f"source. Missing assignments by source: {details}."
        )

    assignments: dict[str, str] = {}
    for case_id in sorted(discovered_ids, key=int):
        memberships = [
            source_assignments.get(case_id)
            for _, source_assignments in assignment_sources
        ]
        if any(split is None for split in memberships):
            # missing_policy='drop': this is the explicit intersection of all
            # supplied coarse/geometry split records.
            continue
        assignments[case_id] = str(memberships[0])

    by_split: dict[str, list[RadiusRefinerGroup]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for group in groups:
        split = assignments.get(group.case_id)
        if split is None:
            continue
        by_split[split].append(replace(group, split=split))
    return RadiusRefinerGroupSplits(
        train=tuple(by_split["train"]),
        val=tuple(by_split["val"]),
        test=tuple(by_split["test"]),
    )

def _view_schedule(
    config: Mapping[str, Any], *, split: str
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        choice_keys = ("train_view_count_choices",)
        probability_keys = ("train_view_count_probabilities",)
        specific_schedule_keys: tuple[str, ...] = ()
        label = "train"
    elif split == "val":
        choice_keys = (
            "validation_view_count_choices",
            "val_view_count_choices",
            "eval_view_count_choices",
            "train_view_count_choices",
        )
        probability_keys = (
            "validation_view_count_probabilities",
            "val_view_count_probabilities",
            "eval_view_count_probabilities",
            "train_view_count_probabilities",
        )
        specific_schedule_keys = (*choice_keys[:-1], *probability_keys[:-1])
        label = "validation"
    else:
        choice_keys = (
            "test_view_count_choices",
            "eval_view_count_choices",
            "train_view_count_choices",
        )
        probability_keys = (
            "test_view_count_probabilities",
            "eval_view_count_probabilities",
            "train_view_count_probabilities",
        )
        specific_schedule_keys = (*choice_keys[:-1], *probability_keys[:-1])
        label = "test"

    raw_weights = config.get("train_view_count_weights")
    if raw_weights is not None and not any(
        config.get(key) is not None for key in specific_schedule_keys
    ):
        raw_min_views = config.get("min_train_views", 1)
        raw_max_views = config.get(
            "max_train_views", config.get("max_views", 7)
        )
        min_views = 1 if raw_min_views is None else int(raw_min_views)
        max_views = 7 if raw_max_views is None else int(raw_max_views)
        if min_views < 1 or max_views > 7 or min_views > max_views:
            raise ValueError(
                "Radius-refiner min_train_views/max_train_views must form "
                f"a valid range within 1..7, got {min_views}..{max_views}."
            )
        normalized = normalize_view_count_weights(
            raw_weights,
            min_views=min_views,
            max_views=max_views,
            setting_name="train_view_count_weights",
        )
        assert normalized is not None
        return (
            np.asarray(list(normalized), dtype=np.int64),
            np.asarray(list(normalized.values()), dtype=np.float64),
        )

    raw_choices = next(
        (config[key] for key in choice_keys if config.get(key) is not None),
        [1, 2, 3, 4, 5, 6, 7],
    )
    raw_probabilities = next(
        (config[key] for key in probability_keys if config.get(key) is not None),
        [0.25, 0.20, 0.16, 0.14, 0.10, 0.08, 0.07],
    )
    choices = np.asarray(raw_choices, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(raw_probabilities, dtype=np.float64).reshape(-1)
    if choices.size == 0 or choices.shape != probabilities.shape:
        raise ValueError(
            f"{label}_view_count_choices and probabilities must be non-empty "
            "arrays of equal length."
        )
    if (
        np.any(choices < 1)
        or np.any(choices > 7)
        or len(set(choices.tolist())) != int(choices.size)
    ):
        raise ValueError(
            f"{label}_view_count_choices must contain unique integers from 1 to 7."
        )
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(
            f"{label}_view_count_probabilities must be finite and non-negative."
        )
    if float(probabilities.sum()) <= 0.0:
        raise ValueError(f"{label}_view_count_probabilities must have positive sum.")
    probabilities /= probabilities.sum()
    return choices, probabilities

class Stage35RadiusRefinerGroupDataset(Dataset):
    """Exhaustive case-group dataset with reproducible visibility-aware views."""

    def __init__(
        self,
        groups: Sequence[RadiusRefinerGroup],
        config: Mapping[str, Any],
        *,
        split: str,
    ) -> None:
        self.groups = list(groups)
        if not self.groups:
            raise ValueError(f"Radius-refiner {split!r} dataset is empty.")
        self.config = dict(config)
        self.online_vggt = uses_online_vggt_features(self.config)
        self.split = _normalize_split_name(split)
        if any(group.split not in {None, self.split} for group in self.groups):
            raise ValueError(f"Dataset {self.split!r} received a group from another split.")
        groups_per_batch = int(self.config.get("groups_per_batch", 1))
        if groups_per_batch != 1:
            raise ValueError(
                "Stage35RadiusRefinerGroupDataset currently requires "
                "groups_per_batch=1 so each variable-size group is one optimiser step."
            )
        sampling_strategy = str(
            self.config.get("group_sampling_strategy", "exhaustive_shuffled")
        ).strip().lower()
        if sampling_strategy != "exhaustive_shuffled":
            raise ValueError(
                "group_sampling_strategy must be 'exhaustive_shuffled'. The "
                "radius-refiner epoch is an exhaustive pass over all case groups."
            )
        self.choices, self.probabilities = _view_schedule(
            self.config, split=self.split
        )
        self.seed = int(self.config.get("seed", 0))
        self.random_view_order = bool(
            self.config.get("radius_refiner_random_view_order", True)
        )
        require_visible_value = self.config.get(
            f"radius_refiner_{self.split}_require_visible_view",
            self.config.get(
                "radius_refiner_require_visible_view",
                self.split == "train",
            ),
        )
        if not isinstance(require_visible_value, bool):
            raise ValueError(
                f"radius_refiner_{self.split}_require_visible_view must be a "
                f"JSON boolean, got {require_visible_value!r}."
            )
        self.require_visible_view = require_visible_value
        default_resample = self.split == "train"
        self.resample_views_each_epoch = bool(
            self.config.get(
                f"radius_refiner_{self.split}_resample_views_each_epoch",
                self.config.get(
                    "radius_refiner_resample_views_each_epoch", default_resample
                ),
            )
        )
        self.epoch = 0

    def __len__(self) -> int:
        # A shuffled DataLoader visits every case group exactly once per epoch.
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self.epoch = int(epoch)

    def _rng(self, group: RadiusRefinerGroup) -> np.random.Generator:
        effective_epoch = self.epoch if self.resample_views_each_epoch else 0
        digest = hashlib.sha256(
            f"{self.seed}:{self.split}:{effective_epoch}:{group.group_id}".encode(
                "utf-8"
            )
        ).digest()
        return np.random.default_rng(
            int.from_bytes(digest[:8], byteorder="big", signed=False)
        )

    @staticmethod
    def _current_to_canonical_views(item: Mapping[str, Any]) -> np.ndarray:
        total = int(np.asarray(item["view_mask"]).shape[0])
        stored = item.get("selected_view_indices")
        if stored is not None:
            stored = np.asarray(stored, dtype=np.int64).reshape(-1)
            if stored.shape == (total,):
                return stored
        return np.arange(total, dtype=np.int64)

    def _select_local_views(
        self,
        group: RadiusRefinerGroup,
        item: Mapping[str, Any],
    ) -> np.ndarray:
        rng = self._rng(group)
        stored_view_mask = np.asarray(item["view_mask"], dtype=np.float32).reshape(-1)
        total = int(stored_view_mask.shape[0])
        valid_local = np.flatnonzero(stored_view_mask > 0.5)
        if valid_local.size == 0:
            raise ValueError(f"Group {group.group_id} has no valid cached views.")
        eligible = self.choices <= int(valid_local.size)
        choices = self.choices[eligible]
        probabilities = self.probabilities[eligible]
        if choices.size == 0 or float(probabilities.sum()) <= 0.0:
            raise ValueError(
                f"Group {group.group_id} has {valid_local.size}/{total} valid "
                "cached views, incompatible "
                "with the configured view-count schedule."
            )
        probabilities = probabilities / probabilities.sum()
        count = int(rng.choice(choices, p=probabilities))

        required_local: int | None = None
        if (
            group.group_type == VISIBLE_AUGMENTED_GROUP
            and self.require_visible_view
        ):
            canonical = self._current_to_canonical_views(item)
            full_visible = np.asarray(
                group.stenosis_view_visible_mask, dtype=bool
            )
            if np.any((canonical < 0) | (canonical >= full_visible.size)):
                raise ValueError(
                    f"Group {group.group_id} cached view indices {canonical.tolist()} "
                    "cannot be mapped to the seven Stage-3.1 views."
                )
            visible_local = np.flatnonzero(
                full_visible[canonical] & (stored_view_mask > 0.5)
            )
            if visible_local.size == 0:
                raise ValueError(
                    f"Visible augmented group {group.group_id} has no visible "
                    "stenosis view in its cached feature subset. Recompute Stage 3.5 "
                    "with all seven views."
                )
            required_local = int(rng.choice(visible_local))

        if required_local is None:
            selected = rng.choice(valid_local, size=count, replace=False)
        elif count == 1:
            selected = np.asarray([required_local], dtype=np.int64)
        else:
            remaining = valid_local[valid_local != required_local]
            extra = rng.choice(remaining, size=count - 1, replace=False)
            selected = np.concatenate(
                (np.asarray([required_local], dtype=np.int64), extra)
            )
        if self.random_view_order:
            selected = rng.permutation(selected)
        else:
            selected = np.sort(selected)
        return np.asarray(selected, dtype=np.int64)

    def _load_member_item(
        self,
        path: Path,
        group: RadiusRefinerGroup,
        variant: str,
    ) -> dict[str, Any]:
        model_config = dict(self.config.get("model", {}) or {})
        merged = dict(self.config)
        merged.update(model_config)
        feature_backbone = str(self.config.get("feature_backbone", "vggt"))
        item = load_precomputed_case(
            path=path,
            feature_backbone=feature_backbone,
            feature_key=str(self.config.get("feature_key", "image_features")),
            image_key=str(self.config.get("image_key", "images")),
            view_feature_key=str(
                self.config.get("view_feature_key", "view_features")
            ),
            num_branches=int(merged.get("num_branches", 7)),
            num_points=int(merged.get("num_points", 200)),
            input_scale_to_mm=float(self.config.get("input_scale_to_mm", 1000.0)),
            view_indices=None,
            dataset_root=path.parent,
            load_images=True,
            feature_metadata_required=(
                False
                if self.online_vggt
                else bool(self.config.get("feature_metadata_required", True))
            ),
            check_feature_finite=bool(
                self.config.get("check_feature_finite", True)
            ),
            expected_vggt_context_mode=(
                None
                if self.online_vggt
                else self.config.get("expected_vggt_context_mode")
            ),
            target_coordinate_frame=str(
                self.config.get("target_coordinate_frame", "projection_centered")
            ),
            zero_cached_image_features=bool(
                self.config.get("zero_cached_image_features", False)
            ),
            require_targets=True,
            load_image_features=not self.online_vggt,
        )
        centerline_mode = normalize_centerline_prediction_mode(
            merged.get("centerline_prediction_mode", "bspline_control_points")
        )
        if centerline_mode != "bspline_control_points":
            raise ValueError(
                "Stage-3.5 radius-refiner training requires "
                "centerline_prediction_mode='bspline_control_points'."
            )
        decoder_architecture = normalize_decoder_architecture(
            merged.get(
                "decoder_architecture", MAIN_FIRST_HIERARCHICAL_DECODER
            )
        )
        coordinate_frame = str(
            self.config.get("target_coordinate_frame", "projection_centered")
        )
        target = load_parametric_target(
            path,
            num_branches=int(merged.get("num_branches", 7)),
            num_points=int(merged.get("num_points", 200)),
            num_control_points=int(merged.get("num_control_points", 20)),
            num_landmarks=int(
                merged.get(
                    "num_landmarks", merged.get("num_control_points", 20)
                )
            ),
            num_radius_coefficients=int(
                merged.get("num_radius_coefficients", 6)
            ),
            num_lesions=int(merged.get("num_lesions", 3)),
            lesion_profile=normalize_lesion_profile(
                merged.get("lesion_profile", "gaussian")
            ),
            projection_center_offset_mm=(
                np.asarray(item["projection_center_offset"], dtype=np.float32)
                if coordinate_frame == "projection_centered"
                else None
            ),
            centerline_prediction_mode=centerline_mode,
            radius_prediction_mode="raw",
            absolute_centerline_parameters=(
                decoder_architecture == ABSOLUTE_PARALLEL_DECODER
            ),
        )
        item.update(target)
        with np.load(path, allow_pickle=False) as payload:
            stored_variant = str(_scalar(payload, "counterfactual_variant", path))
            stored_group_id = str(
                _scalar(payload, "counterfactual_group_id", path)
            )
            if stored_variant != variant or stored_group_id != group.group_id:
                raise ValueError(
                    f"{path} identifies {stored_group_id}:{stored_variant}, "
                    f"expected {group.group_id}:{variant}."
                )
            variant_index = group.variants.index(variant)
            item["counterfactual_variant"] = variant
            item["counterfactual_group_id"] = group.group_id
            item["variant_has_stenosis"] = bool(
                group.variant_has_stenosis[variant_index]
            )
            item["stenosis_view_visible_mask"] = np.asarray(
                group.stenosis_view_visible_mask, dtype=bool
            )
            item["variant_stenosis_view_visible_mask"] = np.asarray(
                group.variant_stenosis_view_visible_masks[variant_index],
                dtype=bool,
            )
            item["stenosis_region_point_mask"] = np.asarray(
                group.stenosis_region_point_mask, dtype=bool
            )
            item["stenosis_xy"] = (
                None
                if "stenosis_xy" not in payload.files
                else np.asarray(payload["stenosis_xy"], dtype=np.float32)
            )
            item["stenosis_xy_valid"] = (
                None
                if "stenosis_xy_valid" not in payload.files
                else np.asarray(payload["stenosis_xy_valid"], dtype=bool)
            )
        return item

    @staticmethod
    def _subset_features(
        features: dict[str, np.ndarray] | np.ndarray, indices: np.ndarray
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(features, dict):
            return {
                key: torch.from_numpy(
                    np.asarray(value, dtype=np.float32)[indices]
                )
                for key, value in features.items()
            }
        return torch.from_numpy(
            np.asarray(features, dtype=np.float32)[indices]
        )

    def _materialize_member(
        self,
        item: Mapping[str, Any],
        indices: np.ndarray,
        *,
        item_index: int,
        case_id: str,
    ) -> dict[str, Any]:
        canonical = self._current_to_canonical_views(item)
        selected_canonical = canonical[indices]
        theta = item.get("theta")
        phi = item.get("phi")
        output: dict[str, Any] = {
            "case_name": f"{case_id}:{item['counterfactual_variant']}",
            "case_id": case_id,
            "path": str(item["path"]),
            "item_index": int(item_index),
            "local_view_indices": torch.from_numpy(indices.copy()),
            "selected_view_indices": torch.from_numpy(
                selected_canonical.astype(np.int64, copy=False)
            ),
            "images": torch.from_numpy(
                np.asarray(item["images"], dtype=np.float32)[indices]
            ),
            "view_features": torch.from_numpy(
                np.asarray(item["view_features"], dtype=np.float32)[indices]
            ),
            "view_mask": torch.from_numpy(
                np.asarray(item["view_mask"], dtype=np.float32)[indices]
            ),
            "image_features": (
                None
                if item["image_features"] is None
                else self._subset_features(item["image_features"], indices)
            ),
            "theta": (
                None
                if theta is None
                else torch.from_numpy(np.asarray(theta, dtype=np.float32)[indices])
            ),
            "phi": (
                None
                if phi is None
                else torch.from_numpy(np.asarray(phi, dtype=np.float32)[indices])
            ),
            "projection_center_offset": torch.from_numpy(
                np.asarray(item["projection_center_offset"], dtype=np.float32)
            ),
            "projection_center_offset_valid": torch.tensor(
                bool(item["projection_center_offset_valid"]), dtype=torch.bool
            ),
        }
        if "target_points" in item:
            output["target_points"] = torch.from_numpy(
                np.asarray(item["target_points"], dtype=np.float32)
            )
            output["target_exist"] = torch.from_numpy(
                np.asarray(item["target_exist"], dtype=np.float32)
            )
            output["artery"] = torch.from_numpy(
                np.asarray(item["artery"], dtype=np.float32)
            )
        if "parametric_target_path" in item:
            output["parametric_target_path"] = str(
                np.asarray(item["parametric_target_path"]).reshape(()).item()
            )
        for key in PARAMETRIC_METADATA_KEYS:
            if key in item:
                output[key] = str(np.asarray(item[key]).reshape(()).item())
        for key in (*PARAMETRIC_BATCH_KEYS, *PARAMETRIC_OPTIONAL_BATCH_KEYS):
            if key in item:
                output[key] = torch.from_numpy(np.asarray(item[key]))
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[int(index)]
        loaded = [
            self._load_member_item(path, group, variant)
            for path, variant in zip(group.paths, group.variants)
        ]
        reference_views = int(np.asarray(loaded[0]["view_mask"]).shape[0])
        for item in loaded[1:]:
            if int(np.asarray(item["view_mask"]).shape[0]) != reference_views:
                raise ValueError(
                    f"Group {group.group_id} variants have different cached view counts."
                )
            if not np.array_equal(
                np.asarray(item["view_mask"]),
                np.asarray(loaded[0]["view_mask"]),
            ):
                raise ValueError(
                    f"Group {group.group_id} variants have different view masks."
                )
            if not np.array_equal(
                self._current_to_canonical_views(item),
                self._current_to_canonical_views(loaded[0]),
            ):
                raise ValueError(
                    f"Group {group.group_id} variants have different cached view mappings."
                )
            for angle_key in ("theta", "phi"):
                reference_angles = loaded[0].get(angle_key)
                current_angles = item.get(angle_key)
                if (reference_angles is None) != (current_angles is None) or (
                    reference_angles is not None
                    and not np.array_equal(
                        np.asarray(current_angles), np.asarray(reference_angles)
                    )
                ):
                    raise ValueError(
                        f"Group {group.group_id} variants have different "
                        f"{angle_key} camera angles."
                    )
        indices = self._select_local_views(group, loaded[0])
        member_items = [
            self._materialize_member(
                item,
                indices,
                item_index=member_index,
                case_id=group.case_id,
            )
            for member_index, item in enumerate(loaded)
        ]
        batch = collate_parametric_batches(member_items)
        batch["group_id"] = group.group_id
        batch["counterfactual_group_id"] = group.group_id
        batch["group_case_id"] = group.case_id
        batch["group_type"] = group.group_type
        batch["group_split"] = group.split or self.split
        batch["group_size"] = len(loaded)
        batch["counterfactual_variant"] = [
            str(item["counterfactual_variant"]) for item in loaded
        ]
        batch["counterfactual_variant_index"] = torch.tensor(
            [COUNTERFACTUAL_VARIANTS.index(name) for name in batch["counterfactual_variant"]],
            dtype=torch.long,
        )
        batch["variant_has_stenosis"] = torch.tensor(
            [bool(item["variant_has_stenosis"]) for item in loaded],
            dtype=torch.bool,
        )
        batch["stenosis_region_point_mask"] = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(item["stenosis_region_point_mask"], dtype=bool)
                )
                for item in loaded
            ],
            dim=0,
        )
        canonical = self._current_to_canonical_views(loaded[0])
        selected_canonical = canonical[indices]
        full_common_visible = np.asarray(
            group.stenosis_view_visible_mask, dtype=bool
        )
        selected_common_visible = full_common_visible[selected_canonical]
        batch["stenosis_view_visible_mask"] = torch.from_numpy(
            np.repeat(
                selected_common_visible.reshape(1, -1), len(loaded), axis=0
            )
        )
        batch["variant_stenosis_view_visible_mask"] = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        item["variant_stenosis_view_visible_mask"], dtype=bool
                    )[selected_canonical]
                )
                for item in loaded
            ],
            dim=0,
        )
        batch["selected_group_local_view_indices"] = torch.from_numpy(indices.copy())
        batch["selected_group_view_indices"] = torch.from_numpy(
            selected_canonical.astype(np.int64, copy=False)
        )
        batch["selected_group_has_visible_stenosis"] = torch.tensor(
            bool(selected_common_visible.any()), dtype=torch.bool
        )

        xy_values: list[torch.Tensor] = []
        xy_valid_values: list[torch.Tensor] = []
        for item in loaded:
            stenosis_xy = item["stenosis_xy"]
            stenosis_xy_valid = item["stenosis_xy_valid"]
            if stenosis_xy is None or stenosis_xy_valid is None:
                xy_values.append(torch.zeros((len(indices), 0, 2), dtype=torch.float32))
                xy_valid_values.append(torch.zeros((len(indices), 0), dtype=torch.bool))
                continue
            xy = np.asarray(stenosis_xy, dtype=np.float32)
            valid = np.asarray(stenosis_xy_valid, dtype=bool)
            if xy.shape[0] != reference_views or valid.shape != xy.shape[:2]:
                raise ValueError(
                    f"{item['path']} stenosis_xy/valid have incompatible shapes "
                    f"{xy.shape} and {valid.shape}; expected [V,P,2] and [V,P]."
                )
            xy_values.append(torch.from_numpy(xy[indices]))
            xy_valid_values.append(torch.from_numpy(valid[indices]))
        batch["stenosis_xy"] = torch.stack(xy_values, dim=0)
        batch["stenosis_xy_valid"] = torch.stack(xy_valid_values, dim=0)
        return batch

def collate_radius_refiner_groups(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Remove the DataLoader's group dimension while enforcing one group/step."""

    if len(batch) != 1:
        raise ValueError(
            "Radius-refiner batches must contain exactly one case group. Set "
            "groups_per_batch=1 and DataLoader(batch_size=1)."
        )
    return batch[0]

def build_radius_refiner_group_datasets(
    config: Mapping[str, Any],
) -> dict[str, Stage35RadiusRefinerGroupDataset]:
    """Discover groups, inherit splits, and build every non-empty dataset."""

    groups = discover_radius_refiner_groups(config)
    splits = inherit_radius_refiner_group_splits(groups, config)
    output: dict[str, Stage35RadiusRefinerGroupDataset] = {}
    for split_name in ("train", "val", "test"):
        split_groups = splits.for_name(split_name)
        if split_groups:
            output[split_name] = Stage35RadiusRefinerGroupDataset(
                split_groups, config, split=split_name
            )
    if "train" not in output:
        raise ValueError("Inherited split contains no Stage-3.5 training groups.")
    return output

def summarize_radius_refiner_groups(
    groups: Iterable[RadiusRefinerGroup],
) -> dict[str, int]:
    summary = {group_type: 0 for group_type in RADIUS_REFINER_GROUP_TYPES}
    summary["total"] = 0
    for group in groups:
        summary[group.group_type] += 1
        summary["total"] += 1
    return summary
