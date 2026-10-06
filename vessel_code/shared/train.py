# Transferred from methods/src/train.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any
import matplotlib
from vessel_code.shared.data import SplitFiles, case_number_from_path, filter_excluded_case_files, filter_excluded_split_files, load_split_record, resolve_excluded_case_numbers, select_case_files, split_case_files_for_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")

DATA_PROJECTION_SOURCE_TO_ISO = 0.75

DATA_PROJECTION_NUM_CIRCLE_POINTS = 120

DATA_DIFF_PROJECTOR_RADIAL_SUBSAMPLES = 2

DATA_DIFF_PROJECTOR_AXIAL_SUBSAMPLES = 4

DATA_DIFF_PROJECTOR_SPLAT_SIGMA_PX = 0.35

DATA_DIFF_PROJECTOR_SPLAT_RADIUS = 1

DATA_DIFF_PROJECTOR_BLUR_SIGMA_PX = 0.25

DATA_DIFF_PROJECTOR_BLUR_KERNEL_SIZE = 3

DATA_DIFF_PROJECTOR_INTENSITY_SCALE = 0.08

def _load_explicit_split(
    split_json_path: str | Path,
    *,
    discovered_files: list[Path],
    config: dict[str, Any],
) -> SplitFiles:
    """Load and validate a user-supplied train/val/test split.

    ``explicit_split_match_mode=case_name`` treats the supplied split as the
    canonical case assignment while remapping every entry to the matching file
    in the currently discovered feature directory. ``case_id`` instead uses
    the numeric case identifier, including the parent directory for layouts
    such as ``<case>/original.npz``. The default remains strict absolute-path
    matching.
    """
    path = Path(split_json_path).expanduser().resolve()
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Split file must contain a JSON object: {path}")
    split = load_split_record(path)

    split_paths = {
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }
    source_counts = {
        split_name: len(paths) for split_name, paths in split_paths.items()
    }
    source_total = sum(source_counts.values())
    declared_counts = payload.get("counts")
    if isinstance(declared_counts, dict):
        for key in ("train", "val", "test", "total"):
            if key not in declared_counts:
                continue
            actual = source_total if key == "total" else source_counts[key]
            if int(declared_counts[key]) != actual:
                raise ValueError(
                    f"Explicit split {path} declares counts[{key}]={declared_counts[key]}, "
                    f"but contains {actual}."
                )

    match_mode = str(
        config.get("explicit_split_match_mode", "exact_path")
    ).strip().lower()
    if match_mode not in {"exact_path", "case_name", "case_id"}:
        raise ValueError(
            "explicit_split_match_mode must be 'exact_path', 'case_name', or "
            "'case_id', "
            f"got {match_mode!r}."
        )
    missing_policy = str(
        config.get("explicit_split_missing_case_policy", "error")
    ).strip().lower()
    if missing_policy not in {"error", "drop"}:
        raise ValueError(
            "explicit_split_missing_case_policy must be 'error' or 'drop', "
            f"got {missing_policy!r}."
        )

    discovered_by_resolved = {
        file_path.expanduser().resolve(): file_path for file_path in discovered_files
    }
    discovered_by_case_name: dict[str, Path] = {}
    discovered_by_case_id: dict[int, Path] = {}
    if match_mode == "case_name":
        duplicate_case_names: dict[str, list[Path]] = {}
        for file_path in discovered_files:
            key = file_path.stem.casefold()
            previous = discovered_by_case_name.get(key)
            if previous is not None:
                duplicate_case_names.setdefault(key, [previous]).append(file_path)
            else:
                discovered_by_case_name[key] = file_path
        if duplicate_case_names:
            details = "; ".join(
                f"{case_name}: {', '.join(str(item) for item in files)}"
                for case_name, files in sorted(duplicate_case_names.items())
            )
            raise ValueError(
                "Cannot remap an explicit split by case name because the current "
                f"dataset contains duplicate case names: {details}"
            )
    elif match_mode == "case_id":
        duplicate_case_ids: dict[int, list[Path]] = {}
        for file_path in discovered_files:
            case_id = case_number_from_path(file_path)
            if case_id is None:
                raise ValueError(
                    "Cannot remap an explicit split by case ID because no numeric "
                    f"identifier can be inferred from discovered file {file_path}."
                )
            previous = discovered_by_case_id.get(case_id)
            if previous is not None:
                duplicate_case_ids.setdefault(case_id, [previous]).append(file_path)
            else:
                discovered_by_case_id[case_id] = file_path
        if duplicate_case_ids:
            details = "; ".join(
                f"{case_id}: {', '.join(str(item) for item in files)}"
                for case_id, files in sorted(duplicate_case_ids.items())
            )
            raise ValueError(
                "Cannot remap an explicit split by case ID because the current "
                f"dataset contains duplicate numeric case IDs: {details}"
            )

    resolved_splits: dict[str, list[Path]] = {}
    membership: dict[Path | str | int, str] = {}
    matched_discovered: set[Path] = set()
    missing_cases: list[dict[str, str]] = []
    for split_name, paths in split_paths.items():
        resolved_splits[split_name] = []
        local_seen: set[Path | str | int] = set()
        for raw_path in paths:
            if match_mode == "case_name":
                identity: Path | str | int = raw_path.stem.casefold()
                matched_path = discovered_by_case_name.get(identity)
            elif match_mode == "case_id":
                source_case_id = case_number_from_path(raw_path)
                if source_case_id is None:
                    raise ValueError(
                        f"Explicit split {path} contains {split_name} path "
                        f"{raw_path}, from which no numeric case ID can be inferred."
                    )
                identity = source_case_id
                matched_path = discovered_by_case_id.get(source_case_id)
            else:
                identity = raw_path.expanduser().resolve()
                matched_path = discovered_by_resolved.get(identity)
            if identity in local_seen:
                raise ValueError(
                    f"Explicit split {path} repeats case {raw_path.stem!r} within "
                    f"{split_name}."
                )
            if identity in membership:
                raise ValueError(
                    f"Explicit split {path} places case {raw_path.stem!r} in both "
                    f"{membership[identity]} and {split_name}."
                )
            local_seen.add(identity)
            membership[identity] = split_name
            if matched_path is None:
                missing_cases.append(
                    {
                        "split": split_name,
                        "case_name": raw_path.stem,
                        "source_path": str(raw_path),
                    }
                )
                if missing_policy == "error":
                    match_description = (
                        f"case name {raw_path.stem!r}"
                        if match_mode == "case_name"
                        else (
                            f"case ID {identity}"
                            if match_mode == "case_id"
                            else f"path {raw_path.expanduser().resolve()}"
                        )
                    )
                    raise ValueError(
                        f"Explicit split {path} contains {split_name} {match_description}, "
                        "which was not discovered under dataset_dir="
                        f"{config.get('dataset_dir', config.get('feature_dataset_dir'))!r}."
                    )
                continue
            resolved_match = matched_path.expanduser().resolve()
            if resolved_match in matched_discovered:
                raise ValueError(
                    f"Explicit split {path} maps more than one source entry to "
                    f"{matched_path}."
                )
            matched_discovered.add(resolved_match)
            resolved_splits[split_name].append(matched_path)

    record_config = payload.get("config")
    if isinstance(record_config, dict):
        recorded_exclusions = set(
            resolve_excluded_case_numbers(
                record_config.get("exclude_case_numbers")
            )
        )
        configured_exclusions = set(
            resolve_excluded_case_numbers(config.get("exclude_case_numbers"))
        )
        if not recorded_exclusions.issubset(configured_exclusions):
            raise ValueError(
                f"Explicit split {path} was generated with exclude_case_numbers="
                f"{record_config.get('exclude_case_numbers')!r}, but the raw training "
                f"config has exclude_case_numbers={config.get('exclude_case_numbers')!r}. "
                "An explicit split cannot restore cases that were excluded when it was created."
            )
        for key in ("num_cases", "case_fraction"):
            if key in record_config and record_config.get(key) != config.get(key):
                raise ValueError(
                    f"Explicit split {path} was generated with {key}="
                    f"{record_config.get(key)!r}, but the raw training config has "
                    f"{key}={config.get(key)!r}."
                )
        for key in ("train_ratio", "val_ratio", "test_ratio"):
            if key not in record_config:
                continue
            recorded = float(record_config[key])
            configured = float(config.get(key, 0.0))
            if not math.isclose(recorded, configured, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Explicit split {path} was generated with {key}={recorded}, "
                    f"but the raw training config has {key}={configured}."
                )

    filtered_split, removed_explicit_cases = filter_excluded_split_files(
        SplitFiles(
            train=resolved_splits["train"],
            val=resolved_splits["val"],
            test=resolved_splits["test"],
        ),
        config.get("exclude_case_numbers"),
    )
    resolved_splits = {
        "train": list(filtered_split.train),
        "val": list(filtered_split.val),
        "test": list(filtered_split.test),
    }
    actual_counts = {
        split_name: len(paths) for split_name, paths in resolved_splits.items()
    }
    actual_total = sum(actual_counts.values())
    validate_ratio_counts = config.get(
        "explicit_split_validate_ratio_counts", True
    )
    if not isinstance(validate_ratio_counts, bool):
        raise ValueError(
            "explicit_split_validate_ratio_counts must be boolean"
        )
    config["explicit_split_validate_ratio_counts"] = validate_ratio_counts
    if match_mode == "exact_path" and validate_ratio_counts:
        eligible_discovered_files, _ = filter_excluded_case_files(
            discovered_files,
            config.get("exclude_case_numbers"),
        )
        selected = select_case_files(
            eligible_discovered_files,
            num_cases=config.get("num_cases"),
            case_fraction=config.get("case_fraction"),
            seed=int(config.get("seed", 0)),
            shuffle=bool(config.get("shuffle_cases", True)),
        )
        expected = split_case_files_for_experiment(
            all_files=eligible_discovered_files,
            selected_files=selected,
            train_ratio=float(config.get("train_ratio", 0.8)),
            val_ratio=float(config.get("val_ratio", 0.1)),
            test_ratio=float(config.get("test_ratio", 0.1)),
            seed=int(config.get("seed", 0)),
            shuffle=bool(config.get("shuffle_cases", True)),
        )
        expected_counts = {
            "train": len(expected.train),
            "val": len(expected.val),
            "test": len(expected.test),
        }
        if (
            not removed_explicit_cases
            and not missing_cases
            and actual_counts != expected_counts
        ):
            raise ValueError(
                f"Explicit split {path} counts {actual_counts} do not match the raw "
                f"training config's num_cases/case_fraction and ratios, which imply "
                f"{expected_counts}."
            )
        if (
            not removed_explicit_cases
            and not missing_cases
            and actual_total != len(selected)
        ):
            raise ValueError(
                f"Explicit split {path} contains {actual_total} cases, but the raw "
                f"training config selects {len(selected)} cases."
            )

    unassigned_discovered_cases = [
        {
            "case_name": file_path.stem,
            "path": str(file_path),
        }
        for file_path in discovered_files
        if file_path.expanduser().resolve() not in matched_discovered
    ]
    config["explicit_split_source_path"] = str(path)
    config["explicit_split_match_mode"] = match_mode
    config["explicit_split_missing_case_policy"] = missing_policy
    config["explicit_split_missing_cases"] = missing_cases
    config["explicit_split_excluded_cases"] = [
        {"case_name": item.stem, "path": str(item)}
        for item in removed_explicit_cases
    ]
    config["explicit_split_unassigned_discovered_cases"] = (
        unassigned_discovered_cases
    )

    exclusion_note = (
        f"; ignored {len(removed_explicit_cases)} configured case(s)"
        if removed_explicit_cases
        else ""
    )
    missing_note = (
        f"; dropped {len(missing_cases)} case(s) absent from the current dataset"
        if missing_cases
        else ""
    )
    print(
        f"[Split] loaded explicit split {path}: "
        f"train={actual_counts['train']} val={actual_counts['val']} "
        f"test={actual_counts['test']}{exclusion_note}{missing_note}"
    )
    if missing_cases:
        print(
            "[Split] absent cases dropped without reshuffling: "
            + ", ".join(
                f"{item['case_name']} ({item['split']})" for item in missing_cases
            )
        )
    return SplitFiles(
        train=resolved_splits["train"],
        val=resolved_splits["val"],
        test=resolved_splits["test"],
    )

_GEOMETRY_LOSS_SCHEDULE_DEFAULTS = {
    "tangent_curve_loss": {
        "start_epoch": 40,
        "end_epoch": 100,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
    "branch_length_loss": {
        "start_epoch": 60,
        "end_epoch": 120,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
    "relative_derivative_loss": {
        "start_epoch": 30,
        "end_epoch": 60,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
    "local_progress_loss": {
        "start_epoch": 1,
        "end_epoch": 20,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
    "asymmetric_curvature_loss": {
        "start_epoch": 20,
        "end_epoch": 60,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
    "bounded_curvature_vector_loss": {
        "start_epoch": 20,
        "end_epoch": 60,
        "start_factor": 0.0,
        "end_factor": 1.0,
    },
}

_HISTORY_LOSS_WEIGHT_DEFAULTS = {
    "existence_loss": ("existence_loss_weight", 0.0),
    "point_loss": ("point_loss_weight", 0.0),
    "xyz_loss": ("xyz_loss_weight", 1.0),
    "radius_loss": ("radius_loss_weight", 1.0),
    "tangent_loss": ("tangent_loss_weight", 0.1),
    "curve_loss": ("curve_loss_weight", 0.1),
    "branch_length_loss": ("branch_length_loss_weight", 0.1),
    "endpoint_vector_loss": ("endpoint_vector_loss_weight", 0.0),
    "chamfer_loss": ("chamfer_loss_weight", 0.1),
    "pred_to_gt_curve_loss": ("pred_to_gt_curve_loss_weight", 0.0),
    "gt_to_pred_curve_loss": ("gt_to_pred_curve_loss_weight", 0.0),
    "detail_loss": ("detail_loss_weight", 0.0),
    "relative_first_difference_loss": ("relative_first_difference_loss_weight", 0.0),
    "relative_second_difference_loss": ("relative_second_difference_loss_weight", 0.0),
    "local_progress_loss": ("local_progress_loss_weight", 0.0),
    "arc_resample_loss": ("arc_resample_loss_weight", 0.0),
    "multiscale_topk_curve_loss": ("multiscale_topk_curve_loss_weight", 0.0),
    "curvature_weighted_xyz_loss": ("curvature_weighted_xyz_loss_weight", 0.0),
    "curvature_weighted_tangent_loss": ("curvature_weighted_tangent_loss_weight", 0.0),
    "curvature_weighted_curve_loss": ("curvature_weighted_curve_loss_weight", 0.0),
    "side_relative_xyz_loss": ("side_relative_xyz_loss_weight", 0.0),
    "asymmetric_curvature_loss": ("asymmetric_curvature_loss_weight", 0.0),
    "curvature_underbend_loss": ("asymmetric_curvature_loss_weight", 0.0),
    "curvature_excess_loss": ("asymmetric_curvature_loss_weight", 0.0),
    "curvature_direction_loss": ("asymmetric_curvature_loss_weight", 0.0),
    "bounded_curvature_vector_loss": ("bounded_curvature_vector_loss_weight", 0.0),
    "bounded_curvature_vector_weighted_loss": (
        "bounded_curvature_vector_loss_weight",
        0.0,
    ),
    "bounded_curvature_under_loss": ("bounded_curvature_vector_loss_weight", 0.0),
    "bounded_curvature_over_loss": ("bounded_curvature_vector_loss_weight", 0.0),
    "bounded_curvature_perpendicular_loss": (
        "bounded_curvature_vector_loss_weight",
        0.0,
    ),
    "bounded_curvature_tangent_loss": ("bounded_curvature_vector_loss_weight", 0.0),
    "bounded_curvature_upper_exceedance_fraction": (
        "bounded_curvature_vector_loss_weight",
        0.0,
    ),
    "bounded_curvature_lower_shortfall_fraction": (
        "bounded_curvature_vector_loss_weight",
        0.0,
    ),
    "bounded_curvature_mean_importance": (
        "bounded_curvature_vector_loss_weight",
        0.0,
    ),
    "bounded_curvature_ratio_p50": ("bounded_curvature_vector_loss_weight", 0.0),
    "bounded_curvature_ratio_p95": ("bounded_curvature_vector_loss_weight", 0.0),
    "attachment_loss": ("attachment_loss_weight", 0.0),
    "occupancy_3d_ssim_loss": ("occupancy_3d_ssim_loss_weight", 0.0),
}

_BRANCH_LENGTH_COMPONENT_HISTORY_KEYS = {
    "branch_length_global_loss",
    "branch_length_local_deficit_loss",
    "branch_length_smooth_excess_loss",
    "branch_length_high_region_fraction",
    "branch_length_smooth_region_fraction",
}

_TERMINAL_POINT_HISTORY_KEYS = {
    "xyz_unweighted_loss",
    "xyz_terminal_start_loss",
    "xyz_terminal_end_loss",
    "centerline_terminal_parameter_mean_weight",
}

_OPTIONAL_HISTORY_LOSS_KEYS = set(_HISTORY_LOSS_WEIGHT_DEFAULTS) | {
    *_BRANCH_LENGTH_COMPONENT_HISTORY_KEYS,
    *_TERMINAL_POINT_HISTORY_KEYS,
    "centerline_3d_dt_loss",
    "proj_loss",
    "proj_weighted_loss",
    "proj_centerline_loss",
    "proj_centerline_unweighted_loss",
    "proj_centerline_mae_px",
    "proj_centerline_valid_fraction",
    "proj_centerline_mask_loss",
    "proj_centerline_mask_distance_px",
    "projected_inside_mask_fraction",
    "proj_centerline_mountain_region_fraction",
    "proj_centerline_mountain_peak_activation",
    "proj_centerline_mountain_mean_weight",
    "proj_centerline_tangent_loss",
    "proj_centerline_tangent_angle_mae_deg",
    "proj_centerline_tangent_valid_fraction",
    "proj_centerline_curvature_loss",
    "proj_centerline_curvature_mae_inv_px",
    "proj_centerline_curvature_valid_fraction",
    "proj_dice_loss",
    "proj_dt_ssim_loss",
    "effective_proj_loss_weight",
    "projection_evidence_coarse_loss",
    "projection_evidence_coarse_loss_weight",
}

_LOSS_CONTRIBUTION_LABELS = {
    "existence_loss": "Existence",
    "point_loss": "Point",
    "xyz_loss": "XYZ",
    "radius_loss": "Radius",
    "tangent_loss": "Tangent",
    "curve_loss": "Curve",
    "branch_length_loss": "Branch length",
    "endpoint_vector_loss": "Endpoint vector",
    "chamfer_loss": "Chamfer",
    "pred_to_gt_curve_loss": "Pred-to-GT curve",
    "gt_to_pred_curve_loss": "GT-to-pred curve",
    "detail_loss": "Detail",
    "relative_first_difference_loss": "Relative H1",
    "relative_second_difference_loss": "Relative H2",
    "local_progress_loss": "Local progress",
    "arc_resample_loss": "Arc resample",
    "multiscale_topk_curve_loss": "Top-k curve",
    "curvature_weighted_xyz_loss": "Curvature-weighted XYZ",
    "curvature_weighted_tangent_loss": "Curvature-weighted tangent",
    "curvature_weighted_curve_loss": "Curvature-weighted curve",
    "side_relative_xyz_loss": "Side-relative XYZ",
    "asymmetric_curvature_loss": "Asymmetric curvature",
    "attachment_loss": "Attachment",
    "occupancy_3d_ssim_loss": "3D occupancy SSIM",
}

_EFFECTIVE_WEIGHT_HISTORY_KEYS = {
    "tangent_loss": "effective_tangent_loss_weight",
    "curve_loss": "effective_curve_loss_weight",
    "branch_length_loss": "effective_branch_length_loss_weight",
    "relative_first_difference_loss": "effective_relative_first_difference_loss_weight",
    "relative_second_difference_loss": "effective_relative_second_difference_loss_weight",
    "local_progress_loss": "effective_local_progress_loss_weight",
    "asymmetric_curvature_loss": "effective_asymmetric_curvature_loss_weight",
}
