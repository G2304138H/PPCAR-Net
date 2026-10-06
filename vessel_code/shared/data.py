# Transferred from methods/src/data.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

RESNET_PRE_FPN_SUFFIXES = ("c2", "c3", "c4", "c5")

CLINICAL_RCA_VIEW_INDICES = (0, 1, 5)

SUPPORTED_FEATURE_BACKBONES = ("resnet_pre_fpn", "vggt", "vggt_omega")

TARGET_COORDINATE_FRAMES = ("absolute_world", "projection_centered")

DEFAULT_EXCLUDED_DATASET_DIR_NAMES = ("removecase",)

def _payload_has_key(payload: Any, key: str) -> bool:
    """Check an NPZ/mapping key without materialising an NPZ array.

    ``NpzFile`` inherits ``Mapping.__contains__``, whose implementation calls
    ``payload[key]``.  For compressed feature members that silently performs a
    complete decompression just to answer a membership question.
    """

    files = getattr(payload, "files", None)
    if files is not None:
        return key in files
    return key in payload

def resolve_zero_cached_image_features(
    config: dict[str, Any],
    default: bool = False,
) -> bool:
    raw_value = config.get("zero_cached_image_features")
    if raw_value is None:
        return bool(default)
    if not isinstance(raw_value, bool):
        raise ValueError("zero_cached_image_features must be a boolean or null when provided.")
    return raw_value

def resolve_lazy_load_image_features(
    config: Mapping[str, Any],
    default: bool = False,
) -> bool:
    """Resolve whether cached backbone arrays are materialised on demand.

    Lazy loading is intentionally opt-in so existing callers retain their
    eager-loading behaviour.  It applies only to precomputed feature NPZs;
    online backbones already have their own image-loading path.
    """

    raw_value = config.get("lazy_load_image_features")
    if raw_value is None:
        return bool(default)
    if not isinstance(raw_value, bool):
        raise ValueError(
            "lazy_load_image_features must be a JSON boolean or null when "
            f"provided, got {raw_value!r}."
        )
    return raw_value

def resolve_target_coordinate_frame(
    config: dict[str, Any],
    default: str = "projection_centered",
) -> str:
    raw_value = config.get("target_coordinate_frame")
    legacy_flag = config.get("center_targets_on_projection_offset")
    if "center_targets_on_projection_offset" in config and not isinstance(legacy_flag, bool):
        raise ValueError("center_targets_on_projection_offset must be a boolean when provided.")
    if raw_value is None and legacy_flag is not None:
        raw_value = "projection_centered" if legacy_flag else "absolute_world"
    value = str(default if raw_value is None else raw_value).strip().lower().replace("-", "_")
    aliases = {
        "absolute": "absolute_world",
        "world": "absolute_world",
        "uncentered": "absolute_world",
        "centered": "projection_centered",
        "projection": "projection_centered",
        "projection_centred": "projection_centered",
    }
    value = aliases.get(value, value)
    if value not in TARGET_COORDINATE_FRAMES:
        raise ValueError(
            f"target_coordinate_frame must be one of {TARGET_COORDINATE_FRAMES}, got {raw_value!r}."
        )
    if legacy_flag is not None and (value == "projection_centered") != legacy_flag:
        raise ValueError(
            "target_coordinate_frame conflicts with center_targets_on_projection_offset. "
            "Use one coordinate-frame setting consistently."
        )
    return value

@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    val: list[Path]
    test: list[Path]

def normalize_feature_backbone(value: str) -> str:
    backbone = str(value).strip().lower().replace("-", "_")
    aliases = {
        "resnet": "resnet_pre_fpn",
        "resent": "resnet_pre_fpn",
        "resent_pre_fpn": "resnet_pre_fpn",
        "resnet_prefpn": "resnet_pre_fpn",
        "resnet_pre_fpn": "resnet_pre_fpn",
        "vggt": "vggt",
        "vggt_original": "vggt",
        "vggt_omega": "vggt_omega",
        "omega": "vggt_omega",
    }
    backbone = aliases.get(backbone, backbone)
    if backbone not in SUPPORTED_FEATURE_BACKBONES:
        raise ValueError(
            f"feature_backbone must be one of {SUPPORTED_FEATURE_BACKBONES}, got {value!r}."
        )
    return backbone

def parse_view_indices(raw_value: Any, default: Sequence[int] = CLINICAL_RCA_VIEW_INDICES) -> tuple[int, ...]:
    if raw_value is None:
        values = tuple(int(value) for value in default)
    elif isinstance(raw_value, str):
        values = tuple(int(part.strip()) for part in raw_value.replace(";", ",").split(",") if part.strip())
    else:
        values = tuple(int(value) for value in raw_value)
    if not values:
        raise ValueError("View index list must contain at least one index.")
    if min(values) < 0:
        raise ValueError(f"View indices must be non-negative, got {values}.")
    if len(set(values)) != len(values):
        raise ValueError(f"View indices contain duplicates: {values}.")
    return values

def resolve_clinical_rca_view_indices(
    config: dict[str, Any],
    feature_backbone: str,
) -> tuple[int, ...] | None:
    """Resolve optional clinical-view subsetting independently of the backbone."""
    normalize_feature_backbone(feature_backbone)
    raw_mode = config.get("view_selection_mode")
    mode = "" if raw_mode is None else str(raw_mode).strip().lower()
    clinical_requested = bool(config.get("clinical_rca_views", False)) or mode in {
        "clinical",
        "clinical_rca",
        "clinical_rca_views",
        "clinica_rca_view",
        "clinica_rca_views",
        "rca_clinical_views",
    }
    if not clinical_requested:
        return None
    return parse_view_indices(config.get("clinical_rca_view_indices", CLINICAL_RCA_VIEW_INDICES))

def resolve_excluded_dataset_dir_names(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        values = DEFAULT_EXCLUDED_DATASET_DIR_NAMES
    elif isinstance(raw_value, str):
        values = tuple(part.strip() for part in raw_value.replace(";", ",").split(",") if part.strip())
    elif isinstance(raw_value, (list, tuple, set)):
        values = tuple(str(value).strip() for value in raw_value if str(value).strip())
    else:
        raise ValueError(
            "exclude_dataset_dir_names must be null, a comma-separated string, or a list of names."
        )
    invalid = [value for value in values if value in {".", ".."} or Path(value).name != value]
    if invalid:
        raise ValueError(
            "exclude_dataset_dir_names must contain directory names, not paths; "
            f"got {invalid}."
        )
    return tuple(dict.fromkeys(value.casefold() for value in values))

def resolve_excluded_case_numbers(raw_value: Any) -> tuple[int, ...]:
    """Normalize configured numeric case identifiers such as ``"0421"``."""
    if raw_value is None:
        values: list[Any] = []
    elif isinstance(raw_value, bool):
        raise ValueError(
            "exclude_case_numbers must be null, an integer/string case number, "
            "or a list of case numbers."
        )
    elif isinstance(raw_value, (int, str)):
        if isinstance(raw_value, str):
            values = [
                part.strip()
                for part in raw_value.replace(";", ",").split(",")
                if part.strip()
            ]
        else:
            values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        raise ValueError(
            "exclude_case_numbers must be null, an integer/string case number, "
            "or a list of case numbers."
        )

    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(
                f"exclude_case_numbers entries must be non-negative integers or digit strings, got {value!r}."
            )
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            number = int(value.strip())
        else:
            raise ValueError(
                f"exclude_case_numbers entries must be non-negative integers or digit strings, got {value!r}."
            )
        if number < 0:
            raise ValueError(
                f"exclude_case_numbers entries must be non-negative, got {value!r}."
            )
        normalized.append(number)
    return tuple(dict.fromkeys(normalized))

def case_number_from_path(path: str | Path) -> int | None:
    """Return the numeric case identifier used by RCA/LCA feature files."""
    value = Path(path)
    if value.parent.name.isdigit():
        return int(value.parent.name)
    groups = re.findall(r"\d+", value.stem)
    return int(groups[-1]) if groups else None

def filter_excluded_case_files(
    files: Iterable[str | Path],
    exclude_case_numbers: Any,
) -> tuple[list[Path], list[Path]]:
    """Partition files into eligible and config-excluded case-number lists."""
    excluded_numbers = set(resolve_excluded_case_numbers(exclude_case_numbers))
    eligible: list[Path] = []
    excluded: list[Path] = []
    for raw_path in files:
        path = Path(raw_path)
        target = excluded if case_number_from_path(path) in excluded_numbers else eligible
        target.append(path)
    return eligible, excluded

def filter_excluded_split_files(
    split: SplitFiles,
    exclude_case_numbers: Any,
) -> tuple[SplitFiles, list[Path]]:
    """Remove excluded cases from every split, including an explicit split."""
    train, removed_train = filter_excluded_case_files(
        split.train, exclude_case_numbers
    )
    val, removed_val = filter_excluded_case_files(split.val, exclude_case_numbers)
    test, removed_test = filter_excluded_case_files(
        split.test, exclude_case_numbers
    )
    removed_by_path = {
        path.expanduser().resolve(): path
        for path in (*removed_train, *removed_val, *removed_test)
    }
    return (
        SplitFiles(train=train, val=val, test=test),
        list(removed_by_path.values()),
    )

def discover_npz_files(
    dataset_dir: str | Path | Iterable[str | Path],
    recursive: bool = True,
    exclude_dir_names: Any = None,
) -> list[Path]:
    if isinstance(dataset_dir, (str, Path)):
        roots = [Path(dataset_dir)]
    else:
        roots = [Path(root) for root in dataset_dir]
    excluded_names = set(resolve_excluded_dataset_dir_names(exclude_dir_names))
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix == ".npz":
                files.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        pattern = "**/*.npz" if recursive else "*.npz"
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.casefold() in excluded_names for part in relative.parts[:-1]):
                continue
            files.append(path)
    files = sorted({path.resolve() for path in files if path.is_file()})
    if not files:
        raise FileNotFoundError(f"No .npz files found under {roots}")
    return files

def select_case_files(
    files: list[Path],
    num_cases: int | None = None,
    case_fraction: float | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[Path]:
    selected = list(files)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(selected)

    if case_fraction is not None:
        fraction = float(case_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"case_fraction must be in (0, 1], got {case_fraction}")
        limit = max(1, int(math.ceil(len(selected) * fraction)))
        selected = selected[:limit]

    if num_cases is not None:
        limit = int(num_cases)
        if limit < 1:
            raise ValueError(f"num_cases must be >= 1, got {num_cases}")
        selected = selected[:limit]

    if not selected:
        raise ValueError("Case selection produced an empty dataset.")
    return selected

def split_case_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
    shuffle: bool = True,
) -> SplitFiles:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0.0) or float(ratios.sum()) <= 0.0:
        raise ValueError(
            f"Split ratios must be non-negative and have positive sum, got {ratios.tolist()}."
        )
    ratios = ratios / ratios.sum()
    ordered = list(files)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(ordered)

    n_total = len(ordered)
    if n_total == 1:
        return SplitFiles(
            train=ordered if ratios[0] > 0.0 else [],
            val=ordered if ratios[0] <= 0.0 and ratios[1] > 0.0 else [],
            test=ordered if ratios[0] <= 0.0 and ratios[1] <= 0.0 else [],
        )

    n_train = int(math.floor(n_total * float(ratios[0])))
    n_val = int(math.floor(n_total * float(ratios[1])))
    if n_total >= 2 and ratios[0] > 0.0 and n_train == 0:
        n_train = 1
    if n_total >= 3 and ratios[1] > 0.0 and n_val == 0:
        n_val = 1
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)
    n_test = n_total - n_train - n_val
    if n_total >= 3 and ratios[2] > 0.0 and n_test == 0:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        n_test = n_total - n_train - n_val

    return SplitFiles(
        train=ordered[:n_train],
        val=ordered[n_train : n_train + n_val],
        test=ordered[n_train + n_val :],
    )

def zero_val_means_train_split(val_ratio: float) -> bool:
    return math.isclose(float(val_ratio), 0.0)

def zero_test_means_remainder_split(test_ratio: float) -> bool:
    return math.isclose(float(test_ratio), 0.0)

def split_case_files_for_experiment(
    all_files: list[Path],
    selected_files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
    shuffle: bool = True,
) -> SplitFiles:
    split = split_case_files(
        selected_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        shuffle=shuffle,
    )
    train_paths = list(split.train)
    val_paths = list(train_paths) if zero_val_means_train_split(val_ratio) else list(split.val)
    if zero_test_means_remainder_split(test_ratio):
        excluded = {path.resolve() for path in train_paths}
        excluded.update(path.resolve() for path in val_paths)
        test_paths = [path for path in all_files if path.resolve() not in excluded]
    else:
        test_paths = list(split.test)
    return SplitFiles(
        train=train_paths,
        val=val_paths,
        test=test_paths,
    )

def _case_file_records(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "case_name": file_path.stem,
            "path": str(file_path),
        }
        for file_path in paths
    ]

def save_split_record(path: Path, split: SplitFiles, config: dict[str, Any]) -> None:
    payload = {
        "schema_version": 2,
        "train": [str(path) for path in split.train],
        "val": [str(path) for path in split.val],
        "test": [str(path) for path in split.test],
        "splits": {
            "train": _case_file_records(split.train),
            "val": _case_file_records(split.val),
            "test": _case_file_records(split.test),
        },
        "counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "total": len(split.train) + len(split.val) + len(split.test),
            "unique_total": len({path.resolve() for path in [*split.train, *split.val, *split.test]}),
        },
        "config": {
            "dataset_dir": config.get("dataset_dir"),
            "feature_backbone": config.get("feature_backbone"),
            "zero_cached_image_features": config.get("zero_cached_image_features"),
            "experiment_name": config.get("experiment_name"),
            "recursive": config.get("recursive"),
            "exclude_dataset_dir_names": config.get("exclude_dataset_dir_names"),
            "exclude_case_numbers": config.get("exclude_case_numbers"),
            "num_cases": config.get("num_cases"),
            "case_fraction": config.get("case_fraction"),
            "train_ratio": config.get("train_ratio"),
            "val_ratio": config.get("val_ratio"),
            "test_ratio": config.get("test_ratio"),
            "seed": config.get("seed"),
            "shuffle_cases": config.get("shuffle_cases"),
            "explicit_split_source_path": config.get(
                "explicit_split_source_path"
            ),
            "explicit_split_match_mode": config.get(
                "explicit_split_match_mode"
            ),
            "explicit_split_missing_case_policy": config.get(
                "explicit_split_missing_case_policy"
            ),
            "explicit_split_missing_cases": config.get(
                "explicit_split_missing_cases"
            ),
            "explicit_split_excluded_cases": config.get(
                "explicit_split_excluded_cases"
            ),
            "explicit_split_unassigned_discovered_cases": config.get(
                "explicit_split_unassigned_discovered_cases"
            ),
            "effective_split_mode": config.get("effective_split_mode"),
            "zero_val_means_train": config.get("zero_val_means_train"),
            "zero_test_means_remainder": config.get("zero_test_means_remainder"),
            "num_discovered_cases": config.get("num_discovered_cases"),
            "num_discovered_cases_before_case_exclusion": config.get(
                "num_discovered_cases_before_case_exclusion"
            ),
            "num_excluded_cases": config.get("num_excluded_cases"),
            "num_selected_cases": config.get("num_selected_cases"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load_split_record(path: str | Path) -> SplitFiles:
    with open(path, "r") as f:
        payload = json.load(f)
    splits = payload.get("splits")
    if isinstance(splits, dict):
        def paths_from_records(split_name: str) -> list[Path]:
            records = splits.get(split_name, [])
            if not isinstance(records, list):
                return []
            out: list[Path] = []
            for record in records:
                if isinstance(record, dict) and record.get("path"):
                    out.append(Path(str(record["path"])))
                elif isinstance(record, str):
                    out.append(Path(record))
            return out

        return SplitFiles(
            train=paths_from_records("train"),
            val=paths_from_records("val"),
            test=paths_from_records("test"),
        )
    return SplitFiles(
        train=[Path(p) for p in payload.get("train", [])],
        val=[Path(p) for p in payload.get("val", [])],
        test=[Path(p) for p in payload.get("test", [])],
    )

def _scale_artery_to_mm(artery: np.ndarray, scale_to_mm: float) -> np.ndarray:
    if float(scale_to_mm) == 1.0:
        return artery.astype(np.float32, copy=False)
    out = artery.astype(np.float32, copy=True)
    out[:, :, :4] *= float(scale_to_mm)
    return out

def _resample_points(points: np.ndarray, out_points: int) -> np.ndarray:
    in_points = int(points.shape[0])
    if in_points == int(out_points):
        return points.astype(np.float32)
    if in_points <= 1:
        return np.repeat(points[:1], repeats=int(out_points), axis=0).astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=in_points, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=int(out_points), dtype=np.float32)
    out = np.zeros((int(out_points), points.shape[1]), dtype=np.float32)
    for channel_index in range(points.shape[1]):
        out[:, channel_index] = np.interp(x_new, x_old, points[:, channel_index]).astype(np.float32)
    return out

def _pick_existing_flag_array(payload: Any, num_branches: int) -> np.ndarray | None:
    for key in ("branch_exist", "branch_exists", "exist_flags", "branch_mask"):
        if _payload_has_key(payload, key):
            arr = np.asarray(payload[key]).astype(np.float32).reshape(-1)
            if arr.size < 1:
                raise ValueError(f"{key} must contain at least one branch label.")
            padded = np.zeros((int(num_branches),), dtype=np.float32)
            copied = min(int(arr.size), int(num_branches))
            padded[:copied] = (arr[:copied] > 0.5).astype(np.float32)
            return padded
    return None

def build_fixed_targets(
    artery: np.ndarray,
    num_branches: int,
    num_points: int,
    exist_flags: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if artery.ndim != 3 or artery.shape[-1] < 4:
        raise ValueError(f"Expected artery shape [M,N,4+], got {artery.shape}")

    branch_points = np.zeros((int(num_branches), int(num_points), 4), dtype=np.float32)
    target_exist = np.zeros((int(num_branches),), dtype=np.float32)
    available_branches = min(int(artery.shape[0]), int(num_branches))
    for branch_idx in range(available_branches):
        branch_points[branch_idx] = _resample_points(
            artery[branch_idx, :, :4].astype(np.float32),
            out_points=int(num_points),
        )

    if exist_flags is not None:
        flags = np.asarray(exist_flags, dtype=np.float32).reshape(-1)
        copied = min(available_branches, int(flags.size))
        target_exist[:copied] = flags[:copied]
    else:
        for branch_idx in range(available_branches):
            if np.any(artery[branch_idx, :, 3] > 1e-6):
                target_exist[branch_idx] = 1.0
    target_exist[0] = 1.0
    return branch_points, target_exist

def _view_features_from_angles(payload: Any) -> np.ndarray | None:
    theta = None
    phi = None
    for key in ("theta_deg", "theta_array", "theta"):
        if _payload_has_key(payload, key):
            theta = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            break
    for key in ("phi_deg", "phi_array", "phi"):
        if _payload_has_key(payload, key):
            phi = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            break
    if theta is None or phi is None:
        return None
    if theta.shape[0] != phi.shape[0]:
        raise ValueError(f"theta and phi view arrays have different lengths: {theta.shape[0]} vs {phi.shape[0]}")
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return np.stack(
        [np.sin(theta_rad), np.cos(theta_rad), np.sin(phi_rad), np.cos(phi_rad)],
        axis=-1,
    ).astype(np.float32)

def _optional_array(payload: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if _payload_has_key(payload, key):
            return np.asarray(payload[key])
    return None

def _load_centerline_dt_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if not _payload_has_key(data, "centerline_dt_volume"):
            raise KeyError(f"{path} does not contain required key 'centerline_dt_volume'.")
        if not _payload_has_key(data, "centerline_dt_origin"):
            raise KeyError(f"{path} does not contain required key 'centerline_dt_origin'.")
        if not _payload_has_key(data, "centerline_dt_spacing"):
            raise KeyError(f"{path} does not contain required key 'centerline_dt_spacing'.")
        volume = np.asarray(data["centerline_dt_volume"], dtype=np.float32)
        origin = np.asarray(data["centerline_dt_origin"], dtype=np.float32).reshape(3)
        spacing = np.asarray(data["centerline_dt_spacing"], dtype=np.float32).reshape(3)
        max_distance = (
            float(np.asarray(data["centerline_dt_max_distance_mm"]).reshape(()))
            if _payload_has_key(data, "centerline_dt_max_distance_mm")
            else float(np.nan)
        )
    if volume.ndim != 3:
        raise ValueError(f"{path} centerline_dt_volume must have shape [D,H,W], got {volume.shape}.")
    if np.any(spacing <= 0.0):
        raise ValueError(f"{path} centerline_dt_spacing must be positive, got {spacing.tolist()}.")
    return {
        "centerline_dt_volume": volume,
        "centerline_dt_origin": origin,
        "centerline_dt_spacing": spacing,
        "centerline_dt_max_distance_mm": np.asarray(max_distance, dtype=np.float32),
        "centerline_dt_valid": np.asarray(True, dtype=np.bool_),
    }

def _format_centerline_dt_resolution_name(raw_value: Any) -> str:
    spacing = parse_centerline_dt_resolution(raw_value)
    if spacing is None:
        raise ValueError("Cannot format a null centerline DT resolution.")

    def fmt(value: float) -> str:
        text = f"{float(value):.6g}"
        return "0" if text == "-0" else text

    if np.allclose(spacing, spacing[0]):
        return fmt(float(spacing[0]))
    return "x".join(fmt(float(value)) for value in spacing.tolist())

def parse_centerline_dt_resolution(raw_value: Any) -> np.ndarray | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text == "":
            return None
        if text.endswith("mm"):
            text = text[:-2]
        if text.startswith("["):
            raw_value = json.loads(text)
        elif "x" in text.lower():
            raw_value = [part.strip() for part in text.lower().split("x") if part.strip()]
        else:
            raw_value = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        spacing = np.asarray([float(raw_value)] * 3, dtype=np.float32)
    else:
        spacing = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        if spacing.shape[0] == 1:
            spacing = np.repeat(spacing, 3)
    if spacing.shape[0] != 3:
        raise ValueError(f"centerline 3D DT resolution must be a scalar or 3-vector, got {spacing.tolist()}.")
    if np.any(spacing <= 0.0):
        raise ValueError(f"centerline 3D DT resolution must be positive, got {spacing.tolist()}.")
    return spacing.astype(np.float32)

def _centerline_dt_root_candidates(
    centerline_dt_dir: str | Path | None,
    centerline_dt_resolution_mm: Any,
) -> list[Path]:
    if centerline_dt_dir is None or str(centerline_dt_dir) == "":
        return []
    root = Path(centerline_dt_dir)
    resolution = parse_centerline_dt_resolution(centerline_dt_resolution_mm)
    if resolution is None:
        return [root]
    name = _format_centerline_dt_resolution_name(resolution)
    variants = [
        name,
        f"{name}mm",
        f"voxel_{name}",
        f"voxel_{name}mm",
        f"resolution_{name}",
        f"resolution_{name}mm",
    ]
    candidates: list[Path] = []
    for variant in variants:
        candidate = root / variant
        if candidate not in candidates:
            candidates.append(candidate)
    candidates.append(root)
    return candidates

def _resolve_centerline_dt_path(
    case_path: Path,
    centerline_dt_dir: str | Path | None,
    dataset_root: str | Path | None,
    centerline_dt_resolution_mm: Any = None,
) -> Path | None:
    dt_roots = _centerline_dt_root_candidates(centerline_dt_dir, centerline_dt_resolution_mm)
    if not dt_roots:
        return None
    candidates: list[Path] = []
    for dt_root in dt_roots:
        if dataset_root is not None:
            try:
                rel_path = case_path.resolve().relative_to(Path(dataset_root).resolve())
                candidates.append(dt_root / rel_path)
            except ValueError:
                pass
        candidates.extend(
            [
                dt_root / case_path.name,
                dt_root / f"{case_path.stem}.npz",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _embedded_centerline_dt_payload(payload: Any) -> dict[str, Any] | None:
    if not _payload_has_key(payload, "centerline_dt_volume"):
        return None
    volume = np.asarray(payload["centerline_dt_volume"], dtype=np.float32)
    origin = np.asarray(payload["centerline_dt_origin"], dtype=np.float32).reshape(3)
    spacing = np.asarray(payload["centerline_dt_spacing"], dtype=np.float32).reshape(3)
    max_distance = (
        float(np.asarray(payload["centerline_dt_max_distance_mm"]).reshape(()))
        if _payload_has_key(payload, "centerline_dt_max_distance_mm")
        else float(np.nan)
    )
    if volume.ndim != 3:
        raise ValueError(f"Embedded centerline_dt_volume must have shape [D,H,W], got {volume.shape}.")
    if np.any(spacing <= 0.0):
        raise ValueError(f"Embedded centerline_dt_spacing must be positive, got {spacing.tolist()}.")
    return {
        "centerline_dt_volume": volume,
        "centerline_dt_origin": origin,
        "centerline_dt_spacing": spacing,
        "centerline_dt_max_distance_mm": np.asarray(max_distance, dtype=np.float32),
        "centerline_dt_valid": np.asarray(True, dtype=np.bool_),
    }

def _feature_keys(feature_backbone: str, feature_key: str) -> list[str]:
    if normalize_feature_backbone(feature_backbone) == "resnet_pre_fpn":
        return [f"{feature_key}_{suffix}" for suffix in RESNET_PRE_FPN_SUFFIXES]
    return [feature_key]

def _stored_feature_shapes(
    *,
    path: Path,
    payload: Any,
    backbone: str,
    feature_key: str,
) -> dict[str, tuple[int, ...]]:
    """Read cached-feature shapes from lightweight NPZ metadata.

    Accessing an array in a compressed ``NpzFile`` decompresses that complete
    member.  Official precompute outputs therefore store shapes separately so
    lazy dataset discovery can validate the cache without materialising its
    large backbone arrays.
    """

    expected_keys = _feature_keys(backbone, feature_key)
    missing = [key for key in expected_keys if not _payload_has_key(payload, key)]
    if missing:
        raise KeyError(f"{path} is missing feature key(s) for {backbone}: {missing}")
    if not _payload_has_key(payload, "image_feature_shapes_json"):
        raise ValueError(
            f"{path} cannot be lazily loaded because it does not contain "
            "image_feature_shapes_json. Re-run feature precomputation with the "
            "current writer, or set lazy_load_image_features=false."
        )
    raw_text = _npz_scalar_string(payload, "image_feature_shapes_json") or "{}"
    try:
        raw_shapes = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} has invalid image_feature_shapes_json.") from exc
    if not isinstance(raw_shapes, dict):
        raise ValueError(
            f"{path} image_feature_shapes_json must encode an object."
        )
    if set(raw_shapes) != set(expected_keys):
        raise ValueError(
            f"{path} cached feature shape keys do not match the requested "
            f"backbone: expected={sorted(expected_keys)}, "
            f"metadata={sorted(raw_shapes)}."
        )
    shapes: dict[str, tuple[int, ...]] = {}
    for key in expected_keys:
        raw_shape = raw_shapes[key]
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ValueError(
                f"{path} feature shape metadata for {key!r} must be a "
                f"non-empty integer list, got {raw_shape!r}."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
            for value in raw_shape
        ):
            raise ValueError(
                f"{path} feature shape metadata for {key!r} must contain "
                f"positive integers, got {raw_shape!r}."
            )
        shapes[key] = tuple(int(value) for value in raw_shape)
    return shapes

def _npz_scalar_string(payload: Any, key: str) -> str | None:
    if not _payload_has_key(payload, key):
        return None
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"Metadata key {key!r} must be scalar, got shape {value.shape}.")
    return str(value.reshape(()).item())

def _validate_cached_features(
    *,
    path: Path,
    payload: Any,
    backbone: str,
    feature_key: str,
    image_key: str,
    features: dict[str, np.ndarray] | np.ndarray | None,
    feature_shapes: Mapping[str, Sequence[int]] | None = None,
    require_metadata: bool,
    check_finite: bool,
    expected_vggt_context_mode: str | None,
) -> dict[str, Any] | None:
    stored_backbone = _npz_scalar_string(payload, "image_feature_backbone")
    stored_source_key = _npz_scalar_string(payload, "image_feature_source_key")
    stored_feature_key = _npz_scalar_string(payload, "image_feature_feature_key")
    metadata_text = _npz_scalar_string(payload, "image_feature_metadata_json")
    if require_metadata:
        missing = [
            key
            for key, value in (
                ("image_feature_backbone", stored_backbone),
                ("image_feature_source_key", stored_source_key),
                ("image_feature_feature_key", stored_feature_key),
                ("image_feature_metadata_json", metadata_text),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{path} is missing required cached-feature metadata {missing}. "
                "Re-run precompute_multiview_image_features.py or set feature_metadata_required=false "
                "only for a deliberate legacy-cache migration."
            )
    if stored_backbone is not None and normalize_feature_backbone(stored_backbone) != backbone:
        raise ValueError(
            f"{path} stores image_feature_backbone={stored_backbone!r}, but training requested {backbone!r}."
        )
    if stored_source_key is not None and stored_source_key != image_key:
        raise ValueError(
            f"{path} features were computed from image key {stored_source_key!r}, but image_key={image_key!r}."
        )
    if stored_feature_key is not None and stored_feature_key != feature_key:
        raise ValueError(
            f"{path} stores feature_key={stored_feature_key!r}, but feature_key={feature_key!r}."
        )

    if features is None:
        if feature_shapes is None:
            raise ValueError(
                f"{path} cached-feature validation requires arrays or shape metadata."
            )
        stored_shapes = {
            str(key): tuple(int(value) for value in shape)
            for key, shape in feature_shapes.items()
        }
        stored_arrays: dict[str, np.ndarray] | None = None
    else:
        feature_arrays = (
            features if isinstance(features, dict) else {feature_key: features}
        )
        stored_arrays = (
            {f"{feature_key}_{key}": value for key, value in feature_arrays.items()}
            if isinstance(features, dict)
            else feature_arrays
        )
        stored_shapes = {
            key: tuple(np.asarray(value).shape)
            for key, value in stored_arrays.items()
        }
    expected_ndim = 4 if backbone == "resnet_pre_fpn" else 3
    view_counts: set[int] = set()
    for key, shape in stored_shapes.items():
        if len(shape) != expected_ndim:
            expected = "[V,C,H,W]" if expected_ndim == 4 else "[V,L,D]"
            raise ValueError(
                f"{path} feature {key!r} must have shape {expected}, got {shape}."
            )
        if shape[0] < 1:
            raise ValueError(f"{path} feature {key!r} contains no views.")
        view_counts.add(int(shape[0]))
        if stored_arrays is not None and check_finite:
            array = np.asarray(stored_arrays[key])
            if not np.isfinite(array).all():
                bad_count = int(
                    array.size - np.count_nonzero(np.isfinite(array))
                )
                raise ValueError(
                    f"{path} feature {key!r} contains {bad_count} "
                    "NaN/Inf value(s)."
                )
    if len(view_counts) != 1:
        raise ValueError(f"{path} cached feature levels have inconsistent view counts: {sorted(view_counts)}.")
    feature_view_count = next(iter(view_counts))
    if _payload_has_key(payload, "image_feature_selected_view_indices"):
        selected = np.asarray(payload["image_feature_selected_view_indices"], dtype=np.int64).reshape(-1)
        if selected.size not in (0, feature_view_count):
            raise ValueError(
                f"{path} image_feature_selected_view_indices has length {selected.size}, "
                f"but cached features contain {feature_view_count} views."
            )
        if selected.size and (np.any(selected < 0) or np.unique(selected).size != selected.size):
            raise ValueError(f"{path} image_feature_selected_view_indices must be unique and non-negative.")

    if _payload_has_key(payload, "image_feature_shapes_json"):
        metadata_shapes = json.loads(
            _npz_scalar_string(payload, "image_feature_shapes_json") or "{}"
        )
        actual_shapes = {key: list(shape) for key, shape in stored_shapes.items()}
        if metadata_shapes != actual_shapes:
            raise ValueError(
                f"{path} cached feature shapes do not match metadata: actual={actual_shapes}, "
                f"metadata={metadata_shapes}."
            )
    if _payload_has_key(payload, "image_feature_keys_json"):
        stored_keys = json.loads(_npz_scalar_string(payload, "image_feature_keys_json") or "[]")
        if not isinstance(stored_keys, list) or set(stored_keys) != set(stored_shapes):
            raise ValueError(
                f"{path} cached feature keys do not match metadata: actual={sorted(stored_shapes)}, "
                f"metadata={stored_keys}."
            )

    metadata = None
    if metadata_text is not None:
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} has invalid image_feature_metadata_json.") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} image_feature_metadata_json must encode an object.")
        metadata_backbone = metadata.get("backbone")
        if metadata_backbone is not None and normalize_feature_backbone(metadata_backbone) != backbone:
            raise ValueError(
                f"{path} metadata JSON backbone={metadata_backbone!r}, but requested backbone={backbone!r}."
            )
        if backbone in ("vggt", "vggt_omega") and expected_vggt_context_mode is not None:
            actual_context = str(metadata.get("vggt_context_mode", "")).strip().lower()
            expected_context = str(expected_vggt_context_mode).strip().lower()
            if actual_context != expected_context:
                raise ValueError(
                    f"{path} VGGT cache uses context_mode={actual_context!r}, expected {expected_context!r}."
                )
    return metadata

def _resolve_local_view_indices(
    path: Path,
    current_num_views: int,
    requested_view_indices: Sequence[int] | None,
    stored_selected_views: np.ndarray | None,
) -> np.ndarray | None:
    if requested_view_indices is None:
        return None
    requested = np.asarray(parse_view_indices(requested_view_indices), dtype=np.int64)
    if int(requested.max()) < int(current_num_views):
        return requested

    if stored_selected_views is not None:
        stored = np.asarray(stored_selected_views, dtype=np.int64).reshape(-1)
        if stored.shape[0] == int(current_num_views):
            lookup = {int(original_index): local_index for local_index, original_index in enumerate(stored.tolist())}
            if all(int(index) in lookup for index in requested.tolist()):
                return np.asarray([lookup[int(index)] for index in requested.tolist()], dtype=np.int64)

    if int(current_num_views) == int(requested.shape[0]):
        return np.arange(int(current_num_views), dtype=np.int64)

    raise ValueError(
        f"{path} has {current_num_views} cached views, but requested clinical_rca_view_indices="
        f"{requested.tolist()} could not be mapped. If the file is already view-subsetted, it should contain "
        "image_feature_selected_view_indices."
    )

def _resolve_stored_original_view_indices(
    path: Path,
    *,
    current_num_views: int,
    source_selected_views: np.ndarray | None,
    feature_selected_views: np.ndarray | None,
) -> np.ndarray | None:
    """Resolve one original-view ID for every currently stored view row.

    ``selected_view_indices`` can already contain the original IDs inherited
    from an upstream Stage-4 file. ``image_feature_selected_view_indices``
    describes which rows the feature precomputation selected from that source.
    When feature selection is present, compose it with the source mapping even
    if both arrays happen to have the same length: the official precomputation
    writer preserves ``selected_view_indices`` in source-row order while it
    subsets/reorders every view-aligned array.
    """

    def validated(
        value: np.ndarray | None,
        *,
        key: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.int64).reshape(-1)
        if array.size == 0:
            return None
        if np.any(array < 0) or np.unique(array).size != array.size:
            raise ValueError(
                f"{path} {key} must contain unique non-negative view IDs."
            )
        return array

    source = validated(source_selected_views, key="selected_view_indices")
    feature = validated(
        feature_selected_views,
        key="image_feature_selected_view_indices",
    )
    count = int(current_num_views)
    if source is not None:
        if feature is not None and feature.shape == (count,):
            if int(feature.max()) >= int(source.size):
                raise ValueError(
                    f"{path} image_feature_selected_view_indices contains row "
                    f"{int(feature.max())}, but selected_view_indices has only "
                    f"{source.size} source rows."
                )
            return source[feature]
        if source.shape == (count,):
            return source
        raise ValueError(
            f"{path} selected_view_indices has length {source.size}, but the "
            f"stored arrays contain {count} views and no valid feature-view "
            "mapping can compose them."
        )
    if feature is not None:
        if feature.shape != (count,):
            raise ValueError(
                f"{path} image_feature_selected_view_indices has length "
                f"{feature.size}, but stored arrays contain {count} views."
            )
        return feature
    return None

def _subset_optional_view_array(
    value: np.ndarray | None,
    local_indices: np.ndarray | None,
    requested_view_indices: Sequence[int] | None,
    current_num_views: int,
) -> np.ndarray | None:
    if value is None or local_indices is None:
        return value
    arr = np.asarray(value)
    if arr.shape[0] == int(current_num_views):
        return arr[local_indices]
    if requested_view_indices is not None:
        requested = np.asarray(requested_view_indices, dtype=np.int64)
        if arr.shape[0] > int(requested.max()):
            return arr[requested]
    return arr

def load_precomputed_case(
    path: str | Path,
    feature_backbone: str,
    feature_key: str = "image_features",
    image_key: str = "images",
    view_feature_key: str = "view_features",
    num_branches: int = 1,
    num_points: int = 200,
    input_scale_to_mm: float = 1.0,
    view_indices: Sequence[int] | None = None,
    centerline_dt_dir: str | Path | None = None,
    centerline_dt_resolution_mm: Any = None,
    centerline_dt_required: bool = False,
    dataset_root: str | Path | None = None,
    load_images: bool = True,
    feature_metadata_required: bool = True,
    check_feature_finite: bool = True,
    expected_vggt_context_mode: str | None = None,
    target_coordinate_frame: str = "absolute_world",
    zero_cached_image_features: bool = False,
    require_targets: bool = True,
    projection_mask_distance_required: bool = False,
    projection_mask_threshold: float = 0.5,
    load_image_features: bool = True,
    lazy_load_image_features: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    backbone = normalize_feature_backbone(feature_backbone)
    coordinate_frame = resolve_target_coordinate_frame(
        {"target_coordinate_frame": target_coordinate_frame},
        default="absolute_world",
    )
    zero_features = resolve_zero_cached_image_features(
        {"zero_cached_image_features": zero_cached_image_features}
    )
    lazy_features = resolve_lazy_load_image_features(
        {"lazy_load_image_features": lazy_load_image_features}
    )
    if lazy_features and not load_image_features:
        raise ValueError(
            "lazy_load_image_features=true requires precomputed image "
            "features; it cannot be combined with load_image_features=false."
        )
    if not np.isfinite(float(input_scale_to_mm)) or float(input_scale_to_mm) <= 0.0:
        raise ValueError(f"input_scale_to_mm must be finite and positive, got {input_scale_to_mm}.")
    if not np.isfinite(float(projection_mask_threshold)) or not 0.0 <= float(
        projection_mask_threshold
    ) <= 1.0:
        raise ValueError(
            "projection_mask_threshold must be finite and in [0, 1], got "
            f"{projection_mask_threshold}."
        )
    embedded_centerline_dt = None
    with np.load(path, allow_pickle=False) as data:
        required_input_keys = [image_key]
        if require_targets:
            required_input_keys.append("artery")
        for key in required_input_keys:
            if not _payload_has_key(data, key):
                raise KeyError(f"{path} does not contain required key {key!r}.")
        if load_image_features:
            missing = [
                key
                for key in _feature_keys(backbone, feature_key)
                if not _payload_has_key(data, key)
            ]
            if missing:
                raise KeyError(
                    f"{path} is missing feature key(s) for {backbone}: {missing}"
                )

        feature_shapes: dict[str, tuple[int, ...]] | None = None
        if load_image_features and lazy_features:
            feature_shapes = _stored_feature_shapes(
                path=path,
                payload=data,
                backbone=backbone,
                feature_key=feature_key,
            )
            features = None
            feature_num_views = int(next(iter(feature_shapes.values()))[0])
        elif load_image_features and backbone == "resnet_pre_fpn":
            features: dict[str, np.ndarray] | np.ndarray | None = {
                suffix: np.asarray(data[f"{feature_key}_{suffix}"], dtype=np.float32)
                for suffix in RESNET_PRE_FPN_SUFFIXES
            }
            feature_num_views = int(
                features[RESNET_PRE_FPN_SUFFIXES[0]].shape[0]
            )
        elif load_image_features:
            features = np.asarray(data[feature_key], dtype=np.float32)
            feature_num_views = int(features.shape[0])
        else:
            features = None
            feature_num_views = int(np.asarray(data[image_key]).shape[0])
        feature_metadata = (
            _validate_cached_features(
                path=path,
                payload=data,
                backbone=backbone,
                feature_key=feature_key,
                image_key=image_key,
                features=features,
                feature_shapes=feature_shapes,
                require_metadata=bool(feature_metadata_required),
                check_finite=bool(check_feature_finite),
                expected_vggt_context_mode=expected_vggt_context_mode,
            )
            if load_image_features
            else None
        )

        images = np.asarray(data[image_key], dtype=np.float32) if load_images else None
        if _payload_has_key(data, view_feature_key):
            view_features = np.asarray(data[view_feature_key], dtype=np.float32)
        else:
            view_features = _view_features_from_angles(data)
            if view_features is None:
                raise KeyError(
                    f"{path} does not contain {view_feature_key!r} or angle arrays "
                    "that can be converted to view features."
                )
        if feature_num_views != view_features.shape[0]:
            raise ValueError(
                f"{path} has mismatched view counts: cached features={feature_num_views}, "
                f"{view_feature_key}={view_features.shape[0]}"
            )
        if images is not None and images.shape[0] != feature_num_views:
            raise ValueError(
                f"{path} has mismatched view counts: {image_key}={images.shape[0]}, "
                f"cached features={feature_num_views}."
            )

        theta = _optional_array(data, ("theta_deg", "theta_array", "theta"))
        phi = _optional_array(data, ("phi_deg", "phi_array", "phi"))
        if _payload_has_key(data, "projection_center_offset"):
            projection_center_offset = np.asarray(data["projection_center_offset"], dtype=np.float32).reshape(3)
            projection_center_offset = projection_center_offset * float(input_scale_to_mm)
            if not np.isfinite(projection_center_offset).all():
                raise ValueError(
                    f"{path} projection_center_offset contains NaN/Inf values: "
                    f"{projection_center_offset.tolist()}. Rebuild or exclude this case."
                )
            projection_center_offset_valid = True
        else:
            projection_center_offset = np.zeros((3,), dtype=np.float32)
            projection_center_offset_valid = False
        if (
            require_targets
            and coordinate_frame == "projection_centered"
            and not projection_center_offset_valid
        ):
            raise ValueError(
                f"{path} does not contain projection_center_offset, but "
                "target_coordinate_frame='projection_centered' requires the exact offset used to render "
                "the input projections. Rebuild the stage-2 case or use target_coordinate_frame='absolute_world'."
            )

        if not np.isfinite(view_features).all():
            nonfinite_count = int(np.size(view_features) - np.isfinite(view_features).sum())
            raise ValueError(
                f"{path} {view_feature_key} contains {nonfinite_count} NaN/Inf value(s). "
                "Rebuild or exclude this case."
            )
        target_points = None
        target_exist = None
        artery = None
        if require_targets:
            artery_absolute = _scale_artery_to_mm(
                np.asarray(data["artery"], dtype=np.float32),
                input_scale_to_mm,
            )
            if not np.isfinite(artery_absolute).all():
                nonfinite_count = int(
                    np.size(artery_absolute)
                    - np.isfinite(artery_absolute).sum()
                )
                raise ValueError(
                    f"{path} artery contains {nonfinite_count} NaN/Inf value(s). "
                    "Rebuild or exclude this case."
                )
            exist_flags = _pick_existing_flag_array(
                data,
                num_branches=num_branches,
            )
            target_points, target_exist = build_fixed_targets(
                artery=artery_absolute,
                num_branches=int(num_branches),
                num_points=int(num_points),
                exist_flags=exist_flags,
            )
            artery = artery_absolute.copy()
            if coordinate_frame == "projection_centered":
                offset = projection_center_offset.reshape(1, 1, 3)
                target_points[target_exist > 0.5, :, :3] -= offset
                valid_artery_branches = np.any(
                    artery[..., 3] > 1e-6,
                    axis=1,
                )
                artery[valid_artery_branches, :, :3] -= offset
            active_target_points = target_points[target_exist > 0.5]
            if not np.isfinite(active_target_points).all():
                nonfinite_count = int(
                    np.size(active_target_points)
                    - np.isfinite(active_target_points).sum()
                )
                raise ValueError(
                    f"{path} produced {nonfinite_count} NaN/Inf target value(s) after scaling and "
                    f"{coordinate_frame!r} coordinate conversion. Rebuild or exclude this case."
                )

        selected_views = _resolve_stored_original_view_indices(
            path,
            current_num_views=feature_num_views,
            source_selected_views=_optional_array(
                data, ("selected_view_indices",)
            ),
            feature_selected_views=_optional_array(
                data, ("image_feature_selected_view_indices",)
            ),
        )
        metadata_json = _npz_scalar_string(data, "image_feature_metadata_json")
        embedded_centerline_dt = _embedded_centerline_dt_payload(data)

    centerline_dt_payload = embedded_centerline_dt if require_targets else None
    centerline_dt_path = (
        _resolve_centerline_dt_path(
            path,
            centerline_dt_dir,
            dataset_root,
            centerline_dt_resolution_mm=centerline_dt_resolution_mm,
        )
        if require_targets
        else None
    )
    if centerline_dt_path is not None:
        centerline_dt_payload = _load_centerline_dt_payload(centerline_dt_path)
    elif bool(centerline_dt_required) and centerline_dt_payload is None:
        raise FileNotFoundError(
            f"No 3D centerline DT found for {path}. Expected embedded centerline_dt_* arrays or a sidecar under "
            f"{centerline_dt_dir!r}."
        )
    requested_dt_spacing = parse_centerline_dt_resolution(centerline_dt_resolution_mm)
    if centerline_dt_payload is not None and requested_dt_spacing is not None:
        actual_dt_spacing = np.asarray(centerline_dt_payload["centerline_dt_spacing"], dtype=np.float32).reshape(3)
        if not np.allclose(actual_dt_spacing, requested_dt_spacing, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"{path} resolved a centerline DT with spacing {actual_dt_spacing.tolist()} mm, "
                f"but {requested_dt_spacing.tolist()} mm was requested. Check the DT root/resolution subdirectory."
            )

    num_views_before_selection = int(feature_num_views)
    local_view_indices = _resolve_local_view_indices(
        path=path,
        current_num_views=num_views_before_selection,
        requested_view_indices=view_indices,
        stored_selected_views=selected_views,
    )
    if local_view_indices is not None:
        if images is not None:
            images = images[local_view_indices]
        view_features = view_features[local_view_indices]
        if isinstance(features, dict):
            features = {key: value[local_view_indices] for key, value in features.items()}
        elif features is not None:
            features = features[local_view_indices]
        theta = _subset_optional_view_array(
            theta,
            local_view_indices,
            requested_view_indices=view_indices,
            current_num_views=num_views_before_selection,
        )
        phi = _subset_optional_view_array(
            phi,
            local_view_indices,
            requested_view_indices=view_indices,
            current_num_views=num_views_before_selection,
        )
        if selected_views is not None and np.asarray(selected_views).reshape(-1).shape[0] == num_views_before_selection:
            selected_views = np.asarray(selected_views, dtype=np.int64).reshape(-1)[local_view_indices]
        elif view_indices is not None:
            selected_views = np.asarray(view_indices, dtype=np.int64)

    if zero_features:
        if features is None and not lazy_features:
            raise ValueError(
                "zero_cached_image_features cannot be used when "
                "load_image_features=False."
            )
        if isinstance(features, dict):
            features = {key: np.zeros_like(value) for key, value in features.items()}
        elif features is not None:
            features = np.zeros_like(features)

    projection_mask_distance_px = None
    if bool(projection_mask_distance_required):
        if images is not None:
            from scipy.ndimage import distance_transform_edt

            masks = images[:, 0] if images.ndim == 4 else images
            projection_mask_distance_px = np.stack(
                [
                    distance_transform_edt(
                        mask <= float(projection_mask_threshold)
                    ).astype(np.float32)
                    for mask in masks
                ],
                axis=0,
            )

    item_feature_shapes = feature_shapes
    if feature_shapes is not None and backbone == "resnet_pre_fpn":
        item_feature_shapes = {
            suffix: tuple(feature_shapes[f"{feature_key}_{suffix}"])
            for suffix in RESNET_PRE_FPN_SUFFIXES
        }

    out = {
        "case_name": path.stem,
        "path": str(path),
        "images": images,
        "image_key": image_key,
        "source_num_views": num_views_before_selection,
        "source_local_view_indices": (
            np.arange(num_views_before_selection, dtype=np.int64)
            if local_view_indices is None
            else np.asarray(local_view_indices, dtype=np.int64)
        ),
        "view_features": view_features.astype(np.float32),
        "view_mask": np.ones((view_features.shape[0],), dtype=np.float32),
        "image_features": features,
        "image_feature_shapes": item_feature_shapes,
        "feature_backbone": backbone,
        "feature_key": feature_key,
        "lazy_load_image_features": lazy_features,
        "check_feature_finite": bool(check_feature_finite),
        "zero_cached_image_features": zero_features,
        "projection_mask_distance_required": bool(
            projection_mask_distance_required
        ),
        "projection_mask_threshold": float(projection_mask_threshold),
        "target_coordinate_frame": coordinate_frame,
        "has_ground_truth": bool(require_targets),
        "theta": None if theta is None else np.asarray(theta, dtype=np.float32),
        "phi": None if phi is None else np.asarray(phi, dtype=np.float32),
        "projection_center_offset": projection_center_offset.astype(np.float32),
        "projection_center_offset_valid": np.asarray(projection_center_offset_valid, dtype=np.bool_),
        "selected_view_indices": (
            None if selected_views is None else np.asarray(selected_views, dtype=np.int64)
        ),
        "image_feature_metadata_json": metadata_json,
        "image_feature_metadata": feature_metadata,
        "centerline_dt_path": None if centerline_dt_path is None else str(centerline_dt_path),
    }
    if require_targets:
        assert target_points is not None
        assert target_exist is not None
        assert artery is not None
        out.update(
            {
                "target_points": target_points.astype(np.float32),
                "target_exist": target_exist.astype(np.float32),
                "artery": artery.astype(np.float32),
            }
        )
    if projection_mask_distance_px is not None:
        out["projection_mask_distance_px"] = projection_mask_distance_px
    if centerline_dt_payload is not None:
        out.update(centerline_dt_payload)
    return out

def ensure_item_images(item: dict[str, Any]) -> np.ndarray:
    """Load and cache a case's image array only when an image-dependent path needs it."""
    if item.get("images") is not None:
        return np.asarray(item["images"], dtype=np.float32)
    path = Path(item["path"])
    image_key = str(item.get("image_key", "images"))
    with np.load(path, allow_pickle=False) as data:
        if not _payload_has_key(data, image_key):
            raise KeyError(f"{path} does not contain required image key {image_key!r}.")
        source_images = np.asarray(data[image_key], dtype=np.float32)
    expected_source_views = int(item.get("source_num_views", source_images.shape[0]))
    if source_images.shape[0] != expected_source_views:
        raise ValueError(
            f"{path} image view count changed after feature loading: "
            f"found {source_images.shape[0]}, expected {expected_source_views}."
        )
    local_indices = np.asarray(
        item.get("source_local_view_indices", np.arange(expected_source_views)),
        dtype=np.int64,
    )
    images = source_images[local_indices]
    expected_views = int(np.asarray(item["view_mask"]).shape[0])
    if images.shape[0] != expected_views:
        raise ValueError(
            f"{path} lazy-loaded {images.shape[0]} images, but cached features contain {expected_views} views."
        )
    item["images"] = images
    return images

def _feature_metadata_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that can affect the selected cached backbone features."""
    raw_backbone = str(metadata.get("backbone", "")).strip().lower().replace("-", "_")
    backbone = {
        "resnet": "resnet_pre_fpn",
        "resnet101": "resnet_pre_fpn",
        "vggt_original": "vggt",
        "omega": "vggt_omega",
    }.get(raw_backbone, raw_backbone)
    signature: dict[str, Any] = {
        "backbone": backbone,
        "image_key": str(metadata.get("image_key") or "images"),
        "feature_key": str(metadata.get("feature_key") or "image_features"),
        "output_dtype": str(metadata.get("output_dtype") or "float32"),
        "view_selection_mode": str(metadata.get("view_selection_mode") or "all"),
        "selected_view_indices": metadata.get("selected_view_indices"),
    }
    if backbone == "resnet_pre_fpn":
        signature.update(
            {
                "seed": int(metadata.get("seed") or 0),
                "image_channels": int(metadata.get("image_channels") or 1),
                # ResNet extraction treats a missing/None setting as False.
                "predictor_pretrained_backbone": bool(
                    metadata.get("predictor_pretrained_backbone") or False
                ),
            }
        )
        return signature

    if backbone in ("vggt", "vggt_omega"):
        expected_variant = "omega" if backbone == "vggt_omega" else "original"
        raw_variant = str(metadata.get("vggt_backbone") or expected_variant).strip().lower()
        variant = "omega" if raw_variant in ("omega", "vggt_omega") else "original"
        signature.update(
            {
                "seed": int(metadata.get("seed") or 0),
                "vggt_context_mode": str(metadata.get("vggt_context_mode") or "all_views"),
                "vggt_backbone": variant,
                "vggt_pretrained": True
                if metadata.get("vggt_pretrained") is None
                else bool(metadata.get("vggt_pretrained")),
                "vggt_load_mode": str(metadata.get("vggt_load_mode") or "package"),
                "vggt_model_name": str(metadata.get("vggt_model_name") or "facebook/VGGT-1B"),
                "vggt_omega_checkpoint_path": (
                    metadata.get("vggt_omega_checkpoint_path") if variant == "omega" else None
                ),
                "vggt_token_dim": int(metadata.get("vggt_token_dim") or 2048),
                "vggt_image_size_mode": str(
                    metadata.get("vggt_image_size_mode") or "resize_to_patch_multiple"
                ),
                "vggt_patch_size": int(
                    metadata.get("vggt_patch_size") or (16 if variant == "omega" else 14)
                ),
                "vggt_target_image_size": int(
                    metadata.get("vggt_target_image_size") or (256 if variant == "omega" else 266)
                ),
            }
        )
    return signature

def load_precomputed_items(
    files: list[Path],
    feature_backbone: str,
    feature_key: str,
    image_key: str,
    view_feature_key: str,
    num_branches: int,
    num_points: int,
    input_scale_to_mm: float,
    view_indices: Sequence[int] | None = None,
    centerline_dt_dir: str | Path | None = None,
    centerline_dt_resolution_mm: Any = None,
    centerline_dt_required: bool = False,
    dataset_root: str | Path | None = None,
    load_images: bool = True,
    feature_metadata_required: bool = True,
    check_feature_finite: bool = True,
    expected_vggt_context_mode: str | None = None,
    desc: str = "Loading precomputed cases",
    target_coordinate_frame: str = "absolute_world",
    zero_cached_image_features: bool = False,
    require_targets: bool = True,
    projection_mask_distance_required: bool = False,
    projection_mask_threshold: float = 0.5,
    load_image_features: bool = True,
    lazy_load_image_features: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    items = [
        load_precomputed_case(
            path=file_path,
            feature_backbone=feature_backbone,
            feature_key=feature_key,
            image_key=image_key,
            view_feature_key=view_feature_key,
            num_branches=num_branches,
            num_points=num_points,
            input_scale_to_mm=input_scale_to_mm,
            target_coordinate_frame=target_coordinate_frame,
            view_indices=view_indices,
            centerline_dt_dir=centerline_dt_dir,
            centerline_dt_resolution_mm=centerline_dt_resolution_mm,
            centerline_dt_required=centerline_dt_required,
            dataset_root=dataset_root,
            load_images=load_images,
            feature_metadata_required=feature_metadata_required,
            check_feature_finite=check_feature_finite,
            expected_vggt_context_mode=expected_vggt_context_mode,
            zero_cached_image_features=zero_cached_image_features,
            require_targets=require_targets,
            projection_mask_distance_required=projection_mask_distance_required,
            projection_mask_threshold=projection_mask_threshold,
            load_image_features=load_image_features,
            lazy_load_image_features=lazy_load_image_features,
        )
        for file_path in tqdm(
            files,
            desc=desc,
            leave=False,
            disable=not bool(show_progress),
        )
    ]
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    reference_signature: dict[str, Any] | None = None
    for item in items:
        features = item["image_features"]
        stored_shapes = item.get("image_feature_shapes")
        if features is None and stored_shapes is None:
            continue
        if features is not None:
            shapes = (
                {
                    key: tuple(np.asarray(value).shape[1:])
                    for key, value in features.items()
                }
                if isinstance(features, dict)
                else {feature_key: tuple(np.asarray(features).shape[1:])}
            )
        else:
            assert isinstance(stored_shapes, Mapping)
            shapes = {
                str(key): tuple(int(value) for value in shape[1:])
                for key, shape in stored_shapes.items()
            }
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError(
                f"Cached feature shapes differ across cases. First case={reference_shapes}; "
                f"{item['path']}={shapes}. All cases in one run must use the same backbone/preprocessing."
            )
        metadata = item.get("image_feature_metadata")
        if isinstance(metadata, dict):
            signature = _feature_metadata_signature(metadata)
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise ValueError(
                    f"Cached feature metadata differs across cases. First case={reference_signature}; "
                    f"{item['path']}={signature}. Do not mix caches produced by different backbone settings."
                )
    return items

def normalize_view_count_weights(
    value: Mapping[int | str, float] | None,
    *,
    min_views: int | None = None,
    max_views: int | None = None,
    setting_name: str = "view_count_weights",
) -> dict[int, float] | None:
    """Validate and normalize a JSON view-count-to-weight mapping."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"{setting_name} must be a non-empty object mapping view counts "
            "to non-negative weights."
        )
    parsed: dict[int, float] = {}
    for raw_count, raw_weight in value.items():
        if isinstance(raw_count, bool):
            raise ValueError(f"{setting_name} keys must be positive integers.")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{setting_name} keys must be positive integers, got "
                f"{raw_count!r}."
            ) from exc
        if str(raw_count).strip() != str(count):
            raise ValueError(
                f"{setting_name} keys must be canonical positive integers, "
                f"got {raw_count!r}."
            )
        if count < 1:
            raise ValueError(
                f"{setting_name} keys must be >= 1, got {count}."
            )
        if count in parsed:
            raise ValueError(f"{setting_name} contains duplicate count {count}.")
        if isinstance(raw_weight, bool):
            raise ValueError(
                f"{setting_name}[{count}] must be a finite number >= 0."
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{setting_name}[{count}] must be a finite number >= 0."
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{setting_name}[{count}] must be a finite number >= 0, "
                f"got {raw_weight!r}."
            )
        if min_views is not None and count < int(min_views):
            raise ValueError(
                f"{setting_name} count {count} is below min_views "
                f"({int(min_views)})."
            )
        if max_views is not None and count > int(max_views):
            raise ValueError(
                f"{setting_name} count {count} exceeds max_views "
                f"({int(max_views)})."
            )
        parsed[count] = weight
    total = math.fsum(parsed.values())
    if total <= 0.0:
        raise ValueError(
            f"{setting_name} must contain at least one positive weight."
        )
    return {count: weight / total for count, weight in sorted(parsed.items())}

class PrecomputedFeatureDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        min_views: int | None = None,
        max_views: int | None = None,
        random_view_order: bool = False,
        random_view_seed: int | None = None,
        view_count_weights: Mapping[int | str, float] | None = None,
    ) -> None:
        self.items = list(items)
        if not self.items:
            raise ValueError("PrecomputedFeatureDataset requires at least one item.")
        self.min_views = None if min_views is None else int(min_views)
        self.max_views = None if max_views is None else int(max_views)
        self.random_view_order = bool(random_view_order)
        self.random_view_seed = (
            None if random_view_seed is None else int(random_view_seed)
        )
        if self.min_views is not None and self.min_views < 1:
            raise ValueError(f"min_views must be >= 1, got {self.min_views}")
        if self.max_views is not None and self.max_views < 1:
            raise ValueError(f"max_views must be >= 1, got {self.max_views}")
        if (
            self.min_views is not None
            and self.max_views is not None
            and self.min_views > self.max_views
        ):
            raise ValueError(
                f"min_views ({self.min_views}) cannot exceed max_views "
                f"({self.max_views})."
            )
        self.view_count_weights = normalize_view_count_weights(
            view_count_weights,
            min_views=self.min_views,
            max_views=self.max_views,
        )

    def __len__(self) -> int:
        return len(self.items)

    def _selected_view_indices(self, item: dict[str, Any]) -> np.ndarray:
        total_views = int(np.asarray(item["view_mask"]).shape[0])
        if total_views < 1:
            raise ValueError(f"Case {item.get('case_name')} has no views.")
        max_views = total_views if self.max_views is None else min(int(self.max_views), total_views)
        min_views = max_views if self.min_views is None else min(int(self.min_views), max_views)
        if min_views > max_views:
            min_views = max_views
        rng: Any = random
        if self.random_view_seed is not None:
            identity = str(
                item.get(
                    "view_selection_identity",
                    item.get("path", item.get("case_name", "")),
                )
            )
            digest = hashlib.sha256(
                f"{self.random_view_seed}:{identity}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], byteorder="big"))
        if self.view_count_weights is not None:
            eligible = [
                (count, weight)
                for count, weight in self.view_count_weights.items()
                if min_views <= count <= max_views and weight > 0.0
            ]
            if not eligible:
                positive_counts = [
                    count
                    for count, weight in self.view_count_weights.items()
                    if weight > 0.0
                ]
                raise ValueError(
                    f"Case {item.get('case_name')} has {total_views} views, but "
                    "none of the positively weighted view counts are feasible: "
                    f"{positive_counts}."
                )
            view_count = rng.choices(
                [count for count, _ in eligible],
                weights=[weight for _, weight in eligible],
                k=1,
            )[0]
        elif min_views < max_views:
            view_count = rng.randint(min_views, max_views)
        else:
            view_count = max_views
        if self.random_view_order:
            return np.asarray(
                rng.sample(range(total_views), k=int(view_count)),
                dtype=np.int64,
            )
        return np.arange(int(view_count), dtype=np.int64)

    @staticmethod
    def _subset_features(
        features: dict[str, np.ndarray] | np.ndarray,
        view_indices: np.ndarray,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(features, dict):
            return {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32)[view_indices])
                for key, value in features.items()
            }
        return torch.from_numpy(np.asarray(features, dtype=np.float32)[view_indices])

    @staticmethod
    def _load_lazy_features(
        item: Mapping[str, Any],
        view_indices: np.ndarray,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Load one cached feature member and retain only requested views."""

        path = Path(str(item["path"]))
        backbone = normalize_feature_backbone(
            str(item.get("feature_backbone", "vggt"))
        )
        feature_key = str(item.get("feature_key", "image_features"))
        raw_shapes = item.get("image_feature_shapes")
        if not isinstance(raw_shapes, Mapping) or not raw_shapes:
            raise ValueError(
                f"{path} lazy feature descriptor is missing "
                "image_feature_shapes. Reload the dataset from the source NPZ."
            )
        shapes = {
            str(key): tuple(int(value) for value in shape)
            for key, shape in raw_shapes.items()
        }
        total_local_views = int(np.asarray(item["view_mask"]).shape[0])
        source_local_indices = np.asarray(
            item.get(
                "source_local_view_indices",
                np.arange(total_local_views, dtype=np.int64),
            ),
            dtype=np.int64,
        ).reshape(-1)
        if source_local_indices.shape != (total_local_views,):
            raise ValueError(
                f"{path} lazy source view mapping has shape "
                f"{source_local_indices.shape}, expected ({total_local_views},)."
            )
        source_indices = source_local_indices[view_indices]
        expected_source_views = int(item.get("source_num_views", -1))
        if expected_source_views < 1:
            raise ValueError(
                f"{path} lazy feature descriptor has invalid source_num_views="
                f"{expected_source_views}."
            )
        if np.any(source_indices < 0) or np.any(
            source_indices >= expected_source_views
        ):
            raise ValueError(
                f"{path} lazy feature source indices {source_indices.tolist()} "
                f"fall outside [0, {expected_source_views})."
            )

        zero_features = bool(item.get("zero_cached_image_features", False))
        check_finite = bool(item.get("check_feature_finite", True))

        def materialise(
            payload: Any,
            *,
            stored_key: str,
            output_key: str,
        ) -> torch.Tensor:
            expected_shape = shapes[output_key]
            if len(expected_shape) < 1 or expected_shape[0] != expected_source_views:
                raise ValueError(
                    f"{path} lazy feature descriptor for {stored_key!r} has "
                    f"shape {expected_shape}, inconsistent with "
                    f"source_num_views={expected_source_views}."
                )
            if not _payload_has_key(payload, stored_key):
                raise KeyError(
                    f"{path} no longer contains lazy feature key {stored_key!r}. "
                    "The cache changed after dataset discovery."
                )
            source = np.asarray(payload[stored_key])
            if tuple(source.shape) != expected_shape:
                raise ValueError(
                    f"{path} lazy feature {stored_key!r} changed shape after "
                    f"dataset discovery: found {tuple(source.shape)}, "
                    f"expected {expected_shape}."
                )
            if check_finite and not np.isfinite(source).all():
                bad_count = int(
                    source.size - np.count_nonzero(np.isfinite(source))
                )
                raise ValueError(
                    f"{path} feature {stored_key!r} contains {bad_count} "
                    "NaN/Inf value(s) during lazy loading."
                )
            if zero_features:
                return torch.zeros(
                    (int(view_indices.size), *expected_shape[1:]),
                    dtype=torch.float32,
                )
            # Advanced indexing makes an owned selected-view copy before the
            # NPZ handle closes.  Cast only that copy, not the full feature.
            selected = np.asarray(source[source_indices], dtype=np.float32)
            return torch.from_numpy(selected)

        with np.load(path, allow_pickle=False) as payload:
            if backbone == "resnet_pre_fpn":
                expected_outputs = set(RESNET_PRE_FPN_SUFFIXES)
                if set(shapes) != expected_outputs:
                    raise ValueError(
                        f"{path} lazy ResNet feature descriptor has keys "
                        f"{sorted(shapes)}, expected {sorted(expected_outputs)}."
                    )
                return {
                    suffix: materialise(
                        payload,
                        stored_key=f"{feature_key}_{suffix}",
                        output_key=suffix,
                    )
                    for suffix in RESNET_PRE_FPN_SUFFIXES
                }
            if set(shapes) != {feature_key}:
                raise ValueError(
                    f"{path} lazy {backbone} feature descriptor has keys "
                    f"{sorted(shapes)}, expected {[feature_key]}."
                )
            return materialise(
                payload,
                stored_key=feature_key,
                output_key=feature_key,
            )

    def item_with_view_indices(
        self,
        index: int,
        view_indices: np.ndarray | Sequence[int],
    ) -> dict[str, Any]:
        """Materialise one item using an explicit local-view subset.

        The ordinary dataset path still obtains this subset from
        :meth:`_selected_view_indices`.  Group-aware callers can use this
        method when several related samples must share a jointly constrained
        subset (for example, Stage-4 branch-visibility sampling).
        """

        item = self.items[int(index)]
        view_indices = np.asarray(view_indices, dtype=np.int64).reshape(-1)
        total_views = int(np.asarray(item["view_mask"]).shape[0])
        if view_indices.size < 1:
            raise ValueError(
                f"Case {item.get('case_name')} explicit view selection is empty."
            )
        if (
            np.any(view_indices < 0)
            or np.any(view_indices >= total_views)
            or np.unique(view_indices).size != view_indices.size
        ):
            raise ValueError(
                f"Case {item.get('case_name')} explicit local view indices "
                f"must be unique values in [0, {total_views}), got "
                f"{view_indices.tolist()}."
            )
        features = item.get("image_features")
        if features is None and bool(item.get("lazy_load_image_features", False)):
            feature_tensors = self._load_lazy_features(item, view_indices)
        else:
            feature_tensors = (
                None
                if features is None
                else self._subset_features(features, view_indices)
            )
        item_images = item.get("images")
        if item_images is None and bool(item.get("lazy_load_images", False)):
            with np.load(Path(item["path"]), allow_pickle=False) as payload:
                image_key = str(item.get("image_key", "images"))
                if not _payload_has_key(payload, image_key):
                    raise KeyError(
                        f"{item['path']} does not contain required image key "
                        f"{image_key!r}."
                    )
                source_images = np.asarray(payload[image_key], dtype=np.float32)
            source_local_indices = np.asarray(
                item.get(
                    "source_local_view_indices",
                    np.arange(source_images.shape[0]),
                ),
                dtype=np.int64,
            )
            item_images = source_images[source_local_indices]
        theta = item.get("theta")
        phi = item.get("phi")
        original_view_indices = item.get("selected_view_indices")
        if original_view_indices is not None:
            original_view_indices = np.asarray(original_view_indices, dtype=np.int64).reshape(-1)
            if original_view_indices.shape[0] == int(np.asarray(item["view_mask"]).shape[0]):
                selected_view_indices = original_view_indices[view_indices]
            else:
                selected_view_indices = view_indices
        else:
            selected_view_indices = view_indices
        out = {
            "case_name": item["case_name"],
            "path": item["path"],
            "item_index": int(index),
            "local_view_indices": torch.from_numpy(view_indices.astype(np.int64, copy=False)),
            "selected_view_indices": torch.from_numpy(selected_view_indices.astype(np.int64, copy=False)),
            "images": (
                None
                if item_images is None
                else torch.from_numpy(
                    np.asarray(item_images, dtype=np.float32)[view_indices]
                )
            ),
            "view_features": torch.from_numpy(np.asarray(item["view_features"], dtype=np.float32)[view_indices]),
            "view_mask": torch.from_numpy(np.asarray(item["view_mask"], dtype=np.float32)[view_indices]),
            "image_features": feature_tensors,
            "theta": None if theta is None else torch.from_numpy(np.asarray(theta, dtype=np.float32)[view_indices]),
            "phi": None if phi is None else torch.from_numpy(np.asarray(phi, dtype=np.float32)[view_indices]),
            "projection_center_offset": torch.from_numpy(
                np.asarray(item["projection_center_offset"], dtype=np.float32)
            ),
            "projection_center_offset_valid": torch.tensor(
                bool(item["projection_center_offset_valid"]),
                dtype=torch.bool,
            ),
        }
        if "target_points" in item:
            out["target_points"] = torch.from_numpy(
                np.asarray(item["target_points"], dtype=np.float32)
            )
            out["target_exist"] = torch.from_numpy(
                np.asarray(item["target_exist"], dtype=np.float32)
            )
            point_valid = item.get("target_point_valid_mask")
            if point_valid is not None:
                out["target_point_valid_mask"] = torch.from_numpy(
                    np.asarray(point_valid, dtype=np.bool_)
                )
            out["artery"] = torch.from_numpy(
                np.asarray(item["artery"], dtype=np.float32)
            )
        if item.get("centerline_dt_volume") is not None:
            out["centerline_dt_volume"] = torch.from_numpy(
                np.asarray(item["centerline_dt_volume"], dtype=np.float32)
            )
            out["centerline_dt_origin"] = torch.from_numpy(
                np.asarray(item["centerline_dt_origin"], dtype=np.float32).reshape(3)
            )
            out["centerline_dt_spacing"] = torch.from_numpy(
                np.asarray(item["centerline_dt_spacing"], dtype=np.float32).reshape(3)
            )
            out["centerline_dt_max_distance_mm"] = torch.tensor(
                float(np.asarray(item.get("centerline_dt_max_distance_mm", np.nan)).reshape(())),
                dtype=torch.float32,
            )
            out["centerline_dt_valid"] = torch.tensor(
                bool(item.get("centerline_dt_valid", True)),
                dtype=torch.bool,
            )
        projection_mask_distance = item.get("projection_mask_distance_px")
        if (
            projection_mask_distance is None
            and bool(item.get("projection_mask_distance_required", False))
        ):
            if item_images is None:
                raise ValueError(
                    "Centreline-to-mask projection loss requires input images "
                    "to compute projection_mask_distance_px lazily."
                )
            from scipy.ndimage import distance_transform_edt

            selected_images = np.asarray(item_images, dtype=np.float32)[
                view_indices
            ]
            masks = (
                selected_images[:, 0]
                if selected_images.ndim == 4
                else selected_images
            )
            out["projection_mask_distance_px"] = torch.from_numpy(
                np.stack(
                    [
                        distance_transform_edt(
                            mask
                            <= float(item.get("projection_mask_threshold", 0.5))
                        ).astype(np.float32)
                        for mask in masks
                    ],
                    axis=0,
                )
            )
        elif projection_mask_distance is not None:
            out["projection_mask_distance_px"] = torch.from_numpy(
                np.asarray(
                    projection_mask_distance, dtype=np.float32
                )[view_indices]
            )
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[int(index)]
        return self.item_with_view_indices(
            int(index),
            self._selected_view_indices(item),
        )

def _pad_view_tensor(tensor: torch.Tensor, max_views: int) -> torch.Tensor:
    if tensor.shape[0] == int(max_views):
        return tensor
    out = tensor.new_zeros((int(max_views), *tensor.shape[1:]))
    out[: tensor.shape[0]] = tensor
    return out

def collate_precomputed_batches(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_views = max(int(item["view_mask"].shape[0]) for item in batch)
    feature_presence = [item.get("image_features") is not None for item in batch]
    if any(feature_presence) and not all(feature_presence):
        raise ValueError(
            "Cannot collate a mixed batch where only some cases contain "
            "precomputed image features."
        )
    first_features = batch[0].get("image_features")
    if first_features is None:
        feature_batch: dict[str, torch.Tensor] | torch.Tensor | None = None
    elif isinstance(first_features, dict):
        feature_batch: dict[str, torch.Tensor] | torch.Tensor = {}
        for key in first_features:
            feature_batch[key] = torch.stack(
                [_pad_view_tensor(item["image_features"][key], max_views) for item in batch],
                dim=0,
            )
    else:
        feature_batch = torch.stack(
            [_pad_view_tensor(item["image_features"], max_views) for item in batch],
            dim=0,
        )

    image_values = [item.get("images") for item in batch]
    image_batch = (
        None
        if any(value is None for value in image_values)
        else torch.stack([_pad_view_tensor(value, max_views) for value in image_values], dim=0)
    )

    out = {
        "case_name": [item["case_name"] for item in batch],
        "path": [item["path"] for item in batch],
        "item_index": torch.tensor([int(item["item_index"]) for item in batch], dtype=torch.long),
        "local_view_indices": torch.stack(
            [_pad_view_tensor(item["local_view_indices"], max_views) for item in batch],
            dim=0,
        ),
        "selected_view_indices": torch.stack(
            [_pad_view_tensor(item["selected_view_indices"], max_views) for item in batch],
            dim=0,
        ),
        "images": image_batch,
        "view_features": torch.stack([_pad_view_tensor(item["view_features"], max_views) for item in batch], dim=0),
        "view_mask": torch.stack([_pad_view_tensor(item["view_mask"], max_views) for item in batch], dim=0),
        "image_features": feature_batch,
        "projection_center_offset": torch.stack([item["projection_center_offset"] for item in batch], dim=0),
        "projection_center_offset_valid": torch.stack(
            [item["projection_center_offset_valid"] for item in batch],
            dim=0,
        ),
    }
    target_presence = ["target_points" in item for item in batch]
    if any(target_presence) and not all(target_presence):
        raise ValueError(
            "A batch cannot mix cases with and without 3D ground truth."
        )
    if all(target_presence):
        out["target_points"] = torch.stack(
            [item["target_points"] for item in batch],
            dim=0,
        )
        out["target_exist"] = torch.stack(
            [item["target_exist"] for item in batch],
            dim=0,
        )
        point_valid_presence = [
            item.get("target_point_valid_mask") is not None for item in batch
        ]
        if any(point_valid_presence) and not all(point_valid_presence):
            raise ValueError(
                "A batch cannot mix targets with and without point-valid masks."
            )
        if all(point_valid_presence):
            out["target_point_valid_mask"] = torch.stack(
                [item["target_point_valid_mask"] for item in batch],
                dim=0,
            )
        out["artery"] = torch.stack(
            [item["artery"] for item in batch],
            dim=0,
        )

    for optional_key in ("theta", "phi"):
        values = [item.get(optional_key) for item in batch]
        if all(value is not None for value in values):
            out[optional_key] = torch.stack([_pad_view_tensor(value, max_views) for value in values], dim=0)
        else:
            out[optional_key] = None

    mask_distance_presence = [
        item.get("projection_mask_distance_px") is not None for item in batch
    ]
    if any(mask_distance_presence) and not all(mask_distance_presence):
        raise ValueError(
            "A batch cannot mix cases with and without "
            "projection_mask_distance_px."
        )
    if all(mask_distance_presence):
        out["projection_mask_distance_px"] = torch.stack(
            [
                _pad_view_tensor(item["projection_mask_distance_px"], max_views)
                for item in batch
            ],
            dim=0,
        )

    dt_volumes = [item.get("centerline_dt_volume") for item in batch]
    if all(value is not None for value in dt_volumes):
        out["centerline_dt_volume"] = dt_volumes
        out["centerline_dt_origin"] = torch.stack([item["centerline_dt_origin"] for item in batch], dim=0)
        out["centerline_dt_spacing"] = torch.stack([item["centerline_dt_spacing"] for item in batch], dim=0)
        out["centerline_dt_max_distance_mm"] = torch.stack(
            [item["centerline_dt_max_distance_mm"] for item in batch],
            dim=0,
        )
        out["centerline_dt_valid"] = torch.stack([item["centerline_dt_valid"] for item in batch], dim=0)
    return out

def infer_feature_shapes(items: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    if not items:
        return {}
    features = items[0].get("image_features")
    if features is None:
        raw_shapes = items[0].get("image_feature_shapes")
        if not isinstance(raw_shapes, Mapping):
            return {}
        logical_view_count = int(np.asarray(items[0]["view_mask"]).shape[0])
        return {
            str(key): (
                logical_view_count,
                *(int(value) for value in tuple(shape)[1:]),
            )
            for key, shape in raw_shapes.items()
        }
    if isinstance(features, dict):
        return {key: tuple(np.asarray(value).shape) for key, value in features.items()}
    return {"image_features": tuple(np.asarray(features).shape)}

def infer_resnet_pre_fpn_channels(items: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
    shapes = infer_feature_shapes(items)
    if not shapes:
        return None
    channels = []
    for suffix in RESNET_PRE_FPN_SUFFIXES:
        shape = shapes.get(suffix)
        if shape is None:
            return None
        if len(shape) < 2:
            raise ValueError(f"Invalid cached ResNet {suffix} feature shape: {shape}")
        channels.append(int(shape[1]))
    return tuple(channels)
