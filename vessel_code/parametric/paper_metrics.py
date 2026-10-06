# Transferred from methods/parametric_methods/paper_metrics.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np

PAPER_METRIC_VOXEL_SHAPE = (128, 128, 128)

PAPER_METRIC_ISOTROPIC_SPACING_MM = 0.5

def _normalized_case_id(raw: Any) -> str:
    value = str(raw).strip()
    tokens = re.findall(r"\d+", value)
    return str(int(tokens[-1])) if tokens else value.casefold()

def _resolve_path(raw: str | Path, *, config_path: Path | None) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() and config_path is not None:
        path = config_path.parent / path
    return path.resolve()

def _finite_vector(raw: Any, *, label: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite three-value vector")
    return value

def _finite_matrix(raw: Any, *, shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return value

def _scalar_npz_value(raw: Any, *, label: str) -> Any:
    value = np.asarray(raw)
    if value.size != 1:
        raise ValueError(f"{label} must contain exactly one scalar value")
    return value.reshape(-1)[0].item()

def _scalar_npz_text(raw: Any, *, label: str) -> str:
    value = _scalar_npz_value(raw, label=label)
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

@dataclass(frozen=True)
class PaperMetricOptions:
    ground_truth_path: Path | None
    ground_truth_dir: Path | None
    save_masks: bool
    ssim_window_size: int
    volume_threshold: float
    branch_probability_threshold: float
    min_target_centerline_containment: float
    volume_key: str
    spacing_key: str
    source_nii_key: str
    origin_override_mm: np.ndarray | None
    direction_override: np.ndarray | None
    affine_override: np.ndarray | None

    def record(self) -> dict[str, Any]:
        return {
            "voxel_shape": list(PAPER_METRIC_VOXEL_SHAPE),
            "target_grid_policy": "full_native_fov_endpoint_aligned",
            "isotropic_metric_spacing_mm": PAPER_METRIC_ISOTROPIC_SPACING_MM,
            "isotropic_metric_grid_policy": (
                "source_origin_and_direction_with_exact_0.5mm_axis_spacing; "
                "shape_rounds_each_native_voxel_center_extent_to_the_nearest_"
                "0.5mm_interval"
            ),
            "ground_truth_resampling": "nearest_neighbor",
            "prediction_rasterization": (
                "union_of_filled_radius_varying_polyline_capsules"
            ),
            "ssim_protocol": "uniform_valid_window_sample_covariance",
            "cldice_protocol": (
                "binary_3d_lee_skeletonization_on_128_voxel_grid_and_"
                "0.5mm_isotropic_grid"
            ),
            "connected_component_protocol": (
                "foreground_26_neighbour_3d_connectivity_without_size_"
                "filtering_on_128_voxel_grid_and_0.5mm_isotropic_grid"
            ),
            "centerline_chamfer_protocol": (
                "sum_of_bidirectional_mean_nearest_neighbor_l2_distances_mm"
            ),
            "coordinate_convention": (
                "ground_truth_array_xyz; world_xyz_mm=index_xyz*spacing_xyz_mm "
                "unless affine/origin/direction metadata is supplied"
            ),
            "ground_truth_path": (
                None if self.ground_truth_path is None else str(self.ground_truth_path)
            ),
            "ground_truth_dir": (
                None if self.ground_truth_dir is None else str(self.ground_truth_dir)
            ),
            "save_masks": self.save_masks,
            "ssim_window_size": self.ssim_window_size,
            "volume_threshold": self.volume_threshold,
            "branch_probability_threshold": self.branch_probability_threshold,
            "min_target_centerline_containment": (
                self.min_target_centerline_containment
            ),
            "volume_key": self.volume_key,
            "spacing_key": self.spacing_key,
            "source_nii_key": self.source_nii_key,
            "origin_override_mm": (
                None
                if self.origin_override_mm is None
                else self.origin_override_mm.astype(float).tolist()
            ),
            "direction_override": (
                None
                if self.direction_override is None
                else self.direction_override.astype(float).tolist()
            ),
            "affine_override": (
                None
                if self.affine_override is None
                else self.affine_override.astype(float).tolist()
            ),
        }

def resolve_paper_metric_options(
    config: Mapping[str, Any],
    *,
    config_path: Path | None,
) -> PaperMetricOptions:
    raw_path = config.get("paper_metric_ground_truth_path")
    raw_dir = config.get("paper_metric_ground_truth_dir")
    has_path = raw_path is not None and str(raw_path).strip() != ""
    has_dir = raw_dir is not None and str(raw_dir).strip() != ""
    if has_path and has_dir:
        raise ValueError(
            "paper_metric_ground_truth_path and paper_metric_ground_truth_dir "
            "are mutually exclusive"
        )
    if not has_path and not has_dir:
        raise ValueError(
            "Paper-mask metric computation requires "
            "paper_metric_ground_truth_path (one case) or "
            "paper_metric_ground_truth_dir (multiple cases)"
        )
    path = (
        None
        if not has_path
        else _resolve_path(raw_path, config_path=config_path)
    )
    directory = (
        None
        if not has_dir
        else _resolve_path(raw_dir, config_path=config_path)
    )
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"Paper-metric ground-truth NPZ not found: {path}")
    if directory is not None and not directory.is_dir():
        raise FileNotFoundError(
            f"Paper-metric ground-truth directory not found: {directory}"
        )

    save_masks = config.get("paper_metric_save_masks", True)
    if not isinstance(save_masks, bool):
        raise ValueError("paper_metric_save_masks must be boolean")
    window_size = int(config.get("paper_metric_ssim_window_size", 7))
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(
            "paper_metric_ssim_window_size must be an odd integer >= 3"
        )
    if window_size > min(PAPER_METRIC_VOXEL_SHAPE):
        raise ValueError(
            "paper_metric_ssim_window_size cannot exceed the 128-voxel grid"
        )
    volume_threshold = float(config.get("paper_metric_volume_threshold", 0.5))
    if not math.isfinite(volume_threshold):
        raise ValueError("paper_metric_volume_threshold must be finite")
    branch_threshold = float(
        config.get("paper_metric_branch_probability_threshold", 0.5)
    )
    if not math.isfinite(branch_threshold) or not 0.0 <= branch_threshold <= 1.0:
        raise ValueError(
            "paper_metric_branch_probability_threshold must be in [0, 1]"
        )
    min_containment = float(
        config.get("paper_metric_min_target_centerline_containment", 0.9)
    )
    if not math.isfinite(min_containment) or not 0.0 <= min_containment <= 1.0:
        raise ValueError(
            "paper_metric_min_target_centerline_containment must be in [0, 1]"
        )

    origin_raw = config.get("paper_metric_volume_origin_mm")
    direction_raw = config.get("paper_metric_volume_direction")
    affine_raw = config.get("paper_metric_index_to_world_affine")
    if affine_raw is not None and (origin_raw is not None or direction_raw is not None):
        raise ValueError(
            "paper_metric_index_to_world_affine cannot be combined with "
            "paper_metric_volume_origin_mm or paper_metric_volume_direction"
        )
    origin = (
        None
        if origin_raw is None
        else _finite_vector(origin_raw, label="paper_metric_volume_origin_mm")
    )
    direction = (
        None
        if direction_raw is None
        else _finite_matrix(
            direction_raw,
            shape=(3, 3),
            label="paper_metric_volume_direction",
        )
    )
    affine = (
        None
        if affine_raw is None
        else _finite_matrix(
            affine_raw,
            shape=(4, 4),
            label="paper_metric_index_to_world_affine",
        )
    )
    if affine is not None and not np.allclose(
        affine[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8
    ):
        raise ValueError(
            "paper_metric_index_to_world_affine must have final row [0,0,0,1]"
        )

    return PaperMetricOptions(
        ground_truth_path=path,
        ground_truth_dir=directory,
        save_masks=save_masks,
        ssim_window_size=window_size,
        volume_threshold=volume_threshold,
        branch_probability_threshold=branch_threshold,
        min_target_centerline_containment=min_containment,
        volume_key=str(config.get("paper_metric_volume_key", "vol")),
        spacing_key=str(config.get("paper_metric_spacing_key", "spacing")),
        source_nii_key=str(config.get("paper_metric_source_nii_key", "source_nii")),
        origin_override_mm=origin,
        direction_override=direction,
        affine_override=affine,
    )

def infer_ground_truth_case_id(
    path: Path,
    *,
    source_nii_key: str = "source_nii",
) -> str | None:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("case_id", "source_case_id", "sample_name"):
            if key in payload.files:
                return _normalized_case_id(
                    _scalar_npz_text(payload[key], label=f"{path} {key}")
                )
        if source_nii_key in payload.files:
            source = _scalar_npz_text(
                payload[source_nii_key],
                label=f"{path} {source_nii_key}",
            )
            tokens = re.findall(r"\d+", Path(source).name)
            if tokens:
                return str(int(tokens[-1]))
    candidate = path.parent.name if path.parent.name.isdigit() else path.stem
    tokens = re.findall(r"\d+", candidate)
    return str(int(tokens[-1])) if tokens else None

def infer_ground_truth_artery_type(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("artery_type", "vessel_type", "anatomy"):
            if key not in payload.files:
                continue
            value = _scalar_npz_text(
                payload[key], label=f"{path} {key}"
            ).strip().upper()
            if value in {"RCA", "LCA"}:
                return value
    for token in (path.stem, *[parent.name for parent in path.parents]):
        matches = re.findall(
            r"(?:^|[^A-Za-z])(RCA|LCA)(?:[^A-Za-z]|$)", token, re.I
        )
        if matches:
            return matches[-1].upper()
    return None

def resolve_ground_truth_volume_paths(
    options: PaperMetricOptions,
    case_ids: Iterable[str],
    *,
    artery_type: str,
) -> dict[str, Path]:
    normalized_artery_type = str(artery_type).strip().upper()
    if normalized_artery_type not in {"RCA", "LCA"}:
        raise ValueError("artery_type must be 'RCA' or 'LCA'")
    requested = {_normalized_case_id(case_id) for case_id in case_ids}
    if not requested:
        raise ValueError("Paper-metric evaluation selected no case IDs")
    if options.ground_truth_path is not None:
        if len(requested) != 1:
            raise ValueError(
                "paper_metric_ground_truth_path can only be used when exactly "
                "one base case is selected"
            )
        requested_id = next(iter(requested))
        inferred = infer_ground_truth_case_id(
            options.ground_truth_path,
            source_nii_key=options.source_nii_key,
        )
        if inferred is not None and inferred != requested_id:
            raise ValueError(
                f"Ground-truth NPZ {options.ground_truth_path} identifies case "
                f"{inferred}, but evaluation selected case {requested_id}"
            )
        inferred_artery = infer_ground_truth_artery_type(
            options.ground_truth_path
        )
        if (
            inferred_artery is not None
            and inferred_artery != normalized_artery_type
        ):
            raise ValueError(
                f"Ground-truth NPZ {options.ground_truth_path} identifies "
                f"{inferred_artery}, but evaluation artery_type is "
                f"{normalized_artery_type}"
            )
        return {requested_id: options.ground_truth_path}

    assert options.ground_truth_dir is not None
    matches: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(options.ground_truth_dir.rglob("*.npz")):
        case_id = infer_ground_truth_case_id(
            path,
            source_nii_key=options.source_nii_key,
        )
        path_artery = infer_ground_truth_artery_type(path)
        if (
            case_id is None
            or case_id not in requested
            or (
                path_artery is not None
                and path_artery != normalized_artery_type
            )
        ):
            continue
        if case_id in matches:
            duplicates.setdefault(case_id, [matches[case_id]]).append(path)
        else:
            matches[case_id] = path
    if duplicates:
        details = "; ".join(
            f"case {case_id}: {paths}" for case_id, paths in duplicates.items()
        )
        raise ValueError(f"Duplicate paper-metric ground-truth volumes: {details}")
    missing = sorted(requested - set(matches))
    if missing:
        raise FileNotFoundError(
            "No paper-metric ground-truth NPZ was found for case IDs "
            f"{missing} under {options.ground_truth_dir}"
        )
    return matches

@dataclass(frozen=True)
class GroundTruthVolume:
    mask: np.ndarray
    spacing_mm: np.ndarray
    index_to_world_affine: np.ndarray
    source_path: Path
    source_nii: str | None
    assumptions: tuple[str, ...]

def _optional_npz_array(
    payload: Any,
    keys: tuple[str, ...],
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key in payload.files:
            return np.asarray(payload[key]), key
    return None, None

def load_ground_truth_volume(
    path: Path,
    options: PaperMetricOptions,
) -> GroundTruthVolume:
    assumptions: list[str] = []
    with np.load(path, allow_pickle=False) as payload:
        if options.volume_key not in payload.files:
            raise KeyError(
                f"{path} does not contain paper-metric volume key "
                f"{options.volume_key!r}; available={payload.files}"
            )
        if options.spacing_key not in payload.files:
            raise KeyError(
                f"{path} does not contain paper-metric spacing key "
                f"{options.spacing_key!r}; available={payload.files}"
            )
        volume = np.asarray(payload[options.volume_key])
        spacing = _finite_vector(
            payload[options.spacing_key],
            label=f"{path} {options.spacing_key}",
        )
        if np.any(spacing <= 0.0):
            raise ValueError(f"{path} spacing values must all be positive")
        if volume.ndim != 3:
            raise ValueError(
                f"{path} {options.volume_key} must be 3D, got {volume.shape}"
            )
        if not np.issubdtype(volume.dtype, np.number) and volume.dtype != np.bool_:
            raise ValueError(f"{path} volume must be numeric or boolean")
        if np.issubdtype(volume.dtype, np.floating) and not np.isfinite(volume).all():
            raise ValueError(f"{path} volume contains NaN/Inf values")
        mask = np.asarray(volume > options.volume_threshold, dtype=np.bool_)
        if not bool(mask.any()):
            raise ValueError(f"{path} ground-truth mask has no foreground voxels")

        source_nii = (
            _scalar_npz_text(
                payload[options.source_nii_key],
                label=f"{path} {options.source_nii_key}",
            )
            if options.source_nii_key in payload.files
            else None
        )
        if options.affine_override is not None:
            affine = options.affine_override.copy()
            assumptions.append("index-to-world affine supplied by evaluation config")
        else:
            stored_affine, affine_key = _optional_npz_array(
                payload,
                ("index_to_world_affine", "affine"),
            )
            if stored_affine is not None:
                affine = _finite_matrix(
                    stored_affine,
                    shape=(4, 4),
                    label=f"{path} {affine_key}",
                )
            else:
                stored_origin, origin_key = _optional_npz_array(
                    payload,
                    ("origin_mm", "origin"),
                )
                stored_direction, direction_key = _optional_npz_array(
                    payload,
                    ("direction", "direction_matrix"),
                )
                if options.origin_override_mm is not None:
                    origin = options.origin_override_mm.copy()
                    assumptions.append("volume origin supplied by evaluation config")
                elif stored_origin is not None:
                    origin = _finite_vector(
                        stored_origin,
                        label=f"{path} {origin_key}",
                    )
                else:
                    origin = np.zeros((3,), dtype=np.float64)
                    assumptions.append(
                        "volume NPZ has no origin/affine; assumed origin [0,0,0] mm"
                    )
                if options.direction_override is not None:
                    direction = options.direction_override.copy()
                    assumptions.append("volume direction supplied by evaluation config")
                elif stored_direction is not None:
                    direction = _finite_matrix(
                        stored_direction,
                        shape=(3, 3),
                        label=f"{path} {direction_key}",
                    )
                else:
                    direction = np.eye(3, dtype=np.float64)
                    assumptions.append(
                        "volume NPZ has no direction/affine; assumed identity direction"
                    )
                affine = np.eye(4, dtype=np.float64)
                affine[:3, :3] = direction @ np.diag(spacing)
                affine[:3, 3] = origin

    if not np.allclose(
        affine[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8
    ):
        raise ValueError(f"{path} index-to-world affine has an invalid final row")
    linear = affine[:3, :3]
    if abs(float(np.linalg.det(linear))) <= 1e-12:
        raise ValueError(f"{path} index-to-world affine is singular")
    return GroundTruthVolume(
        mask=mask,
        spacing_mm=spacing.astype(np.float64),
        index_to_world_affine=affine.astype(np.float64),
        source_path=path,
        source_nii=source_nii,
        assumptions=tuple(assumptions),
    )

def restore_absolute_vessel_coordinates(
    vessel_mm: np.ndarray,
    *,
    target_coordinate_frame: str,
    centering_offset_mm: np.ndarray,
) -> np.ndarray:
    vessel = np.asarray(vessel_mm, dtype=np.float32).copy()
    if vessel.ndim != 3 or vessel.shape[-1] < 4:
        raise ValueError(
            "Decoded vessel must have shape [M,N,>=4], "
            f"got {vessel.shape}"
        )
    frame = str(target_coordinate_frame).strip().lower()
    if frame == "projection_centered":
        offset = _finite_vector(
            centering_offset_mm,
            label="parametric_centering_offset_mm",
        ).astype(np.float32)
        vessel[..., :3] += offset.reshape(1, 1, 3)
    elif frame != "absolute_world":
        raise ValueError(
            "target_coordinate_frame must be 'projection_centered' or "
            f"'absolute_world', got {target_coordinate_frame!r}"
        )
    return vessel

def resample_binary_volume_nearest(
    mask: np.ndarray,
    output_shape: tuple[int, int, int] = PAPER_METRIC_VOXEL_SHAPE,
) -> np.ndarray:
    source = np.asarray(mask, dtype=np.bool_)
    if source.ndim != 3:
        raise ValueError(f"Binary volume must be 3D, got {source.shape}")
    target_shape = tuple(int(value) for value in output_shape)
    if len(target_shape) != 3 or min(target_shape) < 2:
        raise ValueError("output_shape must contain three values >= 2")
    if min(source.shape) < 2:
        raise ValueError(
            "Endpoint-aligned resampling requires every source dimension >= 2"
        )
    indices = [
        np.floor(
            np.linspace(0.0, float(source_size - 1), target_size) + 0.5
        ).astype(np.int64)
        for source_size, target_size in zip(source.shape, target_shape)
    ]
    return source[np.ix_(*indices)].astype(np.bool_, copy=False)

def target_grid_index_to_world_affine(
    *,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    source_index_to_world_affine: np.ndarray,
) -> np.ndarray:
    source_size = np.asarray(source_shape, dtype=np.float64)
    target_size = np.asarray(target_shape, dtype=np.float64)
    if np.any(source_size < 2.0) or np.any(target_size < 2.0):
        raise ValueError(
            "Endpoint-aligned grids require all source and target dimensions >= 2"
        )
    ratio = (source_size - 1.0) / (target_size - 1.0)
    source_affine = np.asarray(source_index_to_world_affine, dtype=np.float64)
    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = source_affine[:3, :3] @ np.diag(ratio)
    target_affine[:3, 3] = source_affine[:3, 3]
    return target_affine

def isotropic_grid_from_source(
    *,
    source_shape: tuple[int, int, int],
    source_index_to_world_affine: np.ndarray,
    spacing_mm: float = PAPER_METRIC_ISOTROPIC_SPACING_MM,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Build an exact-spacing grid aligned to the source voxel axes.

    The source and target origins and axis directions are identical. Each target
    dimension rounds the native voxel-centre extent to the nearest number of
    isotropic intervals, so its far endpoint differs from the native endpoint by
    at most half a target voxel.
    """

    shape = tuple(int(value) for value in source_shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(
            "Isotropic paper-metric resampling requires three source dimensions "
            ">= 2"
        )
    spacing = float(spacing_mm)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(
            "Isotropic paper-metric spacing_mm must be finite and positive"
        )
    source_affine = np.asarray(source_index_to_world_affine, dtype=np.float64)
    if source_affine.shape != (4, 4) or not np.isfinite(source_affine).all():
        raise ValueError("source_index_to_world_affine must be a finite 4x4 matrix")
    linear = source_affine[:3, :3]
    axis_spacing = np.linalg.norm(linear, axis=0)
    if np.any(axis_spacing <= 0.0):
        raise ValueError(
            "source_index_to_world_affine has a zero-length voxel axis"
        )
    direction = linear / axis_spacing[None, :]
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(
            "Exact isotropic paper metrics require orthogonal source voxel axes; "
            "the supplied affine contains shear"
        )
    extent_mm = (np.asarray(shape, dtype=np.float64) - 1.0) * axis_spacing
    interval_count = np.maximum(
        1,
        np.floor(extent_mm / spacing + 0.5).astype(np.int64),
    )
    target_shape = tuple(int(value + 1) for value in interval_count)
    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = direction * spacing
    target_affine[:3, 3] = source_affine[:3, 3]
    return target_shape, target_affine

def resample_binary_volume_nearest_affine(
    mask: np.ndarray,
    *,
    source_index_to_world_affine: np.ndarray,
    output_shape: tuple[int, int, int],
    output_index_to_world_affine: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resampling between grids with matching axis directions."""

    source = np.asarray(mask, dtype=np.bool_)
    if source.ndim != 3:
        raise ValueError(f"Binary volume must be 3D, got {source.shape}")
    target_shape = tuple(int(value) for value in output_shape)
    if len(target_shape) != 3 or min(target_shape) < 1:
        raise ValueError("output_shape must contain three positive values")
    source_affine = np.asarray(source_index_to_world_affine, dtype=np.float64)
    target_affine = np.asarray(output_index_to_world_affine, dtype=np.float64)
    if source_affine.shape != (4, 4) or target_affine.shape != (4, 4):
        raise ValueError("Source and output index-to-world affines must be 4x4")
    source_from_target = np.linalg.inv(source_affine) @ target_affine
    if not np.allclose(
        source_from_target[:3, :3],
        np.diag(np.diag(source_from_target[:3, :3])),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "Nearest affine resampling currently requires matching source and "
            "target voxel-axis directions"
        )
    source_coordinates = [
        source_from_target[axis, axis]
        * np.arange(target_shape[axis], dtype=np.float64)
        + source_from_target[axis, 3]
        for axis in range(3)
    ]
    nearest = [
        np.floor(coordinates + 0.5).astype(np.int64)
        for coordinates in source_coordinates
    ]
    valid = [
        (indices >= 0) & (indices < source.shape[axis])
        for axis, indices in enumerate(nearest)
    ]
    clipped = [
        np.clip(indices, 0, source.shape[axis] - 1)
        for axis, indices in enumerate(nearest)
    ]
    result = source[np.ix_(*clipped)].copy()
    result &= valid[0][:, None, None]
    result &= valid[1][None, :, None]
    result &= valid[2][None, None, :]
    return result

def _indices_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return (
        np.asarray(indices, dtype=np.float64) @ affine[:3, :3].T
        + affine[:3, 3]
    )

def _world_to_indices(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inverse_linear = np.linalg.inv(affine[:3, :3])
    return (
        np.asarray(points, dtype=np.float64) - affine[:3, 3]
    ) @ inverse_linear.T

def rasterize_vessel_to_mask(
    vessel_absolute_mm: np.ndarray,
    branch_exists: np.ndarray,
    *,
    source_volume_shape: tuple[int, int, int],
    source_index_to_world_affine: np.ndarray,
    output_shape: tuple[int, int, int] = PAPER_METRIC_VOXEL_SHAPE,
    output_index_to_world_affine: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vessel = np.asarray(vessel_absolute_mm, dtype=np.float64)
    exists = np.asarray(branch_exists, dtype=np.bool_).reshape(-1)
    if vessel.ndim != 3 or vessel.shape[-1] < 4:
        raise ValueError(f"vessel_absolute_mm must be [M,N,>=4], got {vessel.shape}")
    if exists.shape != (vessel.shape[0],):
        raise ValueError(
            f"branch_exists must have shape {(vessel.shape[0],)}, got {exists.shape}"
        )
    target_shape = tuple(int(value) for value in output_shape)
    target_affine = (
        target_grid_index_to_world_affine(
            source_shape=tuple(int(value) for value in source_volume_shape),
            target_shape=target_shape,
            source_index_to_world_affine=source_index_to_world_affine,
        )
        if output_index_to_world_affine is None
        else _finite_matrix(
            output_index_to_world_affine,
            shape=(4, 4),
            label="output_index_to_world_affine",
        )
    )
    if not np.allclose(
        target_affine[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-8,
    ):
        raise ValueError(
            "output_index_to_world_affine must have final row [0,0,0,1]"
        )
    if abs(float(np.linalg.det(target_affine[:3, :3]))) <= 1e-12:
        raise ValueError("output_index_to_world_affine must be nonsingular")
    target_inverse_linear = np.linalg.inv(target_affine[:3, :3])
    radius_index_extent_per_mm = np.linalg.norm(
        target_inverse_linear,
        axis=1,
    )
    mask = np.zeros(target_shape, dtype=np.bool_)

    if not bool(exists.any()):
        raise ValueError("Predicted vessel has no active branches")
    active_vessel = vessel[exists]
    if not np.isfinite(active_vessel).all():
        raise ValueError("Predicted active branches contain NaN/Inf values")
    if np.any(active_vessel[..., 3] <= 0.0):
        raise ValueError(
            "Predicted active branches contain non-positive radii"
        )
    valid_points = np.broadcast_to(exists[:, None], vessel.shape[:2]).copy()
    centers = vessel[..., :3][valid_points]
    radii = vessel[..., 3][valid_points]
    if centers.size == 0:
        raise ValueError("Predicted vessel has no finite positive-radius points")
    target_center_indices = _world_to_indices(centers, target_affine)
    centers_inside = np.all(
        (target_center_indices >= -0.5)
        & (target_center_indices < np.asarray(target_shape) - 0.5),
        axis=1,
    )

    physical_axis_lengths = np.linalg.norm(target_affine[:3, :3], axis=0)
    maximum_fov_length = float(
        np.max(
            physical_axis_lengths
            * (np.asarray(target_shape, dtype=np.float64) - 1.0)
        )
    )
    if float(np.max(radii)) > maximum_fov_length:
        raise ValueError(
            "Predicted vessel radius exceeds the physical volume extent; "
            f"max radius={float(np.max(radii)):.6g} mm, "
            f"max FOV length={maximum_fov_length:.6g} mm"
        )

    def paint_segment(
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        start_radius: float,
        end_radius: float,
    ) -> None:
        endpoint_indices = _world_to_indices(
            np.stack((start_xyz, end_xyz), axis=0),
            target_affine,
        )
        max_radius = max(float(start_radius), float(end_radius))
        expansion = max_radius * radius_index_extent_per_mm + 1.0
        lower = np.floor(endpoint_indices.min(axis=0) - expansion).astype(np.int64)
        upper = np.ceil(endpoint_indices.max(axis=0) + expansion).astype(np.int64)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(target_shape) - 1)
        if np.any(lower > upper):
            return

        axes = [
            np.arange(lower[axis], upper[axis] + 1, dtype=np.int64)
            for axis in range(3)
        ]
        grid = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        )
        grid_flat = grid.reshape(-1, 3)
        world = _indices_to_world(grid_flat, target_affine)
        segment = end_xyz - start_xyz
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            t = np.zeros((world.shape[0],), dtype=np.float64)
            closest = np.broadcast_to(start_xyz, world.shape)
        else:
            t = np.clip(
                ((world - start_xyz) @ segment) / length_squared,
                0.0,
                1.0,
            )
            closest = start_xyz[None, :] + t[:, None] * segment[None, :]
        local_radius = float(start_radius) + t * (
            float(end_radius) - float(start_radius)
        )
        inside = np.sum((world - closest) ** 2, axis=1) <= local_radius ** 2
        slices = tuple(
            slice(int(lower[axis]), int(upper[axis]) + 1)
            for axis in range(3)
        )
        mask[slices] |= inside.reshape(grid.shape[:3])

    for branch_index in np.flatnonzero(exists):
        branch = vessel[branch_index]
        branch_valid = valid_points[branch_index]
        valid_indices = np.flatnonzero(branch_valid)
        if valid_indices.size == 1:
            point_index = int(valid_indices[0])
            paint_segment(
                branch[point_index, :3],
                branch[point_index, :3],
                float(branch[point_index, 3]),
                float(branch[point_index, 3]),
            )
        for point_index in range(branch.shape[0] - 1):
            if not (branch_valid[point_index] and branch_valid[point_index + 1]):
                continue
            paint_segment(
                branch[point_index, :3],
                branch[point_index + 1, :3],
                float(branch[point_index, 3]),
                float(branch[point_index + 1, 3]),
            )
    return mask, {
        "num_active_branches": int(exists.sum()),
        "num_centerline_points": int(centers.shape[0]),
        "num_centerline_points_inside_volume": int(centers_inside.sum()),
        "num_centerline_points_outside_volume": int((~centers_inside).sum()),
        "centerline_points_inside_volume_fraction": float(centers_inside.mean()),
        "target_index_to_world_affine": target_affine.astype(float).tolist(),
        "target_spacing_mm": np.linalg.norm(
            target_affine[:3, :3], axis=0
        ).astype(float).tolist(),
        "target_origin_mm": target_affine[:3, 3].astype(float).tolist(),
    }

def dice_similarity_3d(predicted: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=np.bool_)
    gt = np.asarray(target, dtype=np.bool_)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(
            f"Dice expects matching 3D masks, got {pred.shape} and {gt.shape}"
        )
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    denominator = pred_count + gt_count
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred, gt).sum())
    return float(2.0 * intersection / denominator)

def connected_component_count_3d(mask: np.ndarray) -> int:
    """Count foreground components using full 26-neighbour connectivity."""

    volume = np.asarray(mask, dtype=np.bool_)
    if volume.ndim != 3:
        raise ValueError(
            "3D connected-component counting expects a 3D mask, got "
            f"{volume.shape}"
        )
    if not bool(volume.any()):
        return 0
    try:
        from scipy.ndimage import label
    except ImportError as error:
        raise ImportError(
            "Paper-metric connected-component counting requires SciPy. "
            "Install the repository requirements."
        ) from error

    occupied_axes = [
        np.flatnonzero(volume.any(axis=tuple(other_axes)))
        for other_axes in ((1, 2), (0, 2), (0, 1))
    ]
    crop = tuple(
        slice(int(indices[0]), int(indices[-1]) + 1)
        for indices in occupied_axes
    )
    _, count = label(
        volume[crop],
        structure=np.ones((3, 3, 3), dtype=np.bool_),
    )
    return int(count)

def connected_component_metrics_3d(
    predicted_volume: np.ndarray,
    target_volume: np.ndarray,
) -> dict[str, int]:
    """Record prediction/target component counts and disconnection flags."""

    predicted = np.asarray(predicted_volume, dtype=np.bool_)
    target = np.asarray(target_volume, dtype=np.bool_)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            "3D connected-component metrics expect matching 3D masks, got "
            f"{predicted.shape} and {target.shape}"
        )
    predicted_count = connected_component_count_3d(predicted)
    target_count = connected_component_count_3d(target)
    return {
        "predicted_connected_components_3d": predicted_count,
        "ground_truth_connected_components_3d": target_count,
        "connected_component_count_absolute_error_3d": abs(
            predicted_count - target_count
        ),
        "predicted_disconnected_3d": int(predicted_count > 1),
        "ground_truth_disconnected_3d": int(target_count > 1),
    }

def symmetric_chamfer_distance(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> float:
    """Return the summed bidirectional mean nearest-neighbour L2 distance.

    This follows the paper-metric definition

    ``mean_p min_q ||p-q||_2 + mean_q min_p ||p-q||_2``

    without the additional factor of one half used by some Chamfer variants.
    Input coordinates therefore determine the output unit; the evaluator
    passes millimetre centreline coordinates.
    """

    predicted = np.asarray(predicted_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if (
        predicted.ndim != 2
        or target.ndim != 2
        or predicted.shape[1:] != (3,)
        or target.shape[1:] != (3,)
    ):
        raise ValueError(
            "Chamfer distance expects point sets shaped [N,3] and [M,3], "
            f"got {predicted.shape} and {target.shape}"
        )
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("Chamfer distance requires two non-empty point sets")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Chamfer distance point sets must be finite")
    block_size = int(chunk_size)
    if block_size < 1:
        raise ValueError("Chamfer distance chunk_size must be >= 1")

    def directed_mean(source: np.ndarray, destination: np.ndarray) -> float:
        minimum_distances: list[np.ndarray] = []
        for start in range(0, source.shape[0], block_size):
            block = source[start : start + block_size]
            squared_distances = np.sum(
                (block[:, None, :] - destination[None, :, :]) ** 2,
                axis=-1,
            )
            minimum_distances.append(
                np.sqrt(np.min(squared_distances, axis=1))
            )
        return float(
            np.concatenate(minimum_distances).mean(dtype=np.float64)
        )

    return float(
        directed_mean(predicted, target)
        + directed_mean(target, predicted)
    )

def skeletonize_binary_volume_3d(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel 3D Lee skeleton with a false padded boundary."""

    volume = np.asarray(mask, dtype=np.bool_)
    if volume.ndim != 3:
        raise ValueError(
            f"3D skeletonization expects a 3D mask, got {volume.shape}"
        )
    skeleton = np.zeros_like(volume, dtype=np.bool_)
    foreground_indices = np.argwhere(volume)
    if foreground_indices.size == 0:
        return skeleton
    try:
        from skimage.morphology import skeletonize
    except ImportError as error:
        raise ImportError(
            "Paper-metric clDice requires scikit-image for 3D Lee "
            "skeletonization. Install the repository requirements."
        ) from error

    lower = foreground_indices.min(axis=0)
    upper = foreground_indices.max(axis=0) + 1
    crop_slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper)
    )
    cropped = volume[crop_slices]
    padded = np.pad(
        cropped,
        1,
        mode="constant",
        constant_values=False,
    )
    padded_skeleton = np.asarray(
        skeletonize(padded, method="lee"),
        dtype=np.bool_,
    )
    skeleton[crop_slices] = padded_skeleton[1:-1, 1:-1, 1:-1]
    return skeleton

def cldice_metrics_3d(
    predicted_volume: np.ndarray,
    target_volume: np.ndarray,
) -> dict[str, Any]:
    """Compute symmetric 3D clDice from thinned prediction/target masks."""

    predicted = np.asarray(predicted_volume, dtype=np.bool_)
    target = np.asarray(target_volume, dtype=np.bool_)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            "3D clDice expects matching 3D masks, got "
            f"{predicted.shape} and {target.shape}"
        )
    predicted_centerline = skeletonize_binary_volume_3d(predicted)
    target_centerline = skeletonize_binary_volume_3d(target)
    predicted_centerline_count = int(predicted_centerline.sum())
    target_centerline_count = int(target_centerline.sum())
    predicted_centerline_in_target = int(
        np.logical_and(predicted_centerline, target).sum()
    )
    target_centerline_in_prediction = int(
        np.logical_and(target_centerline, predicted).sum()
    )
    topology_precision = (
        float(predicted_centerline_in_target / predicted_centerline_count)
        if predicted_centerline_count
        else (1.0 if not bool(target.any()) else 0.0)
    )
    topology_sensitivity = (
        float(target_centerline_in_prediction / target_centerline_count)
        if target_centerline_count
        else (1.0 if not bool(predicted.any()) else 0.0)
    )
    topology_sum = topology_precision + topology_sensitivity
    score = (
        float(
            2.0
            * topology_precision
            * topology_sensitivity
            / topology_sum
        )
        if topology_sum > 0.0
        else 0.0
    )
    return {
        "cldice_3d": score,
        "cldice_loss_3d": float(1.0 - score),
        "cldice_topology_precision": topology_precision,
        "cldice_topology_sensitivity": topology_sensitivity,
        "predicted_centerline_voxels": predicted_centerline_count,
        "ground_truth_centerline_voxels": target_centerline_count,
        "predicted_centerline_in_ground_truth_voxels": (
            predicted_centerline_in_target
        ),
        "ground_truth_centerline_in_prediction_voxels": (
            target_centerline_in_prediction
        ),
        "predicted_centerline_mask": predicted_centerline,
        "ground_truth_centerline_mask": target_centerline,
    }

def structural_similarity_3d(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    window_size: int = 7,
) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(
            f"3D SSIM expects matching 3D masks, got {pred.shape} and {gt.shape}"
        )
    k = int(window_size)
    if k < 3 or k % 2 == 0 or k > min(pred.shape):
        raise ValueError(
            "3D SSIM window_size must be odd, >= 3, and no larger than the "
            "smallest volume dimension"
        )

    def box_mean_valid(volume: np.ndarray) -> np.ndarray:
        integral = np.pad(
            volume,
            ((1, 0), (1, 0), (1, 0)),
            mode="constant",
        ).cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
        sums = (
            integral[k:, k:, k:]
            - integral[:-k, k:, k:]
            - integral[k:, :-k, k:]
            - integral[k:, k:, :-k]
            + integral[:-k, :-k, k:]
            + integral[:-k, k:, :-k]
            + integral[k:, :-k, :-k]
            - integral[:-k, :-k, :-k]
        )
        return sums / float(k ** 3)

    mu_pred = box_mean_valid(pred)
    mu_gt = box_mean_valid(gt)
    covariance_scale = float(k ** 3) / float(k ** 3 - 1)
    pred_variance = covariance_scale * (
        box_mean_valid(pred * pred) - mu_pred * mu_pred
    )
    gt_variance = covariance_scale * (
        box_mean_valid(gt * gt) - mu_gt * mu_gt
    )
    covariance = covariance_scale * (
        box_mean_valid(pred * gt) - mu_pred * mu_gt
    )
    pred_variance = np.maximum(pred_variance, 0.0)
    gt_variance = np.maximum(gt_variance, 0.0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_pred * mu_gt + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        (mu_pred * mu_pred + mu_gt * mu_gt + c1)
        * (pred_variance + gt_variance + c2)
    )
    score = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0.0,
    )
    return float(np.mean(np.clip(score, -1.0, 1.0), dtype=np.float64))

def centerline_ground_truth_overlap(
    vessel_absolute_mm: np.ndarray,
    branch_exists: np.ndarray,
    ground_truth: GroundTruthVolume,
    *,
    point_valid_mask: np.ndarray | None = None,
    require_positive_radius: bool = True,
) -> dict[str, Any]:
    vessel = np.asarray(vessel_absolute_mm, dtype=np.float64)
    exists = np.asarray(branch_exists, dtype=np.bool_).reshape(-1)
    if vessel.ndim != 3 or vessel.shape[-1] < 4:
        raise ValueError(
            f"vessel_absolute_mm must be [M,N,>=4], got {vessel.shape}"
        )
    if exists.shape != (vessel.shape[0],):
        raise ValueError(
            f"branch_exists must have shape {(vessel.shape[0],)}, got "
            f"{exists.shape}"
        )
    point_valid = (
        np.ones(vessel.shape[:2], dtype=np.bool_)
        if point_valid_mask is None
        else np.asarray(point_valid_mask, dtype=np.bool_)
    )
    if point_valid.shape != vessel.shape[:2]:
        raise ValueError(
            f"point_valid_mask must have shape {vessel.shape[:2]}, got "
            f"{point_valid.shape}"
        )
    valid = (
        exists[:, None]
        & point_valid
        & np.isfinite(vessel[..., :3]).all(axis=-1)
    )
    if require_positive_radius:
        valid &= np.isfinite(vessel[..., 3]) & (vessel[..., 3] > 0.0)
    points = vessel[..., :3][valid]
    if not points.size:
        raise ValueError("No valid centerline points are available for containment")
    native_indices = _world_to_indices(
        points,
        ground_truth.index_to_world_affine,
    )
    rounded = np.floor(native_indices + 0.5).astype(np.int64)
    in_bounds = np.all(
        (rounded >= 0) & (rounded < np.asarray(ground_truth.mask.shape)),
        axis=1,
    )
    foreground = np.zeros((points.shape[0],), dtype=np.bool_)
    bounded = rounded[in_bounds]
    foreground[in_bounds] = ground_truth.mask[
        bounded[:, 0], bounded[:, 1], bounded[:, 2]
    ]
    return {
        "centerline_gt_in_bounds_fraction": float(in_bounds.mean()),
        "centerline_gt_foreground_fraction": float(foreground.mean()),
        "centerline_gt_foreground_fraction_given_in_bounds": (
            float(foreground[in_bounds].mean()) if bool(in_bounds.any()) else 0.0
        ),
    }

def compute_paper_mask_metrics(
    *,
    vessel_absolute_mm: np.ndarray,
    branch_exists: np.ndarray,
    ground_truth: GroundTruthVolume,
    ssim_window_size: int,
) -> dict[str, Any]:
    ground_truth_mask = resample_binary_volume_nearest(
        ground_truth.mask,
        PAPER_METRIC_VOXEL_SHAPE,
    )
    if not bool(ground_truth_mask.any()):
        raise ValueError(
            "Paper-metric nearest-neighbor resampling removed all ground-truth "
            f"foreground voxels from {ground_truth.source_path}; use a denser "
            "source mask or a separately specified physical-grid protocol."
        )
    predicted_mask, rasterization = rasterize_vessel_to_mask(
        vessel_absolute_mm,
        branch_exists,
        source_volume_shape=ground_truth.mask.shape,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
        output_shape=PAPER_METRIC_VOXEL_SHAPE,
    )
    cldice_result = cldice_metrics_3d(
        predicted_mask,
        ground_truth_mask,
    )
    connected_component_result = connected_component_metrics_3d(
        predicted_mask,
        ground_truth_mask,
    )
    isotropic_shape, isotropic_affine = isotropic_grid_from_source(
        source_shape=ground_truth.mask.shape,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
    )
    ground_truth_mask_0p5mm = resample_binary_volume_nearest_affine(
        ground_truth.mask,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
        output_shape=isotropic_shape,
        output_index_to_world_affine=isotropic_affine,
    )
    if not bool(ground_truth_mask_0p5mm.any()):
        raise ValueError(
            "Paper-metric 0.5 mm nearest-neighbor resampling removed all "
            f"ground-truth foreground voxels from {ground_truth.source_path}"
        )
    predicted_mask_0p5mm, rasterization_0p5mm = rasterize_vessel_to_mask(
        vessel_absolute_mm,
        branch_exists,
        source_volume_shape=ground_truth.mask.shape,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
        output_shape=isotropic_shape,
        output_index_to_world_affine=isotropic_affine,
    )
    cldice_result_0p5mm = cldice_metrics_3d(
        predicted_mask_0p5mm,
        ground_truth_mask_0p5mm,
    )
    connected_component_result_0p5mm = connected_component_metrics_3d(
        predicted_mask_0p5mm,
        ground_truth_mask_0p5mm,
    )
    metrics: dict[str, int | float] = {
        "paper_mask_dice_3d": dice_similarity_3d(
            predicted_mask, ground_truth_mask
        ),
        "paper_mask_ssim_3d": structural_similarity_3d(
            predicted_mask,
            ground_truth_mask,
            window_size=ssim_window_size,
        ),
        **{
            f"paper_mask_{key}": value
            for key, value in cldice_result.items()
            if isinstance(value, (int, float))
        },
        **{
            f"paper_mask_{key}": value
            for key, value in connected_component_result.items()
        },
        "paper_mask_dice_3d_0p5mm": dice_similarity_3d(
            predicted_mask_0p5mm,
            ground_truth_mask_0p5mm,
        ),
        **{
            f"paper_mask_{key}_0p5mm": value
            for key, value in cldice_result_0p5mm.items()
            if isinstance(value, (int, float))
        },
        **{
            f"paper_mask_{key}_0p5mm": value
            for key, value in connected_component_result_0p5mm.items()
        },
        "paper_mask_predicted_foreground_voxels": int(predicted_mask.sum()),
        "paper_mask_ground_truth_foreground_voxels": int(
            ground_truth_mask.sum()
        ),
        "paper_mask_ground_truth_native_foreground_voxels": int(
            ground_truth.mask.sum()
        ),
        "paper_mask_intersection_voxels": int(
            np.logical_and(predicted_mask, ground_truth_mask).sum()
        ),
        "paper_mask_predicted_foreground_voxels_0p5mm": int(
            predicted_mask_0p5mm.sum()
        ),
        "paper_mask_ground_truth_foreground_voxels_0p5mm": int(
            ground_truth_mask_0p5mm.sum()
        ),
        "paper_mask_intersection_voxels_0p5mm": int(
            np.logical_and(
                predicted_mask_0p5mm,
                ground_truth_mask_0p5mm,
            ).sum()
        ),
        **{
            f"paper_mask_{key}": value
            for key, value in rasterization.items()
            if isinstance(value, (int, float))
        },
        **{
            f"paper_mask_{key}": value
            for key, value in centerline_ground_truth_overlap(
                vessel_absolute_mm,
                branch_exists,
                ground_truth,
            ).items()
        },
    }
    return {
        "predicted_mask": predicted_mask,
        "ground_truth_mask": ground_truth_mask,
        "predicted_centerline_mask": cldice_result[
            "predicted_centerline_mask"
        ],
        "ground_truth_centerline_mask": cldice_result[
            "ground_truth_centerline_mask"
        ],
        "predicted_mask_0p5mm": predicted_mask_0p5mm,
        "ground_truth_mask_0p5mm": ground_truth_mask_0p5mm,
        "predicted_centerline_mask_0p5mm": cldice_result_0p5mm[
            "predicted_centerline_mask"
        ],
        "ground_truth_centerline_mask_0p5mm": cldice_result_0p5mm[
            "ground_truth_centerline_mask"
        ],
        "metrics": metrics,
        "rasterization": rasterization,
        "rasterization_0p5mm": rasterization_0p5mm,
    }

def compute_optimization_inspection_mask_metrics(
    *,
    vessel_absolute_mm: np.ndarray,
    branch_exists: np.ndarray,
    ground_truth: GroundTruthVolume,
) -> dict[str, float]:
    """Compute only the 0.5-mm overlap metrics sampled during optimization."""

    isotropic_shape, isotropic_affine = isotropic_grid_from_source(
        source_shape=ground_truth.mask.shape,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
    )
    ground_truth_mask = resample_binary_volume_nearest_affine(
        ground_truth.mask,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
        output_shape=isotropic_shape,
        output_index_to_world_affine=isotropic_affine,
    )
    if not bool(ground_truth_mask.any()):
        raise ValueError(
            "Optimization-inspection 0.5 mm resampling removed all "
            f"ground-truth foreground voxels from {ground_truth.source_path}"
        )
    predicted_mask, _ = rasterize_vessel_to_mask(
        vessel_absolute_mm,
        branch_exists,
        source_volume_shape=ground_truth.mask.shape,
        source_index_to_world_affine=ground_truth.index_to_world_affine,
        output_shape=isotropic_shape,
        output_index_to_world_affine=isotropic_affine,
    )
    cldice = cldice_metrics_3d(predicted_mask, ground_truth_mask)
    return {
        "paper_mask_dice_3d_0p5mm": dice_similarity_3d(
            predicted_mask,
            ground_truth_mask,
        ),
        "paper_mask_cldice_3d_0p5mm": float(cldice["cldice_3d"]),
    }

def save_paper_mask_artifact(
    path: Path,
    *,
    result: Mapping[str, Any],
    ground_truth: GroundTruthVolume,
    case_id: str,
    split: str,
    prediction_role: str,
    applied_centering_offset_mm: np.ndarray,
    centering_offset_reversed: bool,
    centering_source: str,
    stored_projection_center_offset_mm: np.ndarray,
    stored_projection_center_offset_valid: bool,
    evaluation_num_views: int | None = None,
    side_branch_snapping_applied: bool = False,
    side_branch_translation_mm: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = dict(result["metrics"])
    target_affine = np.asarray(
        result["rasterization"]["target_index_to_world_affine"],
        dtype=np.float64,
    )
    target_affine_0p5mm = np.asarray(
        result["rasterization_0p5mm"]["target_index_to_world_affine"],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        predicted_mask=np.asarray(result["predicted_mask"], dtype=np.uint8),
        ground_truth_mask=np.asarray(
            result["ground_truth_mask"], dtype=np.uint8
        ),
        predicted_centerline_mask=np.asarray(
            result["predicted_centerline_mask"], dtype=np.uint8
        ),
        ground_truth_centerline_mask=np.asarray(
            result["ground_truth_centerline_mask"], dtype=np.uint8
        ),
        predicted_mask_0p5mm=np.asarray(
            result["predicted_mask_0p5mm"], dtype=np.uint8
        ),
        ground_truth_mask_0p5mm=np.asarray(
            result["ground_truth_mask_0p5mm"], dtype=np.uint8
        ),
        predicted_centerline_mask_0p5mm=np.asarray(
            result["predicted_centerline_mask_0p5mm"], dtype=np.uint8
        ),
        ground_truth_centerline_mask_0p5mm=np.asarray(
            result["ground_truth_centerline_mask_0p5mm"], dtype=np.uint8
        ),
        case_id=np.asarray(str(case_id)),
        split=np.asarray(str(split)),
        prediction_role=np.asarray(str(prediction_role)),
        evaluation_num_views=np.asarray(
            -1 if evaluation_num_views is None else int(evaluation_num_views),
            dtype=np.int64,
        ),
        source_ground_truth_path=np.asarray(str(ground_truth.source_path)),
        source_nii=np.asarray("" if ground_truth.source_nii is None else ground_truth.source_nii),
        ground_truth_array_axis_order=np.asarray("XYZ"),
        ground_truth_assumptions=np.asarray(
            ground_truth.assumptions, dtype=np.str_
        ),
        source_volume_shape=np.asarray(ground_truth.mask.shape, dtype=np.int32),
        source_spacing_mm=np.asarray(ground_truth.spacing_mm, dtype=np.float32),
        source_index_to_world_affine=np.asarray(
            ground_truth.index_to_world_affine, dtype=np.float64
        ),
        target_voxel_shape=np.asarray(PAPER_METRIC_VOXEL_SHAPE, dtype=np.int32),
        target_index_to_world_affine=target_affine,
        target_spacing_mm=np.asarray(
            result["rasterization"]["target_spacing_mm"], dtype=np.float64
        ),
        target_origin_mm=np.asarray(
            result["rasterization"]["target_origin_mm"], dtype=np.float64
        ),
        target_voxel_shape_0p5mm=np.asarray(
            result["predicted_mask_0p5mm"].shape, dtype=np.int32
        ),
        target_index_to_world_affine_0p5mm=target_affine_0p5mm,
        target_spacing_mm_0p5mm=np.asarray(
            result["rasterization_0p5mm"]["target_spacing_mm"],
            dtype=np.float64,
        ),
        target_origin_mm_0p5mm=np.asarray(
            result["rasterization_0p5mm"]["target_origin_mm"],
            dtype=np.float64,
        ),
        applied_centering_offset_mm=np.asarray(
            applied_centering_offset_mm, dtype=np.float32
        ).reshape(3),
        centering_offset_reversed=np.asarray(
            bool(centering_offset_reversed), dtype=np.bool_
        ),
        centering_source=np.asarray(str(centering_source)),
        stored_projection_center_offset_mm=np.asarray(
            stored_projection_center_offset_mm, dtype=np.float32
        ).reshape(3),
        stored_projection_center_offset_valid=np.asarray(
            bool(stored_projection_center_offset_valid), dtype=np.bool_
        ),
        side_branch_snapping_applied=np.asarray(
            bool(side_branch_snapping_applied), dtype=np.bool_
        ),
        side_branch_translation_mm=(
            np.empty((0, 3), dtype=np.float32)
            if side_branch_translation_mm is None
            else np.asarray(side_branch_translation_mm, dtype=np.float32)
        ),
        paper_mask_dice_3d=np.asarray(metrics["paper_mask_dice_3d"], dtype=np.float64),
        paper_mask_ssim_3d=np.asarray(metrics["paper_mask_ssim_3d"], dtype=np.float64),
        paper_mask_cldice_3d=np.asarray(
            metrics["paper_mask_cldice_3d"], dtype=np.float64
        ),
        paper_mask_cldice_loss_3d=np.asarray(
            metrics["paper_mask_cldice_loss_3d"], dtype=np.float64
        ),
        paper_mask_cldice_topology_precision=np.asarray(
            metrics["paper_mask_cldice_topology_precision"],
            dtype=np.float64,
        ),
        paper_mask_cldice_topology_sensitivity=np.asarray(
            metrics["paper_mask_cldice_topology_sensitivity"],
            dtype=np.float64,
        ),
        paper_mask_dice_3d_0p5mm=np.asarray(
            metrics["paper_mask_dice_3d_0p5mm"], dtype=np.float64
        ),
        paper_mask_cldice_3d_0p5mm=np.asarray(
            metrics["paper_mask_cldice_3d_0p5mm"], dtype=np.float64
        ),
        paper_mask_cldice_loss_3d_0p5mm=np.asarray(
            metrics["paper_mask_cldice_loss_3d_0p5mm"], dtype=np.float64
        ),
        paper_mask_cldice_topology_precision_0p5mm=np.asarray(
            metrics["paper_mask_cldice_topology_precision_0p5mm"],
            dtype=np.float64,
        ),
        paper_mask_cldice_topology_sensitivity_0p5mm=np.asarray(
            metrics["paper_mask_cldice_topology_sensitivity_0p5mm"],
            dtype=np.float64,
        ),
        paper_mask_predicted_centerline_voxels=np.asarray(
            metrics["paper_mask_predicted_centerline_voxels"],
            dtype=np.int64,
        ),
        paper_mask_ground_truth_centerline_voxels=np.asarray(
            metrics["paper_mask_ground_truth_centerline_voxels"],
            dtype=np.int64,
        ),
        paper_mask_predicted_centerline_in_ground_truth_voxels=np.asarray(
            metrics[
                "paper_mask_predicted_centerline_in_ground_truth_voxels"
            ],
            dtype=np.int64,
        ),
        paper_mask_ground_truth_centerline_in_prediction_voxels=np.asarray(
            metrics[
                "paper_mask_ground_truth_centerline_in_prediction_voxels"
            ],
            dtype=np.int64,
        ),
        paper_mask_predicted_centerline_voxels_0p5mm=np.asarray(
            metrics["paper_mask_predicted_centerline_voxels_0p5mm"],
            dtype=np.int64,
        ),
        paper_mask_ground_truth_centerline_voxels_0p5mm=np.asarray(
            metrics["paper_mask_ground_truth_centerline_voxels_0p5mm"],
            dtype=np.int64,
        ),
        paper_mask_predicted_centerline_in_ground_truth_voxels_0p5mm=np.asarray(
            metrics[
                "paper_mask_predicted_centerline_in_ground_truth_voxels_0p5mm"
            ],
            dtype=np.int64,
        ),
        paper_mask_ground_truth_centerline_in_prediction_voxels_0p5mm=np.asarray(
            metrics[
                "paper_mask_ground_truth_centerline_in_prediction_voxels_0p5mm"
            ],
            dtype=np.int64,
        ),
        paper_mask_ground_truth_native_foreground_voxels=np.asarray(
            metrics["paper_mask_ground_truth_native_foreground_voxels"],
            dtype=np.int64,
        ),
        **{
            key: np.asarray(metrics[key], dtype=np.int64)
            for key in (
                "paper_mask_predicted_connected_components_3d",
                "paper_mask_ground_truth_connected_components_3d",
                "paper_mask_connected_component_count_absolute_error_3d",
                "paper_mask_predicted_disconnected_3d",
                "paper_mask_ground_truth_disconnected_3d",
                "paper_mask_predicted_connected_components_3d_0p5mm",
                "paper_mask_ground_truth_connected_components_3d_0p5mm",
                "paper_mask_connected_component_count_absolute_error_3d_0p5mm",
                "paper_mask_predicted_disconnected_3d_0p5mm",
                "paper_mask_ground_truth_disconnected_3d_0p5mm",
            )
        },
        **(
            {
                "centerline_chamfer_distance_mm": np.asarray(
                    metrics["centerline_chamfer_distance_mm"],
                    dtype=np.float64,
                )
            }
            if "centerline_chamfer_distance_mm" in metrics
            else {}
        ),
    )
