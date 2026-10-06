# Transferred from methods/src/visualization.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
import imageio.v2 as imageio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from vessel_code.geometry.surface_rendering import _render_projection_from_surface_rings, build_surface_coords
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector
from vessel_code.shared.loss import _project_points_uncropped, _projector_main_surface_center

matplotlib.use("Agg")

DATA_PROJECTION_NUM_CIRCLE_POINTS = 120

def _image_views_to_numpy(images: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image views [V,H,W] or [V,C,H,W], got {arr.shape}")
    return np.clip(arr, 0.0, 1.0)

def _valid_branch_indices(target_exist: np.ndarray, num_branches: int) -> list[int]:
    exist = np.asarray(target_exist, dtype=np.float32).reshape(-1)
    out = []
    for branch_idx in range(int(num_branches)):
        if branch_idx == 0 or (branch_idx < exist.shape[0] and float(exist[branch_idx]) > 0.5):
            out.append(branch_idx)
    return out

def save_centerline_xyz_error_profile(
    *,
    target_vessel: np.ndarray | torch.Tensor,
    predicted_vessel: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    out_path: str | Path,
    point_valid: np.ndarray | torch.Tensor | None = None,
    title: str = "Dense centreline XYZ error by ordered point index",
) -> np.ndarray:
    """Plot Euclidean XYZ error along each active dense centreline branch."""

    target = (
        target_vessel.detach().cpu().numpy()
        if isinstance(target_vessel, torch.Tensor)
        else np.asarray(target_vessel)
    )
    predicted = (
        predicted_vessel.detach().cpu().numpy()
        if isinstance(predicted_vessel, torch.Tensor)
        else np.asarray(predicted_vessel)
    )
    if target.shape != predicted.shape or target.ndim != 3:
        raise ValueError(
            "target_vessel and predicted_vessel must have matching [B,N,C] "
            f"shapes, got {target.shape} and {predicted.shape}"
        )
    if target.shape[-1] < 3:
        raise ValueError(
            f"Vessel tensors must contain XYZ coordinates, got shape {target.shape}"
        )

    errors = np.linalg.norm(
        predicted[..., :3] - target[..., :3],
        axis=-1,
    ).astype(np.float32, copy=False)
    finite = np.isfinite(target[..., :3]).all(axis=-1) & np.isfinite(
        predicted[..., :3]
    ).all(axis=-1)
    if point_valid is None:
        valid = finite
    else:
        valid_raw = (
            point_valid.detach().cpu().numpy()
            if isinstance(point_valid, torch.Tensor)
            else np.asarray(point_valid)
        )
        if valid_raw.shape != errors.shape:
            raise ValueError(
                f"point_valid must have shape {errors.shape}, got {valid_raw.shape}"
            )
        valid = np.asarray(valid_raw, dtype=bool) & finite

    exist_values = (
        target_exist.detach().cpu().numpy()
        if isinstance(target_exist, torch.Tensor)
        else np.asarray(target_exist)
    )
    active = _valid_branch_indices(exist_values, target.shape[0])
    if not active:
        return errors
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(active),
        1,
        figsize=(12.0, 3.4 * len(active)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    indices = np.arange(errors.shape[1])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(target.shape[0], 1)))
    for row, branch_index in enumerate(active):
        axis = axes[row, 0]
        branch_values = np.where(valid[branch_index], errors[branch_index], np.nan)
        valid_values = branch_values[np.isfinite(branch_values)]
        color = colors[branch_index]
        axis.plot(
            indices,
            branch_values,
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.0,
        )
        axis.fill_between(
            indices,
            0.0,
            branch_values,
            color=color,
            alpha=0.10,
        )
        if valid_values.size:
            mean_error = float(valid_values.mean())
            axis.axhline(
                mean_error,
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label=f"mean {mean_error:.2f} mm",
            )
            axis.legend(loc="upper right", fontsize=8)
        axis.set_title(f"Branch {branch_index}")
        axis.set_ylabel("Euclidean XYZ error (mm)")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Ordered centreline point index")
    axes[-1, 0].set_xlim(0, max(errors.shape[1] - 1, 1))
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return errors

def save_radius_prediction_profiles(
    *,
    target_vessel: np.ndarray | torch.Tensor,
    predicted_vessel: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    profile_out_path: str | Path,
    error_out_path: str | Path,
    point_valid: np.ndarray | torch.Tensor | None = None,
    target_label: str = "Ground truth",
    profile_title: str = "Ground-truth and predicted radius by ordered point index",
    error_title: str = "Absolute radius prediction error by ordered point index",
) -> np.ndarray:
    """Plot paired radii and absolute radius error for each active branch."""

    def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    target = as_numpy(target_vessel)
    predicted = as_numpy(predicted_vessel)
    if target.shape != predicted.shape or target.ndim != 3:
        raise ValueError(
            "target_vessel and predicted_vessel must have matching [B,N,C] "
            f"shapes, got {target.shape} and {predicted.shape}"
        )
    if target.shape[-1] < 4:
        raise ValueError(
            "Vessel tensors must contain XYZ and radius channels, "
            f"got shape {target.shape}"
        )

    target_radius = target[..., 3].astype(np.float32, copy=False)
    predicted_radius = predicted[..., 3].astype(np.float32, copy=False)
    errors = np.abs(predicted_radius - target_radius).astype(
        np.float32, copy=False
    )
    finite = np.isfinite(target_radius) & np.isfinite(predicted_radius)
    if point_valid is None:
        valid = finite
    else:
        valid_raw = as_numpy(point_valid)
        if valid_raw.shape != errors.shape:
            raise ValueError(
                f"point_valid must have shape {errors.shape}, got {valid_raw.shape}"
            )
        valid = np.asarray(valid_raw, dtype=bool) & finite

    active = _valid_branch_indices(as_numpy(target_exist), target.shape[0])
    if not active:
        return errors

    profile_path = Path(profile_out_path)
    error_path = Path(error_out_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(errors.shape[1])

    profile_figure, profile_axes = plt.subplots(
        len(active),
        1,
        figsize=(12.0, 3.6 * len(active)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    for row, branch_index in enumerate(active):
        axis = profile_axes[row, 0]
        gt_values = np.where(valid[branch_index], target_radius[branch_index], np.nan)
        pred_values = np.where(
            valid[branch_index], predicted_radius[branch_index], np.nan
        )
        branch_errors = errors[branch_index, valid[branch_index]]
        mae = float(branch_errors.mean()) if branch_errors.size else float("nan")
        axis.plot(
            indices,
            gt_values,
            color="#2ca02c",
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            markeredgewidth=0.0,
            label=target_label,
        )
        axis.plot(
            indices,
            pred_values,
            color="#d62728",
            linewidth=1.5,
            linestyle="--",
            label=f"Prediction (MAE {mae:.3f} mm)",
        )
        visible = np.concatenate(
            (gt_values[np.isfinite(gt_values)], pred_values[np.isfinite(pred_values)])
        )
        if visible.size and float(visible.min()) >= 0.0:
            axis.set_ylim(bottom=0.0)
        axis.set_title(f"Branch {branch_index}")
        axis.set_ylabel("Radius (mm)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    profile_axes[-1, 0].set_xlabel("Ordered centreline point index")
    profile_axes[-1, 0].set_xlim(0, max(errors.shape[1] - 1, 1))
    profile_figure.suptitle(profile_title)
    profile_figure.tight_layout()
    profile_figure.savefig(profile_path, bbox_inches="tight")
    plt.close(profile_figure)

    error_figure, error_axes = plt.subplots(
        len(active),
        1,
        figsize=(12.0, 3.4 * len(active)),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(target.shape[0], 1)))
    for row, branch_index in enumerate(active):
        axis = error_axes[row, 0]
        values = np.where(valid[branch_index], errors[branch_index], np.nan)
        valid_values = values[np.isfinite(values)]
        color = colors[branch_index]
        axis.plot(
            indices,
            values,
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.0,
        )
        axis.fill_between(indices, 0.0, values, color=color, alpha=0.10)
        if valid_values.size:
            mean_error = float(valid_values.mean())
            axis.axhline(
                mean_error,
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label=f"mean {mean_error:.3f} mm",
            )
            axis.legend(loc="upper right", fontsize=8)
        axis.set_title(f"Branch {branch_index}")
        axis.set_ylabel("Absolute radius error (mm)")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    error_axes[-1, 0].set_xlabel("Ordered centreline point index")
    error_axes[-1, 0].set_xlim(0, max(errors.shape[1] - 1, 1))
    error_figure.suptitle(error_title)
    error_figure.tight_layout()
    error_figure.savefig(error_path, bbox_inches="tight")
    plt.close(error_figure)
    return errors

def _set_equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = float(max(0.5 * np.max(maxs - mins), 1e-3))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

def _sanitize_vessel_for_surface(vessel: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vessel, dtype=np.float32).copy()
    fallback_arr = None if fallback is None else np.asarray(fallback, dtype=np.float32)
    if fallback_arr is not None and fallback_arr.shape == arr.shape:
        arr = np.where(np.isfinite(arr), arr, fallback_arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    radius = arr[..., 3]
    if fallback_arr is not None and fallback_arr.shape == arr.shape:
        fallback_radius = np.asarray(fallback_arr[..., 3], dtype=np.float32)
        radius = np.where(np.isfinite(radius) & (radius > 0.0), radius, fallback_radius)
    positive = radius[np.isfinite(radius) & (radius > 0.0)]
    replacement = float(np.median(positive)) if positive.size else 1e-3
    arr[..., 3] = np.where(np.isfinite(radius) & (radius > 0.0), radius, replacement)
    return arr.astype(np.float32)

def _surface_coords_for_valid_branches(
    vessel: np.ndarray,
    valid_branches: list[int],
    num_circle_points: int = 24,
) -> list[np.ndarray]:
    surfaces: list[np.ndarray] = []
    for branch_idx in valid_branches:
        if branch_idx < 0 or branch_idx >= int(vessel.shape[0]):
            continue
        branch = np.asarray(vessel[branch_idx : branch_idx + 1], dtype=np.float32).copy()
        branch[..., 3] = np.clip(branch[..., 3], 1e-5, None)
        surfaces.extend(build_surface_coords(branch, num_circle_points=int(num_circle_points)))
    return surfaces

def _center_surface_overlay(
    gt_surfs_raw: list[np.ndarray],
    pred_surfs_raw: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if not gt_surfs_raw and not pred_surfs_raw:
        return [], [], np.zeros((0, 3), dtype=np.float32)
    ref_surfs = gt_surfs_raw if gt_surfs_raw else pred_surfs_raw
    ref_points = np.concatenate([surf.reshape(-1, 3) for surf in ref_surfs], axis=0)
    ref_points = ref_points[np.all(np.isfinite(ref_points), axis=1)]
    center = ref_points.mean(axis=0) if ref_points.size else np.zeros((3,), dtype=np.float32)
    gt_surfs = [np.asarray(surf, dtype=np.float32) - center for surf in gt_surfs_raw]
    pred_surfs = [np.asarray(surf, dtype=np.float32) - center for surf in pred_surfs_raw]
    all_surfs = gt_surfs + pred_surfs
    points = (
        np.concatenate([surf.reshape(-1, 3) for surf in all_surfs], axis=0)
        if all_surfs
        else np.zeros((0, 3), dtype=np.float32)
    )
    return gt_surfs, pred_surfs, points

def _plot_surface_collection(
    ax: Any,
    surfaces: list[np.ndarray],
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    for surf in surfaces:
        ax.plot_surface(
            surf[:, :, 0],
            surf[:, :, 1],
            surf[:, :, 2],
            color=color,
            alpha=float(alpha),
            shade=True,
            linewidth=0,
            antialiased=False,
            rcount=surf.shape[0],
            ccount=surf.shape[1],
            zorder=zorder,
        )

def _surface_coords_and_radii_for_valid_branches(
    vessel: np.ndarray,
    valid_branches: list[int],
    point_valid: np.ndarray | None = None,
    num_circle_points: int = 24,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build branch surfaces and their per-ring radii."""

    surfaces: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    valid_array = None if point_valid is None else np.asarray(point_valid, dtype=bool)
    for branch_idx in valid_branches:
        if branch_idx < 0 or branch_idx >= int(vessel.shape[0]):
            continue
        branch = np.asarray(vessel[branch_idx], dtype=np.float32).copy()
        keep = np.isfinite(branch[:, :3]).all(axis=-1) & np.isfinite(branch[:, 3])
        if valid_array is not None:
            keep &= valid_array[branch_idx]
        branch = branch[keep]
        if branch.shape[0] < 2:
            continue
        branch[:, 3] = np.clip(branch[:, 3], 1e-5, None)
        derivatives = np.zeros_like(branch[:, :3])
        derivatives[0] = branch[1, :3] - branch[0, :3]
        derivatives[-1] = branch[-1, :3] - branch[-2, :3]
        if branch.shape[0] > 2:
            derivatives[1:-1] = 0.5 * (
                branch[2:, :3] - branch[:-2, :3]
            )
        surface_point_indices = np.flatnonzero(
            np.sum(np.abs(derivatives), axis=-1) != 0.0
        )
        branch_surfaces = build_surface_coords(
            branch[None, ...],
            num_circle_points=int(num_circle_points),
        )
        for surface in branch_surfaces:
            centerline_radii = branch[surface_point_indices, 3]
            # ``get_vessel_surface`` expands each endpoint into the same number
            # of concentric rings to close the tube. Those extra rows all belong
            # to the endpoint centreline sample; interpolating the centreline
            # radius over them incorrectly paints interior values onto the caps.
            extra_surface_rows = int(surface.shape[0] - centerline_radii.shape[0])
            if extra_surface_rows < 0 or extra_surface_rows % 2 != 0:
                raise ValueError(
                    "Unexpected vessel-surface/end-cap layout: "
                    f"{surface.shape[0]} surface rings for "
                    f"{centerline_radii.shape[0]} centreline radii."
                )
            rings_per_cap = extra_surface_rows // 2 + 1
            ring_radii = np.concatenate(
                [
                    np.full(
                        rings_per_cap,
                        centerline_radii[0],
                        dtype=np.float32,
                    ),
                    centerline_radii[1:-1].astype(np.float32, copy=False),
                    np.full(
                        rings_per_cap,
                        centerline_radii[-1],
                        dtype=np.float32,
                    ),
                ]
            )
            if ring_radii.shape[0] != surface.shape[0]:
                raise ValueError(
                    "Radius colours do not align with vessel-surface rings: "
                    f"{ring_radii.shape[0]} colours for {surface.shape[0]} rings."
                )
            surfaces.append(np.asarray(surface, dtype=np.float32))
            radii.append(np.asarray(ring_radii, dtype=np.float32))
    return surfaces, radii

def _plot_radius_colored_surface_collection(
    ax: Any,
    surfaces: list[np.ndarray],
    radii: list[np.ndarray],
    *,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> None:
    light_direction = np.asarray([-0.35, -0.45, 0.82], dtype=np.float32)
    light_direction /= np.linalg.norm(light_direction)
    for zorder, (surface, ring_radii) in enumerate(
        zip(surfaces, radii),
        start=1,
    ):
        # ``plot_surface`` colours faces rather than centreline vertices. Use
        # the mean radius of the two rings bounding each surface strip.
        face_radii = 0.5 * (ring_radii[:-1] + ring_radii[1:])
        face_colors = cmap(norm(face_radii))
        facecolors = np.repeat(
            face_colors[:, None, :],
            max(surface.shape[1] - 1, 1),
            axis=1,
        )
        # Add only a small Lambertian brightness variation. Radius continues to
        # determine hue; the 0.86--1.00 multiplier merely makes the tube's
        # curvature legible in a single frame.
        axial_edges = surface[1:, :-1] - surface[:-1, :-1]
        circumferential_edges = surface[:-1, 1:] - surface[:-1, :-1]
        face_normals = np.cross(circumferential_edges, axial_edges)
        normal_lengths = np.linalg.norm(face_normals, axis=-1, keepdims=True)
        unit_normals = np.divide(
            face_normals,
            np.maximum(normal_lengths, 1e-8),
        )
        light_alignment = np.clip(
            np.sum(unit_normals * light_direction, axis=-1),
            -1.0,
            1.0,
        )
        brightness = 0.86 + 0.14 * (0.5 * (light_alignment + 1.0))
        facecolors[..., :3] = np.clip(
            facecolors[..., :3] * brightness[..., None],
            0.0,
            1.0,
        )
        ax.plot_surface(
            surface[:, :, 0],
            surface[:, :, 1],
            surface[:, :, 2],
            facecolors=facecolors,
            alpha=0.98,
            # Shading is already applied above with a deliberately mild range.
            shade=False,
            linewidth=0,
            antialiased=False,
            rcount=surface.shape[0],
            ccount=surface.shape[1],
            zorder=zorder,
        )

def _save_3d_radius_colored_single_surface_gif(
    vessel: np.ndarray | torch.Tensor,
    vessel_exist: np.ndarray | torch.Tensor,
    out_path: str | Path,
    num_frames: int,
    fps: int,
    *,
    point_valid: np.ndarray | torch.Tensor | None = None,
    highlighted_xyz: np.ndarray | torch.Tensor | None = None,
    title: str,
    panel_title: str,
    colormap_name: str,
) -> tuple[float, float]:
    """Save one rotating vessel surface coloured by its per-case radius."""

    def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    vessel_raw = as_numpy(vessel)
    if vessel_raw.ndim != 3 or vessel_raw.shape[-1] < 4:
        raise ValueError(
            "vessel must have shape [B,N,>=4], "
            f"got {vessel_raw.shape}"
        )
    valid_array = None
    if point_valid is not None:
        valid_array = np.asarray(as_numpy(point_valid), dtype=bool)
        if valid_array.shape != vessel_raw.shape[:2]:
            raise ValueError(
                f"point_valid must have shape {vessel_raw.shape[:2]}, "
                f"got {valid_array.shape}"
            )

    sanitized_vessel = _sanitize_vessel_for_surface(vessel_raw)
    valid_branches = _valid_branch_indices(
        as_numpy(vessel_exist),
        int(sanitized_vessel.shape[0]),
    )
    surfaces_raw, ring_radii = _surface_coords_and_radii_for_valid_branches(
        sanitized_vessel,
        valid_branches,
        point_valid=valid_array,
        num_circle_points=24,
    )
    surfaces, _, points_for_axes = _center_surface_overlay(surfaces_raw, [])
    if not surfaces:
        raise ValueError("vessel contains no renderable branches")

    raw_surface_points = np.concatenate(
        [surface.reshape(-1, 3) for surface in surfaces_raw],
        axis=0,
    )
    finite_raw_points = raw_surface_points[
        np.all(np.isfinite(raw_surface_points), axis=-1)
    ]
    surface_center = (
        finite_raw_points.mean(axis=0)
        if finite_raw_points.size
        else np.zeros((3,), dtype=np.float32)
    )

    highlight = np.zeros((0, 3), dtype=np.float32)
    if highlighted_xyz is not None:
        highlight = np.asarray(as_numpy(highlighted_xyz), dtype=np.float32).reshape(-1, 3)
        highlight = highlight[np.all(np.isfinite(highlight), axis=-1)]
        highlight = highlight - surface_center

    radius_parts = [
        np.asarray(values, dtype=np.float32)[np.isfinite(values)]
        for values in ring_radii
    ]
    all_radii = (
        np.concatenate([values for values in radius_parts if values.size])
        if any(values.size for values in radius_parts)
        else np.zeros((0,), dtype=np.float32)
    )
    if all_radii.size:
        radius_min = float(all_radii.min())
        radius_max = float(all_radii.max())
    else:
        radius_min, radius_max = 0.0, 1.0
    if radius_max <= radius_min:
        padding = max(0.05 * abs(radius_min), 1e-3)
        radius_min -= padding
        radius_max += padding

    cmap = LinearSegmentedColormap.from_list(
        colormap_name,
        ["#d73027", "#fee08b", "#1a9850"],
    )
    norm = Normalize(vmin=radius_min, vmax=radius_max, clip=True)
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(all_radii)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_idx in range(frame_count):
        figure = plt.figure(figsize=(7.2, 6.4), dpi=120)
        axis = figure.add_subplot(1, 1, 1, projection="3d")
        _plot_radius_colored_surface_collection(
            axis,
            surfaces,
            ring_radii,
            cmap=cmap,
            norm=norm,
        )
        if highlight.size:
            axis.scatter(
                highlight[:, 0],
                highlight[:, 1],
                highlight[:, 2],
                s=24,
                c="#111111",
                edgecolors="white",
                linewidths=0.7,
                depthshade=False,
            )
        azimuth = 360.0 * float(frame_idx) / float(frame_count)
        elevation = 22.0 + 6.0 * np.sin(
            2.0 * np.pi * float(frame_idx) / float(frame_count)
        )
        _set_equal_3d_axes(axis, points_for_axes)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(panel_title, pad=2.0)
        figure.suptitle(title, y=0.97)
        figure.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.16,
            top=0.90,
        )
        colorbar_axis = figure.add_axes((0.24, 0.07, 0.52, 0.025))
        colorbar = figure.colorbar(
            scalar_mappable,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("Radius (mm): red = smallest, green = largest")
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height,
            width,
            4,
        )[..., :3]
        frames.append(frame.copy())
        plt.close(figure)
    imageio.mimsave(path, frames, fps=max(1, int(fps)), loop=0)
    return radius_min, radius_max

def save_3d_radius_colored_prediction_gif(
    pred_vessel: np.ndarray | torch.Tensor,
    pred_exist: np.ndarray | torch.Tensor,
    out_path: str | Path,
    num_frames: int,
    fps: int,
    *,
    point_valid: np.ndarray | torch.Tensor | None = None,
    title: str = "Predicted radius-coloured artery surface",
) -> tuple[float, float]:
    """Save only the rotating prediction surface coloured by its radius."""

    return _save_3d_radius_colored_single_surface_gif(
        pred_vessel,
        pred_exist,
        out_path,
        num_frames,
        fps,
        point_valid=point_valid,
        title=title,
        panel_title="Prediction",
        colormap_name="artery_radius_red_to_green_prediction",
    )

def save_3d_radius_colored_comparison_gif(
    gt_vessel: np.ndarray | torch.Tensor,
    pred_vessel: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    out_path: str | Path,
    num_frames: int,
    fps: int,
    *,
    point_valid: np.ndarray | torch.Tensor | None = None,
    pred_exist: np.ndarray | torch.Tensor | None = None,
    title: str = "Radius-coloured artery surfaces",
) -> tuple[float, float]:
    """Save GT/pred surfaces on a shared per-case radius colour scale.

    The smallest displayed radius is red and the largest is green. Ground truth
    and prediction share the limits so the same colour denotes the same radius.
    """

    def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    gt_raw = as_numpy(gt_vessel)
    pred_raw = as_numpy(pred_vessel)
    if gt_raw.shape != pred_raw.shape or gt_raw.ndim != 3 or gt_raw.shape[-1] < 4:
        raise ValueError(
            "gt_vessel and pred_vessel must have matching [B,N,>=4] shapes, "
            f"got {gt_raw.shape} and {pred_raw.shape}"
        )
    valid_array = None
    if point_valid is not None:
        valid_array = np.asarray(as_numpy(point_valid), dtype=bool)
        if valid_array.shape != gt_raw.shape[:2]:
            raise ValueError(
                f"point_valid must have shape {gt_raw.shape[:2]}, got {valid_array.shape}"
            )

    gt = _sanitize_vessel_for_surface(gt_raw)
    pred = np.asarray(pred_raw, dtype=np.float32).copy()
    pred[..., :3] = np.where(
        np.isfinite(pred[..., :3]),
        pred[..., :3],
        gt[..., :3],
    )
    pred[..., 3] = np.where(np.isfinite(pred[..., 3]), pred[..., 3], 1e-5)
    pred[..., 3] = np.clip(pred[..., 3], 1e-5, None)

    gt_valid_branches = _valid_branch_indices(
        as_numpy(target_exist),
        min(gt.shape[0], pred.shape[0]),
    )
    pred_valid_branches = _valid_branch_indices(
        as_numpy(target_exist if pred_exist is None else pred_exist),
        min(gt.shape[0], pred.shape[0]),
    )
    pred_valid_array = None if valid_array is None else valid_array.copy()
    if pred_valid_array is not None:
        gt_valid_set = set(gt_valid_branches)
        for branch_index in pred_valid_branches:
            if branch_index not in gt_valid_set:
                pred_valid_array[branch_index] = True
    gt_surfs_raw, gt_radii = _surface_coords_and_radii_for_valid_branches(
        gt,
        gt_valid_branches,
        point_valid=valid_array,
        num_circle_points=24,
    )
    pred_surfs_raw, pred_radii = _surface_coords_and_radii_for_valid_branches(
        pred,
        pred_valid_branches,
        point_valid=pred_valid_array,
        num_circle_points=24,
    )
    gt_surfs, pred_surfs, points_for_axes = _center_surface_overlay(
        gt_surfs_raw,
        pred_surfs_raw,
    )

    # Normalize within each GIF, but use one range for both panels so GT and
    # prediction remain directly comparable. The smallest displayed radius is
    # red and the largest displayed radius is green.
    radius_parts = gt_radii + pred_radii
    if radius_parts:
        all_radii = np.concatenate(radius_parts)
        all_radii = all_radii[np.isfinite(all_radii)]
    else:
        all_radii = np.zeros((0,), dtype=np.float32)
    if all_radii.size:
        radius_min = float(all_radii.min())
        radius_max = float(all_radii.max())
    else:
        radius_min, radius_max = 0.0, 1.0
    if radius_max <= radius_min:
        padding = max(0.05 * abs(radius_min), 1e-3)
        radius_min -= padding
        radius_max += padding

    cmap = LinearSegmentedColormap.from_list(
        "artery_radius_red_to_green",
        ["#d73027", "#fee08b", "#1a9850"],
    )
    norm = Normalize(vmin=radius_min, vmax=radius_max, clip=True)
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(all_radii)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_idx in range(frame_count):
        figure = plt.figure(figsize=(10.8, 5.4), dpi=120)
        gt_axis = figure.add_subplot(1, 2, 1, projection="3d")
        pred_axis = figure.add_subplot(1, 2, 2, projection="3d")
        _plot_radius_colored_surface_collection(
            gt_axis,
            gt_surfs,
            gt_radii,
            cmap=cmap,
            norm=norm,
        )
        _plot_radius_colored_surface_collection(
            pred_axis,
            pred_surfs,
            pred_radii,
            cmap=cmap,
            norm=norm,
        )
        azimuth = 360.0 * float(frame_idx) / float(frame_count)
        elevation = 22.0 + 6.0 * np.sin(
            2.0 * np.pi * float(frame_idx) / float(frame_count)
        )
        for axis, panel_title in (
            (gt_axis, "Ground truth"),
            (pred_axis, "Prediction"),
        ):
            _set_equal_3d_axes(axis, points_for_axes)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            axis.set_title(panel_title, pad=2.0)
        figure.suptitle(title, y=0.97)
        figure.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.16,
            top=0.90,
            wspace=0.01,
        )
        colorbar_axis = figure.add_axes((0.27, 0.07, 0.46, 0.025))
        colorbar = figure.colorbar(
            scalar_mappable,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("Radius (mm): red = smallest, green = largest")
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height,
            width,
            4,
        )[..., :3]
        frames.append(frame.copy())
        plt.close(figure)
    imageio.mimsave(path, frames, fps=max(1, int(fps)), loop=0)
    return radius_min, radius_max

def save_3d_overlay_gif(
    gt_vessel: np.ndarray,
    pred_vessel: np.ndarray,
    target_exist: np.ndarray,
    out_path: Path,
    num_frames: int,
    fps: int,
    title: str = "3D overlay",
    pred_exist: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt = _sanitize_vessel_for_surface(gt_vessel)
    pred = _sanitize_vessel_for_surface(pred_vessel, fallback=gt)
    branch_count = min(gt.shape[0], pred.shape[0])
    gt_valid_branches = _valid_branch_indices(target_exist, branch_count)
    pred_valid_branches = _valid_branch_indices(
        target_exist if pred_exist is None else pred_exist,
        branch_count,
    )
    gt_surfs_raw = _surface_coords_for_valid_branches(
        gt,
        gt_valid_branches,
        num_circle_points=24,
    )
    pred_surfs_raw = _surface_coords_for_valid_branches(
        pred,
        pred_valid_branches,
        num_circle_points=24,
    )
    gt_surfs, pred_surfs, points_for_axes = _center_surface_overlay(
        gt_surfs_raw,
        pred_surfs_raw,
    )

    frames = []
    for frame_idx in range(max(1, int(num_frames))):
        fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        _plot_surface_collection(ax, pred_surfs, color="red", alpha=0.5, zorder=1)
        _plot_surface_collection(ax, gt_surfs, color="green", alpha=0.5, zorder=2)
        _set_equal_3d_axes(ax, points_for_axes)
        ax.set_title(title)
        ax.view_init(elev=22.0, azim=360.0 * float(frame_idx) / float(max(1, int(num_frames))))
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=max(1, int(fps)), loop=0)

def _centerline_overlay_lines(
    gt_vessel: np.ndarray,
    pred_vessel: np.ndarray,
    target_exist: np.ndarray,
    point_valid: np.ndarray | None = None,
    pred_exist: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    gt = np.asarray(gt_vessel, dtype=np.float32)
    pred = np.asarray(pred_vessel, dtype=np.float32)
    if gt.ndim != 3 or pred.ndim != 3 or gt.shape[-1] < 3 or pred.shape[-1] < 3:
        raise ValueError(
            "gt_vessel and pred_vessel must have shape [M,N,>=3], got "
            f"{gt.shape} and {pred.shape}."
        )
    branch_count = min(int(gt.shape[0]), int(pred.shape[0]))
    gt_valid_branches = set(_valid_branch_indices(target_exist, branch_count))
    pred_valid_branches = set(
        _valid_branch_indices(
            target_exist if pred_exist is None else pred_exist,
            branch_count,
        )
    )
    valid_array = None if point_valid is None else np.asarray(point_valid, dtype=bool)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    axis_parts: list[np.ndarray] = []
    for branch_index in sorted(gt_valid_branches | pred_valid_branches):
        shared_count = min(int(gt.shape[1]), int(pred.shape[1]))
        shared_valid = np.ones((shared_count,), dtype=bool)
        if (
            valid_array is not None
            and valid_array.ndim == 2
            and branch_index < valid_array.shape[0]
        ):
            valid_count = min(shared_count, int(valid_array.shape[1]))
            shared_valid[valid_count:] = False
            shared_valid[:valid_count] &= valid_array[branch_index, :valid_count]
        gt_line = gt[branch_index, :shared_count, :3]
        pred_line = pred[branch_index, :shared_count, :3]
        gt_point_valid = shared_valid
        pred_point_valid = (
            shared_valid
            if branch_index in gt_valid_branches
            else np.ones((shared_count,), dtype=bool)
        )
        finite_gt = gt_point_valid & np.isfinite(gt_line).all(axis=-1)
        finite_pred = pred_point_valid & np.isfinite(pred_line).all(axis=-1)
        gt_line = (
            gt_line[finite_gt]
            if branch_index in gt_valid_branches
            else np.zeros((0, 3), dtype=np.float32)
        )
        pred_line = (
            pred_line[finite_pred]
            if branch_index in pred_valid_branches
            else np.zeros((0, 3), dtype=np.float32)
        )
        if gt_line.size == 0 and pred_line.size == 0:
            continue
        lines.append((gt_line, pred_line))
        axis_parts.extend((gt_line, pred_line))
    axis_points = (
        np.concatenate(axis_parts, axis=0)
        if axis_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    return lines, axis_points

def _plot_centerline_overlay_3d(
    ax: Any,
    lines: list[tuple[np.ndarray, np.ndarray]],
    axis_points: np.ndarray,
    title: str,
) -> None:
    for gt_line, pred_line in lines:
        if gt_line.size:
            ax.plot(*gt_line.T, color="#18A558", linewidth=2.2)
        if pred_line.size:
            ax.plot(*pred_line.T, color="#E53935", linewidth=2.0, linestyle="--")
    _set_equal_3d_axes(ax, axis_points)
    ax.set_title(title)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#18A558", linewidth=2.2, label="GT centreline"),
            Line2D(
                [0],
                [0],
                color="#E53935",
                linewidth=2.0,
                linestyle="--",
                label="Pred centreline",
            ),
        ],
        loc="upper right",
    )

def save_3d_centerline_overlay_gif(
    gt_vessel: np.ndarray,
    pred_vessel: np.ndarray,
    target_exist: np.ndarray,
    out_path: Path,
    num_frames: int,
    fps: int,
    title: str = "3D centreline overlay",
    point_valid: np.ndarray | None = None,
    pred_exist: np.ndarray | None = None,
) -> None:
    """Save a rotating, combined GT/predicted centreline overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines, axis_points = _centerline_overlay_lines(
        gt_vessel,
        pred_vessel,
        target_exist,
        point_valid,
        pred_exist,
    )
    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_index in range(frame_count):
        fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        _plot_centerline_overlay_3d(ax, lines, axis_points, title)
        angle = 2.0 * np.pi * float(frame_index) / float(frame_count)
        ax.view_init(
            elev=22.0 + 8.0 * np.sin(angle),
            azim=360.0 * float(frame_index) / float(frame_count),
        )
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height,
            width,
            4,
        )[..., :3]
        frames.append(frame.copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=max(1, int(fps)), loop=0)

def _save_3d_single_surface_gif(
    vessel: np.ndarray | torch.Tensor,
    vessel_exist: np.ndarray | torch.Tensor,
    out_path: Path,
    num_frames: int,
    fps: int,
    *,
    color: str,
    title: str,
    point_valid: np.ndarray | torch.Tensor | None = None,
) -> None:
    """Render one rotating artery surface without a comparison panel."""

    def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vessel_raw = as_numpy(vessel)
    if vessel_raw.ndim != 3 or vessel_raw.shape[-1] < 4:
        raise ValueError(
            "vessel must have shape [B,N,>=4], "
            f"got {vessel_raw.shape}"
        )
    valid_array = None
    if point_valid is not None:
        valid_array = np.asarray(as_numpy(point_valid), dtype=bool)
        if valid_array.shape != vessel_raw.shape[:2]:
            raise ValueError(
                f"point_valid must have shape {vessel_raw.shape[:2]}, "
                f"got {valid_array.shape}"
            )
    sanitized_vessel = _sanitize_vessel_for_surface(vessel_raw)
    valid_branches = _valid_branch_indices(
        as_numpy(vessel_exist),
        sanitized_vessel.shape[0],
    )
    surfaces_raw, _ = _surface_coords_and_radii_for_valid_branches(
        sanitized_vessel,
        valid_branches,
        point_valid=valid_array,
        num_circle_points=24,
    )
    surfaces, _, points_for_axes = _center_surface_overlay(surfaces_raw, [])
    if not surfaces:
        raise ValueError("vessel contains no renderable branches")

    frames: list[np.ndarray] = []
    frame_count = max(1, int(num_frames))
    for frame_idx in range(frame_count):
        fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        _plot_surface_collection(
            ax,
            surfaces,
            color=color,
            alpha=0.62,
            zorder=1,
        )
        _set_equal_3d_axes(ax, points_for_axes)
        ax.set_title(title)
        angle = 2.0 * np.pi * float(frame_idx) / float(frame_count)
        ax.view_init(
            elev=22.0 + 6.0 * np.sin(angle),
            azim=360.0 * float(frame_idx) / float(frame_count),
        )
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(
            fig.canvas.buffer_rgba(),
            dtype=np.uint8,
        ).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=max(1, int(fps)), loop=0)

def save_3d_ground_truth_gif(
    gt_vessel: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    out_path: Path,
    num_frames: int,
    fps: int,
    title: str = "Ground-truth 3D artery",
    *,
    point_valid: np.ndarray | torch.Tensor | None = None,
) -> None:
    """Render only the rotating ground-truth artery surface."""

    _save_3d_single_surface_gif(
        gt_vessel,
        target_exist,
        out_path,
        num_frames,
        fps,
        color="#d62728",
        title=title,
        point_valid=point_valid,
    )

def save_3d_prediction_gif(
    pred_vessel: np.ndarray | torch.Tensor,
    pred_exist: np.ndarray | torch.Tensor,
    out_path: Path,
    num_frames: int,
    fps: int,
    title: str = "Predicted 3D artery",
    *,
    point_valid: np.ndarray | torch.Tensor | None = None,
) -> None:
    """Render a rotating prediction without requiring ground truth."""

    _save_3d_single_surface_gif(
        pred_vessel,
        pred_exist,
        out_path,
        num_frames,
        fps,
        color="#d62728",
        title=title,
        point_valid=point_valid,
    )

def _surface_center_from_vessel_m(vessel_m: np.ndarray) -> np.ndarray:
    surfaces = build_surface_coords(np.asarray(vessel_m, dtype=np.float32), num_circle_points=DATA_PROJECTION_NUM_CIRCLE_POINTS)
    if not surfaces:
        return np.zeros((3,), dtype=np.float32)
    points = np.concatenate([surf.reshape(-1, 3) for surf in surfaces], axis=0)
    points = points[np.all(np.isfinite(points), axis=1)]
    return points.mean(axis=0).astype(np.float32) if points.size else np.zeros((3,), dtype=np.float32)

def _render_vessel_masks_for_views(
    vessel_world: np.ndarray,
    reference_world: np.ndarray,
    vessel_exist: np.ndarray | None,
    reference_exist: np.ndarray | None,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    image_size: int,
    sid: float,
    imager_pixel_spacing: float,
    coord_scale_to_meter: float,
    center_offset_m: np.ndarray | None = None,
) -> np.ndarray:
    vessel_m = np.asarray(vessel_world, dtype=np.float32).copy()
    reference_m = np.asarray(reference_world, dtype=np.float32).copy()
    vessel_m[..., :4] *= float(coord_scale_to_meter)
    reference_m[..., :4] *= float(coord_scale_to_meter)

    if vessel_exist is not None:
        valid_branches = _valid_branch_indices(vessel_exist, vessel_m.shape[0])
        vessel_m = vessel_m[valid_branches]
    if reference_exist is not None:
        reference_branches = _valid_branch_indices(
            reference_exist,
            reference_m.shape[0],
        )
        reference_m = reference_m[reference_branches]
    surfaces = build_surface_coords(vessel_m, num_circle_points=DATA_PROJECTION_NUM_CIRCLE_POINTS)
    if not surfaces:
        return np.zeros((0, int(image_size), int(image_size)), dtype=np.float32)
    center = (
        np.asarray(center_offset_m, dtype=np.float32).reshape(3)
        if center_offset_m is not None
        else _surface_center_from_vessel_m(reference_m)
    )
    centered_surfaces = [surf - center[None, None, :] for surf in surfaces]

    masks = []
    theta = np.asarray(theta_deg, dtype=np.float32).reshape(-1)
    phi = np.asarray(phi_deg, dtype=np.float32).reshape(-1)
    for vi in range(min(theta.shape[0], phi.shape[0])):
        mask = _render_projection_from_surface_rings(
            surface_rings=centered_surfaces,
            theta_deg=float(theta[vi]),
            phi_deg=float(phi[vi]),
            image_dim=int(image_size),
            sid=float(sid),
            imager_pixel_spacing=float(imager_pixel_spacing),
        )
        masks.append(np.clip(mask, 0.0, 1.0).astype(np.float32))
    if not masks:
        return np.zeros((0, int(image_size), int(image_size)), dtype=np.float32)
    return np.stack(masks, axis=0)

def save_2d_overlay(
    input_images: np.ndarray | torch.Tensor,
    gt_vessel: np.ndarray,
    pred_vessel: np.ndarray,
    target_exist: np.ndarray,
    theta_deg: np.ndarray | None,
    phi_deg: np.ndarray | None,
    out_path: Path,
    sid: float,
    imager_pixel_spacing: float,
    coord_scale_to_meter: float,
    center_offset_m: np.ndarray | None = None,
    title: str = "2D overlay",
    pred_exist: np.ndarray | None = None,
) -> None:
    if theta_deg is None or phi_deg is None:
        return
    images = _image_views_to_numpy(input_images)
    pred_masks = _render_vessel_masks_for_views(
        vessel_world=pred_vessel,
        reference_world=gt_vessel,
        vessel_exist=pred_exist,
        reference_exist=target_exist,
        theta_deg=np.asarray(theta_deg, dtype=np.float32),
        phi_deg=np.asarray(phi_deg, dtype=np.float32),
        image_size=int(images.shape[-1]),
        sid=float(sid),
        imager_pixel_spacing=float(imager_pixel_spacing),
        coord_scale_to_meter=float(coord_scale_to_meter),
        center_offset_m=center_offset_m,
    )
    num_views = min(int(images.shape[0]), int(pred_masks.shape[0]))
    if num_views <= 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, num_views, figsize=(3.2 * num_views, 3.4), dpi=150, squeeze=False)
    for vi in range(num_views):
        gt_mask = images[vi] > 0.5
        pred_mask = pred_masks[vi] > 0.5
        overlap = np.logical_and(gt_mask, pred_mask)
        pred_only = np.logical_and(pred_mask, np.logical_not(gt_mask))

        rgb = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        rgb[gt_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rgb[overlap] = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        rgb[pred_only] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axes[0, vi].imshow(rgb, vmin=0.0, vmax=1.0)
        axes[0, vi].set_title(f"view {vi}", fontsize=9)
        axes[0, vi].axis("off")
    fig.suptitle(
        f"{title}: white=input mask, green=overlap, red=prediction only",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_2d_centerline_overlay(
    *,
    input_images: np.ndarray | torch.Tensor,
    gt_vessel: np.ndarray | torch.Tensor,
    pred_vessel: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    theta_deg: np.ndarray | torch.Tensor | None,
    phi_deg: np.ndarray | torch.Tensor | None,
    out_path: Path,
    sid: float,
    imager_pixel_spacing: float,
    coord_scale_to_meter: float,
    center_offset_m: np.ndarray | torch.Tensor | None = None,
    point_valid: np.ndarray | torch.Tensor | None = None,
    pred_exist: np.ndarray | torch.Tensor | None = None,
    mask_threshold: float = 0.5,
    title: str = "Predicted 2D centreline overlay",
) -> None:
    """Plot only the projected prediction on input and binarized-mask views."""
    if theta_deg is None or phi_deg is None:
        return
    images = _image_views_to_numpy(input_images)
    theta = torch.as_tensor(theta_deg, dtype=torch.float32).reshape(-1)
    phi = torch.as_tensor(phi_deg, dtype=torch.float32).reshape(-1)
    num_views = min(int(images.shape[0]), int(theta.shape[0]), int(phi.shape[0]))
    if num_views <= 0:
        return

    gt = torch.as_tensor(gt_vessel, dtype=torch.float32)
    pred = torch.as_tensor(pred_vessel, dtype=torch.float32)
    if gt.dim() != 3 or pred.dim() != 3 or gt.shape[-1] < 4 or pred.shape[-1] < 3:
        raise ValueError(
            "gt_vessel and pred_vessel must have shapes [M,N,>=4] and "
            f"[M,N,>=3], got {tuple(gt.shape)} and {tuple(pred.shape)}."
        )
    branch_count = min(int(gt.shape[0]), int(pred.shape[0]))
    valid_branches = _valid_branch_indices(
        torch.as_tensor(
            target_exist if pred_exist is None else pred_exist
        ).detach().cpu().numpy(),
        branch_count,
    )
    target_valid_branches = set(
        _valid_branch_indices(
            torch.as_tensor(target_exist).detach().cpu().numpy(),
            branch_count,
        )
    )
    valid_array = (
        None
        if point_valid is None
        else torch.as_tensor(point_valid, dtype=torch.bool)
    )
    scale = float(coord_scale_to_meter)
    reference_m = gt.clone()
    reference_m[..., :4] *= scale
    projector = DifferentiableVesselProjector(
        image_size=int(images.shape[-1]),
        sid=float(sid),
        source_to_iso=0.75,
        imager_pixel_spacing=float(imager_pixel_spacing),
        num_circle_points=48,
        center_main_branch=False,
        crop_mode="strict",
    )
    if center_offset_m is None:
        center_m = _projector_main_surface_center(projector, reference_m).detach()
    else:
        center_m = torch.as_tensor(center_offset_m, dtype=torch.float32).reshape(3)

    projected_by_view: list[list[np.ndarray]] = []
    for view_index in range(num_views):
        view_lines: list[np.ndarray] = []
        for branch_index in valid_branches:
            points = pred[branch_index, :, :3] * scale - center_m.view(1, 3)
            if (
                branch_index in target_valid_branches
                and valid_array is not None
                and valid_array.dim() == 2
                and branch_index < valid_array.shape[0]
            ):
                shared_count = min(
                    int(points.shape[0]),
                    int(valid_array.shape[1]),
                )
                points = points[:shared_count][
                    valid_array[branch_index, :shared_count]
                ]
            if points.numel() == 0:
                continue
            xy, geometry_valid = _project_points_uncropped(
                projector,
                points,
                theta[view_index],
                phi[view_index],
            )
            xy_np = xy.detach().cpu().numpy().astype(np.float32)
            xy_np[~geometry_valid.detach().cpu().numpy().astype(bool)] = np.nan
            view_lines.append(xy_np)
        projected_by_view.append(view_lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        num_views,
        2,
        figsize=(7.0, max(3.2, 3.2 * num_views)),
        dpi=150,
        squeeze=False,
    )
    for view_index in range(num_views):
        image = images[view_index]
        backgrounds = (
            (image, "Input image"),
            ((image > float(mask_threshold)).astype(np.float32), "Thresholded mask"),
        )
        for column, (background, background_title) in enumerate(backgrounds):
            axis = axes[view_index, column]
            axis.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
            for line in projected_by_view[view_index]:
                axis.plot(
                    line[:, 0],
                    line[:, 1],
                    color="#E53935",
                    linewidth=1.8,
                )
            axis.set_title(f"View {view_index}: {background_title}", fontsize=9)
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(image.shape[0] - 0.5, -0.5)
            axis.set_aspect("equal")
            axis.axis("off")
    fig.suptitle(f"{title}: red=predicted centreline", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _project_main_centerline_xy(
    projector: Any,
    vessel_world: torch.Tensor,
    reference_world: torch.Tensor,
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    coord_scale_to_meter: float,
    center_offset_world: torch.Tensor | None = None,
) -> list[np.ndarray]:
    vessel_m = vessel_world.clone()
    reference_m = reference_world.clone()
    vessel_m[..., :4] = vessel_m[..., :4] * float(coord_scale_to_meter)
    reference_m[..., :4] = reference_m[..., :4] * float(coord_scale_to_meter)
    if center_offset_world is not None:
        center_m = center_offset_world.to(device=vessel_m.device, dtype=vessel_m.dtype) * float(coord_scale_to_meter)
    else:
        center_m = _projector_main_surface_center(projector, reference_m).detach()
    coords = []
    points = vessel_m[0, :, :3] - center_m.view(1, 3)
    for vi in range(int(theta_deg.shape[0])):
        xy, valid = _project_points_uncropped(projector, points, theta_deg[vi], phi_deg[vi])
        xy_np = xy[valid].detach().cpu().numpy().astype(np.float32)
        coords.append(xy_np)
    return coords

def save_projected_control_point_monitor(
    *,
    input_images: np.ndarray | torch.Tensor,
    gt_vessel: np.ndarray | torch.Tensor,
    pred_vessel: np.ndarray | torch.Tensor,
    gt_control_points_global: np.ndarray | torch.Tensor,
    pred_control_points_global: np.ndarray | torch.Tensor,
    target_exist: np.ndarray | torch.Tensor,
    pred_exist: np.ndarray | torch.Tensor | None = None,
    point_valid: np.ndarray | torch.Tensor,
    theta_deg: np.ndarray | torch.Tensor,
    phi_deg: np.ndarray | torch.Tensor,
    projector: Any,
    coord_scale_to_meter: float,
    clean_out_path: Path,
    mask_out_path: Path,
    center_offset_world: np.ndarray | torch.Tensor | None = None,
    title: str = "Projected centrelines and control points",
    point_label: str = "control",
) -> None:
    """Plot projected GT/predicted centrelines and unconnected control points."""
    images = _image_views_to_numpy(input_images)
    gt_vessel_t = torch.as_tensor(gt_vessel, dtype=torch.float32)
    pred_vessel_t = torch.as_tensor(pred_vessel, dtype=gt_vessel_t.dtype)
    gt_controls_t = torch.as_tensor(
        gt_control_points_global, dtype=gt_vessel_t.dtype
    )
    pred_controls_t = torch.as_tensor(
        pred_control_points_global, dtype=gt_vessel_t.dtype
    )
    theta_t = torch.as_tensor(theta_deg, dtype=gt_vessel_t.dtype).reshape(-1)
    phi_t = torch.as_tensor(phi_deg, dtype=gt_vessel_t.dtype).reshape(-1)
    exists_t = torch.as_tensor(target_exist, dtype=torch.bool).reshape(-1)
    pred_exists_t = torch.as_tensor(
        target_exist if pred_exist is None else pred_exist,
        dtype=torch.bool,
    ).reshape(-1)
    valid_t = torch.as_tensor(point_valid, dtype=torch.bool)
    if gt_vessel_t.dim() != 3 or gt_vessel_t.shape[-1] < 4:
        raise ValueError(
            f"gt_vessel must be [M,N,>=4], got {tuple(gt_vessel_t.shape)}"
        )
    if (
        pred_vessel_t.dim() != 3
        or pred_vessel_t.shape[-1] < 4
        or pred_vessel_t.shape[:2] != gt_vessel_t.shape[:2]
    ):
        raise ValueError("GT and predicted vessels must share branch/point dimensions")
    expected_controls_prefix = (gt_vessel_t.shape[0],)
    if (
        gt_controls_t.dim() != 3
        or gt_controls_t.shape[-1] != 3
        or gt_controls_t.shape[:1] != expected_controls_prefix
    ):
        raise ValueError(
            "gt_control_points_global must be [M,K,3], "
            f"got {tuple(gt_controls_t.shape)}"
        )
    if pred_controls_t.shape != gt_controls_t.shape:
        raise ValueError("GT and predicted control tensors must have identical shapes")
    if valid_t.shape != gt_vessel_t.shape[:2]:
        raise ValueError(
            f"point_valid must be {tuple(gt_vessel_t.shape[:2])}, "
            f"got {tuple(valid_t.shape)}"
        )

    num_views = min(int(images.shape[0]), int(theta_t.numel()), int(phi_t.numel()))
    num_branches = min(
        int(gt_vessel_t.shape[0]),
        int(pred_vessel_t.shape[0]),
        int(gt_controls_t.shape[0]),
        int(exists_t.numel()),
        int(pred_exists_t.numel()),
    )
    if num_views <= 0 or num_branches <= 0:
        return

    scale = float(coord_scale_to_meter)
    gt_xyz_m = gt_vessel_t[..., :3] * scale
    pred_xyz_m = pred_vessel_t[..., :3] * scale
    gt_controls_m = gt_controls_t[..., :3] * scale
    pred_controls_m = pred_controls_t[..., :3] * scale
    if center_offset_world is None:
        reference_m = gt_vessel_t.clone()
        reference_m[..., :4] = reference_m[..., :4] * scale
        center_m = _projector_main_surface_center(projector, reference_m).detach()
    else:
        center_m = torch.as_tensor(
            center_offset_world, dtype=gt_vessel_t.dtype
        ).reshape(3) * scale
    gt_xyz_m = gt_xyz_m - center_m.view(1, 1, 3)
    pred_xyz_m = pred_xyz_m - center_m.view(1, 1, 3)
    gt_controls_m = gt_controls_m - center_m.view(1, 1, 3)
    pred_controls_m = pred_controls_m - center_m.view(1, 1, 3)

    def project_branches(
        points_m: torch.Tensor,
        per_point_valid: torch.Tensor | None,
        branch_exists: torch.Tensor,
    ) -> list[list[np.ndarray]]:
        by_view: list[list[np.ndarray]] = []
        for view_index in range(num_views):
            branches: list[np.ndarray] = []
            for branch_index in range(num_branches):
                if not bool(branch_exists[branch_index]):
                    branches.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                branch_points = points_m[branch_index]
                if per_point_valid is not None:
                    branch_points = branch_points[per_point_valid[branch_index]]
                if branch_points.numel() == 0:
                    branches.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                xy, geometry_valid = _project_points_uncropped(
                    projector,
                    branch_points,
                    theta_t[view_index],
                    phi_t[view_index],
                )
                xy_np = xy.detach().cpu().numpy().astype(np.float32)
                geometry_valid_np = geometry_valid.detach().cpu().numpy().astype(bool)
                xy_np[~geometry_valid_np] = np.nan
                branches.append(xy_np)
            by_view.append(branches)
        return by_view

    pred_point_valid = valid_t.clone()
    target_absent_indices = torch.nonzero(
        ~exists_t[:num_branches],
        as_tuple=False,
    ).flatten()
    pred_point_valid[target_absent_indices] = True
    gt_centerlines = project_branches(gt_xyz_m, valid_t, exists_t)
    pred_centerlines = project_branches(
        pred_xyz_m,
        pred_point_valid,
        pred_exists_t,
    )
    gt_controls = project_branches(gt_controls_m, None, exists_t)
    pred_controls = project_branches(pred_controls_m, None, pred_exists_t)
    normalized_point_label = str(point_label).strip().lower() or "control"
    plural_point_label = (
        f"{normalized_point_label}s"
        if not normalized_point_label.endswith("s")
        else normalized_point_label
    )
    legend_handles = [
        Line2D([0], [0], color="#18A558", linewidth=2.0, label="GT centreline"),
        Line2D(
            [0],
            [0],
            color="#E53935",
            linewidth=2.0,
            linestyle="--",
            label="Pred centreline",
        ),
        Line2D(
            [0],
            [0],
            color="#1687D9",
            marker="o",
            linestyle="none",
            markersize=5,
            label=f"GT {plural_point_label}",
        ),
        Line2D(
            [0],
            [0],
            color="#FF9F1C",
            marker="x",
            linestyle="none",
            markersize=6,
            label=f"Pred {plural_point_label}",
        ),
    ]

    def render(out_path: Path, *, show_mask: bool) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1,
            num_views,
            figsize=(4.0 * num_views, 4.3),
            dpi=150,
            squeeze=False,
        )
        for view_index in range(num_views):
            axis = axes[0, view_index]
            image = images[view_index]
            if show_mask:
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axis.set_facecolor("white")
            for branch_index in range(num_branches):
                gt_curve = gt_centerlines[view_index][branch_index]
                pred_curve = pred_centerlines[view_index][branch_index]
                gt_points = gt_controls[view_index][branch_index]
                pred_points = pred_controls[view_index][branch_index]
                if gt_curve.size:
                    line = axis.plot(
                        gt_curve[:, 0],
                        gt_curve[:, 1],
                        color="#18A558",
                        linewidth=1.8,
                        zorder=3,
                    )[0]
                    if show_mask:
                        line.set_path_effects(
                            [
                                path_effects.Stroke(
                                    linewidth=3.2, foreground="black", alpha=0.75
                                ),
                                path_effects.Normal(),
                            ]
                        )
                if pred_curve.size:
                    line = axis.plot(
                        pred_curve[:, 0],
                        pred_curve[:, 1],
                        color="#E53935",
                        linewidth=1.8,
                        linestyle="--",
                        zorder=4,
                    )[0]
                    if show_mask:
                        line.set_path_effects(
                            [
                                path_effects.Stroke(
                                    linewidth=3.2, foreground="black", alpha=0.75
                                ),
                                path_effects.Normal(),
                            ]
                        )
                gt_finite = np.all(np.isfinite(gt_points), axis=-1)
                pred_finite = np.all(np.isfinite(pred_points), axis=-1)
                if bool(gt_finite.any()):
                    axis.scatter(
                        gt_points[gt_finite, 0],
                        gt_points[gt_finite, 1],
                        s=28,
                        marker="o",
                        facecolors="#1687D9",
                        edgecolors="black" if show_mask else "white",
                        linewidths=0.7,
                        zorder=6,
                    )
                if bool(pred_finite.any()):
                    axis.scatter(
                        pred_points[pred_finite, 0],
                        pred_points[pred_finite, 1],
                        s=34,
                        marker="x",
                        c="#FF9F1C",
                        linewidths=1.5,
                        zorder=7,
                    )
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(image.shape[0] - 0.5, -0.5)
            axis.set_aspect("equal")
            axis.set_title(
                f"view {view_index}: theta={float(theta_t[view_index]):.1f}, "
                f"phi={float(phi_t[view_index]):.1f}",
                fontsize=9,
            )
            axis.axis("off")
        background_label = "mask background" if show_mask else "clean background"
        figure.suptitle(f"{title} ({background_label})", fontsize=11)
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
        figure.savefig(out_path, bbox_inches="tight")
        plt.close(figure)

    render(clean_out_path, show_mask=False)
    render(mask_out_path, show_mask=True)

def save_projection_loss_monitor(
    input_images: np.ndarray | torch.Tensor,
    gt_vessel: torch.Tensor,
    pred_vessel: torch.Tensor,
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    projector: Any,
    coord_scale_to_meter: float,
    out_path: Path,
    center_offset_world: torch.Tensor | None = None,
    target_exist: torch.Tensor | np.ndarray | None = None,
    mask_threshold: float = 0.5,
) -> None:
    images = _image_views_to_numpy(input_images)
    theta = theta_deg.detach()
    phi = phi_deg.detach()
    if target_exist is not None:
        active = torch.as_tensor(target_exist, device=gt_vessel.device, dtype=torch.bool)
        gt_vessel = gt_vessel[active]
        pred_vessel = pred_vessel[active]
    gt_np = gt_vessel.detach().cpu().numpy().astype(np.float32)
    pred_np = pred_vessel.detach().cpu().numpy().astype(np.float32)
    center_offset_m_np = (
        None if center_offset_world is None else (center_offset_world.detach().cpu().numpy().astype(np.float32) * float(coord_scale_to_meter))
    )
    pred_masks = _render_vessel_masks_for_views(
        vessel_world=pred_np,
        reference_world=gt_np,
        vessel_exist=None,
        reference_exist=None,
        theta_deg=theta.cpu().numpy(),
        phi_deg=phi.cpu().numpy(),
        image_size=int(images.shape[-1]),
        sid=float(projector.sid),
        imager_pixel_spacing=float(projector.imager_pixel_spacing),
        coord_scale_to_meter=float(coord_scale_to_meter),
        center_offset_m=center_offset_m_np,
    )
    pred_coords = _project_main_centerline_xy(
        projector,
        pred_vessel,
        gt_vessel,
        theta,
        phi,
        coord_scale_to_meter=float(coord_scale_to_meter),
        center_offset_world=center_offset_world,
    )
    gt_coords = _project_main_centerline_xy(
        projector,
        gt_vessel,
        gt_vessel,
        theta,
        phi,
        coord_scale_to_meter=float(coord_scale_to_meter),
        center_offset_world=center_offset_world,
    )

    num_views = min(int(images.shape[0]), int(pred_masks.shape[0]), int(theta.shape[0]))
    if num_views <= 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(num_views, 4, figsize=(14.0, max(2.8, 2.8 * num_views)), dpi=150, squeeze=False)
    titles = ["Input view", "Pred projection", "Overlay", "Centerline XY (no mask)"]
    for ci, title in enumerate(titles):
        axes[0, ci].set_title(title, fontsize=9)
    for vi in range(num_views):
        inp = images[vi]
        gt_mask = inp > float(mask_threshold)
        pred_mask = pred_masks[vi] > 0.5
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        overlay[gt_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        overlay[np.logical_and(gt_mask, pred_mask)] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        overlay[np.logical_and(pred_mask, np.logical_not(gt_mask))] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        axes[vi, 0].imshow(inp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[vi, 1].imshow(pred_masks[vi], cmap="gray", vmin=0.0, vmax=1.0)
        axes[vi, 2].imshow(overlay, vmin=0.0, vmax=1.0)
        axes[vi, 3].set_facecolor("white")
        axes[vi, 3].set_xlim(-0.5, inp.shape[1] - 0.5)
        axes[vi, 3].set_ylim(inp.shape[0] - 0.5, -0.5)
        axes[vi, 3].set_aspect("equal")
        if vi < len(gt_coords) and gt_coords[vi].size:
            axes[vi, 3].scatter(gt_coords[vi][:, 0], gt_coords[vi][:, 1], s=8, c="lime", marker=".", linewidths=0, label="GT")
        if vi < len(pred_coords) and pred_coords[vi].size:
            axes[vi, 3].scatter(pred_coords[vi][:, 0], pred_coords[vi][:, 1], s=8, c="red", marker=".", linewidths=0, label="Pred")
        for ci in range(4):
            axes[vi, ci].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
