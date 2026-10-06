"""Load one annotated coronary CT component and render seven calibrated masks.

All world coordinates are millimetres in a declared LAS basis. A volume alone
is sufficient for rendering and overlap metrics, but is not a branch annotation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from vessel_code.geometry.projection import render_mesh_views, view_features_from_angles

# Seven projector theta/phi pairs for each artery.
VIEW_ANGLES = {
    'rca': ([20, -60, -40, 75, 0, -20, 75], [70, 90, 80, 80, 65, 90, 110]),
    'lca': ([115, 50, 85, 90, 130, 70, 85], [125, 70, 120, 100, 65, 60, 50]),
}

@dataclass
class AnnotatedCT:
    path: Path
    mask: np.ndarray
    affine_mm: np.ndarray
    artery_type: str
    projection_center_mm: np.ndarray | None
    vessel_code_mm: np.ndarray | None
    theta_deg: np.ndarray
    phi_deg: np.ndarray


def load_annotated_ct(path, artery_type):
    path = Path(path)
    volume_key = 'vol'
    if not isinstance(artery_type, str):
        raise ValueError('Specify artery_type as rca or lca')
    artery_type = artery_type.lower()
    if artery_type not in VIEW_ANGLES:
        raise ValueError('artery_type must be rca or lca')
    with np.load(path, allow_pickle=False) as z:
        if volume_key not in z:
            raise ValueError(f'{path}: missing binary annotation key {volume_key!r}')
        raw = np.asarray(z[volume_key])
        if raw.ndim != 3 or min(raw.shape) < 2 or not np.isfinite(raw).all():
            raise ValueError('vol must be a finite 3D binary annotation with dimensions >= 2')
        if not np.isin(raw, [0, 1]).all():
            raise ValueError('vol must contain 0/1 for one artery component, not CT intensities or multiple labels')
        mask = raw.astype(bool)
        if not mask.any():
            raise ValueError('The annotation is empty')
        frame = str(np.asarray(z['coordinate_frame']).item()).upper() if 'coordinate_frame' in z else None
        if frame is not None and frame != 'LAS':
            raise ValueError('Convert the affine and vessel coordinates to LAS before loading')
        if frame is None:
            raise ValueError("coordinate_frame=LAS is required; reorient to the documented patient frame before loading")
        if 'artery_type' in z and str(np.asarray(z['artery_type']).item()).lower() != artery_type:
            raise ValueError('NPZ artery_type conflicts with the requested anatomy')
        if 'spacing' not in z:
            raise ValueError('spacing is required in the canonical NPZ format')
        spacing = np.asarray(z['spacing'], dtype=float) if 'spacing' in z else None
        if spacing is not None and (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
            raise ValueError('spacing must contain three finite positive millimetre values')
        affine_key = next((k for k in ['index_to_world_affine', 'affine'] if k in z), None)
        if affine_key:
            affine = np.asarray(z[affine_key], dtype=float)
        else:
            if spacing is None:
                raise ValueError('spacing is required when no affine is provided')
            origin = np.asarray(z['origin_mm'], dtype=float) if 'origin_mm' in z else np.zeros(3)
            direction = np.asarray(z['direction'], dtype=float) if 'direction' in z else np.eye(3)
            if origin.shape != (3,) or direction.shape != (3, 3):
                raise ValueError('origin_mm must be [3] and direction must be [3,3]')
            affine = np.eye(4)
            affine[:3, :3] = direction @ np.diag(spacing)
            affine[:3, 3] = origin
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0,0,0,1]) or abs(np.linalg.det(affine[:3,:3])) < 1e-12:
            raise ValueError('Invalid index-to-world affine')
        axis_spacing = np.linalg.norm(affine[:3,:3], axis=0)
        direction = affine[:3,:3] / axis_spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError('Sheared affines are unsupported; resample the annotation first')
        if spacing is not None and not np.allclose(spacing, axis_spacing, rtol=1e-4):
            raise ValueError('spacing disagrees with the affine axis lengths')
        center = np.asarray(z['projection_center_offset_mm'], dtype=float) if 'projection_center_offset_mm' in z else None
        if center is None:
            raise ValueError('projection_center_offset_mm is required; use the same isocentre as the training convention')
        if center is not None and (center.shape != (3,) or not np.isfinite(center).all()):
            raise ValueError('projection_center_offset_mm must be a finite [3] vector')
        vessel = np.asarray(z['vessel_code_mm'], dtype=float) if 'vessel_code_mm' in z else None
        if vessel is not None:
            if vessel.ndim != 3 or vessel.shape[-1] != 4 or not np.isfinite(vessel).all() or np.any(vessel[...,3] < 0):
                raise ValueError('vessel_code_mm must be finite [M,N,4] with nonnegative radii')
            active = np.any(vessel[...,3] > 0, axis=1)
            if not active.any() or np.any(vessel[active,:,3] <= 0):
                raise ValueError('Every annotated branch must have positive radius at all points')
        if ('theta_deg' in z) != ('phi_deg' in z):
            raise ValueError('theta_deg and phi_deg must be provided together')
        theta = np.asarray(z['theta_deg'] if 'theta_deg' in z else VIEW_ANGLES[artery_type][0], dtype=np.float32)
        phi = np.asarray(z['phi_deg'] if 'phi_deg' in z else VIEW_ANGLES[artery_type][1], dtype=np.float32)
        if theta.shape != (7,) or phi.shape != (7,) or not np.isfinite(theta).all() or not np.isfinite(phi).all():
            raise ValueError('Seven finite theta_deg/phi_deg pairs are required')
    return AnnotatedCT(path, mask, affine, artery_type, center, vessel, theta, phi)


def render_ct_case(case):
    from skimage.measure import marching_cubes
    # Pad to close surfaces touching the source boundary; remove that index shift.
    vertices, faces, _, _ = marching_cubes(np.pad(case.mask.astype(np.float32), 1), level=0.5)
    vertices = vertices - 1
    vertices = vertices @ case.affine_mm[:3,:3].T + case.affine_mm[:3,3]
    triangles = vertices[faces]
    center = case.projection_center_mm
    strategy = 'provided_projection_center_offset_mm'
    spacing = 0.55 if case.artery_type == 'rca' else 0.65
    images, _ = render_mesh_views((triangles-center) * 0.001,
        theta_deg=case.theta_deg, phi_deg=case.phi_deg,
        image_dim=256, sid_m=0.9, pixel_spacing_mm=spacing)
    return {
        'images': images.astype(np.float32),
        'theta_deg': case.theta_deg, 'phi_deg': case.phi_deg,
        'view_features': view_features_from_angles(case.theta_deg, case.phi_deg).astype(np.float32),
        'projection_center_offset': (center*0.001).astype(np.float32),
        'projection_center_offset_mm': center,
        'projection_center_strategy': np.asarray(strategy),
        'artery_type': np.asarray(case.artery_type),
        'coordinate_frame': np.asarray('LAS'),
        'image_size': np.asarray(256), 'sid_m': np.asarray(0.9),
        'source_to_iso_m': np.asarray(0.75), 'pixel_spacing_mm': np.asarray(spacing),
        'render_protocol': np.asarray('ct_marching_cubes_surface_stage2_camera'),
    }


class CTProjectionDataset:
    """Lazy sequence of CT annotations and their seven projection masks."""
    def __init__(self, files, artery_type):
        self.files = [Path(p) for p in files]
        self.artery_type = artery_type
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        case = load_annotated_ct(self.files[index], self.artery_type)
        return case, render_ct_case(case)
