# Transferred from methods/eval.py. See TRANSFER_MANIFEST.json.
import numpy as np
import os, sys
from vessel_code.geometry.tube_functions import get_vessel_surface
from vessel_code.geometry.fwd_projection_functions import ray_image_intersection, get_local_params, rotate_volume, convert3D_to_pixels
from skimage import morphology as morph
from skimage import filters
from skimage import draw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def estimate_derivatives(C: np.ndarray) -> np.ndarray:
    dC = np.zeros_like(C)
    if C.shape[0] < 2:
        return dC
    dC[0] = C[1] - C[0]
    dC[-1] = C[-1] - C[-2]
    if C.shape[0] > 2:
        dC[1:-1] = 0.5 * (C[2:] - C[:-2])
    return dC

def build_surface_coords(arr: np.ndarray, num_circle_points: int = 80):
    """
    arr: (M,N,4)
    returns list of branch surfaces, each of shape (K, num_circle_points, 3)
    """
    surface_coords = []

    for branch_idx in range(arr.shape[0]):
        branch = arr[branch_idx]
        C = branch[:, :3]
        radius = np.clip(branch[:, 3], 1e-5, None)
        dC = estimate_derivatives(C)

        # Skip branches with degenerate geometry (all-zero or near-zero centerlines).
        # This happens for non-existent/padding branches in the artery array.
        valid_inds = np.argwhere(np.sum(np.abs(dC), axis=1) != 0).flatten()
        if valid_inds.shape[0] < 2:
            continue

        out = get_vessel_surface(
            C,
            dC,
            branch_points=[],
            num_centerline_points=C.shape[0],
            num_circle_points=num_circle_points,
            radius=radius,
            num_stenoses=0,
            is_main_branch=(branch_idx == 0),
            constant_radius=False,
            return_surface=True
        )

        X, Y, Z = out[0], out[1], out[2]
        surf = np.stack((X, Y, Z), axis=-1)
        surface_coords.append(surf)

    return surface_coords

def _render_projection_from_surface_rings(
    surface_rings,
    theta_deg: float,
    phi_deg: float,
    image_dim: int,
    sid: float,
    imager_pixel_spacing: float,
):
    num_views = 1
    SID = np.ones(num_views) * sid
    distance_source_to_iso = 0.75
    distance_detector_to_iso = SID - distance_source_to_iso
    img_dim = image_dim
    sensor_width = imager_pixel_spacing * img_dim / 1000.0

    all_points = np.concatenate([surf.reshape(-1, 3) for surf in surface_rings], axis=0)
    rotated_spatial_coords_3D = rotate_volume(0, 0, 0, all_points)

    rotated_surface_rings = []
    start = 0
    for surf in surface_rings:
        count = surf.shape[0] * surf.shape[1]
        rotated_surface_rings.append(rotated_spatial_coords_3D[start:start + count].reshape(surf.shape))
        start += count

    V_sensor, V_source, localX, localY = get_local_params(
        np.array([theta_deg], dtype=np.float32),
        np.array([phi_deg], dtype=np.float32),
        num_views,
        distance_detector_to_iso,
        distance_source_to_iso,
        coord_system_change=True,
    )

    mask = np.zeros((img_dim, img_dim), dtype=np.bool_)
    bridge_stride = 2

    for branch_surf in rotated_surface_rings:
        rings = branch_surf.reshape(-1, branch_surf.shape[-2], 3)
        prev_ring_rows = None
        prev_ring_cols = None

        for ring in rings:
            plane_points = ray_image_intersection(
                ring,
                V_source[0],
                localX[0],
                localY[0],
                V_sensor[0],
            )

            if plane_points.shape[0] < 3:
                continue

            ring_px = np.round(
                convert3D_to_pixels(
                    plane_points,
                    0,
                    img_dim,
                    V_sensor,
                    sensor_width,
                    localX,
                    localY,
                )
            )

            in_bounds = np.logical_and(
                np.all(ring_px > 0, axis=1),
                np.all(ring_px < img_dim, axis=1),
            )
            ring_px = ring_px[in_bounds]
            if ring_px.shape[0] < 3:
                continue

            rr = np.clip((img_dim - ring_px[:, 1]).astype(np.int32), 0, img_dim - 1)
            cc = np.clip(ring_px[:, 0].astype(np.int32), 0, img_dim - 1)
            poly_rr, poly_cc = draw.polygon(rr, cc, shape=mask.shape)
            mask[poly_rr, poly_cc] = True

            if prev_ring_rows is not None and prev_ring_cols is not None:
                ring_len = min(len(rr), len(prev_ring_rows))
                for t in range(0, ring_len, bridge_stride):
                    line_rr, line_cc = draw.line(
                        int(prev_ring_rows[t]),
                        int(prev_ring_cols[t]),
                        int(rr[t]),
                        int(cc[t]),
                    )
                    valid = (
                        (line_rr >= 0) & (line_rr < img_dim) &
                        (line_cc >= 0) & (line_cc < img_dim)
                    )
                    mask[line_rr[valid], line_cc[valid]] = True

            prev_ring_rows = rr
            prev_ring_cols = cc

    closed_binary_image = morph.closing(mask, morph.disk(2))
    blurred_binary_image = filters.gaussian(closed_binary_image, sigma=0.5) > 0.25
    return blurred_binary_image.astype(np.float32)
