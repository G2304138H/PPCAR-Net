"""Reproject a predicted vessel code into real X-ray views and draw the overlays (numpy / matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from skimage import draw, filters, morphology as morph

from vessel_code.geometry.fwd_projection_functions import (convert3D_to_pixels, get_local_params,
                                                           ray_image_intersection, rotate_volume)
from vessel_code.geometry.surface_rendering import build_surface_coords

PRED_RGB = np.array([1.0, 0.55, 0.0])      # predicted silhouette on the X-ray (orange)
TP_RGB = np.array([0.1, 0.85, 0.2])        # predicted and segmented
FP_RGB = np.array([1.0, 0.15, 0.15])       # predicted only
FN_RGB = np.array([1.0, 1.0, 1.0])         # segmented only


def render_surface_rings(surface_rings, theta_deg: float, phi_deg: float, image_dim: int, sid: float,
                         source_to_iso: float, imager_pixel_spacing: float) -> np.ndarray:
    """The training renderer (geometry/surface_rendering._render_projection_from_surface_rings), with the
    source-to-isocentre distance as a parameter instead of the fixed 0.75 m. Same rasterisation, closing and blur."""
    num_views = 1
    SID = np.ones(num_views) * sid
    distance_detector_to_iso = SID - source_to_iso
    img_dim = image_dim
    sensor_width = imager_pixel_spacing * img_dim / 1000.0
    all_points = np.concatenate([surf.reshape(-1, 3) for surf in surface_rings], axis=0)
    rotated = rotate_volume(0, 0, 0, all_points)
    rotated_rings, start = [], 0
    for surf in surface_rings:
        count = surf.shape[0] * surf.shape[1]
        rotated_rings.append(rotated[start:start + count].reshape(surf.shape))
        start += count
    V_sensor, V_source, localX, localY = get_local_params(
        np.array([theta_deg], dtype=np.float32), np.array([phi_deg], dtype=np.float32), num_views,
        distance_detector_to_iso, source_to_iso, coord_system_change=True)
    mask = np.zeros((img_dim, img_dim), dtype=np.bool_)
    bridge_stride = 2
    for branch_surf in rotated_rings:
        rings = branch_surf.reshape(-1, branch_surf.shape[-2], 3)
        prev_rr = prev_cc = None
        for ring in rings:
            plane_points = ray_image_intersection(ring, V_source[0], localX[0], localY[0], V_sensor[0])
            if plane_points.shape[0] < 3:
                continue
            ring_px = np.round(convert3D_to_pixels(plane_points, 0, img_dim, V_sensor, sensor_width, localX, localY))
            ring_px = ring_px[np.all(ring_px > 0, axis=1) & np.all(ring_px < img_dim, axis=1)]
            if ring_px.shape[0] < 3:
                continue
            rr = np.clip((img_dim - ring_px[:, 1]).astype(np.int32), 0, img_dim - 1)
            cc = np.clip(ring_px[:, 0].astype(np.int32), 0, img_dim - 1)
            poly_rr, poly_cc = draw.polygon(rr, cc, shape=mask.shape)
            mask[poly_rr, poly_cc] = True
            if prev_rr is not None:
                for t in range(0, min(len(rr), len(prev_rr)), bridge_stride):
                    lr, lc = draw.line(int(prev_rr[t]), int(prev_cc[t]), int(rr[t]), int(cc[t]))
                    ok = (lr >= 0) & (lr < img_dim) & (lc >= 0) & (lc < img_dim)
                    mask[lr[ok], lc[ok]] = True
            prev_rr, prev_cc = rr, cc
    closed = morph.closing(mask, morph.disk(2))
    return (filters.gaussian(closed, sigma=0.5) > 0.25).astype(np.float32)


def reproject_masks(vessel_centered_mm: np.ndarray, active: np.ndarray, theta_deg, phi_deg, camera: dict,
                    num_circle_points: int = 120) -> np.ndarray:
    """Silhouette masks (V,256,256) of the predicted tube surface, rendered with `camera`
    (image_dim, sid [m], source_to_iso [m], imager_pixel_spacing [mm]) - the same camera the refiners used."""
    vessel_m = np.asarray(vessel_centered_mm, np.float32)[np.asarray(active, bool)] * 0.001
    surfaces = build_surface_coords(vessel_m, num_circle_points=num_circle_points)
    dim = int(camera["image_dim"])
    if not surfaces:
        return np.zeros((len(theta_deg), dim, dim), np.float32)
    return np.stack([
        np.clip(render_surface_rings(surfaces, float(t), float(p), dim, float(camera["sid"]),
                                     float(camera.get("source_to_iso", 0.75)),
                                     float(camera["imager_pixel_spacing"])), 0, 1)
        for t, p in zip(np.ravel(theta_deg), np.ravel(phi_deg))
    ]).astype(np.float32)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def translation_to_align(pred: np.ndarray, target: np.ndarray, max_shift: int = 48, sigma: float = 2.0):
    """Integer (d_row, d_col) that best moves `pred` onto `target` (FFT cross-correlation of blurred masks).

    The exported views have no calibrated isocentre (each is centred on its own mask centroid), so a reprojection can be
    off by a translation even when the 3D shape is right. This measures that offset; it changes nothing else.
    """
    a = ndimage.gaussian_filter((pred > 0.5).astype(np.float64), sigma)
    b = ndimage.gaussian_filter((target > 0.5).astype(np.float64), sigma)
    if a.sum() == 0 or b.sum() == 0:
        return 0, 0
    H, W = a.shape
    corr = np.real(np.fft.ifft2(np.fft.fft2(b, s=(2 * H, 2 * W)) * np.conj(np.fft.fft2(a, s=(2 * H, 2 * W)))))
    corr = np.fft.fftshift(corr)                    # zero shift at (H, W)
    win = corr[H - max_shift:H + max_shift + 1, W - max_shift:W + max_shift + 1]
    r, c = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(r - max_shift), int(c - max_shift)


def shift_mask(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    return ndimage.shift(mask.astype(np.float32), (d_row, d_col), order=0, mode="constant", cval=0.0)


def shift_matrix(d_row: float, d_col: float) -> np.ndarray:
    """Homogeneous model->model map for content moved by (d_row, d_col): new(u) = old(u - d)."""
    T = np.eye(3)
    T[0, 2], T[1, 2] = -d_row, -d_col
    return T


def model_to_full(mask_model: np.ndarray, model_to_full_matrix: np.ndarray, full_shape) -> np.ndarray:
    """Model-space mask -> original image grid (inverse of the export resampling)."""
    inv = np.linalg.inv(model_to_full_matrix)            # full -> model
    out = ndimage.affine_transform(mask_model.astype(np.float32), inv[:2, :2], offset=inv[:2, 2],
                                   output_shape=tuple(int(s) for s in full_shape), order=1, mode="constant")
    return out > 0.5


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def overlay_on_xray(raw_u8: np.ndarray, pred: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """X-ray with the predicted silhouette as a translucent fill and a solid outline."""
    g = raw_u8.astype(np.float32) / 255.0 if raw_u8.dtype == np.uint8 else np.clip(raw_u8, 0, 1)
    rgb = np.repeat(g[..., None], 3, -1)
    rgb[pred] = (1 - alpha) * rgb[pred] + alpha * PRED_RGB
    edge = _outline(pred)
    rgb[edge] = PRED_RGB
    return np.clip(rgb, 0, 1)


def overlay_on_mask(seg: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Agreement map: green = both, red = predicted only, white = segmented only, black = neither."""
    seg, pred = seg > 0, pred > 0
    rgb = np.zeros(seg.shape + (3,), np.float32)
    rgb[seg & ~pred] = FN_RGB
    rgb[seg & pred] = TP_RGB
    rgb[pred & ~seg] = FP_RGB
    return rgb


def save_png(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def save_panel(tiles: list[np.ndarray], titles: list[str], path: Path, suptitle: str, legend: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(tiles)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(max(3.8 * n, 7.6), 4.9), squeeze=False)
    for ax, tile, title in zip(axes[0], tiles, titles):
        ax.imshow(tile, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    import textwrap
    fig.suptitle(textwrap.fill(suptitle, width=int(max(3.8 * n, 7.6) * 11)), fontsize=9.5, y=0.985)
    if legend:
        fig.text(0.5, 0.015, legend, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- 3D surface seen from a view
SURFACE_RGB = np.array([0.84, 0.15, 0.16])     # plain predicted surface (same red as the repository's 3D GIFs)
BACKGROUND_RGB = np.array([0.07, 0.07, 0.08])


def radius_colormap():
    """Red (smallest) - yellow - green (largest), as in the repository's radius-coloured GIFs."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("radius", ["#d73027", "#fee08b", "#1a9850"])


def surface_quads(vessel_centered_mm: np.ndarray, active: np.ndarray, num_circle_points: int = 20):
    """Tube surface of the active branches as quads: corners (Q,4,3) in metres and the centreline radius (Q,) in mm."""
    from scipy.spatial import cKDTree
    quads, radii = [], []
    for b in np.flatnonzero(np.asarray(active, bool)):
        branch = np.asarray(vessel_centered_mm[b], np.float32).copy()
        keep = np.isfinite(branch).all(1) & (branch[:, 3] > 0)
        branch = branch[keep]
        if len(branch) < 2:
            continue
        surfs = build_surface_coords(branch[None] * np.array([1e-3, 1e-3, 1e-3, 1e-3], np.float32),
                                     num_circle_points=num_circle_points)
        if not surfs:
            continue
        rings = surfs[0]                                         # (K, C, 3) metres
        tree = cKDTree(branch[:, :3] * 1e-3)
        _, idx = tree.query(rings.mean(1))
        ring_r = branch[idx, 3]                                  # radius of the nearest centreline point (mm)
        K, C = rings.shape[:2]
        k = np.arange(K - 1)[:, None]
        c = np.arange(C)[None, :]
        c1 = (c + 1) % C
        q = np.stack([rings[k, c], rings[k, c1], rings[k + 1, c1], rings[k + 1, c]], -2)   # (K-1, C, 4, 3)
        quads.append(q.reshape(-1, 4, 3))
        radii.append(np.repeat(0.5 * (ring_r[:-1] + ring_r[1:]), C))
    if not quads:
        return np.zeros((0, 4, 3)), np.zeros(0)
    return np.concatenate(quads), np.concatenate(radii)


def render_surface_view(quads_m, quad_radius, theta_deg, phi_deg, camera: dict, model_to_full_matrix, full_shape,
                        shift_rc=(0, 0), radius_range=None):
    """The predicted 3D surface seen along one view's X-ray direction, drawn on that view's full-resolution frame
    (same projection, orientation and position as the reprojected silhouette). Painter's algorithm: quads nearer
    the detector are drawn last, as in the clinical display (seen from the detector side). Returns (plain, radius) RGB."""
    from PIL import Image, ImageDraw
    from vessel_code.geometry.projection import _stage2_camera_parameters, project_points_to_image_rc
    H, W = (int(x) for x in full_shape)
    out_plain = np.ones((H, W, 3)) * BACKGROUND_RGB
    out_radius = out_plain.copy()
    if len(quads_m) == 0:
        return out_plain, out_radius
    sensor, source, lx, ly = _stage2_camera_parameters(np.array([theta_deg]), np.array([phi_deg]),
                                                       sid_m=float(camera["sid"]),
                                                       source_to_iso_m=float(camera.get("source_to_iso", 0.75)))
    pts = quads_m.reshape(-1, 3)
    rc = project_points_to_image_rc(pts, view_index=0, sensor=sensor, source=source, local_x=lx, local_y=ly,
                                    image_dim=int(camera["image_dim"]),
                                    pixel_spacing_mm=float(camera["imager_pixel_spacing"]))
    rc = rc + np.asarray(shift_rc, float)[None]                 # same 2D alignment as the silhouette, if any
    full = (np.asarray(model_to_full_matrix) @ np.c_[rc, np.ones(len(rc))].T).T[:, :2].reshape(-1, 4, 2)
    centre = quads_m.mean(1)
    ray = centre - source[0][None]
    depth = np.linalg.norm(ray, axis=1)
    ray /= depth[:, None]
    n = np.cross(quads_m[:, 2] - quads_m[:, 0], quads_m[:, 3] - quads_m[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.30 + 0.70 * np.abs((n * ray).sum(1))
    ok = np.isfinite(full).all((1, 2))
    order = np.argsort(depth)                                    # near the source first, near the detector last
    order = order[ok[order]]
    lo, hi = radius_range if radius_range else (float(quad_radius.min()), float(quad_radius.max()))
    cm = radius_colormap()
    col_r = cm(np.clip((quad_radius - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3]
    imgs = []
    for colours in (np.repeat(SURFACE_RGB[None], len(quads_m), 0), col_r):
        canvas = Image.new("RGB", (W, H), tuple(int(255 * x) for x in BACKGROUND_RGB))
        draw = ImageDraw.Draw(canvas)
        rgb = np.clip(colours * shade[:, None], 0, 1)
        for i in order:
            draw.polygon([(float(p[1]), float(p[0])) for p in full[i]], fill=tuple(int(255 * x) for x in rgb[i]))
        imgs.append(np.asarray(canvas, np.float32) / 255.0)
    return imgs[0], imgs[1]
