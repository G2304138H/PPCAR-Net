"""Reproject a predicted vessel code into real X-ray views and draw the overlays (numpy / matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from vessel_code.geometry.surface_rendering import build_surface_coords, _render_projection_from_surface_rings

PRED_RGB = np.array([1.0, 0.55, 0.0])      # predicted silhouette on the X-ray (orange)
TP_RGB = np.array([0.1, 0.85, 0.2])        # predicted and segmented
FP_RGB = np.array([1.0, 0.15, 0.15])       # predicted only
FN_RGB = np.array([1.0, 1.0, 1.0])         # segmented only


def reproject_masks(vessel_centered_mm: np.ndarray, active: np.ndarray, theta_deg, phi_deg, camera: dict,
                    num_circle_points: int = 120) -> np.ndarray:
    """Silhouette masks (V,256,256) of the predicted tube surface, rendered with the training projector."""
    vessel_m = np.asarray(vessel_centered_mm, np.float32)[np.asarray(active, bool)] * 0.001
    surfaces = build_surface_coords(vessel_m, num_circle_points=num_circle_points)
    dim = int(camera["image_dim"])
    if not surfaces:
        return np.zeros((len(theta_deg), dim, dim), np.float32)
    return np.stack([
        np.clip(_render_projection_from_surface_rings(surfaces, float(t), float(p), dim, float(camera["sid"]),
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
