"""Healthy reference lumen, lesion detection and virtual lesions.

Reference radius: a non-increasing (isotonic) upper envelope of the lumen radius,
fitted with asymmetric weights so that focal narrowings do not pull it down. This is
the same idea as the "fitted reference diameter" of 3D-QCA / QFR: proximal and distal
healthy segments define the expected lumen, steps at bifurcations are allowed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import isotonic_regression


def _interior_runs(mask: np.ndarray) -> np.ndarray:
    """Keep only True runs with False samples on both sides (a narrowing that recovers).

    A run touching either end is tapering into/out of the segment, not a focal lesion."""
    out = mask.copy()
    n = len(mask)
    if n and mask[0]:
        i = 0
        while i < n and mask[i]:
            out[i] = False
            i += 1
    if n and mask[-1]:
        i = n - 1
        while i >= 0 and mask[i]:
            out[i] = False
            i -= 1
    return out


def _widen(mask: np.ndarray, factor: float = 0.6, minimum: int = 2) -> np.ndarray:
    """Extend every True run by ``factor`` x its length on both sides (lesion shoulders)."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            pad = max(minimum, int(round(factor * (j - i + 1))))
            out[max(0, i - pad): min(n, j + pad + 1)] = True
            i = j + 1
        else:
            i += 1
    return out


def reference_radius(radius: np.ndarray, *, start: int = 0, iterations: int = 6,
                     dip_tolerance: float = 0.03, smooth_samples: float = 2.0) -> np.ndarray:
    """Non-increasing reference radius from sample ``start`` onwards.

    Iterates: isotonic (non-increasing) fit -> samples more than ``dip_tolerance`` below
    the fit are treated as narrowing and replaced by linear interpolation between the
    surrounding healthy samples -> refit. Focal lesions therefore do not pull the
    reference down, while steps at bifurcations and gradual tapering are kept.
    """
    r = np.asarray(radius, dtype=np.float64)
    ref = r.copy()
    seg = r[start:]
    if seg.size < 3:
        return ref
    k = np.arange(seg.size)
    # Start: isotonic fit with asymmetric weights (samples below the fit count little).
    w = np.ones_like(seg)
    for _ in range(4):
        fit = isotonic_regression(seg, weights=w, increasing=False).x
        w = np.where(seg >= fit, 1.0, 0.05)
    for _ in range(iterations):
        dip = _interior_runs(_widen(_interior_runs(seg < fit * (1.0 - dip_tolerance))))
        if dip.all() or not dip.any():
            break
        filled = seg.copy()
        filled[dip] = np.interp(k[dip], k[~dip], seg[~dip])
        new_fit = isotonic_regression(filled, increasing=False).x
        if np.allclose(new_fit, fit, rtol=1e-6):
            fit = new_fit
            break
        fit = new_fit
    if smooth_samples > 0:
        fit = gaussian_filter1d(fit, smooth_samples, mode="nearest")
    ref[start:] = np.maximum(fit, 1e-3)
    # Inside the parent lumen (side branches) keep the first fitted value.
    ref[:start] = ref[start]
    return ref


@dataclass
class Lesion:
    branch: int
    i0: int            # first sample (inclusive)
    i1: int            # last sample (inclusive)
    imin: int          # minimum-lumen sample
    mld_mm: float      # minimum lumen diameter
    ref_d_mm: float    # reference diameter at the throat
    ds_percent: float  # diameter stenosis
    as_percent: float  # area stenosis
    length_mm: float
    dffr: float = float("nan")  # FFR drop across the lesion (filled by the solver)

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in self.__dict__.items()}


def detect_lesions(branch_index: int, s: np.ndarray, radius: np.ndarray, ref: np.ndarray, *,
                   start: int = 0, ds_threshold: float = 0.3, ds_edge: float = 0.10,
                   merge_gap_mm: float = 3.0, min_length_mm: float = 1.0) -> list:
    """Contiguous runs with diameter stenosis >= ``ds_threshold`` extended to ``ds_edge``."""
    ds = 1.0 - radius / np.maximum(ref, 1e-6)
    ds[:start] = 0.0
    core = ds >= ds_threshold
    if not core.any():
        return []
    edge = ds >= ds_edge
    runs = []
    i, n = 0, len(ds)
    while i < n:
        if core[i]:
            a = b = i
            while b + 1 < n and core[b + 1]:
                b += 1
            # Extend to the shoulders: while still narrowed (>= ds_edge) and the narrowing keeps
            # decreasing away from the core (stops at the next dip instead of swallowing it).
            while a - 1 >= start and edge[a - 1] and ds[a - 1] <= ds[a] + 1e-9:
                a -= 1
            while b + 1 < n and edge[b + 1] and ds[b + 1] <= ds[b] + 1e-9:
                b += 1
            runs.append([a, b])
            i = b + 1
        else:
            i += 1
    merged = [runs[0]]
    for a, b in runs[1:]:
        if s[a] - s[merged[-1][1]] <= merge_gap_mm:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    lesions = []
    for a, b in merged:
        b = max(b, a + 1) if a + 1 < n else b
        k = a + int(np.argmax(ds[a : b + 1]))
        length = float(s[b] - s[a])
        if length < min_length_mm:
            continue
        lesions.append(Lesion(branch_index, a, b, k, 2 * float(radius[k]), 2 * float(ref[k]),
                              100 * float(ds[k]), 100 * float(1 - (radius[k] / ref[k]) ** 2), length))
    return lesions


def apply_virtual_lesion(vessel_mm: np.ndarray, branch: int, position: float, sigma: float,
                         severity: float) -> np.ndarray:
    """Insert a Gaussian lesion with the same profile as the model's lesion head.

    r(t) <- r(t) * exp(-log(1/(1-severity)) * exp(-0.5((t-position)/sigma)^2)),
    t in [0,1] over the branch samples; ``severity`` is the diameter stenosis at the throat.
    """
    out = np.array(vessel_mm, dtype=np.float64, copy=True)
    n = out.shape[1]
    t = np.linspace(0.0, 1.0, n)
    depth = -np.log1p(-min(float(severity), 0.999))
    out[branch, :, 3] *= np.exp(-depth * np.exp(-0.5 * ((t - position) / max(sigma, 1e-6)) ** 2))
    return out


def parse_virtual_lesion(spec: str) -> dict:
    """``"branch=0,t=0.35,sigma=0.03,severity=0.6"`` -> kwargs for :func:`apply_virtual_lesion`."""
    values = dict(item.split("=") for item in spec.split(","))
    return {"branch": int(values["branch"]), "position": float(values.get("t", values.get("position"))),
            "sigma": float(values.get("sigma", 0.03)), "severity": float(values["severity"])}
