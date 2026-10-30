"""Monte Carlo propagation of reconstruction and physiology uncertainty to FFR(s).

FFR is very sensitive to the minimum lumen radius (losses scale ~ r^-4), and radii
reconstructed from 2-3 sparse X-ray views carry roughly pixel-level errors. Each draw
perturbs: a global log-radius scale (calibration), a smooth local log-radius field per
branch (reconstruction noise, correlation length ``corr_mm``), the hyperemic flow
(log-normal) and the aortic pressure, then re-runs the full pipeline (reference fit,
lesion detection, BCs, 1D solve).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vessel_code.ffr.solver1d import FFRConfig, FFRResult, run_ffr


@dataclass
class UncertaintyConfig:
    draws: int = 200
    sigma_global_log_radius: float = 0.05     # ~5% radius scale error
    sigma_local_log_radius: float = 0.08      # ~8% local radius error
    corr_mm: float = 5.0
    sigma_log_flow: float = 0.25
    sigma_pa_mmhg: float = 5.0
    seed: int = 0


def _local_field(rng, n: int, spacing_mm: float, sigma: float, corr_mm: float) -> np.ndarray:
    white = rng.standard_normal(n)
    width = max(corr_mm / max(spacing_mm, 1e-6), 1e-3)
    smooth = gaussian_filter1d(white, width, mode="nearest")
    std = smooth.std()
    return sigma * smooth / std if std > 0 else np.zeros(n)


def monte_carlo(vessel_mm: np.ndarray, exists: np.ndarray, artery_type: str, cfg: FFRConfig,
                ucfg: UncertaintyConfig, nominal: FFRResult) -> dict:
    """Return per-branch percentile bands {index: {"p05","p50","p95"}} and distal-FFR stats."""
    rng = np.random.default_rng(ucfg.seed)
    samples = {i: [] for i in nominal.branches}
    distal = {i: [] for i in nominal.branches}
    failures = 0
    for _ in range(ucfg.draws):
        v = np.array(vessel_mm, dtype=np.float64, copy=True)
        g = rng.normal(0.0, ucfg.sigma_global_log_radius)
        for i in np.flatnonzero(exists):
            seg = np.linalg.norm(np.diff(v[i, :, :3], axis=0), axis=1)
            spacing = float(np.mean(seg)) if seg.size else 1.0
            local = _local_field(rng, v.shape[1], spacing, ucfg.sigma_local_log_radius, ucfg.corr_mm)
            v[i, :, 3] *= np.exp(g + local)
        flow_scale = float(np.exp(rng.normal(0.0, ucfg.sigma_log_flow)))
        draw_cfg = replace(cfg, pa_mmhg=float(cfg.pa_mmhg + rng.normal(0.0, ucfg.sigma_pa_mmhg)),
                           hyperemic_velocity_m_s=cfg.hyperemic_velocity_m_s * flow_scale,
                           hyperemia_factor=cfg.hyperemia_factor * flow_scale,
                           frame_count_hyperemia_factor=cfg.frame_count_hyperemia_factor * flow_scale,
                           flow_ml_s=None if cfg.flow_ml_s is None else cfg.flow_ml_s * flow_scale)
        try:
            res = run_ffr(v, exists, artery_type, draw_cfg)
        except Exception:  # noqa: BLE001 - a failed draw is counted, not fatal
            failures += 1
            continue
        for i, b in res.branches.items():
            if i in samples:
                samples[i].append(b.ffr)
                distal[i].append(b.distal_ffr)
    bands, stats = {}, {}
    for i, rows in samples.items():
        if not rows:
            continue
        arr = np.clip(np.stack(rows), 0.0, 1.0)
        bands[i] = {q: np.percentile(arr, p, axis=0) for q, p in (("p05", 5), ("p50", 50), ("p95", 95))}
        d = np.clip(np.asarray(distal[i]), 0.0, 1.0)
        stats[i] = {"distal_p05": float(np.percentile(d, 5)), "distal_p50": float(np.median(d)),
                    "distal_p95": float(np.percentile(d, 95)),
                    "prob_ffr_le_0p80": float(np.mean(d <= 0.80))}
    return {"bands": bands, "stats": stats, "draws": ucfg.draws, "failures": failures}
