"""Hyperemic boundary conditions: total inflow and its split over the outlets.

Total hyperemic inflow (``flow_model``):
  fixed_velocity  Q = v_hyp * mean A_ref of each major vessel (RCA; LM-LAD + LCX),
                  v_hyp = 0.35 m/s (QFR "fixed-flow" fQFR, FAVOR Pilot)
  scaling_law     Q_rest[ml/min] = 1.43 d[mm]^2.55 (van der Giessen et al. 2011), Q = factor * Q_rest
  frame_count     v = L_contrast / (frames / fps) (TIMI frame count), Q = factor * v * A_ref(inlet)
  explicit        Q given in ml/s

Split: at every junction Q_child / Q_continuation = (d_child / d_continuation)^k with
k = 2.27 (van der Giessen et al. 2011; Murray's law is k = 3), using reference diameters
averaged over ``window_mm`` downstream of the junction.
"""
from __future__ import annotations

import numpy as np

from vessel_code.ffr.tree import VesselTree

ML = 1e-6  # m^3 per ml


def mean_over_window(s: np.ndarray, values: np.ndarray, start_index: int, window_mm: float) -> float:
    s0 = s[start_index]
    sel = (s >= s0) & (s <= s0 + window_mm)
    sel[start_index] = True
    return float(np.mean(values[sel]))


def inlet_reference_diameter_mm(tree: VesselTree, ref_radius: dict, window_mm: float = 5.0) -> float:
    main = tree.branches[0]
    return 2.0 * mean_over_window(main.s, ref_radius[0], 0, window_mm)


def total_hyperemic_flow(tree: VesselTree, ref_radius: dict, cfg) -> float:
    """Total inflow (m^3/s)."""
    d_mm = inlet_reference_diameter_mm(tree, ref_radius)
    area_m2 = np.pi * (0.5e-3 * d_mm) ** 2
    model = cfg.flow_model
    if model == "fixed_velocity":
        # QFR fixed-flow: hyperemic velocity x mean reference lumen area of each major vessel
        # (RCA; LM-LAD and LCX for the LCA).
        majors = [0, 1] if tree.artery_type == "lca" and 1 in tree.branches else [0]
        total = 0.0
        for index in majors:
            b = tree.branches[index]
            total += np.pi * np.mean((1e-3 * ref_radius[index][b.start:]) ** 2)
        return cfg.hyperemic_velocity_m_s * total
    if model == "scaling_law":
        q_rest_ml_s = 1.43 * d_mm ** 2.55 / 60.0
        return cfg.hyperemia_factor * q_rest_ml_s * ML
    if model == "frame_count":
        if not cfg.frame_count or not cfg.frame_rate_fps:
            raise ValueError("flow_model=frame_count needs frame_count and frame_rate_fps")
        length_mm = cfg.contrast_path_mm or tree.branches[0].length
        velocity = 1e-3 * length_mm / (cfg.frame_count / cfg.frame_rate_fps)
        return cfg.frame_count_hyperemia_factor * velocity * area_m2
    if model == "explicit":
        if cfg.flow_ml_s is None:
            raise ValueError("flow_model=explicit needs flow_ml_s")
        return float(cfg.flow_ml_s) * ML
    raise ValueError(f"unknown flow_model {model!r}")


def split_flows(tree: VesselTree, ref_radius: dict, q_total: float, *, exponent: float,
                window_mm: float) -> dict:
    """Outlet flow of every branch (m^3/s) from a diameter-power split at each junction."""
    outlet = {}

    def walk(index: int, q_in: float) -> None:
        b = tree.branches[index]
        ref = ref_radius[index]
        q = q_in
        for k, child in b.children:
            c = tree.branches[child]
            d_child = 2 * mean_over_window(c.s, ref_radius[child], c.start, window_mm)
            d_cont = 2 * mean_over_window(b.s, ref, min(k + 1, len(b.s) - 1), window_mm)
            frac = d_child ** exponent / (d_child ** exponent + d_cont ** exponent)
            q_child = frac * q
            q -= q_child
            walk(child, q_child)
        outlet[index] = q

    walk(0, q_total)
    return outlet
