"""Steady hyperemic 1D network model of a coronary tree and FFR along its centrelines.

Per branch, with segment flow Q and lumen area A(s) = pi r(s)^2, the steady 1D
momentum equation (rigid wall, axisymmetric profile of order zeta) gives

    dH/ds = -2 (zeta + 2) pi mu Q / A^2,          H = P + rho Q^2 / (2 A^2)

i.e. total pressure H is lost to viscous friction (zeta = 2: Poiseuille) and the
static pressure follows from H minus the kinetic term (reversible Bernoulli
acceleration). Inside detected lesions the distributed model is replaced by the
Young-Tsai lumped loss (as in Blanco et al. 2018, 1D vs 3D FFR):

    dP_lesion = Kv mu U0 / D0 + Kt rho/2 (A0/As - 1)^2 U0^2,   U0 = Q / A0, Kt = 1.52
    Kv = 32 (0.83 Ls + 3.28 sqrt(As)) / (2 sqrt(A0)) * (0.75 A0/As + 0.25)^2

Junctions: total-pressure continuity (default) or static-pressure continuity.

Boundary conditions:
  bc_mode = "resistance" (default): outlet microvascular resistances R_i are calibrated
      on the healthy reference tree (lumen = reference radius) so that it carries the
      target hyperemic flow with the diameter-power split at aortic pressure Pa; the
      diseased tree is then solved with the same Pa, R_i and venous pressure Pv, so a
      stenosis reduces its own flow (FFR = Pd/Pa with flow adaptation).
  bc_mode = "fixed_flow": the target outlet flows are imposed on the diseased tree
      directly (QFR-like fixed-flow assumption).

Reported FFR(s) = P(s) / Pa. ``ffr`` uses the pressure after kinetic recovery to the
reference lumen (what a pressure wire reads once the jet has recovered, monotone along
a branch); ``ffr_static`` is the local static pressure including the reversible
Bernoulli dip at a throat.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import root

from vessel_code.ffr.boundary import ML, split_flows, total_hyperemic_flow
from vessel_code.ffr.reference import Lesion, detect_lesions, reference_radius
from vessel_code.ffr.tree import VesselTree, build_tree

MMHG = 133.322


@dataclass
class FFRConfig:
    # Physiology
    pa_mmhg: float = 93.0                 # mean aortic pressure at hyperemia
    pv_mmhg: float = 5.0                  # venous / back pressure of the microcirculation
    rho: float = 1060.0                   # kg/m^3
    mu: float = 0.0035                    # Pa s
    profile_zeta: float = 2.0             # 2 = Poiseuille; 9 = blunt profile (Blanco: varpi=11)
    junction: str = "total_pressure"      # or "static"
    # Boundary conditions
    bc_mode: str = "resistance"           # or "fixed_flow"
    flow_model: str = "fixed_velocity"    # fixed_velocity | scaling_law | frame_count | explicit
    hyperemic_velocity_m_s: float = 0.35
    hyperemia_factor: float = 3.0         # scaling_law: hyperemic / resting flow
    frame_count: float | None = None
    frame_rate_fps: float | None = None
    contrast_path_mm: float | None = None
    frame_count_hyperemia_factor: float = 1.0
    flow_ml_s: float | None = None
    split_exponent: float = 2.27
    split_window_mm: float = 10.0
    # Lumen reference and lesions
    ds_threshold: float = 0.30            # diameter stenosis that defines a lesion core
    ds_edge: float = 0.10
    lesion_merge_gap_mm: float = 3.0
    kt: float = 1.52
    use_lesion_model: bool = True
    reference_smooth_samples: float = 2.0

    @classmethod
    def from_dict(cls, values: dict) -> "FFRConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown FFR config keys: {sorted(unknown)}")
        return cls(**values)


@dataclass
class BranchResult:
    index: int
    name: str
    parent: int
    attach: int
    start: int
    xyz_mm: np.ndarray
    radius_mm: np.ndarray
    ref_radius_mm: np.ndarray
    distance_mm: np.ndarray       # along the flow path from the ostium
    pressure_mmhg: np.ndarray     # recovered pressure (see module doc)
    static_mmhg: np.ndarray
    flow_ml_s: np.ndarray         # flow in the segment downstream of each sample
    ffr: np.ndarray
    ffr_static: np.ndarray
    lesions: list = field(default_factory=list)

    @property
    def distal_ffr(self) -> float:
        return float(self.ffr[-1])


@dataclass
class FFRResult:
    artery_type: str
    branches: dict                # index -> BranchResult
    config: FFRConfig
    q_target_ml_s: float
    q_healthy_ml_s: float
    q_disease_ml_s: float
    outlet_resistance: dict       # index -> mmHg s / ml
    converged: bool
    solver_message: str = ""

    def summary(self) -> dict:
        branches = []
        for b in self.branches.values():
            branches.append({
                "index": b.index, "name": b.name, "parent": b.parent,
                "length_mm": float(b.distance_mm[-1] - b.distance_mm[b.start]),
                "distal_ffr": b.distal_ffr, "min_ffr": float(b.ffr.min()),
                "outlet_flow_ml_s": float(b.flow_ml_s[-1]),
                "min_lumen_diameter_mm": float(2 * b.radius_mm[b.start:].min()),
                "lesions": [l.as_dict() for l in b.lesions],
            })
        all_lesions = [l for b in self.branches.values() for l in b.lesions]
        return {
            "artery_type": self.artery_type,
            "vessel_ffr": {b.name: b.distal_ffr for b in self.branches.values() if b.index in (0, 1) and
                           (b.index == 0 or self.artery_type == "lca")},
            "min_ffr": float(min(b.ffr.min() for b in self.branches.values())),
            "ischaemic_ffr_le_0p80": bool(any(b.distal_ffr <= 0.80 for b in self.branches.values()
                                              if b.index in (0, 1))),
            "num_lesions": len(all_lesions),
            "flow_ml_s": {"target_hyperemic": self.q_target_ml_s, "healthy_tree": self.q_healthy_ml_s,
                          "diseased_tree": self.q_disease_ml_s},
            "converged": self.converged,
            "solver_message": self.solver_message,
            "branches": branches,
            "config": asdict(self.config),
        }

    def to_json(self) -> str:
        return json.dumps(self.summary(), indent=2, allow_nan=True)


# --------------------------------------------------------------------------------------
# Core forward pass


@dataclass
class _Geometry:
    radius: dict      # index -> [N] mm
    ref: dict         # index -> [N] mm
    lesions: dict     # index -> list[Lesion]


def _subtree_flows(tree: VesselTree, q_out: dict) -> dict:
    sub = {}
    for index in reversed(tree.order):
        b = tree.branches[index]
        sub[index] = q_out[index] + sum(sub[c] for _, c in b.children)
    return sub


def _young_tsai(q: float, a0: float, a_s: float, length: float, d0: float, cfg: FFRConfig) -> tuple:
    u0 = q / a0
    kv = 32.0 * (0.83 * length + 3.28 * np.sqrt(a_s)) / (2.0 * np.sqrt(a0)) * (0.75 * a0 / a_s + 0.25) ** 2
    viscous = kv * cfg.mu * u0 / d0
    turbulent = cfg.kt * 0.5 * cfg.rho * (a0 / a_s - 1.0) ** 2 * u0 * abs(u0)
    return viscous, turbulent


def _forward(tree: VesselTree, geom: _Geometry, q_out: dict, cfg: FFRConfig) -> dict:
    """Pressures (Pa) along every branch for given outlet flows (m^3/s)."""
    sub = _subtree_flows(tree, q_out)
    pa = cfg.pa_mmhg * MMHG
    kr = 2.0 * (cfg.profile_zeta + 2.0) * np.pi * cfg.mu
    out = {}
    for index in tree.order:
        b = tree.branches[index]
        n = len(b.s)
        r = 1e-3 * geom.radius[index]
        rr = 1e-3 * geom.ref[index]
        area = np.pi * r ** 2
        ref_area = np.pi * rr ** 2
        s = 1e-3 * b.s
        # Segment flows (segment i: sample i -> i+1); the flow at a sample is the downstream segment's.
        q = np.full(n, sub[index])
        for k, c in b.children:
            q[k:] -= sub[c]
        q = np.maximum(q, 0.0)
        h = np.zeros(n)
        st = b.start
        if b.parent < 0:
            # Static pressure at the ostium equals the aortic pressure.
            h[st] = pa + 0.5 * cfg.rho * (q[st] / area[st]) ** 2
        else:
            parent = out[b.parent]
            k = b.attach
            if cfg.junction == "total_pressure":
                h[st] = parent["h"][k]
            else:
                h[st] = parent["p"][k] + 0.5 * cfg.rho * (q[st] / area[st]) ** 2
        h[:st] = h[st]
        lesion_at = {les.i0: les for les in geom.lesions.get(index, [])} if cfg.use_lesion_model else {}
        i = st
        while i < n - 1:
            les = lesion_at.get(i)
            if les is not None and les.i1 > i:
                a, z, m = les.i0, les.i1, les.imin
                a0 = ref_area[m]
                a_s = area[m]
                length = max(s[z] - s[a], 1e-4)
                visc, turb = _young_tsai(q[a], a0, a_s, length, 2 * rr[m], cfg)
                # Distribute: viscous part ~ cumulative int ds/A^2; separation loss after the throat
                inv = 1.0 / area[a : z + 1] ** 2
                cum = np.concatenate(([0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s[a : z + 1]))))
                fv = cum / max(cum[-1], 1e-30)
                recov = np.zeros(z - a + 1)
                after = np.arange(a, z + 1) >= m
                if after.any() and z > m:
                    inv_t, inv_e = 1.0 / area[m] ** 2, 1.0 / area[z] ** 2
                    denom = max(inv_t - inv_e, 1e-30)
                    recov[after] = np.clip((inv_t - 1.0 / area[a : z + 1][after] ** 2) / denom, 0.0, 1.0)
                recov[-1] = 1.0
                h[a : z + 1] = h[a] - visc * fv - turb * recov
                i = z
                continue
            ds = s[i + 1] - s[i]
            h[i + 1] = h[i] - kr * q[i] * ds * 0.5 * (1.0 / area[i] ** 2 + 1.0 / area[i + 1] ** 2)
            i += 1
        kinetic = 0.5 * cfg.rho * (q / area) ** 2
        p = h - kinetic
        p_rec = h - 0.5 * cfg.rho * (q / ref_area) ** 2
        p[:st] = p[st]
        p_rec[:st] = p_rec[st]
        out[index] = {"h": h, "p": p, "p_rec": p_rec, "q": q}
    return out


def _outlet_pressures(tree: VesselTree, fields: dict) -> np.ndarray:
    return np.array([fields[i]["p"][-1] for i in tree.order])


# --------------------------------------------------------------------------------------
# Public API


def prepare_geometry(tree: VesselTree, cfg: FFRConfig) -> tuple:
    ref = {i: reference_radius(b.radius, start=b.start, smooth_samples=cfg.reference_smooth_samples)
           for i, b in tree.branches.items()}
    lesions = {i: detect_lesions(i, b.s, b.radius, ref[i], start=b.start, ds_threshold=cfg.ds_threshold,
                                 ds_edge=cfg.ds_edge, merge_gap_mm=cfg.lesion_merge_gap_mm)
               for i, b in tree.branches.items()}
    disease = _Geometry({i: b.radius for i, b in tree.branches.items()}, ref, lesions)
    healthy = _Geometry(ref, ref, {})
    return disease, healthy


def solve_ffr_1d(tree: VesselTree, cfg: FFRConfig | None = None) -> FFRResult:
    cfg = cfg or FFRConfig()
    disease, healthy = prepare_geometry(tree, cfg)
    q_total = total_hyperemic_flow(tree, healthy.ref, cfg)
    q_target = split_flows(tree, healthy.ref, q_total, exponent=cfg.split_exponent, window_mm=cfg.split_window_mm)
    order = tree.order
    q_vec = np.array([q_target[i] for i in order])
    pv = cfg.pv_mmhg * MMHG
    healthy_fields = _forward(tree, healthy, q_target, cfg)
    p_h = _outlet_pressures(tree, healthy_fields)
    resist = np.maximum(p_h - pv, 1.0) / np.maximum(q_vec, 1e-15)  # Pa s / m^3
    converged, message = True, "fixed flow"
    if cfg.bc_mode == "resistance":
        scale = q_vec.copy()

        def residual(x):
            q = dict(zip(order, np.maximum(x, 0.0) * scale))
            f = _forward(tree, disease, q, cfg)
            return (_outlet_pressures(tree, f) - pv - resist * np.maximum(x, 0.0) * scale) / (cfg.pa_mmhg * MMHG)

        sol = root(residual, np.full(len(order), 0.95), method="hybr", options={"xtol": 1e-10})
        x = np.maximum(sol.x, 0.0)
        if not sol.success or np.max(np.abs(residual(x))) > 1e-6:
            # Damped fixed-point fallback: q_i <- (P_i - Pv) / R_i
            x = np.ones(len(order))
            for _ in range(500):
                q = dict(zip(order, x * scale))
                p = _outlet_pressures(tree, _forward(tree, disease, q, cfg))
                x_new = np.maximum(p - pv, 0.0) / resist / scale
                if np.max(np.abs(x_new - x)) < 1e-9:
                    x = x_new
                    break
                x = 0.5 * x + 0.5 * x_new
            converged = bool(np.max(np.abs(residual(x))) < 1e-5)
            message = f"hybr failed ({sol.message}); fixed-point residual {np.max(np.abs(residual(x))):.2e}"
        else:
            message = "hybr converged"
        q_final = dict(zip(order, x * scale))
    elif cfg.bc_mode == "fixed_flow":
        q_final = dict(q_target)
    else:
        raise ValueError(f"unknown bc_mode {cfg.bc_mode!r}")
    fields = _forward(tree, disease, q_final, cfg)
    pa = cfg.pa_mmhg * MMHG
    results = {}
    for i in order:
        b, f = tree.branches[i], fields[i]
        ffr = f["p_rec"] / pa
        # Recovered FFR is monotone along a branch by construction except for kinetic changes at
        # junctions; enforce it so the colour map is not affected by sub-0.001 numerical noise.
        ffr_mono = ffr.copy()
        ffr_mono[b.start:] = np.minimum.accumulate(ffr[b.start:])
        les = disease.lesions[i]
        for l in les:
            l.dffr = float(ffr_mono[max(l.i0, b.start)] - ffr_mono[l.i1])
        results[i] = BranchResult(
            i, b.name, b.parent, b.attach, b.start, b.xyz, b.radius, disease.ref[i],
            tree.distance_from_ostium(i), f["p_rec"] / MMHG, f["p"] / MMHG, f["q"] / ML,
            ffr_mono, f["p"] / pa, les)
    return FFRResult(tree.artery_type, results, cfg, q_total / ML,
                     float(sum(q_target.values())) / ML, float(sum(q_final.values())) / ML,
                     {i: float(r) * ML / MMHG for i, r in zip(order, resist)}, converged, message)


def run_ffr(vessel_mm: np.ndarray, exists: np.ndarray, artery_type: str = "unknown",
            cfg: FFRConfig | None = None) -> FFRResult:
    tree = build_tree(vessel_mm, exists, artery_type)
    return solve_ffr_1d(tree, cfg)
