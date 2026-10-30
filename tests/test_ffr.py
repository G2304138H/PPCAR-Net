"""FFR branch: analytic checks of the 1D model, topology, references, 3D surface/case export."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vessel_code.ffr.io import load_vessel
from vessel_code.ffr.reference import apply_virtual_lesion, detect_lesions, reference_radius
from vessel_code.ffr.solver1d import MMHG, FFRConfig, run_ffr
from vessel_code.ffr.tree import build_tree

ROOT = Path(__file__).resolve().parents[1]
N = 200


def straight(length=100.0, r0=1.5, r1=None, n=N, offset=(0.0, 0.0, 0.0), direction=(0.0, 0.0, -1.0)):
    r1 = r0 if r1 is None else r1
    t = np.linspace(0.0, 1.0, n)
    xyz = np.asarray(offset) + np.outer(t * length, direction)
    return np.column_stack([xyz, r0 + (r1 - r0) * t])


def tree_vessel(*branches, total=7):
    v = np.zeros((total, N, 4))
    exists = np.zeros(total, dtype=bool)
    for k, b in enumerate(branches):
        v[k] = b
        exists[k] = True
    return v, exists


def test_poiseuille_straight_tube():
    v, e = tree_vessel(straight(r0=1.5))
    cfg = FFRConfig(bc_mode="fixed_flow", flow_model="explicit", flow_ml_s=2.0, profile_zeta=2.0)
    res = run_ffr(v, e, "rca", cfg)
    q, r, length = 2.0e-6, 1.5e-3, 0.1
    expected = 8 * cfg.mu * length * q / (np.pi * r ** 4) / MMHG
    drop = res.branches[0].static_mmhg[0] - res.branches[0].static_mmhg[-1]
    assert drop == pytest.approx(expected, rel=2e-3)
    assert res.branches[0].lesions == []


def test_bernoulli_taper_and_recovered_pressure():
    # Converging tube, inviscid limit: static drop = rho Q^2/2 (1/A1^2 - 1/A0^2)
    v, e = tree_vessel(straight(r0=2.0, r1=1.5))
    cfg = FFRConfig(bc_mode="fixed_flow", flow_model="explicit", flow_ml_s=3.0, mu=1e-9, ds_threshold=0.9)
    res = run_ffr(v, e, "rca", cfg)
    a0, a1 = np.pi * 2.0e-3 ** 2, np.pi * 1.5e-3 ** 2
    expected = 0.5 * cfg.rho * 3e-6 ** 2 * (1 / a1 ** 2 - 1 / a0 ** 2) / MMHG
    b = res.branches[0]
    assert b.static_mmhg[0] - b.static_mmhg[-1] == pytest.approx(expected, rel=1e-4)


def test_young_tsai_lesion_drop_matches_formula():
    base = straight(r0=1.5)
    v, e = tree_vessel(base)
    v = apply_virtual_lesion(v, 0, 0.5, 0.03, 0.6)
    cfg = FFRConfig(bc_mode="fixed_flow", flow_model="explicit", flow_ml_s=2.0)
    res = run_ffr(v, e, "rca", cfg)
    b = res.branches[0]
    assert len(b.lesions) == 1
    les = b.lesions[0]
    assert les.ds_percent == pytest.approx(60.0, abs=1.0)
    # Pressure drop across the lesion = Young-Tsai(Q, A0, As, Ls) (+ viscous outside lesion)
    from vessel_code.ffr.solver1d import _young_tsai
    a0, a_s = np.pi * (1.5e-3) ** 2, np.pi * (les.mld_mm / 2 * 1e-3) ** 2
    visc, turb = _young_tsai(2e-6, a0, a_s, les.length_mm * 1e-3, 3e-3, cfg)
    drop = (b.pressure_mmhg[les.i0] - b.pressure_mmhg[les.i1]) * MMHG
    assert drop == pytest.approx(visc + turb, rel=0.02)
    # FFR decreases monotonically along the branch
    assert np.all(np.diff(b.ffr[b.start:]) <= 1e-12)


def bifurcation():
    main = straight(length=100.0, r0=2.0, r1=1.6)
    k = 60  # side branch leaves at sample 60 of the main branch
    start = main[k, :3]
    side = straight(length=50.0, r0=1.4, r1=1.2, offset=start, direction=(1.0, 0.0, 0.0))
    return tree_vessel(main, side)


def test_topology_and_mass_conservation():
    v, e = bifurcation()
    tree = build_tree(v, e, "rca")
    side = tree.branches[1]
    assert side.parent == 0 and side.attach == 60
    assert side.start > 0  # first samples lie inside the parent lumen
    res = run_ffr(v, e, "rca", FFRConfig())
    main = res.branches[0]
    q_in = main.flow_ml_s[0]
    q_out = sum(b.flow_ml_s[-1] for b in res.branches.values())
    assert q_out == pytest.approx(q_in, rel=1e-9)
    assert main.flow_ml_s[59] - main.flow_ml_s[60] == pytest.approx(res.branches[1].flow_ml_s[-1], rel=1e-9)
    # Murray-type split with exponent 2.27 on the healthy tree
    assert res.converged


def test_resistance_bc_reduces_flow_and_is_milder_than_fixed_flow():
    v, e = bifurcation()
    v = apply_virtual_lesion(v, 0, 0.3, 0.03, 0.65)
    fixed = run_ffr(v, e, "rca", FFRConfig(bc_mode="fixed_flow"))
    adapt = run_ffr(v, e, "rca", FFRConfig(bc_mode="resistance"))
    assert adapt.q_disease_ml_s < adapt.q_healthy_ml_s
    assert adapt.branches[0].distal_ffr > fixed.branches[0].distal_ffr
    # Outlet condition P_i - Pv = R_i Q_i holds after the solve
    for i, b in adapt.branches.items():
        lhs = b.static_mmhg[-1] - adapt.config.pv_mmhg
        assert lhs == pytest.approx(adapt.outlet_resistance[i] * b.flow_ml_s[-1], rel=1e-5)


def test_ffr_monotone_in_severity():
    v, e = bifurcation()
    values = [run_ffr(apply_virtual_lesion(v, 0, 0.3, 0.03, s), e, "rca").branches[0].distal_ffr
              for s in (0.0, 0.4, 0.55, 0.7)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] > 0.95 and values[-1] < 0.85


def test_reference_ignores_focal_narrowing():
    r = np.linspace(2.0, 1.5, N)
    t = np.linspace(0, 1, N)
    narrowed = r * np.exp(np.log(0.5) * np.exp(-0.5 * ((t - 0.4) / 0.03) ** 2))
    ref = reference_radius(narrowed)
    assert np.max(np.abs(ref - r)) < 0.08
    s = np.linspace(0, 100, N)
    les = detect_lesions(0, s, narrowed, ref)
    assert len(les) == 1 and les[0].ds_percent == pytest.approx(50, abs=4)


@pytest.mark.parametrize("artery", ["rca", "lca"])
def test_bundled_examples_healthy(artery):
    vin = load_vessel(ROOT / "examples" / artery / f"{artery}_0457.npz")
    res = run_ffr(vin.vessel_mm, vin.exists, vin.artery_type)
    assert res.converged
    assert all(b.distal_ffr > 0.9 for b in res.branches.values())
    if artery == "lca":
        assert res.branches[2].parent == 1  # OM-type branch attaches to the LCX


def test_uncertainty_runs():
    from vessel_code.ffr.uncertainty import UncertaintyConfig, monte_carlo
    v, e = bifurcation()
    v = apply_virtual_lesion(v, 0, 0.3, 0.03, 0.6)
    cfg = FFRConfig()
    nominal = run_ffr(v, e, "rca", cfg)
    mc = monte_carlo(v, e, "rca", cfg, UncertaintyConfig(draws=20, seed=1), nominal)
    band = mc["bands"][0]
    assert np.all(band["p05"] <= band["p95"])
    assert mc["failures"] == 0


def test_surface_and_openfoam_case(tmp_path):
    pytest.importorskip("skimage")
    from vessel_code.ffr.mesh3d import build_lumen_surface
    from vessel_code.ffr.openfoam import results_3d, update_outlet_flows, write_openfoam_case
    v, e = bifurcation()
    res = run_ffr(v, e, "rca", FFRConfig())
    surf = build_lumen_surface(res, voxel_mm=0.3)
    assert surf.checks["watertight"] and surf.checks["all_caps_found"]
    for cap in surf.checks["caps"].values():
        assert cap["area_mm2"] == pytest.approx(cap["expected_mm2"], rel=0.25)
    case = write_openfoam_case(res, surf, tmp_path / "cfd")
    for f in ("system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict", "0/U", "0/p", "Allrun",
              "constant/triSurface/lumen.stl"):
        assert (case / f).exists(), f
    meta = json.loads((case / "ffr3d_case.json").read_text())
    q = sum(meta["outlet_flow_m3_s"].values())
    assert q == pytest.approx(res.q_disease_ml_s * 1e-6, rel=1e-9)
    # Fake solver output equal to the 1D static pressure -> 3D FFR equals 1D static FFR
    pts = np.asarray(meta["sample_points_m"])
    p = [meta["ffr1d"][str(b)]["ffr_static"][i] * meta["pa_mmhg"] * MMHG / meta["rho"] for b, i in meta["sample_index"]]
    out = case / "postProcessing" / "sampleCentreline" / "100"
    out.mkdir(parents=True)
    np.savetxt(out / "centreline_p.xy", np.column_stack([pts, p]))
    for name in meta["outlet_flow_m3_s"]:
        d = case / "postProcessing" / f"pAvg_{name}" / "100"
        d.mkdir(parents=True)
        b = int(name.split("b")[-1])
        (d / "surfaceFieldValue.dat").write_text(
            f"# t p\n100 {meta['ffr1d'][str(b)]['ffr_static'][-1] * meta['pa_mmhg'] * MMHG / meta['rho']}\n")
    summary = results_3d(case, figure=False)
    for bi, b in summary["branches"].items():
        assert b["distal_ffr3d"] == pytest.approx(meta["ffr1d"][bi]["ffr_static"][-1], rel=1e-6)
    step = update_outlet_flows(case)
    assert step["max_relative_flow_change"] < 1e-3  # consistent pressures -> fixed point
