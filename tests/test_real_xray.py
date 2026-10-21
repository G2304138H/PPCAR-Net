"""Real X-ray keyframe input: loader, view selection and reprojection geometry (numpy only, no model)."""
from pathlib import Path

import numpy as np
import pytest

from vessel_code.geometry.projection import view_features_from_angles
from vessel_code.real_xray import render
from vessel_code.real_xray.data import load_case, select_input_views

ROOT = Path(__file__).resolve().parents[1]


def _write_case(path, artery="lca", n=3, shift=(9, -6)):
    """A case built from the bundled example: its first n masks, a known 2D shift on the last view."""
    ex = np.load(ROOT / f"examples/{artery}/{artery}_0457.npz")
    images = ex["images"][:n].copy()
    images[-1] = render.shift_mask(images[-1], *shift)
    theta, phi = ex["theta_deg"][:n], ex["phi_deg"][:n]
    eye = np.tile(np.eye(3), (n, 1, 1))
    np.savez_compressed(
        path, format_version=np.int32(1), input_format=np.asarray("real_xray_keyframes"),
        images=images, raw_images=images * 0.5, theta_deg=theta, phi_deg=phi,
        view_features=view_features_from_angles(theta, phi),
        primary_angle_deg=(90 - theta + 180) % 360 - 180, secondary_angle_deg=90 - phi,
        clinical_views=ex["clinical_views"][:n], angle_group=np.arange(n, dtype=np.int32),
        raw_full=(images * 255).astype(np.uint8), label_full=(images > 0.5).astype(np.uint8),
        full_shape=np.full((n, 2), 256, np.int32), model_to_full=eye,
        keyframe_vessel_px=images.reshape(n, -1).sum(1).astype(np.int32),
        vessel_type=np.asarray(artery), case_name=np.asarray(path.stem), seg_method=np.asarray("test"),
        image_dim=np.int32(256), sid=np.float32(ex["sid"]), imager_pixel_spacing=np.float32(ex["imager_pixel_spacing"]))
    return ex


@pytest.mark.parametrize("artery", ["lca", "rca"])
def test_reprojection_matches_inputs_and_alignment_recovers_shift(tmp_path, artery):
    ex = _write_case(tmp_path / "case.npz", artery)
    case = load_case(tmp_path / "case.npz", artery)
    vessel = ex["raw_vessel_code_mm"].copy()
    active = vessel[..., 3].max(1) > 0
    vessel[..., :3] -= ex["projection_center_offset"] * float(ex["input_scale_to_mm"])
    masks = render.reproject_masks(vessel, active, case.theta_deg, case.phi_deg, case.camera())
    assert render.dice(masks[0], case.images[0]) > 0.95
    assert render.dice(masks[-1], case.images[-1]) < 0.9            # shifted view
    d = render.translation_to_align(masks[-1], case.images[-1])
    assert d == (9, -6)
    assert render.dice(render.shift_mask(masks[-1], *d), case.images[-1]) > 0.95


def test_model_to_full_roundtrip():
    M = np.array([[-1 / 0.5, 0, 400.0], [0, -1 / 0.5, 380.0], [0, 0, 1]])
    m = np.zeros((256, 256), np.float32)
    m[100:110, 50:60] = 1
    full = render.model_to_full(m, M, (512, 512))
    r, c = np.nonzero(full)
    centre_model = np.array([104.5, 54.5, 1.0])
    expect = M @ centre_model
    assert abs(r.mean() - expect[0]) < 1.5 and abs(c.mean() - expect[1]) < 1.5


def test_view_selection_prefers_separated_directions(tmp_path):
    _write_case(tmp_path / "case.npz", "lca", n=7, shift=(0, 0))
    case = load_case(tmp_path / "case.npz", "lca")
    two = select_input_views(case, 2)
    assert len(set(two)) == 2
    assert select_input_views(case, 3, "first") == [0, 1, 2]
    with pytest.raises(ValueError):
        load_case(tmp_path / "case.npz", "rca")
