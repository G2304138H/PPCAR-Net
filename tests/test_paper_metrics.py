from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vessel_code.parametric.paper_metrics import (
    PAPER_METRIC_ISOTROPIC_SPACING_MM,
    GroundTruthVolume,
    cldice_metrics_3d,
    connected_component_count_3d,
    connected_component_metrics_3d,
    compute_paper_mask_metrics,
    dice_similarity_3d,
    isotropic_grid_from_source,
    load_ground_truth_volume,
    rasterize_vessel_to_mask,
    resample_binary_volume_nearest,
    resample_binary_volume_nearest_affine,
    resolve_ground_truth_volume_paths,
    resolve_paper_metric_options,
    restore_absolute_vessel_coordinates,
    save_paper_mask_artifact,
    skeletonize_binary_volume_3d,
    structural_similarity_3d,
    symmetric_chamfer_distance,
    target_grid_index_to_world_affine,
)


def _write_volume(
    path: Path,
    *,
    shape: tuple[int, int, int] = (5, 5, 5),
    source_nii: str = "/dataset/7.label.nii.gz",
    artery_type: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.zeros(shape, dtype=np.uint8)
    volume[tuple(size // 2 for size in shape)] = 1
    values: dict[str, np.ndarray] = {
        "vol": volume,
        "spacing": np.asarray([0.5, 0.75, 1.25], dtype=np.float32),
        "source_nii": np.asarray(source_nii),
    }
    if artery_type is not None:
        values["artery_type"] = np.asarray(artery_type)
    np.savez_compressed(path, **values)


def test_load_attached_style_npz_uses_spacing_source_and_documented_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artery_mask.npz"
    _write_volume(path, shape=(3, 4, 5), source_nii="/ct/0007.label.nii.gz")
    options = resolve_paper_metric_options(
        {"paper_metric_ground_truth_path": str(path)},
        config_path=None,
    )

    ground_truth = load_ground_truth_volume(path, options)

    assert ground_truth.mask.shape == (3, 4, 5)
    assert ground_truth.mask.dtype == np.bool_
    assert int(ground_truth.mask.sum()) == 1
    np.testing.assert_allclose(ground_truth.spacing_mm, [0.5, 0.75, 1.25])
    np.testing.assert_allclose(
        ground_truth.index_to_world_affine,
        np.diag([0.5, 0.75, 1.25, 1.0]),
    )
    assert ground_truth.source_nii == "/ct/0007.label.nii.gz"
    assert any("assumed origin" in value for value in ground_truth.assumptions)
    assert any("identity direction" in value for value in ground_truth.assumptions)


def test_explicit_ground_truth_path_resolves_selected_case(tmp_path: Path) -> None:
    path = tmp_path / "artery_mask.npz"
    _write_volume(path, source_nii="/ct/subject_0007.label.nii.gz")
    options = resolve_paper_metric_options(
        {"paper_metric_ground_truth_path": str(path)},
        config_path=None,
    )

    assert resolve_ground_truth_volume_paths(
        options,
        ["0007"],
        artery_type="RCA",
    ) == {"7": path.resolve()}


def test_ground_truth_directory_resolution_is_anatomy_aware(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ground_truth"
    rca = root / "RCA" / "case_0007.npz"
    lca = root / "LCA" / "case_0007.npz"
    _write_volume(rca, artery_type="RCA")
    _write_volume(lca, artery_type="LCA")
    options = resolve_paper_metric_options(
        {"paper_metric_ground_truth_dir": str(root)},
        config_path=None,
    )

    assert resolve_ground_truth_volume_paths(
        options,
        ["7"],
        artery_type="RCA",
    ) == {"7": rca.resolve()}


def test_restore_absolute_coordinates_uses_exact_offset_for_centered_frame(
) -> None:
    vessel = np.asarray(
        [[[1.0, 2.0, 3.0, 0.8], [4.0, 5.0, 6.0, 0.6]]],
        dtype=np.float32,
    )
    original = vessel.copy()
    offset = np.asarray([10.0, -20.0, 30.0], dtype=np.float32)

    centered = restore_absolute_vessel_coordinates(
        vessel,
        target_coordinate_frame="projection_centered",
        centering_offset_mm=offset,
    )
    absolute = restore_absolute_vessel_coordinates(
        vessel,
        target_coordinate_frame="absolute_world",
        centering_offset_mm=offset,
    )

    np.testing.assert_allclose(centered[..., :3], vessel[..., :3] + offset)
    np.testing.assert_array_equal(centered[..., 3], vessel[..., 3])
    np.testing.assert_array_equal(absolute, vessel)
    np.testing.assert_array_equal(vessel, original)


def test_endpoint_aligned_resampling_and_affine_preserve_fov_endpoints() -> None:
    source = np.zeros((3, 4, 5), dtype=np.bool_)
    source[0, 0, 0] = True
    source[-1, -1, -1] = True

    resampled = resample_binary_volume_nearest(source, (5, 7, 9))
    target_affine = target_grid_index_to_world_affine(
        source_shape=source.shape,
        target_shape=resampled.shape,
        source_index_to_world_affine=np.asarray(
            [
                [2.0, 0.0, 0.0, 11.0],
                [0.0, 3.0, 0.0, 12.0],
                [0.0, 0.0, 4.0, 13.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    assert resampled.shape == (5, 7, 9)
    assert resampled[0, 0, 0]
    assert resampled[-1, -1, -1]
    np.testing.assert_allclose(np.diag(target_affine)[:3], [1.0, 1.5, 2.0])
    np.testing.assert_allclose(target_affine[:3, 3], [11.0, 12.0, 13.0])
    np.testing.assert_allclose(
        target_affine
        @ np.asarray([4.0, 6.0, 8.0, 1.0]),
        np.asarray([15.0, 21.0, 29.0, 1.0]),
    )


def test_isotropic_grid_and_affine_resampling_use_exact_half_mm_spacing() -> None:
    source = np.zeros((3, 4, 5), dtype=np.bool_)
    source[1, 2, 3] = True
    source_affine = np.asarray(
        [
            [1.0, 0.0, 0.0, 11.0],
            [0.0, 1.5, 0.0, 12.0],
            [0.0, 0.0, 2.0, 13.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    target_shape, target_affine = isotropic_grid_from_source(
        source_shape=source.shape,
        source_index_to_world_affine=source_affine,
    )
    resampled = resample_binary_volume_nearest_affine(
        source,
        source_index_to_world_affine=source_affine,
        output_shape=target_shape,
        output_index_to_world_affine=target_affine,
    )

    assert target_shape == (5, 10, 17)
    np.testing.assert_allclose(
        np.linalg.norm(target_affine[:3, :3], axis=0),
        [PAPER_METRIC_ISOTROPIC_SPACING_MM] * 3,
    )
    np.testing.assert_allclose(target_affine[:3, 3], [11.0, 12.0, 13.0])
    assert resampled[2, 6, 12]


def test_rasterization_builds_a_filled_polyline_tube() -> None:
    vessel = np.asarray(
        [[[2.0, 4.0, 4.0, 1.1], [6.0, 4.0, 4.0, 1.1]]],
        dtype=np.float32,
    )

    mask, record = rasterize_vessel_to_mask(
        vessel,
        np.asarray([True]),
        source_volume_shape=(9, 9, 9),
        source_index_to_world_affine=np.eye(4),
        output_shape=(9, 9, 9),
    )

    assert mask[4, 4, 4]
    assert mask[4, 5, 4]
    assert not mask[4, 6, 4]
    assert mask[2, 4, 4]
    assert mask[6, 4, 4]
    assert record["num_active_branches"] == 1
    assert record["centerline_points_inside_volume_fraction"] == 1.0


def test_dice_is_one_for_identical_and_zero_for_disjoint_masks() -> None:
    first = np.zeros((4, 4, 4), dtype=np.bool_)
    second = np.zeros_like(first)
    first[0, 0, 0] = True
    second[-1, -1, -1] = True

    assert dice_similarity_3d(first, first) == 1.0
    assert dice_similarity_3d(first, second) == 0.0


def test_connected_components_use_26_neighbour_foreground_connectivity() -> None:
    target = np.zeros((7, 7, 7), dtype=np.bool_)
    target[1, 1, 1] = True
    target[2, 2, 2] = True
    predicted = target.copy()
    predicted[5, 5, 5] = True

    metrics = connected_component_metrics_3d(predicted, target)

    assert connected_component_count_3d(np.zeros_like(target)) == 0
    assert connected_component_count_3d(target) == 1
    assert metrics == {
        "predicted_connected_components_3d": 2,
        "ground_truth_connected_components_3d": 1,
        "connected_component_count_absolute_error_3d": 1,
        "predicted_disconnected_3d": 1,
        "ground_truth_disconnected_3d": 0,
    }


def test_symmetric_chamfer_is_sum_of_bidirectional_mean_l2_distances() -> None:
    predicted = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
    )
    target = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )

    assert symmetric_chamfer_distance(target, target) == pytest.approx(0.0)
    # pred -> GT is (0 + 1) / 2 and GT -> pred is (0 + 1) / 2.
    assert symmetric_chamfer_distance(
        predicted, target, chunk_size=1
    ) == pytest.approx(1.0)


def test_symmetric_chamfer_rejects_empty_or_nonfinite_point_sets() -> None:
    points = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        symmetric_chamfer_distance(np.empty((0, 3)), points)
    with pytest.raises(ValueError, match="finite"):
        symmetric_chamfer_distance(
            np.asarray([[np.nan, 0.0, 0.0]]), points
        )


def test_cldice_is_one_for_identical_and_zero_for_disjoint_vessels() -> None:
    first = np.zeros((9, 9, 9), dtype=np.bool_)
    second = np.zeros_like(first)
    first[2:7, 4, 4] = True
    second[2:7, 1, 1] = True

    identical = cldice_metrics_3d(first, first)
    disjoint = cldice_metrics_3d(first, second)

    assert identical["cldice_3d"] == pytest.approx(1.0)
    assert identical["cldice_loss_3d"] == pytest.approx(0.0)
    assert disjoint["cldice_3d"] == pytest.approx(0.0)
    assert disjoint["cldice_loss_3d"] == pytest.approx(1.0)
    assert skeletonize_binary_volume_3d(first).dtype == np.bool_


def test_3d_ssim_is_one_for_identical_volume() -> None:
    mask = np.zeros((7, 7, 7), dtype=np.bool_)
    mask[2:5, 1:6, 3:5] = True

    assert structural_similarity_3d(mask, mask, window_size=3) == pytest.approx(
        1.0
    )


def test_compute_and_save_paper_metric_masks_and_offset_metadata(
    tmp_path: Path,
) -> None:
    native_mask = np.zeros((5, 5, 5), dtype=np.bool_)
    native_mask[1:4, 2, 2] = True
    ground_truth = GroundTruthVolume(
        mask=native_mask,
        spacing_mm=np.ones((3,), dtype=np.float64),
        index_to_world_affine=np.eye(4, dtype=np.float64),
        source_path=tmp_path / "case_7.npz",
        source_nii="/ct/7.label.nii.gz",
        assumptions=("synthetic test geometry",),
    )
    vessel = np.asarray(
        [[[1.0, 2.0, 2.0, 0.55], [3.0, 2.0, 2.0, 0.55]]],
        dtype=np.float32,
    )

    result = compute_paper_mask_metrics(
        vessel_absolute_mm=vessel,
        branch_exists=np.asarray([True]),
        ground_truth=ground_truth,
        ssim_window_size=3,
    )
    result["metrics"]["centerline_chamfer_distance_mm"] = 0.25
    artifact = tmp_path / "masks" / "case_7.npz"
    offset = np.asarray([9.0, 8.0, 7.0], dtype=np.float32)
    save_paper_mask_artifact(
        artifact,
        result=result,
        ground_truth=ground_truth,
        case_id="7",
        split="test",
        prediction_role="final",
        applied_centering_offset_mm=offset,
        centering_offset_reversed=True,
        centering_source="paired_vessel",
        stored_projection_center_offset_mm=np.asarray([1.0, 2.0, 3.0]),
        stored_projection_center_offset_valid=True,
        evaluation_num_views=4,
        side_branch_snapping_applied=True,
        side_branch_translation_mm=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]]
        ),
    )

    assert result["predicted_mask"].shape == (128, 128, 128)
    assert result["ground_truth_mask"].shape == (128, 128, 128)
    assert result["predicted_mask"].dtype == np.bool_
    assert result["ground_truth_mask"].dtype == np.bool_
    assert np.isfinite(result["metrics"]["paper_mask_ssim_3d"])
    assert 0.0 <= result["metrics"]["paper_mask_cldice_3d"] <= 1.0
    assert result["metrics"][
        "paper_mask_predicted_connected_components_3d"
    ] == 1
    assert result["metrics"]["paper_mask_predicted_disconnected_3d"] == 0
    assert result["metrics"][
        "paper_mask_predicted_connected_components_3d_0p5mm"
    ] == 1
    assert result["metrics"]["paper_mask_cldice_loss_3d"] == pytest.approx(
        1.0 - result["metrics"]["paper_mask_cldice_3d"]
    )
    assert result["predicted_centerline_mask"].shape == (128, 128, 128)
    assert result["ground_truth_centerline_mask"].shape == (128, 128, 128)
    assert result["predicted_mask_0p5mm"].shape == (9, 9, 9)
    assert result["ground_truth_mask_0p5mm"].shape == (9, 9, 9)
    assert 0.0 <= result["metrics"]["paper_mask_dice_3d_0p5mm"] <= 1.0
    assert 0.0 <= result["metrics"]["paper_mask_cldice_3d_0p5mm"] <= 1.0
    assert result["metrics"][
        "paper_mask_cldice_loss_3d_0p5mm"
    ] == pytest.approx(
        1.0 - result["metrics"]["paper_mask_cldice_3d_0p5mm"]
    )
    np.testing.assert_allclose(
        result["rasterization_0p5mm"]["target_spacing_mm"],
        [0.5, 0.5, 0.5],
    )
    with np.load(artifact, allow_pickle=False) as payload:
        assert payload["predicted_mask"].shape == (128, 128, 128)
        assert payload["ground_truth_mask"].shape == (128, 128, 128)
        assert payload["predicted_mask"].dtype == np.uint8
        assert payload["ground_truth_mask"].dtype == np.uint8
        assert payload["predicted_centerline_mask"].dtype == np.uint8
        assert payload["ground_truth_centerline_mask"].dtype == np.uint8
        assert payload["predicted_mask_0p5mm"].shape == (9, 9, 9)
        assert payload["ground_truth_mask_0p5mm"].shape == (9, 9, 9)
        assert payload["predicted_centerline_mask_0p5mm"].dtype == np.uint8
        assert payload["ground_truth_centerline_mask_0p5mm"].dtype == np.uint8
        np.testing.assert_allclose(
            payload["target_spacing_mm_0p5mm"], [0.5, 0.5, 0.5]
        )
        np.testing.assert_allclose(payload["applied_centering_offset_mm"], offset)
        assert bool(payload["centering_offset_reversed"].reshape(()))
        assert str(payload["centering_source"].reshape(())) == "paired_vessel"
        assert int(payload["evaluation_num_views"].reshape(())) == 4
        np.testing.assert_allclose(
            payload["stored_projection_center_offset_mm"], [1.0, 2.0, 3.0]
        )
        assert bool(payload["side_branch_snapping_applied"].reshape(()))
        np.testing.assert_allclose(
            payload["side_branch_translation_mm"],
            [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        )
        assert 0.0 <= float(payload["paper_mask_cldice_3d"]) <= 1.0
        assert int(
            payload["paper_mask_predicted_connected_components_3d"]
        ) == 1
        assert int(payload["paper_mask_predicted_disconnected_3d"]) == 0
        assert int(
            payload[
                "paper_mask_predicted_connected_components_3d_0p5mm"
            ]
        ) == 1
        assert 0.0 <= float(payload["paper_mask_dice_3d_0p5mm"]) <= 1.0
        assert 0.0 <= float(payload["paper_mask_cldice_3d_0p5mm"]) <= 1.0
        assert float(payload["centerline_chamfer_distance_mm"]) == pytest.approx(
            0.25
        )


def test_compute_rejects_ground_truth_erased_by_128_grid() -> None:
    native_mask = np.zeros((256, 2, 2), dtype=np.bool_)
    native_mask[1, 0, 0] = True
    ground_truth = GroundTruthVolume(
        mask=native_mask,
        spacing_mm=np.ones((3,), dtype=np.float64),
        index_to_world_affine=np.eye(4, dtype=np.float64),
        source_path=Path("sparse_case.npz"),
        source_nii=None,
        assumptions=(),
    )
    vessel = np.asarray(
        [[[0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5]]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="removed all ground-truth"):
        compute_paper_mask_metrics(
            vessel_absolute_mm=vessel,
            branch_exists=np.asarray([True]),
            ground_truth=ground_truth,
            ssim_window_size=3,
        )
