from __future__ import annotations

import pytest
import torch

from vessel_code.parametric.model import ParametricVesselPredictor
from vessel_code.parametric.radius_refiner_loss import (
    INVISIBLE_STENOSIS,
    NATURAL_NEGATIVE,
    ORIGINAL,
    REMOVED,
    STRENGTHENED,
    VISIBLE_AUGMENTED,
    build_counterfactual_changed_mask_roi,
    compute_counterfactual_radius_refiner_loss,
    local_stenosis_dice_loss,
    whole_artery_dice_loss,
)


def _loss_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "radius_scale_mm": 1.0,
        "radius_refiner_final_radius_loss_weight": 0.0,
        "radius_refiner_intermediate_radius_loss_weight": 0.0,
        "radius_refiner_paired_difference_loss_weight": 0.0,
        "radius_refiner_outside_consistency_loss_weight": 0.0,
        "radius_refiner_conservative_residual_loss_weight": 0.0,
        "radius_refiner_whole_artery_dice_enabled": False,
        "radius_refiner_whole_artery_dice_loss_weight": 0.0,
        "radius_refiner_local_stenosis_dice_enabled": False,
        "radius_refiner_local_stenosis_dice_loss_weight": 0.0,
        "radius_refiner_stenosis_interval_point_weight": 5.0,
    }
    config.update(overrides)
    return config


def _vessel(radius: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    if radius.dim() != 3:
        raise ValueError("Test radius must have shape [B,M,N].")
    xyz = torch.zeros(*radius.shape, 3, dtype=radius.dtype)
    value = torch.cat((xyz, radius.unsqueeze(-1)), dim=-1)
    return value.requires_grad_(requires_grad)


def _output(
    final_radius: torch.Tensor,
    *,
    coarse_radius: torch.Tensor | None = None,
    requires_grad: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    final = _vessel(final_radius, requires_grad=requires_grad)
    coarse = _vessel(
        final_radius if coarse_radius is None else coarse_radius
    ).detach()
    stages = final.detach().clone().unsqueeze(1)
    return (
        {
            "decoded_vessel_mm": final,
            "radius_refiner_coarse_decoded_vessel_mm": coarse,
            "radius_refinement_stage_decoded_vessel_mm": stages,
        },
        final,
    )


def _batch(
    target_radius: torch.Tensor,
    *,
    group_type: str,
    variants: list[str],
    region: torch.Tensor | None = None,
) -> dict[str, object]:
    batch_size, num_branches, num_points = target_radius.shape
    if region is None:
        region = torch.zeros(
            batch_size, num_branches, num_points, dtype=torch.bool
        )
    return {
        "group_type": group_type,
        "counterfactual_variant": variants,
        "target_raw_vessel_mm": _vessel(target_radius),
        "target_point_valid_mask": torch.ones_like(
            target_radius, dtype=torch.bool
        ),
        "target_branch_exist": torch.ones(
            batch_size, num_branches, dtype=torch.float32
        ),
        "stenosis_region_point_mask": region,
    }


@pytest.mark.parametrize(
    ("group_type", "variants", "match"),
    [
        ("unknown", [ORIGINAL], "Unsupported radius-refiner group_type"),
        (
            VISIBLE_AUGMENTED,
            [ORIGINAL],
            "must contain exactly original",
        ),
        (
            NATURAL_NEGATIVE,
            [REMOVED],
            "must contain original only",
        ),
        (
            INVISIBLE_STENOSIS,
            [ORIGINAL, ORIGINAL],
            "must contain original only",
        ),
    ],
)
def test_radius_refiner_loss_rejects_invalid_group_membership(
    group_type: str,
    variants: list[str],
    match: str,
) -> None:
    batch_size = len(variants)
    target = torch.ones(batch_size, 1, 2)
    output, _ = _output(target)
    batch = _batch(
        target,
        group_type=group_type,
        variants=variants,
        region=torch.ones_like(target, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match=match):
        compute_counterfactual_radius_refiner_loss(
            output, batch, _loss_config()
        )


def test_triplet_paired_and_outside_losses_match_exact_deltas() -> None:
    # Variant order is intentionally non-canonical.  The first point is the
    # stenosis interval and the second is outside it.
    variants = [REMOVED, STRENGTHENED, ORIGINAL]
    target_radius = torch.tensor(
        [
            [[3.0, 4.0]],  # removed
            [[1.0, 4.0]],  # strengthened
            [[2.0, 4.0]],  # original
        ]
    )
    predicted_radius = torch.tensor(
        [
            [[4.0, 4.0]],  # removed
            [[1.5, 5.0]],  # strengthened
            [[2.0, 4.0]],  # original
        ]
    )
    region = torch.zeros_like(target_radius, dtype=torch.bool)
    region[:, :, 0] = True
    output, final = _output(predicted_radius, requires_grad=True)
    batch = _batch(
        target_radius,
        group_type=VISIBLE_AUGMENTED,
        variants=variants,
        region=region,
    )

    losses = compute_counterfactual_radius_refiner_loss(
        output,
        batch,
        _loss_config(
            radius_refiner_paired_difference_loss_weight=1.0,
            radius_refiner_outside_consistency_loss_weight=1.0,
        ),
    )

    # Inside Smooth-L1 errors for (strong,removed), (strong,original),
    # (original,removed) are 0.5, 0.5 and 1.0: [0.125, 0.125, 0.5].
    assert losses["radius_refiner_paired_difference_loss"].item() == pytest.approx(
        0.25
    )
    # Outside errors are 1.0, 1.0 and 0.0: [0.5, 0.5, 0.0].
    assert losses[
        "radius_refiner_outside_consistency_loss"
    ].item() == pytest.approx(1.0 / 3.0)
    assert losses["radius_refiner_paired_difference_mae_mm"].item() == pytest.approx(
        2.0 / 3.0
    )
    assert losses["radius_refiner_outside_difference_mae_mm"].item() == pytest.approx(
        2.0 / 3.0
    )
    assert losses["loss"].item() == pytest.approx(7.0 / 12.0)

    losses["loss"].backward()
    assert final.grad is not None
    assert torch.count_nonzero(final.grad[..., 3]) > 0


def test_invisible_stenosis_excludes_interval_target_and_preserves_coarse_radius() -> None:
    target_radius = torch.tensor([[[10.0, 3.0]]])
    predicted_radius = torch.tensor([[[7.0, 4.0]]])
    coarse_radius = torch.tensor([[[5.0, 4.0]]])
    region = torch.tensor([[[True, False]]])
    output, final = _output(
        predicted_radius,
        coarse_radius=coarse_radius,
        requires_grad=True,
    )
    batch = _batch(
        target_radius,
        group_type=INVISIBLE_STENOSIS,
        variants=[ORIGINAL],
        region=region,
    )
    config = _loss_config(
        radius_refiner_final_radius_loss_weight=1.0,
        radius_refiner_conservative_residual_loss_weight=1.0,
    )

    losses = compute_counterfactual_radius_refiner_loss(output, batch, config)

    # The direct target sees only point 1: SmoothL1(4 - 3) = 0.5.
    assert losses["radius_refiner_final_radius_loss"].item() == pytest.approx(0.5)
    # Point 0 instead preserves the coarse value: SmoothL1(7 - 5) = 1.5.
    assert losses[
        "radius_refiner_conservative_residual_loss"
    ].item() == pytest.approx(1.5)
    assert losses["loss"].item() == pytest.approx(2.0)

    # Changing the unobservable stenosis target must not change the objective.
    changed_target = target_radius.clone()
    changed_target[..., 0] = 100.0
    changed_batch = _batch(
        changed_target,
        group_type=INVISIBLE_STENOSIS,
        variants=[ORIGINAL],
        region=region,
    )
    changed_losses = compute_counterfactual_radius_refiner_loss(
        output, changed_batch, config
    )
    torch.testing.assert_close(changed_losses["loss"], losses["loss"])

    losses["loss"].backward()
    assert final.grad is not None
    assert final.grad[0, 0, 0, 3].abs() > 0  # conservative residual
    assert final.grad[0, 0, 1, 3].abs() > 0  # direct radius target


def test_changed_mask_roi_dilation_and_visible_view_filtering() -> None:
    variants = [ORIGINAL, REMOVED, STRENGTHENED]
    target_images = torch.zeros(3, 3, 5, 5)
    strengthened = variants.index(STRENGTHENED)
    target_images[strengthened, 0, 2, 2] = 1.0
    target_images[strengthened, 1, 0, 0] = 1.0

    roi = build_counterfactual_changed_mask_roi(
        target_images=target_images,
        variants=variants,
        dilation_px=1,
        threshold=0.5,
    )

    assert roi.shape == (3, 5, 5)
    assert roi[0, 1:4, 1:4].all()
    assert roi[0].sum().item() == 9
    assert roi[1, :2, :2].all()
    assert roi[1].sum().item() == 4
    assert not roi[2].any()

    predicted_masks = target_images.clone()
    # Make the second view deliberately wrong for every triplet member.  It
    # must not affect visible-only Dice because Stage 3.1 marks only view 0.
    predicted_masks[:, 1] = 1.0 - predicted_masks[:, 1]
    view_mask = torch.ones(3, 3, dtype=torch.bool)
    visible = torch.tensor([[True, False, False]]).expand(3, -1)

    visible_loss, visible_score, visible_count = local_stenosis_dice_loss(
        predicted_masks=predicted_masks,
        target_images=target_images,
        view_mask=view_mask,
        group_visible_view_mask=visible,
        variants=variants,
        dilation_px=1,
        threshold=0.5,
        visible_views_only=True,
    )
    assert visible_count.item() == 3
    assert visible_loss.item() == pytest.approx(0.0)
    assert visible_score.item() == pytest.approx(1.0)

    all_loss, all_score, all_count = local_stenosis_dice_loss(
        predicted_masks=predicted_masks,
        target_images=target_images,
        view_mask=view_mask,
        group_visible_view_mask=visible,
        variants=variants,
        dilation_px=1,
        threshold=0.5,
        visible_views_only=False,
    )
    assert all_count.item() == 6
    assert all_loss.item() > 0.49
    assert all_score.item() < 0.51


def test_whole_artery_dice_uses_complete_masks_and_valid_views_only() -> None:
    target_images = torch.zeros(1, 2, 3, 3)
    target_images[0, 0, 1, 1] = 1.0
    target_images[0, 1, 0, 0] = 1.0
    predicted_masks = target_images.clone()
    predicted_masks[0, 1] = 1.0 - predicted_masks[0, 1]

    valid_only_loss, valid_only_score = whole_artery_dice_loss(
        predicted_masks=predicted_masks,
        target_images=target_images,
        view_mask=torch.tensor([[True, False]]),
        threshold=0.5,
    )
    assert valid_only_loss.item() == pytest.approx(0.0)
    assert valid_only_score.item() == pytest.approx(1.0)

    all_loss, all_score = whole_artery_dice_loss(
        predicted_masks=predicted_masks,
        target_images=target_images,
        view_mask=torch.ones(1, 2, dtype=torch.bool),
        threshold=0.5,
    )
    assert all_loss.item() > 0.49
    assert all_score.item() < 0.51


def test_whole_artery_dice_is_added_to_the_dedicated_objective() -> None:
    target_radius = torch.ones(1, 1, 2)
    output, _ = _output(target_radius)
    rendered = torch.tensor(
        [[[[0.5, 0.5]]]], dtype=torch.float32, requires_grad=True
    )
    output["radius_refiner_final_rendered_masks"] = rendered
    batch = _batch(
        target_radius,
        group_type=NATURAL_NEGATIVE,
        variants=[ORIGINAL],
    )
    batch.update(
        images=torch.tensor([[[[1.0, 0.0]]]]),
        view_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    losses = compute_counterfactual_radius_refiner_loss(
        output,
        batch,
        _loss_config(
            radius_refiner_whole_artery_dice_enabled=True,
            radius_refiner_whole_artery_dice_loss_weight=0.2,
        ),
    )

    assert losses["radius_refiner_whole_artery_dice_loss"].item() == pytest.approx(
        0.5
    )
    assert losses[
        "radius_refiner_whole_artery_dice_weighted_loss"
    ].item() == pytest.approx(0.1)
    assert losses["loss"].item() == pytest.approx(0.1)
    losses["loss"].backward()
    assert rendered.grad is not None
    assert torch.count_nonzero(rendered.grad) > 0


def test_visible_metadata_cannot_silently_produce_an_empty_local_dice_roi() -> None:
    variants = [ORIGINAL, REMOVED, STRENGTHENED]
    target_radius = torch.ones(3, 1, 2)
    region = torch.ones_like(target_radius, dtype=torch.bool)
    output, _ = _output(target_radius)
    output["radius_refiner_final_rendered_masks"] = torch.zeros(3, 1, 5, 5)
    batch = _batch(
        target_radius,
        group_type=VISIBLE_AUGMENTED,
        variants=variants,
        region=region,
    )
    batch.update(
        images=torch.zeros(3, 1, 5, 5),
        view_mask=torch.ones(3, 1, dtype=torch.bool),
        stenosis_view_visible_mask=torch.ones(3, 1, dtype=torch.bool),
    )

    with pytest.raises(RuntimeError, match="empty local Dice ROI"):
        compute_counterfactual_radius_refiner_loss(
            output,
            batch,
            _loss_config(
                radius_refiner_local_stenosis_dice_enabled=True,
                radius_refiner_local_stenosis_dice_loss_weight=0.25,
            ),
        )


def test_direct_radius_loss_backpropagates_to_every_final_radius() -> None:
    target_radius = torch.tensor([[[1.0, 1.0, 1.0]]])
    predicted_radius = torch.tensor([[[1.25, 2.0, 0.5]]])
    output, final = _output(predicted_radius, requires_grad=True)
    batch = _batch(
        target_radius,
        group_type=NATURAL_NEGATIVE,
        variants=[ORIGINAL],
    )

    losses = compute_counterfactual_radius_refiner_loss(
        output,
        batch,
        _loss_config(radius_refiner_final_radius_loss_weight=1.0),
    )
    losses["loss"].backward()

    assert final.grad is not None
    assert torch.all(final.grad[..., 3] != 0)
    assert torch.count_nonzero(final.grad[..., :3]) == 0


def test_direct_radius_loss_masks_absent_branch() -> None:
    target_radius = torch.ones(1, 2, 3)
    output, final = _output(target_radius + 1.0, requires_grad=True)
    batch = _batch(
        target_radius,
        group_type=NATURAL_NEGATIVE,
        variants=[ORIGINAL],
    )
    batch["target_branch_exist"] = torch.tensor([[1.0, 0.0]])

    losses = compute_counterfactual_radius_refiner_loss(
        output,
        batch,
        _loss_config(radius_refiner_final_radius_loss_weight=1.0),
    )
    losses["loss"].backward()

    assert final.grad is not None
    assert torch.all(final.grad[:, 0, :, 3] != 0)
    assert torch.count_nonzero(final.grad[:, 1]) == 0


def test_radius_refiner_canonicalizes_triplet_geometry_but_retains_radii() -> None:
    torch.manual_seed(17)
    model = ParametricVesselPredictor(
        feature_backbone="vggt",
        num_points=11,
        num_branches=2,
        num_control_points=4,
        num_radius_coefficients=1,
        num_lesions=0,
        radius_prediction_mode="raw",
        view_feat_dim=4,
        model_dim=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_attention_heads=4,
        mlp_hidden_dim=32,
        dropout=0.0,
        vggt_token_dim=8,
        use_bspline_control_refiner=True,
        bspline_refiner_num_stages=1,
        bspline_refiner_evidence_hidden_dim=32,
        bspline_refiner_patch_size=3,
        bspline_refiner_use_learned_image_features=False,
        bspline_refiner_use_distance_transform=False,
        bspline_refiner_image_size=16,
        bspline_refiner_residual_scale_mm=2.0,
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=1,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_radial_subsamples=1,
        radius_refiner_render_axial_subsamples=0,
    ).eval()
    inputs = {
        "views": torch.randn(3, 3, 4),
        "view_mask": torch.ones(3, 3, dtype=torch.bool),
        "image_features": torch.randn(3, 3, 5, 8),
        "images": torch.rand(3, 3, 16, 16),
    }
    # Member 1 represents original in [removed, original, strengthened].
    canonical_index = 1

    with torch.no_grad():
        for parameter in model.branch_exist_head.parameters():
            parameter.zero_()
        model.branch_exist_head[-1].bias.fill_(20.0)
        ordinary = model(**inputs)
        ordinary_xyz = ordinary["decoded_vessel_mm"][..., :3]
        reference_xyz = ordinary_xyz[
            canonical_index : canonical_index + 1
        ]
        ordinary_p95 = torch.quantile(
            torch.linalg.vector_norm(ordinary_xyz - reference_xyz, dim=-1)
            .flatten(start_dim=1),
            0.95,
            dim=1,
        )
        assert ordinary_p95.max().item() > 0.0
        threshold = ordinary_p95.max().item() * 0.5

        output = model(
            **inputs,
            radius_refiner_canonical_geometry_member_index=canonical_index,
            radius_refiner_canonical_geometry_p95_threshold_mm=threshold,
        )

    precanonical = output["radius_refiner_precanonical_decoded_vessel_mm"]
    expected_p95 = torch.quantile(
        torch.linalg.vector_norm(
            precanonical[..., :3]
            - precanonical[canonical_index : canonical_index + 1, ..., :3],
            dim=-1,
        ).flatten(start_dim=1),
        0.95,
        dim=1,
    )
    torch.testing.assert_close(
        output["radius_refiner_precanonical_geometry_p95_mm"],
        expected_p95,
    )
    assert expected_p95.max().item() > threshold
    assert torch.equal(
        output["radius_refiner_geometry_canonicalization_applied"],
        torch.ones(3),
    )
    assert torch.equal(
        output["radius_refiner_canonical_geometry_member_index"],
        torch.full((3,), canonical_index, dtype=torch.long),
    )

    expected_xyz = precanonical[
        canonical_index : canonical_index + 1, ..., :3
    ].expand(3, -1, -1, -1)
    torch.testing.assert_close(
        output["radius_refiner_coarse_decoded_vessel_mm"][..., :3],
        expected_xyz,
    )
    torch.testing.assert_close(output["decoded_vessel_mm"][..., :3], expected_xyz)
    # Canonicalization is geometry-only: every member keeps its own coarse
    # radius instead of copying the canonical member's radius profile.
    torch.testing.assert_close(
        output["radius_refiner_coarse_decoded_vessel_mm"][..., 3],
        precanonical[..., 3],
    )
