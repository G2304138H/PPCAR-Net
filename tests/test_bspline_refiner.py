from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vessel_code.parametric.centerline_probability import (
    CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMA_VERSION,
    CenterlineProbabilityPredictor,
)
from vessel_code.parametric.loss import (
    _bspline_refined_branch_existence_loss,
    compute_parametric_loss,
)
from vessel_code.parametric.model import (
    AbsoluteParallelParametricVesselPredictor,
    LegacyParallelParametricVesselPredictor,
    ParametricVesselPredictor,
    build_model_from_config,
    infer_checkpoint_decoder_architecture,
)
from vessel_code.parametric.train import (
    _coarse_refiner_monitor_output,
    configure_bspline_refiner_only_training,
    configure_pretrained_centerline_predictor,
    load_config,
    refiner_branch_mask_for_batch,
    set_parametric_model_training_mode,
    validate_geometry_refiner_coarse_config,
)


def _small_model_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "feature_backbone": "vggt",
        "num_points": 11,
        "num_branches": 2,
        "num_control_points": 4,
        "num_radius_coefficients": 1,
        "num_lesions": 0,
        "radius_prediction_mode": "raw",
        "view_feat_dim": 4,
        "model_dim": 16,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_attention_heads": 4,
        "mlp_hidden_dim": 32,
        "dropout": 0.0,
        "vggt_token_dim": 8,
        "use_bspline_control_refiner": True,
        "bspline_refiner_num_stages": 3,
        "bspline_refiner_evidence_hidden_dim": 32,
        "bspline_refiner_patch_size": 3,
        "bspline_refiner_use_learned_image_features": False,
        "bspline_refiner_use_distance_transform": False,
        "bspline_refiner_image_size": 16,
        "bspline_refiner_residual_scale_mm": 2.0,
    }
    kwargs.update(overrides)
    return kwargs


def _small_model(**overrides: object) -> ParametricVesselPredictor:
    return ParametricVesselPredictor(**_small_model_kwargs(**overrides))


def _small_legacy_model(
    **overrides: object,
) -> LegacyParallelParametricVesselPredictor:
    return LegacyParallelParametricVesselPredictor(
        **_small_model_kwargs(**overrides)
    )


def _small_absolute_model(
    **overrides: object,
) -> AbsoluteParallelParametricVesselPredictor:
    return AbsoluteParallelParametricVesselPredictor(
        **_small_model_kwargs(**overrides)
    )


def _forward(model: ParametricVesselPredictor) -> dict[str, torch.Tensor]:
    return model(
        views=torch.randn(2, 3, 4),
        view_mask=torch.ones(2, 3, dtype=torch.bool),
        image_features=torch.randn(2, 3, 5, 8),
        images=torch.rand(2, 3, 16, 16),
    )


def test_bspline_refiner_returns_coarse_and_each_refined_stage() -> None:
    model = _small_model()
    output = _forward(model)

    assert output["coarse_centerline_parameters_mm"].shape == (2, 2, 4, 3)
    assert output["refinement_stage_centerline_parameters_mm"].shape == (
        2,
        3,
        2,
        4,
        3,
    )
    assert output["refinement_stage_decoded_vessel_mm"].shape == (
        2,
        3,
        2,
        11,
        4,
    )
    assert output["bspline_refinement_residual_mm"].shape == (2, 3, 2, 4, 3)
    assert torch.allclose(
        output["decoded_vessel_mm"],
        output["refinement_stage_decoded_vessel_mm"][:, -1],
    )
    # Zero initialization makes refinement begin as an identity update.
    assert torch.allclose(
        output["centerline_parameters_mm"],
        output["coarse_centerline_parameters_mm"],
    )
    coarse_monitor_output = _coarse_refiner_monitor_output(output)
    assert coarse_monitor_output is not None
    assert torch.equal(
        coarse_monitor_output["decoded_vessel_mm"],
        output["coarse_decoded_vessel_mm"],
    )
    assert not any(
        key.startswith("coarse_") for key in coarse_monitor_output
    )


def test_branch_existence_refinement_is_disabled_by_default() -> None:
    model = _small_model()
    output = _forward(model)

    assert model.bspline_refiner_refine_branch_existence is False
    assert not any(
        "branch_existence_residual_head" in key
        for key in model.state_dict()
    )
    assert "coarse_branch_exist_logits" not in output
    assert "refinement_stage_branch_exist_logits" not in output


@pytest.mark.parametrize(
    "factory",
    (_small_model, _small_legacy_model, _small_absolute_model),
)
def test_branch_existence_refinement_returns_coarse_stage_and_final_logits(
    factory,
) -> None:
    model = factory(bspline_refiner_refine_branch_existence=True)
    output = _forward(model)

    assert output["coarse_branch_exist_logits"].shape == (2, 2)
    assert output["refinement_stage_branch_exist_logits"].shape == (2, 3, 2)
    assert output["refinement_stage_branch_exist_probs"].shape == (2, 3, 2)
    assert output[
        "bspline_refinement_branch_exist_residual_logits"
    ].shape == (2, 3, 2)
    torch.testing.assert_close(
        output["branch_exist_logits"],
        output["coarse_branch_exist_logits"],
    )
    torch.testing.assert_close(
        output["branch_exist_probs"],
        output["coarse_branch_exist_probs"],
    )
    assert torch.equal(output["branch_exist_probs"][:, 0], torch.ones(2))

    coarse_monitor_output = _coarse_refiner_monitor_output(output)
    assert coarse_monitor_output is not None
    assert torch.equal(
        coarse_monitor_output["branch_exist_logits"],
        output["coarse_branch_exist_logits"],
    )
    assert torch.equal(
        coarse_monitor_output["branch_exist_probs"],
        output["coarse_branch_exist_probs"],
    )


def test_branch_existence_refinement_only_removes_coarse_positive_side_slots() -> None:
    model = _small_model(
        bspline_refiner_refine_branch_existence=True,
        bspline_refiner_branch_existence_max_logit_decrease=4.0,
    )
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.branch_existence_residual_head is not None
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
        existence_head = (
            model.bspline_control_refiner.branch_existence_residual_head[-1]
        )
        existence_head.weight.zero_()
        existence_head.bias.fill_(-1.0)
        model.bspline_control_refiner.residual_head[-1].weight.zero_()
        model.bspline_control_refiner.residual_head[-1].bias.fill_(1.0)

    supplied_geometry_mask = torch.tensor([[True, False]])
    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
        refiner_branch_mask=supplied_geometry_mask,
    )

    assert torch.equal(
        output["bspline_refiner_existence_candidate_mask"],
        torch.tensor([[False, True]]),
    )
    # The GT-derived geometry mask is deliberately not an input to the
    # existence-refining trajectory. A coarse-positive candidate therefore
    # follows the same geometry path in training and inference.
    assert torch.equal(
        output["bspline_refiner_active_branch_mask"],
        torch.tensor([[True, True]]),
    )
    assert torch.count_nonzero(
        output["bspline_refinement_residual_mm"][:, :, 1]
    ) > 0
    torch.testing.assert_close(
        output["branch_exist_logits"][:, 0],
        output["coarse_branch_exist_logits"][:, 0],
    )
    torch.testing.assert_close(
        output["branch_exist_logits"][:, 1],
        output["coarse_branch_exist_logits"][:, 1] - 4.0,
    )
    assert torch.all(
        output["branch_exist_logits"] <= output["coarse_branch_exist_logits"]
    )


def test_existence_forward_features_do_not_depend_on_ground_truth_mask() -> None:
    torch.manual_seed(7)
    model = _small_model(bspline_refiner_refine_branch_existence=True)
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.branch_existence_residual_head is not None
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
        model.bspline_control_refiner.residual_head[-1].weight.zero_()
        model.bspline_control_refiner.residual_head[-1].bias.fill_(0.5)
        existence_output = (
            model.bspline_control_refiner.branch_existence_residual_head[-1]
        )
        existence_output.weight.fill_(0.05)
        existence_output.bias.fill_(-0.1)

    inputs = {
        "views": torch.randn(1, 3, 4),
        "view_mask": torch.ones(1, 3, dtype=torch.bool),
        "image_features": torch.randn(1, 3, 5, 8),
        "images": torch.rand(1, 3, 16, 16),
    }
    gt_absent = model(
        **inputs,
        refiner_branch_mask=torch.tensor([[True, False]]),
    )
    gt_present = model(
        **inputs,
        refiner_branch_mask=torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(
        gt_absent["refinement_stage_branch_exist_logits"],
        gt_present["refinement_stage_branch_exist_logits"],
    )
    torch.testing.assert_close(
        gt_absent["refinement_stage_centerline_parameters_mm"],
        gt_present["refinement_stage_centerline_parameters_mm"],
    )


def test_later_existence_stage_can_restore_an_earlier_removal() -> None:
    class StagewiseExistenceHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            value = -1.0 if self.calls == 0 else 1.0
            self.calls += 1
            return state.new_full((*state.shape[:-1], 1), value)

    model = _small_model(bspline_refiner_refine_branch_existence=True)
    assert model.bspline_control_refiner is not None
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
    model.bspline_control_refiner.branch_existence_residual_head = (
        StagewiseExistenceHead()
    )

    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
    )
    coarse_side_logit = output["coarse_branch_exist_logits"][0, 1]
    stage_side_logits = output[
        "refinement_stage_branch_exist_logits"
    ][0, :, 1]

    assert stage_side_logits[0] < coarse_side_logit
    torch.testing.assert_close(stage_side_logits[1:], coarse_side_logit.expand(2))
    assert torch.all(stage_side_logits <= coarse_side_logit)


@pytest.mark.parametrize(
    "factory",
    (_small_model, _small_legacy_model, _small_absolute_model),
)
def test_existence_refiner_can_retain_fixed_lca_main_slots(factory) -> None:
    model = factory(
        num_branches=3,
        fixed_main_branch_count=2,
        bspline_refiner_refine_branch_existence=True,
    )
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.branch_existence_residual_head is not None
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
        existence_output = (
            model.bspline_control_refiner.branch_existence_residual_head[-1]
        )
        existence_output.weight.zero_()
        existence_output.bias.fill_(-1.0)

    output = _forward(model)

    torch.testing.assert_close(
        output["branch_exist_logits"][:, :2],
        output["coarse_branch_exist_logits"][:, :2],
    )
    assert torch.equal(
        output["branch_exist_probs"][:, :2],
        torch.ones_like(output["branch_exist_probs"][:, :2]),
    )
    assert torch.all(
        output["branch_exist_logits"][:, 2]
        < output["coarse_branch_exist_logits"][:, 2]
    )


def test_existence_refiner_can_remove_positive_tokens_after_target_scope() -> None:
    model = _small_model(
        num_branches=4,
        fixed_main_branch_count=2,
        target_num_branches=2,
        bspline_refiner_refine_branch_existence=True,
    )
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(4.0)

    output = _forward(model)

    assert torch.equal(
        output["bspline_refiner_active_branch_mask"],
        torch.ones((2, 4), dtype=torch.bool),
    )
    assert torch.equal(
        output["bspline_refiner_existence_candidate_mask"],
        torch.tensor([[False, False, True, True]]).expand(2, -1),
    )


def test_removal_only_refiner_excludes_coarse_false_negative_geometry() -> None:
    model = _small_model(bspline_refiner_refine_branch_existence=True)
    assert model.bspline_control_refiner is not None
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(-2.0)
        model.bspline_control_refiner.residual_head[-1].weight.zero_()
        model.bspline_control_refiner.residual_head[-1].bias.fill_(1.0)

    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
        # The side branch exists in GT, but its coarse prediction is absent.
        refiner_branch_mask=torch.tensor([[True, True]]),
    )

    assert torch.equal(
        output["bspline_refiner_existence_candidate_mask"],
        torch.tensor([[False, False]]),
    )
    assert torch.equal(
        output["bspline_refiner_active_branch_mask"],
        torch.tensor([[True, False]]),
    )
    assert torch.count_nonzero(
        output["bspline_refinement_residual_mm"][:, :, 1]
    ) == 0
    torch.testing.assert_close(
        output["branch_exist_logits"],
        output["coarse_branch_exist_logits"],
    )


def test_removal_only_branch_existence_loss_excludes_coarse_negative_slots() -> None:
    stage_logits = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, -1.0],
                [0.0, 0.5, 0.8, -1.0],
            ]
        ],
        requires_grad=True,
    )
    output = {
        "coarse_branch_exist_logits": torch.tensor(
            [[0.0, 1.0, 1.0, -1.0]]
        ),
        "refinement_stage_branch_exist_logits": stage_logits,
        "bspline_refiner_existence_candidate_mask": torch.tensor(
            [[False, True, True, False]]
        ),
    }
    target = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    parts = _bspline_refined_branch_existence_loss(
        output,
        target,
        {
            "artery_type": "RCA",
            "loss": {
                "bspline_refiner_branch_existence_loss_weight": 1.0,
                "bspline_refiner_branch_existence_intermediate_loss_weight": 0.25,
                "bspline_refiner_branch_existence_false_positive_weight": 2.0,
                "bspline_refiner_branch_existence_true_positive_weight": 1.0,
            },
        },
    )
    parts["loss"].backward()

    assert torch.isfinite(parts["loss"])
    assert parts["coarse_false_positive_count"] == pytest.approx(1.0)
    assert stage_logits.grad is not None
    assert torch.all(stage_logits.grad[:, :, 1] > 0.0)
    assert torch.all(stage_logits.grad[:, :, 2] < 0.0)
    assert torch.count_nonzero(stage_logits.grad[:, :, 3]) == 0


def test_removal_only_branch_existence_loss_is_finite_without_candidates() -> None:
    stage_logits = torch.zeros(1, 2, 2, requires_grad=True)
    parts = _bspline_refined_branch_existence_loss(
        {
            "coarse_branch_exist_logits": torch.zeros(1, 2),
            "refinement_stage_branch_exist_logits": stage_logits,
            "bspline_refiner_existence_candidate_mask": torch.zeros(
                1, 2, dtype=torch.bool
            ),
        },
        torch.tensor([[1.0, 0.0]]),
        {"artery_type": "RCA"},
    )
    parts["loss"].backward()

    assert parts["loss"].detach().item() == pytest.approx(0.0)
    assert stage_logits.grad is not None
    assert torch.count_nonzero(stage_logits.grad) == 0


def test_refined_existence_loss_trains_later_tokens_as_absent() -> None:
    stage_logits = torch.zeros(1, 2, 4, requires_grad=True)
    parts = _bspline_refined_branch_existence_loss(
        {
            "coarse_branch_exist_logits": torch.ones(1, 4),
            "refinement_stage_branch_exist_logits": stage_logits,
            "bspline_refiner_existence_candidate_mask": torch.tensor(
                [[False, True, True, True]]
            ),
        },
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        {"artery_type": "RCA"},
        torch.ones((1, 4), dtype=torch.bool),
    )
    parts["loss"].backward()

    assert stage_logits.grad is not None
    assert torch.all(stage_logits.grad[:, :, 1] < 0.0)
    assert torch.all(stage_logits.grad[:, :, 2:] > 0.0)


def test_false_positive_weight_scales_an_fp_only_existence_batch() -> None:
    output = {
        "coarse_branch_exist_logits": torch.tensor([[0.0, 1.0]]),
        "refinement_stage_branch_exist_logits": torch.tensor(
            [[[0.0, 1.0]]]
        ),
        "bspline_refiner_existence_candidate_mask": torch.tensor(
            [[False, True]]
        ),
    }
    target = torch.tensor([[1.0, 0.0]])
    unit_weight = _bspline_refined_branch_existence_loss(
        output,
        target,
        {
            "artery_type": "RCA",
            "loss": {
                "bspline_refiner_branch_existence_false_positive_weight": 1.0
            },
        },
    )
    double_weight = _bspline_refined_branch_existence_loss(
        output,
        target,
        {
            "artery_type": "RCA",
            "loss": {
                "bspline_refiner_branch_existence_false_positive_weight": 2.0
            },
        },
    )

    torch.testing.assert_close(
        double_weight["loss"],
        2.0 * unit_weight["loss"],
    )


def test_refined_existence_loss_reaches_refiner_but_not_frozen_coarse_head() -> None:
    model = _small_model(bspline_refiner_refine_branch_existence=True)
    configure_bspline_refiner_only_training(
        model,
        {"train_bspline_refiner_only": True},
    )
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
        refiner_branch_mask=torch.tensor([[True, False]]),
    )
    parts = _bspline_refined_branch_existence_loss(
        output,
        torch.tensor([[1.0, 0.0]]),
        {"artery_type": "RCA"},
    )
    parts["loss"].backward()

    assert model.bspline_control_refiner is not None
    existence_head = model.bspline_control_refiner.branch_existence_residual_head
    assert existence_head is not None
    assert existence_head[-1].weight.grad is not None
    assert torch.count_nonzero(existence_head[-1].weight.grad) > 0
    assert all(
        parameter.grad is None
        for parameter in model.branch_exist_head.parameters()
    )


def test_compute_parametric_loss_adds_refined_existence_once() -> None:
    model = _small_model(bspline_refiner_refine_branch_existence=True)
    with torch.no_grad():
        model.branch_exist_head[-1].weight.zero_()
        model.branch_exist_head[-1].bias.fill_(2.0)
    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
    )
    target_vessel = output["decoded_vessel_mm"].detach().clone()
    target_parameters = output["centerline_parameters_mm"].detach().clone()
    losses = compute_parametric_loss(
        output,
        {
            "target_branch_exist": torch.tensor([[1.0, 0.0]]),
            "target_centerline_parameters_mm": target_parameters,
            "target_attachment_index": torch.full((1, 2), -1),
            "target_raw_vessel_mm": target_vessel,
            "target_reconstructed_vessel_mm": target_vessel,
            "target_point_valid_mask": torch.ones(
                1, 2, 11, dtype=torch.bool
            ),
        },
        {
            "artery_type": "RCA",
            "centerline_prediction_mode": "bspline_control_points",
            "radius_prediction_mode": "raw",
            "loss": {
                "branch_exist_loss_weight": 0.0,
                "centerline_control_loss_weight": 0.0,
                "attachment_loss_weight": 0.0,
                "side_relative_xyz_loss_weight": 0.0,
                "decoded_xyz_loss_weight": 0.0,
                "decoded_radius_loss_weight": 0.0,
                "bspline_refiner_coarse_loss_weight": 0.0,
                "bspline_refiner_intermediate_loss_weight": 0.0,
                "bspline_refiner_branch_existence_loss_weight": 1.0,
                "bspline_refiner_branch_existence_intermediate_loss_weight": 0.25,
            },
        },
    )

    expected_existence_total = (
        losses["bspline_refiner_branch_existence_weighted_loss"]
        + losses[
            "bspline_refiner_branch_existence_intermediate_weighted_loss"
        ]
    )
    torch.testing.assert_close(losses["loss"], expected_existence_total)
    assert losses[
        "bspline_refiner_branch_existence_coarse_false_positive_count"
    ] == pytest.approx(1.0)
    assert losses["loss"] > 0.0


def test_existence_refinement_is_finite_with_only_fixed_main_branch() -> None:
    model = _small_model(
        num_branches=1,
        bspline_refiner_refine_branch_existence=True,
    )
    output = _forward(model)
    parts = _bspline_refined_branch_existence_loss(
        output,
        torch.ones(2, 1),
        {"artery_type": "RCA"},
    )

    assert torch.isfinite(parts["loss"])
    assert parts["candidate_count"] == pytest.approx(0.0)
    torch.testing.assert_close(
        output["branch_exist_logits"],
        output["coarse_branch_exist_logits"],
    )


def test_branch_existence_refiner_config_requires_boolean_switch() -> None:
    with pytest.raises(
        ValueError,
        match="bspline_refiner_refine_branch_existence must be a JSON boolean",
    ):
        build_model_from_config(
            {
                **_small_model_kwargs(),
                "bspline_refiner_refine_branch_existence": "true",
            },
            view_feat_dim=4,
            inferred_feature_dim=8,
        )


def test_model_config_separates_branch_tokens_from_target_count() -> None:
    model = build_model_from_config(
        {
            **_small_model_kwargs(),
            "num_branches": 2,
            "model": {"num_branches": 4},
        },
        view_feat_dim=4,
        inferred_feature_dim=8,
    )

    assert model.num_branches == 4
    assert model.target_num_branches == 2




def test_absolute_parallel_refiner_keeps_every_stage_in_global_frame() -> None:
    model = _small_absolute_model()
    output = _forward(model)

    assert "attachment_logits" not in output
    assert "attachment_probabilities" not in output
    stage_parameters = output["refinement_stage_centerline_parameters_mm"]
    stage_vessels = output["refinement_stage_decoded_vessel_mm"]
    for stage_index in range(stage_parameters.shape[1]):
        expected_centerlines = model._decode_local_centerlines(
            stage_parameters[:, stage_index]
        )
        torch.testing.assert_close(
            stage_vessels[:, stage_index, ..., :3],
            expected_centerlines,
        )
    torch.testing.assert_close(
        output["decoded_vessel_mm"],
        output["coarse_decoded_vessel_mm"],
    )


def test_absolute_parallel_checkpoint_can_warm_start_refiner() -> None:
    baseline = _small_absolute_model(use_bspline_control_refiner=False)
    refined = _small_absolute_model()

    incompatible = refined.load_state_dict(baseline.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert "bspline_refiner_anchor_indices" in incompatible.missing_keys
    assert any(
        key.startswith("bspline_control_refiner.")
        for key in incompatible.missing_keys
    )
    assert (
        infer_checkpoint_decoder_architecture(refined.state_dict())
        == "absolute_parallel"
    )


def test_geometry_only_refiner_can_warm_start_existence_refinement() -> None:
    geometry_only = _small_model()
    with_existence = _small_model(
        bspline_refiner_refine_branch_existence=True
    )

    incompatible = with_existence.load_state_dict(
        geometry_only.state_dict(),
        strict=False,
    )

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("bspline_control_refiner.branch_existence_")
        for key in incompatible.missing_keys
    )


def test_absolute_parallel_radius_head_scope_uses_shared_head() -> None:
    model = _small_absolute_model(num_branches=2)
    config = {"train_radius_head_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.raw_radius_head is not None
    assert not hasattr(model, "side_raw_radius_head")
    expected = {id(parameter) for parameter in model.raw_radius_head.parameters()}
    assert {id(parameter) for parameter in trainable} == expected
    assert model.raw_radius_head.training is True


def test_absolute_parallel_refiner_reports_latency_and_receives_gradient() -> None:
    model = _small_absolute_model()
    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
        record_coarse_prediction_timing=True,
        record_refiner_timing=True,
    )

    assert output["coarse_prediction_elapsed_ms"].shape == (1,)
    assert torch.isfinite(output["coarse_prediction_elapsed_ms"]).all()
    assert torch.all(output["coarse_prediction_elapsed_ms"] > 0.0)
    assert output["bspline_refiner_elapsed_ms"].shape == (1,)
    assert output["refiner_total_elapsed_ms"].shape == (1,)
    loss = output["decoded_vessel_mm"][..., :3].square().mean()
    loss.backward()
    assert model.bspline_control_refiner is not None
    final_layer = model.bspline_control_refiner.residual_head[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0
























def test_geometry_refiner_rejects_coarse_configuration_drift() -> None:
    coarse = {
        "feature_backbone": "vggt",
        "expected_vggt_context_mode": "per_view",
        "check_feature_finite": False,
        "parametric_target_source": "feature_file",
        "artery_type": "RCA",
        "num_branches": 7,
        "num_points": 200,
        "centerline_prediction_mode": "bspline_control_points",
        "num_control_points": 20,
        "radius_prediction_mode": "raw",
        "min_train_views": 1,
        "max_train_views": 7,
        "train_view_count_weights": {"1": 0.5, "2": 0.5},
        "model": {
            "decoder_architecture": "absolute_parallel",
            "model_dim": 256,
            "num_decoder_layers": 2,
            "dropout": 0.25,
            "use_bspline_control_refiner": False,
        },
    }
    refiner = deepcopy(coarse)
    refiner["train_bspline_refiner_only"] = True
    refiner["parametric_target_source"] = "directory"
    refiner["parametric_target_dir"] = "/targets"
    refiner["model"]["use_bspline_control_refiner"] = True

    compared = validate_geometry_refiner_coarse_config(refiner, coarse)

    assert "decoder_architecture" in compared
    assert "train_view_count_weights" in compared
    assert "check_feature_finite" in compared
    assert "dropout" in compared
    assert "parametric_target_source" not in compared
    assert refiner["initial_checkpoint_coarse_config_match"] is True

    refiner["model"]["num_decoder_layers"] = 3
    with pytest.raises(
        ValueError,
        match="num_decoder_layers: checkpoint=2, refiner=3",
    ):
        validate_geometry_refiner_coarse_config(refiner, coarse)


def test_geometry_refiner_warns_for_branch_variant_training_protocol_drift() -> None:
    coarse = {
        "feature_backbone": "vggt",
        "artery_type": "RCA",
        "num_branches": 7,
        "branch_variant_group_training": True,
        "branch_variant_train_sampling": "all",
        "branch_variant_resample_each_epoch": True,
        "model": {
            "decoder_architecture": "absolute_parallel",
            "model_dim": 256,
        },
    }
    refiner = deepcopy(coarse)
    refiner["train_bspline_refiner_only"] = True
    refiner["branch_variant_group_training"] = False
    refiner["branch_variant_train_sampling"] = "random_one"
    refiner["branch_variant_resample_each_epoch"] = False

    with pytest.warns(
        RuntimeWarning,
        match=(
            "settings control data sampling/grouping rather than the frozen "
            "coarse-model architecture"
        ),
    ):
        compared = validate_geometry_refiner_coarse_config(refiner, coarse)

    assert "decoder_architecture" in compared
    assert "branch_variant_group_training" not in compared
    assert refiner["initial_checkpoint_coarse_config_match"] is True
    assert refiner["initial_checkpoint_training_protocol_mismatches"] == [
        "branch_variant_group_training: checkpoint=True, refiner=False",
        "branch_variant_train_sampling: checkpoint='all', "
        "refiner='random_one'",
        "branch_variant_resample_each_epoch: checkpoint=True, refiner=False",
    ]










def test_bspline_refiner_can_report_coarse_to_final_latency() -> None:
    model = _small_model()
    output = model(
        views=torch.randn(1, 3, 4),
        view_mask=torch.ones(1, 3, dtype=torch.bool),
        image_features=torch.randn(1, 3, 5, 8),
        images=torch.rand(1, 3, 16, 16),
        record_coarse_prediction_timing=True,
        record_refiner_timing=True,
    )

    assert output["coarse_prediction_elapsed_ms"].shape == (1,)
    assert torch.isfinite(output["coarse_prediction_elapsed_ms"]).all()
    assert torch.all(output["coarse_prediction_elapsed_ms"] > 0.0)
    assert output["bspline_refiner_elapsed_ms"].shape == (1,)
    assert output["refiner_total_elapsed_ms"].shape == (1,)
    assert torch.isfinite(output["bspline_refiner_elapsed_ms"]).all()
    assert torch.all(output["bspline_refiner_elapsed_ms"] > 0.0)
    assert torch.all(
        output["refiner_total_elapsed_ms"]
        >= output["bspline_refiner_elapsed_ms"]
    )


def test_coarse_prediction_timing_is_disabled_by_default() -> None:
    output = _forward(_small_model())

    assert "coarse_prediction_elapsed_ms" not in output


def test_bspline_refiner_receives_gradient_from_refined_geometry() -> None:
    model = _small_model()
    output = _forward(model)
    loss = output["decoded_vessel_mm"][..., :3].square().mean()
    loss.backward()

    assert model.bspline_control_refiner is not None
    final_layer = model.bspline_control_refiner.residual_head[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0


def test_refiner_only_training_freezes_coarse_parameters_and_dropout_mode() -> None:
    model = _small_model(dropout=0.25)
    config = {"train_bspline_refiner_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.bspline_control_refiner is not None
    refiner_parameter_ids = {
        id(parameter) for parameter in model.bspline_control_refiner.parameters()
    }
    assert {id(parameter) for parameter in trainable} == refiner_parameter_ids
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("bspline_control_refiner.")
    )
    assert model.training is False
    assert model.branch_decoder.training is False
    assert model.bspline_control_refiner.training is True
    assert config["train_bspline_refiner_only_effective"] is True
    assert config["num_frozen_parameters"] > 0

    output = _forward(model)
    output["decoded_vessel_mm"][..., :3].square().mean().backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("bspline_control_refiner.")
    )
    assert any(parameter.grad is not None for parameter in trainable)


def test_refiner_training_keeps_patch_probability_predictor_frozen(
    tmp_path: Path,
) -> None:
    source = CenterlineProbabilityPredictor(
        learned_feature_dim=4,
        centerline_head_hidden_dim=3,
        centerline_map_size=8,
    )
    checkpoint_path = tmp_path / "centerline_probability.pt"
    torch.save(
        {
            "schema_version": CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMA_VERSION,
            "task": "single_view_centerline_probability",
            "epoch": 4,
            "model_state_dict": source.state_dict(),
            "config": {},
        },
        checkpoint_path,
    )
    model = _small_model(
        bspline_refiner_use_learned_image_features=True,
        bspline_refiner_learned_feature_dim=4,
        bspline_refiner_use_centerline_probability_patch_evidence=True,
        bspline_refiner_centerline_probability_patch_size=3,
        bspline_refiner_use_separate_centerline_encoder=True,
        bspline_refiner_centerline_map_size=8,
        bspline_refiner_centerline_head_hidden_dim=3,
    )
    config = {
        "train_bspline_refiner_only": True,
        "model": {
            "bspline_refiner_use_centerline_probability_patch_evidence": True,
            "bspline_refiner_pretrained_centerline_checkpoint": str(
                checkpoint_path
            ),
            "bspline_refiner_freeze_pretrained_centerline_predictor": True,
        },
    }

    info = configure_pretrained_centerline_predictor(
        model,
        config,
        load_weights=True,
    )
    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert info is not None
    assert info["frozen"] is True
    assert info["separate_encoder"] is True
    assert model.bspline_control_refiner is not None
    refiner = model.bspline_control_refiner
    assert refiner.centerline_image_feature_encoder is not None
    assert refiner.input_centerline_probability_head is not None
    assert refiner.centerline_probability_patch_projection is not None
    torch.testing.assert_close(
        refiner.centerline_image_feature_encoder[0].weight,
        source.image_feature_encoder[0].weight,
    )
    frozen_parameter_ids = {
        id(parameter)
        for module in (
            refiner.centerline_image_feature_encoder,
            refiner.input_centerline_probability_head,
        )
        for parameter in module.parameters()
    }
    trainable_parameter_ids = {id(parameter) for parameter in trainable}
    patch_parameter_ids = {
        id(parameter)
        for parameter in (
            refiner.centerline_probability_patch_projection.parameters()
        )
    }
    assert frozen_parameter_ids.isdisjoint(trainable_parameter_ids)
    assert patch_parameter_ids.issubset(trainable_parameter_ids)
    frozen_parameters = (
        *refiner.centerline_image_feature_encoder.parameters(),
        *refiner.input_centerline_probability_head.parameters(),
    )
    assert all(not parameter.requires_grad for parameter in frozen_parameters)
    assert refiner.centerline_image_feature_encoder.training is False
    assert refiner.input_centerline_probability_head.training is False


def test_refiner_only_scope_includes_branch_existence_head() -> None:
    model = _small_model(bspline_refiner_refine_branch_existence=True)
    trainable = configure_bspline_refiner_only_training(
        model,
        {"train_bspline_refiner_only": True},
    )

    assert model.bspline_control_refiner is not None
    existence_head = (
        model.bspline_control_refiner.branch_existence_residual_head
    )
    assert existence_head is not None
    trainable_ids = {id(parameter) for parameter in trainable}
    assert {
        id(parameter) for parameter in existence_head.parameters()
    }.issubset(trainable_ids)
    assert all(
        not parameter.requires_grad
        for parameter in model.branch_exist_head.parameters()
    )


def test_refiner_only_training_requires_enabled_refiner() -> None:
    model = _small_model(use_bspline_control_refiner=False)
    with pytest.raises(ValueError, match="use_bspline_control_refiner=true"):
        configure_bspline_refiner_only_training(
            model,
            {"train_bspline_refiner_only": True},
        )


@pytest.mark.parametrize(
    "config",
    [
        {
            "train_bspline_refiner_only": True,
            "train_radius_refiner_only": True,
        },
        {
            "train_bspline_refiner_only": True,
            "train_radius_head_only": True,
        },
        {
            "train_radius_refiner_only": True,
            "train_radius_head_only": True,
        },
        {
            "train_bspline_refiner_only": True,
            "train_bspline_refiner_and_radius_head_only": True,
        },
        {
            "train_radius_refiner_only": True,
            "train_bspline_refiner_and_radius_head_only": True,
        },
        {
            "train_radius_head_only": True,
            "train_bspline_refiner_and_radius_head_only": True,
        },
    ],
)
def test_specialized_training_modes_are_mutually_exclusive(
    config: dict[str, bool],
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_bspline_refiner_only_training(_small_model(), config)


def test_radius_head_only_training_selects_only_rca_main_head() -> None:
    model = _small_model(num_branches=1, dropout=0.25)
    config = {"train_radius_head_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.raw_radius_head is not None
    main_head_parameter_ids = {
        id(parameter) for parameter in model.raw_radius_head.parameters()
    }
    assert {id(parameter) for parameter in trainable} == main_head_parameter_ids
    assert all(
        parameter.requires_grad == name.startswith("raw_radius_head.")
        for name, parameter in model.named_parameters()
    )
    assert model.training is False
    assert model.branch_decoder.training is False
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.training is False
    assert model.raw_radius_head.training is True
    assert model.side_raw_radius_head is not None
    assert model.side_raw_radius_head.training is False
    assert config["train_radius_head_only_effective"] is True
    assert config["trainable_refiner_module"] is None
    assert config["trainable_modules"] == ["raw_radius_head"]

    output = _forward(model)
    output["decoded_vessel_mm"][..., 3].square().mean().backward()
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("raw_radius_head.")
    )

    set_parametric_model_training_mode(model, training=False, config=config)
    assert model.training is False
    assert model.raw_radius_head.training is False


def test_radius_head_only_training_selects_both_lca_radius_heads() -> None:
    model = _small_model(num_branches=2, dropout=0.25)
    config = {"train_radius_head_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.raw_radius_head is not None
    assert model.side_raw_radius_head is not None
    radius_head_parameter_ids = {
        id(parameter)
        for head in (model.raw_radius_head, model.side_raw_radius_head)
        for parameter in head.parameters()
    }
    assert {id(parameter) for parameter in trainable} == radius_head_parameter_ids
    assert all(
        parameter.requires_grad
        == (
            name.startswith("raw_radius_head.")
            or name.startswith("side_raw_radius_head.")
        )
        for name, parameter in model.named_parameters()
    )
    assert model.training is False
    assert model.branch_decoder.training is False
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.training is False
    assert model.raw_radius_head.training is True
    assert model.side_raw_radius_head.training is True
    assert config["trainable_modules"] == [
        "raw_radius_head",
        "side_raw_radius_head",
    ]

    output = _forward(model)
    output["decoded_vessel_mm"][..., 3].square().mean().backward()
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not (
            name.startswith("raw_radius_head.")
            or name.startswith("side_raw_radius_head.")
        )
    )

    set_parametric_model_training_mode(model, training=False, config=config)
    assert model.raw_radius_head.training is False
    assert model.side_raw_radius_head.training is False


def test_radius_head_only_training_rejects_incompatible_modules() -> None:
    with pytest.raises(ValueError, match="use_bspline_control_refiner=true"):
        configure_bspline_refiner_only_training(
            _small_model(use_bspline_control_refiner=False),
            {"train_radius_head_only": True},
        )

    with pytest.raises(ValueError, match="radius_prediction_mode='raw'"):
        configure_bspline_refiner_only_training(
            _small_model(
                radius_prediction_mode="parametric",
                num_radius_coefficients=4,
            ),
            {"train_radius_head_only": True},
        )

    radius_refiner_model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=1,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_axial_subsamples=0,
    )
    with pytest.raises(ValueError, match="use_radius_evidence_refiner=false"):
        configure_bspline_refiner_only_training(
            radius_refiner_model,
            {"train_radius_head_only": True},
        )

    missing_side_head_model = _small_model(num_branches=2)
    missing_side_head_model.side_raw_radius_head = None
    with pytest.raises(ValueError, match="side_raw_radius_head"):
        configure_bspline_refiner_only_training(
            missing_side_head_model,
            {"train_radius_head_only": True},
        )


def test_joint_bspline_refiner_radius_head_scope_selects_rca_modules() -> None:
    model = _small_model(num_branches=1, dropout=0.25)
    config = {"train_bspline_refiner_and_radius_head_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.bspline_control_refiner is not None
    assert model.raw_radius_head is not None
    selected_parameter_ids = {
        id(parameter)
        for module in (model.bspline_control_refiner, model.raw_radius_head)
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in trainable} == selected_parameter_ids
    assert all(
        parameter.requires_grad
        == (
            name.startswith("bspline_control_refiner.")
            or name.startswith("raw_radius_head.")
        )
        for name, parameter in model.named_parameters()
    )
    assert model.training is False
    assert model.branch_decoder.training is False
    assert model.bspline_control_refiner.training is True
    assert model.raw_radius_head.training is True
    assert model.side_raw_radius_head is not None
    assert model.side_raw_radius_head.training is False
    assert config[
        "train_bspline_refiner_and_radius_head_only_effective"
    ] is True
    assert config["trainable_refiner_module"] == "bspline_control_refiner"
    assert config["trainable_modules"] == [
        "bspline_control_refiner",
        "raw_radius_head",
    ]

    output = _forward(model)
    (
        output["decoded_vessel_mm"][..., :3].square().mean()
        + output["decoded_vessel_mm"][..., 3].square().mean()
    ).backward()
    assert any(
        parameter.grad is not None
        for parameter in model.bspline_control_refiner.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.raw_radius_head.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not (
            name.startswith("bspline_control_refiner.")
            or name.startswith("raw_radius_head.")
        )
    )


def test_joint_bspline_refiner_radius_head_scope_selects_both_lca_heads() -> None:
    model = _small_model(num_branches=2, dropout=0.25)
    config = {"train_bspline_refiner_and_radius_head_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.bspline_control_refiner is not None
    assert model.raw_radius_head is not None
    assert model.side_raw_radius_head is not None
    selected_modules = (
        model.bspline_control_refiner,
        model.raw_radius_head,
        model.side_raw_radius_head,
    )
    selected_parameter_ids = {
        id(parameter)
        for module in selected_modules
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in trainable} == selected_parameter_ids
    assert config["trainable_modules"] == [
        "bspline_control_refiner",
        "raw_radius_head",
        "side_raw_radius_head",
    ]
    assert all(module.training is True for module in selected_modules)
    assert model.branch_decoder.training is False

    output = _forward(model)
    (
        output["decoded_vessel_mm"][..., :3].square().mean()
        + output["decoded_vessel_mm"][..., 3].square().mean()
    ).backward()
    for module in selected_modules:
        assert any(parameter.grad is not None for parameter in module.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not (
            name.startswith("bspline_control_refiner.")
            or name.startswith("raw_radius_head.")
            or name.startswith("side_raw_radius_head.")
        )
    )


def test_joint_bspline_refiner_radius_head_scope_rejects_incompatible_model(
) -> None:
    joint_config = {"train_bspline_refiner_and_radius_head_only": True}
    with pytest.raises(ValueError, match="use_bspline_control_refiner=true"):
        configure_bspline_refiner_only_training(
            _small_model(use_bspline_control_refiner=False),
            joint_config,
        )

    with pytest.raises(ValueError, match="radius_prediction_mode='raw'"):
        configure_bspline_refiner_only_training(
            _small_model(
                radius_prediction_mode="parametric",
                num_radius_coefficients=4,
            ),
            joint_config,
        )

    radius_refiner_model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=1,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_axial_subsamples=0,
    )
    with pytest.raises(ValueError, match="use_radius_evidence_refiner=false"):
        configure_bspline_refiner_only_training(
            radius_refiner_model,
            joint_config,
        )

    missing_side_head_model = _small_model(num_branches=2)
    missing_side_head_model.side_raw_radius_head = None
    with pytest.raises(ValueError, match="side_raw_radius_head"):
        configure_bspline_refiner_only_training(
            missing_side_head_model,
            joint_config,
        )


def test_bspline_refiner_distance_direction_path_is_finite() -> None:
    model = _small_model(
        bspline_refiner_num_stages=1,
        bspline_refiner_use_distance_transform=True,
        bspline_refiner_distance_transform_num_iters=2,
    )

    output = _forward(model)

    assert torch.isfinite(output["decoded_vessel_mm"]).all()
    assert torch.isfinite(output["bspline_refinement_residual_mm"]).all()


def test_bspline_refiner_requires_images() -> None:
    model = _small_model()
    with pytest.raises(ValueError, match="requires input images"):
        model(
            views=torch.randn(1, 2, 4),
            view_mask=torch.ones(1, 2, dtype=torch.bool),
            image_features=torch.randn(1, 2, 5, 8),
        )


def test_radius_refiner_runs_after_geometry_and_changes_only_radius() -> None:
    model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=2,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_residual_scale_mm=0.5,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_radial_subsamples=1,
        radius_refiner_render_axial_subsamples=0,
    )

    output = _forward(model)

    assert output["radius_refiner_coarse_decoded_vessel_mm"].shape == (
        2,
        2,
        11,
        4,
    )
    assert output["radius_refinement_stage_decoded_vessel_mm"].shape == (
        2,
        2,
        2,
        11,
        4,
    )
    assert output["radius_refinement_residual_mm"].shape == (2, 2, 2, 11)
    assert output["radius_refiner_coarse_rendered_masks"].shape == (
        2,
        3,
        16,
        16,
    )
    assert output["radius_refiner_final_rendered_masks"].shape == (
        2,
        3,
        16,
        16,
    )
    assert torch.equal(
        output["decoded_vessel_mm"][..., :3],
        output["radius_refiner_coarse_decoded_vessel_mm"][..., :3],
    )
    # The zero-initialized radius head starts as an exact identity update.
    assert torch.allclose(
        output["decoded_vessel_mm"][..., 3],
        output["radius_refiner_coarse_decoded_vessel_mm"][..., 3],
    )
    assert torch.equal(
        output["raw_radius_mm"],
        output["decoded_vessel_mm"][..., 3],
    )
    assert torch.equal(
        output["coarse_raw_radius_mm"],
        output["radius_refiner_coarse_decoded_vessel_mm"][..., 3],
    )
    assert torch.count_nonzero(output["radius_refinement_residual_mm"]) == 0

    output["decoded_vessel_mm"][..., 3].square().mean().backward()
    assert model.radius_evidence_refiner is not None
    final_layer = model.radius_evidence_refiner.residual_head[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0


def test_radius_refiner_masks_predicted_absent_branch_without_target_input() -> None:
    model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=2,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_residual_scale_mm=0.5,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_radial_subsamples=1,
        radius_refiner_render_axial_subsamples=0,
        radius_refiner_branch_probability_threshold=0.5,
    )
    assert model.radius_evidence_refiner is not None
    with torch.no_grad():
        for parameter in model.branch_exist_head.parameters():
            parameter.zero_()
        model.branch_exist_head[-1].bias.fill_(-20.0)
        model.radius_evidence_refiner.residual_head[-1].bias.fill_(1.0)

    output = _forward(model)

    assert torch.equal(
        output["radius_refiner_active_branch_mask"],
        torch.tensor([[True, False], [True, False]]),
    )
    residual = output["radius_refinement_residual_mm"]
    assert float(residual[:, :, 0].abs().sum()) > 0.0
    assert torch.count_nonzero(residual[:, :, 1]) == 0
    assert torch.equal(
        output["decoded_vessel_mm"][:, 1, :, 3],
        output["radius_refiner_coarse_decoded_vessel_mm"][:, 1, :, 3],
    )


def test_ground_truth_override_gates_geometry_and_radius_refiners_together() -> None:
    model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=1,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_axial_subsamples=0,
    )
    assert model.bspline_control_refiner is not None
    assert model.radius_evidence_refiner is not None
    with torch.no_grad():
        for parameter in model.branch_exist_head.parameters():
            parameter.zero_()
        model.branch_exist_head[-1].bias.fill_(-20.0)
        model.bspline_control_refiner.residual_head[-1].bias.fill_(1.0)
        model.radius_evidence_refiner.residual_head[-1].bias.fill_(1.0)
    supplied_mask = torch.tensor([[True, True], [True, False]])

    output = model(
        views=torch.randn(2, 3, 4),
        view_mask=torch.ones(2, 3, dtype=torch.bool),
        image_features=torch.randn(2, 3, 5, 8),
        images=torch.rand(2, 3, 16, 16),
        refiner_branch_mask=supplied_mask,
    )

    assert torch.equal(output["bspline_refiner_active_branch_mask"], supplied_mask)
    assert torch.equal(output["radius_refiner_active_branch_mask"], supplied_mask)
    assert float(output["bspline_refinement_residual_mm"][0, :, 1].abs().sum()) > 0
    assert torch.count_nonzero(
        output["bspline_refinement_residual_mm"][1, :, 1]
    ) == 0
    assert float(output["radius_refinement_residual_mm"][0, :, 1].abs().sum()) > 0
    assert torch.count_nonzero(
        output["radius_refinement_residual_mm"][1, :, 1]
    ) == 0


def test_lca_fixed_main_branches_remain_active_with_predicted_gating() -> None:
    model = _small_model(fixed_main_branch_count=2)
    with torch.no_grad():
        for parameter in model.branch_exist_head.parameters():
            parameter.zero_()
        model.branch_exist_head[-1].bias.fill_(-20.0)

    output = _forward(model)

    assert torch.equal(
        output["bspline_refiner_active_branch_mask"],
        torch.ones(2, 2, dtype=torch.bool),
    )


def test_known_total_branch_count_keeps_fixed_main_and_top_coarse_sides() -> None:
    model = _small_model(
        num_branches=6,
        fixed_main_branch_count=2,
        target_num_branches=6,
    )
    output = {
        "branch_exist_probs": torch.tensor(
            [
                [1.0, 1.0, 0.30, 0.90, 0.80, 0.10],
                [1.0, 1.0, 0.70, 0.20, 0.60, 0.80],
            ]
        )
    }

    mask = model._count_limited_refiner_branch_mask(
        output=output,
        total_branch_count=4,
        override=None,
    )

    assert mask is not None
    assert mask.tolist() == [
        [True, True, False, True, True, False],
        [True, True, True, False, False, True],
    ]
    assert torch.equal(
        output["prediction_count_limited_branch_mask"], mask
    )


def test_known_total_branch_count_is_forwarded_to_geometry_refiner() -> None:
    model = _small_model()

    output = model(
        views=torch.randn(2, 3, 4),
        view_mask=torch.ones(2, 3, dtype=torch.bool),
        image_features=torch.randn(2, 3, 5, 8),
        images=torch.rand(2, 3, 16, 16),
        refiner_total_branch_count=1,
    )

    expected = torch.tensor([[True, False], [True, False]])
    assert torch.equal(
        output["prediction_count_limited_branch_mask"], expected
    )
    assert torch.equal(output["bspline_refiner_active_branch_mask"], expected)


def test_ground_truth_training_mask_forces_lca_main_structure() -> None:
    batch = {
        "target_branch_exist": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
    }
    config = {
        "artery_type": "LCA",
        "refiner_branch_existence_source": "ground_truth",
    }

    mask = refiner_branch_mask_for_batch(batch, config, torch.device("cpu"))

    assert torch.equal(
        mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )


def test_radius_refiner_reuses_projection_context_across_stages(
    monkeypatch,
) -> None:
    model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=2,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_radial_subsamples=1,
        radius_refiner_render_axial_subsamples=0,
    )
    refiner = model.radius_evidence_refiner
    assert refiner is not None
    calls = {"point_grid": 0, "cameras": 0, "geometry": 0}

    original_point_grid = refiner._project_points_to_grid
    original_cameras = refiner.surface_projector.prepare_cameras
    original_geometry = refiner.surface_projector.prepare_surface_geometry

    def counted_point_grid(*args, **kwargs):
        calls["point_grid"] += 1
        return original_point_grid(*args, **kwargs)

    def counted_cameras(*args, **kwargs):
        calls["cameras"] += 1
        return original_cameras(*args, **kwargs)

    def counted_geometry(*args, **kwargs):
        calls["geometry"] += 1
        return original_geometry(*args, **kwargs)

    monkeypatch.setattr(refiner, "_project_points_to_grid", counted_point_grid)
    monkeypatch.setattr(
        refiner.surface_projector, "prepare_cameras", counted_cameras
    )
    monkeypatch.setattr(
        refiner.surface_projector,
        "prepare_surface_geometry",
        counted_geometry,
    )

    _forward(model)

    assert calls == {"point_grid": 1, "cameras": 1, "geometry": 2}


def test_radius_refiner_only_training_freezes_geometry_refiner() -> None:
    model = _small_model(
        use_radius_evidence_refiner=True,
        radius_refiner_num_stages=1,
        radius_refiner_evidence_hidden_dim=32,
        radius_refiner_profile_samples=5,
        radius_refiner_profile_half_width_px=4.0,
        radius_refiner_image_size=16,
        radius_refiner_render_num_circle_points=8,
        radius_refiner_render_axial_subsamples=0,
    )
    config = {"train_radius_refiner_only": True}

    trainable = configure_bspline_refiner_only_training(model, config)
    set_parametric_model_training_mode(model, training=True, config=config)

    assert model.radius_evidence_refiner is not None
    radius_parameter_ids = {
        id(parameter)
        for parameter in model.radius_evidence_refiner.parameters()
    }
    assert {id(parameter) for parameter in trainable} == radius_parameter_ids
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("radius_evidence_refiner.")
    )
    assert model.training is False
    assert model.bspline_control_refiner is not None
    assert model.bspline_control_refiner.training is False
    assert model.radius_evidence_refiner.training is True
    assert config["train_radius_refiner_only_effective"] is True

    output = _forward(model)
    output["decoded_vessel_mm"][..., 3].square().mean().backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("radius_evidence_refiner.")
    )
    assert any(parameter.grad is not None for parameter in trainable)


def test_radius_refiner_rejects_unsupported_model_modes() -> None:
    with pytest.raises(ValueError, match="use_bspline_control_refiner=true"):
        _small_model(
            use_bspline_control_refiner=False,
            use_radius_evidence_refiner=True,
        )
    with pytest.raises(ValueError, match="radius_prediction_mode='raw'"):
        _small_model(
            radius_prediction_mode="parametric",
            num_radius_coefficients=4,
            use_radius_evidence_refiner=True,
        )
    with pytest.raises(ValueError, match="branch_probability_threshold"):
        _small_model(radius_refiner_branch_probability_threshold=1.5)








def test_bspline_refiner_rejects_landmark_mode_and_invalid_stage_count() -> None:
    with pytest.raises(ValueError, match="supported only"):
        _small_model(
            centerline_prediction_mode="adaptive_landmarks",
            num_landmarks=6,
        )
    with pytest.raises(ValueError, match="num_stages"):
        _small_model(bspline_refiner_num_stages=0)
    with pytest.raises(ValueError, match="candidate_threshold"):
        _small_model(
            bspline_refiner_refine_branch_existence=True,
            bspline_refiner_branch_existence_candidate_threshold=1.5,
        )
    with pytest.raises(ValueError, match="candidate_threshold"):
        _small_model(
            bspline_refiner_refine_branch_existence=True,
            bspline_refiner_branch_existence_candidate_threshold=0.25,
        )
    with pytest.raises(ValueError, match="max_logit_decrease"):
        _small_model(
            bspline_refiner_refine_branch_existence=True,
            bspline_refiner_branch_existence_max_logit_decrease=0.0,
        )
    with pytest.raises(ValueError, match="requires use_bspline_control_refiner"):
        _small_model(
            use_bspline_control_refiner=False,
            bspline_refiner_refine_branch_existence=True,
        )


def test_coarse_and_intermediate_losses_are_added_without_reweighting_final(
    monkeypatch,
) -> None:
    decoded = torch.zeros(1, 1, 2, 4)
    decoded[..., 3] = 1.0
    empty_side_offsets = torch.empty(1, 0, 2, 4)
    final_parameters = torch.ones(1, 1, 2, 3)
    coarse_parameters = torch.full_like(final_parameters, 2.0)
    first_stage_parameters = torch.full_like(final_parameters, 3.0)
    second_stage_parameters = torch.full_like(final_parameters, 5.0)
    output = {
        "branch_exist_logits": torch.zeros(1, 1),
        "branch_exist_probs": torch.ones(1, 1),
        "centerline_parameters_mm": final_parameters,
        "centerline_control_points_mm": final_parameters,
        "radius_baseline_coefficients_log_mm": torch.zeros(1, 1, 1),
        "raw_radius_log_mm": torch.zeros(1, 1, 0),
        "raw_radius_mm": torch.zeros(1, 1, 0),
        "lesion_exist_logits": torch.zeros(1, 1, 0),
        "lesion_geometry": torch.zeros(1, 1, 0, 3),
        "attachment_logits": torch.zeros(1, 1, 2),
        "decoded_vessel_mm": decoded,
        "side_centerline_offsets_mm": torch.zeros(1, 1, 2, 3),
        "side_branch_relative_code_mm": empty_side_offsets,
        "coarse_centerline_parameters_mm": coarse_parameters,
        "coarse_decoded_vessel_mm": decoded,
        "coarse_side_centerline_offsets_mm": torch.zeros(1, 1, 2, 3),
        "coarse_side_branch_relative_code_mm": empty_side_offsets,
        "refinement_stage_centerline_parameters_mm": torch.stack(
            [first_stage_parameters, second_stage_parameters, final_parameters],
            dim=1,
        ),
        "refinement_stage_decoded_vessel_mm": torch.stack(
            [decoded, decoded, decoded], dim=1
        ),
        "refinement_stage_side_centerline_offsets_mm": torch.stack(
            [
                torch.zeros(1, 1, 2, 3),
                torch.zeros(1, 1, 2, 3),
                torch.zeros(1, 1, 2, 3),
            ],
            dim=1,
        ),
        "refinement_stage_side_branch_relative_code_mm": torch.stack(
            [empty_side_offsets, empty_side_offsets, empty_side_offsets], dim=1
        ),
        "bspline_refinement_residual_mm": torch.zeros(1, 3, 1, 2, 3),
        "coarse_branch_exist_logits": torch.zeros(1, 1),
        "coarse_branch_exist_probs": torch.ones(1, 1),
        "refinement_stage_branch_exist_logits": torch.zeros(1, 3, 1),
        "refinement_stage_branch_exist_probs": torch.ones(1, 3, 1),
        "bspline_refinement_branch_exist_residual_logits": torch.zeros(
            1, 3, 1
        ),
        "bspline_refiner_existence_candidate_mask": torch.zeros(
            1, 1, dtype=torch.bool
        ),
    }
    target = decoded.clone()
    batch = {
        "target_branch_exist": torch.ones(1, 1),
        "target_centerline_parameters_mm": torch.zeros(1, 1, 2, 3),
        "target_radius_baseline_coefficients_log_mm": torch.zeros(1, 1, 1),
        "target_lesion_exist": torch.zeros(1, 1, 0),
        "target_lesion_geometry": torch.zeros(1, 1, 0, 3),
        "target_attachment_index": torch.full((1, 1), -1),
        "target_raw_vessel_mm": target,
        "target_reconstructed_vessel_mm": target,
        "target_point_valid_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }
    losses = compute_parametric_loss(
        output,
        batch,
        {
            "centerline_prediction_mode": "bspline_control_points",
            "radius_prediction_mode": "parametric",
            "loss": {
                "control_point_scale_mm": 1.0,
                "centerline_control_loss_type": "mse",
                "centerline_control_loss_weight": 1.0,
                "branch_exist_loss_weight": 0.0,
                "radius_coefficient_loss_weight": 0.0,
                "lesion_exist_loss_weight": 0.0,
                "lesion_geometry_loss_weight": 0.0,
                "attachment_loss_weight": 0.0,
                "side_relative_xyz_loss_weight": 0.0,
                "decoded_xyz_loss_weight": 0.0,
                "decoded_radius_loss_weight": 0.0,
                "bspline_refiner_coarse_loss_weight": 0.1,
                "bspline_refiner_intermediate_loss_weight": 0.25,
            },
        },
    )

    assert losses["centerline_control_loss"] == pytest.approx(1.0)
    assert losses["bspline_refiner_coarse_auxiliary_loss"] == pytest.approx(4.0)
    assert losses["bspline_refiner_stage_1_auxiliary_loss"] == pytest.approx(9.0)
    assert losses["bspline_refiner_stage_2_auxiliary_loss"] == pytest.approx(25.0)
    assert losses["bspline_refiner_branch_existence_loss"] == pytest.approx(0.0)
    assert losses["loss"] == pytest.approx(
        1.0 + 0.1 * 4.0 + 0.25 * ((9.0 + 25.0) / 2.0)
    )

    import vessel_code.parametric.loss as loss_module

    recursive_calls = 0
    original_compute = loss_module.compute_parametric_loss

    def counted_compute(*args, **kwargs):
        nonlocal recursive_calls
        recursive_calls += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(
        loss_module, "compute_parametric_loss", counted_compute
    )
    zero_auxiliary_config = {
        "centerline_prediction_mode": "bspline_control_points",
        "radius_prediction_mode": "parametric",
        "loss": {
            "control_point_scale_mm": 1.0,
            "centerline_control_loss_type": "mse",
            "centerline_control_loss_weight": 1.0,
            "branch_exist_loss_weight": 0.0,
            "radius_coefficient_loss_weight": 0.0,
            "lesion_exist_loss_weight": 0.0,
            "lesion_geometry_loss_weight": 0.0,
            "attachment_loss_weight": 0.0,
            "side_relative_xyz_loss_weight": 0.0,
            "decoded_xyz_loss_weight": 0.0,
            "decoded_radius_loss_weight": 0.0,
            "bspline_refiner_coarse_loss_weight": 0.0,
            "bspline_refiner_intermediate_loss_weight": 0.0,
        },
    }
    zero_auxiliary_losses = counted_compute(
        output, batch, zero_auxiliary_config
    )

    assert recursive_calls == 1
    assert zero_auxiliary_losses[
        "bspline_refiner_coarse_auxiliary_loss"
    ] == pytest.approx(0.0)
    assert zero_auxiliary_losses[
        "bspline_refiner_intermediate_auxiliary_loss"
    ] == pytest.approx(0.0)

    def unexpected_projection(**_kwargs):
        raise AssertionError(
            "projection work must be skipped before its start epoch"
        )

    monkeypatch.setattr(
        loss_module,
        "compute_centerline_projection_losses",
        unexpected_projection,
    )
    scheduled_config = {
        **zero_auxiliary_config,
        "enable_projection_2d_loss": True,
        "proj_loss_start_epoch": 10,
    }
    scheduled_losses = original_compute(
        output,
        batch,
        scheduled_config,
        projector=object(),
        epoch=1,
    )
    assert "projection_2d_loss" not in scheduled_losses
