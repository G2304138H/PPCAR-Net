# Transferred from methods/parametric_methods/model.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import copy
import math
import time
from typing import Any, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vessel_code.preprocessing.vessel_code_transform_utils import bspline_basis_matrix
from vessel_code.geometry.differentiable_projector import DifferentiableVesselProjector, PreparedSurfaceGeometry, ProjectionCameras
from vessel_code.backbones.vggt_support.model_realdata import _ProjectionEvidenceRefiner
from vessel_code.parametric.centerline_heatmap import GAUSSIAN_CENTERLINE_TARGET, normalize_centerline_target_mode, render_centerline_heatmap_from_grid
from vessel_code.parametric.representation import constrain_lesion_geometry, decode_bspline_centerlines_local, decode_centerlines, decode_landmark_centerlines, decode_landmark_centerlines_local, decode_radii, decode_raw_radii, normalize_centerline_prediction_mode, normalize_lesion_profile, normalize_radius_prediction_mode
from vessel_code.shared.data import normalize_feature_backbone
from vessel_code.shared.branch_visibility import required_branch_count, resolve_artery_type
from vessel_code.shared.camera_encoding import build_camera_token_layout
from vessel_code.shared.model import ABSOLUTE_PARALLEL_DECODER, LEGACY_PARALLEL_DECODER, MAIN_FIRST_HIERARCHICAL_DECODER, PrecomputedFeatureVesselPredictor, normalize_decoder_architecture

def _synchronize_for_model_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

def _synchronize_for_refiner_timing(device: torch.device) -> None:
    _synchronize_for_model_timing(device)

class _RadiusRefinerProjectionContext(NamedTuple):
    """View- and geometry-invariant radius-refiner projection tensors."""

    input_masks: torch.Tensor
    centerline_m: torch.Tensor
    profile_grid: torch.Tensor
    valid_profiles: torch.Tensor
    theta_deg: torch.Tensor
    phi_deg: torch.Tensor
    cameras: ProjectionCameras
    surface_geometries: tuple[PreparedSurfaceGeometry, ...]
    branch_mask: torch.Tensor

class _BSplineControlPointProjectionRefiner(_ProjectionEvidenceRefiner):
    """Use local multi-view evidence to predict bounded control-point updates."""

    def __init__(
        self,
        *,
        model_dim: int,
        view_feat_dim: int,
        num_branches: int,
        num_control_points: int,
        num_attention_heads: int,
        evidence_hidden_dim: int,
        patch_size: int,
        use_learned_image_features: bool,
        learned_feature_dim: int,
        use_distance_transform: bool,
        distance_transform_num_iters: int,
        dropout: float,
        image_size: int,
        sid: float,
        source_to_iso: float,
        imager_pixel_spacing: float,
        residual_scale_mm: float,
        control_position_scale_mm: float,
        refine_branch_existence: bool,
        branch_existence_max_logit_decrease: float,
        coord_scale_to_meter: float,
        target_coordinate_frame: str,
        use_spatial_vggt_features: bool,
        spatial_vggt_feature_dim: int,
        vggt_token_dim: int,
        vggt_image_size_mode: str,
        vggt_patch_size: int,
        vggt_target_image_height: int,
        vggt_target_image_width: int,
        use_3d_candidates: bool,
        candidate_pattern: str,
        candidate_spacing_mm: tuple[float, ...],
        candidate_hidden_dim: int,
        candidate_num_attention_heads: int,
        candidate_score_temperature: float,
        use_unexplained_centerline_evidence: bool,
        use_centerline_probability_patch_evidence: bool,
        centerline_probability_patch_size: int,
        detach_centerline_probability_patch_evidence: bool,
        centerline_probability_target_mode: str,
        centerline_probability_target_gamma: float,
        centerline_probability_mask_threshold: float,
        centerline_probability_unexplained_power: float,
        use_separate_centerline_encoder: bool,
        centerline_map_size: int,
        centerline_head_hidden_dim: int,
        centerline_map_sigma_px: float,
        centerline_map_radius_px: int,
        unexplained_centerline_pool_kernel: int,
        detach_unexplained_centerline_evidence: bool,
        anchor_basis_matrix: torch.Tensor,
    ) -> None:
        super().__init__(
            model_dim=model_dim,
            view_feat_dim=view_feat_dim,
            num_attention_heads=num_attention_heads,
            evidence_hidden_dim=evidence_hidden_dim,
            patch_size=patch_size,
            use_learned_image_features=use_learned_image_features,
            learned_feature_dim=learned_feature_dim,
            use_distance_transform=use_distance_transform,
            distance_transform_num_iters=distance_transform_num_iters,
            dropout=dropout,
            image_size=image_size,
            sid=sid,
            source_to_iso=source_to_iso,
            imager_pixel_spacing=imager_pixel_spacing,
        )
        residual_scale_mm = float(residual_scale_mm)
        control_position_scale_mm = float(control_position_scale_mm)
        coord_scale_to_meter = float(coord_scale_to_meter)
        if self.image_size < 2:
            raise ValueError(
                "bspline_refiner_image_size must be >= 2, "
                f"got {self.image_size}."
            )
        if not math.isfinite(residual_scale_mm) or residual_scale_mm <= 0.0:
            raise ValueError(
                "bspline_refiner_residual_scale_mm must be finite and > 0, "
                f"got {residual_scale_mm}."
            )
        if (
            not math.isfinite(control_position_scale_mm)
            or control_position_scale_mm <= 0.0
        ):
            raise ValueError(
                "bspline_refiner_control_position_scale_mm must be finite and > 0, "
                f"got {control_position_scale_mm}."
            )
        if not math.isfinite(coord_scale_to_meter) or coord_scale_to_meter <= 0.0:
            raise ValueError(
                "projection_coord_scale_to_meter must be finite and > 0, "
                f"got {coord_scale_to_meter}."
            )
        target_coordinate_frame = str(target_coordinate_frame).strip().lower()
        if target_coordinate_frame not in {"projection_centered", "absolute_world"}:
            raise ValueError(
                "target_coordinate_frame must be 'projection_centered' or "
                f"'absolute_world', got {target_coordinate_frame!r}."
            )

        self.num_branches = int(num_branches)
        self.num_control_points = int(num_control_points)
        self.control_index_embedding = nn.Embedding(
            self.num_control_points, self.model_dim
        )
        self.branch_index_embedding = nn.Embedding(
            self.num_branches, self.model_dim
        )
        self.control_position_encoder = nn.Sequential(
            nn.Linear(3, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.query_fusion = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )
        self.distance_direction_encoder: nn.Module | None = (
            nn.Sequential(
                nn.Linear(2, self.model_dim),
                nn.GELU(),
                nn.LayerNorm(self.model_dim),
            )
            if self.use_distance_transform
            else None
        )
        self.use_spatial_vggt_features = bool(use_spatial_vggt_features)
        self.spatial_vggt_feature_dim = int(spatial_vggt_feature_dim)
        self.vggt_token_dim = int(vggt_token_dim)
        self.vggt_image_size_mode = str(vggt_image_size_mode).strip().lower()
        self.vggt_patch_size = int(vggt_patch_size)
        self.vggt_target_image_height = int(vggt_target_image_height)
        self.vggt_target_image_width = int(vggt_target_image_width)
        if self.vggt_image_size_mode not in {
            "pad_to_patch_multiple",
            "resize_to_patch_multiple",
            "error",
        }:
            raise ValueError(
                "vggt_image_size_mode must be one of "
                "['pad_to_patch_multiple', 'resize_to_patch_multiple', "
                f"'error'], got {vggt_image_size_mode!r}."
            )
        if self.vggt_patch_size < 1:
            raise ValueError(
                f"vggt_patch_size must be >= 1, got {self.vggt_patch_size}."
            )
        if (
            self.vggt_target_image_height < 1
            or self.vggt_target_image_width < 1
            or self.vggt_target_image_height % self.vggt_patch_size != 0
            or self.vggt_target_image_width % self.vggt_patch_size != 0
        ):
            raise ValueError(
                "VGGT target image dimensions must be positive multiples of "
                f"vggt_patch_size={self.vggt_patch_size}, got "
                f"{(self.vggt_target_image_height, self.vggt_target_image_width)}."
            )
        if self.vggt_token_dim < 1:
            raise ValueError(
                f"vggt_token_dim must be >= 1, got {self.vggt_token_dim}."
            )
        if self.spatial_vggt_feature_dim < 1:
            raise ValueError(
                "bspline_refiner_spatial_vggt_feature_dim must be >= 1, got "
                f"{self.spatial_vggt_feature_dim}."
            )
        if self.use_spatial_vggt_features:
            self.spatial_vggt_token_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.vggt_token_dim),
                nn.Linear(
                    self.vggt_token_dim,
                    self.spatial_vggt_feature_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(self.spatial_vggt_feature_dim),
            )
            self.spatial_vggt_evidence_projection: nn.Module | None = nn.Linear(
                self.spatial_vggt_feature_dim,
                self.model_dim,
            )
            # Loading an older refiner remains an exact functional warm start:
            # the newly sampled VGGT evidence initially contributes zero.
            nn.init.zeros_(self.spatial_vggt_evidence_projection.weight)
            nn.init.zeros_(self.spatial_vggt_evidence_projection.bias)
        else:
            self.spatial_vggt_token_projection = None
            self.spatial_vggt_evidence_projection = None
        self.use_3d_candidates = bool(use_3d_candidates)
        candidate_pattern = str(candidate_pattern).strip().lower()
        if candidate_pattern == "axis_7":
            candidate_unit_offsets = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float32,
            )
        elif candidate_pattern == "grid_27":
            axes = torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float32)
            zz, yy, xx = torch.meshgrid(axes, axes, axes, indexing="ij")
            grid_offsets = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
            center = torch.all(grid_offsets == 0.0, dim=-1)
            candidate_unit_offsets = torch.cat(
                (grid_offsets[center], grid_offsets[~center]), dim=0
            )
        else:
            raise ValueError(
                "bspline_refiner_candidate_pattern must be 'axis_7' or "
                f"'grid_27', got {candidate_pattern!r}."
            )
        inverse_candidate_indices = []
        for offset in candidate_unit_offsets:
            inverse_matches = torch.nonzero(
                torch.all(candidate_unit_offsets == -offset, dim=-1),
                as_tuple=False,
            ).reshape(-1)
            if int(inverse_matches.numel()) != 1:
                raise RuntimeError(
                    "Every 3D candidate offset must have exactly one symmetric "
                    f"inverse; failed for {offset.tolist()}."
                )
            inverse_candidate_indices.append(int(inverse_matches.item()))
        self.candidate_pattern = candidate_pattern
        spacing_values = tuple(float(value) for value in candidate_spacing_mm)
        if not spacing_values or any(
            not math.isfinite(value) or value <= 0.0
            for value in spacing_values
        ):
            raise ValueError(
                "bspline_refiner_candidate_spacing_mm must contain finite, "
                f"positive values, got {candidate_spacing_mm!r}."
            )
        self.candidate_hidden_dim = int(candidate_hidden_dim)
        self.candidate_num_attention_heads = int(
            candidate_num_attention_heads
        )
        self.candidate_score_temperature = float(candidate_score_temperature)
        if self.candidate_hidden_dim < 1:
            raise ValueError(
                "bspline_refiner_candidate_hidden_dim must be >= 1, got "
                f"{self.candidate_hidden_dim}."
            )
        if (
            self.candidate_num_attention_heads < 1
            or self.candidate_hidden_dim % self.candidate_num_attention_heads != 0
        ):
            raise ValueError(
                "bspline_refiner_candidate_num_attention_heads must be positive "
                "and divide bspline_refiner_candidate_hidden_dim, got "
                f"{self.candidate_num_attention_heads} and "
                f"{self.candidate_hidden_dim}."
            )
        if (
            not math.isfinite(self.candidate_score_temperature)
            or self.candidate_score_temperature <= 0.0
        ):
            raise ValueError(
                "bspline_refiner_candidate_score_temperature must be finite and "
                f"> 0, got {self.candidate_score_temperature}."
            )
        self.use_unexplained_centerline_evidence = bool(
            use_unexplained_centerline_evidence
        )
        self.use_centerline_probability_patch_evidence = bool(
            use_centerline_probability_patch_evidence
        )
        self.centerline_probability_patch_size = int(
            centerline_probability_patch_size
        )
        self.detach_centerline_probability_patch_evidence = bool(
            detach_centerline_probability_patch_evidence
        )
        self.centerline_probability_target_mode = normalize_centerline_target_mode(
            centerline_probability_target_mode
        )
        self.centerline_probability_target_gamma = float(
            centerline_probability_target_gamma
        )
        self.centerline_probability_mask_threshold = float(
            centerline_probability_mask_threshold
        )
        self.centerline_probability_unexplained_power = float(
            centerline_probability_unexplained_power
        )
        if (
            not math.isfinite(self.centerline_probability_target_gamma)
            or self.centerline_probability_target_gamma <= 0.0
        ):
            raise ValueError(
                "bspline_refiner_centerline_probability_target_gamma must be "
                "finite and > 0, got "
                f"{self.centerline_probability_target_gamma}."
            )
        if (
            not math.isfinite(self.centerline_probability_mask_threshold)
            or not 0.0 <= self.centerline_probability_mask_threshold <= 1.0
        ):
            raise ValueError(
                "bspline_refiner_centerline_probability_mask_threshold must "
                "be finite and in [0,1], got "
                f"{self.centerline_probability_mask_threshold}."
            )
        if (
            not math.isfinite(self.centerline_probability_unexplained_power)
            or self.centerline_probability_unexplained_power < 1.0
        ):
            raise ValueError(
                "bspline_refiner_centerline_probability_unexplained_power "
                "must be finite and >= 1, got "
                f"{self.centerline_probability_unexplained_power}."
            )
        self.uses_centerline_probability_predictor = bool(
            self.use_unexplained_centerline_evidence
            or self.use_centerline_probability_patch_evidence
        )
        self.use_separate_centerline_encoder = bool(
            use_separate_centerline_encoder
        )
        self.centerline_map_size = int(centerline_map_size)
        self.centerline_head_hidden_dim = int(centerline_head_hidden_dim)
        self.centerline_map_sigma_px = float(centerline_map_sigma_px)
        self.centerline_map_radius_px = int(centerline_map_radius_px)
        self.unexplained_centerline_pool_kernel = int(
            unexplained_centerline_pool_kernel
        )
        self.detach_unexplained_centerline_evidence = bool(
            detach_unexplained_centerline_evidence
        )
        if self.use_unexplained_centerline_evidence and not self.use_3d_candidates:
            raise ValueError(
                "bspline_refiner_use_unexplained_centerline_evidence=true "
                "requires bspline_refiner_use_3d_candidates=true."
            )
        if (
            self.uses_centerline_probability_predictor
            and self.image_feature_encoder is None
        ):
            raise ValueError(
                "Centreline-probability evidence requires "
                "bspline_refiner_use_learned_image_features=true."
            )
        if (
            self.centerline_probability_patch_size < 1
            or self.centerline_probability_patch_size % 2 == 0
        ):
            raise ValueError(
                "bspline_refiner_centerline_probability_patch_size must be an "
                "odd integer >= 1, got "
                f"{self.centerline_probability_patch_size}."
            )
        if self.centerline_map_size < 2:
            raise ValueError(
                "bspline_refiner_centerline_map_size must be >= 2, got "
                f"{self.centerline_map_size}."
            )
        if self.centerline_head_hidden_dim < 1:
            raise ValueError(
                "bspline_refiner_centerline_head_hidden_dim must be >= 1, got "
                f"{self.centerline_head_hidden_dim}."
            )
        if (
            not math.isfinite(self.centerline_map_sigma_px)
            or self.centerline_map_sigma_px <= 0.0
        ):
            raise ValueError(
                "bspline_refiner_centerline_map_sigma_px must be finite "
                f"and > 0, got {self.centerline_map_sigma_px}."
            )
        if self.centerline_map_radius_px < 0:
            raise ValueError(
                "bspline_refiner_centerline_map_radius_px must be >= 0, "
                f"got {self.centerline_map_radius_px}."
            )
        if (
            self.unexplained_centerline_pool_kernel < 1
            or self.unexplained_centerline_pool_kernel % 2 == 0
        ):
            raise ValueError(
                "bspline_refiner_unexplained_centerline_pool_kernel must be "
                "an odd integer >= 1, got "
                f"{self.unexplained_centerline_pool_kernel}."
            )
        if self.use_3d_candidates:
            anchor_basis_matrix = anchor_basis_matrix.detach().to(
                dtype=torch.float64, device="cpu"
            )
            if anchor_basis_matrix.shape != (
                self.num_control_points,
                self.num_control_points,
            ):
                raise ValueError(
                    "Candidate anchor basis must have shape [C,C], got "
                    f"{tuple(anchor_basis_matrix.shape)} for "
                    f"C={self.num_control_points}."
                )
            anchor_rank = int(
                torch.linalg.matrix_rank(anchor_basis_matrix).item()
            )
            anchor_condition = float(torch.linalg.cond(anchor_basis_matrix).item())
            if (
                anchor_rank != self.num_control_points
                or not math.isfinite(anchor_condition)
                or anchor_condition > 1.0e6
            ):
                raise ValueError(
                    "The B-spline candidate anchor basis must be invertible and "
                    "well-conditioned; got rank/size "
                    f"{anchor_rank}/{self.num_control_points} and condition "
                    f"{anchor_condition:.6g}."
                )
            anchor_to_control = torch.linalg.inv(anchor_basis_matrix)
        else:
            anchor_basis_matrix = torch.eye(
                self.num_control_points, dtype=torch.float64
            )
            anchor_to_control = anchor_basis_matrix
        self.register_buffer(
            "candidate_unit_offsets",
            candidate_unit_offsets,
            persistent=False,
        )
        self.register_buffer(
            "candidate_inverse_indices",
            torch.tensor(inverse_candidate_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "candidate_spacing_mm",
            torch.tensor(spacing_values, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "candidate_anchor_basis_matrix",
            anchor_basis_matrix.to(dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "candidate_anchor_to_control_matrix",
            anchor_to_control.to(dtype=torch.float32),
            persistent=False,
        )
        candidate_evidence_dim = 1 + self.view_feat_dim
        if self.use_distance_transform:
            candidate_evidence_dim += 3
        if self.image_feature_encoder is not None:
            candidate_evidence_dim += self.learned_feature_dim
        if self.use_spatial_vggt_features:
            candidate_evidence_dim += self.spatial_vggt_feature_dim
        self.candidate_evidence_dim = candidate_evidence_dim
        self.candidate_score_aux_dim = (
            2 if self.use_spatial_vggt_features else 1
        )
        self.centerline_image_feature_encoder: nn.Module | None = (
            copy.deepcopy(self.image_feature_encoder)
            if self.uses_centerline_probability_predictor
            and self.use_separate_centerline_encoder
            and self.image_feature_encoder is not None
            else None
        )
        if self.uses_centerline_probability_predictor:
            # Adding an ablation-only module must not advance the global RNG
            # and silently change every subsequently initialized refiner
            # weight under the same training seed.
            with torch.random.fork_rng(devices=[]):
                self.input_centerline_probability_head: nn.Module | None = (
                    nn.Sequential(
                        nn.Conv2d(
                            self.learned_feature_dim,
                            self.centerline_head_hidden_dim,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            self.centerline_head_hidden_dim,
                            1,
                            kernel_size=1,
                        ),
                    )
                )
                self.candidate_centerline_evidence_projection: nn.Module | None = (
                    nn.Linear(3, self.candidate_hidden_dim)
                    if self.use_unexplained_centerline_evidence
                    else None
                )
                if self.candidate_centerline_evidence_projection is not None:
                    # Preserve the exact function of a trained step-3 candidate
                    # checkpoint. The new maps begin contributing only as this
                    # adapter learns, without changing the old encoder width.
                    nn.init.zeros_(
                        self.candidate_centerline_evidence_projection.weight
                    )
                    nn.init.zeros_(
                        self.candidate_centerline_evidence_projection.bias
                    )
                self.centerline_probability_patch_projection: nn.Module | None = (
                    nn.Sequential(
                        nn.LayerNorm(
                            self.centerline_probability_patch_size
                            * self.centerline_probability_patch_size
                        ),
                        nn.Linear(
                            self.centerline_probability_patch_size
                            * self.centerline_probability_patch_size,
                            self.model_dim,
                        ),
                    )
                    if self.use_centerline_probability_patch_evidence
                    else None
                )
                if self.centerline_probability_patch_projection is not None:
                    # This additive adapter makes an older refiner an exact
                    # functional warm start until the new evidence is learned.
                    nn.init.zeros_(
                        self.centerline_probability_patch_projection[-1].weight
                    )
                    nn.init.zeros_(
                        self.centerline_probability_patch_projection[-1].bias
                    )
        else:
            self.input_centerline_probability_head = None
            self.candidate_centerline_evidence_projection = None
            self.centerline_probability_patch_projection = None
        if self.use_3d_candidates:
            self.candidate_evidence_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(candidate_evidence_dim, self.candidate_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.candidate_hidden_dim),
                nn.Linear(self.candidate_hidden_dim, self.candidate_hidden_dim),
            )
            self.candidate_query_projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.model_dim),
                nn.Linear(self.model_dim, self.candidate_hidden_dim),
            )
            self.candidate_offset_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(3, self.candidate_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.candidate_hidden_dim),
            )
            self.candidate_stage_embedding: nn.Module | None = nn.Embedding(
                len(spacing_values), self.candidate_hidden_dim
            )
            self.candidate_view_attention: nn.Module | None = nn.MultiheadAttention(
                self.candidate_hidden_dim,
                self.candidate_num_attention_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.candidate_query_norm: nn.Module | None = nn.LayerNorm(
                self.candidate_hidden_dim
            )
            self.candidate_output_norm: nn.Module | None = nn.LayerNorm(
                self.candidate_hidden_dim
            )
            self.candidate_ffn: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.candidate_hidden_dim),
                nn.Linear(self.candidate_hidden_dim, 2 * self.candidate_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(2 * self.candidate_hidden_dim, self.candidate_hidden_dim),
            )
            self.candidate_score_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(
                    self.candidate_hidden_dim + self.candidate_score_aux_dim
                ),
                nn.Linear(
                    self.candidate_hidden_dim + self.candidate_score_aux_dim,
                    self.candidate_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.candidate_hidden_dim, 1),
            )
            # Uniform scores over every symmetric candidate set give exactly
            # zero initial displacement while retaining immediate gradients.
            nn.init.zeros_(self.candidate_score_head[-1].weight)
            nn.init.zeros_(self.candidate_score_head[-1].bias)
        else:
            self.candidate_evidence_encoder = None
            self.candidate_query_projection = None
            self.candidate_offset_encoder = None
            self.candidate_stage_embedding = None
            self.candidate_view_attention = None
            self.candidate_query_norm = None
            self.candidate_output_norm = None
            self.candidate_ffn = None
            self.candidate_score_head = None
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 3),
        )
        # An enabled refiner initially behaves exactly like the coarse model.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.refine_branch_existence = bool(refine_branch_existence)
        branch_existence_max_logit_decrease = float(
            branch_existence_max_logit_decrease
        )
        if (
            not math.isfinite(branch_existence_max_logit_decrease)
            or branch_existence_max_logit_decrease <= 0.0
        ):
            raise ValueError(
                "bspline_refiner_branch_existence_max_logit_decrease must be "
                "finite and > 0, got "
                f"{branch_existence_max_logit_decrease}."
            )
        self.branch_existence_max_logit_decrease = (
            branch_existence_max_logit_decrease
        )
        if self.refine_branch_existence:
            self.branch_existence_logit_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(1, self.model_dim),
                nn.GELU(),
                nn.LayerNorm(self.model_dim),
            )
            self.branch_existence_residual_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.model_dim),
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
                nn.Linear(self.model_dim, 1),
            )
            # The optional existence refiner also starts as an exact identity.
            # Each stage proposes a signed update.  The caller bounds the
            # cumulative result to [coarse - maximum, coarse], which permits a
            # later stage to undo an over-aggressive earlier removal without
            # ever increasing a branch above its coarse existence score.
            nn.init.zeros_(self.branch_existence_residual_head[-1].weight)
            nn.init.zeros_(self.branch_existence_residual_head[-1].bias)
        else:
            self.branch_existence_logit_encoder = None
            self.branch_existence_residual_head = None
        self.register_buffer(
            "residual_scale_mm",
            torch.tensor(residual_scale_mm, dtype=torch.float32),
            persistent=True,
        )
        self.control_position_scale_mm = control_position_scale_mm
        self.coord_scale_to_meter = coord_scale_to_meter
        self.target_coordinate_frame = target_coordinate_frame

    def prepare_image_maps(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute image-only evidence once and reuse it across refinement stages."""
        if images.dim() == 4:
            batch_size, num_views, image_h, image_w = images.shape
            image_in = images.reshape(
                batch_size * num_views, 1, image_h, image_w
            )
        elif images.dim() == 5:
            batch_size, num_views, _, image_h, image_w = images.shape
            image_in = images[:, :, :1].reshape(
                batch_size * num_views, 1, image_h, image_w
            )
        else:
            raise ValueError(
                "images must have shape [B,V,H,W] or [B,V,C,H,W], got "
                f"{tuple(images.shape)}."
            )
        distance_map = (
            self._distance_transform_2d(image_in)
            if self.use_distance_transform
            else None
        )
        feature_map = (
            self.image_feature_encoder(image_in)
            if self.image_feature_encoder is not None
            else None
        )
        return distance_map, feature_map

    def prepare_input_centerline_probability_map(
        self,
        prepared_feature_map: torch.Tensor | None,
        *,
        batch_size: int,
        num_views: int,
        images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Predict a low-resolution 2D centreline map once per input view."""

        if not self.uses_centerline_probability_predictor:
            return None, None
        centerline_feature_map = prepared_feature_map
        if self.centerline_image_feature_encoder is not None:
            if images is None:
                raise ValueError(
                    "The separate centreline encoder requires input images."
                )
            if images.ndim == 4:
                image_in = images.reshape(
                    int(batch_size) * int(num_views),
                    1,
                    int(images.shape[-2]),
                    int(images.shape[-1]),
                )
            elif images.ndim == 5:
                image_in = images[:, :, :1].reshape(
                    int(batch_size) * int(num_views),
                    1,
                    int(images.shape[-2]),
                    int(images.shape[-1]),
                )
            else:
                raise ValueError(
                    "Separate centreline encoder images must have shape "
                    f"[B,V,H,W] or [B,V,C,H,W], got {tuple(images.shape)}."
                )
            centerline_feature_map = self.centerline_image_feature_encoder(
                image_in
            )
        if centerline_feature_map is None:
            raise ValueError(
                "Centreline-probability evidence requires the prepared learned "
                "image feature map."
            )
        expected_leading = int(batch_size) * int(num_views)
        if int(centerline_feature_map.shape[0]) != expected_leading:
            raise ValueError(
                "Prepared learned image features do not match the current "
                "batch/view shape: got leading size "
                f"{int(centerline_feature_map.shape[0])}, expected "
                f"{expected_leading}."
            )
        assert self.input_centerline_probability_head is not None
        compact_features = F.interpolate(
            centerline_feature_map,
            size=(self.centerline_map_size, self.centerline_map_size),
            mode="bilinear",
            align_corners=True,
        )
        logits = self.input_centerline_probability_head(compact_features)
        return logits, torch.sigmoid(logits)

    def _sample_centerline_probability_patch(
        self,
        *,
        probability_map: torch.Tensor,
        points_m: torch.Tensor,
        views: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a local patch from the predicted map for every point/view.

        Patch offsets are measured in centreline-map pixels, not raw detector
        pixels. For a 64x64 map and patch size 9, each point therefore sees a
        9x9 neighbourhood whose outer sample centres span roughly 32x32 pixels
        of a 256x256 input.
        """

        grid, _ = self._project_points_to_grid(points_m=points_m, views=views)
        batch_size, num_views, num_points, _ = grid.shape
        expected_shape = (
            batch_size * num_views,
            1,
            self.centerline_map_size,
            self.centerline_map_size,
        )
        if tuple(probability_map.shape) != expected_shape:
            raise ValueError(
                "Prepared centreline probability map has shape "
                f"{tuple(probability_map.shape)}, expected {expected_shape}."
            )
        sample_map = (
            probability_map.detach()
            if self.detach_centerline_probability_patch_evidence
            else probability_map
        )
        patch_radius = self.centerline_probability_patch_size // 2
        offsets = torch.arange(
            -patch_radius,
            patch_radius + 1,
            device=grid.device,
            dtype=grid.dtype,
        )
        normalized_offsets = offsets * (
            2.0 / float(self.centerline_map_size - 1)
        )
        yy, xx = torch.meshgrid(
            normalized_offsets,
            normalized_offsets,
            indexing="ij",
        )
        patch_offsets = torch.stack((xx, yy), dim=-1).reshape(
            1,
            1,
            1,
            self.centerline_probability_patch_size
            * self.centerline_probability_patch_size,
            2,
        )
        patch_grid = grid.unsqueeze(-2) + patch_offsets
        sampled = F.grid_sample(
            sample_map,
            patch_grid.reshape(
                batch_size * num_views,
                num_points
                * self.centerline_probability_patch_size
                * self.centerline_probability_patch_size,
                1,
                2,
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled[:, 0, :, 0].reshape(
            batch_size,
            num_views,
            num_points,
            self.centerline_probability_patch_size
            * self.centerline_probability_patch_size,
        )
        return sampled.permute(0, 2, 1, 3)

    def prepare_unexplained_centerline_evidence_map(
        self,
        *,
        input_centerline_probability_map: torch.Tensor | None,
        current_centerlines_mm: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        branch_mask: torch.Tensor,
        projection_center_offset_mm: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build input, explained, and residual centreline evidence maps.

        This renders centreline points only, not vessel radii or tube surfaces.
        The residual is positive evidence for image centreline locations that
        the current 3D prediction does not yet explain; it is not a penalty for
        two true branches that overlap within one projection.
        """

        if not self.use_unexplained_centerline_evidence:
            return None, None, None
        if input_centerline_probability_map is None:
            raise ValueError(
                "Unexplained-centerline evidence requires the predicted input "
                "centreline probability map."
            )
        if current_centerlines_mm.dim() != 4 or int(
            current_centerlines_mm.shape[-1]
        ) != 3:
            raise ValueError(
                "current_centerlines_mm must have shape [B,M,N,3], got "
                f"{tuple(current_centerlines_mm.shape)}."
            )
        batch_size, num_branches, num_curve_points, _ = (
            current_centerlines_mm.shape
        )
        num_views = int(views.shape[1])
        if branch_mask.shape != (batch_size, num_branches):
            raise ValueError(
                "branch_mask must match current centreline branches, got "
                f"{tuple(branch_mask.shape)} and expected "
                f"{(batch_size, num_branches)}."
            )
        expected_map_shape = (
            batch_size * num_views,
            1,
            self.centerline_map_size,
            self.centerline_map_size,
        )
        if tuple(input_centerline_probability_map.shape) != expected_map_shape:
            raise ValueError(
                "Input centreline probability map has shape "
                f"{tuple(input_centerline_probability_map.shape)}, expected "
                f"{expected_map_shape}."
            )

        evidence_centerlines_mm = current_centerlines_mm
        evidence_input_map = input_centerline_probability_map
        if self.detach_unexplained_centerline_evidence:
            # This evidence is an observation for the candidate scorer, not a
            # second recursive geometry loss. Avoid retaining a four-stage
            # rasterization graph while leaving the separately returned head
            # logits attached for their explicit auxiliary supervision.
            evidence_centerlines_mm = evidence_centerlines_mm.detach()
            evidence_input_map = evidence_input_map.detach()
        centered_points_mm = evidence_centerlines_mm
        if self.target_coordinate_frame == "absolute_world":
            if projection_center_offset_mm is None:
                raise ValueError(
                    "Unexplained-centerline evidence in absolute_world "
                    "coordinates requires projection_center_offset."
                )
            centered_points_mm = centered_points_mm - (
                projection_center_offset_mm[:, None, None, :]
            )
        projected_grid, geometry_valid = self._project_points_to_grid(
            points_m=centered_points_mm.reshape(batch_size, -1, 3)
            * self.coord_scale_to_meter,
            views=views,
        )
        active_points = branch_mask.to(
            device=geometry_valid.device, dtype=torch.bool
        ).unsqueeze(-1).expand(
            batch_size, num_branches, num_curve_points
        ).reshape(batch_size, -1)
        valid = geometry_valid & active_points[:, None, :]
        if view_mask is not None:
            if view_mask.shape != (batch_size, num_views):
                raise ValueError(
                    "view_mask must have shape [B,V], got "
                    f"{tuple(view_mask.shape)} and expected "
                    f"{(batch_size, num_views)}."
                )
            valid_views = view_mask.to(
                device=valid.device, dtype=torch.bool
            )
            valid = valid & valid_views.unsqueeze(-1)
        else:
            valid_views = torch.ones(
                (batch_size, num_views),
                device=valid.device,
                dtype=torch.bool,
            )

        predicted_map = render_centerline_heatmap_from_grid(
            projected_grid,
            valid,
            map_size=self.centerline_map_size,
            sigma_px=self.centerline_map_sigma_px,
            radius_px=self.centerline_map_radius_px,
        ).reshape(
            batch_size * num_views,
            1,
            self.centerline_map_size,
            self.centerline_map_size,
        )
        valid_view_map = valid_views.reshape(
            batch_size * num_views, 1, 1, 1
        ).to(dtype=predicted_map.dtype)
        input_map = evidence_input_map * valid_view_map
        unexplained_source = input_map.pow(
            self.centerline_probability_unexplained_power
        )
        unexplained_map = unexplained_source * (1.0 - predicted_map)
        if self.unexplained_centerline_pool_kernel > 1:
            unexplained_map = F.max_pool2d(
                unexplained_map,
                kernel_size=self.unexplained_centerline_pool_kernel,
                stride=1,
                padding=self.unexplained_centerline_pool_kernel // 2,
            )
        evidence_map = torch.cat(
            (input_map, predicted_map, unexplained_map), dim=1
        )
        if self.detach_unexplained_centerline_evidence:
            evidence_map = evidence_map.detach()
        return evidence_map, predicted_map.detach(), unexplained_map.detach()

    def prepare_spatial_vggt_map(
        self,
        image_features: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Project cached VGGT patch tokens once into a compact spatial map.

        VGGT may prepend camera/register tokens. Only the final patch-grid
        tokens have a detector location and are therefore retained here.
        """

        if not self.use_spatial_vggt_features:
            return None
        if isinstance(image_features, dict):
            raise ValueError(
                "Spatial VGGT refiner evidence requires tensor image_features "
                "with shape [B,V,L,D], not a feature pyramid dictionary."
            )
        if image_features.dim() != 4:
            raise ValueError(
                "Spatial VGGT refiner evidence requires image_features with "
                f"shape [B,V,L,D], got {tuple(image_features.shape)}."
            )
        if int(image_features.shape[-1]) != self.vggt_token_dim:
            raise ValueError(
                "Cached VGGT token width does not match vggt_token_dim: got "
                f"{int(image_features.shape[-1])}, expected {self.vggt_token_dim}."
            )
        assert self.spatial_vggt_token_projection is not None
        reference_parameter = next(
            self.spatial_vggt_token_projection.parameters()
        )
        image_features = image_features.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        num_locations = int(image_features.shape[2])
        layout = build_camera_token_layout(
            feature_backbone="vggt",
            num_locations=num_locations,
            image_height=self.image_size,
            image_width=self.image_size,
            resnet_pool_size=0,
            vggt_patch_size=self.vggt_patch_size,
            vggt_target_image_height=self.vggt_target_image_height,
            vggt_target_image_width=self.vggt_target_image_width,
            device=image_features.device,
            dtype=torch.float32,
        )
        spatial_count = (
            self.vggt_target_image_height // self.vggt_patch_size
        ) * (self.vggt_target_image_width // self.vggt_patch_size)
        spatial_mask = layout.spatial_mask
        if int(spatial_mask.sum().item()) != spatial_count:
            raise RuntimeError(
                "VGGT token layout did not identify the expected number of "
                f"patch tokens: got {int(spatial_mask.sum().item())}, "
                f"expected {spatial_count}."
            )
        prefix_count = num_locations - spatial_count
        expected_mask = torch.arange(
            num_locations, device=image_features.device
        ) >= prefix_count
        if not torch.equal(spatial_mask, expected_mask):
            raise RuntimeError(
                "Spatial VGGT sampling requires prefix/register tokens followed "
                "by the row-major patch grid."
            )
        spatial_tokens = image_features[:, :, spatial_mask]
        compact_tokens = self.spatial_vggt_token_projection(spatial_tokens)
        batch_size, num_views = compact_tokens.shape[:2]
        grid_height = self.vggt_target_image_height // self.vggt_patch_size
        grid_width = self.vggt_target_image_width // self.vggt_patch_size
        return compact_tokens.reshape(
            batch_size,
            num_views,
            grid_height,
            grid_width,
            self.spatial_vggt_feature_dim,
        ).permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_views,
            self.spatial_vggt_feature_dim,
            grid_height,
            grid_width,
        )

    def _raw_projection_grid_to_vggt_grid(
        self,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Map raw-image ``align_corners=True`` coordinates to VGGT patches."""

        raw_x = (grid[..., 0] + 1.0) * (float(self.image_size - 1) / 2.0)
        raw_y = (grid[..., 1] + 1.0) * (float(self.image_size - 1) / 2.0)
        if self.vggt_image_size_mode == "resize_to_patch_multiple":
            # F.interpolate(..., align_corners=False) preserves this normalized
            # pixel-centre coordinate even when 256 is resized to 266.
            norm_x = 2.0 * (raw_x + 0.5) / float(self.image_size) - 1.0
            norm_y = 2.0 * (raw_y + 0.5) / float(self.image_size) - 1.0
        else:
            if self.vggt_image_size_mode == "error" and (
                self.image_size != self.vggt_target_image_height
                or self.image_size != self.vggt_target_image_width
            ):
                raise ValueError(
                    "vggt_image_size_mode='error' requires the refiner image "
                    "dimensions to equal the cached VGGT target dimensions."
                )
            if (
                self.image_size > self.vggt_target_image_height
                or self.image_size > self.vggt_target_image_width
            ):
                raise ValueError(
                    "pad_to_patch_multiple cannot map a refiner image larger "
                    "than the cached VGGT target image."
                )
            norm_x = (
                2.0 * (raw_x + 0.5) / float(self.vggt_target_image_width) - 1.0
            )
            norm_y = (
                2.0 * (raw_y + 0.5) / float(self.vggt_target_image_height) - 1.0
            )
        return torch.stack((norm_x, norm_y), dim=-1)

    def _sample_spatial_vggt_features(
        self,
        *,
        spatial_map: torch.Tensor,
        points_m: torch.Tensor,
        views: torch.Tensor,
    ) -> torch.Tensor:
        grid, _ = self._project_points_to_grid(points_m=points_m, views=views)
        batch_size, num_views, num_points, _ = grid.shape
        if int(spatial_map.shape[0]) != batch_size * num_views:
            raise ValueError(
                "Prepared spatial VGGT map does not match the current batch/view "
                f"shape: got leading size {int(spatial_map.shape[0])}, expected "
                f"{batch_size * num_views}."
            )
        feature_grid = self._raw_projection_grid_to_vggt_grid(grid)
        sampled = F.grid_sample(
            spatial_map,
            feature_grid.reshape(batch_size * num_views, num_points, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled[:, :, :, 0].transpose(1, 2).reshape(
            batch_size,
            num_views,
            num_points,
            self.spatial_vggt_feature_dim,
        )
        return sampled.permute(0, 2, 1, 3)

    @staticmethod
    def prepare_distance_gradient_map(
        distance_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if distance_map is None:
            return None
        gradient_x = torch.zeros_like(distance_map)
        gradient_y = torch.zeros_like(distance_map)
        gradient_x[..., 1:-1] = 0.5 * (
            distance_map[..., 2:] - distance_map[..., :-2]
        )
        gradient_x[..., 0] = distance_map[..., 1] - distance_map[..., 0]
        gradient_x[..., -1] = distance_map[..., -1] - distance_map[..., -2]
        gradient_y[..., 1:-1, :] = 0.5 * (
            distance_map[..., 2:, :] - distance_map[..., :-2, :]
        )
        gradient_y[..., 0, :] = distance_map[..., 1, :] - distance_map[..., 0, :]
        gradient_y[..., -1, :] = distance_map[..., -1, :] - distance_map[..., -2, :]
        return torch.cat([gradient_x, gradient_y], dim=1)

    def _sample_distance_direction(
        self,
        *,
        distance_gradient_map: torch.Tensor,
        points_m: torch.Tensor,
        views: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the local detector-space direction toward/away from the mask."""
        grid, _ = self._project_points_to_grid(points_m=points_m, views=views)
        batch_size, num_views, num_points, _ = grid.shape
        sampled = F.grid_sample(
            distance_gradient_map,
            grid.reshape(batch_size * num_views, num_points, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled[:, :, :, 0].transpose(1, 2).reshape(
            batch_size, num_views, num_points, 2
        )
        # The distance map is diagonal-normalized; restore an approximately
        # pixel-scaled derivative before the learned projection.
        return sampled.permute(0, 2, 1, 3) * float(self.image_size)

    @staticmethod
    def _sample_candidate_map(
        feature_map: torch.Tensor,
        *,
        grid: torch.Tensor,
        batch_size: int,
        num_views: int,
        num_points: int,
        num_candidates: int,
        align_corners: bool,
        padding_mode: str,
    ) -> torch.Tensor:
        num_channels = int(feature_map.shape[1])
        sampled = F.grid_sample(
            feature_map,
            grid.reshape(
                batch_size * num_views,
                num_points * num_candidates,
                1,
                2,
            ),
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return sampled[:, :, :, 0].transpose(1, 2).reshape(
            batch_size,
            num_views,
            num_points,
            num_candidates,
            num_channels,
        ).permute(0, 2, 3, 1, 4)

    def _sample_candidate_evidence(
        self,
        *,
        candidate_points_m: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        prepared_distance_map: torch.Tensor | None,
        prepared_distance_gradient_map: torch.Tensor | None,
        prepared_feature_map: torch.Tensor | None,
        prepared_spatial_vggt_map: torch.Tensor | None,
        prepared_unexplained_centerline_evidence_map: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Sample compact centre evidence for every 3D candidate in one pass."""

        batch_size, num_points, num_candidates, _ = candidate_points_m.shape
        num_views = int(views.shape[1])
        if images.dim() == 4:
            image_h, image_w = images.shape[-2:]
            image_in = images.reshape(
                batch_size * num_views, 1, image_h, image_w
            )
        elif images.dim() == 5:
            image_h, image_w = images.shape[-2:]
            image_in = images[:, :, :1].reshape(
                batch_size * num_views, 1, image_h, image_w
            )
        else:
            raise ValueError(
                "images must have shape [B,V,H,W] or [B,V,C,H,W], got "
                f"{tuple(images.shape)}."
            )
        flat_points_m = candidate_points_m.reshape(
            batch_size, num_points * num_candidates, 3
        )
        grid, valid = self._project_points_to_grid(
            points_m=flat_points_m,
            views=views,
        )
        pieces = [
            self._sample_candidate_map(
                image_in,
                grid=grid,
                batch_size=batch_size,
                num_views=num_views,
                num_points=num_points,
                num_candidates=num_candidates,
                align_corners=True,
                padding_mode="zeros",
            )
        ]
        if self.use_distance_transform:
            if (
                prepared_distance_map is None
                or prepared_distance_gradient_map is None
            ):
                raise ValueError(
                    "3D candidate evidence requires prepared distance and "
                    "distance-gradient maps when distance evidence is enabled."
                )
            pieces.append(
                self._sample_candidate_map(
                    prepared_distance_map,
                    grid=grid,
                    batch_size=batch_size,
                    num_views=num_views,
                    num_points=num_points,
                    num_candidates=num_candidates,
                    align_corners=True,
                    padding_mode="border",
                )
            )
            pieces.append(
                self._sample_candidate_map(
                    prepared_distance_gradient_map,
                    grid=grid,
                    batch_size=batch_size,
                    num_views=num_views,
                    num_points=num_points,
                    num_candidates=num_candidates,
                    align_corners=True,
                    padding_mode="border",
                )
                * float(self.image_size)
            )
        if self.image_feature_encoder is not None:
            if prepared_feature_map is None:
                raise ValueError(
                    "3D candidate evidence requires the prepared learned image "
                    "feature map when learned evidence is enabled."
                )
            pieces.append(
                self._sample_candidate_map(
                    prepared_feature_map,
                    grid=grid,
                    batch_size=batch_size,
                    num_views=num_views,
                    num_points=num_points,
                    num_candidates=num_candidates,
                    align_corners=True,
                    padding_mode="zeros",
                )
            )
        sampled_spatial_vggt: torch.Tensor | None = None
        if self.use_spatial_vggt_features:
            if prepared_spatial_vggt_map is None:
                raise ValueError(
                    "3D candidate evidence requires prepared spatial VGGT maps."
                )
            vggt_grid = self._raw_projection_grid_to_vggt_grid(grid)
            sampled_spatial_vggt = self._sample_candidate_map(
                prepared_spatial_vggt_map,
                grid=vggt_grid,
                batch_size=batch_size,
                num_views=num_views,
                num_points=num_points,
                num_candidates=num_candidates,
                align_corners=False,
                padding_mode="border",
            )
            pieces.append(sampled_spatial_vggt)
        sampled_centerline_evidence: torch.Tensor | None = None
        if self.use_unexplained_centerline_evidence:
            if prepared_unexplained_centerline_evidence_map is None:
                raise ValueError(
                    "3D candidate scoring requires prepared unexplained-"
                    "centerline evidence maps when that experiment is enabled."
                )
            sampled_centerline_evidence = self._sample_candidate_map(
                prepared_unexplained_centerline_evidence_map,
                grid=grid,
                batch_size=batch_size,
                num_views=num_views,
                num_points=num_points,
                num_candidates=num_candidates,
                align_corners=True,
                padding_mode="zeros",
            )
        view_features = views[:, :, None, None, :].expand(
            batch_size,
            num_views,
            num_points,
            num_candidates,
            self.view_feat_dim,
        ).permute(0, 2, 3, 1, 4)
        pieces.append(view_features)
        evidence = torch.cat(pieces, dim=-1)
        candidate_valid = valid.reshape(
            batch_size, num_views, num_points, num_candidates
        ).permute(0, 2, 3, 1)
        return (
            evidence,
            candidate_valid,
            sampled_spatial_vggt,
            sampled_centerline_evidence,
        )

    def _control_to_anchor_residual(
        self,
        control_residual_mm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "ac,bmcd->bmad",
            self.candidate_anchor_basis_matrix.to(control_residual_mm),
            control_residual_mm,
        )

    def _anchor_to_control_residual(
        self,
        anchor_residual_mm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "ca,bmad->bmcd",
            self.candidate_anchor_to_control_matrix.to(anchor_residual_mm),
            anchor_residual_mm,
        )

    def _score_3d_candidates(
        self,
        *,
        stage_index: int,
        candidate_centers_mm: torch.Tensor,
        point_state: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        evidence_controls: torch.Tensor,
        prepared_distance_map: torch.Tensor | None,
        prepared_distance_gradient_map: torch.Tensor | None,
        prepared_feature_map: torch.Tensor | None,
        prepared_spatial_vggt_map: torch.Tensor | None,
        prepared_unexplained_centerline_evidence_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_3d_candidates:
            raise RuntimeError("3D candidate scoring was called while disabled.")
        if not 0 <= int(stage_index) < int(self.candidate_spacing_mm.numel()):
            raise ValueError(
                f"Candidate stage index {stage_index} is outside the configured "
                f"{int(self.candidate_spacing_mm.numel())} spacings."
            )
        batch_size, num_branches, num_controls, _ = candidate_centers_mm.shape
        num_points = num_branches * num_controls
        spacing_mm = self.candidate_spacing_mm[int(stage_index)].to(
            candidate_centers_mm
        )
        offsets_mm = self.candidate_unit_offsets.to(candidate_centers_mm) * spacing_mm
        num_candidates = int(offsets_mm.shape[0])
        candidate_points_mm = (
            candidate_centers_mm.reshape(batch_size, num_points, 1, 3)
            + offsets_mm.reshape(1, 1, num_candidates, 3)
        )
        candidate_points_m = candidate_points_mm * self.coord_scale_to_meter
        (
            evidence_raw,
            valid,
            sampled_spatial_vggt,
            sampled_centerline_evidence,
        ) = self._sample_candidate_evidence(
            candidate_points_m=candidate_points_m,
            images=images,
            views=views,
            prepared_distance_map=prepared_distance_map,
            prepared_distance_gradient_map=prepared_distance_gradient_map,
            prepared_feature_map=prepared_feature_map,
            prepared_spatial_vggt_map=prepared_spatial_vggt_map,
            prepared_unexplained_centerline_evidence_map=(
                prepared_unexplained_centerline_evidence_map
            ),
        )
        valid = valid & evidence_controls.reshape(
            batch_size, num_points, 1, 1
        )
        if view_mask is not None:
            valid = valid & view_mask.to(
                device=valid.device, dtype=torch.bool
            )[:, None, None, :]

        assert self.candidate_evidence_encoder is not None
        assert self.candidate_query_projection is not None
        assert self.candidate_offset_encoder is not None
        assert self.candidate_stage_embedding is not None
        assert self.candidate_view_attention is not None
        assert self.candidate_query_norm is not None
        assert self.candidate_output_norm is not None
        assert self.candidate_ffn is not None
        assert self.candidate_score_head is not None
        evidence = self.candidate_evidence_encoder(evidence_raw)
        if sampled_centerline_evidence is not None:
            assert self.candidate_centerline_evidence_projection is not None
            evidence = evidence + self.candidate_centerline_evidence_projection(
                sampled_centerline_evidence
            )
        max_spacing = self.candidate_spacing_mm.max().to(offsets_mm)
        offset_tokens = self.candidate_offset_encoder(
            offsets_mm / max_spacing.clamp_min(1.0e-6)
        )
        stage_id = torch.tensor(
            int(stage_index), device=point_state.device, dtype=torch.long
        )
        stage_token = self.candidate_stage_embedding(stage_id)
        query = self.candidate_query_projection(
            point_state.reshape(batch_size, num_points, self.model_dim)
        )[:, :, None, :] + offset_tokens[None, None, :, :] + stage_token
        query = query.reshape(
            batch_size * num_points * num_candidates,
            1,
            self.candidate_hidden_dim,
        )
        evidence = evidence.reshape(
            batch_size * num_points * num_candidates,
            int(views.shape[1]),
            self.candidate_hidden_dim,
        )
        key_padding_mask = ~valid.reshape(
            batch_size * num_points * num_candidates,
            int(views.shape[1]),
        )
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask = key_padding_mask.clone()
        evidence = evidence.clone()
        key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_masked
        evidence[:, 0] = torch.where(
            all_masked[:, None],
            torch.zeros_like(evidence[:, 0]),
            evidence[:, 0],
        )
        attended, _ = self.candidate_view_attention(
            self.candidate_query_norm(query),
            evidence,
            evidence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        candidate_state = self.candidate_output_norm(
            query + self.dropout(attended)
        )
        candidate_state = candidate_state + self.dropout(
            self.candidate_ffn(candidate_state)
        )
        valid_fraction = valid.to(dtype=candidate_state.dtype).mean(
            dim=-1, keepdim=True
        ).reshape(-1, 1, 1)
        score_auxiliary = [valid_fraction]
        if sampled_spatial_vggt is not None:
            normalized_vggt = F.normalize(
                sampled_spatial_vggt,
                p=2.0,
                dim=-1,
                eps=1.0e-6,
            )
            pairwise_similarity = torch.einsum(
                "bnkvd,bnkwd->bnkvw",
                normalized_vggt,
                normalized_vggt,
            )
            pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
            pair_valid = pair_valid & ~torch.eye(
                int(views.shape[1]),
                device=valid.device,
                dtype=torch.bool,
            ).reshape(1, 1, 1, int(views.shape[1]), int(views.shape[1]))
            pair_count = pair_valid.sum(dim=(-2, -1), keepdim=True)
            mean_similarity = (
                (pairwise_similarity * pair_valid.to(pairwise_similarity)).sum(
                    dim=(-2, -1), keepdim=True
                )
                / pair_count.clamp_min(1).to(pairwise_similarity)
            )
            mean_similarity = torch.where(
                pair_count > 0,
                mean_similarity,
                torch.zeros_like(mean_similarity),
            ).reshape(-1, 1, 1)
            score_auxiliary.append(mean_similarity)
        score_input = torch.cat(
            (candidate_state, *score_auxiliary), dim=-1
        )
        scores = self.candidate_score_head(score_input).reshape(
            batch_size, num_points, num_candidates
        )
        observable = valid.any(dim=-1)
        # Keep candidate availability symmetric so zero-initialized, uniform
        # scores still have exactly zero displacement near image boundaries.
        # A directional candidate is usable only when its opposite direction
        # is also observable. The zero-offset candidate is always a safe
        # fallback, including when every projection is outside/padded.
        symmetric_observable = observable & observable.index_select(
            dim=-1,
            index=self.candidate_inverse_indices,
        )
        symmetric_observable = symmetric_observable.clone()
        symmetric_observable[..., 0] = True
        scores = scores.masked_fill(
            ~symmetric_observable,
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(
            scores / self.candidate_score_temperature,
            dim=-1,
        )
        anchor_residual_mm = torch.einsum(
            "bnk,kd->bnd", probabilities, offsets_mm
        ).reshape(batch_size, num_branches, num_controls, 3)
        anchor_residual_mm = torch.where(
            evidence_controls.unsqueeze(-1),
            anchor_residual_mm,
            torch.zeros_like(anchor_residual_mm),
        )
        control_residual_mm = self._anchor_to_control_residual(
            anchor_residual_mm
        )
        return (
            control_residual_mm,
            anchor_residual_mm,
            probabilities.reshape(
                batch_size, num_branches, num_controls, num_candidates
            ),
        )

    def forward(
        self,
        *,
        branch_tokens: torch.Tensor,
        control_points_mm: torch.Tensor,
        anchor_points_mm: torch.Tensor,
        current_centerlines_mm: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        point_state: torch.Tensor | None,
        projection_center_offset_mm: torch.Tensor | None,
        prepared_distance_map: torch.Tensor | None,
        prepared_distance_gradient_map: torch.Tensor | None,
        prepared_feature_map: torch.Tensor | None,
        prepared_spatial_vggt_map: torch.Tensor | None,
        prepared_input_centerline_probability_map: torch.Tensor | None,
        stage_index: int,
        branch_mask: torch.Tensor | None = None,
        centerline_evidence_branch_mask: torch.Tensor | None = None,
        existence_candidate_mask: torch.Tensor | None = None,
        current_branch_exist_logits: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size, num_branches, num_controls, _ = control_points_mm.shape
        expected = (self.num_branches, self.num_control_points)
        if (num_branches, num_controls) != expected:
            raise ValueError(
                "B-spline refiner control shape does not match its configuration: "
                f"got {(num_branches, num_controls)}, expected {expected}."
            )
        if anchor_points_mm.shape != control_points_mm.shape:
            raise ValueError(
                "anchor_points_mm must match control_points_mm, got "
                f"{tuple(anchor_points_mm.shape)} and {tuple(control_points_mm.shape)}."
            )
        if int(images.shape[-2]) != self.image_size or int(images.shape[-1]) != self.image_size:
            raise ValueError(
                "B-spline refiner image dimensions must match "
                f"bspline_refiner_image_size={self.image_size}, got "
                f"{tuple(images.shape[-2:])}."
            )

        points_mm = anchor_points_mm
        if self.target_coordinate_frame == "absolute_world":
            if projection_center_offset_mm is None:
                raise ValueError(
                    "B-spline refinement in absolute_world coordinates requires "
                    "projection_center_offset."
                )
            points_mm = points_mm - projection_center_offset_mm[:, None, None, :]
        points_m = (
            points_mm.reshape(batch_size, num_branches * num_controls, 3)
            * self.coord_scale_to_meter
        )
        evidence_raw, valid = self._sample_features(
            images=images,
            views=views,
            points_m=points_m,
            prepared_distance_map=prepared_distance_map,
            prepared_feature_map=prepared_feature_map,
        )
        if branch_mask is None:
            branch_mask = torch.ones(
                (batch_size, num_branches),
                device=control_points_mm.device,
                dtype=torch.bool,
            )
        else:
            branch_mask = branch_mask.to(
                device=control_points_mm.device, dtype=torch.bool
            )
            if branch_mask.shape != (batch_size, num_branches):
                raise ValueError(
                    "B-spline refiner branch_mask must have shape [B,M], got "
                    f"{tuple(branch_mask.shape)} and expected "
                    f"{(batch_size, num_branches)}."
                )
        active_controls = branch_mask.unsqueeze(-1).expand(
            batch_size, num_branches, num_controls
        )
        if self.refine_branch_existence:
            if existence_candidate_mask is None:
                raise ValueError(
                    "Existence refinement requires existence_candidate_mask."
                )
            existence_candidate_mask = existence_candidate_mask.to(
                device=control_points_mm.device,
                dtype=torch.bool,
            )
            if existence_candidate_mask.shape != (batch_size, num_branches):
                raise ValueError(
                    "existence_candidate_mask must have shape [B,M], got "
                    f"{tuple(existence_candidate_mask.shape)} and expected "
                    f"{(batch_size, num_branches)}."
                )
            evidence_controls = (
                branch_mask | existence_candidate_mask
            ).unsqueeze(-1).expand(batch_size, num_branches, num_controls)
        else:
            evidence_controls = active_controls
        if centerline_evidence_branch_mask is None:
            centerline_evidence_branch_mask = branch_mask
        else:
            centerline_evidence_branch_mask = centerline_evidence_branch_mask.to(
                device=control_points_mm.device,
                dtype=torch.bool,
            )
            if centerline_evidence_branch_mask.shape != (
                batch_size,
                num_branches,
            ):
                raise ValueError(
                    "centerline_evidence_branch_mask must have shape [B,M], "
                    f"got {tuple(centerline_evidence_branch_mask.shape)} and "
                    f"expected {(batch_size, num_branches)}."
                )
        (
            prepared_unexplained_centerline_evidence_map,
            predicted_centerline_map,
            unexplained_centerline_map,
        ) = self.prepare_unexplained_centerline_evidence_map(
            input_centerline_probability_map=(
                prepared_input_centerline_probability_map
            ),
            current_centerlines_mm=current_centerlines_mm,
            views=views,
            view_mask=view_mask,
            branch_mask=centerline_evidence_branch_mask,
            projection_center_offset_mm=projection_center_offset_mm,
        )
        # Geometry gating and existence evidence have different jobs.  In the
        # legacy path an inactive branch is removed from attention entirely.
        # With existence refinement enabled, every candidate branch must still
        # see its projected image evidence so a GT-absent coarse false positive
        # can be classified for removal; only its XYZ residual remains gated.
        valid = valid & evidence_controls.reshape(
            batch_size, num_branches * num_controls, 1
        )
        evidence = self.evidence_encoder(evidence_raw)
        if self.use_centerline_probability_patch_evidence:
            if prepared_input_centerline_probability_map is None:
                raise ValueError(
                    "Prepared centreline probability maps are required when "
                    "bspline_refiner_use_centerline_probability_patch_evidence="
                    "true."
                )
            assert self.centerline_probability_patch_projection is not None
            centerline_probability_patch = (
                self._sample_centerline_probability_patch(
                    probability_map=prepared_input_centerline_probability_map,
                    points_m=points_m,
                    views=views,
                )
            )
            evidence = evidence + self.centerline_probability_patch_projection(
                centerline_probability_patch
            )
        if self.use_spatial_vggt_features:
            if prepared_spatial_vggt_map is None:
                raise ValueError(
                    "Prepared spatial VGGT features are required when "
                    "bspline_refiner_use_spatial_vggt_features=true."
                )
            assert self.spatial_vggt_evidence_projection is not None
            sampled_vggt = self._sample_spatial_vggt_features(
                spatial_map=prepared_spatial_vggt_map,
                points_m=points_m,
                views=views,
            )
            evidence = evidence + self.spatial_vggt_evidence_projection(
                sampled_vggt
            )
        if self.distance_direction_encoder is not None:
            if prepared_distance_gradient_map is None:
                raise ValueError(
                    "Prepared distance-gradient maps are required when B-spline "
                    "refiner distance-transform evidence is enabled."
                )
            distance_direction = self._sample_distance_direction(
                distance_gradient_map=prepared_distance_gradient_map,
                points_m=points_m,
                views=views,
            )
            evidence = evidence + self.distance_direction_encoder(
                distance_direction
            )

        control_ids = torch.arange(num_controls, device=control_points_mm.device)
        branch_ids = torch.arange(num_branches, device=control_points_mm.device)
        index_tokens = self.control_index_embedding(control_ids)[None, None]
        branch_index_tokens = self.branch_index_embedding(branch_ids)[None, :, None]
        point_tokens = (
            branch_tokens.unsqueeze(2)
            + index_tokens
            + branch_index_tokens
            + self.control_position_encoder(
                control_points_mm / self.control_position_scale_mm
            )
        )
        if point_state is not None:
            if point_state.shape != point_tokens.shape:
                raise ValueError(
                    "point_state must match control query shape, got "
                    f"{tuple(point_state.shape)} and {tuple(point_tokens.shape)}."
                )
            point_tokens = point_tokens + point_state
        point_tokens = self.query_fusion(point_tokens)
        flat_tokens = point_tokens.reshape(
            batch_size * num_branches * num_controls, 1, self.model_dim
        )
        num_views = int(evidence.shape[2])
        if view_mask is not None:
            valid = valid & view_mask.to(
                device=valid.device, dtype=torch.bool
            ).unsqueeze(1)
        key_padding_mask = ~valid.reshape(
            batch_size * num_branches * num_controls, num_views
        )
        evidence = evidence.reshape(
            batch_size * num_branches * num_controls, num_views, self.model_dim
        )
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask = key_padding_mask.clone()
        evidence = evidence.clone()
        key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_masked
        evidence[:, 0] = torch.where(
            all_masked[:, None], torch.zeros_like(evidence[:, 0]), evidence[:, 0]
        )

        attended, _ = self.attn(
            self.norm_q(flat_tokens),
            evidence,
            evidence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        next_state = self.norm_out(flat_tokens + self.dropout(attended))
        next_state = next_state + self.dropout(self.ffn(next_state))
        next_state = next_state.reshape(
            batch_size, num_branches, num_controls, self.model_dim
        )
        residual_mm = torch.tanh(self.residual_head(next_state)) * (
            self.residual_scale_mm.to(control_points_mm)
        )
        candidate_control_residual_mm: torch.Tensor | None = None
        candidate_anchor_residual_mm: torch.Tensor | None = None
        candidate_probabilities: torch.Tensor | None = None
        if self.use_3d_candidates:
            preliminary_anchor_points_mm = points_mm + (
                self._control_to_anchor_residual(residual_mm)
            )
            (
                candidate_control_residual_mm,
                candidate_anchor_residual_mm,
                candidate_probabilities,
            ) = self._score_3d_candidates(
                stage_index=stage_index,
                candidate_centers_mm=preliminary_anchor_points_mm,
                point_state=next_state,
                images=images,
                views=views,
                view_mask=view_mask,
                evidence_controls=evidence_controls,
                prepared_distance_map=prepared_distance_map,
                prepared_distance_gradient_map=prepared_distance_gradient_map,
                prepared_feature_map=prepared_feature_map,
                prepared_spatial_vggt_map=prepared_spatial_vggt_map,
                prepared_unexplained_centerline_evidence_map=(
                    prepared_unexplained_centerline_evidence_map
                ),
            )
            residual_mm = residual_mm + candidate_control_residual_mm
        branch_existence_logit_residual: torch.Tensor | None = None
        if self.branch_existence_residual_head is not None:
            if current_branch_exist_logits is None:
                raise ValueError(
                    "Existence refinement requires current branch logits."
                )
            if current_branch_exist_logits.shape != (batch_size, num_branches):
                raise ValueError(
                    "current_branch_exist_logits must have shape [B,M], got "
                    f"{tuple(current_branch_exist_logits.shape)} and expected "
                    f"{(batch_size, num_branches)}."
                )
            assert self.branch_existence_logit_encoder is not None
            branch_state = next_state.mean(dim=2)
            branch_state = branch_state + self.branch_existence_logit_encoder(
                current_branch_exist_logits.unsqueeze(-1)
            )
            raw_logit_residual = self.branch_existence_residual_head(
                branch_state
            ).squeeze(-1)
            branch_existence_logit_residual = torch.tanh(
                raw_logit_residual
            ) * self.branch_existence_max_logit_decrease
        residual_mm = torch.where(
            active_controls.unsqueeze(-1),
            residual_mm,
            torch.zeros_like(residual_mm),
        )
        next_state = torch.where(
            evidence_controls.unsqueeze(-1),
            next_state,
            torch.zeros_like(next_state),
        )
        return (
            control_points_mm + residual_mm,
            next_state,
            residual_mm,
            branch_existence_logit_residual,
            candidate_control_residual_mm,
            candidate_anchor_residual_mm,
            candidate_probabilities,
            predicted_centerline_map,
            unexplained_centerline_map,
        )

class _RadiusSurfaceProjectionRefiner(_ProjectionEvidenceRefiner):
    """Iteratively correct dense raw radii from rendered-mask discrepancies."""

    def __init__(
        self,
        *,
        model_dim: int,
        view_feat_dim: int,
        num_branches: int,
        num_points: int,
        num_attention_heads: int,
        evidence_hidden_dim: int,
        profile_samples: int,
        profile_half_width_px: float,
        dropout: float,
        image_size: int,
        sid: float,
        source_to_iso: float,
        imager_pixel_spacing: float,
        residual_scale_mm: float,
        radius_value_scale_mm: float,
        min_radius_mm: float,
        coord_scale_to_meter: float,
        target_coordinate_frame: str,
        render_num_circle_points: int,
        render_radial_subsamples: int,
        render_axial_subsamples: int,
    ) -> None:
        profile_samples = int(profile_samples)
        if profile_samples < 3 or profile_samples % 2 == 0:
            raise ValueError(
                "radius_refiner_profile_samples must be an odd integer >= 3, "
                f"got {profile_samples}."
            )
        super().__init__(
            model_dim=model_dim,
            view_feat_dim=view_feat_dim,
            num_attention_heads=num_attention_heads,
            evidence_hidden_dim=evidence_hidden_dim,
            patch_size=profile_samples,
            use_learned_image_features=False,
            learned_feature_dim=1,
            use_distance_transform=False,
            distance_transform_num_iters=1,
            dropout=dropout,
            image_size=image_size,
            sid=sid,
            source_to_iso=source_to_iso,
            imager_pixel_spacing=imager_pixel_spacing,
        )
        self.num_branches = int(num_branches)
        self.num_points = int(num_points)
        self.profile_samples = profile_samples
        self.profile_half_width_px = float(profile_half_width_px)
        self.radius_value_scale_mm = float(radius_value_scale_mm)
        self.min_radius_mm = float(min_radius_mm)
        self.coord_scale_to_meter = float(coord_scale_to_meter)
        self.target_coordinate_frame = str(target_coordinate_frame).strip().lower()
        for name, value in (
            ("radius_refiner_profile_half_width_px", self.profile_half_width_px),
            ("radius_refiner_residual_scale_mm", float(residual_scale_mm)),
            ("radius_refiner_radius_value_scale_mm", self.radius_value_scale_mm),
            ("radius_refiner_min_radius_mm", self.min_radius_mm),
            ("projection_coord_scale_to_meter", self.coord_scale_to_meter),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}.")
        if self.target_coordinate_frame not in {
            "projection_centered",
            "absolute_world",
        }:
            raise ValueError(
                "target_coordinate_frame must be 'projection_centered' or "
                f"'absolute_world', got {target_coordinate_frame!r}."
            )

        # Four transverse profiles: input mask, rendered surface, signed
        # difference, and absolute difference, followed by the view encoding.
        evidence_input_dim = 4 * self.profile_samples + self.view_feat_dim
        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_input_dim, int(evidence_hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(evidence_hidden_dim)),
            nn.Linear(int(evidence_hidden_dim), self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.point_index_embedding = nn.Embedding(self.num_points, self.model_dim)
        self.branch_index_embedding = nn.Embedding(
            self.num_branches, self.model_dim
        )
        self.radius_encoder = nn.Sequential(
            nn.Linear(1, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.query_fusion = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )
        self.sequence_norm = nn.LayerNorm(self.model_dim)
        self.sequence_conv_1 = nn.Conv1d(
            self.model_dim,
            self.model_dim,
            kernel_size=5,
            padding=2,
        )
        self.sequence_conv_2 = nn.Conv1d(
            self.model_dim,
            self.model_dim,
            kernel_size=5,
            padding=2,
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, int(evidence_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(evidence_hidden_dim), 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer(
            "residual_scale_mm",
            torch.tensor(float(residual_scale_mm), dtype=torch.float32),
            persistent=True,
        )
        self.surface_projector = DifferentiableVesselProjector(
            image_size=int(image_size),
            sid=float(sid),
            source_to_iso=float(source_to_iso),
            imager_pixel_spacing=float(imager_pixel_spacing),
            num_circle_points=int(render_num_circle_points),
            radial_subsamples=int(render_radial_subsamples),
            axial_subsamples=int(render_axial_subsamples),
            center_main_branch=False,
            crop_mode="none",
        )

    @staticmethod
    def _input_masks(images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            return images
        if images.dim() == 5:
            return images[:, :, 0]
        raise ValueError(
            "images must have shape [B,V,H,W] or [B,V,C,H,W], got "
            f"{tuple(images.shape)}."
        )

    @staticmethod
    def _view_angles_deg(views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.rad2deg(torch.atan2(views[..., 0], views[..., 1]))
        phi = torch.rad2deg(torch.atan2(views[..., 2], views[..., 3]))
        return theta, phi

    def _projection_vessel_m(
        self,
        vessel_mm: torch.Tensor,
        projection_center_offset_mm: torch.Tensor | None,
    ) -> torch.Tensor:
        vessel_m = torch.cat(
            [vessel_mm[..., :3].detach(), vessel_mm[..., 3:4]],
            dim=-1,
        ) * self.coord_scale_to_meter
        if self.target_coordinate_frame == "absolute_world":
            if projection_center_offset_mm is None:
                raise ValueError(
                    "Radius refinement in absolute_world coordinates requires "
                    "projection_center_offset."
                )
            vessel_m = vessel_m.clone()
            vessel_m[..., :3] = vessel_m[..., :3] - (
                projection_center_offset_mm[:, None, None, :]
                * self.coord_scale_to_meter
            )
        return vessel_m

    def render_masks(
        self,
        *,
        vessel_mm: torch.Tensor,
        views: torch.Tensor,
        projection_center_offset_mm: torch.Tensor | None,
        projection_context: _RadiusRefinerProjectionContext | None = None,
        branch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if projection_context is not None:
            if branch_mask is not None and not torch.equal(
                branch_mask.to(
                    device=projection_context.branch_mask.device,
                    dtype=torch.bool,
                ),
                projection_context.branch_mask,
            ):
                raise ValueError(
                    "branch_mask disagrees with the prepared radius-refiner "
                    "projection context."
                )
            branch_mask = projection_context.branch_mask
        if projection_context is None:
            vessel_m = self._projection_vessel_m(
                vessel_mm,
                projection_center_offset_mm,
            )
        else:
            vessel_m = torch.cat(
                [
                    projection_context.centerline_m,
                    vessel_mm[..., 3:4] * self.coord_scale_to_meter,
                ],
                dim=-1,
            )
        if projection_context is None:
            theta_deg, phi_deg = self._view_angles_deg(views)
        else:
            theta_deg = projection_context.theta_deg
            phi_deg = projection_context.phi_deg
        cameras = (
            None if projection_context is None else projection_context.cameras
        )
        prepared_geometries = (
            None
            if projection_context is None
            else projection_context.surface_geometries
        )
        return self.surface_projector.render_multiview_batch(
            vessel_m,
            theta_deg,
            phi_deg,
            cameras=cameras,
            prepared_geometries=prepared_geometries,
            branch_mask=branch_mask,
        )[:, :, 0]

    def prepare_projection_context(
        self,
        *,
        vessel_mm: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        projection_center_offset_mm: torch.Tensor | None,
        branch_mask: torch.Tensor | None = None,
    ) -> _RadiusRefinerProjectionContext:
        """Prepare tensors that are unchanged by radius-only refinement."""

        input_masks = self._input_masks(images).to(
            device=vessel_mm.device,
            dtype=vessel_mm.dtype,
        ).clamp(0.0, 1.0)
        batch_size, num_views, height, width = input_masks.shape
        if branch_mask is None:
            branch_mask = torch.ones(
                (batch_size, self.num_branches),
                device=vessel_mm.device,
                dtype=torch.bool,
            )
        else:
            branch_mask = branch_mask.to(
                device=vessel_mm.device,
                dtype=torch.bool,
            )
            if branch_mask.shape != (batch_size, self.num_branches):
                raise ValueError(
                    "Radius-refiner branch_mask must have shape [B,M], got "
                    f"{tuple(branch_mask.shape)} and expected "
                    f"{(batch_size, self.num_branches)}."
                )
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                "Radius-refiner image dimensions must match "
                f"radius_refiner_image_size={self.image_size}, got "
                f"{(height, width)}."
            )
        vessel_m = self._projection_vessel_m(
            vessel_mm,
            projection_center_offset_mm,
        )
        points_m = vessel_m[..., :3].reshape(
            batch_size,
            self.num_branches * self.num_points,
            3,
        )
        center_grid, valid = self._project_points_to_grid(
            points_m=points_m,
            views=views,
        )
        center_grid = center_grid.reshape(
            batch_size,
            num_views,
            self.num_branches,
            self.num_points,
            2,
        )
        valid = valid.reshape(
            batch_size,
            num_views,
            self.num_branches,
            self.num_points,
        )
        valid = valid & branch_mask[:, None, :, None]

        tangent = torch.zeros_like(center_grid)
        tangent[..., 0, :] = center_grid[..., 1, :] - center_grid[..., 0, :]
        tangent[..., -1, :] = center_grid[..., -1, :] - center_grid[..., -2, :]
        if self.num_points > 2:
            tangent[..., 1:-1, :] = 0.5 * (
                center_grid[..., 2:, :] - center_grid[..., :-2, :]
            )
        tangent = self._safe_normalize(tangent)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        pixel_to_grid = 2.0 / float(max(self.image_size - 1, 1))
        offsets = torch.linspace(
            -self.profile_half_width_px,
            self.profile_half_width_px,
            steps=self.profile_samples,
            device=vessel_mm.device,
            dtype=vessel_mm.dtype,
        ) * pixel_to_grid
        profile_grid = (
            center_grid.unsqueeze(-2)
            + normal.unsqueeze(-2) * offsets.view(1, 1, 1, 1, -1, 1)
        ).reshape(
            batch_size * num_views,
            self.num_branches * self.num_points,
            self.profile_samples,
            2,
        )
        theta_deg, phi_deg = self._view_angles_deg(views)
        cameras = self.surface_projector.prepare_cameras(
            theta_deg.reshape(-1).to(vessel_mm),
            phi_deg.reshape(-1).to(vessel_mm),
        )
        surface_geometries = tuple(
            self.surface_projector.prepare_surface_geometry(vessel_m[index])
            for index in range(batch_size)
        )
        return _RadiusRefinerProjectionContext(
            input_masks=input_masks,
            centerline_m=vessel_m[..., :3],
            profile_grid=profile_grid,
            valid_profiles=valid.permute(0, 2, 3, 1),
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            cameras=cameras,
            surface_geometries=surface_geometries,
            branch_mask=branch_mask,
        )

    def _sample_transverse_profiles(
        self,
        *,
        input_masks: torch.Tensor,
        rendered_masks: torch.Tensor,
        views: torch.Tensor,
        profile_grid: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, height, width = input_masks.shape

        signed_difference = input_masks - rendered_masks
        evidence_maps = torch.stack(
            [
                input_masks,
                rendered_masks,
                signed_difference,
                signed_difference.abs(),
            ],
            dim=2,
        ).reshape(batch_size * num_views, 4, height, width)
        sampled = F.grid_sample(
            evidence_maps,
            profile_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 1, 3).reshape(
            batch_size,
            num_views,
            self.num_branches,
            self.num_points,
            4 * self.profile_samples,
        )
        view_features = views[:, :, None, None, :].expand(
            batch_size,
            num_views,
            self.num_branches,
            self.num_points,
            self.view_feat_dim,
        )
        evidence = torch.cat([sampled, view_features], dim=-1)
        return evidence.permute(0, 2, 3, 1, 4), valid

    def forward(
        self,
        *,
        branch_tokens: torch.Tensor,
        vessel_mm: torch.Tensor,
        rendered_masks: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        point_state: torch.Tensor | None,
        projection_center_offset_mm: torch.Tensor | None,
        projection_context: _RadiusRefinerProjectionContext | None = None,
        branch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_vessel_shape = (
            vessel_mm.shape[0],
            self.num_branches,
            self.num_points,
            4,
        )
        if vessel_mm.shape != expected_vessel_shape:
            raise ValueError(
                "Radius-refiner vessel shape does not match its configuration: "
                f"got {tuple(vessel_mm.shape)}, expected {expected_vessel_shape}."
            )
        if projection_context is None:
            projection_context = self.prepare_projection_context(
                vessel_mm=vessel_mm,
                images=images,
                views=views,
                projection_center_offset_mm=projection_center_offset_mm,
                branch_mask=branch_mask,
            )
        elif branch_mask is not None and not torch.equal(
            branch_mask.to(
                device=projection_context.branch_mask.device,
                dtype=torch.bool,
            ),
            projection_context.branch_mask,
        ):
            raise ValueError(
                "branch_mask disagrees with the prepared radius-refiner "
                "projection context."
            )
        branch_mask = projection_context.branch_mask
        evidence_raw, valid = self._sample_transverse_profiles(
            input_masks=projection_context.input_masks,
            rendered_masks=rendered_masks,
            views=views,
            profile_grid=projection_context.profile_grid,
            valid=projection_context.valid_profiles,
        )
        evidence = self.evidence_encoder(evidence_raw)
        batch_size, num_branches, num_points, num_views = evidence.shape[:4]
        if view_mask is not None:
            valid = valid & view_mask.to(
                device=valid.device,
                dtype=torch.bool,
            )[:, None, None, :]

        point_ids = torch.arange(num_points, device=vessel_mm.device)
        branch_ids = torch.arange(num_branches, device=vessel_mm.device)
        query = (
            branch_tokens[:, :, None, :]
            + self.point_index_embedding(point_ids)[None, None, :, :]
            + self.branch_index_embedding(branch_ids)[None, :, None, :]
            + self.radius_encoder(
                vessel_mm[..., 3:4] / self.radius_value_scale_mm
            )
        )
        if point_state is not None:
            if point_state.shape != query.shape:
                raise ValueError(
                    "point_state must match the radius query shape, got "
                    f"{tuple(point_state.shape)} and {tuple(query.shape)}."
                )
            query = query + point_state
        query = self.query_fusion(query)

        flat_count = batch_size * num_branches * num_points
        evidence = evidence.reshape(flat_count, num_views, self.model_dim)
        key_padding_mask = ~valid.reshape(flat_count, num_views)
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask = key_padding_mask.clone()
        evidence = evidence.clone()
        key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~all_masked
        evidence[:, 0] = torch.where(
            all_masked[:, None], torch.zeros_like(evidence[:, 0]), evidence[:, 0]
        )
        query_flat = query.reshape(flat_count, 1, self.model_dim)
        attended, _ = self.attn(
            self.norm_q(query_flat),
            evidence,
            evidence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        next_state = self.norm_out(query_flat + self.dropout(attended))
        next_state = next_state + self.dropout(self.ffn(next_state))
        next_state = next_state.reshape(
            batch_size,
            num_branches,
            num_points,
            self.model_dim,
        )
        sequence_input = self.sequence_norm(next_state).reshape(
            batch_size * num_branches,
            num_points,
            self.model_dim,
        ).transpose(1, 2)
        sequence_update = self.sequence_conv_2(
            self.dropout(F.gelu(self.sequence_conv_1(sequence_input)))
        ).transpose(1, 2).reshape_as(next_state)
        next_state = next_state + self.dropout(sequence_update)

        residual_mm = torch.tanh(self.residual_head(next_state)[..., 0]) * (
            self.residual_scale_mm.to(vessel_mm)
        )
        active_points = branch_mask.unsqueeze(-1)
        residual_mm = torch.where(
            active_points,
            residual_mm,
            torch.zeros_like(residual_mm),
        )
        next_state = torch.where(
            active_points.unsqueeze(-1),
            next_state,
            torch.zeros_like(next_state),
        )
        active_refined_radius_mm = torch.clamp(
            vessel_mm[..., 3] + residual_mm,
            min=self.min_radius_mm,
        )
        refined_radius_mm = torch.where(
            active_points,
            active_refined_radius_mm,
            vessel_mm[..., 3],
        )
        return refined_radius_mm, next_state, residual_mm

class ParametricVesselPredictor(PrecomputedFeatureVesselPredictor):
    """Predict parametric centrelines with parametric or pointwise radii."""

    def __init__(
        self,
        *,
        feature_backbone: str,
        num_points: int = 200,
        num_branches: int = 7,
        num_control_points: int = 20,
        num_landmarks: int | None = None,
        num_radius_coefficients: int = 6,
        num_lesions: int = 3,
        lesion_profile: str = "gaussian",
        centerline_prediction_mode: str | None = "bspline_control_points",
        radius_prediction_mode: str | None = "parametric",
        centerline_spline_degree: int = 3,
        catmull_rom_alpha: float = 0.5,
        catmull_rom_dense_samples: int = 1000,
        radius_spline_degree: int = 3,
        control_point_output_scale_mm: float = 100.0,
        centerline_output_scale_mm: float | None = None,
        view_feat_dim: int = 4,
        model_dim: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 8,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_encoded_view_dir: bool = True,
        use_post_backbone_transformer_encoder: bool | None = None,
        use_ray_camera_encoding: bool = False,
        ray_camera_encoding_dim: int = 128,
        use_prope_attention: bool = False,
        prope_frequency_base: float = 100.0,
        camera_encoding_image_height: int = 256,
        camera_encoding_image_width: int = 256,
        camera_encoding_sid: float = 0.9,
        camera_encoding_source_to_iso: float = 0.75,
        camera_encoding_imager_pixel_spacing: float = 0.55,
        vggt_patch_size: int = 14,
        vggt_target_image_height: int = 266,
        vggt_target_image_width: int = 266,
        vggt_token_dim: int = 2048,
        vggt_image_size_mode: str = "resize_to_patch_multiple",
        resnet_pre_fpn_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        image_fpn_pool_size: int = 8,
        use_learned_side_parent_projection: bool = True,
        use_bspline_control_refiner: bool = False,
        bspline_refiner_num_stages: int = 4,
        bspline_refiner_evidence_hidden_dim: int = 256,
        bspline_refiner_patch_size: int = 9,
        bspline_refiner_use_learned_image_features: bool = True,
        bspline_refiner_learned_feature_dim: int = 32,
        bspline_refiner_use_distance_transform: bool = True,
        bspline_refiner_distance_transform_num_iters: int = 64,
        bspline_refiner_image_size: int = 256,
        bspline_refiner_sid: float = 0.9,
        bspline_refiner_source_to_iso: float = 0.75,
        bspline_refiner_imager_pixel_spacing: float = 0.55,
        bspline_refiner_residual_scale_mm: float = 5.0,
        bspline_refiner_control_position_scale_mm: float = 100.0,
        bspline_refiner_refine_branch_existence: bool = False,
        bspline_refiner_branch_existence_candidate_threshold: float = 0.5,
        bspline_refiner_branch_existence_max_logit_decrease: float = 10.0,
        bspline_refiner_use_spatial_vggt_features: bool = False,
        bspline_refiner_spatial_vggt_feature_dim: int = 64,
        bspline_refiner_use_3d_candidates: bool = False,
        bspline_refiner_candidate_pattern: str = "axis_7",
        bspline_refiner_candidate_spacing_mm: tuple[float, ...] = (
            4.0,
            2.0,
            1.0,
            0.5,
        ),
        bspline_refiner_candidate_hidden_dim: int = 64,
        bspline_refiner_candidate_num_attention_heads: int = 4,
        bspline_refiner_candidate_score_temperature: float = 1.0,
        bspline_refiner_use_unexplained_centerline_evidence: bool = False,
        bspline_refiner_use_centerline_probability_patch_evidence: bool = False,
        bspline_refiner_centerline_probability_patch_size: int = 9,
        bspline_refiner_detach_centerline_probability_patch_evidence: bool = True,
        bspline_refiner_centerline_probability_target_mode: str = (
            GAUSSIAN_CENTERLINE_TARGET
        ),
        bspline_refiner_centerline_probability_target_gamma: float = 2.0,
        bspline_refiner_centerline_probability_mask_threshold: float = 0.5,
        bspline_refiner_centerline_probability_unexplained_power: float = 1.0,
        bspline_refiner_use_separate_centerline_encoder: bool = False,
        bspline_refiner_centerline_map_size: int = 64,
        bspline_refiner_centerline_head_hidden_dim: int = 16,
        bspline_refiner_centerline_map_sigma_px: float = 1.25,
        bspline_refiner_centerline_map_radius_px: int = 2,
        bspline_refiner_unexplained_centerline_pool_kernel: int = 5,
        bspline_refiner_detach_unexplained_centerline_evidence: bool = True,
        use_radius_evidence_refiner: bool = False,
        radius_refiner_num_stages: int = 3,
        radius_refiner_evidence_hidden_dim: int = 256,
        radius_refiner_profile_samples: int = 25,
        radius_refiner_profile_half_width_px: float = 16.0,
        radius_refiner_image_size: int = 256,
        radius_refiner_sid: float = 0.9,
        radius_refiner_source_to_iso: float = 0.75,
        radius_refiner_imager_pixel_spacing: float = 0.55,
        radius_refiner_residual_scale_mm: float = 1.0,
        radius_refiner_radius_value_scale_mm: float = 5.0,
        radius_refiner_min_radius_mm: float = 0.05,
        radius_refiner_render_num_circle_points: int = 24,
        radius_refiner_render_radial_subsamples: int = 1,
        radius_refiner_render_axial_subsamples: int = 2,
        radius_refiner_branch_probability_threshold: float = 0.5,
        fixed_main_branch_count: int = 1,
        target_num_branches: int | None = None,
        projection_coord_scale_to_meter: float = 0.001,
        target_coordinate_frame: str = "projection_centered",
    ) -> None:
        super().__init__(
            feature_backbone=feature_backbone,
            num_points=num_points,
            num_branches=num_branches,
            view_feat_dim=view_feat_dim,
            model_dim=model_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            point_head_mode="point",
            use_encoded_view_dir=use_encoded_view_dir,
            use_post_backbone_transformer_encoder=(
                use_post_backbone_transformer_encoder
            ),
            use_ray_camera_encoding=use_ray_camera_encoding,
            ray_camera_encoding_dim=ray_camera_encoding_dim,
            use_prope_attention=use_prope_attention,
            prope_frequency_base=prope_frequency_base,
            camera_encoding_image_height=camera_encoding_image_height,
            camera_encoding_image_width=camera_encoding_image_width,
            camera_encoding_sid=camera_encoding_sid,
            camera_encoding_source_to_iso=camera_encoding_source_to_iso,
            camera_encoding_imager_pixel_spacing=(
                camera_encoding_imager_pixel_spacing
            ),
            vggt_patch_size=vggt_patch_size,
            vggt_target_image_height=vggt_target_image_height,
            vggt_target_image_width=vggt_target_image_width,
            vggt_token_dim=vggt_token_dim,
            resnet_pre_fpn_channels=resnet_pre_fpn_channels,
            image_fpn_pool_size=image_fpn_pool_size,
            use_learned_side_parent_projection=use_learned_side_parent_projection,
            use_centerline_point_index_embedding=False,
            use_radius_point_index_embedding=False,
            use_projection_evidence_residual_refiner=False,
        )
        del self.branch_point_head
        del self.side_branch_point_head
        self.centerline_prediction_mode = normalize_centerline_prediction_mode(
            centerline_prediction_mode
        )
        self.num_control_points = int(num_control_points)
        self.num_landmarks = int(
            self.num_control_points if num_landmarks is None else num_landmarks
        )
        self.num_centerline_parameters = (
            self.num_landmarks
            if self.centerline_prediction_mode == "adaptive_landmarks"
            else self.num_control_points
        )
        self.num_radius_coefficients = int(num_radius_coefficients)
        self.num_lesions = int(num_lesions)
        self.lesion_profile = normalize_lesion_profile(lesion_profile)
        self.radius_prediction_mode = normalize_radius_prediction_mode(
            radius_prediction_mode
        )
        self.centerline_output_scale_mm = float(
            control_point_output_scale_mm
            if centerline_output_scale_mm is None
            else centerline_output_scale_mm
        )
        # Retain the historical attribute for legacy checkpoint/config consumers.
        self.control_point_output_scale_mm = self.centerline_output_scale_mm
        self.catmull_rom_alpha = float(catmull_rom_alpha)
        self.catmull_rom_dense_samples = int(catmull_rom_dense_samples)
        if (
            self.centerline_prediction_mode == "bspline_control_points"
            and self.num_control_points < int(centerline_spline_degree) + 1
        ):
            raise ValueError("num_control_points must be at least centerline_spline_degree + 1")
        if (
            self.centerline_prediction_mode == "adaptive_landmarks"
            and self.num_landmarks < 4
        ):
            raise ValueError("num_landmarks must be at least four for Catmull--Rom decoding")
        if not 0.0 <= self.catmull_rom_alpha <= 1.0:
            raise ValueError("catmull_rom_alpha must be in [0,1]")
        if self.catmull_rom_dense_samples < 2:
            raise ValueError("catmull_rom_dense_samples must be at least two")
        if (
            self.radius_prediction_mode == "parametric"
            and self.num_radius_coefficients < int(radius_spline_degree) + 1
        ):
            raise ValueError("num_radius_coefficients must be at least radius_spline_degree + 1")
        if self.num_lesions < 0:
            raise ValueError("num_lesions must be non-negative")

        hidden = int(mlp_hidden_dim)
        smaller_hidden = max(128, hidden // 2)
        self.centerline_control_head = nn.Sequential(
            nn.Linear(self.model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_centerline_parameters * 3),
        )
        self.side_centerline_control_head = copy.deepcopy(
            self.centerline_control_head
        )
        if self.radius_prediction_mode == "parametric":
            self.radius_coefficient_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.model_dim, smaller_hidden),
                nn.GELU(),
                nn.Linear(smaller_hidden, self.num_radius_coefficients),
            )
            self.side_radius_coefficient_head: nn.Module | None = copy.deepcopy(
                self.radius_coefficient_head
            )
            self.raw_radius_head: nn.Module | None = None
            self.side_raw_radius_head: nn.Module | None = None
        else:
            self.radius_coefficient_head = None
            self.side_radius_coefficient_head = None
            self.raw_radius_head = nn.Sequential(
                nn.Linear(self.model_dim, smaller_hidden),
                nn.GELU(),
                nn.Linear(smaller_hidden, self.num_points),
            )
            self.side_raw_radius_head = copy.deepcopy(self.raw_radius_head)
        lesion_geometry_dim = 3 if self.lesion_profile == "gaussian" else 5
        if self.radius_prediction_mode == "parametric" and self.num_lesions > 0:
            self.lesion_exist_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.model_dim, smaller_hidden),
                nn.GELU(),
                nn.Linear(smaller_hidden, self.num_lesions),
            )
            self.lesion_geometry_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.model_dim, smaller_hidden),
                nn.GELU(),
                nn.Linear(smaller_hidden, self.num_lesions * lesion_geometry_dim),
            )
            self.side_lesion_exist_head: nn.Module | None = copy.deepcopy(
                self.lesion_exist_head
            )
            self.side_lesion_geometry_head: nn.Module | None = copy.deepcopy(
                self.lesion_geometry_head
            )
        else:
            self.lesion_exist_head = None
            self.lesion_geometry_head = None
            self.side_lesion_exist_head = None
            self.side_lesion_geometry_head = None
        self.lesion_geometry_dim = lesion_geometry_dim

        t = np.linspace(0.0, 1.0, self.num_points, dtype=np.float64)
        if self.centerline_prediction_mode == "bspline_control_points":
            centerline_basis, centerline_knots = bspline_basis_matrix(
                t, self.num_control_points, degree=int(centerline_spline_degree)
            )
            self.register_buffer(
                "centerline_basis", torch.from_numpy(centerline_basis).float()
            )
            self.register_buffer(
                "centerline_knot_vector",
                torch.from_numpy(centerline_knots).float(),
            )
        else:
            self.register_buffer("centerline_basis", torch.empty(self.num_points, 0))
            self.register_buffer("centerline_knot_vector", torch.empty(0))
        if self.radius_prediction_mode == "parametric":
            radius_basis, radius_knots = bspline_basis_matrix(
                t, self.num_radius_coefficients, degree=int(radius_spline_degree)
            )
            self.register_buffer("radius_basis", torch.from_numpy(radius_basis).float())
            self.register_buffer(
                "radius_knot_vector", torch.from_numpy(radius_knots).float()
            )
        else:
            self.register_buffer("radius_basis", torch.empty(self.num_points, 0))
            self.register_buffer("radius_knot_vector", torch.empty(0))

        self.use_bspline_control_refiner = bool(use_bspline_control_refiner)
        self.vggt_image_size_mode = str(vggt_image_size_mode).strip().lower()
        self.bspline_refiner_num_stages = int(bspline_refiner_num_stages)
        self.bspline_refiner_refine_branch_existence = bool(
            bspline_refiner_refine_branch_existence
        )
        self.bspline_refiner_branch_existence_candidate_threshold = float(
            bspline_refiner_branch_existence_candidate_threshold
        )
        if (
            not math.isfinite(
                self.bspline_refiner_branch_existence_candidate_threshold
            )
            or not 0.5
            <= self.bspline_refiner_branch_existence_candidate_threshold
            <= 1.0
        ):
            raise ValueError(
                "bspline_refiner_branch_existence_candidate_threshold must "
                "be finite and in [0.5, 1], got "
                f"{bspline_refiner_branch_existence_candidate_threshold}."
            )
        if (
            self.bspline_refiner_refine_branch_existence
            and not self.use_bspline_control_refiner
        ):
            raise ValueError(
                "bspline_refiner_refine_branch_existence=true requires "
                "use_bspline_control_refiner=true."
            )
        self.bspline_refiner_use_spatial_vggt_features = bool(
            bspline_refiner_use_spatial_vggt_features
        )
        self.bspline_refiner_use_3d_candidates = bool(
            bspline_refiner_use_3d_candidates
        )
        self.bspline_refiner_use_unexplained_centerline_evidence = bool(
            bspline_refiner_use_unexplained_centerline_evidence
        )
        self.bspline_refiner_use_centerline_probability_patch_evidence = bool(
            bspline_refiner_use_centerline_probability_patch_evidence
        )
        self.bspline_refiner_use_separate_centerline_encoder = bool(
            bspline_refiner_use_separate_centerline_encoder
        )
        if (
            self.bspline_refiner_use_spatial_vggt_features
            and not self.use_bspline_control_refiner
        ):
            raise ValueError(
                "bspline_refiner_use_spatial_vggt_features=true requires "
                "use_bspline_control_refiner=true."
            )
        if (
            self.bspline_refiner_use_spatial_vggt_features
            and self.feature_backbone not in {"vggt", "vggt_omega"}
        ):
            raise ValueError(
                "Spatial VGGT refiner evidence requires feature_backbone to be "
                f"'vggt' or 'vggt_omega', got {self.feature_backbone!r}."
            )
        if (
            self.bspline_refiner_use_3d_candidates
            and not self.use_bspline_control_refiner
        ):
            raise ValueError(
                "bspline_refiner_use_3d_candidates=true requires "
                "use_bspline_control_refiner=true."
            )
        if (
            self.bspline_refiner_use_3d_candidates
            and not self.bspline_refiner_use_spatial_vggt_features
        ):
            raise ValueError(
                "bspline_refiner_use_3d_candidates=true requires the preceding "
                "bspline_refiner_use_spatial_vggt_features experiment."
            )
        if (
            self.bspline_refiner_use_unexplained_centerline_evidence
            and not self.bspline_refiner_use_3d_candidates
        ):
            raise ValueError(
                "bspline_refiner_use_unexplained_centerline_evidence=true "
                "requires the preceding bspline_refiner_use_3d_candidates "
                "experiment."
            )
        if (
            self.bspline_refiner_use_centerline_probability_patch_evidence
            and not self.use_bspline_control_refiner
        ):
            raise ValueError(
                "bspline_refiner_use_centerline_probability_patch_evidence=true "
                "requires use_bspline_control_refiner=true."
            )
        candidate_spacing_values = tuple(
            float(value) for value in bspline_refiner_candidate_spacing_mm
        )
        if (
            self.bspline_refiner_use_3d_candidates
            and len(candidate_spacing_values) != self.bspline_refiner_num_stages
        ):
            raise ValueError(
                "bspline_refiner_candidate_spacing_mm must have exactly one "
                "value per refinement stage: got "
                f"{len(candidate_spacing_values)} values for "
                f"{self.bspline_refiner_num_stages} stages."
            )
        if self.use_bspline_control_refiner:
            if self.centerline_prediction_mode != "bspline_control_points":
                raise ValueError(
                    "use_bspline_control_refiner=true is supported only for "
                    "centerline_prediction_mode='bspline_control_points'."
                )
            if self.bspline_refiner_num_stages < 1:
                raise ValueError(
                    "bspline_refiner_num_stages must be >= 1 when refinement is "
                    f"enabled, got {self.bspline_refiner_num_stages}."
                )
            anchor_indices = torch.argmax(self.centerline_basis, dim=0)
            if (
                self.bspline_refiner_use_3d_candidates
                and int(torch.unique(anchor_indices).numel())
                != self.num_control_points
            ):
                raise ValueError(
                    "3D candidate refinement requires one unique on-curve "
                    "anchor per B-spline control."
                )
            anchor_basis_matrix = self.centerline_basis.index_select(
                dim=0,
                index=anchor_indices,
            )
            self.register_buffer(
                "bspline_refiner_anchor_indices",
                anchor_indices.to(dtype=torch.long),
                persistent=True,
            )
            self.bspline_control_refiner: nn.Module | None = (
                _BSplineControlPointProjectionRefiner(
                    model_dim=self.model_dim,
                    view_feat_dim=view_feat_dim,
                    num_branches=self.num_branches,
                    num_control_points=self.num_control_points,
                    num_attention_heads=num_attention_heads,
                    evidence_hidden_dim=bspline_refiner_evidence_hidden_dim,
                    patch_size=bspline_refiner_patch_size,
                    use_learned_image_features=(
                        bspline_refiner_use_learned_image_features
                    ),
                    learned_feature_dim=bspline_refiner_learned_feature_dim,
                    use_distance_transform=bspline_refiner_use_distance_transform,
                    distance_transform_num_iters=(
                        bspline_refiner_distance_transform_num_iters
                    ),
                    dropout=dropout,
                    image_size=bspline_refiner_image_size,
                    sid=bspline_refiner_sid,
                    source_to_iso=bspline_refiner_source_to_iso,
                    imager_pixel_spacing=bspline_refiner_imager_pixel_spacing,
                    residual_scale_mm=bspline_refiner_residual_scale_mm,
                    control_position_scale_mm=(
                        bspline_refiner_control_position_scale_mm
                    ),
                    refine_branch_existence=(
                        self.bspline_refiner_refine_branch_existence
                    ),
                    branch_existence_max_logit_decrease=(
                        bspline_refiner_branch_existence_max_logit_decrease
                    ),
                    coord_scale_to_meter=projection_coord_scale_to_meter,
                    target_coordinate_frame=target_coordinate_frame,
                    use_spatial_vggt_features=(
                        self.bspline_refiner_use_spatial_vggt_features
                    ),
                    spatial_vggt_feature_dim=(
                        bspline_refiner_spatial_vggt_feature_dim
                    ),
                    vggt_token_dim=vggt_token_dim,
                    vggt_image_size_mode=self.vggt_image_size_mode,
                    vggt_patch_size=vggt_patch_size,
                    vggt_target_image_height=vggt_target_image_height,
                    vggt_target_image_width=vggt_target_image_width,
                    use_3d_candidates=self.bspline_refiner_use_3d_candidates,
                    candidate_pattern=bspline_refiner_candidate_pattern,
                    candidate_spacing_mm=candidate_spacing_values,
                    candidate_hidden_dim=bspline_refiner_candidate_hidden_dim,
                    candidate_num_attention_heads=(
                        bspline_refiner_candidate_num_attention_heads
                    ),
                    candidate_score_temperature=(
                        bspline_refiner_candidate_score_temperature
                    ),
                    use_unexplained_centerline_evidence=(
                        self.bspline_refiner_use_unexplained_centerline_evidence
                    ),
                    use_centerline_probability_patch_evidence=(
                        self.bspline_refiner_use_centerline_probability_patch_evidence
                    ),
                    centerline_probability_patch_size=(
                        bspline_refiner_centerline_probability_patch_size
                    ),
                    detach_centerline_probability_patch_evidence=(
                        bspline_refiner_detach_centerline_probability_patch_evidence
                    ),
                    centerline_probability_target_mode=(
                        bspline_refiner_centerline_probability_target_mode
                    ),
                    centerline_probability_target_gamma=(
                        bspline_refiner_centerline_probability_target_gamma
                    ),
                    centerline_probability_mask_threshold=(
                        bspline_refiner_centerline_probability_mask_threshold
                    ),
                    centerline_probability_unexplained_power=(
                        bspline_refiner_centerline_probability_unexplained_power
                    ),
                    use_separate_centerline_encoder=(
                        self.bspline_refiner_use_separate_centerline_encoder
                    ),
                    centerline_map_size=bspline_refiner_centerline_map_size,
                    centerline_head_hidden_dim=(
                        bspline_refiner_centerline_head_hidden_dim
                    ),
                    centerline_map_sigma_px=(
                        bspline_refiner_centerline_map_sigma_px
                    ),
                    centerline_map_radius_px=(
                        bspline_refiner_centerline_map_radius_px
                    ),
                    unexplained_centerline_pool_kernel=(
                        bspline_refiner_unexplained_centerline_pool_kernel
                    ),
                    detach_unexplained_centerline_evidence=(
                        bspline_refiner_detach_unexplained_centerline_evidence
                    ),
                    anchor_basis_matrix=anchor_basis_matrix,
                )
            )
        else:
            self.register_buffer(
                "bspline_refiner_anchor_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.bspline_control_refiner = None

        self.use_radius_evidence_refiner = bool(use_radius_evidence_refiner)
        self.radius_refiner_num_stages = int(radius_refiner_num_stages)
        self.radius_refiner_branch_probability_threshold = float(
            radius_refiner_branch_probability_threshold
        )
        self.fixed_main_branch_count = int(fixed_main_branch_count)
        if not 1 <= self.fixed_main_branch_count <= self.num_branches:
            raise ValueError(
                "fixed_main_branch_count must be between 1 and num_branches, "
                f"got {self.fixed_main_branch_count} and {self.num_branches}."
            )
        self.target_num_branches = int(
            self.num_branches
            if target_num_branches is None
            else target_num_branches
        )
        if not (
            self.fixed_main_branch_count
            <= self.target_num_branches
            <= self.num_branches
        ):
            raise ValueError(
                "target_num_branches must be between fixed_main_branch_count "
                f"and num_branches, got {self.target_num_branches}, "
                f"{self.fixed_main_branch_count}, and {self.num_branches}."
            )
        if not math.isfinite(
            self.radius_refiner_branch_probability_threshold
        ) or not 0.0 <= self.radius_refiner_branch_probability_threshold <= 1.0:
            raise ValueError(
                "radius_refiner_branch_probability_threshold must be finite "
                "and in [0, 1], got "
                f"{radius_refiner_branch_probability_threshold}."
            )
        if self.use_radius_evidence_refiner:
            if not self.use_bspline_control_refiner:
                raise ValueError(
                    "use_radius_evidence_refiner=true requires "
                    "use_bspline_control_refiner=true so geometry refinement "
                    "runs before radius refinement."
                )
            if self.centerline_prediction_mode != "bspline_control_points":
                raise ValueError(
                    "use_radius_evidence_refiner=true requires "
                    "centerline_prediction_mode='bspline_control_points'."
                )
            if self.radius_prediction_mode != "raw":
                raise ValueError(
                    "use_radius_evidence_refiner=true currently requires "
                    "radius_prediction_mode='raw'."
                )
            if self.radius_refiner_num_stages < 1:
                raise ValueError(
                    "radius_refiner_num_stages must be >= 1 when radius "
                    f"refinement is enabled, got {self.radius_refiner_num_stages}."
                )
            self.radius_evidence_refiner: nn.Module | None = (
                _RadiusSurfaceProjectionRefiner(
                    model_dim=self.model_dim,
                    view_feat_dim=view_feat_dim,
                    num_branches=self.num_branches,
                    num_points=self.num_points,
                    num_attention_heads=num_attention_heads,
                    evidence_hidden_dim=radius_refiner_evidence_hidden_dim,
                    profile_samples=radius_refiner_profile_samples,
                    profile_half_width_px=radius_refiner_profile_half_width_px,
                    dropout=dropout,
                    image_size=radius_refiner_image_size,
                    sid=radius_refiner_sid,
                    source_to_iso=radius_refiner_source_to_iso,
                    imager_pixel_spacing=(
                        radius_refiner_imager_pixel_spacing
                    ),
                    residual_scale_mm=radius_refiner_residual_scale_mm,
                    radius_value_scale_mm=(
                        radius_refiner_radius_value_scale_mm
                    ),
                    min_radius_mm=radius_refiner_min_radius_mm,
                    coord_scale_to_meter=projection_coord_scale_to_meter,
                    target_coordinate_frame=target_coordinate_frame,
                    render_num_circle_points=(
                        radius_refiner_render_num_circle_points
                    ),
                    render_radial_subsamples=(
                        radius_refiner_render_radial_subsamples
                    ),
                    render_axial_subsamples=(
                        radius_refiner_render_axial_subsamples
                    ),
                )
            )
        else:
            self.radius_evidence_refiner = None

    def _decode_local_centerlines(
        self,
        centerline_parameters_mm: torch.Tensor,
    ) -> torch.Tensor:
        if self.centerline_prediction_mode == "adaptive_landmarks":
            return decode_landmark_centerlines_local(
                centerline_parameters_mm,
                num_points=self.num_points,
                catmull_rom_alpha=self.catmull_rom_alpha,
                dense_samples=self.catmull_rom_dense_samples,
            )
        return decode_bspline_centerlines_local(
            centerline_parameters_mm, self.centerline_basis
        )

    def _predict_radius(
        self,
        tokens: torch.Tensor,
        *,
        side_branch: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_branches = tokens.shape[:2]
        if self.radius_prediction_mode == "parametric":
            coefficient_head = (
                self.side_radius_coefficient_head
                if side_branch
                else self.radius_coefficient_head
            )
            assert coefficient_head is not None
            radius_coefficients = coefficient_head(tokens)
            if self.num_lesions > 0:
                lesion_exist_head = (
                    self.side_lesion_exist_head
                    if side_branch
                    else self.lesion_exist_head
                )
                lesion_geometry_head = (
                    self.side_lesion_geometry_head
                    if side_branch
                    else self.lesion_geometry_head
                )
                assert (
                    lesion_exist_head is not None
                    and lesion_geometry_head is not None
                )
                lesion_exist_logits = lesion_exist_head(tokens)
                lesion_geometry_raw = lesion_geometry_head(tokens).view(
                    batch_size,
                    num_branches,
                    self.num_lesions,
                    self.lesion_geometry_dim,
                )
                lesion_geometry = constrain_lesion_geometry(
                    lesion_geometry_raw, self.lesion_profile
                )
            else:
                lesion_exist_logits = tokens.new_zeros(
                    (batch_size, num_branches, 0)
                )
                lesion_geometry = tokens.new_zeros(
                    (batch_size, num_branches, 0, self.lesion_geometry_dim)
                )
            radius_mm = decode_radii(
                radius_coefficients,
                lesion_exist_logits,
                lesion_geometry,
                self.radius_basis,
                self.lesion_profile,
            )
            raw_radius_log_mm = tokens.new_zeros(
                (batch_size, num_branches, 0)
            )
            raw_radius_mm = raw_radius_log_mm
        else:
            raw_head = (
                self.side_raw_radius_head
                if side_branch
                else self.raw_radius_head
            )
            assert raw_head is not None
            raw_radius_log_mm = raw_head(tokens)
            raw_radius_mm = decode_raw_radii(raw_radius_log_mm)
            radius_mm = raw_radius_mm
            radius_coefficients = tokens.new_zeros(
                (batch_size, num_branches, 0)
            )
            lesion_exist_logits = tokens.new_zeros(
                (batch_size, num_branches, 0)
            )
            lesion_geometry = tokens.new_zeros(
                (batch_size, num_branches, 0, self.lesion_geometry_dim)
            )
        return {
            "radius_baseline_coefficients_log_mm": radius_coefficients,
            "raw_radius_log_mm": raw_radius_log_mm,
            "raw_radius_mm": raw_radius_mm,
            "lesion_exist_logits": lesion_exist_logits,
            "lesion_geometry": lesion_geometry,
            "radius_mm": radius_mm,
        }

    def _run_bspline_control_refiner(
        self,
        *,
        output: dict[str, torch.Tensor],
        branch_tokens: torch.Tensor,
        image_features: torch.Tensor | dict[str, torch.Tensor],
        images: torch.Tensor | None,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        projection_center_offset: torch.Tensor | None,
        refiner_branch_mask: torch.Tensor | None,
        record_refiner_timing: bool,
    ) -> tuple[dict[str, torch.Tensor], float | None]:
        """Refine relative or absolute B-spline controls in their native frame.

        Hierarchical and legacy-parallel branches are reassembled from their
        attachment-relative side controls. Absolute-parallel branches are
        decoded directly and remain independent; the refiner is free to move
        them using image evidence but no attachment is imposed.
        """

        refiner = self.bspline_control_refiner
        if refiner is None:
            return output, None
        if images is None:
            raise ValueError(
                "use_bspline_control_refiner=true requires input images in "
                "ParametricVesselPredictor.forward."
            )
        centerline_parameters = output["centerline_parameters_mm"]
        vessel_mm = output["decoded_vessel_mm"]
        batch_size = int(centerline_parameters.shape[0])
        predicted_centerline_evidence_branch_mask = (
            self._resolve_refiner_branch_mask(
                output=output,
                override=None,
            )
        )
        active_branch_mask = self._resolve_refiner_branch_mask(
            output=output,
            override=refiner_branch_mask,
        )
        coarse_branch_exist_logits = output["branch_exist_logits"]
        coarse_branch_exist_probs = output["branch_exist_probs"]
        existence_candidate_mask: torch.Tensor | None = None
        current_branch_exist_logits: torch.Tensor | None = None
        if self.bspline_refiner_refine_branch_existence:
            count_limited_mask = output.get(
                "prediction_count_limited_branch_mask"
            )
            if count_limited_mask is None:
                coarse_positive_mask = coarse_branch_exist_probs.detach() >= (
                    self.bspline_refiner_branch_existence_candidate_threshold
                )
                coarse_positive_mask = coarse_positive_mask.clone()
                coarse_positive_mask[
                    :, : self.fixed_main_branch_count
                ] = True
                # Existence refinement must have the same forward inputs in
                # training and inference: the GT existence label is
                # supervision, never an input to the classification
                # trajectory. Therefore teacher-forced training masks remain
                # ignored in the ordinary path.
                active_branch_mask = coarse_positive_mask
                predicted_centerline_evidence_branch_mask = (
                    coarse_positive_mask
                )
            else:
                # Prediction-only evaluation may know the total topology but
                # not the side-branch identities. Its mask is derived only
                # from coarse probabilities, so it is safe to use as the
                # inference trajectory for both geometry and removal-only
                # existence refinement.
                active_branch_mask = count_limited_mask
                predicted_centerline_evidence_branch_mask = (
                    count_limited_mask
                )
            existence_candidate_mask = active_branch_mask.clone()
            existence_candidate_mask[:, : self.fixed_main_branch_count] = False
            current_branch_exist_logits = coarse_branch_exist_logits
            output["bspline_refiner_existence_candidate_mask"] = (
                existence_candidate_mask
            )
        output["bspline_refiner_active_branch_mask"] = active_branch_mask
        if refiner.use_unexplained_centerline_evidence:
            output[
                "bspline_refinement_centerline_evidence_branch_mask"
            ] = predicted_centerline_evidence_branch_mask
        expected_token_shape = (
            batch_size,
            self.num_branches,
            self.model_dim,
        )
        if tuple(branch_tokens.shape) != expected_token_shape:
            raise ValueError(
                "B-spline refinement requires one branch token per decoded "
                f"branch; got {tuple(branch_tokens.shape)}, expected "
                f"{expected_token_shape}."
            )
        absolute_coordinates = (
            self.decoder_architecture == ABSOLUTE_PARALLEL_DECODER
        )
        if refiner.use_3d_candidates and not absolute_coordinates:
            raise ValueError(
                "bspline_refiner_use_3d_candidates=true currently requires "
                "decoder_architecture='absolute_parallel' so candidate anchor "
                "displacements map exactly back to independent B-spline controls."
            )
        attachment_probabilities: torch.Tensor | None = None
        if not absolute_coordinates:
            attachment_probabilities = output["attachment_probabilities"]
            if attachment_probabilities.shape != (
                batch_size,
                self.num_branches,
                self.num_points,
            ):
                raise ValueError(
                    "attachment_probabilities must match the decoded branch "
                    f"grid, got {tuple(attachment_probabilities.shape)}."
                )

        started: float | None = None
        if record_refiner_timing:
            _synchronize_for_refiner_timing(centerline_parameters.device)
            started = time.perf_counter()
        current_parameters = centerline_parameters
        current_vessel = vessel_mm
        point_state: torch.Tensor | None = None
        stage_parameters: list[torch.Tensor] = []
        stage_vessels: list[torch.Tensor] = []
        stage_side_offsets: list[torch.Tensor] = []
        stage_side_relative: list[torch.Tensor] = []
        stage_residuals: list[torch.Tensor] = []
        stage_branch_exist_logits: list[torch.Tensor] = []
        stage_branch_exist_probs: list[torch.Tensor] = []
        stage_branch_exist_residuals: list[torch.Tensor] = []
        stage_candidate_control_residuals: list[torch.Tensor] = []
        stage_candidate_anchor_residuals: list[torch.Tensor] = []
        stage_candidate_probabilities: list[torch.Tensor] = []
        stage_predicted_centerline_maps: list[torch.Tensor] = []
        stage_unexplained_centerline_maps: list[torch.Tensor] = []
        refiner_images = images.to(
            device=current_parameters.device,
            dtype=current_parameters.dtype,
        )
        prepared_distance_map, prepared_feature_map = refiner.prepare_image_maps(
            refiner_images
        )
        prepared_distance_gradient_map = refiner.prepare_distance_gradient_map(
            prepared_distance_map
        )
        (
            prepared_input_centerline_logits,
            prepared_input_centerline_probability_map,
        ) = refiner.prepare_input_centerline_probability_map(
            prepared_feature_map,
            batch_size=batch_size,
            num_views=int(views.shape[1]),
            images=refiner_images,
        )
        if prepared_input_centerline_logits is not None:
            output["bspline_refinement_input_centerline_logits"] = (
                prepared_input_centerline_logits.reshape(
                    batch_size,
                    int(views.shape[1]),
                    refiner.centerline_map_size,
                    refiner.centerline_map_size,
                )
            )
            assert prepared_input_centerline_probability_map is not None
            diagnostic_centerline_probabilities = (
                prepared_input_centerline_probability_map.detach().reshape(
                    batch_size,
                    int(views.shape[1]),
                    refiner.centerline_map_size,
                    refiner.centerline_map_size,
                )
            )
            if view_mask is not None:
                diagnostic_centerline_probabilities = (
                    diagnostic_centerline_probabilities
                    * view_mask.to(
                        device=diagnostic_centerline_probabilities.device,
                        dtype=diagnostic_centerline_probabilities.dtype,
                    )[..., None, None]
                )
            output["bspline_refinement_input_centerline_probabilities"] = (
                diagnostic_centerline_probabilities
            )
        prepared_spatial_vggt_map = refiner.prepare_spatial_vggt_map(
            image_features
        )
        center_offset_mm = (
            None
            if projection_center_offset is None
            else projection_center_offset.to(
                device=current_parameters.device,
                dtype=current_parameters.dtype,
            )
        )

        for _stage_index in range(self.bspline_refiner_num_stages):
            anchor_points_mm = current_vessel[..., :3].index_select(
                dim=2,
                index=self.bspline_refiner_anchor_indices,
            )
            (
                current_parameters,
                point_state,
                residual_mm,
                branch_exist_logit_residual,
                candidate_control_residual_mm,
                candidate_anchor_residual_mm,
                candidate_probabilities,
                predicted_centerline_map,
                unexplained_centerline_map,
            ) = refiner(
                branch_tokens=branch_tokens,
                control_points_mm=current_parameters,
                anchor_points_mm=anchor_points_mm,
                current_centerlines_mm=current_vessel[..., :3],
                images=refiner_images,
                views=views,
                view_mask=view_mask,
                point_state=point_state,
                projection_center_offset_mm=center_offset_mm,
                prepared_distance_map=prepared_distance_map,
                prepared_distance_gradient_map=prepared_distance_gradient_map,
                prepared_feature_map=prepared_feature_map,
                prepared_spatial_vggt_map=prepared_spatial_vggt_map,
                prepared_input_centerline_probability_map=(
                    prepared_input_centerline_probability_map
                ),
                stage_index=_stage_index,
                branch_mask=active_branch_mask,
                centerline_evidence_branch_mask=(
                    predicted_centerline_evidence_branch_mask
                ),
                existence_candidate_mask=existence_candidate_mask,
                current_branch_exist_logits=current_branch_exist_logits,
            )
            if refiner.use_3d_candidates:
                if (
                    candidate_control_residual_mm is None
                    or candidate_anchor_residual_mm is None
                    or candidate_probabilities is None
                ):
                    raise RuntimeError(
                        "Enabled 3D candidate refinement did not return its "
                        "stage diagnostics."
                    )
                stage_candidate_control_residuals.append(
                    candidate_control_residual_mm
                )
                stage_candidate_anchor_residuals.append(
                    candidate_anchor_residual_mm
                )
                stage_candidate_probabilities.append(candidate_probabilities)
            if refiner.use_unexplained_centerline_evidence:
                if (
                    predicted_centerline_map is None
                    or unexplained_centerline_map is None
                ):
                    raise RuntimeError(
                        "Enabled unexplained-centerline evidence did not return "
                        "its stage maps."
                    )
                stage_predicted_centerline_maps.append(
                    predicted_centerline_map.reshape(
                        batch_size,
                        int(views.shape[1]),
                        refiner.centerline_map_size,
                        refiner.centerline_map_size,
                    )
                )
                stage_unexplained_centerline_maps.append(
                    unexplained_centerline_map.reshape(
                        batch_size,
                        int(views.shape[1]),
                        refiner.centerline_map_size,
                        refiner.centerline_map_size,
                    )
                )
            if self.bspline_refiner_refine_branch_existence:
                if (
                    branch_exist_logit_residual is None
                    or existence_candidate_mask is None
                    or current_branch_exist_logits is None
                ):
                    raise RuntimeError(
                        "Enabled B-spline branch-existence refinement did not "
                        "return its required stage tensors."
                    )
                proposed_branch_exist_logits = (
                    current_branch_exist_logits
                    + branch_exist_logit_residual
                )
                minimum_branch_exist_logits = coarse_branch_exist_logits - (
                    refiner.branch_existence_max_logit_decrease
                )
                bounded_branch_exist_logits = torch.maximum(
                    torch.minimum(
                        proposed_branch_exist_logits,
                        coarse_branch_exist_logits,
                    ),
                    minimum_branch_exist_logits,
                )
                next_branch_exist_logits = torch.where(
                    existence_candidate_mask,
                    bounded_branch_exist_logits,
                    coarse_branch_exist_logits,
                )
                applied_logit_residual = (
                    next_branch_exist_logits - current_branch_exist_logits
                )
                current_branch_exist_logits = next_branch_exist_logits
                current_branch_exist_probs = (
                    self._branch_existence_probabilities(
                        current_branch_exist_logits
                    )
                )
                stage_branch_exist_logits.append(current_branch_exist_logits)
                stage_branch_exist_probs.append(current_branch_exist_probs)
                stage_branch_exist_residuals.append(applied_logit_residual)
            local_centerlines_mm = self._decode_local_centerlines(
                current_parameters
            )
            current_main_centerline_mm = local_centerlines_mm[:, :1]
            if self.num_branches > 1:
                current_side_offsets_mm = (
                    local_centerlines_mm[:, 1:]
                    - local_centerlines_mm[:, 1:, :1]
                )
                if absolute_coordinates:
                    current_side_centerlines_mm = local_centerlines_mm[:, 1:]
                else:
                    assert attachment_probabilities is not None
                    current_attachment_coordinates = torch.sum(
                        attachment_probabilities[:, 1:].unsqueeze(-1)
                        * current_main_centerline_mm,
                        dim=2,
                    )
                    current_side_centerlines_mm = (
                        current_side_offsets_mm
                        + current_attachment_coordinates.unsqueeze(2)
                    )
            else:
                current_side_offsets_mm = local_centerlines_mm.new_empty(
                    (batch_size, 0, self.num_points, 3)
                )
                current_side_centerlines_mm = current_side_offsets_mm
            current_centerlines_mm = torch.cat(
                [current_main_centerline_mm, current_side_centerlines_mm],
                dim=1,
            )
            current_vessel = torch.cat(
                [current_centerlines_mm, vessel_mm[..., 3:].clone()], dim=-1
            )
            current_all_side_offsets = torch.cat(
                [
                    current_main_centerline_mm.new_zeros(
                        (batch_size, 1, self.num_points, 3)
                    ),
                    current_side_offsets_mm,
                ],
                dim=1,
            )
            current_side_relative = torch.cat(
                [
                    current_side_offsets_mm,
                    current_vessel[:, 1:, :, 3:4],
                ],
                dim=-1,
            )
            stage_parameters.append(current_parameters)
            stage_vessels.append(current_vessel)
            stage_side_offsets.append(current_all_side_offsets)
            stage_side_relative.append(current_side_relative)
            stage_residuals.append(residual_mm)

        output["coarse_centerline_parameters_mm"] = centerline_parameters
        output["coarse_decoded_vessel_mm"] = vessel_mm
        output["coarse_side_centerline_offsets_mm"] = output[
            "side_centerline_offsets_mm"
        ]
        output["coarse_side_branch_relative_code_mm"] = output[
            "side_branch_relative_code_mm"
        ]
        output["refinement_stage_centerline_parameters_mm"] = torch.stack(
            stage_parameters, dim=1
        )
        output["refinement_stage_decoded_vessel_mm"] = torch.stack(
            stage_vessels, dim=1
        )
        output["refinement_stage_side_centerline_offsets_mm"] = torch.stack(
            stage_side_offsets, dim=1
        )
        output["refinement_stage_side_branch_relative_code_mm"] = torch.stack(
            stage_side_relative, dim=1
        )
        output["bspline_refinement_residual_mm"] = torch.stack(
            stage_residuals, dim=1
        )
        if refiner.use_3d_candidates:
            output["bspline_refinement_candidate_control_residual_mm"] = (
                torch.stack(stage_candidate_control_residuals, dim=1)
            )
            output["bspline_refinement_candidate_anchor_residual_mm"] = (
                torch.stack(stage_candidate_anchor_residuals, dim=1)
            )
            output["bspline_refinement_candidate_probabilities"] = torch.stack(
                stage_candidate_probabilities, dim=1
            )
        if refiner.use_unexplained_centerline_evidence:
            output[
                "bspline_refinement_stage_input_predicted_centerline_maps"
            ] = (
                torch.stack(stage_predicted_centerline_maps, dim=1)
            )
            output[
                "bspline_refinement_stage_input_unexplained_centerline_maps"
            ] = (
                torch.stack(stage_unexplained_centerline_maps, dim=1)
            )
        output["centerline_parameters_mm"] = stage_parameters[-1]
        output["centerline_control_points_mm"] = stage_parameters[-1]
        output["decoded_vessel_mm"] = stage_vessels[-1]
        output["side_centerline_offsets_mm"] = stage_side_offsets[-1]
        output["side_branch_relative_code_mm"] = stage_side_relative[-1]
        if self.bspline_refiner_refine_branch_existence:
            output["coarse_branch_exist_logits"] = coarse_branch_exist_logits
            output["coarse_branch_exist_probs"] = coarse_branch_exist_probs
            output["refinement_stage_branch_exist_logits"] = torch.stack(
                stage_branch_exist_logits,
                dim=1,
            )
            output["refinement_stage_branch_exist_probs"] = torch.stack(
                stage_branch_exist_probs,
                dim=1,
            )
            output[
                "bspline_refinement_branch_exist_residual_logits"
            ] = torch.stack(stage_branch_exist_residuals, dim=1)
            output["branch_exist_logits"] = stage_branch_exist_logits[-1]
            output["branch_exist_probs"] = stage_branch_exist_probs[-1]

        elapsed_ms: float | None = None
        if started is not None:
            _synchronize_for_refiner_timing(centerline_parameters.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        return output, elapsed_ms

    def _resolve_refiner_branch_mask(
        self,
        *,
        output: dict[str, torch.Tensor],
        override: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a validated teacher-forced or predicted refiner mask."""

        probabilities = output.get("branch_exist_probs")
        if probabilities is None:
            raise ValueError("Refinement requires branch_exist_probs.")
        expected_shape = (int(probabilities.shape[0]), self.num_branches)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "Refinement requires branch_exist_probs with shape "
                f"{expected_shape}, got {tuple(probabilities.shape)}."
            )
        if override is None:
            active = probabilities.detach() >= (
                self.radius_refiner_branch_probability_threshold
            )
        else:
            active = override.to(device=probabilities.device, dtype=torch.bool)
            if active.shape != expected_shape:
                raise ValueError(
                    "refiner_branch_mask must have shape [B,M], got "
                    f"{tuple(active.shape)} and expected {expected_shape}."
                )
        active = active.clone()
        active[:, : self.fixed_main_branch_count] = True
        active[:, self.target_num_branches :] = False
        return active

    def _count_limited_refiner_branch_mask(
        self,
        *,
        output: dict[str, torch.Tensor],
        total_branch_count: int | None,
        override: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Select an exact known count using coarse side-existence ranking."""

        if total_branch_count is None:
            return override
        if override is not None:
            raise ValueError(
                "refiner_total_branch_count cannot be combined with an "
                "explicit refiner_branch_mask."
            )
        if isinstance(total_branch_count, bool) or not isinstance(
            total_branch_count, int
        ):
            raise ValueError("refiner_total_branch_count must be an integer.")
        count = int(total_branch_count)
        if not self.fixed_main_branch_count <= count <= self.target_num_branches:
            raise ValueError(
                "refiner_total_branch_count must be between "
                f"fixed_main_branch_count={self.fixed_main_branch_count} and "
                f"target_num_branches={self.target_num_branches}, got {count}."
            )
        probabilities = output.get("branch_exist_probs")
        if probabilities is None:
            raise ValueError(
                "Count-limited refinement requires coarse branch_exist_probs."
            )
        expected_shape = (int(probabilities.shape[0]), self.num_branches)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "Count-limited refinement requires branch_exist_probs with "
                f"shape {expected_shape}, got {tuple(probabilities.shape)}."
            )

        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active[:, : self.fixed_main_branch_count] = True
        optional_count = count - self.fixed_main_branch_count
        if optional_count > 0:
            optional_probabilities = probabilities.detach()[
                :,
                self.fixed_main_branch_count : self.target_num_branches,
            ]
            # Stable sorting makes equal-probability ties deterministic: the
            # lower (earlier) trained query index wins.
            ranked_optional_indices = torch.argsort(
                optional_probabilities,
                dim=1,
                descending=True,
                stable=True,
            )[:, :optional_count]
            ranked_optional_indices = (
                ranked_optional_indices + self.fixed_main_branch_count
            )
            active.scatter_(1, ranked_optional_indices, True)
        output["prediction_count_limited_branch_mask"] = active
        return active

    def _branch_existence_probabilities(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Return optional-query probabilities with fixed main slots forced on."""

        expected_shape = (int(logits.shape[0]), self.num_branches)
        if logits.shape != expected_shape:
            raise ValueError(
                "branch existence logits must have shape "
                f"{expected_shape}, got {tuple(logits.shape)}."
            )
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.clone()
        probabilities[:, : self.fixed_main_branch_count] = 1.0
        return probabilities

    def _run_radius_evidence_refiner(
        self,
        *,
        output: dict[str, torch.Tensor],
        branch_tokens: torch.Tensor,
        images: torch.Tensor | None,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        projection_center_offset: torch.Tensor | None,
        refiner_branch_mask: torch.Tensor | None,
        record_refiner_timing: bool,
        canonical_geometry_member_index: int | None = None,
        canonical_geometry_p95_threshold_mm: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], float | None]:
        """Apply the optional radius refiner to either decoder architecture."""

        radius_refiner = self.radius_evidence_refiner
        if radius_refiner is None:
            return output, None
        if images is None:
            raise ValueError(
                "use_radius_evidence_refiner=true requires input images in "
                "ParametricVesselPredictor.forward."
            )
        assert isinstance(radius_refiner, _RadiusSurfaceProjectionRefiner)
        current_vessel = output["decoded_vessel_mm"]
        batch_size = int(current_vessel.shape[0])
        active_branch_mask = self._resolve_refiner_branch_mask(
            output=output,
            override=refiner_branch_mask,
        )
        output["radius_refiner_active_branch_mask"] = active_branch_mask
        started: float | None = None
        if record_refiner_timing:
            _synchronize_for_refiner_timing(current_vessel.device)
            started = time.perf_counter()
        if canonical_geometry_member_index is not None:
            canonical_index = int(canonical_geometry_member_index)
            if not 0 <= canonical_index < batch_size:
                raise ValueError(
                    "canonical_geometry_member_index must identify a member of "
                    f"the current batch, got {canonical_index} for B={batch_size}."
                )
            if canonical_geometry_p95_threshold_mm is None:
                raise ValueError(
                    "canonical_geometry_member_index requires "
                    "canonical_geometry_p95_threshold_mm."
                )
            geometry_threshold = float(canonical_geometry_p95_threshold_mm)
            if not math.isfinite(geometry_threshold) or geometry_threshold < 0.0:
                raise ValueError(
                    "canonical_geometry_p95_threshold_mm must be finite and "
                    f">= 0, got {geometry_threshold}."
                )
            pre_canonical_vessel = current_vessel
            canonical_xyz = pre_canonical_vessel[
                canonical_index : canonical_index + 1, ..., :3
            ].detach()
            member_distances = torch.linalg.vector_norm(
                pre_canonical_vessel[..., :3].detach() - canonical_xyz,
                dim=-1,
            )
            active_point_mask = active_branch_mask.unsqueeze(-1).expand_as(
                member_distances
            )
            member_p95_mm = torch.stack(
                [
                    torch.quantile(
                        member_distances[index][active_point_mask[index]],
                        0.95,
                    )
                    for index in range(batch_size)
                ]
            )
            apply_canonical = member_p95_mm.max() > geometry_threshold
            output[
                "radius_refiner_precanonical_decoded_vessel_mm"
            ] = pre_canonical_vessel
            output[
                "radius_refiner_precanonical_geometry_p95_mm"
            ] = member_p95_mm
            output[
                "radius_refiner_geometry_canonicalization_applied"
            ] = apply_canonical.to(member_p95_mm).expand(batch_size)
            output["radius_refiner_canonical_geometry_member_index"] = (
                torch.full(
                    (batch_size,),
                    canonical_index,
                    device=current_vessel.device,
                    dtype=torch.long,
                )
            )
            selected_xyz = torch.where(
                apply_canonical,
                canonical_xyz.expand(batch_size, -1, -1, -1),
                current_vessel[..., :3],
            )
            current_vessel = torch.cat(
                [selected_xyz, current_vessel[..., 3:4]], dim=-1
            )
        elif canonical_geometry_p95_threshold_mm is not None:
            raise ValueError(
                "canonical_geometry_p95_threshold_mm requires "
                "canonical_geometry_member_index."
            )

        radius_coarse_vessel = current_vessel
        radius_stage_vessels: list[torch.Tensor] = []
        radius_stage_residuals: list[torch.Tensor] = []
        radius_point_state: torch.Tensor | None = None
        radius_images = images.to(
            device=current_vessel.device,
            dtype=current_vessel.dtype,
        )
        center_offset_mm = (
            None
            if projection_center_offset is None
            else projection_center_offset.to(
                device=current_vessel.device,
                dtype=current_vessel.dtype,
            )
        )
        projection_context = radius_refiner.prepare_projection_context(
            vessel_mm=current_vessel,
            images=radius_images,
            views=views,
            projection_center_offset_mm=center_offset_mm,
            branch_mask=active_branch_mask,
        )
        rendered_masks = radius_refiner.render_masks(
            vessel_mm=current_vessel,
            views=views,
            projection_center_offset_mm=center_offset_mm,
            projection_context=projection_context,
        )
        coarse_rendered_masks = rendered_masks
        for _stage_index in range(self.radius_refiner_num_stages):
            refined_radius_mm, radius_point_state, residual_mm = radius_refiner(
                branch_tokens=branch_tokens,
                vessel_mm=current_vessel,
                rendered_masks=rendered_masks,
                images=radius_images,
                views=views,
                view_mask=view_mask,
                point_state=radius_point_state,
                projection_center_offset_mm=center_offset_mm,
                projection_context=projection_context,
                branch_mask=active_branch_mask,
            )
            current_vessel = torch.cat(
                [
                    current_vessel[..., :3],
                    refined_radius_mm.unsqueeze(-1),
                ],
                dim=-1,
            )
            rendered_masks = radius_refiner.render_masks(
                vessel_mm=current_vessel,
                views=views,
                projection_center_offset_mm=center_offset_mm,
                projection_context=projection_context,
            )
            radius_stage_vessels.append(current_vessel)
            radius_stage_residuals.append(residual_mm)

        output["radius_refiner_coarse_decoded_vessel_mm"] = radius_coarse_vessel
        output["radius_refinement_stage_decoded_vessel_mm"] = torch.stack(
            radius_stage_vessels,
            dim=1,
        )
        output["radius_refinement_residual_mm"] = torch.stack(
            radius_stage_residuals,
            dim=1,
        )
        output["radius_refiner_coarse_rendered_masks"] = coarse_rendered_masks
        output["radius_refiner_final_rendered_masks"] = rendered_masks
        output["coarse_raw_radius_mm"] = output["raw_radius_mm"]
        output["coarse_raw_radius_log_mm"] = output["raw_radius_log_mm"]
        output["refined_raw_radius_mm"] = current_vessel[..., 3]
        output["raw_radius_mm"] = current_vessel[..., 3]
        output["raw_radius_log_mm"] = torch.log(
            current_vessel[..., 3].clamp_min(radius_refiner.min_radius_mm)
        )
        output["decoded_vessel_mm"] = current_vessel
        if self.num_branches > 1:
            output["side_branch_relative_code_mm"] = torch.cat(
                [
                    output["side_branch_relative_code_mm"][..., :3],
                    current_vessel[:, 1:, :, 3:4],
                ],
                dim=-1,
            )

        elapsed_ms: float | None = None
        if started is not None:
            _synchronize_for_refiner_timing(current_vessel.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        if output["decoded_vessel_mm"].shape[0] != batch_size:
            raise RuntimeError("Radius refinement changed the batch dimension.")
        return output, elapsed_ms

    def forward(
        self,
        *,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        image_features: torch.Tensor | dict[str, torch.Tensor],
        images: torch.Tensor | None = None,
        projection_center_offset: torch.Tensor | None = None,
        record_coarse_prediction_timing: bool = False,
        record_refiner_timing: bool = False,
        return_refiner_context: bool = False,
        refiner_branch_mask: torch.Tensor | None = None,
        refiner_total_branch_count: int | None = None,
        radius_refiner_canonical_geometry_member_index: int | None = None,
        radius_refiner_canonical_geometry_p95_threshold_mm: float | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        coarse_prediction_started: float | None = None
        if record_coarse_prediction_timing:
            _synchronize_for_model_timing(views.device)
            coarse_prediction_started = time.perf_counter()
        memory, padding_mask = self._encode_view_memory(
            views=views,
            view_mask=view_mask,
            image_features=image_features,
        )
        main_token, initial_side_tokens = self._decode_main_and_side_tokens(
            memory, padding_mask
        )
        batch_size = main_token.shape[0]
        main_centerline_parameters = self.centerline_control_head(main_token).view(
            batch_size, 1, self.num_centerline_parameters, 3
        )
        main_centerline_parameters = (
            main_centerline_parameters * self.centerline_output_scale_mm
        )
        main_centerline_mm = self._decode_local_centerlines(
            main_centerline_parameters
        )
        main_radius = self._predict_radius(main_token, side_branch=False)
        main_vessel_mm = torch.cat(
            (main_centerline_mm, main_radius["radius_mm"].unsqueeze(-1)), dim=-1
        )

        side_state = self._condition_side_tokens(
            main_token=main_token,
            initial_side_tokens=initial_side_tokens,
            main_branch_points=main_vessel_mm,
        )
        conditioned_side_tokens = side_state["conditioned_side_tokens"]
        if self.num_branches > 1:
            side_centerline_parameters = self.side_centerline_control_head(
                conditioned_side_tokens
            ).view(
                batch_size,
                self.num_branches - 1,
                self.num_centerline_parameters,
                3,
            )
            side_centerline_parameters = (
                side_centerline_parameters * self.centerline_output_scale_mm
            )
            side_centerline_offsets_mm = self._decode_local_centerlines(
                side_centerline_parameters
            )
            side_centerline_offsets_mm = (
                side_centerline_offsets_mm
                - side_centerline_offsets_mm[..., :1, :]
            )
            side_centerlines_mm = (
                side_centerline_offsets_mm
                + side_state["attachment_coordinates"].unsqueeze(2)
            )
            side_radius = self._predict_radius(
                conditioned_side_tokens, side_branch=True
            )
            side_vessel_mm = torch.cat(
                (side_centerlines_mm, side_radius["radius_mm"].unsqueeze(-1)),
                dim=-1,
            )
            side_exist_logits = self.branch_exist_head(
                conditioned_side_tokens
            ).squeeze(-1)
        else:
            side_centerline_parameters = main_centerline_parameters.new_empty(
                (batch_size, 0, self.num_centerline_parameters, 3)
            )
            side_centerline_offsets_mm = main_centerline_mm.new_empty(
                (batch_size, 0, self.num_points, 3)
            )
            side_vessel_mm = main_vessel_mm.new_empty(
                (batch_size, 0, self.num_points, 4)
            )
            side_exist_logits = main_token.new_empty((batch_size, 0))
            side_radius = {
                "radius_baseline_coefficients_log_mm": main_radius[
                    "radius_baseline_coefficients_log_mm"
                ].new_empty(
                    (batch_size, 0, main_radius[
                        "radius_baseline_coefficients_log_mm"
                    ].shape[-1])
                ),
                "raw_radius_log_mm": main_radius["raw_radius_log_mm"].new_empty(
                    (batch_size, 0, main_radius["raw_radius_log_mm"].shape[-1])
                ),
                "raw_radius_mm": main_radius["raw_radius_mm"].new_empty(
                    (batch_size, 0, main_radius["raw_radius_mm"].shape[-1])
                ),
                "radius_mm": main_radius["radius_mm"].new_empty(
                    (batch_size, 0, self.num_points)
                ),
                "lesion_exist_logits": main_radius[
                    "lesion_exist_logits"
                ].new_empty(
                    (batch_size, 0, main_radius["lesion_exist_logits"].shape[-1])
                ),
                "lesion_geometry": main_radius["lesion_geometry"].new_empty(
                    (
                        batch_size,
                        0,
                        main_radius["lesion_geometry"].shape[-2],
                        self.lesion_geometry_dim,
                    )
                ),
            }

        centerline_parameters = torch.cat(
            [main_centerline_parameters, side_centerline_parameters], dim=1
        )
        vessel_mm = torch.cat([main_vessel_mm, side_vessel_mm], dim=1)
        radius_coefficients = torch.cat(
            [
                main_radius["radius_baseline_coefficients_log_mm"],
                side_radius["radius_baseline_coefficients_log_mm"],
            ],
            dim=1,
        )
        raw_radius_log_mm = torch.cat(
            [main_radius["raw_radius_log_mm"], side_radius["raw_radius_log_mm"]],
            dim=1,
        )
        raw_radius_mm = torch.cat(
            [main_radius["raw_radius_mm"], side_radius["raw_radius_mm"]], dim=1
        )
        lesion_exist_logits = torch.cat(
            [
                main_radius["lesion_exist_logits"],
                side_radius["lesion_exist_logits"],
            ],
            dim=1,
        )
        lesion_geometry = torch.cat(
            [main_radius["lesion_geometry"], side_radius["lesion_geometry"]],
            dim=1,
        )
        main_exist_logits = main_token.new_full((batch_size, 1), 20.0)
        branch_exist_logits = torch.cat(
            [main_exist_logits, side_exist_logits], dim=1
        )
        branch_exist_probs = self._branch_existence_probabilities(
            branch_exist_logits
        )
        main_attachment_logits = main_token.new_zeros(
            (batch_size, 1, self.num_points)
        )
        main_attachment_probabilities = main_token.new_zeros(
            (batch_size, 1, self.num_points)
        )
        main_attachment_probabilities[..., 0] = 1.0
        attachment_logits = torch.cat(
            [main_attachment_logits, side_state["attachment_logits"]], dim=1
        )
        attachment_probabilities = torch.cat(
            [
                main_attachment_probabilities,
                side_state["attachment_probabilities"],
            ],
            dim=1,
        )
        attachment_indices = torch.cat(
            [
                torch.zeros(
                    (batch_size, 1),
                    device=main_token.device,
                    dtype=torch.long,
                ),
                side_state["attachment_indices"],
            ],
            dim=1,
        )
        output = {
            "centerline_parameters_mm": centerline_parameters,
            "radius_baseline_coefficients_log_mm": radius_coefficients,
            "raw_radius_log_mm": raw_radius_log_mm,
            "raw_radius_mm": raw_radius_mm,
            "attachment_logits": attachment_logits,
            "attachment_probabilities": attachment_probabilities,
            "attachment_indices": attachment_indices,
            "lesion_exist_logits": lesion_exist_logits,
            "lesion_geometry": lesion_geometry,
            "branch_exist_logits": branch_exist_logits,
            "branch_exist_probs": branch_exist_probs,
            "decoded_vessel_mm": vessel_mm,
            "side_centerline_offsets_mm": torch.cat(
                [
                    main_centerline_mm.new_zeros(
                        (batch_size, 1, self.num_points, 3)
                    ),
                    side_centerline_offsets_mm,
                ],
                dim=1,
            ),
            "side_branch_relative_code_mm": torch.cat(
                [
                    side_centerline_offsets_mm,
                    side_radius["radius_mm"].unsqueeze(-1),
                ],
                dim=-1,
            ),
        }
        if self.centerline_prediction_mode == "adaptive_landmarks":
            output["centerline_landmarks_mm"] = centerline_parameters
        else:
            output["centerline_control_points_mm"] = centerline_parameters

        refiner_branch_mask = self._count_limited_refiner_branch_mask(
            output=output,
            total_branch_count=refiner_total_branch_count,
            override=refiner_branch_mask,
        )

        if coarse_prediction_started is not None:
            _synchronize_for_model_timing(centerline_parameters.device)
            coarse_prediction_elapsed_ms = (
                time.perf_counter() - coarse_prediction_started
            ) * 1000.0
            output["coarse_prediction_elapsed_ms"] = (
                centerline_parameters.new_full(
                    (batch_size,), coarse_prediction_elapsed_ms
                )
            )

        branch_tokens = torch.cat(
            [main_token, conditioned_side_tokens], dim=1
        )
        refinement_total_started: float | None = None
        bspline_refiner_elapsed_ms: float | None = None
        radius_refiner_started: float | None = None
        radius_refiner_elapsed_ms: float | None = None
        if self.bspline_control_refiner is not None:
            if record_refiner_timing:
                _synchronize_for_refiner_timing(centerline_parameters.device)
                refinement_total_started = time.perf_counter()
            output, bspline_refiner_elapsed_ms = (
                self._run_bspline_control_refiner(
                    output=output,
                    branch_tokens=branch_tokens,
                    image_features=image_features,
                    images=images,
                    views=views,
                    view_mask=view_mask,
                    projection_center_offset=projection_center_offset,
                    refiner_branch_mask=refiner_branch_mask,
                    record_refiner_timing=record_refiner_timing,
                )
            )

        if self.radius_evidence_refiner is not None:
            if record_refiner_timing:
                _synchronize_for_refiner_timing(output["decoded_vessel_mm"].device)
                radius_refiner_started = time.perf_counter()
                if refinement_total_started is None:
                    refinement_total_started = radius_refiner_started
            output, radius_refiner_elapsed_ms = (
                self._run_radius_evidence_refiner(
                    output=output,
                    branch_tokens=branch_tokens,
                    images=images,
                    views=views,
                    view_mask=view_mask,
                    projection_center_offset=projection_center_offset,
                    refiner_branch_mask=refiner_branch_mask,
                    record_refiner_timing=record_refiner_timing,
                    canonical_geometry_member_index=(
                        radius_refiner_canonical_geometry_member_index
                    ),
                    canonical_geometry_p95_threshold_mm=(
                        radius_refiner_canonical_geometry_p95_threshold_mm
                    ),
                )
            )
        if record_refiner_timing and refinement_total_started is not None:
            _synchronize_for_refiner_timing(output["decoded_vessel_mm"].device)
            total_elapsed_ms = (
                time.perf_counter() - refinement_total_started
            ) * 1000.0
            timing_tensor = output["decoded_vessel_mm"].new_full(
                (batch_size,), total_elapsed_ms
            )
            output["refiner_total_elapsed_ms"] = timing_tensor
            if bspline_refiner_elapsed_ms is not None:
                output["bspline_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), bspline_refiner_elapsed_ms
                )
            if radius_refiner_elapsed_ms is not None:
                output["radius_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), radius_refiner_elapsed_ms
                )
        if return_refiner_context:
            # Evaluation-only context for rerunning the learned radius refiner
            # after an external geometry post-processing step. It is omitted by
            # default so training/inference outputs and checkpoints stay stable.
            output["_evaluation_refiner_branch_tokens"] = branch_tokens
        return output

class LegacyParallelParametricVesselPredictor(ParametricVesselPredictor):
    """Historical shared-decoder architecture used by older checkpoints.

    Before the main-first hierarchical decoder was introduced, every branch was
    decoded in parallel by ``branch_decoder``.  Side tokens were then fused with
    the main token through ``side_parent_fusion`` and all branches shared the
    same centreline, radius, and attachment heads.  Reconstructing that module
    graph is necessary for strict loading; ``strict=False`` would silently leave
    newly created side-branch modules randomly initialized.
    """

    def __init__(self, **kwargs: Any) -> None:
        mlp_hidden_dim = int(kwargs.get("mlp_hidden_dim", 1024))
        super().__init__(**kwargs)
        self.decoder_architecture = LEGACY_PARALLEL_DECODER

        # Remove modules that exist only in the newer main-first hierarchical
        # architecture.  These names must not appear in the legacy state dict.
        for name in (
            "side_branch_decoder",
            "side_parent_projection",
            "side_attachment_condition",
            "side_centerline_control_head",
            "side_radius_coefficient_head",
            "side_raw_radius_head",
            "side_lesion_exist_head",
            "side_lesion_geometry_head",
        ):
            if hasattr(self, name):
                delattr(self, name)

        self.side_parent_fusion = nn.Sequential(
            nn.Linear(2 * self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        smaller_hidden = max(128, mlp_hidden_dim // 2)
        self.attachment_head = nn.Sequential(
            nn.Linear(self.model_dim, smaller_hidden),
            nn.GELU(),
            nn.Linear(smaller_hidden, self.num_points),
        )

    def forward(
        self,
        *,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        image_features: torch.Tensor | dict[str, torch.Tensor],
        images: torch.Tensor | None = None,
        projection_center_offset: torch.Tensor | None = None,
        record_coarse_prediction_timing: bool = False,
        record_refiner_timing: bool = False,
        return_refiner_context: bool = False,
        refiner_branch_mask: torch.Tensor | None = None,
        refiner_total_branch_count: int | None = None,
        radius_refiner_canonical_geometry_member_index: int | None = None,
        radius_refiner_canonical_geometry_p95_threshold_mm: float | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        coarse_prediction_started: float | None = None
        if record_coarse_prediction_timing:
            _synchronize_for_model_timing(views.device)
            coarse_prediction_started = time.perf_counter()
        memory, padding_mask = self._encode_view_memory(
            views=views,
            view_mask=view_mask,
            image_features=image_features,
        )
        queries = self.branch_queries.unsqueeze(0).expand(
            memory.shape[0], -1, -1
        )
        decoded = self.branch_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        if self.num_branches > 1:
            main_token = decoded[:, :1].expand(-1, self.num_branches - 1, -1)
            side_tokens = self.side_parent_fusion(
                torch.cat((decoded[:, 1:], main_token), dim=-1)
            )
            decoded = torch.cat((decoded[:, :1], side_tokens), dim=1)

        batch_size = decoded.shape[0]
        centerline_parameters = self.centerline_control_head(decoded).view(
            batch_size,
            self.num_branches,
            self.num_centerline_parameters,
            3,
        )
        centerline_parameters = (
            centerline_parameters * self.centerline_output_scale_mm
        )
        attachment_logits = self.attachment_head(decoded)
        branch_exist_logits = self.branch_exist_head(decoded).squeeze(-1)

        if self.centerline_prediction_mode == "adaptive_landmarks":
            centerlines_mm, attachment_probabilities = decode_landmark_centerlines(
                centerline_parameters,
                attachment_logits,
                num_points=self.num_points,
                catmull_rom_alpha=self.catmull_rom_alpha,
                dense_samples=self.catmull_rom_dense_samples,
            )
        else:
            centerlines_mm, attachment_probabilities = decode_centerlines(
                centerline_parameters,
                attachment_logits,
                self.centerline_basis,
            )

        radius = self._predict_radius(decoded, side_branch=False)
        vessel_mm = torch.cat(
            (centerlines_mm, radius["radius_mm"].unsqueeze(-1)), dim=-1
        )
        branch_exist_probs = self._branch_existence_probabilities(
            branch_exist_logits
        )
        output = {
            "centerline_parameters_mm": centerline_parameters,
            "radius_baseline_coefficients_log_mm": radius[
                "radius_baseline_coefficients_log_mm"
            ],
            "raw_radius_log_mm": radius["raw_radius_log_mm"],
            "raw_radius_mm": radius["raw_radius_mm"],
            "attachment_logits": attachment_logits,
            "attachment_probabilities": attachment_probabilities,
            "lesion_exist_logits": radius["lesion_exist_logits"],
            "lesion_geometry": radius["lesion_geometry"],
            "branch_exist_logits": branch_exist_logits,
            "branch_exist_probs": branch_exist_probs,
            "decoded_vessel_mm": vessel_mm,
            "side_centerline_offsets_mm": torch.cat(
                [
                    centerlines_mm.new_zeros(
                        (batch_size, 1, self.num_points, 3)
                    ),
                    (
                        centerlines_mm[:, 1:]
                        - centerlines_mm[:, 1:, :1]
                    ),
                ],
                dim=1,
            ),
            "side_branch_relative_code_mm": torch.cat(
                [
                    (
                        centerlines_mm[:, 1:]
                        - centerlines_mm[:, 1:, :1]
                    ),
                    radius["radius_mm"][:, 1:].unsqueeze(-1),
                ],
                dim=-1,
            ),
        }
        if self.centerline_prediction_mode == "adaptive_landmarks":
            output["centerline_landmarks_mm"] = centerline_parameters
        else:
            output["centerline_control_points_mm"] = centerline_parameters

        refiner_branch_mask = self._count_limited_refiner_branch_mask(
            output=output,
            total_branch_count=refiner_total_branch_count,
            override=refiner_branch_mask,
        )

        if coarse_prediction_started is not None:
            _synchronize_for_model_timing(centerline_parameters.device)
            coarse_prediction_elapsed_ms = (
                time.perf_counter() - coarse_prediction_started
            ) * 1000.0
            output["coarse_prediction_elapsed_ms"] = (
                centerline_parameters.new_full(
                    (batch_size,), coarse_prediction_elapsed_ms
                )
            )

        refinement_total_started: float | None = None
        if record_refiner_timing and (
            self.bspline_control_refiner is not None
            or self.radius_evidence_refiner is not None
        ):
            _synchronize_for_refiner_timing(centerline_parameters.device)
            refinement_total_started = time.perf_counter()
        output, bspline_refiner_elapsed_ms = self._run_bspline_control_refiner(
            output=output,
            branch_tokens=decoded,
            image_features=image_features,
            images=images,
            views=views,
            view_mask=view_mask,
            projection_center_offset=projection_center_offset,
            refiner_branch_mask=refiner_branch_mask,
            record_refiner_timing=record_refiner_timing,
        )
        output, radius_refiner_elapsed_ms = self._run_radius_evidence_refiner(
            output=output,
            branch_tokens=decoded,
            images=images,
            views=views,
            view_mask=view_mask,
            projection_center_offset=projection_center_offset,
            refiner_branch_mask=refiner_branch_mask,
            record_refiner_timing=record_refiner_timing,
            canonical_geometry_member_index=(
                radius_refiner_canonical_geometry_member_index
            ),
            canonical_geometry_p95_threshold_mm=(
                radius_refiner_canonical_geometry_p95_threshold_mm
            ),
        )
        if record_refiner_timing and refinement_total_started is not None:
            _synchronize_for_refiner_timing(output["decoded_vessel_mm"].device)
            total_elapsed_ms = (
                time.perf_counter() - refinement_total_started
            ) * 1000.0
            timing_tensor = output["decoded_vessel_mm"].new_full(
                (batch_size,), total_elapsed_ms
            )
            output["refiner_total_elapsed_ms"] = timing_tensor
            if bspline_refiner_elapsed_ms is not None:
                output["bspline_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), bspline_refiner_elapsed_ms
                )
            if radius_refiner_elapsed_ms is not None:
                output["radius_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), radius_refiner_elapsed_ms
                )
        if return_refiner_context:
            output["_evaluation_refiner_branch_tokens"] = decoded
        return output

class AbsoluteParallelParametricVesselPredictor(
    LegacyParallelParametricVesselPredictor
):
    """Decode independent branches directly in one absolute coordinate frame.

    All branches share the decoder and prediction heads, but an identity target
    mask prevents cross-branch query attention. No main-token fusion,
    attachment head, or attachment-based translation is present in this
    architecture.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.decoder_architecture = ABSOLUTE_PARALLEL_DECODER
        for name in ("side_parent_fusion", "attachment_head"):
            if hasattr(self, name):
                delattr(self, name)
        self.register_buffer(
            "_absolute_parallel_decoder_marker",
            torch.ones((), dtype=torch.uint8),
            persistent=True,
        )

    def forward(
        self,
        *,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        image_features: torch.Tensor | dict[str, torch.Tensor],
        images: torch.Tensor | None = None,
        projection_center_offset: torch.Tensor | None = None,
        record_coarse_prediction_timing: bool = False,
        record_refiner_timing: bool = False,
        return_refiner_context: bool = False,
        refiner_branch_mask: torch.Tensor | None = None,
        refiner_total_branch_count: int | None = None,
        radius_refiner_canonical_geometry_member_index: int | None = None,
        radius_refiner_canonical_geometry_p95_threshold_mm: float | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        coarse_prediction_started: float | None = None
        if record_coarse_prediction_timing:
            _synchronize_for_model_timing(views.device)
            coarse_prediction_started = time.perf_counter()
        memory, padding_mask = self._encode_view_memory(
            views=views,
            view_mask=view_mask,
            image_features=image_features,
        )
        queries = self.branch_queries.unsqueeze(0).expand(
            memory.shape[0], -1, -1
        )
        target_mask = ~torch.eye(
            self.num_branches,
            device=queries.device,
            dtype=torch.bool,
        )
        decoded = self.branch_decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=target_mask,
            memory_key_padding_mask=padding_mask,
        )

        batch_size = decoded.shape[0]
        centerline_parameters = self.centerline_control_head(decoded).view(
            batch_size,
            self.num_branches,
            self.num_centerline_parameters,
            3,
        )
        centerline_parameters = (
            centerline_parameters * self.centerline_output_scale_mm
        )
        centerlines_mm = self._decode_local_centerlines(
            centerline_parameters
        )
        radius = self._predict_radius(decoded, side_branch=False)
        vessel_mm = torch.cat(
            (centerlines_mm, radius["radius_mm"].unsqueeze(-1)), dim=-1
        )
        branch_exist_logits = self.branch_exist_head(decoded).squeeze(-1)
        branch_exist_probs = self._branch_existence_probabilities(
            branch_exist_logits
        )
        side_offsets_mm = (
            centerlines_mm[:, 1:] - centerlines_mm[:, 1:, :1]
        )
        output = {
            "centerline_parameters_mm": centerline_parameters,
            "radius_baseline_coefficients_log_mm": radius[
                "radius_baseline_coefficients_log_mm"
            ],
            "raw_radius_log_mm": radius["raw_radius_log_mm"],
            "raw_radius_mm": radius["raw_radius_mm"],
            "lesion_exist_logits": radius["lesion_exist_logits"],
            "lesion_geometry": radius["lesion_geometry"],
            "branch_exist_logits": branch_exist_logits,
            "branch_exist_probs": branch_exist_probs,
            "decoded_vessel_mm": vessel_mm,
            "side_centerline_offsets_mm": torch.cat(
                [
                    centerlines_mm.new_zeros(
                        (batch_size, 1, self.num_points, 3)
                    ),
                    side_offsets_mm,
                ],
                dim=1,
            ),
            "side_branch_relative_code_mm": torch.cat(
                [
                    side_offsets_mm,
                    radius["radius_mm"][:, 1:].unsqueeze(-1),
                ],
                dim=-1,
            ),
        }
        if self.centerline_prediction_mode == "adaptive_landmarks":
            output["centerline_landmarks_mm"] = centerline_parameters
        else:
            output["centerline_control_points_mm"] = centerline_parameters

        refiner_branch_mask = self._count_limited_refiner_branch_mask(
            output=output,
            total_branch_count=refiner_total_branch_count,
            override=refiner_branch_mask,
        )

        if coarse_prediction_started is not None:
            _synchronize_for_model_timing(centerline_parameters.device)
            coarse_prediction_elapsed_ms = (
                time.perf_counter() - coarse_prediction_started
            ) * 1000.0
            output["coarse_prediction_elapsed_ms"] = (
                centerline_parameters.new_full(
                    (batch_size,), coarse_prediction_elapsed_ms
                )
            )

        refinement_total_started: float | None = None
        if record_refiner_timing and (
            self.bspline_control_refiner is not None
            or self.radius_evidence_refiner is not None
        ):
            _synchronize_for_refiner_timing(centerline_parameters.device)
            refinement_total_started = time.perf_counter()
        output, bspline_refiner_elapsed_ms = self._run_bspline_control_refiner(
            output=output,
            branch_tokens=decoded,
            image_features=image_features,
            images=images,
            views=views,
            view_mask=view_mask,
            projection_center_offset=projection_center_offset,
            refiner_branch_mask=refiner_branch_mask,
            record_refiner_timing=record_refiner_timing,
        )
        output, radius_refiner_elapsed_ms = self._run_radius_evidence_refiner(
            output=output,
            branch_tokens=decoded,
            images=images,
            views=views,
            view_mask=view_mask,
            projection_center_offset=projection_center_offset,
            refiner_branch_mask=refiner_branch_mask,
            record_refiner_timing=record_refiner_timing,
            canonical_geometry_member_index=(
                radius_refiner_canonical_geometry_member_index
            ),
            canonical_geometry_p95_threshold_mm=(
                radius_refiner_canonical_geometry_p95_threshold_mm
            ),
        )
        if record_refiner_timing and refinement_total_started is not None:
            _synchronize_for_refiner_timing(output["decoded_vessel_mm"].device)
            total_elapsed_ms = (
                time.perf_counter() - refinement_total_started
            ) * 1000.0
            timing_tensor = output["decoded_vessel_mm"].new_full(
                (batch_size,), total_elapsed_ms
            )
            output["refiner_total_elapsed_ms"] = timing_tensor
            if bspline_refiner_elapsed_ms is not None:
                output["bspline_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), bspline_refiner_elapsed_ms
                )
            if radius_refiner_elapsed_ms is not None:
                output["radius_refiner_elapsed_ms"] = timing_tensor.new_full(
                    (batch_size,), radius_refiner_elapsed_ms
                )
        if return_refiner_context:
            output["_evaluation_refiner_branch_tokens"] = decoded
        return output

def infer_checkpoint_decoder_architecture(
    state_dict: dict[str, torch.Tensor],
) -> str:
    """Infer the parametric decoder graph from unambiguous state-dict keys."""

    keys = tuple(state_dict)
    has_absolute_marker = any(
        key == "_absolute_parallel_decoder_marker"
        or key.endswith("._absolute_parallel_decoder_marker")
        for key in keys
    )
    has_legacy_fusion = any(
        key == "side_parent_fusion.0.weight"
        or key.endswith(".side_parent_fusion.0.weight")
        for key in keys
    )
    has_hierarchical_decoder = any(
        key.startswith("side_branch_decoder.")
        or ".side_branch_decoder." in key
        for key in keys
    )
    if sum(
        (has_absolute_marker, has_legacy_fusion, has_hierarchical_decoder)
    ) > 1:
        raise ValueError(
            "Checkpoint mixes absolute-parallel, legacy side_parent_fusion, "
            "and/or hierarchical side_branch_decoder signatures; its "
            "architecture is ambiguous."
        )
    if has_absolute_marker:
        return ABSOLUTE_PARALLEL_DECODER
    if has_legacy_fusion:
        return LEGACY_PARALLEL_DECODER
    return MAIN_FIRST_HIERARCHICAL_DECODER

def build_model_from_config(
    config: dict[str, Any],
    view_feat_dim: int,
    inferred_feature_dim: int | None = None,
    *,
    decoder_architecture: str | None = None,
) -> ParametricVesselPredictor:
    feature_backbone = normalize_feature_backbone(config.get("feature_backbone", "vggt"))
    merged = dict(config)
    merged.update(dict(config.get("model", {}) or {}))
    vggt_dim = merged.get("vggt_token_dim", inferred_feature_dim or 2048)
    model_num_branches = int(merged.get("num_branches", 7))
    target_num_branches = int(
        config.get("num_branches", model_num_branches)
    )
    if inferred_feature_dim is not None and feature_backbone.startswith("vggt"):
        vggt_dim = int(inferred_feature_dim)
    default_camera_size = 256
    camera_size = merged.get(
        "camera_encoding_image_size",
        merged.get("projection_evidence_image_size", default_camera_size),
    )
    if camera_size is None:
        camera_height = camera_width = default_camera_size
    elif isinstance(camera_size, (int, float)):
        camera_height = camera_width = int(camera_size)
    elif isinstance(camera_size, (list, tuple)) and len(camera_size) == 2:
        camera_height, camera_width = int(camera_size[0]), int(camera_size[1])
    else:
        raise ValueError(
            "camera_encoding_image_size must be an integer or [height, width], "
            f"got {camera_size!r}."
        )
    default_vggt_target_size = 256 if feature_backbone == "vggt_omega" else 266
    vggt_target_size = merged.get(
        "vggt_target_image_size", default_vggt_target_size
    )
    if vggt_target_size is None:
        vggt_target_height = vggt_target_width = default_vggt_target_size
    elif isinstance(vggt_target_size, (int, float)):
        vggt_target_height = vggt_target_width = int(vggt_target_size)
    elif isinstance(vggt_target_size, (list, tuple)) and len(vggt_target_size) == 2:
        vggt_target_height = int(vggt_target_size[0])
        vggt_target_width = int(vggt_target_size[1])
    else:
        raise ValueError(
            "vggt_target_image_size must be an integer or [height, width], "
            f"got {vggt_target_size!r}."
        )
    post_backbone_default = feature_backbone == "resnet_pre_fpn"
    architecture = normalize_decoder_architecture(
        merged.get("decoder_architecture", MAIN_FIRST_HIERARCHICAL_DECODER)
        if decoder_architecture is None
        else decoder_architecture
    )
    if architecture == MAIN_FIRST_HIERARCHICAL_DECODER:
        predictor_class = ParametricVesselPredictor
    elif architecture == LEGACY_PARALLEL_DECODER:
        predictor_class = LegacyParallelParametricVesselPredictor
    elif architecture == ABSOLUTE_PARALLEL_DECODER:
        predictor_class = AbsoluteParallelParametricVesselPredictor
    else:
        raise AssertionError(f"Unhandled decoder architecture: {architecture}")
    refine_branch_existence = merged.get(
        "bspline_refiner_refine_branch_existence",
        False,
    )
    if not isinstance(refine_branch_existence, bool):
        raise ValueError(
            "bspline_refiner_refine_branch_existence must be a JSON boolean, "
            f"got {refine_branch_existence!r}."
        )
    use_spatial_vggt_features = merged.get(
        "bspline_refiner_use_spatial_vggt_features",
        False,
    )
    if not isinstance(use_spatial_vggt_features, bool):
        raise ValueError(
            "bspline_refiner_use_spatial_vggt_features must be a JSON boolean, "
            f"got {use_spatial_vggt_features!r}."
        )
    if use_spatial_vggt_features and feature_backbone not in {
        "vggt",
        "vggt_omega",
    }:
        raise ValueError(
            "bspline_refiner_use_spatial_vggt_features=true requires a VGGT "
            f"feature backbone, got {feature_backbone!r}."
        )
    if use_spatial_vggt_features and merged.get(
        "zero_cached_image_features", False
    ):
        raise ValueError(
            "Spatial VGGT refiner evidence cannot be enabled together with "
            "zero_cached_image_features=true because that removes the signal "
            "being sampled."
        )
    use_3d_candidates = merged.get(
        "bspline_refiner_use_3d_candidates",
        False,
    )
    if not isinstance(use_3d_candidates, bool):
        raise ValueError(
            "bspline_refiner_use_3d_candidates must be a JSON boolean, got "
            f"{use_3d_candidates!r}."
        )
    if use_3d_candidates and not use_spatial_vggt_features:
        raise ValueError(
            "bspline_refiner_use_3d_candidates=true requires "
            "bspline_refiner_use_spatial_vggt_features=true."
        )
    if use_3d_candidates and architecture != ABSOLUTE_PARALLEL_DECODER:
        raise ValueError(
            "bspline_refiner_use_3d_candidates=true currently requires "
            "decoder_architecture='absolute_parallel'."
        )
    use_unexplained_centerline_evidence = merged.get(
        "bspline_refiner_use_unexplained_centerline_evidence",
        False,
    )
    if not isinstance(use_unexplained_centerline_evidence, bool):
        raise ValueError(
            "bspline_refiner_use_unexplained_centerline_evidence must be a "
            "JSON boolean, got "
            f"{use_unexplained_centerline_evidence!r}."
        )
    if use_unexplained_centerline_evidence and not use_3d_candidates:
        raise ValueError(
            "bspline_refiner_use_unexplained_centerline_evidence=true "
            "requires bspline_refiner_use_3d_candidates=true."
        )
    use_centerline_probability_patch_evidence = merged.get(
        "bspline_refiner_use_centerline_probability_patch_evidence",
        False,
    )
    if not isinstance(use_centerline_probability_patch_evidence, bool):
        raise ValueError(
            "bspline_refiner_use_centerline_probability_patch_evidence must be "
            "a JSON boolean, got "
            f"{use_centerline_probability_patch_evidence!r}."
        )
    use_refiner_learned_image_features = merged.get(
        "bspline_refiner_use_learned_image_features",
        True,
    )
    if not isinstance(use_refiner_learned_image_features, bool):
        raise ValueError(
            "bspline_refiner_use_learned_image_features must be a JSON "
            f"boolean, got {use_refiner_learned_image_features!r}."
        )
    if (
        (
            use_unexplained_centerline_evidence
            or use_centerline_probability_patch_evidence
        )
        and not use_refiner_learned_image_features
    ):
        raise ValueError(
            "Centreline-probability evidence requires "
            "bspline_refiner_use_learned_image_features=true."
        )
    detach_unexplained_centerline_evidence = merged.get(
        "bspline_refiner_detach_unexplained_centerline_evidence",
        True,
    )
    if not isinstance(detach_unexplained_centerline_evidence, bool):
        raise ValueError(
            "bspline_refiner_detach_unexplained_centerline_evidence must be a "
            "JSON boolean, got "
            f"{detach_unexplained_centerline_evidence!r}."
        )
    detach_centerline_probability_patch_evidence = merged.get(
        "bspline_refiner_detach_centerline_probability_patch_evidence",
        True,
    )
    if not isinstance(detach_centerline_probability_patch_evidence, bool):
        raise ValueError(
            "bspline_refiner_detach_centerline_probability_patch_evidence must "
            "be a JSON boolean, got "
            f"{detach_centerline_probability_patch_evidence!r}."
        )
    candidate_spacing_value = merged.get(
        "bspline_refiner_candidate_spacing_mm",
        (4.0, 2.0, 1.0, 0.5),
    )
    if isinstance(candidate_spacing_value, bool) or not isinstance(
        candidate_spacing_value, (list, tuple)
    ):
        raise ValueError(
            "bspline_refiner_candidate_spacing_mm must be a JSON array with "
            f"one value per stage, got {candidate_spacing_value!r}."
        )
    candidate_spacing_mm = tuple(
        float(value) for value in candidate_spacing_value
    )
    return predictor_class(
        feature_backbone=feature_backbone,
        num_points=int(merged.get("num_points", 200)),
        num_branches=model_num_branches,
        num_control_points=int(merged.get("num_control_points", 20)),
        num_landmarks=int(
            merged.get("num_landmarks", merged.get("num_control_points", 20))
        ),
        num_radius_coefficients=int(merged.get("num_radius_coefficients", 6)),
        num_lesions=int(merged.get("num_lesions", 3)),
        lesion_profile=str(merged.get("lesion_profile", "gaussian")),
        centerline_prediction_mode=merged.get(
            "centerline_prediction_mode", "bspline_control_points"
        ),
        radius_prediction_mode=merged.get("radius_prediction_mode", "parametric"),
        centerline_spline_degree=int(merged.get("centerline_spline_degree", 3)),
        catmull_rom_alpha=float(merged.get("catmull_rom_alpha", 0.5)),
        catmull_rom_dense_samples=int(
            merged.get("catmull_rom_dense_samples", 1000)
        ),
        radius_spline_degree=int(merged.get("radius_spline_degree", 3)),
        control_point_output_scale_mm=float(merged.get("control_point_output_scale_mm", 100.0)),
        centerline_output_scale_mm=(
            None
            if "centerline_output_scale_mm" not in merged
            else float(merged["centerline_output_scale_mm"])
        ),
        view_feat_dim=int(view_feat_dim),
        model_dim=int(merged.get("model_dim", 512)),
        num_encoder_layers=int(merged.get("num_encoder_layers", 2)),
        num_decoder_layers=int(merged.get("num_decoder_layers", 2)),
        num_attention_heads=int(merged.get("num_attention_heads", 8)),
        mlp_hidden_dim=int(merged.get("mlp_hidden_dim", 1024)),
        dropout=float(merged.get("dropout", 0.1)),
        use_encoded_view_dir=bool(merged.get("use_encoded_view_dir", True)),
        use_post_backbone_transformer_encoder=bool(
            merged.get(
                "use_post_backbone_transformer_encoder", post_backbone_default
            )
        ),
        use_ray_camera_encoding=bool(
            merged.get("use_ray_camera_encoding", False)
        ),
        ray_camera_encoding_dim=int(merged.get("ray_camera_encoding_dim", 128)),
        use_prope_attention=bool(merged.get("use_prope_attention", False)),
        prope_frequency_base=float(merged.get("prope_frequency_base", 100.0)),
        camera_encoding_image_height=camera_height,
        camera_encoding_image_width=camera_width,
        camera_encoding_sid=float(
            merged.get("camera_encoding_sid", merged.get("proj_loss_sid", 0.9))
        ),
        camera_encoding_source_to_iso=float(
            merged.get(
                "camera_encoding_source_to_iso",
                merged.get("proj_loss_source_to_iso", 0.75),
            )
        ),
        camera_encoding_imager_pixel_spacing=float(
            merged.get(
                "camera_encoding_imager_pixel_spacing",
                merged.get("proj_loss_imager_pixel_spacing", 0.55),
            )
        ),
        vggt_patch_size=int(
            merged.get(
                "vggt_patch_size",
                16 if feature_backbone == "vggt_omega" else 14,
            )
        ),
        vggt_target_image_height=vggt_target_height,
        vggt_target_image_width=vggt_target_width,
        vggt_token_dim=int(vggt_dim),
        vggt_image_size_mode=str(
            merged.get("vggt_image_size_mode", "resize_to_patch_multiple")
        ),
        resnet_pre_fpn_channels=tuple(
            int(value)
            for value in merged.get("resnet_pre_fpn_channels", (256, 512, 1024, 2048))
        ),
        image_fpn_pool_size=int(merged.get("image_fpn_pool_size", 8)),
        use_learned_side_parent_projection=bool(
            merged.get("use_learned_side_parent_projection", True)
        ),
        use_bspline_control_refiner=bool(
            merged.get("use_bspline_control_refiner", False)
        ),
        bspline_refiner_num_stages=int(
            merged.get("bspline_refiner_num_stages", 4)
        ),
        bspline_refiner_evidence_hidden_dim=int(
            merged.get("bspline_refiner_evidence_hidden_dim", 256)
        ),
        bspline_refiner_patch_size=int(
            merged.get("bspline_refiner_patch_size", 9)
        ),
        bspline_refiner_use_learned_image_features=(
            use_refiner_learned_image_features
        ),
        bspline_refiner_learned_feature_dim=int(
            merged.get("bspline_refiner_learned_feature_dim", 32)
        ),
        bspline_refiner_use_distance_transform=bool(
            merged.get("bspline_refiner_use_distance_transform", True)
        ),
        bspline_refiner_distance_transform_num_iters=int(
            merged.get("bspline_refiner_distance_transform_num_iters", 64)
        ),
        bspline_refiner_image_size=int(
            merged.get("bspline_refiner_image_size", 256)
        ),
        bspline_refiner_sid=float(
            merged.get("bspline_refiner_sid", merged.get("proj_loss_sid", 0.9))
        ),
        bspline_refiner_source_to_iso=float(
            merged.get(
                "bspline_refiner_source_to_iso",
                merged.get("proj_loss_source_to_iso", 0.75),
            )
        ),
        bspline_refiner_imager_pixel_spacing=float(
            merged.get(
                "bspline_refiner_imager_pixel_spacing",
                merged.get("proj_loss_imager_pixel_spacing", 0.55),
            )
        ),
        bspline_refiner_residual_scale_mm=float(
            merged.get("bspline_refiner_residual_scale_mm", 5.0)
        ),
        bspline_refiner_control_position_scale_mm=float(
            merged.get("bspline_refiner_control_position_scale_mm", 100.0)
        ),
        bspline_refiner_refine_branch_existence=refine_branch_existence,
        bspline_refiner_branch_existence_candidate_threshold=float(
            merged.get(
                "bspline_refiner_branch_existence_candidate_threshold",
                0.5,
            )
        ),
        bspline_refiner_branch_existence_max_logit_decrease=float(
            merged.get(
                "bspline_refiner_branch_existence_max_logit_decrease",
                10.0,
            )
        ),
        bspline_refiner_use_spatial_vggt_features=use_spatial_vggt_features,
        bspline_refiner_spatial_vggt_feature_dim=int(
            merged.get("bspline_refiner_spatial_vggt_feature_dim", 64)
        ),
        bspline_refiner_use_3d_candidates=use_3d_candidates,
        bspline_refiner_candidate_pattern=str(
            merged.get("bspline_refiner_candidate_pattern", "axis_7")
        ),
        bspline_refiner_candidate_spacing_mm=candidate_spacing_mm,
        bspline_refiner_candidate_hidden_dim=int(
            merged.get("bspline_refiner_candidate_hidden_dim", 64)
        ),
        bspline_refiner_candidate_num_attention_heads=int(
            merged.get(
                "bspline_refiner_candidate_num_attention_heads",
                4,
            )
        ),
        bspline_refiner_candidate_score_temperature=float(
            merged.get("bspline_refiner_candidate_score_temperature", 1.0)
        ),
        bspline_refiner_use_unexplained_centerline_evidence=(
            use_unexplained_centerline_evidence
        ),
        bspline_refiner_use_centerline_probability_patch_evidence=(
            use_centerline_probability_patch_evidence
        ),
        bspline_refiner_centerline_probability_patch_size=int(
            merged.get(
                "bspline_refiner_centerline_probability_patch_size",
                9,
            )
        ),
        bspline_refiner_detach_centerline_probability_patch_evidence=(
            detach_centerline_probability_patch_evidence
        ),
        bspline_refiner_centerline_probability_target_mode=str(
            merged.get(
                "bspline_refiner_centerline_probability_target_mode",
                GAUSSIAN_CENTERLINE_TARGET,
            )
        ),
        bspline_refiner_centerline_probability_target_gamma=float(
            merged.get(
                "bspline_refiner_centerline_probability_target_gamma",
                2.0,
            )
        ),
        bspline_refiner_centerline_probability_mask_threshold=float(
            merged.get(
                "bspline_refiner_centerline_probability_mask_threshold",
                0.5,
            )
        ),
        bspline_refiner_centerline_probability_unexplained_power=float(
            merged.get(
                "bspline_refiner_centerline_probability_unexplained_power",
                1.0,
            )
        ),
        bspline_refiner_use_separate_centerline_encoder=bool(
            merged.get(
                "bspline_refiner_use_separate_centerline_encoder",
                False,
            )
        ),
        bspline_refiner_centerline_map_size=int(
            merged.get("bspline_refiner_centerline_map_size", 64)
        ),
        bspline_refiner_centerline_head_hidden_dim=int(
            merged.get("bspline_refiner_centerline_head_hidden_dim", 16)
        ),
        bspline_refiner_centerline_map_sigma_px=float(
            merged.get("bspline_refiner_centerline_map_sigma_px", 1.25)
        ),
        bspline_refiner_centerline_map_radius_px=int(
            merged.get("bspline_refiner_centerline_map_radius_px", 2)
        ),
        bspline_refiner_unexplained_centerline_pool_kernel=int(
            merged.get(
                "bspline_refiner_unexplained_centerline_pool_kernel",
                5,
            )
        ),
        bspline_refiner_detach_unexplained_centerline_evidence=(
            detach_unexplained_centerline_evidence
        ),
        use_radius_evidence_refiner=bool(
            merged.get("use_radius_evidence_refiner", False)
        ),
        radius_refiner_num_stages=int(
            merged.get("radius_refiner_num_stages", 3)
        ),
        radius_refiner_evidence_hidden_dim=int(
            merged.get("radius_refiner_evidence_hidden_dim", 256)
        ),
        radius_refiner_profile_samples=int(
            merged.get("radius_refiner_profile_samples", 25)
        ),
        radius_refiner_profile_half_width_px=float(
            merged.get("radius_refiner_profile_half_width_px", 16.0)
        ),
        radius_refiner_image_size=int(
            merged.get("radius_refiner_image_size", 256)
        ),
        radius_refiner_sid=float(
            merged.get("radius_refiner_sid", merged.get("proj_loss_sid", 0.9))
        ),
        radius_refiner_source_to_iso=float(
            merged.get(
                "radius_refiner_source_to_iso",
                merged.get("proj_loss_source_to_iso", 0.75),
            )
        ),
        radius_refiner_imager_pixel_spacing=float(
            merged.get(
                "radius_refiner_imager_pixel_spacing",
                merged.get("proj_loss_imager_pixel_spacing", 0.55),
            )
        ),
        radius_refiner_residual_scale_mm=float(
            merged.get("radius_refiner_residual_scale_mm", 1.0)
        ),
        radius_refiner_radius_value_scale_mm=float(
            merged.get("radius_refiner_radius_value_scale_mm", 5.0)
        ),
        radius_refiner_min_radius_mm=float(
            merged.get("radius_refiner_min_radius_mm", 0.05)
        ),
        radius_refiner_render_num_circle_points=int(
            merged.get("radius_refiner_render_num_circle_points", 24)
        ),
        radius_refiner_render_radial_subsamples=int(
            merged.get("radius_refiner_render_radial_subsamples", 1)
        ),
        radius_refiner_render_axial_subsamples=int(
            merged.get("radius_refiner_render_axial_subsamples", 2)
        ),
        radius_refiner_branch_probability_threshold=float(
            merged.get("radius_refiner_branch_probability_threshold", 0.5)
        ),
        fixed_main_branch_count=required_branch_count(
            resolve_artery_type(merged)
        ),
        target_num_branches=target_num_branches,
        projection_coord_scale_to_meter=float(
            merged.get("projection_coord_scale_to_meter", 0.001)
        ),
        target_coordinate_frame=str(
            merged.get("target_coordinate_frame", "projection_centered")
        ),
    )

def build_model_for_checkpoint(
    config: dict[str, Any],
    view_feat_dim: int,
    inferred_feature_dim: int | None,
    state_dict: dict[str, torch.Tensor],
) -> ParametricVesselPredictor:
    """Build the exact decoder family encoded by a checkpoint state dict."""

    architecture = infer_checkpoint_decoder_architecture(state_dict)
    return build_model_from_config(
        config,
        view_feat_dim,
        inferred_feature_dim,
        decoder_architecture=architecture,
    )
