# Transferred from methods/src/model.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import copy
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from vessel_code.backbones.image_support.model_realdata import _fpn_2d_sine_position_encoding
from vessel_code.backbones.vggt_support.model_realdata import _ProjectionEvidenceResidualRefiner
from vessel_code.shared.camera_encoding import CArmCameraGeometry, CameraTokenLayout, PRoPETransformerEncoder, build_camera_token_layout, build_carm_camera_geometry, plucker_ray_embeddings
from vessel_code.shared.data import RESNET_PRE_FPN_SUFFIXES, normalize_feature_backbone

MAIN_FIRST_HIERARCHICAL_DECODER = "main_first_hierarchical"

LEGACY_PARALLEL_DECODER = "legacy_parallel"

ABSOLUTE_PARALLEL_DECODER = "absolute_parallel"

def normalize_decoder_architecture(value: Any) -> str:
    """Normalize the supported branch-decoder graph names."""

    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        MAIN_FIRST_HIERARCHICAL_DECODER: MAIN_FIRST_HIERARCHICAL_DECODER,
        "hierarchical": MAIN_FIRST_HIERARCHICAL_DECODER,
        "main_plus_side": MAIN_FIRST_HIERARCHICAL_DECODER,
        LEGACY_PARALLEL_DECODER: LEGACY_PARALLEL_DECODER,
        "parallel": LEGACY_PARALLEL_DECODER,
        "shared": LEGACY_PARALLEL_DECODER,
        "shared_all_branches": LEGACY_PARALLEL_DECODER,
        ABSOLUTE_PARALLEL_DECODER: ABSOLUTE_PARALLEL_DECODER,
        "absolute": ABSOLUTE_PARALLEL_DECODER,
        "independent_absolute": ABSOLUTE_PARALLEL_DECODER,
    }
    if key not in aliases:
        raise ValueError(
            "decoder_architecture must be 'main_first_hierarchical', "
            f"'legacy_parallel', or 'absolute_parallel', got {value!r}."
        )
    return aliases[key]

class ResNetPreFPNTokenProjector(nn.Module):
    """Trainable FPN over cached ResNet C2-C5 feature maps."""

    def __init__(
        self,
        model_dim: int,
        in_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        pool_size: int = 8,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.pool_size = int(pool_size)
        if len(in_channels) != 4:
            raise ValueError(f"in_channels must contain four values for C2-C5, got {in_channels}")
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(int(channels), self.model_dim, kernel_size=1) for channels in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(self.model_dim, self.model_dim, kernel_size=3, padding=1) for _ in range(4)]
        )
        self.fpn_level_embeddings = nn.Parameter(torch.empty(4, self.model_dim))
        nn.init.normal_(self.fpn_level_embeddings, mean=0.0, std=0.02)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [key for key in RESNET_PRE_FPN_SUFFIXES if key not in features]
        if missing:
            raise KeyError(f"resnet_pre_fpn image_features is missing keys: {missing}")
        c2, c3, c4, c5 = [features[key] for key in RESNET_PRE_FPN_SUFFIXES]
        batch_size, num_views = c2.shape[:2]
        maps = []
        for value in (c2, c3, c4, c5):
            if value.dim() != 5:
                raise ValueError(f"ResNet pre-FPN maps must have shape [B,V,C,H,W], got {tuple(value.shape)}")
            if value.shape[:2] != (batch_size, num_views):
                raise ValueError("All ResNet pre-FPN maps must share batch and view dimensions.")
            maps.append(value.reshape(batch_size * num_views, *value.shape[2:]).to(dtype=self.lateral_convs[0].weight.dtype))

        pyramid = []
        previous = None
        for feature, lateral_conv, output_conv in zip(
            (maps[3], maps[2], maps[1], maps[0]),
            reversed(self.lateral_convs),
            reversed(self.output_convs),
        ):
            lateral = lateral_conv(feature)
            if previous is not None:
                lateral = lateral + F.interpolate(previous, size=lateral.shape[-2:], mode="nearest")
            previous = lateral
            output = output_conv(lateral)
            if self.pool_size > 0:
                output = F.adaptive_avg_pool2d(output, (self.pool_size, self.pool_size))
            pyramid.append(output)

        tokens = []
        for level_index, output in enumerate(reversed(pyramid)):
            position = _fpn_2d_sine_position_encoding(
                output.shape[-2],
                output.shape[-1],
                output.shape[1],
                device=output.device,
                dtype=output.dtype,
            )
            level = self.fpn_level_embeddings[level_index].to(dtype=output.dtype).view(1, -1, 1, 1)
            tokens.append((output + position + level).flatten(2).transpose(1, 2))
        tokens = torch.cat(tokens, dim=1)
        return tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])

class PrecomputedFeatureVesselPredictor(nn.Module):
    """
    Multi-view vessel-code predictor that consumes cached image features.

    Supported cached features:
    - vggt / vggt_omega: image_features [B,V,L,Dv]
    - resnet_pre_fpn: image_features_{c2,c3,c4,c5} [B,V,C,H,W], then trainable FPN

    Both ResNet and VGGT tokens can retain or bypass an additional cross-view
    Transformer through ``use_post_backbone_transformer_encoder``. The token
    fusion can independently include a global four-angle feature and a dense
    patch-centre Plucker ray. The optional post-backbone Transformer can use
    either standard self-attention or PRoPE projective attention.
    """

    def __init__(
        self,
        feature_backbone: str,
        num_points: int = 200,
        num_branches: int = 1,
        view_feat_dim: int = 4,
        model_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_attention_heads: int = 8,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
        point_head_mode: str = "split_xyzr",
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
        resnet_pre_fpn_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        image_fpn_pool_size: int = 8,
        use_learned_side_parent_projection: bool = True,
        use_centerline_point_index_embedding: bool = True,
        centerline_point_embed_dim: int = 32,
        use_radius_point_index_embedding: bool = True,
        use_projection_evidence_residual_refiner: bool = False,
        projection_evidence_hidden_dim: int | None = None,
        projection_evidence_patch_size: int = 3,
        projection_evidence_use_learned_image_features: bool = True,
        projection_evidence_learned_feature_dim: int = 32,
        projection_evidence_use_distance_transform: bool = True,
        projection_evidence_distance_transform_num_iters: int = 64,
        projection_evidence_image_size: int = 256,
        projection_evidence_sid: float = 0.9,
        projection_evidence_source_to_iso: float = 0.75,
        projection_evidence_imager_pixel_spacing: float = 0.55,
        projection_evidence_coord_scale_to_meter: float = 0.001,
        projection_evidence_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_backbone = normalize_feature_backbone(feature_backbone)
        self.num_branches = int(num_branches)
        self.num_points = int(num_points)
        self.model_dim = int(model_dim)
        self.decoder_architecture = MAIN_FIRST_HIERARCHICAL_DECODER
        self.use_learned_side_parent_projection = bool(
            use_learned_side_parent_projection
        )
        self.point_head_mode = str(point_head_mode).strip().lower()
        self.use_encoded_view_dir = bool(use_encoded_view_dir)
        self.use_post_backbone_transformer_encoder = (
            self.feature_backbone == "resnet_pre_fpn"
            if use_post_backbone_transformer_encoder is None
            else bool(use_post_backbone_transformer_encoder)
        )
        self.use_ray_camera_encoding = bool(use_ray_camera_encoding)
        self.use_prope_attention = bool(use_prope_attention)
        self.camera_encoding_image_height = int(camera_encoding_image_height)
        self.camera_encoding_image_width = int(camera_encoding_image_width)
        self.camera_encoding_sid = float(camera_encoding_sid)
        self.camera_encoding_source_to_iso = float(camera_encoding_source_to_iso)
        self.camera_encoding_imager_pixel_spacing = float(
            camera_encoding_imager_pixel_spacing
        )
        self.vggt_patch_size = int(vggt_patch_size)
        self.vggt_target_image_height = int(vggt_target_image_height)
        self.vggt_target_image_width = int(vggt_target_image_width)
        self.view_feat_dim = int(view_feat_dim)
        if self.num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {self.num_branches}")
        if self.num_points < 2:
            raise ValueError(f"num_points must be >= 2, got {self.num_points}")
        if self.point_head_mode not in ("point", "split_xyzr"):
            raise ValueError(f"point_head_mode must be 'point' or 'split_xyzr', got {point_head_mode!r}.")
        if (self.use_ray_camera_encoding or self.use_prope_attention) and self.view_feat_dim != 4:
            raise ValueError(
                "Ray camera encoding and PRoPE currently require view_feat_dim=4 "
                "with [sin(theta), cos(theta), sin(phi), cos(phi)]."
            )
        if self.use_prope_attention and not self.use_post_backbone_transformer_encoder:
            raise ValueError(
                "use_prope_attention=true requires "
                "use_post_backbone_transformer_encoder=true."
            )
        if self.camera_encoding_image_height < 1 or self.camera_encoding_image_width < 1:
            raise ValueError("Camera encoding image height and width must be positive.")

        self.resnet_pre_fpn = None
        if self.feature_backbone == "resnet_pre_fpn":
            self.resnet_pre_fpn = ResNetPreFPNTokenProjector(
                model_dim=self.model_dim,
                in_channels=tuple(int(v) for v in resnet_pre_fpn_channels),
                pool_size=int(image_fpn_pool_size),
            )
            image_token_dim = self.model_dim
            self.image_token_projection = nn.Identity()
        else:
            image_token_dim = int(vggt_token_dim)
            self.image_token_projection = nn.Sequential(
                nn.Linear(image_token_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
            )
            image_token_dim = self.model_dim

        conditioning_dim = 0
        if self.use_encoded_view_dir:
            # Fuse the view direction with every image token in the same way
            # for both cached backbones.  In particular, VGGT no longer uses a
            # separate additive direction embedding.
            self.view_encoder = nn.Sequential(
                nn.Linear(self.view_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            conditioning_dim += 128
        else:
            self.view_encoder = None
        if self.use_ray_camera_encoding:
            ray_dim = int(ray_camera_encoding_dim)
            if ray_dim < 1:
                raise ValueError("ray_camera_encoding_dim must be positive.")
            self.ray_encoder = nn.Sequential(
                nn.Linear(6, ray_dim),
                nn.GELU(),
                nn.Linear(ray_dim, ray_dim),
            )
            conditioning_dim += ray_dim
        else:
            self.ray_encoder = None
        if conditioning_dim > 0:
            self.token_projection = nn.Sequential(
                nn.Linear(image_token_dim + conditioning_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
            )
        else:
            self.token_projection = nn.Identity()

        self.view_encoder_transformer: nn.TransformerEncoder | None = None
        self.prope_encoder: PRoPETransformerEncoder | None = None
        if self.use_post_backbone_transformer_encoder:
            if self.use_prope_attention:
                self.prope_encoder = PRoPETransformerEncoder(
                    model_dim=self.model_dim,
                    num_heads=int(num_attention_heads),
                    mlp_hidden_dim=int(mlp_hidden_dim),
                    dropout=float(dropout),
                    num_layers=int(num_encoder_layers),
                    frequency_base=float(prope_frequency_base),
                )
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.model_dim,
                    nhead=int(num_attention_heads),
                    dim_feedforward=int(mlp_hidden_dim),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.view_encoder_transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(num_encoder_layers),
                    norm=nn.LayerNorm(self.model_dim),
                )
        self.branch_queries = nn.Parameter(torch.randn(self.num_branches, self.model_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(mlp_hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.branch_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(num_decoder_layers),
            norm=nn.LayerNorm(self.model_dim),
        )
        # The main branch must be decoded before the side branches.  Keep the
        # historical ``branch_decoder`` name for the main decoder so existing
        # main-branch checkpoints retain their learned lookup weights.
        side_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(mlp_hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.side_branch_decoder = nn.TransformerDecoder(
            side_decoder_layer,
            num_layers=int(num_decoder_layers),
            norm=nn.LayerNorm(self.model_dim),
        )
        self.side_parent_projection = (
            nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
                nn.Linear(self.model_dim, self.model_dim),
            )
            if self.use_learned_side_parent_projection
            else nn.Identity()
        )
        self.attachment_head = nn.Sequential(
            nn.Linear(2 * self.model_dim + 5, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden_dim), 1),
        )
        self.side_attachment_condition = nn.Sequential(
            nn.Linear(2 * self.model_dim + 5, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

        self.use_centerline_point_index_embedding = bool(use_centerline_point_index_embedding)
        self.use_radius_point_index_embedding = bool(use_radius_point_index_embedding)
        self.centerline_point_embed_dim = int(centerline_point_embed_dim)
        if self.point_head_mode != "split_xyzr" and (
            self.use_centerline_point_index_embedding or self.use_radius_point_index_embedding
        ):
            raise ValueError("Point-index embeddings require point_head_mode='split_xyzr'.")

        if self.point_head_mode == "point":
            self.branch_point_head = nn.Sequential(
                nn.Linear(self.model_dim, int(mlp_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(mlp_hidden_dim), self.num_points * 4),
            )
            self.side_branch_point_head = copy.deepcopy(self.branch_point_head)
        else:
            if self.use_centerline_point_index_embedding:
                self.centerline_point_index_embed = nn.Embedding(self.num_points, self.centerline_point_embed_dim)
                self.side_centerline_point_index_embed = nn.Embedding(
                    self.num_points, self.centerline_point_embed_dim
                )
                self.branch_centerline_head = nn.Sequential(
                    nn.Linear(self.model_dim + self.centerline_point_embed_dim, int(mlp_hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(mlp_hidden_dim), 3),
                )
            else:
                self.branch_centerline_head = nn.Sequential(
                    nn.Linear(self.model_dim, int(mlp_hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(mlp_hidden_dim), self.num_points * 3),
                )
            self.side_branch_centerline_head = copy.deepcopy(self.branch_centerline_head)
            if self.use_radius_point_index_embedding:
                self.radius_point_index_embed = nn.Embedding(self.num_points, self.centerline_point_embed_dim)
                self.side_radius_point_index_embed = nn.Embedding(
                    self.num_points, self.centerline_point_embed_dim
                )
                self.branch_radius_head = nn.Sequential(
                    nn.Linear(self.model_dim + self.centerline_point_embed_dim, max(1, int(mlp_hidden_dim) // 2)),
                    nn.GELU(),
                    nn.Linear(max(1, int(mlp_hidden_dim) // 2), 1),
                )
            else:
                self.branch_radius_head = nn.Sequential(
                    nn.Linear(self.model_dim, max(1, int(mlp_hidden_dim) // 2)),
                    nn.GELU(),
                    nn.Linear(max(1, int(mlp_hidden_dim) // 2), self.num_points),
                )
            self.side_branch_radius_head = copy.deepcopy(self.branch_radius_head)
        self.branch_exist_head = nn.Sequential(
            nn.Linear(self.model_dim, max(1, int(mlp_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(1, int(mlp_hidden_dim) // 2), 1),
        )
        self.projection_evidence_coord_scale_to_meter = float(projection_evidence_coord_scale_to_meter)
        self.projection_evidence_residual_refiner = None
        if bool(use_projection_evidence_residual_refiner):
            if self.view_feat_dim != 4:
                raise ValueError("Projection evidence residual refiner requires view_feat_dim=4.")
            self.projection_evidence_residual_refiner = _ProjectionEvidenceResidualRefiner(
                model_dim=self.model_dim,
                view_feat_dim=self.view_feat_dim,
                num_attention_heads=int(num_attention_heads),
                evidence_hidden_dim=(
                    int(projection_evidence_hidden_dim)
                    if projection_evidence_hidden_dim is not None
                    else max(128, int(mlp_hidden_dim) // 2)
                ),
                patch_size=int(projection_evidence_patch_size),
                use_learned_image_features=bool(projection_evidence_use_learned_image_features),
                learned_feature_dim=int(projection_evidence_learned_feature_dim),
                use_distance_transform=bool(projection_evidence_use_distance_transform),
                distance_transform_num_iters=int(projection_evidence_distance_transform_num_iters),
                dropout=float(dropout),
                image_size=int(projection_evidence_image_size),
                sid=float(projection_evidence_sid),
                source_to_iso=float(projection_evidence_source_to_iso),
                imager_pixel_spacing=float(projection_evidence_imager_pixel_spacing),
                residual_scale=float(projection_evidence_residual_scale),
            )

    def _pointwise_head_input(self, decoded: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        batch_size, num_branches, model_dim = decoded.shape
        point_ids = torch.arange(self.num_points, device=decoded.device)
        point_embed = embedding(point_ids).view(1, 1, self.num_points, -1).expand(
            batch_size,
            num_branches,
            self.num_points,
            -1,
        )
        decoded = decoded.view(batch_size, num_branches, 1, model_dim).expand(
            batch_size,
            num_branches,
            self.num_points,
            model_dim,
        )
        return torch.cat([decoded, point_embed], dim=-1)

    def _decode_branch_points(
        self,
        decoded: torch.Tensor,
        *,
        side_branch: bool = False,
    ) -> torch.Tensor:
        batch_size = decoded.shape[0]
        num_branches = decoded.shape[1]
        if self.point_head_mode == "point":
            head = self.side_branch_point_head if side_branch else self.branch_point_head
            return head(decoded).view(batch_size, num_branches, self.num_points, 4)
        centerline_head = (
            self.side_branch_centerline_head
            if side_branch
            else self.branch_centerline_head
        )
        radius_head = (
            self.side_branch_radius_head if side_branch else self.branch_radius_head
        )
        if self.use_centerline_point_index_embedding:
            centerline_embedding = (
                self.side_centerline_point_index_embed
                if side_branch
                else self.centerline_point_index_embed
            )
            xyz = centerline_head(
                self._pointwise_head_input(decoded, centerline_embedding)
            )
        else:
            xyz = centerline_head(decoded).view(
                batch_size, num_branches, self.num_points, 3
            )
        if self.use_radius_point_index_embedding:
            radius_embedding = (
                self.side_radius_point_index_embed
                if side_branch
                else self.radius_point_index_embed
            )
            radius = radius_head(
                self._pointwise_head_input(decoded, radius_embedding)
            )
        else:
            radius = radius_head(decoded).view(
                batch_size, num_branches, self.num_points, 1
            )
        return torch.cat([xyz, radius], dim=-1)

    def _encode_view_memory(
        self,
        *,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        image_features: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens, camera_geometry, layout = self._build_tokens_and_camera_context(
            views=views,
            image_features=image_features,
        )
        padding_mask = None
        if view_mask is not None:
            if view_mask.dim() != 2:
                raise ValueError(
                    f"view_mask must have shape [B,V], got {tuple(view_mask.shape)}"
                )
            padding_mask = self._expand_view_padding_mask(view_mask, tokens.shape[1])
        if self.prope_encoder is not None:
            if camera_geometry is None or layout is None:
                raise RuntimeError("PRoPE encoder is missing its camera context.")
            batch_size, num_views = views.shape[:2]
            num_locations = layout.rope_xy.shape[0]
            token_view_indices = torch.arange(
                num_views, device=tokens.device, dtype=torch.long
            ).repeat_interleave(num_locations)
            rope_xy = layout.rope_xy.to(device=tokens.device, dtype=tokens.dtype).repeat(
                num_views, 1
            )
            memory = self.prope_encoder(
                tokens,
                world_to_camera=camera_geometry.world_to_camera,
                intrinsics=camera_geometry.intrinsics,
                rope_xy=rope_xy,
                token_view_indices=token_view_indices,
                padding_mask=padding_mask,
                image_height=self.camera_encoding_image_height,
                image_width=self.camera_encoding_image_width,
            )
        elif self.view_encoder_transformer is None:
            memory = tokens
        elif padding_mask is not None:
            memory = self.view_encoder_transformer(
                tokens, src_key_padding_mask=padding_mask
            )
        else:
            memory = self.view_encoder_transformer(tokens)
        return memory, padding_mask

    def _decode_main_and_side_tokens(
        self,
        memory: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the parent token first, then use it to query side evidence."""
        batch_size = memory.shape[0]
        main_query = self.branch_queries[:1].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        main_token = self.branch_decoder(
            tgt=main_query,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        if self.num_branches == 1:
            return main_token, main_token.new_empty((batch_size, 0, self.model_dim))

        side_queries = self.branch_queries[1:].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        parent_token = main_token[:, 0]
        if (
            not self.use_learned_side_parent_projection
            and parent_token.shape[-1] != side_queries.shape[-1]
        ):
            raise RuntimeError(
                "Direct side-query conditioning requires the main and side token "
                f"dimensions to match, got {parent_token.shape[-1]} and "
                f"{side_queries.shape[-1]}."
            )
        parent_update = self.side_parent_projection(parent_token).unsqueeze(1)
        initial_side_tokens = self.side_branch_decoder(
            tgt=side_queries + parent_update,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        return main_token, initial_side_tokens

    def _condition_side_tokens(
        self,
        *,
        main_token: torch.Tensor,
        initial_side_tokens: torch.Tensor,
        main_branch_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict parent attachments and add the local parent descriptor."""
        if main_branch_points.dim() != 4 or main_branch_points.shape[1:] != (
            1,
            self.num_points,
            4,
        ):
            raise ValueError(
                "main_branch_points must have shape "
                f"[B,1,{self.num_points},4], got {tuple(main_branch_points.shape)}"
            )
        batch_size, num_side_branches = initial_side_tokens.shape[:2]
        if num_side_branches == 0:
            empty_logits = main_branch_points.new_empty(
                (batch_size, 0, self.num_points)
            )
            return {
                "conditioned_side_tokens": initial_side_tokens,
                "attachment_logits": empty_logits,
                "attachment_probabilities": empty_logits,
                "attachment_indices": torch.empty(
                    (batch_size, 0),
                    device=main_branch_points.device,
                    dtype=torch.long,
                ),
                "attachment_coordinates": main_branch_points.new_empty(
                    (batch_size, 0, 3)
                ),
            }

        main_code = main_branch_points[:, 0, :, :4]
        position = torch.linspace(
            0.0,
            1.0,
            steps=self.num_points,
            device=main_code.device,
            dtype=main_code.dtype,
        ).view(1, self.num_points, 1)
        point_descriptors = torch.cat(
            [main_code, position.expand(batch_size, -1, -1)], dim=-1
        )
        side_at_points = initial_side_tokens.unsqueeze(2).expand(
            -1, -1, self.num_points, -1
        )
        main_at_points = main_token.expand(-1, num_side_branches, -1).unsqueeze(
            2
        ).expand(-1, -1, self.num_points, -1)
        descriptor_at_sides = point_descriptors.unsqueeze(1).expand(
            -1, num_side_branches, -1, -1
        )
        attachment_logits = self.attachment_head(
            torch.cat(
                [side_at_points, main_at_points, descriptor_at_sides], dim=-1
            )
        ).squeeze(-1)
        attachment_probabilities = torch.softmax(attachment_logits, dim=-1)
        attachment_indices = attachment_probabilities.argmax(dim=-1)
        soft_attachment_coordinates = torch.einsum(
            "bsn,bnd->bsd",
            attachment_probabilities,
            main_code[..., :3],
        )
        soft_local_descriptors = torch.einsum(
            "bsn,bnd->bsd",
            attachment_probabilities,
            point_descriptors,
        )
        main_for_sides = main_token.expand(-1, num_side_branches, -1)
        conditioned_side_tokens = initial_side_tokens + self.side_attachment_condition(
            torch.cat(
                [
                    initial_side_tokens,
                    main_for_sides,
                    soft_local_descriptors,
                ],
                dim=-1,
            )
        )
        return {
            "conditioned_side_tokens": conditioned_side_tokens,
            "attachment_logits": attachment_logits,
            "attachment_probabilities": attachment_probabilities,
            "attachment_indices": attachment_indices,
            "attachment_coordinates": soft_attachment_coordinates,
        }

    def _assemble_side_points(
        self,
        *,
        side_relative_points: torch.Tensor,
        main_branch_points: torch.Tensor,
        attachment_probabilities: torch.Tensor,
        code_mean: torch.Tensor | None,
        code_std: torch.Tensor | None,
    ) -> torch.Tensor:
        """Translate normalized relative side geometry to its soft parent attachment."""
        side_relative_points = side_relative_points.clone()
        side_relative_points[..., :3] = (
            side_relative_points[..., :3]
            - side_relative_points[..., :1, :3]
        )
        if code_mean is None and code_std is None:
            origins = torch.einsum(
                "bsn,bnd->bsd",
                attachment_probabilities,
                main_branch_points[:, 0, :, :3],
            )
            side_relative_points[..., :3] = (
                side_relative_points[..., :3] + origins.unsqueeze(2)
            )
            return side_relative_points
        if code_mean is None or code_std is None:
            raise ValueError("code_mean and code_std must either both be provided or both be None.")
        mean_shape_ok = (
            code_mean.dim() == 4
            and code_mean.shape[-1] == 4
            and code_mean.shape[-2] in (1, self.num_points)
        )
        std_shape_ok = (
            code_std.dim() == 4
            and code_std.shape[-1] == 4
            and code_std.shape[-2] in (1, self.num_points)
        )
        if not mean_shape_ok or not std_shape_ok:
            raise ValueError(
                "Hierarchical side-branch assembly requires code_mean/code_std "
                f"ending in [1 or {self.num_points},4], got "
                f"{tuple(code_mean.shape)} and {tuple(code_std.shape)}"
            )

        mean_xyz = code_mean[..., :3].to(side_relative_points)
        std_xyz = code_std[..., :3].to(side_relative_points)
        main_world_xyz = (
            main_branch_points[:, 0, :, :3] * std_xyz[:, 0]
            + mean_xyz[:, 0]
        )
        origins_world = torch.einsum(
            "bsn,bnd->bsd", attachment_probabilities, main_world_xyz
        )
        side_world_xyz = (
            origins_world.unsqueeze(2)
            + side_relative_points[..., :3] * std_xyz
        )
        side_relative_points[..., :3] = (
            side_world_xyz - mean_xyz
        ) / std_xyz.clamp_min(1e-6)
        return side_relative_points

    def _image_tokens(self, image_features: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if self.feature_backbone == "resnet_pre_fpn":
            if not isinstance(image_features, dict):
                raise ValueError("resnet_pre_fpn expects image_features to be a dict containing c2/c3/c4/c5.")
            assert self.resnet_pre_fpn is not None
            return self.resnet_pre_fpn(image_features)
        if isinstance(image_features, dict):
            raise ValueError(f"{self.feature_backbone} expects image_features to be a tensor, not a dict.")
        if image_features.dim() != 4:
            raise ValueError(f"image_features must have shape [B,V,L,D], got {tuple(image_features.shape)}")
        return self.image_token_projection(image_features)

    def _camera_context(
        self,
        *,
        views: torch.Tensor,
        num_locations: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[CArmCameraGeometry, CameraTokenLayout]:
        geometry = build_carm_camera_geometry(
            views,
            image_height=self.camera_encoding_image_height,
            image_width=self.camera_encoding_image_width,
            sid=self.camera_encoding_sid,
            source_to_iso=self.camera_encoding_source_to_iso,
            imager_pixel_spacing=self.camera_encoding_imager_pixel_spacing,
        )
        layout = build_camera_token_layout(
            feature_backbone=self.feature_backbone,
            num_locations=int(num_locations),
            image_height=self.camera_encoding_image_height,
            image_width=self.camera_encoding_image_width,
            resnet_pool_size=(
                0 if self.resnet_pre_fpn is None else self.resnet_pre_fpn.pool_size
            ),
            vggt_patch_size=self.vggt_patch_size,
            vggt_target_image_height=self.vggt_target_image_height,
            vggt_target_image_width=self.vggt_target_image_width,
            device=device,
            dtype=dtype,
        )
        return geometry, layout

    def _build_tokens_and_camera_context(
        self,
        views: torch.Tensor,
        image_features: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        CArmCameraGeometry | None,
        CameraTokenLayout | None,
    ]:
        image_tokens = self._image_tokens(image_features)
        batch_size, num_views, num_locations, _ = image_tokens.shape
        if views.shape[:2] != (batch_size, num_views):
            raise ValueError(f"views must have shape [B,V,F], got {tuple(views.shape)}")

        geometry = None
        layout = None
        if self.use_ray_camera_encoding or self.use_prope_attention:
            geometry, layout = self._camera_context(
                views=views,
                num_locations=num_locations,
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )

        conditioning = []
        if self.view_encoder is not None:
            view_feat = self.view_encoder(
                views.reshape(batch_size * num_views, -1)
            ).reshape(batch_size, num_views, -1)
            conditioning.append(
                view_feat.unsqueeze(2).expand(-1, -1, num_locations, -1)
            )
        if self.ray_encoder is not None:
            if geometry is None or layout is None:
                raise RuntimeError("Ray encoder is missing its camera context.")
            rays = plucker_ray_embeddings(geometry, layout).to(dtype=image_tokens.dtype)
            encoded_rays = self.ray_encoder(rays.reshape(-1, 6)).reshape(
                batch_size, num_views, num_locations, -1
            )
            encoded_rays = encoded_rays * layout.spatial_mask.to(
                device=encoded_rays.device, dtype=encoded_rays.dtype
            ).view(1, 1, num_locations, 1)
            conditioning.append(
                encoded_rays
            )
        tokens = (
            self.token_projection(torch.cat((image_tokens, *conditioning), dim=-1))
            if conditioning
            else image_tokens
        )
        return (
            tokens.reshape(batch_size, num_views * num_locations, self.model_dim),
            geometry,
            layout,
        )

    def _build_tokens(
        self,
        views: torch.Tensor,
        image_features: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        tokens, _, _ = self._build_tokens_and_camera_context(
            views=views,
            image_features=image_features,
        )
        return tokens

    def _expand_view_padding_mask(self, view_mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
        batch_size, num_views = view_mask.shape
        if num_tokens % num_views != 0:
            raise ValueError(f"Cannot expand view mask with {num_views} views over {num_tokens} tokens")
        num_locations = num_tokens // num_views
        return ~view_mask.to(dtype=torch.bool).unsqueeze(-1).expand(
            batch_size,
            num_views,
            num_locations,
        ).reshape(batch_size, num_tokens)

    def forward(
        self,
        images: torch.Tensor | None = None,
        views: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        image_features: torch.Tensor | dict[str, torch.Tensor] | None = None,
        code_mean: torch.Tensor | None = None,
        code_std: torch.Tensor | None = None,
        coord_scale_to_meter: float | None = None,
        return_coarse: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if views is None:
            raise ValueError("views must be provided.")
        if image_features is None:
            raise ValueError("This model requires precomputed image_features.")
        memory, padding_mask = self._encode_view_memory(
            views=views,
            view_mask=view_mask,
            image_features=image_features,
        )
        main_token, initial_side_tokens = self._decode_main_and_side_tokens(
            memory, padding_mask
        )
        main_branch_points = self._decode_branch_points(main_token)
        coarse_main_branch_points = main_branch_points
        residual = None
        if self.projection_evidence_residual_refiner is not None:
            if images is None:
                raise ValueError("Projection evidence residual refiner requires original images.")
            main_branch_points, residual = self.projection_evidence_residual_refiner(
                coarse_branch_points=coarse_main_branch_points,
                pooled_token=main_token[:, 0, :],
                images=images,
                views=views,
                view_mask=view_mask,
                code_mean=code_mean,
                code_std=code_std,
                coord_scale_to_meter=(
                    self.projection_evidence_coord_scale_to_meter
                    if coord_scale_to_meter is None
                    else float(coord_scale_to_meter)
                ),
            )

        side_state = self._condition_side_tokens(
            main_token=main_token,
            initial_side_tokens=initial_side_tokens,
            main_branch_points=main_branch_points,
        )
        conditioned_side_tokens = side_state["conditioned_side_tokens"]
        if self.num_branches > 1:
            side_relative_points = self._decode_branch_points(
                conditioned_side_tokens,
                side_branch=True,
            )
            side_branch_points = self._assemble_side_points(
                side_relative_points=side_relative_points,
                main_branch_points=main_branch_points,
                attachment_probabilities=side_state[
                    "attachment_probabilities"
                ],
                code_mean=code_mean,
                code_std=code_std,
            )
            branch_points = torch.cat(
                [main_branch_points, side_branch_points], dim=1
            )
            side_exist_logits = self.branch_exist_head(
                conditioned_side_tokens
            ).squeeze(-1)
        else:
            side_relative_points = main_branch_points.new_empty(
                (main_branch_points.shape[0], 0, self.num_points, 4)
            )
            branch_points = main_branch_points
            side_exist_logits = main_branch_points.new_empty(
                (main_branch_points.shape[0], 0)
            )
        main_exist_logits = main_branch_points.new_full(
            (main_branch_points.shape[0], 1), 20.0
        )
        branch_exist_logits = torch.cat(
            [main_exist_logits, side_exist_logits], dim=1
        )
        branch_exist_probs = torch.cat(
            [
                torch.ones_like(main_exist_logits),
                torch.sigmoid(side_exist_logits),
            ],
            dim=1,
        )
        main_attachment_logits = main_branch_points.new_zeros(
            (main_branch_points.shape[0], 1, self.num_points)
        )
        main_attachment_probabilities = main_branch_points.new_zeros(
            (main_branch_points.shape[0], 1, self.num_points)
        )
        main_attachment_probabilities[..., 0] = 1.0
        main_attachment_indices = torch.zeros(
            (main_branch_points.shape[0], 1),
            device=main_branch_points.device,
            dtype=torch.long,
        )
        side_offsets_full = torch.cat(
            [
                main_branch_points.new_zeros(
                    (main_branch_points.shape[0], 1, self.num_points, 3)
                ),
                (
                    side_relative_points[..., :3]
                    - side_relative_points[..., :1, :3]
                ),
            ],
            dim=1,
        )
        out = {
            "branch_points": branch_points,
            "branch_exist_logits": branch_exist_logits,
            "branch_exist_probs": branch_exist_probs,
            "attachment_logits": torch.cat(
                [main_attachment_logits, side_state["attachment_logits"]], dim=1
            ),
            "attachment_probabilities": torch.cat(
                [
                    main_attachment_probabilities,
                    side_state["attachment_probabilities"],
                ],
                dim=1,
            ),
            "attachment_indices": torch.cat(
                [main_attachment_indices, side_state["attachment_indices"]], dim=1
            ),
            "side_branch_offsets": side_offsets_full,
            "side_branch_relative_code": torch.cat(
                [
                    (
                        side_relative_points[..., :3]
                        - side_relative_points[..., :1, :3]
                    ),
                    side_relative_points[..., 3:4],
                ],
                dim=-1,
            ),
        }
        if residual is not None:
            out["projection_evidence_residual"] = residual
        if bool(return_coarse) and self.projection_evidence_residual_refiner is not None:
            coarse_branch_points = branch_points.clone()
            coarse_branch_points[:, :1] = coarse_main_branch_points
            out["coarse_branch_points"] = coarse_branch_points
            out["coarse_main_branch_points"] = coarse_main_branch_points
        return out
