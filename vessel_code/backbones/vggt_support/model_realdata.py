# Transferred from methods/multiview_model/model_realdata.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(trainable)

class VGGTViewTokenVesselPredictor(nn.Module):
    """
    VGGT-token multi-view vessel-code predictor.

    The model keeps the public forward contract used by train_realdata.py:
    - branch_points: [B, M, N, 4] in normalized vessel-code space
    - branch_exist_logits/probs: [B, M]

    VGGT itself is an optional dependency. It is loaded lazily through either
    VGGT.from_pretrained(...) or torch.hub depending on vggt_load_mode.
    """

    def __init__(
        self,
        num_points: int = 200,
        num_branches: int = 1,
        view_feat_dim: int = 4,
        model_dim: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 8,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
        point_head_mode: str = "split_xyzr",
        prediction_code_type: str = "vessel",
        use_encoded_view_dir: bool = True,
        vggt_backbone: str = "omega",
        vggt_token_dim: int = 2048,
        vggt_pretrained: bool = True,
        vggt_finetune: bool = False,
        vggt_model_name: str = "facebook/VGGT-1B",
        vggt_load_mode: str = "package",
        vggt_torchhub_repo: str = "facebookresearch/vggt",
        vggt_omega_checkpoint_path: str | None = None,
        vggt_image_size_mode: str = "resize_to_patch_multiple",
        vggt_patch_size: int = 16,
        vggt_target_image_size: int | None = 256,
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
        self.num_branches = int(num_branches)
        self.num_points = int(num_points)
        if self.num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {self.num_branches}")
        if self.num_points < 2:
            raise ValueError(f"num_points must be >= 2, got {self.num_points}")
        self.prediction_code_type = str(prediction_code_type).strip().lower()
        if self.prediction_code_type != "vessel":
            raise ValueError("VGGTViewTokenVesselPredictor supports only prediction_code_type='vessel'.")
        self.point_head_mode = str(point_head_mode).strip().lower()
        if self.point_head_mode not in ("point", "split_xyzr"):
            raise ValueError(f"point_head_mode must be one of ['point', 'split_xyzr'], got {point_head_mode}")

        self.model_dim = int(model_dim)
        self.view_feat_dim = int(view_feat_dim)
        self.use_encoded_view_dir = bool(use_encoded_view_dir)
        self.vggt_backbone = str(vggt_backbone).strip().lower()
        if self.vggt_backbone not in ("omega", "vggt_omega", "original", "vggt"):
            raise ValueError("vggt_backbone must be one of ['omega', 'original'].")
        self.vggt_token_dim = int(vggt_token_dim)
        self.vggt_pretrained = bool(vggt_pretrained)
        self.vggt_finetune = bool(vggt_finetune)
        self.vggt_model_name = str(vggt_model_name)
        self.vggt_load_mode = str(vggt_load_mode).strip().lower()
        self.vggt_torchhub_repo = str(vggt_torchhub_repo)
        self.vggt_omega_checkpoint_path = None if vggt_omega_checkpoint_path is None else str(vggt_omega_checkpoint_path)
        self.vggt_image_size_mode = str(vggt_image_size_mode).strip().lower()
        if self.vggt_image_size_mode not in ("pad_to_patch_multiple", "resize_to_patch_multiple", "error"):
            raise ValueError(
                "vggt_image_size_mode must be one of "
                "['pad_to_patch_multiple', 'resize_to_patch_multiple', 'error']."
            )
        self.vggt_patch_size = int(vggt_patch_size)
        if self.vggt_patch_size < 1:
            raise ValueError(f"vggt_patch_size must be >= 1, got {self.vggt_patch_size}")
        self.vggt_target_image_size = None if vggt_target_image_size is None else int(vggt_target_image_size)
        if self.vggt_target_image_size is not None and self.vggt_target_image_size < 1:
            raise ValueError(f"vggt_target_image_size must be >= 1, got {self.vggt_target_image_size}")
        if (
            self.vggt_target_image_size is not None
            and self.vggt_target_image_size % self.vggt_patch_size != 0
        ):
            raise ValueError(
                f"vggt_target_image_size={self.vggt_target_image_size} must be divisible by "
                f"vggt_patch_size={self.vggt_patch_size}."
            )
        self.vggt: nn.Module | None = None

        self.token_projection = nn.Sequential(
            nn.Linear(self.vggt_token_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        if self.use_encoded_view_dir:
            self.view_encoder = nn.Sequential(
                nn.Linear(self.view_feat_dim, max(128, self.model_dim // 4)),
                nn.GELU(),
                nn.Linear(max(128, self.model_dim // 4), self.model_dim),
                nn.LayerNorm(self.model_dim),
            )
        else:
            self.view_encoder = None

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
        else:
            if self.use_centerline_point_index_embedding:
                self.centerline_point_index_embed = nn.Embedding(self.num_points, self.centerline_point_embed_dim)
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
            if self.use_radius_point_index_embedding:
                self.radius_point_index_embed = nn.Embedding(self.num_points, self.centerline_point_embed_dim)
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
        self.branch_exist_head = nn.Sequential(
            nn.Linear(self.model_dim, max(1, int(mlp_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(1, int(mlp_hidden_dim) // 2), 1),
        )
        self.use_projection_evidence_residual_refiner = bool(use_projection_evidence_residual_refiner)
        self.projection_evidence_coord_scale_to_meter = float(projection_evidence_coord_scale_to_meter)
        if self.use_projection_evidence_residual_refiner and self.num_branches != 1:
            raise ValueError("Projection evidence residual refinement currently supports only num_branches=1.")
        if self.use_projection_evidence_residual_refiner:
            self.projection_evidence_residual_refiner = _ProjectionEvidenceResidualRefiner(
                model_dim=self.model_dim,
                view_feat_dim=int(view_feat_dim),
                num_attention_heads=int(num_attention_heads),
                evidence_hidden_dim=(
                    int(projection_evidence_hidden_dim)
                    if projection_evidence_hidden_dim is not None
                    else int(mlp_hidden_dim) // 2
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
        else:
            self.projection_evidence_residual_refiner = None

    def _load_vggt(self) -> nn.Module:
        if self.vggt is not None:
            return self.vggt
        if self.vggt_backbone in ("omega", "vggt_omega"):
            try:
                from vggt_omega.models import VGGTOmega  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "VGGT-Omega is not installed. Install facebookresearch/vggt-omega "
                    "or set vggt_backbone='original' to use the original VGGT package."
                ) from exc
            model = VGGTOmega(patch_size=self.vggt_patch_size)
            if self.vggt_pretrained:
                if not self.vggt_omega_checkpoint_path:
                    raise ValueError(
                        "vggt_backbone='omega' with vggt_pretrained=True requires "
                        "vggt_omega_checkpoint_path pointing to the downloaded VGGT-Omega checkpoint."
                    )
                state = torch.load(self.vggt_omega_checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                    state = state["model"]
                model.load_state_dict(state)
        elif self.vggt_load_mode in ("package", "python_package"):
            try:
                from vggt.models.vggt import VGGT  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "VGGT is not installed. Install the VGGT package, or set "
                    "vggt_load_mode='torchhub' with network/cache access."
                ) from exc
            if self.vggt_pretrained and hasattr(VGGT, "from_pretrained"):
                model = VGGT.from_pretrained(self.vggt_model_name)
            else:
                model = VGGT()
        elif self.vggt_load_mode in ("torchhub", "hub"):
            model = torch.hub.load(
                self.vggt_torchhub_repo,
                "VGGT",
                pretrained=self.vggt_pretrained,
            )
        else:
            raise ValueError(
                f"Unsupported vggt_load_mode={self.vggt_load_mode!r}; use 'package' or 'torchhub'."
            )
        _set_module_trainable(model, self.vggt_finetune)
        # The backbone is loaded lazily, potentially after the parent predictor
        # has already been put in evaluation mode.  A newly constructed module
        # otherwise defaults to training mode even when every weight is frozen.
        model.train(bool(self.training) and self.vggt_finetune)
        try:
            ref_param = next(self.token_projection.parameters())
            model = model.to(device=ref_param.device)
        except StopIteration:
            pass
        self.vggt = model
        return model

    @staticmethod
    def _prepare_vggt_images(
        images: torch.Tensor,
        image_size_mode: str = "resize_to_patch_multiple",
        patch_size: int = 14,
        target_image_size: int | None = None,
    ) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(2)
        if images.dim() != 5:
            raise ValueError(f"images must have shape [B,V,H,W] or [B,V,C,H,W], got {tuple(images.shape)}")
        if images.shape[2] == 1:
            images = images.expand(-1, -1, 3, -1, -1)
        elif images.shape[2] != 3:
            raise ValueError(f"VGGT expects 1 or 3 image channels, got {images.shape[2]}")
        images = images.clamp(0.0, 1.0)
        mode = str(image_size_mode).strip().lower()
        patch = int(patch_size)
        if patch < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if target_image_size is not None:
            target = int(target_image_size)
            if target < 1:
                raise ValueError(f"target_image_size must be >= 1, got {target}")
            if target % patch != 0:
                raise ValueError(f"target_image_size={target} must be divisible by patch_size={patch}")
            target_h = target
            target_w = target
        else:
            target_h = ((height + patch - 1) // patch) * patch
            target_w = ((width + patch - 1) // patch) * patch
        if (height, width) == (target_h, target_w):
            return images
        if mode == "error":
            raise ValueError(
                f"VGGT input image size {(height, width)} is not divisible by patch size {patch}; "
                f"use vggt_image_size_mode='pad_to_patch_multiple' or 'resize_to_patch_multiple'."
            )
        if mode == "pad_to_patch_multiple":
            return F.pad(images, (0, target_w - width, 0, target_h - height), mode="constant", value=0.0)
        if mode == "resize_to_patch_multiple":
            batch_size, num_views, channels = images.shape[:3]
            flat = images.reshape(batch_size * num_views, channels, height, width)
            resized = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return resized.reshape(batch_size, num_views, channels, target_h, target_w)
        raise ValueError(
            "image_size_mode must be one of ['pad_to_patch_multiple', 'resize_to_patch_multiple', 'error']."
        )

    @staticmethod
    def _first_tensor(value) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for key in ("view_tokens", "tokens", "aggregated_tokens", "image_tokens", "patch_tokens"):
                found = VGGTViewTokenVesselPredictor._first_tensor(value.get(key))
                if found is not None:
                    return found
            for nested in value.values():
                found = VGGTViewTokenVesselPredictor._first_tensor(nested)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for nested in reversed(value):
                found = VGGTViewTokenVesselPredictor._first_tensor(nested)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _last_tensor(value) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            for nested in reversed(value):
                found = VGGTViewTokenVesselPredictor._last_tensor(nested)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for nested in reversed(list(value.values())):
                found = VGGTViewTokenVesselPredictor._last_tensor(nested)
                if found is not None:
                    return found
        return None

    def _extract_vggt_tokens(self, images: torch.Tensor) -> torch.Tensor:
        model = self._load_vggt()
        ctx = torch.enable_grad() if self.vggt_finetune else torch.no_grad()
        with ctx:
            if self.vggt_backbone in ("omega", "vggt_omega") and hasattr(model, "aggregator"):
                raw = model.aggregator(images)
                if not (isinstance(raw, (list, tuple)) and len(raw) >= 1):
                    raise RuntimeError("Unsupported VGGT-Omega aggregator output.")
                tokens = self._last_tensor(raw[0])
            elif hasattr(model, "aggregator"):
                raw = model.aggregator(images)
                if isinstance(raw, (list, tuple)) and len(raw) >= 1:
                    tokens = self._last_tensor(raw[0])
                else:
                    tokens = self._first_tensor(raw)
            else:
                raw = model(images)
                tokens = self._first_tensor(raw)
        if tokens is None:
            raise RuntimeError("Could not find view tokens in VGGT output.")
        if tokens.dim() == 4:
            return tokens
        if tokens.dim() == 3:
            batch_size, num_views = images.shape[:2]
            if tokens.shape[0] == batch_size and tokens.shape[1] % num_views == 0:
                return tokens.reshape(batch_size, num_views, tokens.shape[1] // num_views, tokens.shape[2])
            if tokens.shape[0] == batch_size * num_views:
                return tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])
        raise RuntimeError(f"Unsupported VGGT token shape {tuple(tokens.shape)}")

    def _pointwise_head_input(self, decoded: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        batch_size, num_branches, model_dim = decoded.shape
        point_ids = torch.arange(self.num_points, device=decoded.device)
        point_embed = embedding(point_ids).view(1, 1, self.num_points, -1).expand(
            batch_size, num_branches, self.num_points, -1
        )
        decoded = decoded.view(batch_size, num_branches, 1, model_dim).expand(
            batch_size, num_branches, self.num_points, model_dim
        )
        return torch.cat([decoded, point_embed], dim=-1)

    def _decode_branch_points(self, decoded: torch.Tensor) -> torch.Tensor:
        batch_size = decoded.shape[0]
        if self.point_head_mode == "point":
            return self.branch_point_head(decoded).view(batch_size, self.num_branches, self.num_points, 4)
        if self.use_centerline_point_index_embedding:
            xyz = self.branch_centerline_head(self._pointwise_head_input(decoded, self.centerline_point_index_embed))
        else:
            xyz = self.branch_centerline_head(decoded).view(batch_size, self.num_branches, self.num_points, 3)
        if self.use_radius_point_index_embedding:
            radius = self.branch_radius_head(self._pointwise_head_input(decoded, self.radius_point_index_embed))
        else:
            radius = self.branch_radius_head(decoded).view(batch_size, self.num_branches, self.num_points, 1)
        return torch.cat([xyz, radius], dim=-1)

    def _expand_view_padding_mask(self, view_mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
        batch_size, num_views = view_mask.shape
        if num_tokens % num_views != 0:
            raise ValueError(f"Cannot expand view mask with {num_views} views over {num_tokens} tokens")
        num_locations = num_tokens // num_views
        return ~view_mask.to(dtype=torch.bool).unsqueeze(-1).expand(batch_size, num_views, num_locations).reshape(
            batch_size,
            num_tokens,
        )

    def forward(
        self,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        code_mean: torch.Tensor | None = None,
        code_std: torch.Tensor | None = None,
        coord_scale_to_meter: float | None = None,
        return_coarse: bool = False,
        image_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if image_features is None:
            images_5d = self._prepare_vggt_images(
                images,
                image_size_mode=self.vggt_image_size_mode,
                patch_size=self.vggt_patch_size,
                target_image_size=self.vggt_target_image_size,
            )
            vggt_tokens = self._extract_vggt_tokens(images_5d)
        else:
            vggt_tokens = image_features.to(dtype=images.dtype)
        batch_size, num_views, num_locations, token_dim = vggt_tokens.shape
        if int(token_dim) != self.vggt_token_dim:
            raise RuntimeError(
                f"VGGT token dim is {token_dim}, but vggt_token_dim={self.vggt_token_dim}. "
                "Update the config to match the installed VGGT backbone."
            )
        tokens = self.token_projection(vggt_tokens)
        if self.view_encoder is not None:
            if views.shape[:2] != (batch_size, num_views):
                raise ValueError(f"views must have shape [B,V,F] matching images, got {tuple(views.shape)}")
            view_tokens = self.view_encoder(views.reshape(batch_size * num_views, -1)).reshape(
                batch_size,
                num_views,
                1,
                self.model_dim,
            )
            tokens = tokens + view_tokens
        tokens = tokens.reshape(batch_size, num_views * num_locations, self.model_dim)
        if view_mask is not None:
            padding_mask = self._expand_view_padding_mask(view_mask, tokens.shape[1])
            memory = self.view_encoder_transformer(tokens, src_key_padding_mask=padding_mask)
        else:
            padding_mask = None
            memory = self.view_encoder_transformer(tokens)

        branch_queries = self.branch_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.branch_decoder(
            tgt=branch_queries,
            memory=memory,
            memory_key_padding_mask=padding_mask,
        )
        branch_points = self._decode_branch_points(decoded)
        coarse_branch_points = branch_points
        residual = None
        if self.projection_evidence_residual_refiner is not None:
            branch_points, residual = self.projection_evidence_residual_refiner(
                coarse_branch_points=coarse_branch_points,
                pooled_token=decoded[:, 0, :],
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
        branch_exist_logits = self.branch_exist_head(decoded).squeeze(-1)
        if self.num_branches == 1:
            branch_exist_probs = torch.ones_like(branch_exist_logits)
        else:
            branch_exist_probs = torch.sigmoid(branch_exist_logits)
            branch_exist_probs = torch.cat([torch.ones_like(branch_exist_probs[:, :1]), branch_exist_probs[:, 1:]], dim=1)
        out = {
            "branch_points": branch_points,
            "branch_exist_logits": branch_exist_logits,
            "branch_exist_probs": branch_exist_probs,
        }
        if residual is not None:
            out["projection_evidence_residual"] = residual
        if bool(return_coarse) and self.projection_evidence_residual_refiner is not None:
            out["coarse_branch_points"] = coarse_branch_points
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vggt is not None:
            self.vggt.train(bool(mode) and self.vggt_finetune)
        return self

class _ProjectionEvidenceRefiner(nn.Module):
    def __init__(
        self,
        model_dim: int,
        view_feat_dim: int,
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
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.view_feat_dim = int(view_feat_dim)
        self.patch_size = int(patch_size)
        if self.patch_size < 1:
            raise ValueError(f"projection_evidence_patch_size must be >= 1, got {patch_size}")
        if self.patch_size % 2 == 0:
            raise ValueError(f"projection_evidence_patch_size must be odd, got {patch_size}")
        if self.view_feat_dim != 4:
            raise ValueError(
                "Projection evidence refinement currently requires trig view features "
                "[sin(theta), cos(theta), sin(phi), cos(phi)] with view_feat_dim=4."
            )

        self.use_learned_image_features = bool(use_learned_image_features)
        self.learned_feature_dim = int(learned_feature_dim)
        self.use_distance_transform = bool(use_distance_transform)
        self.distance_transform_num_iters = int(distance_transform_num_iters)
        if self.use_distance_transform and self.distance_transform_num_iters < 1:
            raise ValueError(
                "projection_evidence_distance_transform_num_iters must be >= 1, "
                f"got {distance_transform_num_iters}"
            )
        if self.use_learned_image_features and self.learned_feature_dim < 1:
            raise ValueError(
                f"projection_evidence_learned_feature_dim must be >= 1, got {learned_feature_dim}"
            )

        if self.use_learned_image_features:
            self.image_feature_encoder = nn.Sequential(
                nn.Conv2d(1, self.learned_feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.learned_feature_dim, self.learned_feature_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.image_feature_encoder = None

        raw_dim = self.patch_size * self.patch_size
        evidence_in_dim = raw_dim + self.view_feat_dim
        if self.use_distance_transform:
            evidence_in_dim += 1
        if self.use_learned_image_features:
            evidence_in_dim += self.learned_feature_dim

        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_in_dim, int(evidence_hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(evidence_hidden_dim)),
            nn.Linear(int(evidence_hidden_dim), self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=int(num_attention_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(self.model_dim)
        self.norm_out = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, int(evidence_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(evidence_hidden_dim), self.model_dim),
        )

        self.image_size = int(image_size)
        self.sid = float(sid)
        self.source_to_iso = float(source_to_iso)
        self.imager_pixel_spacing = float(imager_pixel_spacing)
        circle_angles = torch.linspace(0.0, 2.0 * torch.pi, steps=120)
        self.register_buffer("circle_cos", torch.cos(circle_angles), persistent=False)
        self.register_buffer("circle_sin", torch.sin(circle_angles), persistent=False)

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

    def _camera_basis_from_views(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = torch.atan2(views[..., 0], views[..., 1])
        phi = torch.atan2(views[..., 2], views[..., 3])
        device = views.device
        dtype = views.dtype
        batch_size, num_views = views.shape[:2]

        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        cos_p = torch.cos(phi)
        sin_p = torch.sin(phi)

        r_ap1_native = torch.stack(
            [
                torch.stack([cos_t, -sin_t, zero], dim=-1),
                torch.stack([sin_t, cos_t, zero], dim=-1),
                torch.stack([zero, zero, one], dim=-1),
            ],
            dim=-2,
        )
        r_ap2_native = torch.stack(
            [
                torch.stack([one, zero, zero], dim=-1),
                torch.stack([zero, cos_p, sin_p], dim=-1),
                torch.stack([zero, -sin_p, cos_p], dim=-1),
            ],
            dim=-2,
        )

        coord = views.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        inv_coord = torch.linalg.inv(coord)
        coord_b = coord.view(1, 1, 3, 3)
        inv_coord_b = inv_coord.view(1, 1, 3, 3)
        r_total = coord_b @ r_ap1_native @ r_ap2_native @ inv_coord_b

        detector_to_iso = self.sid - self.source_to_iso
        axis_z = views.new_tensor([0.0, 0.0, 1.0]).view(1, 1, 3, 1)
        axis_y = views.new_tensor([0.0, 1.0, 0.0]).view(1, 1, 3, 1)
        axis_neg_x = views.new_tensor([-1.0, 0.0, 0.0]).view(1, 1, 3, 1)

        v_sensor = (r_total @ (axis_z * detector_to_iso)).squeeze(-1)
        v_source = -v_sensor / max(float(detector_to_iso), 1e-8) * self.source_to_iso
        local_x = self._safe_normalize((r_total @ axis_y).squeeze(-1))
        local_y = self._safe_normalize((r_total @ axis_neg_x).squeeze(-1))
        return v_sensor, v_source, local_x, local_y

    def _project_points_to_grid(self, points_m: torch.Tensor, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # points_m: [B,N,3], views: [B,V,4] -> grid [B,V,N,2], valid [B,V,N]
        v_sensor, v_source, local_x, local_y = self._camera_basis_from_views(views)
        points = points_m[:, None, :, :]
        v_sensor = v_sensor[:, :, None, :]
        v_source = v_source[:, :, None, :]
        local_x = local_x[:, :, None, :]
        local_y = local_y[:, :, None, :]

        n = torch.cross(local_x, local_y, dim=-1)
        u = v_source - points
        denom = torch.sum(u * n, dim=-1)
        valid = torch.abs(denom) > 1e-8
        denom_safe = torch.where(valid, denom, torch.ones_like(denom))

        w = points - v_sensor
        scale = -torch.sum(w * n, dim=-1) / denom_safe
        proj_points = points + scale.unsqueeze(-1) * u

        img_dim = float(self.image_size)
        img_dim_t = points.new_tensor(img_dim)
        sensor_width = self.imager_pixel_spacing * img_dim / 1000.0
        sensor_width_t = points.new_tensor(sensor_width)

        local_origin = v_sensor + sensor_width_t * (
            ((1.0 - img_dim_t / 2.0) / img_dim_t) * local_x
            + ((1.0 - img_dim_t / 2.0) / img_dim_t) * local_y
        )
        local_xmax = v_sensor + sensor_width_t * (
            ((img_dim_t - img_dim_t / 2.0) / img_dim_t) * local_x
            + ((1.0 - img_dim_t / 2.0) / img_dim_t) * local_y
        )
        local_ymax = v_sensor + sensor_width_t * (
            ((1.0 - img_dim_t / 2.0) / img_dim_t) * local_x
            + ((img_dim_t - img_dim_t / 2.0) / img_dim_t) * local_y
        )

        rel = proj_points - local_origin
        vx_scale = torch.sum(rel * local_x, dim=-1) / torch.clamp(torch.sum(local_x * local_x, dim=-1), min=1e-8)
        vy_scale = torch.sum(rel * local_y, dim=-1) / torch.clamp(torch.sum(local_y * local_y, dim=-1), min=1e-8)
        vx_projected = vx_scale.unsqueeze(-1) * local_x
        vy_projected = vy_scale.unsqueeze(-1) * local_y

        sign_x = torch.sign(torch.sum((local_xmax - local_origin) * vx_projected, dim=-1))
        sign_y = torch.sign(torch.sum((local_ymax - local_origin) * vy_projected, dim=-1))
        x_px = sign_x * img_dim_t * torch.linalg.norm(vx_projected, dim=-1) / torch.clamp(
            torch.linalg.norm(local_xmax - local_origin, dim=-1),
            min=1e-8,
        )
        y_px = sign_y * img_dim_t * torch.linalg.norm(vy_projected, dim=-1) / torch.clamp(
            torch.linalg.norm(local_ymax - local_origin, dim=-1),
            min=1e-8,
        )
        y_px = img_dim_t - y_px

        valid = valid & (x_px >= 0.0) & (x_px <= img_dim - 1.0) & (y_px >= 0.0) & (y_px <= img_dim - 1.0)
        grid_x = (x_px / max(img_dim - 1.0, 1.0)) * 2.0 - 1.0
        # grid_sample row coordinate follows image row; projector y_px is converted to row as H - y_px.
        row_px = img_dim_t - y_px
        grid_y = (row_px / max(img_dim - 1.0, 1.0)) * 2.0 - 1.0
        return torch.stack([grid_x, grid_y], dim=-1), valid

    def _distance_transform_2d(self, images: torch.Tensor) -> torch.Tensor:
        """Approximate normalized distance to the foreground mask on-device."""
        if images.dim() != 4 or images.shape[1] != 1:
            raise ValueError(f"images must have shape [B,1,H,W], got {tuple(images.shape)}")

        h, w = int(images.shape[-2]), int(images.shape[-1])
        max_dist = float(max(h, w))
        per_image_max = images.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        soft_mask = (images / per_image_max).clamp(0.0, 1.0)
        distance = (1.0 - soft_mask) * max_dist
        num_iters = min(self.distance_transform_num_iters, max(h, w))
        for _ in range(num_iters):
            padded = F.pad(distance, (1, 1, 1, 1), mode="constant", value=max_dist)
            neighbor_min = -F.max_pool2d(-padded, kernel_size=3, stride=1)
            distance = torch.minimum(distance, neighbor_min + 1.0)

        diagonal = float((h * h + w * w) ** 0.5)
        return distance / max(diagonal, 1e-6)

    @staticmethod
    def _estimate_derivatives(centerline: torch.Tensor) -> torch.Tensor:
        d = torch.zeros_like(centerline)
        n = int(centerline.shape[-2])
        if n < 2:
            return d
        d[:, 0] = centerline[:, 1] - centerline[:, 0]
        d[:, -1] = centerline[:, -1] - centerline[:, -2]
        if n > 2:
            d[:, 1:-1] = 0.5 * (centerline[:, 2:] - centerline[:, :-2])
        return d

    def _main_branch_surface_center(self, branch_points_m: torch.Tensor) -> torch.Tensor:
        centerline = branch_points_m[:, :, :3]
        radius = torch.clamp(branch_points_m[:, :, 3], min=1e-8)
        batch_size, num_points, _ = centerline.shape
        derivatives = self._estimate_derivatives(centerline)

        if num_points > 1:
            ref_index = torch.argmin(torch.abs(centerline[:, 1, :]), dim=-1)
        else:
            ref_index = torch.zeros((batch_size,), device=centerline.device, dtype=torch.long)
        normal = torch.zeros((batch_size, 3), device=centerline.device, dtype=centerline.dtype)
        normal.scatter_(1, ref_index[:, None], 1.0)

        normals = []
        convecs = []
        for point_idx in range(num_points):
            tangent = derivatives[:, point_idx]
            convec = self._safe_normalize(torch.cross(normal, tangent, dim=-1))
            normal = self._safe_normalize(torch.cross(tangent, convec, dim=-1))
            normals.append(normal)
            convecs.append(convec)
        normals_t = torch.stack(normals, dim=1)
        convecs_t = torch.stack(convecs, dim=1)

        center_sum = centerline.new_zeros((batch_size, 3))
        center_count = 0
        for point_idx in range(num_points):
            if point_idx == 0:
                rho = torch.linspace(
                    1.0 / 50.0,
                    1.0,
                    steps=49,
                    device=centerline.device,
                    dtype=centerline.dtype,
                )
            elif point_idx == num_points - 1:
                rho = torch.flip(
                    torch.linspace(
                        1.0 / 50.0,
                        1.0,
                        steps=49,
                        device=centerline.device,
                        dtype=centerline.dtype,
                    ),
                    dims=[0],
                )
            else:
                rho = torch.ones((1,), device=centerline.device, dtype=centerline.dtype)

            ring = (
                centerline[:, point_idx, None, None, :]
                + rho[None, :, None, None]
                * radius[:, point_idx, None, None, None]
                * (
                    self.circle_cos[None, None, :, None] * normals_t[:, point_idx, None, None, :]
                    + self.circle_sin[None, None, :, None] * convecs_t[:, point_idx, None, None, :]
                )
            )
            center_sum = center_sum + ring.reshape(batch_size, -1, 3).sum(dim=1)
            center_count += int(ring.shape[1] * ring.shape[2])

        if center_count <= 0:
            return centerline.mean(dim=1)
        return center_sum / float(center_count)

    def _sample_features(
        self,
        images: torch.Tensor,
        views: torch.Tensor,
        points_m: torch.Tensor,
        *,
        prepared_distance_map: torch.Tensor | None = None,
        prepared_feature_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.dim() == 4:
            batch_size, num_views, image_h, image_w = images.shape
            image_in = images.reshape(batch_size * num_views, 1, image_h, image_w)
        elif images.dim() == 5:
            batch_size, num_views, num_channels, image_h, image_w = images.shape
            image_in = images[:, :, :1].reshape(batch_size * num_views, 1, image_h, image_w)
        else:
            raise ValueError(f"images must have shape [B,V,H,W] or [B,V,C,H,W], got {tuple(images.shape)}")

        grid, valid = self._project_points_to_grid(points_m=points_m, views=views)
        _, _, num_points, _ = grid.shape
        grid_flat = grid.reshape(batch_size * num_views, num_points, 1, 2)

        offsets = torch.linspace(
            -(self.patch_size // 2),
            self.patch_size // 2,
            steps=self.patch_size,
            device=images.device,
            dtype=images.dtype,
        )
        if image_w > 1:
            offsets_x = offsets * (2.0 / float(image_w - 1))
        else:
            offsets_x = offsets * 0.0
        if image_h > 1:
            offsets_y = offsets * (2.0 / float(image_h - 1))
        else:
            offsets_y = offsets * 0.0
        yy, xx = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
        patch_offsets = torch.stack([xx, yy], dim=-1).reshape(1, 1, self.patch_size * self.patch_size, 2)
        patch_grid = grid_flat + patch_offsets

        raw_patch = F.grid_sample(
            image_in,
            patch_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        raw_patch = raw_patch[:, 0].reshape(
            batch_size,
            num_views,
            num_points,
            self.patch_size * self.patch_size,
        )

        pieces = [raw_patch]
        if self.use_distance_transform:
            distance_map = (
                self._distance_transform_2d(image_in)
                if prepared_distance_map is None
                else prepared_distance_map
            )
            distance_sample = F.grid_sample(
                distance_map,
                grid_flat,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            distance_sample = distance_sample[:, 0, :, 0].reshape(
                batch_size,
                num_views,
                num_points,
                1,
            )
            pieces.append(distance_sample)
        if self.image_feature_encoder is not None:
            fmap = (
                self.image_feature_encoder(image_in)
                if prepared_feature_map is None
                else prepared_feature_map
            )
            center_sample = F.grid_sample(
                fmap,
                grid_flat,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            center_sample = center_sample[:, :, :, 0].transpose(1, 2).reshape(
                batch_size,
                num_views,
                num_points,
                self.learned_feature_dim,
            )
            pieces.append(center_sample)

        view_feat = views[:, :, None, :].expand(batch_size, num_views, num_points, self.view_feat_dim)
        pieces.append(view_feat)
        evidence_raw = torch.cat(pieces, dim=-1)
        return evidence_raw.permute(0, 2, 1, 3), valid.permute(0, 2, 1)

    def forward(
        self,
        point_tokens: torch.Tensor,
        coarse_branch_points: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        code_mean: torch.Tensor | None,
        code_std: torch.Tensor | None,
        coord_scale_to_meter: float,
    ) -> torch.Tensor:
        if code_mean is None or code_std is None:
            points_world = coarse_branch_points
        else:
            points_world = coarse_branch_points * code_std.to(coarse_branch_points) + code_mean.to(coarse_branch_points)
        branch_points_m = points_world[:, 0, :, :4] * float(coord_scale_to_meter)
        center_m = self._main_branch_surface_center(branch_points_m).unsqueeze(1)
        points_m = branch_points_m[:, :, :3] - center_m

        evidence_raw, valid = self._sample_features(images=images, views=views, points_m=points_m)
        batch_size, num_points, num_views = evidence_raw.shape[:3]
        evidence = self.evidence_encoder(evidence_raw)

        if view_mask is not None:
            valid = valid & view_mask.to(device=valid.device, dtype=torch.bool).unsqueeze(1)
        key_padding_mask = ~valid.reshape(batch_size * num_points, num_views)
        all_masked = key_padding_mask.all(dim=1)
        if bool(all_masked.any().item()):
            key_padding_mask[all_masked, 0] = False
            evidence = evidence.reshape(batch_size * num_points, num_views, self.model_dim)
            evidence = evidence.clone()
            evidence[all_masked, 0, :] = 0.0
        else:
            evidence = evidence.reshape(batch_size * num_points, num_views, self.model_dim)

        q = self.norm_q(point_tokens).reshape(batch_size * num_points, 1, self.model_dim)
        refined, _ = self.attn(q, evidence, evidence, key_padding_mask=key_padding_mask, need_weights=False)
        out = point_tokens.reshape(batch_size * num_points, 1, self.model_dim) + self.dropout(refined)
        out = self.norm_out(out)
        out = out + self.dropout(self.ffn(out))
        return out.reshape(batch_size, num_points, self.model_dim)

class _ProjectionEvidenceResidualRefiner(_ProjectionEvidenceRefiner):
    def __init__(
        self,
        model_dim: int,
        view_feat_dim: int,
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
        residual_scale: float,
        center_points_on_main_surface: bool = True,
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
        residual_scale = float(residual_scale)
        if residual_scale < 0.0:
            raise ValueError(f"projection_evidence_residual_scale must be >= 0, got {residual_scale}")
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32),
            persistent=True,
        )
        self.center_points_on_main_surface = bool(center_points_on_main_surface)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.model_dim + 5, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, int(evidence_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(evidence_hidden_dim), 4),
        )

    def forward(
        self,
        coarse_branch_points: torch.Tensor,
        pooled_token: torch.Tensor,
        images: torch.Tensor,
        views: torch.Tensor,
        view_mask: torch.Tensor | None,
        code_mean: torch.Tensor | None,
        code_std: torch.Tensor | None,
        coord_scale_to_meter: float,
        projection_center_offset: torch.Tensor | None = None,
        prepared_distance_map: torch.Tensor | None = None,
        prepared_feature_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coarse_branch_points.dim() != 4 or coarse_branch_points.shape[-1] < 4:
            raise ValueError(
                "coarse_branch_points must have shape [B,M,N,4+] for residual projection evidence refinement, "
                f"got {tuple(coarse_branch_points.shape)}"
            )
        batch_size, num_branches, num_points = coarse_branch_points.shape[:3]
        if pooled_token.dim() == 2:
            pooled_token = pooled_token[:, None, :].expand(-1, num_branches, -1)
        if (
            pooled_token.dim() != 3
            or pooled_token.shape[:2] != (batch_size, num_branches)
        ):
            raise ValueError(
                "pooled_token must have shape [B,D] or [B,M,D] matching "
                "coarse_branch_points, "
                f"got {tuple(pooled_token.shape)}"
            )

        if code_mean is None or code_std is None:
            points_world = coarse_branch_points
        else:
            points_world = coarse_branch_points * code_std.to(coarse_branch_points) + code_mean.to(coarse_branch_points)
        branch_points_m = points_world[..., :4] * float(coord_scale_to_meter)
        if self.center_points_on_main_surface:
            center_m = self._main_branch_surface_center(
                branch_points_m[:, 0]
            ).view(batch_size, 1, 1, 3)
            points_m = branch_points_m[..., :3] - center_m
        elif projection_center_offset is not None:
            offset = projection_center_offset.to(branch_points_m)
            if tuple(offset.shape) != (batch_size, 3):
                raise ValueError(
                    "projection_center_offset must have shape [B,3], got "
                    f"{tuple(offset.shape)}."
                )
            points_m = branch_points_m[..., :3] - offset.view(
                batch_size, 1, 1, 3
            ) * float(coord_scale_to_meter)
        else:
            points_m = branch_points_m[..., :3]

        evidence_raw, valid = self._sample_features(
            images=images,
            views=views,
            points_m=points_m.reshape(batch_size, num_branches * num_points, 3),
            prepared_distance_map=prepared_distance_map,
            prepared_feature_map=prepared_feature_map,
        )
        num_views = evidence_raw.shape[2]
        evidence = self.evidence_encoder(evidence_raw)

        if view_mask is not None:
            valid = valid & view_mask.to(device=valid.device, dtype=torch.bool).unsqueeze(1)
        key_padding_mask = ~valid.reshape(
            batch_size * num_branches * num_points, num_views
        )
        all_masked = key_padding_mask.all(dim=1)
        evidence = evidence.reshape(
            batch_size * num_branches * num_points,
            num_views,
            self.model_dim,
        )
        if bool(all_masked.any().item()):
            key_padding_mask[all_masked, 0] = False
            evidence = evidence.clone()
            evidence[all_masked, 0, :] = 0.0

        coarse_point_code = coarse_branch_points[..., :4]
        t = torch.linspace(
            0.0,
            1.0,
            steps=num_points,
            device=coarse_branch_points.device,
            dtype=coarse_branch_points.dtype,
        ).view(1, 1, num_points, 1).expand(
            batch_size, num_branches, -1, -1
        )
        pooled = pooled_token[:, :, None, :].expand(
            batch_size, num_branches, num_points, -1
        )
        point_query = self.query_encoder(torch.cat([pooled, coarse_point_code, t], dim=-1))

        flat_point_count = batch_size * num_branches * num_points
        q = self.norm_q(point_query).reshape(flat_point_count, 1, self.model_dim)
        refined, _ = self.attn(q, evidence, evidence, key_padding_mask=key_padding_mask, need_weights=False)
        out = point_query.reshape(flat_point_count, 1, self.model_dim) + self.dropout(refined)
        out = self.norm_out(out)
        out = out + self.dropout(self.ffn(out))
        residual = torch.tanh(self.residual_head(out[:, 0, :])).reshape(
            batch_size, num_branches, num_points, 4
        )
        residual = residual * self.residual_scale.to(device=residual.device, dtype=residual.dtype)

        refined_branch_points = coarse_branch_points.clone()
        refined_branch_points[..., :4] = refined_branch_points[..., :4] + residual
        return refined_branch_points, residual
