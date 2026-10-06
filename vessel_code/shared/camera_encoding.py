# Transferred from methods/src/camera_encoding.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class CArmCameraGeometry:
    """Nominal pinhole geometry reconstructed from the stored C-arm angles."""

    world_to_camera: torch.Tensor  # [B,V,4,4]
    intrinsics: torch.Tensor  # [B,V,3,3], in pixels
    source_positions: torch.Tensor  # [B,V,3], in metres

@dataclass(frozen=True)
class CameraTokenLayout:
    """Detector positions associated with one view's cached feature tokens."""

    pixel_xy: torch.Tensor  # [L,2], detector pixel centres (column, row)
    rope_xy: torch.Tensor  # [L,2], patch-grid positions used by PRoPE
    spatial_mask: torch.Tensor  # [L], false for non-spatial prefix/register tokens

def angles_from_four_view_features(views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover theta/phi radians from [sin(theta), cos(theta), sin(phi), cos(phi)]."""

    if views.dim() != 3 or views.shape[-1] != 4:
        raise ValueError(
            "Camera conditioning requires views with shape [B,V,4] containing "
            "[sin(theta), cos(theta), sin(phi), cos(phi)]; got "
            f"{tuple(views.shape)}."
        )
    theta = torch.atan2(views[..., 0], views[..., 1])
    phi = torch.atan2(views[..., 2], views[..., 3])
    return theta, phi

def build_carm_camera_geometry(
    views: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    sid: float,
    source_to_iso: float,
    imager_pixel_spacing: float,
) -> CArmCameraGeometry:
    """Construct the nominal source-detector pinhole camera for every view.

    The rotation and axis conventions intentionally match
    ``DifferentiableVesselProjector._camera_basis``. Distances are configured in
    metres, while detector spacing is configured in millimetres per pixel.
    """

    if int(image_height) < 1 or int(image_width) < 1:
        raise ValueError("Camera image height and width must be positive.")
    if not math.isfinite(float(sid)) or float(sid) <= 0.0:
        raise ValueError(f"camera_encoding_sid must be finite and > 0, got {sid}.")
    if not math.isfinite(float(source_to_iso)) or not 0.0 < float(source_to_iso) < float(sid):
        raise ValueError(
            "camera_encoding_source_to_iso must be finite and lie in (0, SID); "
            f"got source_to_iso={source_to_iso}, SID={sid}."
        )
    if not math.isfinite(float(imager_pixel_spacing)) or float(imager_pixel_spacing) <= 0.0:
        raise ValueError(
            "camera_encoding_imager_pixel_spacing must be finite and > 0; "
            f"got {imager_pixel_spacing}."
        )

    theta, phi = angles_from_four_view_features(views)
    device, dtype = views.device, views.dtype
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)

    r_ap1_native = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta), zero), dim=-1),
            torch.stack((torch.sin(theta), torch.cos(theta), zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    r_ap2_native = torch.stack(
        (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, torch.cos(phi), torch.sin(phi)), dim=-1),
            torch.stack((zero, -torch.sin(phi), torch.cos(phi)), dim=-1),
        ),
        dim=-2,
    )
    coord = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        device=device,
        dtype=dtype,
    )
    r_total = coord @ r_ap1_native @ r_ap2_native @ coord.transpose(-1, -2)

    axis_x = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
    axis_y = torch.tensor((-1.0, 0.0, 0.0), device=device, dtype=dtype)
    axis_z = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
    local_x = torch.einsum("bvij,j->bvi", r_total, axis_x)
    local_y = torch.einsum("bvij,j->bvi", r_total, axis_y)
    forward = torch.einsum("bvij,j->bvi", r_total, axis_z)
    source_positions = -float(source_to_iso) * forward

    rotation = torch.stack((local_x, local_y, forward), dim=-2)
    translation = -torch.einsum("bvij,bvj->bvi", rotation, source_positions)
    world_to_camera = torch.zeros(
        (*views.shape[:2], 4, 4), device=device, dtype=dtype
    )
    world_to_camera[..., :3, :3] = rotation
    world_to_camera[..., :3, 3] = translation
    world_to_camera[..., 3, 3] = 1.0

    spacing_m = float(imager_pixel_spacing) / 1000.0
    focal_px = float(sid) / spacing_m
    intrinsics = torch.zeros(
        (*views.shape[:2], 3, 3), device=device, dtype=dtype
    )
    intrinsics[..., 0, 0] = focal_px
    intrinsics[..., 1, 1] = focal_px
    intrinsics[..., 0, 2] = float(image_width) / 2.0
    intrinsics[..., 1, 2] = float(image_height) / 2.0
    intrinsics[..., 2, 2] = 1.0
    return CArmCameraGeometry(
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        source_positions=source_positions,
    )

def build_camera_token_layout(
    *,
    feature_backbone: str,
    num_locations: int,
    image_height: int,
    image_width: int,
    resnet_pool_size: int,
    vggt_patch_size: int,
    vggt_target_image_height: int,
    vggt_target_image_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> CameraTokenLayout:
    """Map each cached token to a detector position.

    ResNet C2-C5 tokens comprise repeated pooled grids, one per FPN level.
    VGGT tokens may contain non-spatial prefix/register tokens before a single
    patch grid; those prefix tokens are assigned the centre position for PRoPE
    and a zero ray embedding.
    """

    if feature_backbone == "resnet_pre_fpn":
        grid_height = grid_width = int(resnet_pool_size)
        if grid_height < 1:
            raise ValueError(
                "Ray/PRoPE camera conditioning requires image_fpn_pool_size > 0 "
                "for resnet_pre_fpn."
            )
        spatial_count = grid_height * grid_width
        if int(num_locations) % spatial_count != 0:
            raise ValueError(
                f"ResNet token count {num_locations} is not a multiple of the "
                f"configured {grid_height}x{grid_width} pooled grid."
            )
        repeats = int(num_locations) // spatial_count
        prefix_count = 0
    else:
        patch_size = int(vggt_patch_size)
        if patch_size < 1:
            raise ValueError("vggt_patch_size must be positive for camera conditioning.")
        if (
            int(vggt_target_image_height) % patch_size != 0
            or int(vggt_target_image_width) % patch_size != 0
        ):
            raise ValueError(
                "vggt_target_image_size must be divisible by vggt_patch_size; "
                f"got {(vggt_target_image_height, vggt_target_image_width)} and "
                f"patch_size={patch_size}."
            )
        grid_height = int(vggt_target_image_height) // patch_size
        grid_width = int(vggt_target_image_width) // patch_size
        spatial_count = grid_height * grid_width
        prefix_count = int(num_locations) - spatial_count
        repeats = 1
        if prefix_count < 0:
            raise ValueError(
                f"VGGT token count {num_locations} is smaller than the expected "
                f"{grid_height}x{grid_width} spatial grid ({spatial_count}). Check "
                "vggt_target_image_size and vggt_patch_size against the cache metadata."
            )

    rows, columns = torch.meshgrid(
        torch.arange(grid_height, device=device, dtype=dtype),
        torch.arange(grid_width, device=device, dtype=dtype),
        indexing="ij",
    )
    base_rope_xy = torch.stack((columns.reshape(-1), rows.reshape(-1)), dim=-1)
    base_pixel_xy = torch.stack(
        (
            (columns.reshape(-1) + 0.5) * (float(image_width) / grid_width),
            (rows.reshape(-1) + 0.5) * (float(image_height) / grid_height),
        ),
        dim=-1,
    )
    rope_xy = base_rope_xy.repeat(repeats, 1)
    pixel_xy = base_pixel_xy.repeat(repeats, 1)
    spatial_mask = torch.ones(pixel_xy.shape[0], device=device, dtype=torch.bool)
    if prefix_count:
        prefix_pixel = torch.tensor(
            (float(image_width) / 2.0, float(image_height) / 2.0),
            device=device,
            dtype=dtype,
        ).expand(prefix_count, -1)
        prefix_rope = torch.zeros((prefix_count, 2), device=device, dtype=dtype)
        pixel_xy = torch.cat((prefix_pixel, pixel_xy), dim=0)
        rope_xy = torch.cat((prefix_rope, rope_xy), dim=0)
        spatial_mask = torch.cat(
            (torch.zeros(prefix_count, device=device, dtype=torch.bool), spatial_mask),
            dim=0,
        )
    if pixel_xy.shape[0] != int(num_locations):
        raise RuntimeError(
            f"Internal camera-token layout error: built {pixel_xy.shape[0]} positions "
            f"for {num_locations} tokens."
        )
    return CameraTokenLayout(
        pixel_xy=pixel_xy,
        rope_xy=rope_xy,
        spatial_mask=spatial_mask,
    )

def plucker_ray_embeddings(
    geometry: CArmCameraGeometry,
    layout: CameraTokenLayout,
) -> torch.Tensor:
    """Return patch-centre Plucker rays [B,V,L,6] in (o x d, d) order.

    This follows the token-level camera conditioning used by GS-LRM and LVSM:
    a six-dimensional Plucker ray map is concatenated with the corresponding
    image patch before learned projection. Here cached tokens replace raw image
    patches, so a learned MLP embeds the ray at each token's spatial centre.
    """

    intrinsics = geometry.intrinsics
    world_to_camera = geometry.world_to_camera
    batch_size, num_views = intrinsics.shape[:2]
    num_locations = layout.pixel_xy.shape[0]
    pixel_xy = layout.pixel_xy.to(device=intrinsics.device, dtype=intrinsics.dtype)
    u = pixel_xy[:, 0].view(1, 1, num_locations)
    v = pixel_xy[:, 1].view(1, 1, num_locations)
    camera_direction = torch.stack(
        (
            (u - intrinsics[..., 0, 2].unsqueeze(-1))
            / intrinsics[..., 0, 0].unsqueeze(-1),
            (v - intrinsics[..., 1, 2].unsqueeze(-1))
            / intrinsics[..., 1, 1].unsqueeze(-1),
            torch.ones(
                (batch_size, num_views, num_locations),
                device=intrinsics.device,
                dtype=intrinsics.dtype,
            ),
        ),
        dim=-1,
    )
    rotation = world_to_camera[..., :3, :3]
    world_direction = torch.einsum(
        "bvji,bvlj->bvli", rotation, camera_direction
    )
    world_direction = F.normalize(world_direction, dim=-1)
    origin = geometry.source_positions.unsqueeze(2).expand(-1, -1, num_locations, -1)
    moment = torch.cross(origin, world_direction, dim=-1)
    rays = torch.cat((moment, world_direction), dim=-1)
    spatial_mask = layout.spatial_mask.to(device=rays.device).view(1, 1, -1, 1)
    return rays * spatial_mask.to(dtype=rays.dtype)

def _lift_intrinsics(matrix: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((*matrix.shape[:-2], 4, 4), device=matrix.device, dtype=matrix.dtype)
    out[..., :3, :3] = matrix
    out[..., 3, 3] = 1.0
    return out

def _invert_se3(matrix: torch.Tensor) -> torch.Tensor:
    rotation_inverse = matrix[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rotation_inverse
    out[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation_inverse, matrix[..., :3, 3]
    )
    out[..., 3, 3] = 1.0
    return out

def _normalized_projective_matrices(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.zeros_like(intrinsics)
    normalized[..., 0, 0] = intrinsics[..., 0, 0] / float(image_width)
    normalized[..., 1, 1] = intrinsics[..., 1, 1] / float(image_height)
    normalized[..., 0, 2] = intrinsics[..., 0, 2] / float(image_width) - 0.5
    normalized[..., 1, 2] = intrinsics[..., 1, 2] / float(image_height) - 0.5
    normalized[..., 2, 2] = 1.0
    inverse_intrinsics = torch.zeros_like(normalized)
    inverse_intrinsics[..., 0, 0] = 1.0 / normalized[..., 0, 0]
    inverse_intrinsics[..., 1, 1] = 1.0 / normalized[..., 1, 1]
    inverse_intrinsics[..., 0, 2] = (
        -normalized[..., 0, 2] / normalized[..., 0, 0]
    )
    inverse_intrinsics[..., 1, 2] = (
        -normalized[..., 1, 2] / normalized[..., 1, 1]
    )
    inverse_intrinsics[..., 2, 2] = 1.0
    projective = _lift_intrinsics(normalized) @ world_to_camera
    projective_inverse = _invert_se3(world_to_camera) @ _lift_intrinsics(
        inverse_intrinsics
    )
    return projective, projective_inverse

def _apply_camera_matrix(
    features: torch.Tensor,
    matrix: torch.Tensor,
    token_view_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, sequence_length, feature_dim = features.shape
    if feature_dim % 4 != 0:
        raise ValueError(f"PRoPE projective feature block must be divisible by 4, got {feature_dim}.")
    if token_view_indices.shape != (sequence_length,):
        raise ValueError("token_view_indices must have one entry per sequence token.")
    per_token_matrix = matrix.index_select(1, token_view_indices)
    grouped = features.reshape(batch_size, num_heads, sequence_length, feature_dim // 4, 4)
    return torch.einsum("bnij,bhnrj->bhnri", per_token_matrix, grouped).reshape_as(features)

def _apply_rope(
    features: torch.Tensor,
    positions: torch.Tensor,
    *,
    frequency_base: float,
    inverse: bool = False,
) -> torch.Tensor:
    if features.shape[-1] % 2 != 0:
        raise ValueError("Each PRoPE x/y rotary block must have even dimension.")
    half = features.shape[-1] // 2
    frequencies = float(frequency_base) ** (
        -torch.arange(half, device=features.device, dtype=torch.float32)
        / float(max(half, 1))
    )
    angles = positions.to(device=features.device, dtype=torch.float32).view(1, 1, -1, 1)
    angles = angles * frequencies.view(1, 1, 1, -1)
    cosine = torch.cos(angles).to(dtype=features.dtype)
    sine = torch.sin(angles).to(dtype=features.dtype)
    first, second = features[..., :half], features[..., half:]
    if inverse:
        return torch.cat((cosine * first - sine * second, sine * first + cosine * second), dim=-1)
    return torch.cat((cosine * first + sine * second, -sine * first + cosine * second), dim=-1)

class PRoPESelfAttention(nn.Module):
    """Projective Positional Encoding attention from Li et al., NeurIPS 2025.

    The implementation follows the paper/official code's block-diagonal split:
    one half of every head is transformed with the lifted projective camera
    matrix and one quarter each uses x/y RoPE. Token positions are supplied
    explicitly so the formulation also supports repeated ResNet FPN grids and
    VGGT prefix/register tokens.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        frequency_base: float = 100.0,
    ) -> None:
        super().__init__()
        if int(model_dim) % int(num_heads) != 0:
            raise ValueError("model_dim must be divisible by num_heads for PRoPE.")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        if self.head_dim % 8 != 0:
            raise ValueError(
                "PRoPE requires attention head_dim divisible by 8 so the projective, "
                f"x-RoPE and y-RoPE blocks are valid; got head_dim={self.head_dim}."
            )
        if not math.isfinite(float(frequency_base)) or float(frequency_base) <= 0.0:
            raise ValueError("prope_frequency_base must be finite and positive.")
        self.dropout = float(dropout)
        self.frequency_base = float(frequency_base)
        self.qkv_projection = nn.Linear(self.model_dim, 3 * self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, self.model_dim)

    def _apply_input_encoding(
        self,
        features: torch.Tensor,
        *,
        projective_matrix: torch.Tensor,
        rope_xy: torch.Tensor,
        token_view_indices: torch.Tensor,
    ) -> torch.Tensor:
        projective_dim = self.head_dim // 2
        rotary_dim = self.head_dim // 4
        projective, rope_x, rope_y = torch.split(
            features, (projective_dim, rotary_dim, rotary_dim), dim=-1
        )
        projective = _apply_camera_matrix(
            projective, projective_matrix, token_view_indices
        )
        rope_x = _apply_rope(
            rope_x,
            rope_xy[:, 0],
            frequency_base=self.frequency_base,
        )
        rope_y = _apply_rope(
            rope_y,
            rope_xy[:, 1],
            frequency_base=self.frequency_base,
        )
        return torch.cat((projective, rope_x, rope_y), dim=-1)

    def _apply_output_encoding(
        self,
        features: torch.Tensor,
        *,
        projective_matrix: torch.Tensor,
        rope_xy: torch.Tensor,
        token_view_indices: torch.Tensor,
    ) -> torch.Tensor:
        projective_dim = self.head_dim // 2
        rotary_dim = self.head_dim // 4
        projective, rope_x, rope_y = torch.split(
            features, (projective_dim, rotary_dim, rotary_dim), dim=-1
        )
        projective = _apply_camera_matrix(
            projective, projective_matrix, token_view_indices
        )
        rope_x = _apply_rope(
            rope_x,
            rope_xy[:, 0],
            frequency_base=self.frequency_base,
            inverse=True,
        )
        rope_y = _apply_rope(
            rope_y,
            rope_xy[:, 1],
            frequency_base=self.frequency_base,
            inverse=True,
        )
        return torch.cat((projective, rope_x, rope_y), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        rope_xy: torch.Tensor,
        token_view_indices: torch.Tensor,
        padding_mask: torch.Tensor | None,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = tokens.shape
        qkv = self.qkv_projection(tokens).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        projective, projective_inverse = _normalized_projective_matrices(
            world_to_camera,
            intrinsics,
            image_height=int(image_height),
            image_width=int(image_width),
        )
        projective = projective.to(dtype=query.dtype)
        projective_inverse = projective_inverse.to(dtype=query.dtype)
        query = self._apply_input_encoding(
            query,
            projective_matrix=projective.transpose(-1, -2),
            rope_xy=rope_xy,
            token_view_indices=token_view_indices,
        )
        key = self._apply_input_encoding(
            key,
            projective_matrix=projective_inverse,
            rope_xy=rope_xy,
            token_view_indices=token_view_indices,
        )
        value = self._apply_input_encoding(
            value,
            projective_matrix=projective_inverse,
            rope_xy=rope_xy,
            token_view_indices=token_view_indices,
        )
        attention_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "PRoPE padding mask must have shape [B,N], got "
                    f"{tuple(padding_mask.shape)}."
                )
            attention_mask = torch.zeros(
                (batch_size, 1, 1, sequence_length),
                device=tokens.device,
                dtype=query.dtype,
            )
            attention_mask = attention_mask.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = self._apply_output_encoding(
            output,
            projective_matrix=projective,
            rope_xy=rope_xy,
            token_view_indices=token_view_indices,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.model_dim)
        output = self.output_projection(output)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return output

class PRoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        frequency_base: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(model_dim))
        self.norm2 = nn.LayerNorm(int(model_dim))
        self.self_attention = PRoPESelfAttention(
            model_dim=int(model_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            frequency_base=float(frequency_base),
        )
        self.dropout1 = nn.Dropout(float(dropout))
        self.linear1 = nn.Linear(int(model_dim), int(mlp_hidden_dim))
        self.linear2 = nn.Linear(int(mlp_hidden_dim), int(model_dim))
        self.dropout_ff = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor, **camera_kwargs: object) -> torch.Tensor:
        tokens = tokens + self.dropout1(
            self.self_attention(self.norm1(tokens), **camera_kwargs)
        )
        hidden = self.linear2(self.dropout_ff(F.gelu(self.linear1(self.norm2(tokens)))))
        return tokens + self.dropout2(hidden)

class PRoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        num_layers: int,
        frequency_base: float = 100.0,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("PRoPE encoder requires num_encoder_layers >= 1.")
        self.layers = nn.ModuleList(
            [
                PRoPETransformerEncoderLayer(
                    model_dim=int(model_dim),
                    num_heads=int(num_heads),
                    mlp_hidden_dim=int(mlp_hidden_dim),
                    dropout=float(dropout),
                    frequency_base=float(frequency_base),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, tokens: torch.Tensor, **camera_kwargs: object) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, **camera_kwargs)
        return self.norm(tokens)
