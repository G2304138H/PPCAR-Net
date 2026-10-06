# Transferred from methods/parametric_methods/centerline_heatmap.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import NamedTuple, Tuple, Union
import torch
import torch.nn.functional as F

HeatmapSize = Union[int, Tuple[int, int]]

GAUSSIAN_CENTERLINE_TARGET = "gaussian"

MASK_NORMALIZED_CENTERLINE_DISTANCE_TARGET = (
    "mask_normalized_centerline_distance"
)

class CenterlineTargetMaps(NamedTuple):
    """Dense target plus the two binary maps used to construct it."""

    target: torch.Tensor
    centerline: torch.Tensor
    vessel_mask: torch.Tensor

def normalize_centerline_target_mode(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "gaussian": GAUSSIAN_CENTERLINE_TARGET,
        "gaussian_heatmap": GAUSSIAN_CENTERLINE_TARGET,
        "mask_normalized_affinity": MASK_NORMALIZED_CENTERLINE_DISTANCE_TARGET,
        "mask_normalized_centerline_distance": (
            MASK_NORMALIZED_CENTERLINE_DISTANCE_TARGET
        ),
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "centerline target mode must be 'gaussian' or "
            "'mask_normalized_centerline_distance', got "
            f"{value!r}."
        ) from error

def _heatmap_shape(map_size: HeatmapSize) -> Tuple[int, int]:
    if isinstance(map_size, int):
        height = width = int(map_size)
    elif isinstance(map_size, tuple) and len(map_size) == 2:
        height, width = (int(map_size[0]), int(map_size[1]))
    else:
        raise ValueError(
            "map_size must be an integer or a (height, width) tuple, got "
            f"{map_size!r}."
        )
    if height < 2 or width < 2:
        raise ValueError(
            "Centerline heatmap dimensions must both be >= 2, got "
            f"{(height, width)}."
        )
    return height, width

def render_centerline_heatmap_from_grid(
    projected_grid: torch.Tensor,
    valid: torch.Tensor,
    *,
    map_size: HeatmapSize,
    sigma_px: float = 1.25,
    radius_px: int = 2,
) -> torch.Tensor:
    """Rasterize projected centreline points into a soft union heatmap.

    Args:
        projected_grid: ``[B,V,P,2]`` raster coordinates normalized for
            ``grid_sample(..., align_corners=True)``. The final coordinate is
            ordered ``(x, y)`` where ``y`` is the image row direction.
        valid: Caller-combined ``[B,V,P]`` mask. It should already include
            projection, branch, point and view validity.
        map_size: Output side length or ``(height, width)``.
        sigma_px: Gaussian standard deviation in *output-map* pixels.
        radius_px: Integer Gaussian support radius in output-map pixels.

    Returns:
        A differentiable soft union heatmap with shape ``[B,V,H,W]``. Repeated
        or overlapping branch points saturate the same occupancy map; they do
        not repel one another or create branch-specific penalties.
    """

    if projected_grid.ndim != 4 or int(projected_grid.shape[-1]) != 2:
        raise ValueError(
            "projected_grid must have shape [B,V,P,2], got "
            f"{tuple(projected_grid.shape)}."
        )
    if valid.shape != projected_grid.shape[:-1]:
        raise ValueError(
            "valid must have shape [B,V,P] matching projected_grid, got "
            f"{tuple(valid.shape)} and {tuple(projected_grid.shape)}."
        )
    height, width = _heatmap_shape(map_size)
    sigma_px = float(sigma_px)
    radius_px = int(radius_px)
    if not math.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError(
            f"sigma_px must be finite and > 0, got {sigma_px}."
        )
    if radius_px < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}.")

    batch_size, num_views, num_points = projected_grid.shape[:3]
    batch_views = int(batch_size) * int(num_views)
    valid = valid.to(device=projected_grid.device, dtype=torch.bool)
    finite = torch.isfinite(projected_grid).all(dim=-1)
    valid = valid & finite
    safe_grid = torch.where(
        valid.unsqueeze(-1), projected_grid, torch.zeros_like(projected_grid)
    )

    # align_corners=True maps normalized endpoints exactly to the first/last
    # pixel centres. This also lets a low-resolution evidence map share the
    # same normalized projection grid as the full-resolution input image.
    x = (safe_grid[..., 0] + 1.0) * (float(width - 1) / 2.0)
    y = (safe_grid[..., 1] + 1.0) * (float(height - 1) / 2.0)
    centre_x = torch.round(x)
    centre_y = torch.round(y)

    offset_values = torch.arange(
        -radius_px,
        radius_px + 1,
        device=projected_grid.device,
        dtype=projected_grid.dtype,
    )
    offset_y = offset_values[:, None].expand(
        2 * radius_px + 1, 2 * radius_px + 1
    ).reshape(-1)
    offset_x = offset_values[None, :].expand(
        2 * radius_px + 1, 2 * radius_px + 1
    ).reshape(-1)

    px = centre_x.unsqueeze(-1) + offset_x
    py = centre_y.unsqueeze(-1) + offset_y
    inside = (
        valid.unsqueeze(-1)
        & (px >= 0.0)
        & (px < float(width))
        & (py >= 0.0)
        & (py < float(height))
    )
    px_safe = px.clamp(0.0, float(width - 1))
    py_safe = py.clamp(0.0, float(height - 1))
    distance_squared = (x.unsqueeze(-1) - px_safe).square() + (
        y.unsqueeze(-1) - py_safe
    ).square()
    weights = torch.exp(
        -distance_squared / (2.0 * sigma_px * sigma_px)
    ) * inside.to(dtype=projected_grid.dtype)

    linear_indices = py_safe.long() * width + px_safe.long()
    view_offsets = (
        torch.arange(batch_views, device=projected_grid.device)
        * (height * width)
    ).reshape(int(batch_size), int(num_views), 1, 1)
    linear_indices = linear_indices + view_offsets
    accumulated = projected_grid.new_zeros(
        batch_views * height * width
    )
    if int(num_points) > 0:
        accumulated.index_add_(
            0,
            linear_indices.reshape(-1),
            weights.reshape(-1),
        )
    accumulated = accumulated.reshape(
        int(batch_size), int(num_views), height, width
    )
    return 1.0 - torch.exp(-accumulated)

def _resize_vessel_mask(
    input_masks: torch.Tensor,
    *,
    map_size: tuple[int, int],
    threshold: float,
) -> torch.Tensor:
    if input_masks.ndim != 4:
        raise ValueError(
            "input_masks must have shape [B,V,H,W], got "
            f"{tuple(input_masks.shape)}."
        )
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"mask_threshold must be finite and in [0,1], got {threshold}."
        )
    batch_size, num_views, input_height, input_width = input_masks.shape
    flat = (input_masks > threshold).to(dtype=input_masks.dtype).reshape(
        int(batch_size) * int(num_views), 1, int(input_height), int(input_width)
    )
    output_height, output_width = map_size
    if (input_height, input_width) == map_size:
        resized = flat
    elif output_height <= input_height and output_width <= input_width:
        # Max pooling preserves thin side branches that could disappear under
        # an area average before thresholding.
        resized = F.adaptive_max_pool2d(flat, map_size)
    else:
        resized = F.interpolate(flat, size=map_size, mode="nearest")
    return (resized[:, 0] > 0.5).reshape(
        int(batch_size), int(num_views), output_height, output_width
    )

def _jump_flood_distance(seed: torch.Tensor) -> torch.Tensor:
    """Approximate Euclidean distance to a binary seed map on-device.

    Jump flooding needs logarithmically many propagation rounds, avoiding the
    one-full-image-iteration-per-pixel cost of a naive morphological distance
    transform. Exact EDT precision is unnecessary for this smooth target.
    """

    if seed.ndim != 4:
        raise ValueError(
            f"seed must have shape [N,C,H,W], got {tuple(seed.shape)}."
        )
    num_items, num_channels, height, width = seed.shape
    dtype = (
        torch.float32
        if seed.dtype in {torch.bool, torch.float16, torch.bfloat16}
        else seed.dtype
    )
    y, x = torch.meshgrid(
        torch.arange(height, device=seed.device, dtype=dtype),
        torch.arange(width, device=seed.device, dtype=dtype),
        indexing="ij",
    )
    coordinates = torch.stack((y, x), dim=0).reshape(1, 1, 2, height, width)
    coordinates = coordinates.expand(num_items, num_channels, 2, height, width)
    nearest = torch.where(
        seed.to(dtype=torch.bool).unsqueeze(2),
        coordinates,
        coordinates.new_full((), -1.0),
    )

    jump = 1
    while jump < max(height, width):
        jump *= 2
    jump //= 2
    while jump >= 1:
        flat_nearest = nearest.reshape(
            num_items * num_channels, 2, height, width
        )
        padded = F.pad(
            flat_nearest,
            (jump, jump, jump, jump),
            mode="constant",
            value=-1.0,
        )
        candidates = []
        for offset_y in (-jump, 0, jump):
            for offset_x in (-jump, 0, jump):
                start_y = jump + offset_y
                start_x = jump + offset_x
                candidates.append(
                    padded[
                        :,
                        :,
                        start_y : start_y + height,
                        start_x : start_x + width,
                    ]
                )
        candidate_coordinates = torch.stack(candidates, dim=1).reshape(
            num_items, num_channels, 9, 2, height, width
        )
        valid = candidate_coordinates[:, :, :, 0] >= 0.0
        distance_squared = (
            candidate_coordinates - coordinates.unsqueeze(2)
        ).square().sum(dim=3)
        distance_squared = distance_squared.masked_fill(
            ~valid, torch.inf
        )
        best = distance_squared.argmin(dim=2, keepdim=True)
        nearest = torch.gather(
            candidate_coordinates,
            dim=2,
            index=best.unsqueeze(3).expand(
                num_items, num_channels, 1, 2, height, width
            ),
        ).squeeze(2)
        jump //= 2

    valid_nearest = nearest[:, :, 0] >= 0.0
    distance = (nearest - coordinates).square().sum(dim=2).sqrt()
    maximum = math.hypot(float(height), float(width))
    return torch.where(
        valid_nearest,
        distance,
        distance.new_full((), maximum),
    ).to(dtype=seed.dtype if seed.is_floating_point() else torch.float32)

def render_centerline_target_maps_from_grid(
    projected_grid: torch.Tensor,
    valid: torch.Tensor,
    input_masks: torch.Tensor,
    *,
    map_size: HeatmapSize,
    target_mode: str = GAUSSIAN_CENTERLINE_TARGET,
    sigma_px: float = 1.25,
    radius_px: int = 2,
    mask_threshold: float = 0.5,
    affinity_gamma: float = 2.0,
) -> CenterlineTargetMaps:
    """Render a Gaussian heatmap or mask-normalized centreline affinity.

    The affinity is exactly zero outside the vessel mask, zero on its inner
    boundary (unless that pixel is itself a projected centreline sample), and
    rises continuously toward the projected ground-truth centreline:

    ``(d_boundary / (d_boundary + d_centerline + eps)) ** gamma``.
    """

    mode = normalize_centerline_target_mode(target_mode)
    height, width = _heatmap_shape(map_size)
    if projected_grid.ndim != 4 or int(projected_grid.shape[-1]) != 2:
        raise ValueError(
            "projected_grid must have shape [B,V,P,2], got "
            f"{tuple(projected_grid.shape)}."
        )
    batch_size, num_views = projected_grid.shape[:2]
    if input_masks.shape[:2] != (batch_size, num_views):
        raise ValueError(
            "input_masks batch/view dimensions must match projected_grid, got "
            f"{tuple(input_masks.shape[:2])} and {(batch_size, num_views)}."
        )
    vessel_mask = _resize_vessel_mask(
        input_masks.to(device=projected_grid.device, dtype=projected_grid.dtype),
        map_size=(height, width),
        threshold=mask_threshold,
    )
    # Radius zero means each projected sample occupies exactly one output-map
    # pixel. Thresholding removes sub-pixel Gaussian amplitude and leaves the
    # singular projected centreline seed requested by the target definition.
    centerline = render_centerline_heatmap_from_grid(
        projected_grid,
        valid,
        map_size=(height, width),
        sigma_px=0.5,
        radius_px=0,
    ) > 0.0
    centerline = centerline & vessel_mask

    if mode == GAUSSIAN_CENTERLINE_TARGET:
        target = render_centerline_heatmap_from_grid(
            projected_grid,
            valid,
            map_size=(height, width),
            sigma_px=sigma_px,
            radius_px=radius_px,
        )
        return CenterlineTargetMaps(
            target=target,
            centerline=centerline.to(dtype=target.dtype),
            vessel_mask=vessel_mask.to(dtype=target.dtype),
        )

    affinity_gamma = float(affinity_gamma)
    if not math.isfinite(affinity_gamma) or affinity_gamma <= 0.0:
        raise ValueError(
            "affinity_gamma must be finite and > 0, got "
            f"{affinity_gamma}."
        )
    flat_mask = vessel_mask.reshape(
        int(batch_size) * int(num_views), 1, height, width
    )
    padded_background = F.pad(
        (~flat_mask).to(dtype=projected_grid.dtype),
        (1, 1, 1, 1),
        mode="constant",
        value=1.0,
    )
    adjacent_to_background = F.max_pool2d(
        padded_background, kernel_size=3, stride=1
    ) > 0.0
    boundary = flat_mask & adjacent_to_background
    flat_centerline = centerline.reshape(
        int(batch_size) * int(num_views), 1, height, width
    )
    seeds = torch.cat((boundary, flat_centerline), dim=1)
    distances = _jump_flood_distance(seeds)
    boundary_distance = distances[:, 0]
    centerline_distance = distances[:, 1]
    denominator = boundary_distance + centerline_distance
    affinity = boundary_distance / denominator.clamp_min(1.0e-6)
    affinity = affinity.pow(affinity_gamma) * flat_mask[:, 0].to(
        dtype=affinity.dtype
    )
    # Preserve an exact ridge target even where a very thin vessel makes its
    # projected centreline coincide with the discrete mask boundary.
    affinity = torch.where(
        flat_centerline[:, 0], torch.ones_like(affinity), affinity
    )
    has_centerline = flat_centerline.flatten(start_dim=1).any(dim=1)
    affinity = affinity * has_centerline[:, None, None].to(affinity.dtype)
    target = affinity.reshape(
        int(batch_size), int(num_views), height, width
    ).to(dtype=projected_grid.dtype)
    return CenterlineTargetMaps(
        target=target,
        centerline=centerline.to(dtype=target.dtype),
        vessel_mask=vessel_mask.to(dtype=target.dtype),
    )
