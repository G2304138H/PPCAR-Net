# Transferred from methods/src/mountain_weighting.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn.functional as F

DEFAULT_MOUNTAIN_WIDTH_PAIRS: tuple[tuple[int, int], ...] = (
    (8, 8),
    (8, 12),
    (12, 8),
    (16, 16),
    (24, 24),
)

DEFAULT_MOUNTAIN_SCALE_FRACTIONS: tuple[float, ...] = (
    0.02,
    0.04,
    0.06,
    0.08,
    0.12,
    0.16,
)

MOUNTAIN_DETECTORS: tuple[str, ...] = (
    "local_prominence",
    "rise_fall",
    "sizer_derivative",
)

def normalize_mountain_detector(raw_value: Any) -> str:
    """Return a canonical detector name while accepting descriptive aliases."""

    normalized = str(raw_value or "local_prominence").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "prominence": "local_prominence",
        "local": "local_prominence",
        "asymmetric_prominence": "local_prominence",
        "rise_and_fall": "rise_fall",
        "risefall": "rise_fall",
        "sizer": "sizer_derivative",
        "sizer_inspired": "sizer_derivative",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MOUNTAIN_DETECTORS:
        raise ValueError(
            "mountain detector must be one of "
            f"{MOUNTAIN_DETECTORS}, got {raw_value!r}."
        )
    return normalized

def parse_mountain_scale_fractions(raw_value: Any) -> tuple[float, ...]:
    """Parse relative side-width scales used by the SiZer-inspired detector."""

    if raw_value is None:
        return DEFAULT_MOUNTAIN_SCALE_FRACTIONS
    if isinstance(raw_value, str):
        entries: list[Any] = [
            entry.strip()
            for entry in raw_value.replace(";", ",").split(",")
            if entry.strip()
        ]
    elif isinstance(raw_value, (list, tuple)):
        entries = list(raw_value)
    else:
        entries = [raw_value]

    fractions: list[float] = []
    for entry in entries:
        value = float(entry)
        if not math.isfinite(value) or not 0.0 < value < 0.5:
            raise ValueError(
                "Mountain scale fractions must be finite and in (0,0.5), "
                f"got {entry!r}."
            )
        if value not in fractions:
            fractions.append(value)
    if not fractions:
        raise ValueError("At least one mountain scale fraction is required.")
    return tuple(sorted(fractions))

def parse_mountain_width_pairs(raw_value: Any) -> tuple[tuple[int, int], ...]:
    """Parse symmetric widths or explicit asymmetric ``(left, right)`` pairs."""

    if raw_value is None:
        return DEFAULT_MOUNTAIN_WIDTH_PAIRS
    if isinstance(raw_value, str):
        entries: list[Any] = [
            entry.strip()
            for entry in raw_value.replace(";", ",").split(",")
            if entry.strip()
        ]
    elif isinstance(raw_value, (list, tuple)):
        entries = list(raw_value)
    else:
        entries = [raw_value]

    pairs: list[tuple[int, int]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized = entry.lower().replace("x", ":")
            if ":" in normalized:
                left_raw, right_raw = normalized.split(":", maxsplit=1)
                left, right = int(left_raw), int(right_raw)
            else:
                left = right = int(normalized)
        elif isinstance(entry, (list, tuple)):
            if len(entry) != 2:
                raise ValueError(
                    "Each mountain width pair must contain [left, right], "
                    f"got {entry!r}."
                )
            left, right = int(entry[0]), int(entry[1])
        else:
            left = right = int(entry)
        if left < 1 or right < 1:
            raise ValueError(
                "Mountain left/right widths must be positive integers, "
                f"got {(left, right)}."
            )
        pair = (left, right)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("At least one mountain width pair is required.")
    return tuple(pairs)

def detached_asymmetric_mountain_regions(
    error_profile: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    width_pairs: Any = DEFAULT_MOUNTAIN_WIDTH_PAIRS,
    min_prominence: float = 1.0,
    temperature: float = 0.25,
    smooth_kernel: int = 5,
    edge_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Detect asymmetric mountain-shaped regions in ordered error profiles.

    The detector is intentionally evaluated without autograd. A candidate centre
    must be higher than both its left and right local means. Independent
    left/right widths allow gradual-rise/fast-fall and fast-rise/gradual-fall
    mountains. Candidate activations are expanded into tapered regional masks.
    """

    if error_profile.dim() < 1:
        raise ValueError("error_profile must have at least one dimension.")
    prominence_threshold = float(min_prominence)
    activation_temperature = float(temperature)
    edge = float(edge_weight)
    kernel_size = int(smooth_kernel)
    if not math.isfinite(prominence_threshold) or prominence_threshold <= 0.0:
        raise ValueError(
            "min_prominence must be finite and > 0, "
            f"got {min_prominence}."
        )
    if not math.isfinite(activation_temperature) or activation_temperature <= 0.0:
        raise ValueError(
            f"temperature must be finite and > 0, got {temperature}."
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"smooth_kernel must be a positive odd integer, got {smooth_kernel}."
        )
    if not math.isfinite(edge) or not 0.0 <= edge <= 1.0:
        raise ValueError(
            f"edge_weight must be finite and in [0,1], got {edge_weight}."
        )
    parsed_widths = parse_mountain_width_pairs(width_pairs)
    if valid is None:
        valid_mask = torch.ones_like(error_profile, dtype=torch.bool)
    else:
        if valid.shape != error_profile.shape:
            raise ValueError(
                "valid must match error_profile, got "
                f"{tuple(valid.shape)} and {tuple(error_profile.shape)}."
            )
        valid_mask = valid.to(device=error_profile.device, dtype=torch.bool)

    with torch.no_grad():
        finite = torch.isfinite(error_profile)
        valid_detached = valid_mask.detach() & finite
        clean = torch.where(
            valid_detached,
            error_profile.detach(),
            torch.zeros_like(error_profile),
        )
        point_count = int(clean.shape[-1])
        flat_error = clean.reshape(-1, point_count)
        flat_valid = valid_detached.reshape(-1, point_count)
        valid_float = flat_valid.to(dtype=flat_error.dtype)

        if kernel_size > 1:
            radius = kernel_size // 2
            smoothing_kernel = flat_error.new_ones((1, 1, kernel_size))
            smooth_sum = F.conv1d(
                F.pad(
                    (flat_error * valid_float).unsqueeze(1),
                    (radius, radius),
                ),
                smoothing_kernel,
            ).squeeze(1)
            smooth_count = F.conv1d(
                F.pad(valid_float.unsqueeze(1), (radius, radius)),
                smoothing_kernel,
            ).squeeze(1)
            smoothed = smooth_sum / smooth_count.clamp_min(1.0)
        else:
            smoothed = flat_error

        cumulative_error = F.pad(torch.cumsum(smoothed, dim=-1), (1, 0))
        cumulative_valid = F.pad(torch.cumsum(valid_float, dim=-1), (1, 0))
        indices = torch.arange(point_count, device=flat_error.device)
        region = torch.zeros_like(flat_error)
        peak_activation = flat_error.new_zeros(())

        for left_width, right_width in parsed_widths:
            if point_count <= left_width + right_width:
                continue
            left_start = (indices - left_width).clamp_min(0)
            right_end = (indices + right_width + 1).clamp_max(point_count)
            left_sum = cumulative_error[:, indices] - cumulative_error[:, left_start]
            right_sum = (
                cumulative_error[:, right_end]
                - cumulative_error[:, indices + 1]
            )
            left_count = (
                cumulative_valid[:, indices] - cumulative_valid[:, left_start]
            )
            right_count = (
                cumulative_valid[:, right_end]
                - cumulative_valid[:, indices + 1]
            )
            candidate_valid = (
                flat_valid
                & (left_count >= float(left_width))
                & (right_count >= float(right_width))
            )
            left_mean = left_sum / left_count.clamp_min(1.0)
            right_mean = right_sum / right_count.clamp_min(1.0)
            prominence = torch.minimum(
                smoothed - left_mean,
                smoothed - right_mean,
            )
            positive_scale = (
                prominence / prominence_threshold
            ).clamp(min=0.0, max=1.0)
            activation = torch.sigmoid(
                (prominence - prominence_threshold) / activation_temperature
            ) * positive_scale
            activation = torch.where(
                candidate_valid & (prominence > 0.0),
                activation,
                torch.zeros_like(activation),
            )
            peak_activation = torch.maximum(
                peak_activation,
                activation.max(),
            )

            # At output index j, inspect candidate centres
            # i in [j-right_width, j+left_width]. The reversed offset order
            # aligns each centre activation with its tapered support at j.
            candidate_windows = F.pad(
                activation,
                (right_width, left_width),
            ).unfold(
                dimension=-1,
                size=left_width + right_width + 1,
                step=1,
            )
            offsets = torch.arange(
                right_width,
                -left_width - 1,
                -1,
                device=flat_error.device,
                dtype=flat_error.dtype,
            )
            left_taper = 1.0 - (1.0 - edge) * (
                offsets.abs() / float(left_width)
            )
            right_taper = 1.0 - (1.0 - edge) * (
                offsets.abs() / float(right_width)
            )
            taper = torch.where(offsets < 0.0, left_taper, right_taper)
            spread = (candidate_windows * taper.view(1, 1, -1)).amax(dim=-1)
            region = torch.maximum(region, spread)

        region = region.clamp(0.0, 1.0) * valid_float
        valid_count = valid_float.sum().clamp_min(1.0)
        region_fraction = region.sum() / valid_count
        region_mask = region.reshape(error_profile.shape).detach()

    return {
        "region_mask": region_mask,
        "region_fraction": region_fraction.detach(),
        "peak_activation": peak_activation.detach(),
    }

def _smooth_flat_profile(
    flat_error: torch.Tensor,
    valid_float: torch.Tensor,
    kernel_size: int,
    *,
    gaussian: bool = False,
) -> torch.Tensor:
    """Smooth flattened profiles without allowing invalid values into averages."""

    if kernel_size <= 1:
        return flat_error
    radius = kernel_size // 2
    if gaussian:
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=flat_error.device,
            dtype=flat_error.dtype,
        )
        sigma = max(float(radius) / 2.0, 0.5)
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
    else:
        kernel = flat_error.new_ones(kernel_size)
    kernel = kernel.view(1, 1, -1)
    weighted_sum = F.conv1d(
        F.pad((flat_error * valid_float).unsqueeze(1), (radius, radius)),
        kernel,
    ).squeeze(1)
    weight_sum = F.conv1d(
        F.pad(valid_float.unsqueeze(1), (radius, radius)),
        kernel,
    ).squeeze(1)
    return weighted_sum / weight_sum.clamp_min(torch.finfo(flat_error.dtype).eps)

def _prepare_detached_profiles(
    error_profile: torch.Tensor,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if error_profile.dim() < 1:
        raise ValueError("error_profile must have at least one dimension.")
    if valid is None:
        valid_mask = torch.ones_like(error_profile, dtype=torch.bool)
    else:
        if valid.shape != error_profile.shape:
            raise ValueError(
                "valid must match error_profile, got "
                f"{tuple(valid.shape)} and {tuple(error_profile.shape)}."
            )
        valid_mask = valid.to(device=error_profile.device, dtype=torch.bool)
    finite = torch.isfinite(error_profile)
    valid_detached = valid_mask.detach() & finite
    clean = torch.where(
        valid_detached,
        error_profile.detach(),
        torch.zeros_like(error_profile),
    )
    point_count = int(clean.shape[-1])
    flat_error = clean.reshape(-1, point_count)
    flat_valid = valid_detached.reshape(-1, point_count)
    valid_float = flat_valid.to(dtype=flat_error.dtype)
    return flat_error, flat_valid, valid_float, tuple(error_profile.shape)

def _validate_region_detector_parameters(
    *,
    min_prominence: float,
    temperature: float,
    smooth_kernel: int,
    edge_weight: float,
    min_slope_fraction: float,
    min_area_fraction: float,
    min_region_width: int,
) -> tuple[float, float, int, float, float, float, int]:
    prominence_threshold = float(min_prominence)
    activation_temperature = float(temperature)
    kernel_size = int(smooth_kernel)
    edge = float(edge_weight)
    slope_fraction = float(min_slope_fraction)
    area_fraction = float(min_area_fraction)
    region_width = int(min_region_width)
    if not math.isfinite(prominence_threshold) or prominence_threshold <= 0.0:
        raise ValueError(
            "min_prominence must be finite and > 0, "
            f"got {min_prominence}."
        )
    if not math.isfinite(activation_temperature) or activation_temperature <= 0.0:
        raise ValueError(
            f"temperature must be finite and > 0, got {temperature}."
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"smooth_kernel must be a positive odd integer, got {smooth_kernel}."
        )
    if not math.isfinite(edge) or not 0.0 <= edge <= 1.0:
        raise ValueError(
            f"edge_weight must be finite and in [0,1], got {edge_weight}."
        )
    if not math.isfinite(slope_fraction) or not 0.5 < slope_fraction <= 1.0:
        raise ValueError(
            "min_slope_fraction must be finite and in (0.5,1], "
            f"got {min_slope_fraction}."
        )
    if not math.isfinite(area_fraction) or area_fraction <= 0.0:
        raise ValueError(
            "min_area_fraction must be finite and > 0, "
            f"got {min_area_fraction}."
        )
    if region_width < 3:
        raise ValueError(
            f"min_region_width must be at least 3, got {min_region_width}."
        )
    return (
        prominence_threshold,
        activation_temperature,
        kernel_size,
        edge,
        slope_fraction,
        area_fraction,
        region_width,
    )

def _soft_threshold(
    value: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    positive_scale = (value / float(threshold)).clamp(min=0.0, max=1.0)
    return (
        torch.sigmoid((value - float(threshold)) / float(temperature))
        * positive_scale
    )

def _rise_fall_regions_from_smoothed(
    smoothed: torch.Tensor,
    flat_valid: torch.Tensor,
    *,
    width_pairs: tuple[tuple[int, int], ...],
    min_prominence: float,
    temperature: float,
    edge_weight: float,
    min_slope_fraction: float,
    min_area_fraction: float,
    min_region_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find sustained rise-then-fall segments and score their full support."""

    point_count = int(smoothed.shape[-1])
    region = torch.zeros_like(smoothed)
    peak_activation = smoothed.new_zeros(())
    area_threshold = float(min_prominence) * float(min_area_fraction)

    for left_width, right_width in width_pairs:
        segment_size = left_width + right_width + 1
        if (
            segment_size > point_count
            or segment_size < int(min_region_width)
        ):
            continue
        segments = smoothed.unfold(-1, segment_size, 1)
        segment_valid = flat_valid.unfold(-1, segment_size, 1).all(dim=-1)
        centre = segments[..., left_width]
        rise = centre - segments[..., 0]
        fall = centre - segments[..., -1]
        prominence = torch.minimum(rise, fall)
        prominence_activation = _soft_threshold(
            prominence,
            threshold=min_prominence,
            temperature=temperature,
        )

        differences = segments[..., 1:] - segments[..., :-1]
        rising_fraction = (
            differences[..., :left_width] > 0.0
        ).to(dtype=smoothed.dtype).mean(dim=-1)
        falling_fraction = (
            differences[..., left_width:] < 0.0
        ).to(dtype=smoothed.dtype).mean(dim=-1)
        denominator = max(1.0 - float(min_slope_fraction), 1e-6)
        rising_score = (
            (rising_fraction - float(min_slope_fraction)) / denominator
        ).clamp(min=0.0, max=1.0)
        falling_score = (
            (falling_fraction - float(min_slope_fraction)) / denominator
        ).clamp(min=0.0, max=1.0)
        slope_score = torch.minimum(rising_score, falling_score)

        interpolation = torch.linspace(
            0.0,
            1.0,
            segment_size,
            device=smoothed.device,
            dtype=smoothed.dtype,
        )
        baseline = (
            segments[..., :1] * (1.0 - interpolation)
            + segments[..., -1:] * interpolation
        )
        residual = (segments - baseline).clamp_min(0.0)
        area_mean = residual.mean(dim=-1)
        area_activation = _soft_threshold(
            area_mean,
            threshold=area_threshold,
            temperature=max(float(temperature) * float(min_area_fraction), 1e-6),
        )
        activation = (
            prominence_activation * slope_score * area_activation
        )
        activation = torch.where(
            segment_valid & (prominence > 0.0),
            activation,
            torch.zeros_like(activation),
        )
        peak_activation = torch.maximum(peak_activation, activation.max())

        residual_peak = residual.amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(smoothed.dtype).eps
        )
        residual_shape = (residual / residual_peak).clamp(0.0, 1.0)
        regional_shape = (
            float(edge_weight)
            + (1.0 - float(edge_weight)) * residual_shape
        )
        support = activation.unsqueeze(-1) * regional_shape
        candidate_count = int(support.shape[-2])
        for offset in range(segment_size):
            region[:, offset : offset + candidate_count] = torch.maximum(
                region[:, offset : offset + candidate_count],
                support[..., offset],
            )

    return region.clamp(0.0, 1.0), peak_activation

def detached_rise_fall_mountain_regions(
    error_profile: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    width_pairs: Any = DEFAULT_MOUNTAIN_WIDTH_PAIRS,
    min_prominence: float = 1.0,
    temperature: float = 0.25,
    smooth_kernel: int = 5,
    edge_weight: float = 0.25,
    min_slope_fraction: float = 0.65,
    min_area_fraction: float = 0.25,
    min_region_width: int = 7,
) -> dict[str, torch.Tensor]:
    """Detect sustained asymmetric rise-then-fall regions at explicit widths."""

    (
        prominence_threshold,
        activation_temperature,
        kernel_size,
        edge,
        slope_fraction,
        area_fraction,
        region_width,
    ) = _validate_region_detector_parameters(
        min_prominence=min_prominence,
        temperature=temperature,
        smooth_kernel=smooth_kernel,
        edge_weight=edge_weight,
        min_slope_fraction=min_slope_fraction,
        min_area_fraction=min_area_fraction,
        min_region_width=min_region_width,
    )
    parsed_widths = parse_mountain_width_pairs(width_pairs)
    with torch.no_grad():
        flat_error, flat_valid, valid_float, original_shape = (
            _prepare_detached_profiles(error_profile, valid)
        )
        smoothed = _smooth_flat_profile(
            flat_error,
            valid_float,
            kernel_size,
        )
        region, peak_activation = _rise_fall_regions_from_smoothed(
            smoothed,
            flat_valid,
            width_pairs=parsed_widths,
            min_prominence=prominence_threshold,
            temperature=activation_temperature,
            edge_weight=edge,
            min_slope_fraction=slope_fraction,
            min_area_fraction=area_fraction,
            min_region_width=region_width,
        )
        region = region * valid_float
        valid_count = valid_float.sum().clamp_min(1.0)
        region_fraction = region.sum() / valid_count
        region_mask = region.reshape(original_shape).detach()
    return {
        "region_mask": region_mask,
        "region_fraction": region_fraction.detach(),
        "peak_activation": peak_activation.detach(),
    }

def detached_sizer_mountain_regions(
    error_profile: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    scale_fractions: Any = DEFAULT_MOUNTAIN_SCALE_FRACTIONS,
    min_prominence: float = 1.0,
    temperature: float = 0.25,
    edge_weight: float = 0.25,
    min_slope_fraction: float = 0.65,
    min_area_fraction: float = 0.25,
    min_region_width: int = 7,
    min_scale_persistence: int = 2,
) -> dict[str, torch.Tensor]:
    """Detect rise-fall regions that persist across relative smoothing scales.

    Side widths are generated from fractions of the ordered profile length.
    Each scale uses Gaussian smoothing and symmetric plus 3:2 asymmetric
    supports. A point is retained only when at least ``min_scale_persistence``
    scale levels agree that it belongs to a mountain region.
    """

    persistence = int(min_scale_persistence)
    if persistence < 1:
        raise ValueError(
            "min_scale_persistence must be at least 1, "
            f"got {min_scale_persistence}."
        )
    (
        prominence_threshold,
        activation_temperature,
        _,
        edge,
        slope_fraction,
        area_fraction,
        region_width,
    ) = _validate_region_detector_parameters(
        min_prominence=min_prominence,
        temperature=temperature,
        smooth_kernel=1,
        edge_weight=edge_weight,
        min_slope_fraction=min_slope_fraction,
        min_area_fraction=min_area_fraction,
        min_region_width=min_region_width,
    )
    fractions = parse_mountain_scale_fractions(scale_fractions)

    with torch.no_grad():
        flat_error, flat_valid, valid_float, original_shape = (
            _prepare_detached_profiles(error_profile, valid)
        )
        point_count = int(flat_error.shape[-1])
        minimum_side = max(2, int(math.ceil((region_width - 1) / 2.0)))
        side_widths: list[int] = []
        for fraction in fractions:
            width = max(minimum_side, int(round(fraction * point_count)))
            width = min(width, max((point_count - 1) // 2, 1))
            if width not in side_widths:
                side_widths.append(width)

        scale_regions: list[torch.Tensor] = []
        scale_peaks: list[torch.Tensor] = []
        for width in side_widths:
            gaussian_radius = max(1, int(round(width / 2.0)))
            gaussian_kernel = 2 * gaussian_radius + 1
            smoothed = _smooth_flat_profile(
                flat_error,
                valid_float,
                gaussian_kernel,
                gaussian=True,
            )
            asymmetric_width = max(width + 1, int(round(1.5 * width)))
            scale_width_pairs = (
                (width, width),
                (width, asymmetric_width),
                (asymmetric_width, width),
            )
            scale_region, scale_peak = _rise_fall_regions_from_smoothed(
                smoothed,
                flat_valid,
                width_pairs=scale_width_pairs,
                min_prominence=prominence_threshold,
                temperature=activation_temperature,
                edge_weight=edge,
                min_slope_fraction=slope_fraction,
                min_area_fraction=area_fraction,
                min_region_width=region_width,
            )
            scale_regions.append(scale_region)
            scale_peaks.append(scale_peak)

        if scale_regions:
            stacked = torch.stack(scale_regions, dim=0)
            support_count = (stacked > 0.0).sum(dim=0)
            persistent = support_count >= persistence
            region = torch.where(
                persistent,
                stacked.amax(dim=0),
                torch.zeros_like(stacked[0]),
            )
            peak_activation = region.max()
        else:
            region = torch.zeros_like(flat_error)
            peak_activation = flat_error.new_zeros(())
        region = region.clamp(0.0, 1.0) * valid_float
        valid_count = valid_float.sum().clamp_min(1.0)
        region_fraction = region.sum() / valid_count
        region_mask = region.reshape(original_shape).detach()
    return {
        "region_mask": region_mask,
        "region_fraction": region_fraction.detach(),
        "peak_activation": peak_activation.detach(),
    }

def detached_mountain_regions(
    error_profile: torch.Tensor,
    *,
    detector: str = "local_prominence",
    valid: torch.Tensor | None = None,
    width_pairs: Any = DEFAULT_MOUNTAIN_WIDTH_PAIRS,
    scale_fractions: Any = DEFAULT_MOUNTAIN_SCALE_FRACTIONS,
    min_prominence: float = 1.0,
    temperature: float = 0.25,
    smooth_kernel: int = 5,
    edge_weight: float = 0.25,
    min_slope_fraction: float = 0.65,
    min_area_fraction: float = 0.25,
    min_region_width: int = 7,
    min_scale_persistence: int = 2,
) -> dict[str, torch.Tensor]:
    """Dispatch to one of the named detached mountain-region detectors."""

    mode = normalize_mountain_detector(detector)
    if mode == "local_prominence":
        return detached_asymmetric_mountain_regions(
            error_profile,
            valid=valid,
            width_pairs=width_pairs,
            min_prominence=min_prominence,
            temperature=temperature,
            smooth_kernel=smooth_kernel,
            edge_weight=edge_weight,
        )
    if mode == "rise_fall":
        return detached_rise_fall_mountain_regions(
            error_profile,
            valid=valid,
            width_pairs=width_pairs,
            min_prominence=min_prominence,
            temperature=temperature,
            smooth_kernel=smooth_kernel,
            edge_weight=edge_weight,
            min_slope_fraction=min_slope_fraction,
            min_area_fraction=min_area_fraction,
            min_region_width=min_region_width,
        )
    return detached_sizer_mountain_regions(
        error_profile,
        valid=valid,
        scale_fractions=scale_fractions,
        min_prominence=min_prominence,
        temperature=temperature,
        edge_weight=edge_weight,
        min_slope_fraction=min_slope_fraction,
        min_area_fraction=min_area_fraction,
        min_region_width=min_region_width,
        min_scale_persistence=min_scale_persistence,
    )

def detached_mountain_weighted_mean(
    point_loss: torch.Tensor,
    *,
    error_profile: torch.Tensor,
    valid: torch.Tensor,
    weight_boost: float,
    detector: str = "local_prominence",
    width_pairs: Any = DEFAULT_MOUNTAIN_WIDTH_PAIRS,
    scale_fractions: Any = DEFAULT_MOUNTAIN_SCALE_FRACTIONS,
    min_prominence: float = 1.0,
    temperature: float = 0.25,
    smooth_kernel: int = 5,
    edge_weight: float = 0.25,
    min_slope_fraction: float = 0.65,
    min_area_fraction: float = 0.25,
    min_region_width: int = 7,
    min_scale_persistence: int = 2,
) -> dict[str, torch.Tensor]:
    """Apply detached mountain-region weights to a differentiable point loss."""

    if point_loss.shape != error_profile.shape or valid.shape != point_loss.shape:
        raise ValueError(
            "point_loss, error_profile, and valid must have matching shapes, got "
            f"{tuple(point_loss.shape)}, {tuple(error_profile.shape)}, and "
            f"{tuple(valid.shape)}."
        )
    boost = float(weight_boost)
    if not math.isfinite(boost) or boost < 0.0:
        raise ValueError(
            f"weight_boost must be finite and non-negative, got {weight_boost}."
        )
    valid_mask = valid.to(device=point_loss.device, dtype=torch.bool)
    valid_float = valid_mask.to(dtype=point_loss.dtype)
    safe_point_loss = torch.where(
        valid_mask & torch.isfinite(point_loss),
        point_loss,
        torch.zeros_like(point_loss),
    )
    valid_count = valid_float.sum().clamp_min(1.0)
    unweighted_loss = (safe_point_loss * valid_float).sum() / valid_count

    if boost > 0.0:
        detection = detached_mountain_regions(
            error_profile,
            detector=detector,
            valid=valid_mask,
            width_pairs=width_pairs,
            scale_fractions=scale_fractions,
            min_prominence=min_prominence,
            temperature=temperature,
            smooth_kernel=smooth_kernel,
            edge_weight=edge_weight,
            min_slope_fraction=min_slope_fraction,
            min_area_fraction=min_area_fraction,
            min_region_width=min_region_width,
            min_scale_persistence=min_scale_persistence,
        )
        region_mask = detection["region_mask"]
        peak_activation = detection["peak_activation"]
        region_fraction = detection["region_fraction"]
    else:
        region_mask = torch.zeros_like(point_loss).detach()
        peak_activation = point_loss.new_zeros(())
        region_fraction = point_loss.new_zeros(())

    weights = valid_float * (1.0 + boost * region_mask)
    weighted_loss = (safe_point_loss * weights).sum() / weights.sum().clamp_min(1.0)
    mean_weight = weights.sum() / valid_count
    return {
        "loss": weighted_loss,
        "unweighted_loss": unweighted_loss,
        "region_mask": region_mask,
        "region_fraction": region_fraction,
        "peak_activation": peak_activation,
        "mean_weight": mean_weight.detach(),
    }

__all__ = [
    "DEFAULT_MOUNTAIN_SCALE_FRACTIONS",
    "DEFAULT_MOUNTAIN_WIDTH_PAIRS",
    "MOUNTAIN_DETECTORS",
    "detached_asymmetric_mountain_regions",
    "detached_mountain_regions",
    "detached_mountain_weighted_mean",
    "detached_rise_fall_mountain_regions",
    "detached_sizer_mountain_regions",
    "normalize_mountain_detector",
    "parse_mountain_scale_fractions",
    "parse_mountain_width_pairs",
]
