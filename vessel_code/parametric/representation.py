# Transferred from methods/parametric_methods/representation.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import torch

LESION_PROFILES = ("gaussian", "asymmetric_segment")

RADIUS_PREDICTION_MODES = ("parametric", "raw")

CENTERLINE_PREDICTION_MODES = ("bspline_control_points", "adaptive_landmarks")

def normalize_centerline_prediction_mode(value: str | None) -> str:
    mode = (
        "bspline_control_points"
        if value is None
        else str(value).strip().lower().replace("-", "_")
    )
    aliases = {
        "bspline": "bspline_control_points",
        "b_spline": "bspline_control_points",
        "control_points": "bspline_control_points",
        "landmarks": "adaptive_landmarks",
        "catmull_rom": "adaptive_landmarks",
    }
    mode = aliases.get(mode, mode)
    if mode not in CENTERLINE_PREDICTION_MODES:
        raise ValueError(
            f"centerline_prediction_mode must be one of "
            f"{CENTERLINE_PREDICTION_MODES}, got {value!r}"
        )
    return mode

def normalize_radius_prediction_mode(value: str | None) -> str:
    mode = "parametric" if value is None else str(value).strip().lower().replace("-", "_")
    if mode not in RADIUS_PREDICTION_MODES:
        raise ValueError(
            f"radius_prediction_mode must be one of {RADIUS_PREDICTION_MODES}, "
            f"got {value!r}"
        )
    return mode

def normalize_lesion_profile(value: str) -> str:
    profile = str(value).strip().lower().replace("-", "_")
    if profile not in LESION_PROFILES:
        raise ValueError(f"lesion_profile must be one of {LESION_PROFILES}, got {value!r}")
    return profile

def constrain_lesion_geometry(raw: torch.Tensor, lesion_profile: str) -> torch.Tensor:
    """Map unconstrained head outputs to valid normalized lesion geometry."""
    profile = normalize_lesion_profile(lesion_profile)
    if profile == "gaussian":
        if raw.shape[-1] != 3:
            raise ValueError(f"Gaussian geometry must end in 3 values, got {tuple(raw.shape)}")
        position = torch.sigmoid(raw[..., 0])
        sigma = 0.005 + 0.295 * torch.sigmoid(raw[..., 1])
        severity = 0.999 * torch.sigmoid(raw[..., 2])
        return torch.stack((position, sigma, severity), dim=-1)

    if raw.shape[-1] != 5:
        raise ValueError(f"Asymmetric geometry must end in 5 values, got {tuple(raw.shape)}")
    start = torch.sigmoid(raw[..., 0])
    end = start + (1.0 - start) * torch.sigmoid(raw[..., 1])
    entry_tau = 0.001 + 0.299 * torch.sigmoid(raw[..., 2])
    exit_tau = 0.001 + 0.299 * torch.sigmoid(raw[..., 3])
    severity = 0.999 * torch.sigmoid(raw[..., 4])
    return torch.stack((start, end, entry_tau, exit_tau, severity), dim=-1)

def decode_centerlines(
    control_points_mm: torch.Tensor,
    attachment_logits: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode global main and parent-attachment-relative side centrelines."""
    centerlines_local = decode_bspline_centerlines_local(control_points_mm, basis)
    return assemble_relative_centerlines(centerlines_local, attachment_logits)

def decode_bspline_centerlines_local(
    control_points_mm: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Decode compact B-spline parameters without applying branch translation."""
    if control_points_mm.dim() != 4 or control_points_mm.shape[-1] != 3:
        raise ValueError(f"control_points_mm must be [B,M,K,3], got {tuple(control_points_mm.shape)}")
    return torch.einsum("nk,bmkd->bmnd", basis, control_points_mm)

def assemble_relative_centerlines(
    centerlines_local: torch.Tensor,
    attachment_logits: torch.Tensor,
    *,
    hard_attachment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach local side curves to a decoded main curve."""
    if centerlines_local.dim() != 4 or centerlines_local.shape[-1] != 3:
        raise ValueError(
            f"centerlines_local must be [B,M,N,3], got {tuple(centerlines_local.shape)}"
        )
    if centerlines_local.shape[1] == 1:
        probabilities = torch.ones(
            (*centerlines_local.shape[:2], centerlines_local.shape[2]),
            device=centerlines_local.device,
            dtype=centerlines_local.dtype,
        )
        return centerlines_local, probabilities
    if attachment_logits.shape != centerlines_local.shape[:3]:
        raise ValueError(
            f"attachment_logits must be [B,M,N]={tuple(centerlines_local.shape[:3])}, "
            f"got {tuple(attachment_logits.shape)}"
        )
    probabilities = torch.softmax(attachment_logits, dim=-1)
    main = centerlines_local[:, 0]
    if hard_attachment:
        attachment_indices = probabilities[:, 1:].argmax(dim=-1)
        side_origins = torch.gather(
            main,
            1,
            attachment_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
    else:
        side_origins = torch.einsum(
            "bmn,bnd->bmd", probabilities[:, 1:], main
        )
    decoded = centerlines_local.clone()
    side_relative = centerlines_local[:, 1:] - centerlines_local[:, 1:, :1]
    decoded[:, 1:] = side_relative + side_origins.unsqueeze(2)
    return decoded, probabilities

def _batched_catmull_rom_dense(
    landmarks_mm: torch.Tensor,
    *,
    alpha: float,
    dense_samples: int,
) -> torch.Tensor:
    """Evaluate a centripetal Catmull--Rom curve for every flattened branch.

    The segment identity is determined by the ordered landmark index. Chord-based
    knots control the local curve shape, matching the offline adaptive transform.
    """
    if landmarks_mm.dim() != 3 or landmarks_mm.shape[-1] != 3:
        raise ValueError(
            f"landmarks_mm must be [Q,K,3], got {tuple(landmarks_mm.shape)}"
        )
    num_curves, num_landmarks, _ = landmarks_mm.shape
    if num_landmarks < 4:
        raise ValueError("Catmull--Rom decoding requires at least four landmarks")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"catmull_rom_alpha must be in [0,1], got {alpha}")
    sample_count = max(int(dense_samples), int(num_landmarks) * 50)
    if sample_count < 2:
        raise ValueError("dense_samples must be at least two")

    chord_lengths = torch.linalg.vector_norm(
        landmarks_mm[:, 1:] - landmarks_mm[:, :-1], dim=-1
    )
    if float(alpha) == 0.0:
        intervals = torch.ones_like(chord_lengths)
    else:
        # Clamp before a fractional power so coincident predicted landmarks do
        # not create an infinite derivative at zero during early training.
        intervals = chord_lengths.clamp_min(1e-16).pow(float(alpha))
    intervals = intervals.clamp_min(1e-8)
    knots = torch.cat(
        (
            intervals.new_zeros((num_curves, 1)),
            torch.cumsum(intervals, dim=-1),
        ),
        dim=-1,
    )
    knots = knots / knots[:, -1:].clamp_min(1e-12)

    semantic_query = torch.linspace(
        0.0,
        1.0,
        sample_count,
        device=landmarks_mm.device,
        dtype=landmarks_mm.dtype,
    )
    scaled_query = semantic_query * float(num_landmarks - 1)
    segment = torch.floor(scaled_query).to(dtype=torch.long)
    segment = segment.clamp(min=0, max=num_landmarks - 2)
    fraction = (scaled_query - segment.to(dtype=scaled_query.dtype)).clamp(0.0, 1.0)
    curve_index = torch.arange(
        num_curves, device=landmarks_mm.device, dtype=torch.long
    ).view(-1, 1)
    segment_batch = segment.view(1, -1).expand(num_curves, -1)

    p1 = landmarks_mm[curve_index, segment_batch]
    p2 = landmarks_mm[curve_index, segment_batch + 1]
    previous_index = (segment_batch - 1).clamp_min(0)
    next_index = (segment_batch + 2).clamp_max(num_landmarks - 1)
    p0_stored = landmarks_mm[curve_index, previous_index]
    p3_stored = landmarks_mm[curve_index, next_index]
    start_segment = segment_batch == 0
    end_segment = segment_batch == num_landmarks - 2
    p0 = torch.where(start_segment.unsqueeze(-1), 2.0 * p1 - p2, p0_stored)
    p3 = torch.where(end_segment.unsqueeze(-1), 2.0 * p2 - p1, p3_stored)

    t1 = knots[curve_index, segment_batch]
    t2 = knots[curve_index, segment_batch + 1]
    t0_stored = knots[curve_index, previous_index]
    t3_stored = knots[curve_index, next_index]
    t0 = torch.where(start_segment, t1 - (t2 - t1), t0_stored)
    t3 = torch.where(end_segment, t2 + (t2 - t1), t3_stored)
    query = t1 + fraction.view(1, -1) * (t2 - t1)

    def blend(
        left: torch.Tensor,
        right: torch.Tensor,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
    ) -> torch.Tensor:
        denominator = (right_t - left_t).clamp_min(1e-12)
        left_weight = ((right_t - query) / denominator).unsqueeze(-1)
        right_weight = ((query - left_t) / denominator).unsqueeze(-1)
        return left_weight * left + right_weight * right

    a1 = blend(p0, p1, t0, t1)
    a2 = blend(p1, p2, t1, t2)
    a3 = blend(p2, p3, t2, t3)
    b1 = blend(a1, a2, t0, t2)
    b2 = blend(a2, a3, t1, t3)
    return blend(b1, b2, t1, t2)

def _resample_curves_uniform_arc(
    dense_curves: torch.Tensor, num_points: int
) -> torch.Tensor:
    if dense_curves.dim() != 3 or dense_curves.shape[-1] != 3:
        raise ValueError(
            f"dense_curves must be [Q,D,3], got {tuple(dense_curves.shape)}"
        )
    if int(num_points) < 2:
        raise ValueError("num_points must be at least two")
    segment_lengths = torch.linalg.vector_norm(
        dense_curves[:, 1:] - dense_curves[:, :-1], dim=-1
    )
    cumulative = torch.cat(
        (
            segment_lengths.new_zeros((segment_lengths.shape[0], 1)),
            torch.cumsum(segment_lengths, dim=-1),
        ),
        dim=-1,
    )
    total_length = cumulative[:, -1:]
    arc_t = cumulative / total_length.clamp_min(1e-12)
    query = torch.linspace(
        0.0,
        1.0,
        int(num_points),
        device=dense_curves.device,
        dtype=dense_curves.dtype,
    ).view(1, -1).expand(dense_curves.shape[0], -1)
    right = torch.searchsorted(arc_t.contiguous(), query.contiguous(), right=True)
    left = (right - 1).clamp(min=0, max=dense_curves.shape[1] - 2)
    right = left + 1
    curve_index = torch.arange(
        dense_curves.shape[0], device=dense_curves.device
    ).view(-1, 1)
    left_t = arc_t[curve_index, left]
    right_t = arc_t[curve_index, right]
    fraction = ((query - left_t) / (right_t - left_t).clamp_min(1e-12)).clamp(
        0.0, 1.0
    )
    left_xyz = dense_curves[curve_index, left]
    right_xyz = dense_curves[curve_index, right]
    decoded = left_xyz + fraction.unsqueeze(-1) * (right_xyz - left_xyz)
    degenerate = total_length.squeeze(-1) <= 1e-12
    if bool(degenerate.any()):
        decoded = torch.where(
            degenerate.view(-1, 1, 1),
            dense_curves[:, :1].expand(-1, int(num_points), -1),
            decoded,
        )
    return decoded

def decode_landmark_centerlines(
    landmarks_mm: torch.Tensor,
    attachment_logits: torch.Tensor,
    *,
    num_points: int,
    catmull_rom_alpha: float = 0.5,
    dense_samples: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode on-curve landmarks and attach relative side branches.

    Main-branch landmarks are global. Side-branch landmarks are local to their
    first point and are translated to a soft attachment on the decoded main branch.
    """
    centerlines_local = decode_landmark_centerlines_local(
        landmarks_mm,
        num_points=num_points,
        catmull_rom_alpha=catmull_rom_alpha,
        dense_samples=dense_samples,
    )
    return assemble_relative_centerlines(centerlines_local, attachment_logits)

def decode_landmark_centerlines_local(
    landmarks_mm: torch.Tensor,
    *,
    num_points: int,
    catmull_rom_alpha: float = 0.5,
    dense_samples: int = 1000,
) -> torch.Tensor:
    """Decode Catmull-Rom landmark parameters without branch translation."""
    if landmarks_mm.dim() != 4 or landmarks_mm.shape[-1] != 3:
        raise ValueError(
            f"landmarks_mm must be [B,M,K,3], got {tuple(landmarks_mm.shape)}"
        )
    batch_size, num_branches, num_landmarks, _ = landmarks_mm.shape
    dense = _batched_catmull_rom_dense(
        landmarks_mm.reshape(batch_size * num_branches, num_landmarks, 3),
        alpha=float(catmull_rom_alpha),
        dense_samples=int(dense_samples),
    )
    centerlines_local = _resample_curves_uniform_arc(
        dense, int(num_points)
    ).reshape(batch_size, num_branches, int(num_points), 3)
    return centerlines_local

def _gaussian_deficit(
    t: torch.Tensor,
    existence_probability: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    position, sigma, severity = geometry.unbind(dim=-1)
    log_depth = -torch.log1p(-severity.clamp(max=1.0 - 1e-6))
    profile = torch.exp(-0.5 * ((t - position.unsqueeze(-1)) / sigma.unsqueeze(-1).clamp_min(1e-6)) ** 2)
    return (existence_probability * log_depth).unsqueeze(-1) * profile

def _asymmetric_deficit(
    t: torch.Tensor,
    existence_probability: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    start, end, entry_tau, exit_tau, severity = geometry.unbind(dim=-1)
    entry = torch.sigmoid((t - start.unsqueeze(-1)) / entry_tau.unsqueeze(-1).clamp_min(1e-6))
    exit_profile = torch.sigmoid((end.unsqueeze(-1) - t) / exit_tau.unsqueeze(-1).clamp_min(1e-6))
    profile = entry * exit_profile
    profile = profile / profile.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    log_depth = -torch.log1p(-severity.clamp(max=1.0 - 1e-6))
    return (existence_probability * log_depth).unsqueeze(-1) * profile

def decode_radii(
    baseline_coefficients_log_mm: torch.Tensor,
    lesion_exist_logits: torch.Tensor,
    lesion_geometry: torch.Tensor,
    basis: torch.Tensor,
    lesion_profile: str,
) -> torch.Tensor:
    profile = normalize_lesion_profile(lesion_profile)
    baseline_log_radius = torch.einsum("nk,bmk->bmn", basis, baseline_coefficients_log_mm)
    existence_probability = torch.sigmoid(lesion_exist_logits)
    t = torch.linspace(
        0.0,
        1.0,
        basis.shape[0],
        device=basis.device,
        dtype=basis.dtype,
    ).view(1, 1, 1, -1)
    if lesion_geometry.shape[2] == 0:
        total_deficit = baseline_log_radius.new_zeros(baseline_log_radius.shape)
    elif profile == "gaussian":
        total_deficit = _gaussian_deficit(t, existence_probability, lesion_geometry).sum(dim=2)
    else:
        total_deficit = _asymmetric_deficit(t, existence_probability, lesion_geometry).sum(dim=2)
    return torch.exp((baseline_log_radius - total_deficit).clamp(min=-8.0, max=5.0))

def decode_raw_radii(raw_radius_log_mm: torch.Tensor) -> torch.Tensor:
    """Convert pointwise log-radius predictions to positive radii in millimetres."""
    if raw_radius_log_mm.dim() != 3:
        raise ValueError(
            "raw_radius_log_mm must be [B,M,N], "
            f"got {tuple(raw_radius_log_mm.shape)}"
        )
    return torch.exp(raw_radius_log_mm.clamp(min=-8.0, max=5.0))

def decode_parametric_vessel(
    *,
    centerline_control_points_mm: torch.Tensor,
    attachment_logits: torch.Tensor,
    radius_baseline_coefficients_log_mm: torch.Tensor,
    lesion_exist_logits: torch.Tensor,
    lesion_geometry: torch.Tensor,
    centerline_basis: torch.Tensor,
    radius_basis: torch.Tensor,
    lesion_profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    centerlines, attachment_probabilities = decode_centerlines(
        centerline_control_points_mm, attachment_logits, centerline_basis
    )
    radius = decode_radii(
        radius_baseline_coefficients_log_mm,
        lesion_exist_logits,
        lesion_geometry,
        radius_basis,
        lesion_profile,
    )
    return torch.cat((centerlines, radius.unsqueeze(-1)), dim=-1), attachment_probabilities
