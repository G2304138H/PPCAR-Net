# Transferred from methods/losses/error_thresholding.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Iterable
import torch

@dataclass(frozen=True)
class ErrorThresholdSettings:
    """Resolved settings for one configured per-element error threshold."""

    enabled: bool
    configured_threshold: float
    effective_threshold: float
    sigmoid_k: float
    schedule_enabled: bool
    start_epoch: int
    decay_every_epochs: int
    decay_factor: float

def _thresholding_config(config: dict[str, Any]) -> dict[str, Any]:
    loss_config = config.get("loss", {})
    if loss_config is None:
        loss_config = {}
    if not isinstance(loss_config, dict):
        raise ValueError("config['loss'] must be an object when provided.")
    raw = loss_config.get(
        "error_thresholding",
        config.get("error_thresholding", {}),
    )
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("error_thresholding must be an object keyed by loss name.")
    return raw

def _find_component_config(
    config: dict[str, Any],
    component_name: str,
    aliases: Iterable[str],
) -> tuple[str, dict[str, Any]] | None:
    thresholding = _thresholding_config(config)
    matches: list[tuple[str, dict[str, Any]]] = []
    for name in (component_name, *aliases):
        if name not in thresholding:
            continue
        raw = thresholding[name]
        if not isinstance(raw, dict):
            raise ValueError(
                f"error_thresholding.{name} must be an object, got {type(raw).__name__}."
            )
        matches.append((name, raw))
    if not matches:
        return None
    return next(
        (match for match in matches if bool(match[1].get("enabled", False))),
        matches[0],
    )

def resolve_error_threshold_settings(
    config: dict[str, Any],
    component_name: str,
    *,
    epoch: int | None,
    aliases: Iterable[str] = (),
) -> ErrorThresholdSettings:
    """Resolve one component's settings for a one-based epoch.

    A scheduled threshold is zero before ``start_epoch``. At ``start_epoch`` it
    jumps to ``threshold`` and is multiplied by ``decay_factor`` after every
    ``decay_every_epochs`` complete epochs.
    """

    found = _find_component_config(config, component_name, aliases)
    if found is None:
        return ErrorThresholdSettings(
            enabled=False,
            configured_threshold=0.0,
            effective_threshold=0.0,
            sigmoid_k=1.0,
            schedule_enabled=False,
            start_epoch=1,
            decay_every_epochs=1,
            decay_factor=1.0,
        )
    config_name, component = found
    enabled = bool(component.get("enabled", False))
    if not enabled:
        return ErrorThresholdSettings(
            enabled=False,
            configured_threshold=0.0,
            effective_threshold=0.0,
            sigmoid_k=1.0,
            schedule_enabled=False,
            start_epoch=1,
            decay_every_epochs=1,
            decay_factor=1.0,
        )
    configured_threshold = float(component.get("threshold", 0.0))
    sigmoid_k = float(component.get("sigmoid_k", 10.0))
    if not math.isfinite(configured_threshold) or configured_threshold < 0.0:
        raise ValueError(
            f"error_thresholding.{config_name}.threshold must be finite and >= 0."
        )
    if not math.isfinite(sigmoid_k) or sigmoid_k <= 0.0:
        raise ValueError(
            f"error_thresholding.{config_name}.sigmoid_k must be finite and > 0."
        )
    raw_schedule = component.get("schedule", {})
    if raw_schedule is None:
        raw_schedule = {}
    if not isinstance(raw_schedule, dict):
        raise ValueError(
            f"error_thresholding.{config_name}.schedule must be an object."
        )
    schedule_enabled = bool(raw_schedule.get("enabled", False))
    start_epoch = int(raw_schedule.get("start_epoch", 1))
    decay_every_epochs = int(raw_schedule.get("decay_every_epochs", 1))
    decay_factor = float(raw_schedule.get("decay_factor", 0.5))
    if start_epoch < 1:
        raise ValueError(
            f"error_thresholding.{config_name}.schedule.start_epoch must be >= 1."
        )
    if decay_every_epochs < 1:
        raise ValueError(
            "error_thresholding."
            f"{config_name}.schedule.decay_every_epochs must be >= 1."
        )
    if (
        not math.isfinite(decay_factor)
        or decay_factor <= 0.0
        or decay_factor > 1.0
    ):
        raise ValueError(
            "error_thresholding."
            f"{config_name}.schedule.decay_factor must be finite and in (0, 1]."
        )

    effective_threshold = configured_threshold
    if schedule_enabled:
        resolved_epoch = (
            int(epoch)
            if epoch is not None
            else int(config.get("num_epochs", start_epoch))
        )
        if resolved_epoch < 1:
            raise ValueError(f"epoch must be one-based and >= 1, got {epoch}.")
        if resolved_epoch < start_epoch:
            effective_threshold = 0.0
        else:
            completed_periods = (
                resolved_epoch - start_epoch
            ) // decay_every_epochs
            effective_threshold = configured_threshold * (
                decay_factor**completed_periods
            )

    return ErrorThresholdSettings(
        enabled=enabled,
        configured_threshold=configured_threshold,
        effective_threshold=effective_threshold,
        sigmoid_k=sigmoid_k,
        schedule_enabled=schedule_enabled,
        start_epoch=start_epoch,
        decay_every_epochs=decay_every_epochs,
        decay_factor=decay_factor,
    )

def error_thresholded_mean(
    errors: torch.Tensor,
    *,
    valid: torch.Tensor,
    threshold: float,
    sigmoid_k: float,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Reduce per-element errors using a soft estimate of the active count.

    For a positive threshold ``t``, this computes

    ``sum(w * relu(error - t)) /
    stopgrad(sum(w * sigmoid(k * (error - t))))``.

    An effective threshold of zero deliberately uses the ordinary weighted
    mean. This makes a scheduled zero-threshold warm-up exactly equivalent to
    training on all supplied errors instead of approximating the count with a
    sigmoid. The positive-threshold soft count is recomputed on every forward
    pass, then detached so it contributes no gradient.
    """

    if not torch.is_floating_point(errors):
        raise ValueError("errors must be a floating-point tensor.")
    if errors.shape != valid.shape:
        raise ValueError(
            "errors and valid must have matching shapes, got "
            f"{tuple(errors.shape)} and {tuple(valid.shape)}."
        )
    if sample_weights is not None and sample_weights.shape != errors.shape:
        raise ValueError(
            "sample_weights must match errors, got "
            f"{tuple(sample_weights.shape)} and {tuple(errors.shape)}."
        )
    threshold_value = float(threshold)
    k_value = float(sigmoid_k)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("threshold must be finite and >= 0.")
    if not math.isfinite(k_value) or k_value <= 0.0:
        raise ValueError("sigmoid_k must be finite and > 0.")
    valid_mask = valid.to(device=errors.device, dtype=torch.bool)
    valid_weight = valid_mask.to(dtype=errors.dtype)
    if sample_weights is not None:
        valid_weight = valid_weight * sample_weights.to(
            device=errors.device,
            dtype=errors.dtype,
        ).clamp_min(0.0)
    safe_errors = torch.where(valid_mask, errors, torch.zeros_like(errors))
    total_weight = valid_weight.sum()
    fraction_denominator = total_weight.clamp_min(
        torch.finfo(errors.dtype).eps
    )

    if threshold_value == 0.0:
        loss = (safe_errors * valid_weight).sum() / fraction_denominator
        soft_active_fraction = (total_weight > 0.0).to(dtype=errors.dtype)
        hard_exceedance_fraction = (
            ((safe_errors > 0.0).to(dtype=errors.dtype) * valid_weight).sum()
            / fraction_denominator
        )
    else:
        shifted = safe_errors - threshold_value
        numerator = (
            torch.relu(shifted) * valid_weight
        ).sum()
        soft_count = (
            torch.sigmoid(k_value * shifted) * valid_weight
        ).sum()
        denominator = soft_count.detach().clamp_min(1.0)
        loss = numerator / denominator
        soft_active_fraction = (
            soft_count.detach() / fraction_denominator.detach()
        )
        hard_exceedance_fraction = (
            (
                (safe_errors > threshold_value).to(dtype=errors.dtype)
                * valid_weight
            ).sum()
            / fraction_denominator
        )

    return {
        "loss": loss,
        "effective_threshold": errors.new_tensor(threshold_value),
        "soft_active_fraction": soft_active_fraction.detach(),
        "hard_exceedance_fraction": hard_exceedance_fraction.detach(),
    }

def configured_error_thresholded_mean(
    errors: torch.Tensor,
    *,
    valid: torch.Tensor,
    config: dict[str, Any],
    component_name: str,
    epoch: int | None,
    aliases: Iterable[str] = (),
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Apply ``error_thresholded_mean`` when the named component is enabled."""

    settings = resolve_error_threshold_settings(
        config,
        component_name,
        epoch=epoch,
        aliases=aliases,
    )
    if not settings.enabled:
        return None
    return error_thresholded_mean(
        errors,
        valid=valid,
        threshold=settings.effective_threshold,
        sigmoid_k=settings.sigmoid_k,
        sample_weights=sample_weights,
    )

def configured_error_threshold_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Return enabled component names for startup validation and logging."""

    names: list[str] = []
    for name, raw in _thresholding_config(config).items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"error_thresholding.{name} must be an object, got {type(raw).__name__}."
            )
        if bool(raw.get("enabled", False)):
            names.append(str(name))
    return tuple(names)

__all__ = [
    "ErrorThresholdSettings",
    "configured_error_threshold_names",
    "configured_error_thresholded_mean",
    "error_thresholded_mean",
    "resolve_error_threshold_settings",
]
