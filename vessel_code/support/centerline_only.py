# Transferred from methods/centerline_only.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from typing import Any

_PIPELINE_RADIUS_LOSS_KEYS = {
    "src": (
        "radius_loss_weight",
        "occupancy_3d_ssim_loss_weight",
    ),
    "parametric": (
        "radius_coefficient_loss_weight",
        "decoded_radius_loss_weight",
        "lesion_exist_loss_weight",
        "lesion_geometry_loss_weight",
    ),
}

_SURFACE_PROJECTION_WEIGHT_KEYS = (
    "proj_dice_weight",
    "proj_geometry_dice_weight",
    "proj_radius_dice_weight",
    "proj_joint_dice_weight",
    "proj_dt_ssim_weight",
)

def only_centerline_from_config(config: dict[str, Any]) -> bool:
    """Return the canonical centreline-only training flag."""
    value = config.get("only_centerline", False)
    if not isinstance(value, bool):
        raise ValueError(
            "only_centerline must be a JSON boolean (true or false), "
            f"got {value!r}."
        )
    return value

def configure_only_centerline_losses(
    config: dict[str, Any],
    *,
    pipeline: str,
    announce: bool = False,
) -> bool:
    """Force radius- and surface-supervision weights to zero when requested.

    The model architecture is intentionally unchanged: it can still emit a
    radius channel, but that channel contributes no supervised training signal.
    Original configured values are retained in resolved-config metadata.
    """
    pipeline_name = str(pipeline).strip().lower()
    if pipeline_name not in _PIPELINE_RADIUS_LOSS_KEYS:
        raise ValueError(
            f"Unsupported only-centerline pipeline {pipeline!r}; "
            f"expected one of {sorted(_PIPELINE_RADIUS_LOSS_KEYS)}."
        )

    enabled = only_centerline_from_config(config)
    config["only_centerline"] = enabled
    config["only_centerline_effective"] = enabled
    if not enabled:
        return False

    loss_config = config.setdefault("loss", {})
    if not isinstance(loss_config, dict):
        raise ValueError("loss must be a JSON object")

    stored_original = config.get("only_centerline_overridden_loss_weights", {})
    original_weights: dict[str, float] = (
        {
            str(key): float(value)
            for key, value in stored_original.items()
            if value is not None
        }
        if isinstance(stored_original, dict)
        else {}
    )

    def remember(key: str, value: Any) -> None:
        if key not in original_weights and value is not None:
            original_weights[key] = float(value)

    for key in _PIPELINE_RADIUS_LOSS_KEYS[pipeline_name]:
        if key in loss_config:
            remember(key, loss_config[key])
        elif key in config:
            remember(key, config[key])
        loss_config[key] = 0.0
        if key in config:
            config[key] = 0.0

    for key in _SURFACE_PROJECTION_WEIGHT_KEYS:
        if key in config:
            remember(key, config[key])
        config[key] = 0.0
    # Split Dice requires a positive component weight, so disable its routing
    # switch as well as zeroing every surface loss.
    config["proj_dice_gradient_split_enabled"] = False

    config["only_centerline_overridden_loss_weights"] = original_weights
    config["only_centerline_monitor_mode"] = "centerline_overlays"
    if announce:
        changed = [
            f"{key}={value:g}->0"
            for key, value in original_weights.items()
            if value != 0.0
        ]
        suffix = ", ".join(changed) if changed else "all relevant weights already zero"
        print(
            "Only-centreline mode enabled: radius/surface supervision disabled; "
            f"{suffix}."
        )
    return True
