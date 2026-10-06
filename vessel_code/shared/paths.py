# Transferred from methods/src/paths.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from pathlib import Path
from typing import Any
from vessel_code.shared.data import normalize_feature_backbone

def backbone_experiment_group(feature_backbone: str) -> str:
    backbone = normalize_feature_backbone(feature_backbone)
    if backbone == "resnet_pre_fpn":
        return "ResNet_backbone_experiments"
    if backbone == "vggt":
        return "VGGT_backbone_experiments"
    if backbone == "vggt_omega":
        return "VGGT_Omega_backbone_experiments"
    raise ValueError(f"Unsupported feature_backbone={feature_backbone!r}")

def _experiment_name(config: dict[str, Any]) -> str:
    raw = config.get("experiment_name")
    if raw is None or str(raw).strip() == "":
        raise ValueError("Missing required config value experiment_name for automatic experiment directory creation.")
    name = str(raw).strip().replace("/", "_")
    if name in {".", ".."}:
        raise ValueError(f"Invalid experiment_name={raw!r}")
    return name

def configured_train_output_dir(config: dict[str, Any]) -> Path:
    """Return the configured output root before adding experiment subdirs.

    ``output_dir`` is the canonical training configuration key.  ``out_dir``
    remains a read-only compatibility alias for older SRC configurations, and
    ``output_root_dir`` preserves the original root in a resolved config.
    """

    raw = (
        config.get("output_root_dir")
        or config.get("output_dir")
        or config.get("out_dir")
        or "outputs/precomputed_multiview_train"
    )
    return Path(str(raw)).expanduser().resolve()

def resolve_train_experiment_dir(config: dict[str, Any], feature_backbone: str) -> Path:
    explicit_experiment_dir = config.get("experiment_dir")
    if explicit_experiment_dir not in {None, ""}:
        return Path(str(explicit_experiment_dir)).expanduser().resolve()

    base = configured_train_output_dir(config)
    experiment_name = _experiment_name(config)
    if not bool(config.get("auto_experiment_dir", True)):
        # Older configs sometimes stored the complete experiment directory in
        # output_dir/out_dir. Keep that representation idempotent while making
        # the canonical contract output_dir/<experiment_name>.
        if base.name == experiment_name:
            return base
        return (base / experiment_name).resolve()

    group = backbone_experiment_group(feature_backbone)
    if base.name == experiment_name and base.parent.name == group:
        return base.resolve()
    if base.name == group:
        return (base / experiment_name).resolve()
    return (base / group / experiment_name).resolve()

def resolve_train_output_config(
    config: dict[str, Any], feature_backbone: str
) -> Path:
    """Resolve and store the canonical training experiment-directory fields."""

    output_root_dir = configured_train_output_dir(config)
    experiment_dir = resolve_train_experiment_dir(config, feature_backbone)
    config["output_root_dir"] = str(output_root_dir)
    config["output_dir"] = str(experiment_dir)
    config["experiment_dir"] = str(experiment_dir)
    config["experiment_name"] = experiment_dir.name
    # Compatibility for older SRC checkpoint and evaluation readers.
    config["out_dir"] = str(experiment_dir)
    return experiment_dir
