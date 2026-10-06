# Transferred from methods/parametric_methods/centerline_probability.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping
import torch
import torch.nn as nn
from vessel_code.parametric.centerline_heatmap import GAUSSIAN_CENTERLINE_TARGET, MASK_NORMALIZED_CENTERLINE_DISTANCE_TARGET, normalize_centerline_target_mode

CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMA_VERSION = 2

SUPPORTED_CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMAS = frozenset((1, 2))

def centerline_probability_target_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the checkpointed semantic contract for the dense target."""

    model = centerline_probability_model_config(config)
    mode = normalize_centerline_target_mode(
        config.get("target_mode", GAUSSIAN_CENTERLINE_TARGET)
    )
    output: dict[str, Any] = {
        "mode": mode,
        "centerline_map_size": int(model["centerline_map_size"]),
        "mask_threshold": float(config.get("target_mask_threshold", 0.5)),
    }
    if (
        not math.isfinite(output["mask_threshold"])
        or not 0.0 <= output["mask_threshold"] <= 1.0
    ):
        raise ValueError(
            "target_mask_threshold must be finite and in [0,1], got "
            f"{output['mask_threshold']}."
        )
    if mode == GAUSSIAN_CENTERLINE_TARGET:
        output.update(
            {
                "sigma_px": float(config.get("target_sigma_px", 1.25)),
                "radius_px": int(config.get("target_radius_px", 2)),
            }
        )
        if not math.isfinite(output["sigma_px"]) or output["sigma_px"] <= 0.0:
            raise ValueError("target_sigma_px must be finite and > 0.")
        if output["radius_px"] < 0:
            raise ValueError("target_radius_px must be >= 0.")
    else:
        output["affinity_gamma"] = float(
            config.get("target_affinity_gamma", 2.0)
        )
        if (
            not math.isfinite(output["affinity_gamma"])
            or output["affinity_gamma"] <= 0.0
        ):
            raise ValueError("target_affinity_gamma must be finite and > 0.")
    return output

def checkpoint_centerline_probability_target_config(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    stored = checkpoint.get("target_representation")
    if isinstance(stored, Mapping):
        output = dict(stored)
        output["mode"] = normalize_centerline_target_mode(output.get("mode"))
        return output
    source_config = checkpoint.get("config", {})
    if isinstance(source_config, Mapping):
        return centerline_probability_target_config(source_config)
    # Schema-1 checkpoints predate target metadata and were Gaussian-only.
    return {"mode": GAUSSIAN_CENTERLINE_TARGET}

def centerline_probability_model_config(
    config: Mapping[str, Any],
) -> dict[str, int]:
    raw_model = config.get("model", {})
    if not isinstance(raw_model, Mapping):
        raise ValueError("model must be a JSON object.")
    return {
        "learned_feature_dim": int(raw_model.get("learned_feature_dim", 32)),
        "centerline_head_hidden_dim": int(
            raw_model.get("centerline_head_hidden_dim", 16)
        ),
        "centerline_map_size": int(raw_model.get("centerline_map_size", 64)),
    }

def load_centerline_probability_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source = Path(path).expanduser().resolve()
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Centreline probability checkpoint must be a dictionary: {source}."
        )
    schema = int(checkpoint.get("schema_version", -1))
    if schema not in SUPPORTED_CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMAS:
        raise ValueError(
            f"Unsupported centreline checkpoint schema {schema}; expected "
            f"one of {sorted(SUPPORTED_CENTERLINE_PROBABILITY_CHECKPOINT_SCHEMAS)}."
        )
    task = str(checkpoint.get("task", "")).strip()
    if task and task != "single_view_centerline_probability":
        raise ValueError(
            f"Checkpoint {source} declares task={task!r}, not "
            "'single_view_centerline_probability'."
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"{source} does not contain model_state_dict.")
    return checkpoint, state

def load_pretrained_centerline_predictor_into_refiner(
    refiner: nn.Module,
    checkpoint_path: str | Path,
    *,
    freeze: bool = True,
) -> dict[str, Any]:
    """Load a standalone predictor into the shared or separate refiner path."""

    checkpoint, state = load_centerline_probability_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    source_target = checkpoint_centerline_probability_target_config(checkpoint)
    source_config = checkpoint.get("config", {})
    if isinstance(source_config, Mapping):
        source_model = source_config.get("model", {})
        if isinstance(source_model, Mapping):
            source_map_size = int(
                source_model.get(
                    "centerline_map_size",
                    getattr(refiner, "centerline_map_size", -1),
                )
            )
            target_map_size = int(getattr(refiner, "centerline_map_size", -1))
            if source_map_size != target_map_size:
                raise ValueError(
                    "Standalone and refiner centreline map sizes differ: "
                    f"checkpoint={source_map_size}, refiner={target_map_size}."
                )
    expected_mode = getattr(
        refiner, "centerline_probability_target_mode", None
    )
    source_mode = normalize_centerline_target_mode(
        source_target.get("mode", GAUSSIAN_CENTERLINE_TARGET)
    )
    if expected_mode is not None:
        expected_mode = normalize_centerline_target_mode(expected_mode)
        if source_mode != expected_mode:
            raise ValueError(
                "Standalone and refiner centreline target modes differ: "
                f"checkpoint={source_mode!r}, refiner={expected_mode!r}."
            )
        if "mask_threshold" in source_target:
            source_mask_threshold = float(source_target["mask_threshold"])
            expected_mask_threshold = float(
                getattr(
                    refiner,
                    "centerline_probability_mask_threshold",
                    source_mask_threshold,
                )
            )
            if not math.isclose(
                source_mask_threshold,
                expected_mask_threshold,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    "Standalone and refiner centreline mask thresholds differ: "
                    f"checkpoint={source_mask_threshold}, "
                    f"refiner={expected_mask_threshold}."
                )
        if source_mode == MASK_NORMALIZED_CENTERLINE_DISTANCE_TARGET:
            source_gamma = float(source_target.get("affinity_gamma", 2.0))
            expected_gamma = float(
                getattr(refiner, "centerline_probability_target_gamma", 2.0)
            )
            if not math.isclose(
                source_gamma,
                expected_gamma,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    "Standalone and refiner centreline affinity gamma differ: "
                    f"checkpoint={source_gamma}, refiner={expected_gamma}."
                )
        elif isinstance(source_config, Mapping):
            # For legacy Gaussian predictors these settings define the input
            # map semantics and should remain exact across pretraining/refining.
            target_settings = (
                ("target_sigma_px", "centerline_map_sigma_px", float),
                ("target_radius_px", "centerline_map_radius_px", int),
            )
            for source_key, refiner_attribute, caster in target_settings:
                if source_key not in source_config or not hasattr(
                    refiner, refiner_attribute
                ):
                    continue
                source_value = caster(source_config[source_key])
                target_value = caster(getattr(refiner, refiner_attribute))
                matches = (
                    source_value == target_value
                    if caster is int
                    else math.isclose(
                        source_value,
                        target_value,
                        rel_tol=0.0,
                        abs_tol=1.0e-8,
                    )
                )
                if not matches:
                    raise ValueError(
                        "Standalone and refiner centreline target settings "
                        f"differ: checkpoint {source_key}={source_value}, "
                        f"refiner {refiner_attribute}={target_value}."
                    )
    separate_encoder = getattr(
        refiner, "centerline_image_feature_encoder", None
    )
    encoder = (
        separate_encoder
        if separate_encoder is not None
        else getattr(refiner, "image_feature_encoder", None)
    )
    head = getattr(refiner, "input_centerline_probability_head", None)
    if encoder is None or head is None:
        raise ValueError(
            "The B-spline refiner must enable learned image features and "
            "at least one centreline-probability evidence path before loading "
            "a pretrained centreline predictor."
        )

    encoder_prefix = "image_feature_encoder."
    head_prefix = "input_centerline_probability_head."
    encoder_state = {
        key[len(encoder_prefix) :]: value
        for key, value in state.items()
        if key.startswith(encoder_prefix)
    }
    head_state = {
        key[len(head_prefix) :]: value
        for key, value in state.items()
        if key.startswith(head_prefix)
    }
    if not encoder_state or not head_state:
        raise KeyError(
            "Standalone checkpoint lacks image encoder or centreline-head weights."
        )
    try:
        encoder.load_state_dict(encoder_state, strict=True)
        head.load_state_dict(head_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Standalone centreline predictor is architecture-incompatible with "
            "the configured B-spline refiner. Match learned_feature_dim and "
            "centerline_head_hidden_dim."
        ) from error

    setattr(refiner, "pretrained_centerline_predictor_frozen", bool(freeze))
    setattr(
        refiner,
        "pretrained_centerline_predictor_checkpoint",
        str(Path(checkpoint_path).expanduser().resolve()),
    )
    if freeze:
        for module in (encoder, head):
            module.requires_grad_(False)
            module.eval()
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "separate_encoder": separate_encoder is not None,
        "frozen": bool(freeze),
        "target_representation": source_target,
    }

def enforce_frozen_centerline_predictor(refiner: nn.Module) -> None:
    """Reapply freezing after a refiner-only scope enables its parent module."""

    if not bool(
        getattr(refiner, "pretrained_centerline_predictor_frozen", False)
    ):
        return
    encoder = getattr(refiner, "centerline_image_feature_encoder", None)
    if encoder is None:
        encoder = getattr(refiner, "image_feature_encoder", None)
    head = getattr(refiner, "input_centerline_probability_head", None)
    for module in (encoder, head):
        if module is not None:
            module.requires_grad_(False)
            module.eval()

class CenterlineProbabilityPredictor(nn.Module):
    """Predict a dense 2D centreline map from one vessel-mask view.

    The submodule names and layer shapes deliberately match the corresponding
    geometry-refiner modules so a standalone checkpoint can be transplanted
    without reshaping or approximate key conversion.
    """

    def __init__(
        self,
        *,
        learned_feature_dim: int = 32,
        centerline_head_hidden_dim: int = 16,
        centerline_map_size: int = 64,
    ) -> None:
        super().__init__()
        self.learned_feature_dim = int(learned_feature_dim)
        self.centerline_head_hidden_dim = int(centerline_head_hidden_dim)
        self.centerline_map_size = int(centerline_map_size)
        if self.learned_feature_dim < 1:
            raise ValueError("learned_feature_dim must be >= 1.")
        if self.centerline_head_hidden_dim < 1:
            raise ValueError("centerline_head_hidden_dim must be >= 1.")
        if self.centerline_map_size < 2:
            raise ValueError("centerline_map_size must be >= 2.")

        self.image_feature_encoder = nn.Sequential(
            nn.Conv2d(1, self.learned_feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.learned_feature_dim,
                self.learned_feature_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )
        self.input_centerline_probability_head = nn.Sequential(
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or int(image.shape[1]) != 1:
            raise ValueError(
                "CenterlineProbabilityPredictor expects [B,1,H,W], got "
                f"{tuple(image.shape)}."
            )
        features = self.image_feature_encoder(image)
        compact = F.interpolate(
            features,
            size=(self.centerline_map_size, self.centerline_map_size),
            mode="bilinear",
            align_corners=True,
        )
        return self.input_centerline_probability_head(compact)
