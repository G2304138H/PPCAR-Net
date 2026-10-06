# Transferred from methods/parametric_methods/online_vggt.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from vessel_code.backbones.vggt_support.model_realdata import VGGTViewTokenVesselPredictor
from vessel_code.shared.data import normalize_feature_backbone

ONLINE_VGGT_CACHE_SCHEMA_VERSION = 1

def _merged_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    merged.update(dict(config.get("model", {}) or {}))
    return merged

def _json_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]

class OnlineVGGTFeatureProvider:
    """Read-through frozen-VGGT provider keyed by case and contextual view subset.

    In ``all_views`` mode, VGGT receives every selected view for one case in a
    single call.  Cases with different selected-view counts are never padded
    together before the backbone, because a padded image would become part of
    VGGT's context.  In ``per_view`` mode, every selected image is instead sent
    through VGGT with a view dimension of one, matching per-view precomputation.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: torch.device,
        raw_image_dataset_dir: str | Path,
    ) -> None:
        merged = _merged_config(config)
        self.device = torch.device(device)
        self.raw_image_dataset_dir = Path(raw_image_dataset_dir).expanduser().resolve()
        self.feature_backbone = normalize_feature_backbone(
            merged.get("feature_backbone", "vggt")
        )
        if self.feature_backbone not in ("vggt", "vggt_omega"):
            raise ValueError(
                "OnlineVGGTFeatureProvider requires feature_backbone='vggt' "
                "or 'vggt_omega'."
            )

        expected_context_value = merged.get("expected_vggt_context_mode")
        expected_context = (
            None
            if expected_context_value is None
            or not str(expected_context_value).strip()
            else str(expected_context_value).strip().lower()
        )
        # During evaluation, this field describes the feature semantics used
        # to train the checkpoint.  Prefer it over a legacy
        # ``vggt_context_mode`` value inherited from that training config.
        self.context_mode = str(
            expected_context
            or merged.get("vggt_context_mode", "all_views")
        ).strip().lower()
        if self.context_mode not in {"all_views", "per_view"}:
            raise ValueError(
                "On-the-fly parametric VGGT requires "
                "expected_vggt_context_mode (or vggt_context_mode) to be "
                "'all_views' or 'per_view'."
            )
        self.view_batch_size = int(
            merged.get(
                "evaluation_backbone_view_batch_size",
                merged.get("feature_view_batch_size", 0),
            )
        )
        if self.view_batch_size < 0:
            raise ValueError(
                "evaluation_backbone_view_batch_size must be >= 0."
            )
        if bool(merged.get("vggt_finetune", False)):
            raise ValueError(
                "The online parametric feature path is deliberately frozen; "
                "set vggt_finetune=false."
            )

        expected_variant = (
            "omega" if self.feature_backbone == "vggt_omega" else "original"
        )
        raw_variant = str(
            merged.get("vggt_backbone", expected_variant)
        ).strip().lower()
        vggt_variant = (
            "omega" if raw_variant in {"omega", "vggt_omega"} else "original"
        )
        if vggt_variant != expected_variant:
            raise ValueError(
                f"feature_backbone={self.feature_backbone!r} conflicts with "
                f"vggt_backbone={raw_variant!r}."
            )

        default_patch_size = 16 if vggt_variant == "omega" else 14
        default_target_size = 256 if vggt_variant == "omega" else 266
        self.token_dim = int(merged.get("vggt_token_dim", 2048))
        self.canonicalize_view_order = bool(
            merged.get("online_vggt_cache_canonicalize_view_order", True)
        )
        cache_dir_value = merged.get("online_vggt_cache_dir")
        self.cache_dir = (
            None
            if cache_dir_value is None or not str(cache_dir_value).strip()
            else Path(str(cache_dir_value)).expanduser().resolve()
        )
        self.cache_read = bool(
            merged.get("online_vggt_cache_read", self.cache_dir is not None)
        )
        self.cache_write = bool(merged.get("online_vggt_cache_write", False))
        if (self.cache_read or self.cache_write) and self.cache_dir is None:
            raise ValueError(
                "online_vggt_cache_read/write requires online_vggt_cache_dir."
            )
        cache_dtype_name = str(
            merged.get("online_vggt_cache_dtype", "float16")
        ).strip().lower()
        cache_dtypes = {"float16": np.float16, "float32": np.float32}
        if cache_dtype_name not in cache_dtypes:
            raise ValueError(
                "online_vggt_cache_dtype must be 'float16' or 'float32'."
            )
        self.cache_dtype_name = cache_dtype_name
        self.cache_dtype = cache_dtypes[cache_dtype_name]
        self.cache_compressed = bool(
            merged.get("online_vggt_cache_compressed", True)
        )

        omega_checkpoint = merged.get("vggt_omega_checkpoint_path")
        omega_checkpoint_path = (
            None
            if omega_checkpoint is None
            else Path(str(omega_checkpoint)).expanduser().resolve()
        )
        omega_checkpoint_fingerprint = None
        if omega_checkpoint_path is not None and omega_checkpoint_path.is_file():
            checkpoint_stat = omega_checkpoint_path.stat()
            omega_checkpoint_fingerprint = {
                "path": str(omega_checkpoint_path),
                "size": int(checkpoint_stat.st_size),
                "mtime_ns": int(checkpoint_stat.st_mtime_ns),
            }
        self.feature_signature = {
            "schema_version": ONLINE_VGGT_CACHE_SCHEMA_VERSION,
            "feature_backbone": self.feature_backbone,
            "vggt_context_mode": self.context_mode,
            "vggt_backbone": vggt_variant,
            "vggt_pretrained": bool(merged.get("vggt_pretrained", True)),
            "vggt_model_name": str(
                merged.get("vggt_model_name", "facebook/VGGT-1B")
            ),
            "vggt_load_mode": str(merged.get("vggt_load_mode", "package")),
            "vggt_torchhub_repo": str(
                merged.get("vggt_torchhub_repo", "facebookresearch/vggt")
            ),
            "vggt_omega_checkpoint_path": (
                None
                if omega_checkpoint_path is None
                else str(omega_checkpoint_path)
            ),
            "vggt_omega_checkpoint_fingerprint": omega_checkpoint_fingerprint,
            "vggt_token_dim": self.token_dim,
            "vggt_image_size_mode": str(
                merged.get(
                    "vggt_image_size_mode", "resize_to_patch_multiple"
                )
            ),
            "vggt_patch_size": int(
                merged.get("vggt_patch_size", default_patch_size)
            ),
            "vggt_target_image_size": merged.get(
                "vggt_target_image_size", default_target_size
            ),
            "canonicalize_view_order": self.canonicalize_view_order,
            "user_cache_tag": merged.get("online_vggt_cache_tag"),
        }
        self.signature_hash = _json_hash(self.feature_signature)
        if self.cache_write:
            assert self.cache_dir is not None
            (self.cache_dir / self.signature_hash).mkdir(parents=True, exist_ok=True)

        self.predictor = VGGTViewTokenVesselPredictor(
            num_points=int(merged.get("num_points", 200)),
            num_branches=int(merged.get("num_branches", 1)),
            view_feat_dim=int(merged.get("view_feat_dim", 4)),
            model_dim=int(merged.get("model_dim", 512)),
            vggt_backbone=vggt_variant,
            vggt_token_dim=self.token_dim,
            vggt_pretrained=bool(merged.get("vggt_pretrained", True)),
            vggt_finetune=False,
            vggt_model_name=self.feature_signature["vggt_model_name"],
            vggt_load_mode=self.feature_signature["vggt_load_mode"],
            vggt_torchhub_repo=self.feature_signature["vggt_torchhub_repo"],
            vggt_omega_checkpoint_path=self.feature_signature[
                "vggt_omega_checkpoint_path"
            ],
            vggt_image_size_mode=self.feature_signature["vggt_image_size_mode"],
            vggt_patch_size=self.feature_signature["vggt_patch_size"],
            vggt_target_image_size=self.feature_signature[
                "vggt_target_image_size"
            ],
        ).to(self.device)
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)

        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_stale_or_invalid": 0,
            "extraction_batches": 0,
            "extracted_cases": 0,
        }

    def resolved_config(self) -> dict[str, Any]:
        return {
            "online_vggt_feature_signature": dict(self.feature_signature),
            "online_vggt_feature_signature_hash": self.signature_hash,
            "online_vggt_cache_dir": (
                None if self.cache_dir is None else str(self.cache_dir)
            ),
            "online_vggt_cache_read": self.cache_read,
            "online_vggt_cache_write": self.cache_write,
            "online_vggt_cache_dtype": self.cache_dtype_name,
            "online_vggt_cache_compressed": self.cache_compressed,
            "online_vggt_cache_canonicalize_view_order": (
                self.canonicalize_view_order
            ),
            "online_vggt_cache_tag": self.feature_signature["user_cache_tag"],
            "resolved_online_vggt_context_mode": self.context_mode,
            "evaluation_backbone_view_batch_size": self.view_batch_size,
        }

    def stats(self) -> dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}

    def _relative_case_path(self, source_path: Path) -> Path:
        source = source_path.expanduser().resolve()
        try:
            relative = source.relative_to(self.raw_image_dataset_dir)
        except ValueError:
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
            relative = Path(f"external_{digest}_{source.stem}.npz")
        return relative.with_suffix("")

    def _cache_path(
        self, source_path: Path, contextual_view_indices: tuple[int, ...]
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        subset = "-".join(f"{index:03d}" for index in contextual_view_indices)
        return (
            self.cache_dir
            / self.signature_hash
            / self._relative_case_path(source_path)
            / f"views_{subset}.npz"
        )

    @staticmethod
    def _source_fingerprint(source_path: Path) -> dict[str, Any]:
        stat = source_path.stat()
        return {
            "source_path": str(source_path.expanduser().resolve()),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }

    def _cache_metadata(
        self,
        source_path: Path,
        contextual_view_indices: tuple[int, ...],
        feature_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        return {
            "schema_version": ONLINE_VGGT_CACHE_SCHEMA_VERSION,
            "signature_hash": self.signature_hash,
            "contextual_view_indices": list(contextual_view_indices),
            "feature_shape": list(feature_shape),
            **self._source_fingerprint(source_path),
        }

    def _load_cache(
        self,
        source_path: Path,
        contextual_view_indices: tuple[int, ...],
    ) -> torch.Tensor | None:
        cache_path = self._cache_path(source_path, contextual_view_indices)
        if not self.cache_read or cache_path is None or not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                features = np.asarray(payload["image_features"])
                metadata = json.loads(
                    str(np.asarray(payload["metadata_json"]).reshape(()).item())
                )
            expected_source = self._source_fingerprint(source_path)
            valid = (
                int(metadata.get("schema_version", -1))
                == ONLINE_VGGT_CACHE_SCHEMA_VERSION
                and metadata.get("signature_hash") == self.signature_hash
                and tuple(metadata.get("contextual_view_indices", ()))
                == contextual_view_indices
                and tuple(metadata.get("feature_shape", ())) == features.shape
                and all(metadata.get(key) == value for key, value in expected_source.items())
                and features.ndim == 3
                and int(features.shape[0]) == len(contextual_view_indices)
                and int(features.shape[-1]) == self.token_dim
                and np.isfinite(features).all()
            )
        except Exception:
            valid = False
            features = np.empty((0,), dtype=np.float32)
        if not valid:
            self._stats["cache_stale_or_invalid"] += 1
            return None
        self._stats["cache_hits"] += 1
        return torch.from_numpy(features.astype(np.float32, copy=False)).to(
            self.device, non_blocking=True
        )

    def _write_cache(
        self,
        source_path: Path,
        contextual_view_indices: tuple[int, ...],
        features: torch.Tensor,
    ) -> None:
        cache_path = self._cache_path(source_path, contextual_view_indices)
        if not self.cache_write or cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        array = features.detach().cpu().numpy().astype(self.cache_dtype, copy=False)
        metadata = self._cache_metadata(
            source_path, contextual_view_indices, tuple(array.shape)
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}.",
                suffix=".npz",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                save = np.savez_compressed if self.cache_compressed else np.savez
                save(
                    handle,
                    image_features=array,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            temporary_path.replace(cache_path)
            self._stats["cache_writes"] += 1
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _round_trip_cache_precision(self, features: torch.Tensor) -> torch.Tensor:
        """Make a newly computed value identical to what a later run reloads."""

        array = features.detach().cpu().numpy().astype(self.cache_dtype, copy=False)
        return torch.from_numpy(array.astype(np.float32, copy=False)).to(
            self.device, non_blocking=True
        )

    def _extract_group(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        if self.context_mode == "all_views":
            prepared = VGGTViewTokenVesselPredictor._prepare_vggt_images(
                images,
                image_size_mode=self.predictor.vggt_image_size_mode,
                patch_size=self.predictor.vggt_patch_size,
                target_image_size=self.predictor.vggt_target_image_size,
            )
            with torch.no_grad():
                features = self.predictor._extract_vggt_tokens(prepared)
            self._stats["extraction_batches"] += 1
        else:
            batch_size, num_views = int(images.shape[0]), int(images.shape[1])
            flat_images = images.reshape(
                batch_size * num_views, *images.shape[2:]
            )
            step = (
                self.view_batch_size
                if self.view_batch_size > 0
                else int(flat_images.shape[0])
            )
            feature_chunks: list[torch.Tensor] = []
            for start in range(0, int(flat_images.shape[0]), step):
                # The leading dimension may batch independent images for
                # efficiency, but VGGT always sees exactly one contextual view
                # per sample.
                prepared = VGGTViewTokenVesselPredictor._prepare_vggt_images(
                    flat_images[start : start + step].unsqueeze(1),
                    image_size_mode=self.predictor.vggt_image_size_mode,
                    patch_size=self.predictor.vggt_patch_size,
                    target_image_size=self.predictor.vggt_target_image_size,
                )
                with torch.no_grad():
                    chunk = self.predictor._extract_vggt_tokens(prepared)
                if chunk.ndim != 4 or int(chunk.shape[1]) != 1:
                    raise RuntimeError(
                        "Online per-view VGGT returned shape "
                        f"{tuple(chunk.shape)}; expected [B,1,L,{self.token_dim}]."
                    )
                feature_chunks.append(chunk[:, 0])
                self._stats["extraction_batches"] += 1
            flat_features = torch.cat(feature_chunks, dim=0)
            features = flat_features.reshape(
                batch_size,
                num_views,
                *flat_features.shape[1:],
            )
        if features.ndim != 4 or int(features.shape[-1]) != self.token_dim:
            raise RuntimeError(
                "Online VGGT returned shape "
                f"{tuple(features.shape)}; expected [B,V,L,{self.token_dim}]."
            )
        return features.detach().to(dtype=torch.float32)

    def features_for_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch.get("images")
        if images is None:
            raise ValueError(
                "On-the-fly VGGT requires raw input images in every batch."
            )
        view_mask = batch["view_mask"]
        selected_indices = batch["selected_view_indices"]
        source_paths = batch["path"]
        if not isinstance(source_paths, list) or len(source_paths) != int(images.shape[0]):
            raise ValueError("Online VGGT batch paths do not match the batch size.")

        records: list[dict[str, Any]] = []
        missing_groups: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for batch_index, raw_path in enumerate(source_paths):
            count = int((view_mask[batch_index] > 0.5).sum().item())
            if count < 1:
                raise ValueError(f"Online VGGT case {raw_path} has no selected views.")
            requested_indices = np.asarray(
                selected_indices[batch_index, :count].detach().cpu(), dtype=np.int64
            )
            if np.unique(requested_indices).size != count:
                raise ValueError(
                    f"Online VGGT case {raw_path} has duplicate source-view indices "
                    f"{requested_indices.tolist()}."
                )
            canonical_order = (
                np.argsort(requested_indices, kind="stable")
                if self.canonicalize_view_order
                else np.arange(count, dtype=np.int64)
            )
            contextual_indices = tuple(
                int(value) for value in requested_indices[canonical_order]
            )
            source_path = Path(str(raw_path)).expanduser().resolve()
            self._stats["requests"] += 1
            cached = self._load_cache(source_path, contextual_indices)
            record = {
                "batch_index": batch_index,
                "source_path": source_path,
                "contextual_indices": contextual_indices,
                "canonical_order": canonical_order,
                "features": cached,
            }
            records.append(record)
            if cached is None:
                self._stats["cache_misses"] += 1
                order_tensor = torch.as_tensor(
                    canonical_order,
                    device=images.device,
                    dtype=torch.long,
                )
                record["canonical_images"] = images[
                    batch_index, :count
                ].index_select(0, order_tensor)
                group_key = (
                    count,
                    tuple(int(value) for value in record["canonical_images"].shape[1:]),
                )
                missing_groups[group_key].append(record)

        for group_records in missing_groups.values():
            group_images = torch.stack(
                [record["canonical_images"] for record in group_records], dim=0
            )
            group_features = self._extract_group(group_images)
            self._stats["extracted_cases"] += len(group_records)
            for record, features in zip(group_records, group_features):
                self._write_cache(
                    record["source_path"],
                    record["contextual_indices"],
                    features,
                )
                record["features"] = (
                    self._round_trip_cache_precision(features)
                    if self.cache_write
                    else features
                )

        reordered_features: list[torch.Tensor] = []
        for record in records:
            canonical_features = record["features"]
            if canonical_features is None:
                raise RuntimeError("Online VGGT failed to produce contextual features.")
            inverse_order = np.argsort(record["canonical_order"], kind="stable")
            inverse_tensor = torch.as_tensor(
                inverse_order,
                device=canonical_features.device,
                dtype=torch.long,
            )
            reordered_features.append(
                canonical_features.index_select(0, inverse_tensor).to(
                    dtype=torch.float32
                )
            )

        max_views = int(view_mask.shape[1])
        first = reordered_features[0]
        output = first.new_zeros(
            (len(reordered_features), max_views, *first.shape[1:])
        )
        for index, features in enumerate(reordered_features):
            output[index, : features.shape[0]] = features
        return output
