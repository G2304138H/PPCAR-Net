# Transferred from methods/parametric_methods/sampling_schedule.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch.utils.data import Dataset
from vessel_code.parametric.data import ParametricBranchVariantGroupDataset, ParametricFeatureDataset, collate_parametric_batches

"""Record and replay the exact parametric-training sampling trajectory."""

TRAINING_SAMPLING_SCHEDULE_SCHEMA_VERSION = 1

TRAINING_SAMPLING_SCHEDULE_MODES = frozenset({"off", "record", "replay"})

def _load_schedule_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training sampling schedule not found: {path}")
    if path.suffix.lower() == ".jsonl":
        payload: dict[str, Any] | None = None
        with open(path, "r") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL training sampling schedule record at "
                        f"{path}:{line_number}."
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Training sampling schedule record at {path}:"
                        f"{line_number} must be a JSON object."
                    )
                record_type = record.get("record_type")
                if record_type == "header":
                    if payload is not None:
                        raise ValueError(
                            f"Training sampling schedule {path} contains more "
                            "than one header."
                        )
                    payload = {
                        "schema_version": record.get("schema_version"),
                        "sampling_policy": record.get("sampling_policy", {}),
                        "epochs": {},
                    }
                elif record_type == "epoch":
                    if payload is None:
                        raise ValueError(
                            f"Training sampling schedule {path} has an epoch "
                            "before its header."
                        )
                    epoch_key = str(int(record.get("epoch")))
                    if epoch_key in payload["epochs"]:
                        raise ValueError(
                            f"Training sampling schedule {path} repeats epoch "
                            f"{epoch_key}."
                        )
                    payload["epochs"][epoch_key] = {
                        "num_steps": record.get("num_steps"),
                        "steps": record.get("steps"),
                    }
                else:
                    raise ValueError(
                        f"Training sampling schedule record at {path}:"
                        f"{line_number} has unknown record_type={record_type!r}."
                    )
        if payload is None:
            raise ValueError(f"Training sampling schedule {path} is empty.")
    else:
        with open(path, "r") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Training sampling schedule must be a JSON object: {path}")
    if int(payload.get("schema_version", -1)) != TRAINING_SAMPLING_SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported training sampling schedule schema in {path}: "
            f"{payload.get('schema_version')!r}."
        )
    epochs = payload.get("epochs")
    if not isinstance(epochs, dict):
        raise ValueError(f"Training sampling schedule {path} has no epochs object.")
    return payload

def resolve_training_sampling_schedule(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> tuple[str, Path | None]:
    raw_mode = config.get("training_sampling_schedule_mode", "off")
    mode = str(raw_mode).strip().lower()
    if mode not in TRAINING_SAMPLING_SCHEDULE_MODES:
        raise ValueError(
            "training_sampling_schedule_mode must be 'off', 'record', or "
            f"'replay', got {raw_mode!r}."
        )
    raw_path = config.get("training_sampling_schedule_path")
    if mode == "off":
        if raw_path is not None and str(raw_path).strip():
            raise ValueError(
                "training_sampling_schedule_path is set while "
                "training_sampling_schedule_mode='off'. Select 'record' or "
                "'replay', or clear the path."
            )
        return mode, None
    if raw_path is None or not str(raw_path).strip():
        if mode == "replay":
            raise ValueError(
                "training_sampling_schedule_mode='replay' requires "
                "training_sampling_schedule_path."
            )
        return mode, (output_dir / "training_sampling_schedule.jsonl").resolve()
    return mode, Path(str(raw_path)).expanduser().resolve()

def sampling_policy_from_config(
    config: Mapping[str, Any],
    *,
    grouped: bool,
    normalized_view_count_weights: Mapping[int, float] | None,
    replacement_sampling: bool,
) -> dict[str, Any]:
    min_views = config.get("min_train_views", config.get("max_views"))
    max_views = config.get("max_train_views", config.get("max_views"))
    if normalized_view_count_weights is not None:
        view_count_method = "weighted_categorical"
    elif min_views is not None and max_views is not None and int(min_views) != int(max_views):
        view_count_method = "uniform_integer_inclusive"
    else:
        view_count_method = "fixed"
    visibility = config.get("branch_visibility_sampling")
    return {
        "seed": int(config.get("seed", 0)),
        "sampling_unit": "physical_case_group" if grouped else "case_file",
        "case_order_method": (
            "torch_random_with_replacement"
            if replacement_sampling
            else "torch_random_permutation"
        ),
        "batch_size": (
            1 if grouped else int(config.get("batch_size", 4))
        ),
        "min_steps_per_epoch": config.get("min_steps_per_epoch"),
        "view_count_method": view_count_method,
        "view_randomness_scope": (
            "sha256_seeded_per_case_and_epoch"
            if grouped
            else "global_python_random_stream"
        ),
        "min_train_views": None if min_views is None else int(min_views),
        "max_train_views": None if max_views is None else int(max_views),
        "normalized_view_count_probabilities": (
            None
            if normalized_view_count_weights is None
            else {
                str(count): float(probability)
                for count, probability in normalized_view_count_weights.items()
            }
        ),
        "view_order_method": (
            "random_without_replacement"
            if bool(config.get("train_random_view_order", False))
            else "first_k_in_source_order"
        ),
        "branch_variant_group_training": bool(grouped),
        "branch_variant_train_sampling": (
            str(config.get("branch_variant_train_sampling", "all"))
            if grouped
            else None
        ),
        "branch_variant_resample_each_epoch": (
            bool(config.get("branch_variant_resample_each_epoch", True))
            if grouped
            else None
        ),
        "branch_visibility_sampling": (
            dict(visibility) if isinstance(visibility, Mapping) else None
        ),
    }

def sampling_step_from_batch(batch: Mapping[str, Any], step: int) -> dict[str, Any]:
    view_mask = batch.get("view_mask")
    local_indices = batch.get("local_view_indices")
    selected_indices = batch.get("selected_view_indices")
    if not all(torch.is_tensor(value) for value in (view_mask, local_indices, selected_indices)):
        raise ValueError(
            "Training sampling recording requires tensor view_mask, "
            "local_view_indices, and selected_view_indices entries."
        )
    assert isinstance(view_mask, torch.Tensor)
    assert isinstance(local_indices, torch.Tensor)
    assert isinstance(selected_indices, torch.Tensor)
    if view_mask.ndim < 2:
        raise ValueError(
            f"Training batch view_mask must be [B,V,...], got {tuple(view_mask.shape)}."
        )
    batch_size = int(view_mask.shape[0])

    def sequence(name: str) -> Sequence[Any]:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            if value.ndim < 1 or int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"Training batch {name} does not match batch size {batch_size}."
                )
            return value.detach().cpu().tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"Training batch {name} must be a batch sequence.")
        if len(value) != batch_size:
            raise ValueError(
                f"Training batch {name} has {len(value)} entries, expected {batch_size}."
            )
        return value

    case_names = sequence("case_name")
    case_ids = sequence("case_id")
    paths = sequence("path")
    item_indices = sequence("item_index")
    variants_raw = batch.get("branch_subset_variant")
    variants = (
        [None] * batch_size
        if variants_raw is None
        else list(variants_raw)
    )
    if len(variants) != batch_size:
        raise ValueError(
            "Training batch branch_subset_variant does not match its batch size."
        )

    samples: list[dict[str, Any]] = []
    for sample_index in range(batch_size):
        count = int((view_mask[sample_index] > 0.5).sum().item())
        if count < 1:
            raise ValueError(
                f"Cannot record a zero-view training sample at step {step}."
            )
        local = [
            int(value)
            for value in local_indices[sample_index, :count].detach().cpu().tolist()
        ]
        selected = [
            int(value)
            for value in selected_indices[sample_index, :count]
            .detach()
            .cpu()
            .tolist()
        ]
        samples.append(
            {
                "case_name": str(case_names[sample_index]),
                "case_id": str(case_ids[sample_index]),
                "path": str(paths[sample_index]),
                "item_index": int(item_indices[sample_index]),
                "num_views": count,
                "local_view_indices": local,
                "selected_view_indices": selected,
                "branch_subset_variant": (
                    None
                    if variants[sample_index] is None
                    else str(variants[sample_index])
                ),
            }
        )

    grouped = bool(batch.get("branch_variant_group_mode", False))
    record: dict[str, Any] = {
        "step": int(step),
        "grouped": grouped,
        "samples": samples,
    }
    if grouped:
        record.update(
            {
                "group_id": str(batch["branch_variant_group_id"]),
                "group_case_id": str(batch["branch_variant_group_case_id"]),
                "group_stage": str(batch["branch_variant_group_stage"]),
                "policy_skipped_variants": [
                    str(value)
                    for value in batch.get(
                        "branch_visibility_policy_skipped_variants", ()
                    )
                ],
            }
        )
    return record

def _comparable_sampling_step(record: Mapping[str, Any]) -> dict[str, Any]:
    samples = []
    for raw_sample in record.get("samples", []):
        sample = dict(raw_sample)
        samples.append(
            {
                key: sample.get(key)
                for key in (
                    "case_name",
                    "case_id",
                    "num_views",
                    "local_view_indices",
                    "selected_view_indices",
                    "branch_subset_variant",
                )
            }
        )
    output = {
        "step": int(record.get("step", -1)),
        "grouped": bool(record.get("grouped", False)),
        "samples": samples,
    }
    if output["grouped"]:
        output.update(
            {
                "group_id": record.get("group_id"),
                "group_case_id": record.get("group_case_id"),
                "group_stage": record.get("group_stage"),
                "policy_skipped_variants": record.get(
                    "policy_skipped_variants", []
                ),
            }
        )
    return output

class TrainingSamplingSchedule:
    """Accumulate a schedule or validate batches materialised from one."""

    def __init__(
        self,
        *,
        mode: str,
        path: Path,
        sampling_policy: Mapping[str, Any],
        resume_recording: bool = False,
    ) -> None:
        if mode not in {"record", "replay"}:
            raise ValueError(f"TrainingSamplingSchedule cannot use mode {mode!r}.")
        self.mode = mode
        self.path = path.expanduser().resolve()
        self._current_epoch: int | None = None
        self._current_steps: list[dict[str, Any]] = []
        self._expected_steps: list[dict[str, Any]] = []
        loaded_existing = bool(
            mode == "replay" or (resume_recording and self.path.is_file())
        )
        if loaded_existing:
            self.payload = _load_schedule_payload(self.path)
        else:
            self.payload = {
                "schema_version": TRAINING_SAMPLING_SCHEDULE_SCHEMA_VERSION,
                "sampling_policy": dict(sampling_policy),
                "epochs": {},
            }
        self.payload.setdefault("sampling_policy", dict(sampling_policy))
        if (
            mode == "record"
            and resume_recording
            and loaded_existing
            and self.payload["sampling_policy"] != dict(sampling_policy)
        ):
            raise ValueError(
                f"Cannot resume sampling-schedule recording because the "
                f"sampling policy changed: {self.path}."
            )
        if mode == "record" and not loaded_existing:
            self._initialise_record_file()

    @property
    def available_epochs(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in self.payload["epochs"]))

    def steps_for_epoch(self, epoch: int) -> list[dict[str, Any]]:
        raw = self.payload["epochs"].get(str(int(epoch)))
        if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
            raise ValueError(
                f"Training sampling schedule {self.path} has no complete epoch "
                f"{int(epoch)}; available epochs={list(self.available_epochs)}."
            )
        steps = [dict(value) for value in raw["steps"]]
        if int(raw.get("num_steps", -1)) != len(steps):
            raise ValueError(
                f"Training sampling schedule {self.path} epoch {epoch} reports "
                f"num_steps={raw.get('num_steps')!r}, but stores {len(steps)} steps."
            )
        for expected_step, record in enumerate(steps):
            if int(record.get("step", -1)) != expected_step:
                raise ValueError(
                    f"Training sampling schedule {self.path} epoch {epoch} has "
                    f"non-contiguous step numbering at position {expected_step}."
                )
        return steps

    def start_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        self._current_steps = []
        if self.mode == "record" and str(self._current_epoch) in self.payload["epochs"]:
            # A training epoch may have completed and been recorded before a
            # later validation/checkpoint failure. Re-running from the previous
            # checkpoint replaces that orphaned schedule epoch.
            del self.payload["epochs"][str(self._current_epoch)]
            self._rewrite_record_file()
        self._expected_steps = (
            self.steps_for_epoch(epoch) if self.mode == "replay" else []
        )

    def observe_batch(self, batch: Mapping[str, Any], step: int) -> None:
        if self._current_epoch is None:
            raise RuntimeError("Training sampling schedule epoch was not started.")
        actual = sampling_step_from_batch(batch, step)
        if self.mode == "record":
            self._current_steps.append(actual)
            return
        if int(step) >= len(self._expected_steps):
            raise ValueError(
                f"Replay produced unexpected step {step} in epoch "
                f"{self._current_epoch}; schedule contains "
                f"{len(self._expected_steps)} steps."
            )
        expected = self._expected_steps[int(step)]
        if _comparable_sampling_step(actual) != _comparable_sampling_step(expected):
            raise ValueError(
                f"Replayed training sample differs from the schedule at epoch "
                f"{self._current_epoch}, step {step}. Expected "
                f"{_comparable_sampling_step(expected)!r}, got "
                f"{_comparable_sampling_step(actual)!r}."
            )
        self._current_steps.append(actual)

    def finish_epoch(self) -> None:
        if self._current_epoch is None:
            raise RuntimeError("Training sampling schedule epoch was not started.")
        epoch = self._current_epoch
        if self.mode == "replay":
            expected_count = len(self._expected_steps)
            if len(self._current_steps) != expected_count:
                raise ValueError(
                    f"Replay epoch {epoch} yielded {len(self._current_steps)} "
                    f"steps, expected {expected_count}."
                )
        else:
            epoch_payload = {
                "num_steps": len(self._current_steps),
                "steps": self._current_steps,
            }
            self.payload["epochs"][str(epoch)] = epoch_payload
            self._write_epoch(epoch, epoch_payload)
        self._current_epoch = None
        self._current_steps = []
        self._expected_steps = []

    def _initialise_record_file(self) -> None:
        self.payload["epochs"] = {}
        self._rewrite_record_file()

    def _rewrite_record_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with open(temporary, "w") as handle:
            if self.path.suffix.lower() == ".jsonl":
                header = {
                    "record_type": "header",
                    "schema_version": self.payload["schema_version"],
                    "sampling_policy": self.payload["sampling_policy"],
                }
                handle.write(json.dumps(header, separators=(",", ":")) + "\n")
                for epoch in self.available_epochs:
                    epoch_payload = self.payload["epochs"][str(epoch)]
                    record = {
                        "record_type": "epoch",
                        "epoch": epoch,
                        **dict(epoch_payload),
                    }
                    handle.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
            else:
                json.dump(self.payload, handle, indent=2)
        temporary.replace(self.path)

    def _write_epoch(self, epoch: int, payload: Mapping[str, Any]) -> None:
        if self.path.suffix.lower() != ".jsonl":
            self._rewrite_record_file()
            return
        record = {
            "record_type": "epoch",
            "epoch": int(epoch),
            **dict(payload),
        }
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

class ReplayTrainingBatchDataset(Dataset):
    """Turn recorded optimizer steps back into already-collated batches."""

    def __init__(
        self,
        dataset: ParametricFeatureDataset | ParametricBranchVariantGroupDataset,
        schedule: TrainingSamplingSchedule,
        *,
        initial_epoch: int,
    ) -> None:
        if schedule.mode != "replay":
            raise ValueError("ReplayTrainingBatchDataset requires a replay schedule.")
        self.dataset = dataset
        self.schedule = schedule
        self.base = dataset.base
        self._steps: list[dict[str, Any]] = []
        self._ordinary_index_by_path: dict[str, int] = {}
        self._ordinary_indices_by_identity: dict[tuple[str, str], list[int]] = {}
        self._group_index_by_id: dict[str, int] = {}
        if isinstance(dataset, ParametricBranchVariantGroupDataset):
            self._group_index_by_id = {
                group.group_id: index
                for index, (group, _indices) in enumerate(dataset.groups)
            }
            if len(self._group_index_by_id) != len(dataset.groups):
                raise ValueError("Grouped training dataset contains duplicate group IDs.")
        else:
            for index, item in enumerate(dataset.items):
                name = str(item["case_name"])
                case_id = str(item.get("case_id", name))
                path = str(Path(str(item["path"])).expanduser().resolve())
                if path in self._ordinary_index_by_path:
                    raise ValueError(
                        "Training schedule replay requires unique source paths; "
                        f"duplicate={path!r}."
                    )
                self._ordinary_index_by_path[path] = index
                self._ordinary_indices_by_identity.setdefault(
                    (name, case_id), []
                ).append(index)
        self.set_epoch(initial_epoch)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
        self._steps = self.schedule.steps_for_epoch(int(epoch))

    def __len__(self) -> int:
        return len(self._steps)

    @staticmethod
    def _validate_materialised_sample(
        output: Mapping[str, Any],
        scheduled: Mapping[str, Any],
    ) -> None:
        count = int(output["view_mask"].shape[0])
        actual_local = [int(value) for value in output["local_view_indices"].tolist()]
        actual_selected = [
            int(value) for value in output["selected_view_indices"].tolist()
        ]
        if (
            count != int(scheduled["num_views"])
            or actual_local != [int(value) for value in scheduled["local_view_indices"]]
            or actual_selected
            != [int(value) for value in scheduled["selected_view_indices"]]
        ):
            raise ValueError(
                f"Current dataset cannot reproduce scheduled views for "
                f"{scheduled.get('case_name')!r}: scheduled local/original="
                f"{scheduled.get('local_view_indices')}/"
                f"{scheduled.get('selected_view_indices')}, current="
                f"{actual_local}/{actual_selected}."
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        step = self._steps[int(index)]
        scheduled_samples = step.get("samples")
        if not isinstance(scheduled_samples, list) or not scheduled_samples:
            raise ValueError(
                f"Sampling schedule step {index} contains no samples."
            )
        grouped = bool(step.get("grouped", False))
        if grouped:
            if not isinstance(self.dataset, ParametricBranchVariantGroupDataset):
                raise ValueError(
                    "Sampling schedule contains grouped steps but the current "
                    "training dataset is ordinary."
                )
            group_id = str(step.get("group_id"))
            if group_id not in self._group_index_by_id:
                raise ValueError(
                    f"Sampling schedule group {group_id!r} is absent from the "
                    "current training dataset."
                )
            first = scheduled_samples[0]
            variants = [
                str(sample["branch_subset_variant"])
                for sample in scheduled_samples
            ]
            return self.dataset.item_from_sampling_schedule(
                self._group_index_by_id[group_id],
                local_view_indices=first["local_view_indices"],
                variants=variants,
                policy_skipped_variants=step.get(
                    "policy_skipped_variants", ()
                ),
            )

        if isinstance(self.dataset, ParametricBranchVariantGroupDataset):
            raise ValueError(
                "Sampling schedule contains ordinary steps but the current "
                "training dataset is grouped."
            )
        materialised = []
        for scheduled in scheduled_samples:
            case_name = str(scheduled["case_name"])
            case_id = str(scheduled["case_id"])
            scheduled_path = str(
                Path(str(scheduled["path"])).expanduser().resolve()
            )
            item_index = self._ordinary_index_by_path.get(scheduled_path)
            if item_index is None:
                candidates = self._ordinary_indices_by_identity.get(
                    (case_name, case_id), []
                )
                if len(candidates) != 1:
                    raise ValueError(
                        f"Sampling schedule case {case_name!r}/{case_id!r} "
                        "does not have one unambiguous match in the current "
                        f"training dataset; candidates={len(candidates)}."
                    )
                item_index = candidates[0]
            item = self.dataset.item_with_view_indices(
                item_index,
                scheduled["local_view_indices"],
            )
            if str(item.get("case_id")) != case_id:
                raise ValueError(
                    f"Sampling schedule case ID mismatch for {case_name!r}: "
                    f"schedule={scheduled['case_id']!r}, current={item.get('case_id')!r}."
                )
            self._validate_materialised_sample(item, scheduled)
            materialised.append(item)
        return collate_parametric_batches(materialised)
