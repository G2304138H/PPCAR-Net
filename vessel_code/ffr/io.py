"""Load a 3D artery (vessel code) from the NPZ files this repository produces or consumes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Keys tried in order. Each entry: (array key, multiplier to mm or None for "use input_scale_to_mm").
_VESSEL_KEYS = (
    ("vessel_code_mm", 1.0),            # evaluate.py refined.npz / coarse.npz (absolute LAS mm)
    ("decoded_vessel_mm", 1.0),         # raw model output dumps
    ("pred_vessel_code_mm", 1.0),       # real-xray prediction npz (if saved under this name)
    ("predicted_vessel_code_mm", 1.0),
    ("raw_vessel_code_mm", 1.0),        # prepared examples / CT-derived reference
    ("reconstructed_vessel_code_mm", 1.0),
    ("artery", None),                   # prepared examples, stored in metres x input_scale_to_mm
)
_EXIST_KEYS = ("branch_exists", "pred_branch_exists", "branch_exist", "active_branches")


@dataclass
class VesselInput:
    vessel_mm: np.ndarray            # [M, N, 4] absolute xyz (mm) + radius (mm)
    exists: np.ndarray               # [M] bool
    artery_type: str                 # "lca" | "rca" | "unknown"
    source: str
    key: str
    meta: dict = field(default_factory=dict)


def _artery_from_npz(data, path: Path) -> str:
    for key in ("artery_type", "vessel_type"):
        if key in data.files:
            value = str(np.asarray(data[key]).item()).lower()
            if value in ("lca", "rca"):
                return value
    name = str(path).lower()
    for candidate in ("lca", "rca"):
        if f"/{candidate}" in name or name.startswith(candidate) or f"{candidate}_" in Path(name).name:
            return candidate
    return "unknown"


def load_vessel(path: str | Path, *, key: str | None = None, artery_type: str | None = None) -> VesselInput:
    """Read a vessel-code NPZ.

    Accepts ``refined.npz``/``coarse.npz`` from ``vessel_code.evaluate``, the bundled
    prepared examples (ground-truth artery) and generic prediction dumps holding a
    ``[M,N,4]`` array of xyz + radius in millimetres. ``key`` forces a specific array.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    candidates = [(key, 1.0)] if key else list(_VESSEL_KEYS)
    vessel = used_key = None
    for name, scale in candidates:
        if name in data.files:
            array = np.asarray(data[name], dtype=np.float64)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3 or array.shape[-1] != 4:
                continue
            if scale is None:
                scale = float(data["input_scale_to_mm"]) if "input_scale_to_mm" in data.files else 1000.0
            vessel, used_key = array * scale, name
            break
    if vessel is None:
        raise KeyError(
            f"{path}: no [M,N,4] vessel-code array found (tried {[k for k, _ in candidates]}); "
            "pass --key to name it"
        )
    exists = None
    for name in _EXIST_KEYS:
        if name in data.files:
            exists = np.asarray(data[name]).astype(bool).reshape(-1)
            if exists.shape[0] != vessel.shape[0]:
                exists = None
            else:
                break
    radius_ok = np.all(vessel[..., 3] > 0, axis=1)
    if exists is None:
        exists = radius_ok
    else:
        exists = exists & radius_ok
    if not exists[0]:
        raise ValueError(f"{path}: main branch (index 0) is missing or has non-positive radii")
    meta = {}
    for name in ("projection_center_offset_mm", "projection_center_offset", "case_id", "sample_name"):
        if name in data.files:
            meta[name] = np.asarray(data[name]).tolist()
    artery = (artery_type or _artery_from_npz(data, path)).lower()
    return VesselInput(vessel, exists, artery, str(path), used_key, meta)
