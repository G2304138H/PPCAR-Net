"""Read real X-ray keyframe cases exported by angioseg-nnunet (scripts/export_recon_inputs.py) and choose input views."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from vessel_code.geometry.projection import view_features_from_angles

FORMAT_VERSION = 1
MODEL_CAMERA = {  # what the released checkpoints were trained with (configs/*/train_*.json)
    "lca": dict(image_dim=256, sid=0.9, source_to_iso=0.75, imager_pixel_spacing=0.65),
    "rca": dict(image_dim=256, sid=0.9, source_to_iso=0.75, imager_pixel_spacing=0.55),
}
# Acquisition geometry of the CoronaryDominance systems (dataset authors' reference values, medians; mm).
DEFAULT_ACQUISITION_GEOMETRY = {
    "allura": {"sid_mm": 1137.0, "sod_mm": 765.0},
    "azurion": {"sid_mm": 1195.0, "sod_mm": 810.0},
}


def model_iso_pixel_mm(artery: str) -> float:
    """Pixel size at the isocentre of the 256x256 model images (the grid step 08 resamples every view onto)."""
    c = MODEL_CAMERA[artery]
    return c["imager_pixel_spacing"] * c["source_to_iso"] / c["sid"]


def inference_camera(case: "RealXrayCase", config: dict) -> dict:
    """Camera for the refiners and the reprojection of one case.

    camera = "training"    : the checkpoint's own camera (SID 0.9 m, SOD 0.75 m, 0.65 / 0.55 mm).
    camera = "acquisition" : SID and SOD of the case's acquisition system (config "acquisition_geometry", keyed by the
        `system` stored by step 08: allura / azurion; "default" for anything else). The detector pixel spacing is the
        one that keeps the exported grid: isocentre pixel (0.542 / 0.458 mm) x SID / SOD, unless
        "imager_pixel_spacing_mm" is set; then a mismatch with the grid is reported in `warning`.
    """
    mode = config.get("camera", "acquisition")
    cam = dict(MODEL_CAMERA[case.artery_type], mode=mode, system=case.meta.get("system", ""))
    if mode == "training":
        return cam
    if mode != "acquisition":
        raise ValueError("camera must be 'training' or 'acquisition'")
    table = {**DEFAULT_ACQUISITION_GEOMETRY, **config.get("acquisition_geometry", {})}
    geo = table.get(cam["system"]) or table.get("default")
    if geo is None:
        raise ValueError(f"{case.name}: no acquisition geometry for system '{cam['system']}' (add it, or 'default', "
                         f"to acquisition_geometry in the config)")
    sid_m, sod_m = float(geo["sid_mm"]) / 1000.0, float(geo["sod_mm"]) / 1000.0
    if not 0 < sod_m < sid_m:
        raise ValueError(f"acquisition geometry needs 0 < SOD < SID, got SID {sid_m} m, SOD {sod_m} m")
    grid_px = model_iso_pixel_mm(case.artery_type) * sid_m / sod_m
    px = float(config.get("imager_pixel_spacing_mm") or grid_px)
    cam.update(sid=sid_m, source_to_iso=sod_m, imager_pixel_spacing=px)
    if abs(px / grid_px - 1) > 0.01:
        cam["warning"] = (f"imager_pixel_spacing_mm {px:.4f} gives {px * sod_m / sid_m:.4f} mm at the isocentre, "
                          f"but the exported images use {model_iso_pixel_mm(case.artery_type):.4f}")
    return cam


@dataclass
class RealXrayCase:
    path: Path
    name: str
    artery_type: str
    images: np.ndarray            # (V,256,256) binary keyframe masks in model space
    raw_images: np.ndarray        # (V,256,256) keyframe X-ray in model space, [0,1]
    theta_deg: np.ndarray         # (V,)
    phi_deg: np.ndarray           # (V,)
    view_features: np.ndarray     # (V,4)
    raw_full: np.ndarray          # (V,Hmax,Wmax) uint8, original orientation
    label_full: np.ndarray        # (V,Hmax,Wmax) uint8
    full_shape: np.ndarray        # (V,2)
    model_to_full: np.ndarray     # (V,3,3)
    clinical_views: list[str]
    angle_group: np.ndarray       # (V,)
    keyframe_vessel_px: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        return int(self.images.shape[0])

    def camera(self) -> dict:
        return dict(MODEL_CAMERA[self.artery_type])


def load_case(path: str | Path, artery_type: str) -> RealXrayCase:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if "input_format" not in z.files or str(z["input_format"]) != "real_xray_keyframes":
            raise ValueError(f"{path}: not a real X-ray keyframe case (input_format missing)")
        if int(z["format_version"]) > FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {int(z['format_version'])} is newer than this reader")
        artery = str(z["vessel_type"]).lower()
        if artery != artery_type:
            raise ValueError(f"{path}: case artery {artery} != requested {artery_type}")
        cam = MODEL_CAMERA[artery]
        for key, ref in (("image_dim", cam["image_dim"]), ("sid", cam["sid"]),
                         ("imager_pixel_spacing", cam["imager_pixel_spacing"])):
            if not np.isclose(float(z[key]), ref):
                raise ValueError(f"{path}: {key}={float(z[key])} but the {artery.upper()} model expects {ref}")
        images = np.asarray(z["images"], np.float32)
        V = images.shape[0]
        if images.shape[1:] != (cam["image_dim"],) * 2 or images.min() < 0 or images.max() > 1:
            raise ValueError(f"{path}: images must be (V,256,256) in [0,1]")
        theta, phi = np.asarray(z["theta_deg"], np.float32), np.asarray(z["phi_deg"], np.float32)
        feats = np.asarray(z["view_features"], np.float32)
        if not np.allclose(feats, view_features_from_angles(theta, phi), atol=1e-5):
            raise ValueError(f"{path}: view_features disagree with theta/phi")
        meta_keys = ["case_id", "study_id", "subset", "seg_method", "seg_dir", "mask_key", "orientation",
                     "center_mode", "scale_mode", "scale_full_to_model", "src_iso_pixel_mm", "angle_convention",
                     "system", "scale_calibration"]
        meta = {k: (z[k].item() if z[k].ndim == 0 else z[k].tolist()) for k in meta_keys if k in z.files}
        for k in ("primary_angle_deg", "secondary_angle_deg", "keyframe_index", "n_frames",
                  "crop_clipped_fraction", "nearest_anchor_deg", "video_stem", "video_path"):
            if k in z.files:
                meta[k] = z[k].tolist()
        return RealXrayCase(
            path=path, name=str(z["case_name"]) if "case_name" in z.files else path.stem, artery_type=artery,
            images=images, raw_images=np.asarray(z["raw_images"], np.float32), theta_deg=theta, phi_deg=phi,
            view_features=feats, raw_full=np.asarray(z["raw_full"]), label_full=np.asarray(z["label_full"]),
            full_shape=np.asarray(z["full_shape"], np.int64), model_to_full=np.asarray(z["model_to_full"], np.float64),
            clinical_views=[str(s) for s in z["clinical_views"]],
            angle_group=np.asarray(z["angle_group"], np.int64) if "angle_group" in z.files else np.arange(V),
            keyframe_vessel_px=np.asarray(z["keyframe_vessel_px"]) if "keyframe_vessel_px" in z.files else images.reshape(V, -1).sum(1),
            meta=meta,
        )


def _directions(case: RealXrayCase) -> np.ndarray:
    """Unit viewing directions from the DICOM angles (only angles between them are used)."""
    p = np.deg2rad(np.asarray(case.meta.get("primary_angle_deg", 90 - case.theta_deg), np.float64))
    s = np.deg2rad(np.asarray(case.meta.get("secondary_angle_deg", 90 - case.phi_deg), np.float64))
    return np.stack([np.sin(p) * np.cos(s), np.cos(p) * np.cos(s), np.sin(s)], 1)


def select_input_views(case: RealXrayCase, num_views: int, strategy: str = "max_separation",
                       max_useful_separation_deg: float = 90.0) -> list[int]:
    """Choose `num_views` input views from distinct viewing directions.

    first          : the first views in file order with distinct directions
    max_separation : one view per direction (the one with most vessel), then the set whose smallest pairwise angle is
                     largest (angles above max_useful_separation_deg count as equal); ties -> more vessel pixels
    """
    groups: dict[int, int] = {}
    for i in range(case.num_views):          # representative per direction: most vessel pixels
        g = int(case.angle_group[i])
        if g not in groups or case.keyframe_vessel_px[i] > case.keyframe_vessel_px[groups[g]]:
            groups[g] = i
    reps = sorted(groups.values())
    if num_views > len(reps):
        raise ValueError(f"{case.name}: {num_views} input views requested but only {len(reps)} distinct directions")
    if strategy == "first":
        firsts = []
        for i in range(case.num_views):
            if int(case.angle_group[i]) not in {int(case.angle_group[j]) for j in firsts}:
                firsts.append(i)
        return firsts[:num_views]
    if strategy != "max_separation":
        raise ValueError(f"unknown view selection {strategy}")
    if num_views == 1:
        return [max(reps, key=lambda i: case.keyframe_vessel_px[i])]
    d = _directions(case)
    best, best_key = None, None
    for combo in combinations(reps, num_views):
        ang = min(np.degrees(np.arccos(np.clip(d[a] @ d[b], -1, 1))) for a, b in combinations(combo, 2))
        key = (min(ang, max_useful_separation_deg), int(sum(case.keyframe_vessel_px[i] for i in combo)))
        if best_key is None or key > best_key:
            best, best_key = list(combo), key
    return best


def resolve_input_views(case: RealXrayCase, config: dict) -> list[int]:
    """Config `input_views`: "auto" (default), a list of indices for every case, or {case_name: [indices]}."""
    spec = config.get("input_views", "auto")
    if isinstance(spec, dict):
        spec = spec.get(case.name, "auto")
    if spec == "auto":
        return select_input_views(case, int(config.get("num_views", 2)), config.get("view_selection", "max_separation"))
    idx = [int(i) for i in spec]
    if not idx or len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) >= case.num_views or len(idx) > 7:
        raise ValueError(f"{case.name}: invalid input_views {idx} for {case.num_views} views")
    return idx
