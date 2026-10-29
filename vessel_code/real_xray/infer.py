"""Reconstruct real X-ray angiography cases (keyframe masks exported by angioseg-nnunet) and visualise the result.

    python -m vessel_code.real_xray.infer --config configs/real_xray/lca.json [--shard K --num_shards N]

For every case (one study x artery, all its views) the model sees `num_views` input views; every other view is held out
and only used to check the reconstruction. Output per case, under <output_dir>/<case_name>/:
    surface_3d.gif                      rotating predicted artery surface
    surface_3d_radius.gif               the same, coloured by the predicted radius
    reprojection/input_views_on_xray.png      predicted surface reprojected onto the input views, over the raw X-ray
    reprojection/input_views_on_mask.png      ... over the segmentation mask (green both, red predicted only, white mask only)
    reprojection/heldout_views_on_xray.png    the same for the views NOT given to the model
    reprojection/heldout_views_on_mask.png
    reprojection/views/v<i>_<input|heldout>_<xray|mask>.png   full-resolution single views
    prediction_refined.npz / prediction_coarse.npz            vessel code (centred mm), branch existence, input views
    case_summary.json                   input views, per-view reprojection Dice (refined and coarse)
and <output_dir>/summary.csv with one row per case.
Overlays are drawn on the original keyframe in its original (DICOM display) orientation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from vessel_code.config import load_config
from vessel_code.real_xray.data import RealXrayCase, inference_camera, load_case, resolve_input_views
from vessel_code.real_xray import render

SUMMARY_FIELDS = ["case_name", "artery", "seg_method", "subset", "system", "iso_pixel_mm", "camera",
                  "camera_sid_mm", "camera_sod_mm", "camera_pixel_mm", "n_views", "input_views", "input_clinical_views",
                  "heldout_views", "dice_input_refined", "dice_heldout_refined",
                  "dice_input_refined_aligned", "dice_heldout_refined_aligned",
                  "dice_input_coarse", "dice_heldout_coarse", "dice_input_coarse_aligned",
                  "dice_heldout_coarse_aligned", "n_active_branches", "status"]


def validate_config(config: dict) -> None:
    if config.get("artery_type") not in ("rca", "lca"):
        raise ValueError("artery_type must be rca or lca")
    n = config.get("num_views", 2)
    if type(n) is not int or not 1 <= n <= 7:
        raise ValueError("num_views must be an integer from 1 to 7")
    if config.get("output_mode", "full") not in ("full", "summary"):
        raise ValueError("output_mode must be full or summary")
    if config.get("view_selection", "max_separation") not in ("max_separation", "first"):
        raise ValueError("view_selection must be max_separation or first")
    for key in ("checkpoint", "input_dir", "output_dir"):
        if not config.get(key):
            raise ValueError(f"{key} is required")


# ----------------------------------------------------------------------------- model
def load_model(config: dict):
    import torch
    from vessel_code.config import validate_model_config
    from vessel_code.parametric.model import build_model_for_checkpoint
    from vessel_code.parametric.online_vggt import OnlineVGGTFeatureProvider

    ckpt_path = Path(config["checkpoint"])
    # Research checkpoints contain config and numpy RNG state, so load trusted local files only.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt["config"]
    validate_model_config(model_config)
    if model_config["artery_type"].lower() != config["artery_type"]:
        raise ValueError("Checkpoint and config artery_type disagree")
    if model_config.get("target_coordinate_frame", "projection_centered") != "projection_centered":
        raise ValueError("A projection-centered checkpoint is required")
    dev = config.get("device", "auto")
    device = torch.device(dev if dev != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_for_checkpoint(model_config, ckpt["view_feat_dim"], ckpt.get("inferred_feature_dim"),
                                       ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    provider_config = dict(model_config, online_vggt_cache_read=False, online_vggt_cache_write=False,
                           online_vggt_cache_dir=None)
    provider = OnlineVGGTFeatureProvider(provider_config, device=device, raw_image_dataset_dir=Path(config["input_dir"]))
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return model, provider, device, sha


def predict(model, provider, device, case: RealXrayCase, inputs: list[int]) -> dict:
    import torch
    idx = np.asarray(inputs, np.int64)
    batch = {
        "images": torch.from_numpy(case.images[idx][None]).to(device),
        "view_mask": torch.ones(1, len(idx), dtype=torch.bool, device=device),
        # positions within the selected set (already ascending, so VGGT's canonical order = this order)
        "selected_view_indices": torch.arange(len(idx), device=device)[None],
        "path": [str(case.path)],
    }
    with torch.inference_mode():
        features = provider.features_for_batch(batch)
        return model(views=torch.from_numpy(case.view_features[idx][None]).to(device), view_mask=batch["view_mask"],
                     image_features=features, images=batch["images"])


def set_model_camera(model, cam: dict) -> int:
    """Set SID / SOD / detector pixel spacing on every module that projects into the views (B-spline and radius
    refiners and their surface projector). They are plain attributes read at every forward pass, so this changes the
    camera without touching the weights. The coarse predictor has no camera input (use_ray_camera_encoding=false).
    Returns how many modules were updated."""
    n = 0
    for m in model.modules():
        if all(hasattr(m, a) for a in ("sid", "source_to_iso", "imager_pixel_spacing")):
            if int(getattr(m, "image_size", cam["image_dim"])) != int(cam["image_dim"]):
                raise ValueError(f"{type(m).__name__} works on {m.image_size}px images, camera has {cam['image_dim']}")
            m.sid = float(cam["sid"])
            m.source_to_iso = float(cam["source_to_iso"])
            m.imager_pixel_spacing = float(cam["imager_pixel_spacing"])
            n += 1
    return n


# ----------------------------------------------------------------------------- outputs
SHEET_COLUMNS = ["X-ray (keyframe)", "X-ray + reprojected 3D", "segmentation (input mask)",
                 "segmentation vs reprojected 3D", "3D prediction, this view", "3D prediction, radius"]


def save_case_sheet(case: RealXrayCase, vessel, active, inputs, views, masks_shown, shifts, cam, role, aligned,
                    path: Path) -> Path:
    """One PNG per case: one row per view (keyframe), six panels from left to right, all in the original
    orientation and resolution of the keyframe and in the same position (incl. the 2D alignment, if used)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    quads, qr = render.surface_quads(vessel, active)
    rng = (float(qr.min()), float(qr.max())) if len(qr) else (0.0, 1.0)
    V = case.num_views
    Ht = V * 2.6 + 1.3                                           # figure height (inches)
    fig, axes = plt.subplots(V, 6, figsize=(6 * 2.55 + 2.0, Ht), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.995, top=1 - 0.80 / Ht, bottom=0.55 / Ht, wspace=0.04, hspace=0.06)
    for i in range(V):
        H, W = case.full_shape[i]
        raw = case.raw_full[i, :H, :W]
        seg = case.label_full[i, :H, :W] > 0
        pred = render.model_to_full(masks_shown[i], case.model_to_full[i], (H, W))
        plain, rad = render.render_surface_view(quads, qr, case.theta_deg[i], case.phi_deg[i], cam,
                                                case.model_to_full[i], (H, W), shift_rc=shifts[i], radius_range=rng)
        panels = [np.repeat((raw.astype(np.float32) / 255.0)[..., None], 3, -1), render.overlay_on_xray(raw, pred),
                  np.repeat(seg[..., None].astype(np.float32), 3, -1), render.overlay_on_mask(seg, pred), plain, rad]
        for j, (ax, im) in enumerate(zip(axes[i], panels)):
            ax.imshow(im, interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(SHEET_COLUMNS[j], fontsize=9)
        v = views[i]
        is_input = v["role"] == "input"
        kf = case.meta.get("keyframe_index", [None] * V)[i]
        nf = case.meta.get("n_frames", [None] * V)[i]
        dice = v.get(f"dice_{role}_aligned" if aligned else f"dice_{role}", float("nan"))
        label = (f"v{i}  {case.clinical_views[i]}\n"
                 f"{'INPUT to 3D reconstruction' if is_input else 'NOT input (held-out)'}"
                 f"{' - same angle as an input' if v['same_direction_as_an_input'] else ''}\n"
                 f"keyframe {kf}/{nf}   Dice {dice:.2f}")
        axes[i, 0].set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8,
                              color="#11703a" if is_input else "#a34a00", fontweight="bold" if is_input else "normal")
        for ax in axes[i]:
            for sp in ax.spines.values():
                sp.set_edgecolor("#11703a" if is_input else "#cccccc")
                sp.set_linewidth(2.2 if is_input else 0.8)
    md = {g: [v.get(f"dice_{role}_aligned" if aligned else f"dice_{role}") for v in views if v["role"] == g]
          for g in ("input", "heldout")}
    mean = {g: (np.nanmean(x) if x else float("nan")) for g, x in md.items()}
    fig.suptitle(
        f"{case.name}   [{case.meta.get('seg_method', '?')}]   {case.meta.get('system', '')} "
        f"{float(case.meta.get('src_iso_pixel_mm', float('nan'))):.3f} mm/px   camera {cam.get('mode', '')} "
        f"(SID {cam['sid'] * 1000:.0f}, SOD {cam['source_to_iso'] * 1000:.0f} mm)   {role} reconstruction\n"
        f"inputs: {', '.join(f'v{k}' for k in inputs)}   mean Dice{' (2D-aligned)' if aligned else ''}: "
        f"input {mean['input']:.2f}, held-out {mean['heldout']:.2f}", fontsize=10.5, y=1 - 0.12 / Ht, va="top")
    fig.text(0.01, 0.12 / Ht, "orange = reprojected 3D surface;  green = both, red = reprojection only, white = "
             "segmentation only\n3D panels: predicted surface seen along this view's X-ray direction (original image "
             "orientation)" + (";  reprojections shifted by their best 2D translation" if aligned else ""),
             fontsize=8, va="bottom")
    cax = fig.add_axes((0.80, 0.22 / Ht, 0.18, 0.09 / Ht))
    cb = fig.colorbar(ScalarMappable(Normalize(*rng), render.radius_colormap()), cax=cax, orientation="horizontal")
    cb.ax.set_title("radius (mm)", fontsize=8, pad=2)
    cb.ax.tick_params(labelsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def visualize_case(case: RealXrayCase, roles: dict, inputs: list[int], dest: Path, config: dict) -> dict:
    """GIFs + reprojection overlays + per-view Dice. `roles` = {role: (vessel_centered_mm [M,N,4], active [M])}."""
    cam = case.meta.get("inference_camera") or inference_camera(case, config)
    tag = f"{case.name} [{case.meta.get('seg_method', '?')}]"
    input_set = set(inputs)
    input_groups = {int(case.angle_group[i]) for i in inputs}
    align = bool(config.get("align_reprojection", True))
    max_shift = int(config.get("max_align_shift_px", 48))
    views = []
    reproj, aligned = {}, {}
    for role, (vessel, active) in roles.items():
        reproj[role] = render.reproject_masks(vessel, active, case.theta_deg, case.phi_deg, cam)
        aligned[role] = np.zeros_like(reproj[role])
    for i in range(case.num_views):
        v = dict(index=i, clinical_view=case.clinical_views[i], theta_deg=float(case.theta_deg[i]),
                 phi_deg=float(case.phi_deg[i]), role="input" if i in input_set else "heldout",
                 same_direction_as_an_input=bool(int(case.angle_group[i]) in input_groups and i not in input_set))
        for role, masks in reproj.items():
            d = render.translation_to_align(masks[i], case.images[i], max_shift)
            aligned[role][i] = render.shift_mask(masks[i], *d)
            v[f"dice_{role}"] = render.dice(masks[i], case.images[i])
            v[f"dice_{role}_aligned"] = render.dice(aligned[role][i], case.images[i])
            v[f"align_shift_px_{role}"] = list(d)
        views.append(v)

    # ---- reprojection overlays (refined, or coarse if the checkpoint has no refiner)
    show = "refined" if "refined" in reproj else "coarse"
    masks_shown = aligned[show] if align else reproj[show]
    full_mode = config.get("output_mode", "full") == "full"
    rdir = dest / "reprojection"
    for group in (("input", "heldout") if full_mode else ()):
        ids = [v["index"] for v in views if v["role"] == group]
        tiles_x, tiles_m, titles = [], [], []
        for i in ids:
            H, W = case.full_shape[i]
            pred_full = render.model_to_full(masks_shown[i], case.model_to_full[i], (H, W))
            raw = case.raw_full[i, :H, :W]
            seg = case.label_full[i, :H, :W] > 0
            xr, mk = render.overlay_on_xray(raw, pred_full), render.overlay_on_mask(seg, pred_full)
            render.save_png(xr, rdir / "views" / f"v{i}_{group}_xray.png")
            render.save_png(mk, rdir / "views" / f"v{i}_{group}_mask.png")
            tiles_x.append(xr)
            tiles_m.append(mk)
            v = views[i]
            note = " (same angle as an input)" if v["same_direction_as_an_input"] else ""
            if align:
                dr, dc = v[f"align_shift_px_{show}"]
                score = (f"Dice {v[f'dice_{show}_aligned']:.3f} after 2D shift ({dr:+d},{dc:+d}) px; "
                         f"{v[f'dice_{show}']:.3f} without")
            else:
                score = f"Dice {v[f'dice_{show}']:.3f}"
            titles.append(f"v{i}: {case.clinical_views[i]}{note}\n{score}")
        if ids:
            what = "input views" if group == "input" else "held-out views (not given to the model)"
            how = " (translation-aligned)" if align else ""
            render.save_panel(tiles_x, titles, rdir / f"{group}_views_on_xray.png",
                              f"{tag}: {show} reconstruction reprojected onto the {what}{how}",
                              "orange = reprojected predicted surface (original image orientation)")
            render.save_panel(tiles_m, titles, rdir / f"{group}_views_on_mask.png",
                              f"{tag}: {show} reconstruction vs segmentation, {what}{how}",
                              "green = both, red = reprojection only, white = segmentation only")

    # ---- rotating 3D surfaces
    if full_mode and config.get("save_gifs", True):
        from vessel_code.shared.visualization import save_3d_prediction_gif, save_3d_radius_colored_prediction_gif
        vessel, active = roles[show]
        frames, fps = int(config.get("gif_frames", 60)), int(config.get("gif_fps", 12))
        exist = active.astype(np.float32)
        save_3d_prediction_gif(vessel, exist, dest / "surface_3d.gif", frames, fps,
                               title=f"{case.name}: predicted 3D artery")
        rmin, rmax = save_3d_radius_colored_prediction_gif(vessel, exist, dest / "surface_3d_radius.gif", frames, fps,
                                                           title=f"{case.name}: predicted radius")
    else:
        rmin = rmax = None

    # ---- one-page summary (both modes)
    vessel, active = roles[show]
    shifts = [tuple(v[f"align_shift_px_{show}"]) if align else (0, 0) for v in views]
    sheet = save_case_sheet(case, vessel, active, inputs, views, masks_shown, shifts, cam, show, align,
                            dest / "summary.png")
    if config.get("_summary_dir"):
        import shutil
        Path(config["_summary_dir"]).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sheet, Path(config["_summary_dir"]) / f"{case.name}.png")

    def mean(key, group):
        vals = [v[key] for v in views if v["role"] == group and np.isfinite(v.get(key, np.nan))]
        return float(np.mean(vals)) if vals else None

    mean_dice = {}
    for r in reproj:
        for g in ("input", "heldout"):
            mean_dice[f"{g}_{r}"] = mean(f"dice_{r}", g)
            mean_dice[f"{g}_{r}_aligned"] = mean(f"dice_{r}_aligned", g)
    return dict(
        case_name=case.name, artery=case.artery_type.upper(), npz=str(case.path), meta=case.meta,
        input_views=inputs, input_clinical_views=[case.clinical_views[i] for i in inputs],
        heldout_views=[v["index"] for v in views if v["role"] == "heldout"],
        input_recentering_shift_px=case.meta.get("input_recentering_shift_px"),
        n_active_branches={r: int(a.sum()) for r, (_, a) in roles.items()},
        radius_range_mm=[rmin, rmax], views=views, shown_role=show, overlays_aligned=align,
        mean_dice=mean_dice,
    )


def recenter_inputs(case: RealXrayCase, inputs: list[int], vessel: np.ndarray, active: np.ndarray,
                    max_shift: int) -> list[list[int]]:
    """Shift each input view so that it agrees with the reprojection of the current prediction (in place).

    The exported views are centred on their own mask centroid, not on a common isocentre; this moves them into the
    frame of the model's own 3D estimate. model_to_full is updated, so overlays stay on the original images.
    """
    cam = case.meta.get("inference_camera") or case.camera()
    masks = render.reproject_masks(vessel, active, case.theta_deg[inputs], case.phi_deg[inputs], cam)
    shifts = []
    for j, i in enumerate(inputs):
        dr, dc = render.translation_to_align(masks[j], case.images[i], max_shift)   # moves prediction onto input
        dr, dc = -dr, -dc                                                            # move input onto prediction
        case.images[i] = render.shift_mask(case.images[i], dr, dc)
        case.raw_images[i] = ndimage.shift(case.raw_images[i], (dr, dc), order=1).astype(np.float32)
        case.model_to_full[i] = case.model_to_full[i] @ render.shift_matrix(dr, dc)
        shifts.append([dr, dc])
    return shifts


def write_summary(out: Path, rows: list[dict], shard: int) -> None:
    with open(out / f"summary_shard{shard:03d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    merged = []
    for p in sorted(out.glob("summary_shard*.csv")):
        merged.extend(csv.DictReader(open(p, newline="")))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)


def summary_row(s: dict) -> dict:
    fmt = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
    md = s["mean_dice"]
    row = dict(case_name=s["case_name"], artery=s["artery"], seg_method=s["meta"].get("seg_method", ""),
               subset=s["meta"].get("subset", ""), system=s["meta"].get("system", ""),
               iso_pixel_mm=s["meta"].get("src_iso_pixel_mm", ""),
               camera=(s["meta"].get("inference_camera") or {}).get("mode", ""),
               camera_sid_mm=round(1000 * (s["meta"].get("inference_camera") or {}).get("sid", float("nan")), 1),
               camera_sod_mm=round(1000 * (s["meta"].get("inference_camera") or {}).get("source_to_iso", float("nan")), 1),
               camera_pixel_mm=round((s["meta"].get("inference_camera") or {}).get("imager_pixel_spacing", float("nan")), 4),
               n_views=len(s["views"]), input_views=";".join(map(str, s["input_views"])),
               input_clinical_views=" | ".join(s["input_clinical_views"]),
               heldout_views=";".join(map(str, s["heldout_views"])),
               n_active_branches=s["n_active_branches"].get(s["shown_role"]), status="ok")
    for k in SUMMARY_FIELDS:
        if k.startswith("dice_"):
            row[k] = fmt(md.get(k[len("dice_"):]))
    return row


def run(config: dict, shard: int = 0, num_shards: int = 1) -> Path:
    from vessel_code.evaluate import prediction_roles
    validate_config(config)
    artery = config["artery_type"]
    root = Path(config["input_dir"])
    files = sorted(root.glob("*.npz"))
    if config.get("case_ids"):
        wanted = set(map(str, config["case_ids"]))
        missing = wanted - {p.stem for p in files}
        if missing:
            raise ValueError(f"Missing requested cases: {sorted(missing)}")
        files = [p for p in files if p.stem in wanted]
    files = [p for k, p in enumerate(files) if k % num_shards == shard]
    if config.get("limit"):
        files = files[: int(config["limit"])]
    if not files:
        raise ValueError(f"No cases to run in {root}")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    config = dict(config, _summary_dir=str(out / "_summary"))     # flat folder with every case's summary.png

    model, provider, device, sha = load_model(config)
    (out / "run.json").write_text(json.dumps({"config": config, "checkpoint_sha256": sha}, indent=2) + "\n")
    print(f"[real_xray] {len(files)} cases (shard {shard}/{num_shards}) on {device} -> {out}", flush=True)
    rows, t0 = [], time.time()
    for n, path in enumerate(files, 1):
        dest = out / path.stem
        done = dest / "case_summary.json"
        if done.exists() and not config.get("overwrite", False):
            rows.append(summary_row(json.loads(done.read_text())))
            continue
        try:
            case = load_case(path, artery)
            inputs = sorted(resolve_input_views(case, config))
            thr = float(config.get("branch_threshold", 0.5))
            cam = inference_camera(case, config)
            n_mod = set_model_camera(model, cam)
            if n_mod == 0 and cam["mode"] != "training":
                raise RuntimeError("no camera-using module found in the model; cannot apply the acquisition camera")
            if cam.get("warning"):
                print(f"  [camera] {case.name}: {cam['warning']}", flush=True)
            case.meta["inference_camera"] = cam
            output = predict(model, provider, device, case, inputs)
            roles = prediction_roles(output, artery, threshold=thr)
            total = np.zeros((len(inputs), 2), int)
            for _ in range(int(config.get("input_recentering_iterations", 0))):
                vessel, active = roles["refined" if "refined" in roles else "coarse"]
                total += np.asarray(recenter_inputs(case, inputs, vessel, active,
                                                    int(config.get("max_recentering_shift_px", 40))))
                output = predict(model, provider, device, case, inputs)
                roles = prediction_roles(output, artery, threshold=thr)
            if config.get("input_recentering_iterations", 0):
                case.meta["input_recentering_shift_px"] = total.tolist()
            dest.mkdir(parents=True, exist_ok=True)
            for role, (vessel, active) in roles.items():
                np.savez_compressed(dest / f"prediction_{role}.npz", vessel_code_centered_mm=vessel,
                                    branch_exists=active, input_views=np.asarray(inputs),
                                    theta_deg=case.theta_deg, phi_deg=case.phi_deg,
                                    camera_sid_m=cam["sid"], camera_source_to_iso_m=cam["source_to_iso"],
                                    camera_imager_pixel_spacing_mm=cam["imager_pixel_spacing"],
                                    camera_mode=np.asarray(cam["mode"]))
            summary = visualize_case(case, roles, inputs, dest, config)
            done.write_text(json.dumps(summary, indent=2, default=float) + "\n")
            rows.append(summary_row(summary))
            md = summary["mean_dice"]
            print(f"  {n}/{len(files)} {case.name}: inputs {inputs} | Dice input "
                  f"{md.get('input_refined', md.get('input_coarse'))} held-out "
                  f"{md.get('heldout_refined', md.get('heldout_coarse'))} | {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep going over a large set; the error is recorded in summary.csv
            rows.append(dict(case_name=path.stem, artery=artery.upper(), status=f"error: {type(e).__name__}: {e}"))
            print(f"  [error] {path.stem}: {type(e).__name__}: {e}", flush=True)
            if config.get("raise_errors", False):
                raise
    write_summary(out, rows, shard)
    print(f"[real_xray] done -> {out / 'summary.csv'}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--input_dir", help="override config input_dir")
    p.add_argument("--output_dir", help="override config output_dir")
    p.add_argument("--num_views", type=int, help="override config num_views")
    p.add_argument("--camera", choices=["acquisition", "training"], help="override config camera")
    p.add_argument("--output_mode", choices=["full", "summary"], help="full: GIFs + per-view PNGs + summary.png; "
                   "summary: only summary.png per case")
    p.add_argument("--limit", type=int, help="only the first N cases (test run)")
    p.add_argument("--case_ids", nargs="+", help="only these cases (file names without .npz)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    a = p.parse_args()
    config = load_config(a.config)
    for key in ("input_dir", "output_dir", "num_views", "camera", "limit", "case_ids", "output_mode"):
        if getattr(a, key) is not None:
            config[key] = getattr(a, key)
    run(config, a.shard, a.num_shards)


if __name__ == "__main__":
    main()
