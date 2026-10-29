# Real X-ray angiography (CoronaryDominance) inference

This branch runs the released RCA/LCA checkpoints on real angiograms whose vessels were segmented by
[angioseg-nnunet](https://github.com/G2304138H/angioseg-nnunet) (nnU-Net, XRAVOS or SMART). Nothing in the model,
the checkpoints or the synthetic-CT pipeline changes; the additions are `vessel_code/real_xray/`, `configs/real_xray/`
and `tests/test_real_xray.py`.

## 1. Input: one NPZ per study × artery

Made by `scripts/export_recon_inputs.py` in angioseg-nnunet (`slurm/08_export_recon_inputs.sbatch`). Every video of
the case is one view; its keyframe is the frame with the most segmented vessel pixels.

| Key | Shape | Content |
| --- | --- | --- |
| `images` | `[V,256,256]` float32 | binary keyframe masks in the model's camera space (model input) |
| `raw_images` | `[V,256,256]` float32 | keyframe X-ray in the same space, `[0,1]` |
| `theta_deg`, `phi_deg`, `view_features` | `[V]`, `[V]`, `[V,4]` | model camera angles and their sin/cos |
| `primary_angle_deg`, `secondary_angle_deg` | `[V]` | DICOM positioner angles (+ = LAO / CRA) |
| `raw_full`, `label_full`, `full_shape` | `[V,H,W]`, `[V,2]` | keyframe X-ray and label map at full resolution, original orientation |
| `model_to_full` | `[V,3,3]` | maps model pixel `[row, col, 1]` to the full-resolution image |
| `angle_group` | `[V]` | views with identical angles share a group |
| `vessel_type`, `case_name`, `seg_method`, … | scalars | bookkeeping |

The files live in `<export>/<method>/lca/` and `<export>/<method>/rca/`; point `input_dir` at one of these folders.

### How real views are mapped into the model camera

- **Angles.** `theta = wrap(90° − primary)`, `phi = 90° − secondary` for both arteries (paper, App. A.1; primary + = LAO,
  secondary + = CRA). The projector looks along `(sin φ cos θ, sin φ sin θ, cos φ)` in the LAS frame of the training CTs,
  and the training masks were rendered with it, so this pairs each real image with the θ at which the model saw that
  side of the heart. The clinical labels attached to the RCA camera table in `docs/data_format.md` (e.g. `LAO 60` →
  θ = −60) are 90° off that projector. On real angiograms the RCA in LAO 25–35 (the "C") matches the training anatomy
  at the paper's θ ≈ 60, not at the label θ ≈ −30 (near-vertical vessel). The checkpoints are unchanged; physically the
  RCA training views lie around LAO 15 and LAO 70 (plus views from behind), so real RCA views are often 15–30° from the
  nearest one. The exporter's `--angle_convention code` (RCA `theta = −primary`) is kept for comparison. Each case file
  stores `nearest_anchor_deg`, the angle between each view and the closest training view.
- **Orientation.** In the model's images, rows point towards the patient's head and columns towards the patient's
  right; DICOM display has the head at the top and the patient's left on the right. The exporter therefore rotates
  every frame by 180° (`--orientation rot180`).
- **Scale.** The model images have 0.65 mm (LCA) / 0.55 mm (RCA) detector pixels with SID 0.90 m and source–isocentre
  0.75 m, i.e. 0.542 / 0.458 mm per pixel at the isocentre. The CoronaryDominance files keep no geometry; the dataset
  authors' reference values give 0.3418 mm × 765 / 1137 = **0.2300 mm** per pixel at the isocentre for the Allura
  subsets (`main_normal`, `main_others`). The Azurion subset (`domain_shift_dataset`) was cropped and resized, so its
  pixel size is measured from the data (angioseg-nnunet step 11) and passed to the exporter. Each case stores
  `system`, `src_iso_pixel_mm` and `scale_calibration`, and `summary.csv` repeats subset, system and pixel size, so
  results can be reported per system. Magnification differs (real SID/SOD ≈ 1.49 vs 1.2 in the model), but the
  depth dependence of magnification is set by the SOD (765–810 vs 750 mm), so matching the isocentre pixel size
  is the right correction.
- **Centring.** Training projections are centred on the 3D main-artery centroid (the isocentre is moved there before
  projecting). Real views keep their raw C-arm geometry by default: the image centre is the isocentre, which is
  consistent between views as long as the table is not moved, but the artery is not centred the way the training
  projections are. `--center mask_centroid` instead centres each view on its own mask (centred, but not consistent
  between views). See section 4.

## 2. Run

**Environment:** the existing venv `/export/home2/reny0012/vir_env/vesseltree` (PyTorch, this package, VGGT); no extra
setup. VGGT-1B is loaded from the usual HuggingFace cache; if a compute node has no internet and the weights are
cached, add `HF_HUB_OFFLINE=1` to the job's `--export`.

**GPU job** (`slurm/real_xray_infer.sbatch`, one job, all cases; set `REPO_ROOT` if the clone is not at
`/export/home2/reny0012/code/PPCAR-Net`):

```bash
sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,LIMIT=5 slurm/real_xray_infer.sbatch   # test: first 5 cases
sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2 slurm/real_xray_infer.sbatch           # all LCA cases
# options: CAMERA=acquisition|training, NUM_VIEWS=2, CASES="Study00605_RCA Study00087_RCA"
```
Input: `/dataset/reny0012/angioseg/recon_inputs/<METHOD>/<ART>/`; output:
`/dataset/reny0012/angioseg/recon_outputs/<METHOD>_<camera>[_first<LIMIT>]/<ART>/`.

Directly: `python -m vessel_code.real_xray.infer --config configs/real_xray/lca.json --input_dir ... --output_dir ...`
with optional `--camera`, `--num_views`, `--limit`, `--case_ids`, `--shard K --num_shards N`.

### Camera at inference

The coarse predictor has no camera input (`use_ray_camera_encoding: false` in both checkpoints); only the B-spline
geometry refiner and the radius refiner (and its surface projector) project into the views. Their SID, SOD and
detector pixel spacing are plain attributes, set per case before the forward pass, so no weights change.

- `camera: "acquisition"` (default): SID and SOD of the case's system from `acquisition_geometry` in the config
  (Allura 1137 / 765 mm, Azurion 1195 / 810 mm; keyed by `system` from step 08, `default` otherwise). The detector pixel
  spacing is derived so that the camera describes the exported 256x256 grid: isocentre pixel (0.5417 mm LCA,
  0.4583 mm RCA) x SID / SOD, e.g. 0.8051 / 0.6812 mm for Allura. Leave `imager_pixel_spacing_mm` at `null`: the real
  detector pixel (0.3418 mm) belongs to the original 512 frames, not to the resampled grid; a value that does not match
  the grid is reported as a warning.
- `camera: "training"`: the checkpoint's camera (900 / 750 mm, 0.65 / 0.55 mm).

The reprojections and overlays use the same camera, and `summary.csv` records it (`camera`, `camera_sid_mm`,
`camera_sod_mm`, `camera_pixel_mm`). With matched isocentre pixel size the two cameras differ only in perspective
(source–isocentre 765 vs 750 mm): on a synthetic Allura acquisition (real SID/SOD/pixel, through step 08) the true
artery reprojects at Dice 0.974 with either camera and no 2D shift.

Config keys (`configs/real_xray/{lca,rca}.json`):

| Key | Default | Meaning |
| --- | --- | --- |
| `output_mode` | `full` | `full`: GIFs, per-view overlays and `summary.png`; `summary`: only `summary.png` per case |
| `camera` | `acquisition` | `acquisition` or `training` (see above) |
| `acquisition_geometry` | Allura 1137/765, Azurion 1195/810 mm | SID / SOD per system (`sid_mm`, `sod_mm`), plus `default` |
| `imager_pixel_spacing_mm` | null | null = derived from the exported grid (recommended) |
| `num_views` | 2 | input views given to the model (1–7) |
| `view_selection` | `max_separation` | `max_separation`: one view per direction (the one with most vessel), then the set with the largest smallest angle between views (capped at 90°); `first`: file order |
| `input_views` | `auto` | or a list of view indices for every case, or `{"Study01126_LCA": [0, 2]}` |
| `input_recentering_iterations` | 0 | experimental: shift the input views onto the reprojection of the prediction and predict again (max 40 px per iteration) |
| `align_reprojection` | true | overlays use the reprojection moved by its best 2D translation; both Dice values are reported |
| `branch_threshold` | 0.5 | branch-existence probability threshold |
| `save_gifs`, `gif_frames`, `gif_fps` | true, 60, 12 | rotating 3D surfaces |
| `case_ids`, `limit`, `overwrite` | null, –, false | case selection; finished cases are skipped unless `overwrite` |

Every case is independent; an error is recorded in `summary.csv` and the run continues.

## 3. Output

Two modes (`output_mode`, or `OUTPUT_MODE=` for the sbatch):

- **`summary`** - one PNG per case, `<case>/summary.png`, also collected in `<output_dir>/_summary/<case>.png` for quick
  browsing. One row per view (its keyframe), six panels from left to right, all in the keyframe's original orientation
  and resolution:
  1. raw X-ray, 2. raw X-ray + reprojected 3D surface (orange), 3. segmentation mask (the reconstruction input),
  4. segmentation vs reprojection (green both, red reprojection only, white mask only), 5. the predicted 3D surface seen
  along this view's X-ray direction (shaded, same camera and position as the reprojection), 6. the same coloured by the
  predicted radius (red smallest - green largest, colour bar in mm).
  The row label gives the view (e.g. `LAO 30, CRA 0`), **`INPUT to 3D reconstruction`** (green, framed) or
  `NOT input (held-out)` (orange), the keyframe (frame / frames) and the view's Dice. The title gives the case, the
  segmentation method, the system and pixel size, the camera, the inputs and the mean input / held-out Dice.
- **`full`** (default) - the summary PNG plus:

```text
<output_dir>/
  summary.csv                               one row per case: system, camera, input views, mean reprojection Dice
  run.json
  _summary/<case>.png                       every case's summary sheet in one folder
  <case_name>/
    summary.png                             the summary sheet (both modes)
    surface_3d.gif                          rotating predicted 3D surface                         (full only)
    surface_3d_radius.gif                   the same, coloured by predicted radius                (full only)
    reprojection/input_views_on_xray.png    reprojection onto the input views, over the raw X-ray (full only)
    reprojection/input_views_on_mask.png    ... against the segmentation                          (full only)
    reprojection/heldout_views_on_xray.png  views not given to the model                          (full only)
    reprojection/heldout_views_on_mask.png                                                        (full only)
    reprojection/views/v<i>_<input|heldout>_<xray|mask>.png   full-resolution single views        (full only)
    prediction_refined.npz, prediction_coarse.npz             vessel code [M,200,4] (centred mm), branch_exists, camera
    case_summary.json                       per-view Dice (raw and translation-aligned) and the alignment shift
```

Overlays and 3D panels are drawn on the full-resolution keyframe in its original (DICOM display) orientation. The 3D
panels are rendered with the same camera as the reprojection (perspective from the source, surfaces nearer the detector
in front, as in the clinical display); on synthetic data their footprint matches the segmentation at Dice 0.99. A
held-out view with the same angles as an input view (another video at that angle) is labelled "same angle as an input".
`summary` mode takes about 10 s per case on CPU for the drawing (plus the GPU forward pass); `full` about twice that.

## 4. Reading the reprojection numbers

- `dice_*` compares the reprojected surface with the segmentation mask in model space, exactly as exported.
- `dice_*_aligned` first moves the reprojection by its best 2D translation (up to 48 px). Without a calibrated
  isocentre (table panning, or `--center mask_centroid`), a correct 3D shape can still be off by a translation; with
  the true 3D artery rendered as synthetic angiograms and centred per view, the aligned Dice is 0.96–0.98 in input and
  held-out views while the raw Dice is 0.14–0.66. The aligned value measures shape agreement; the shift is stored per
  view.
- On held-out views both numbers are an honest check of the 3D shape: the model never saw these masks.

## 5. Checks before trusting a full run

1. Open a few `_qa/<case>.png` sheets from the exporter: row 2 shows each view in model space.
2. Run 10–20 cases and compare the held-out aligned Dice for `--orientation rot180` (default) against `none`;
   the right orientation gives clearly higher held-out agreement. Do the same for RCA with
   `--angle_convention paper` (default) against `code`, and for `--center image_center` (default) against
   `mask_centroid`.
3. Compare Allura and Azurion cases in `summary.csv` (`system` column). For Azurion the image centre is not the
   isocentre (varying crop), so expect larger 2D shifts there.

Real angiograms differ from the CT-derived training projections (segmentation errors, catheter, overlap, unknown
magnification), so treat these reconstructions as qualitative until validated against a reference.
