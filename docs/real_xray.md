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

- **Angles.** The exporter follows the convention of this code and its checkpoints (the clinical labels of the camera
  tables in `docs/data_format.md`): LCA `theta = wrap(90° − primary)`, RCA `theta = −primary`, both `phi = 90° − secondary`
  (primary + = LAO, secondary + = CRA). Every training anchor label maps exactly onto its projector angles.
  The paper (App. A.1) writes `theta = wrap(90° − primary)` for RCA as well, 90° from the RCA camera table; the
  exporter's `--angle_convention paper` reproduces that for comparison. Each case file stores `nearest_anchor_deg`, the
  angle between each view and the closest training view.
- **Orientation.** In the model's images, rows point towards the patient's head and columns towards the patient's
  right; DICOM display has the head at the top and the patient's left on the right. The exporter therefore rotates
  every frame by 180° (`--orientation rot180`).
- **Scale.** The model images have 0.65 mm (LCA) / 0.55 mm (RCA) detector pixels with SID 0.90 m and source–isocentre
  0.75 m, i.e. 0.542 / 0.458 mm per pixel at the isocentre. The CoronaryDominance files keep no SID, SOD or pixel
  spacing, so the exporter assumes 0.28 mm per pixel at the isocentre for the 512×512 frames
  (`--src_iso_pixel_mm`), one scale per case. `--scale_mode extent` instead matches the median vessel extent to that
  of the training masks.
- **Centring.** Training projections are centred on the 3D main-artery centroid (the isocentre is moved there before
  projecting). Real views keep their raw C-arm geometry by default: the image centre is the isocentre, which is
  consistent between views as long as the table is not moved, but the artery is not centred the way the training
  projections are. `--center mask_centroid` instead centres each view on its own mask (centred, but not consistent
  between views). See section 4.

## 2. Run

```bash
python -m vessel_code.real_xray.infer --config configs/real_xray/lca.json \
    --input_dir /dataset/reny0012/angioseg/recon_inputs/nnunet_A/lca \
    --output_dir outputs/real_xray/nnunet_A/lca
```

Config keys (`configs/real_xray/{lca,rca}.json`):

| Key | Default | Meaning |
| --- | --- | --- |
| `num_views` | 2 | input views given to the model (1–7) |
| `view_selection` | `max_separation` | `max_separation`: one view per direction (the one with most vessel), then the set with the largest smallest angle between views (capped at 90°); `first`: file order |
| `input_views` | `auto` | or a list of view indices for every case, or `{"Study01126_LCA": [0, 2]}` |
| `input_recentering_iterations` | 0 | experimental: shift the input views onto the reprojection of the prediction and predict again (max 40 px per iteration) |
| `align_reprojection` | true | overlays use the reprojection moved by its best 2D translation; both Dice values are reported |
| `branch_threshold` | 0.5 | branch-existence probability threshold |
| `save_gifs`, `gif_frames`, `gif_fps` | true, 60, 12 | rotating 3D surfaces |
| `case_ids`, `limit`, `overwrite` | null, –, false | case selection; finished cases are skipped unless `overwrite` |

`--shard K --num_shards N` splits the cases for Slurm arrays. Every case is independent; an error is recorded in
`summary.csv` and the run continues.

Slurm example (one GPU; `--array=0-3` with `--num_shards 4` to split):

```bash
#!/bin/bash
#SBATCH -J ppcar_realxray
#SBATCH -o /export/home2/reny0012/result/runlog/ppcar_realxray_%A_%a.log
#SBATCH -e /export/home2/reny0012/result/runlog/ppcar_realxray_%A_%a.err
#SBATCH -p RTXA6Kq
#SBATCH -w node11
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0
set -euo pipefail
module load python3
source /export/home2/reny0012/vir_env/<your vessel-code venv>/bin/activate
module load cuda12.5/toolkit/12.5.1
cd /export/home2/reny0012/code/PPCAR-Net
METHOD=${METHOD:-nnunet_A}; ART=${ART:-lca}
python -m vessel_code.real_xray.infer --config configs/real_xray/$ART.json \
  --input_dir /dataset/reny0012/angioseg/recon_inputs/$METHOD/$ART \
  --output_dir /dataset/reny0012/angioseg/recon_outputs/$METHOD/$ART \
  --shard ${SLURM_ARRAY_TASK_ID:-0} --num_shards ${NUM_SHARDS:-1}
```
Submit with e.g. `sbatch --export=ALL,METHOD=xravos_A,ART=rca realxray.sbatch`.

## 3. Output

```text
<output_dir>/
  summary.csv                               one row per case: input views, mean reprojection Dice
  run.json
  <case_name>/
    surface_3d.gif                          rotating predicted 3D surface
    surface_3d_radius.gif                   the same, coloured by predicted radius
    reprojection/input_views_on_xray.png    reprojection onto the input views, over the raw X-ray
    reprojection/input_views_on_mask.png    ... against the segmentation (green both, red prediction only, white mask only)
    reprojection/heldout_views_on_xray.png  views not given to the model
    reprojection/heldout_views_on_mask.png
    reprojection/views/v<i>_<input|heldout>_<xray|mask>.png   full-resolution single views
    prediction_refined.npz, prediction_coarse.npz             vessel code [M,200,4] (centred mm), branch_exists
    case_summary.json                       per-view Dice (raw and translation-aligned) and the alignment shift
```

Overlays are drawn on the full-resolution keyframe in its original (DICOM display) orientation. A held-out view
with the same angles as an input view (another video at that angle) is labelled "same angle as an input".

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
   `--angle_convention code` (default) against `paper`, and for `--center image_center` (default) against
   `mask_centroid`.
3. If arteries look too small or too large relative to the training masks, change `--src_iso_pixel_mm` or use
   `--scale_mode extent` in the exporter.

Real angiograms differ from the CT-derived training projections (segmentation errors, catheter, overlap, unknown
magnification), so treat these reconstructions as qualitative until validated against a reference.
