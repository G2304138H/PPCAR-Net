<h1 align="center">PPCAR-Net: Projection-Refined Parametric 3D Coronary Artery Reconstruction from Sparse X-ray Angiographic Views</h1>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b" alt="arXiv: coming soon">
  <a href="https://G2304138H.github.io/PPCAR-Net/"><img src="https://img.shields.io/badge/Project%20Page-website-4c9c2e" alt="Project Page"></a>
</p>

<!-- Add the arXiv URL once the preprint is public. -->

<p align="center">
  Yu Ren<sup>1,2</sup>, Hwee Kuan Lee<sup>1</sup>, Tat-Jen Cham<sup>2</sup>, Jonathan Yap<sup>3</sup>, Khung Keong Yeo<sup>3</sup>
</p>

<p align="center">
  <sup>1</sup>Bioinformatics Institute, Agency for Science, Technology and Research (A*STAR)<br>
  <sup>2</sup>College of Computing and Data Science, Nanyang Technological University<br>
  <sup>3</sup>National Heart Centre Singapore
</p>

```bibtex
% BibTeX citation will be added when the arXiv preprint is available.
```

## Overview

PPCAR-Net (Projection-Refined Parametric 3D Coronary Artery Reconstruction Network) reconstructs explicit, branch-structured 3D coronary arteries from a small number (1–7) of calibrated 2D artery segmentation masks, with separate models for the right coronary artery (RCA) and left coronary artery (LCA). Each artery is represented by branch-existence predictions, B-spline control points defining continuous branch centrelines, and dense radii associated with sampled centreline points. We call this representation **vessel code**. A coarse predictor combines frozen VGGT image features with learned branch queries to estimate the initial vessel code. Projection-guided geometry and radius refiners then project this coarse reconstruction into the input views, sample local image evidence, and apply learned residual corrections to the centrelines and radii. The resulting representation directly provides centreline geometry and vessel thickness for 3D surface reconstruction, without requiring graph extraction from a predicted volume.

Together, our model provides:

- **Explicit and editable anatomy:** branch presence, centreline geometry and local vessel radius are directly accessible for visualization and downstream analysis.
- **Continuous branch geometry:** B-spline curves preserve connectivity within each branch without a separate centreline or graph-extraction step.
- **Sparse-view reconstruction:** learned coronary anatomy patterns complement the input evidence when depth and branch structure are ambiguous in a few projections.
- **Fast, evidence-guided refinement:** learned geometry and radius corrections adapt the coarse prediction to the observed views without iterative per-case optimisation.

This repository includes our model and pretrained checkpoints, two runnable examples, and instructions for preparing your own data for inference.

## Installation

Use Python 3.10 or later in an isolated environment. Install the PyTorch build appropriate for your machine, then run from the repository root:

```bash
python -m pip install -r requirements.txt
```

This installs the package and its VGGT dependency. The first model run obtains `facebook/VGGT-1B` through its pretrained loader; the reconstruction checkpoint does not include this image backbone. CUDA is recommended for practical inference.

## Run the bundled examples

Two prepared examples are included, each containing seven projection masks, their camera directions and reference vessel annotations:

- `examples/rca/rca_0457.npz`
- `examples/lca/lca_0457.npz`

The reconstruction checkpoints are included at `ckpt/rca/best.pt` and `ckpt/lca/best.pt`. Run either command from the repository root:

```bash
# RCA example
python -m vessel_code.evaluate --config configs/rca/evaluate_example.json

# LCA example
python -m vessel_code.evaluate --config configs/lca/evaluate_example.json
```

These configurations use two input views and save coarse/refined prediction NPZs, projection-overlay figures and 3D reconstruction figures under `outputs/rca_example/` or `outputs/lca_example/`. Set `num_views` from 1 to 7 to use the first N views; choose a fresh `output_dir` for each run.

The examples use `input_format: "prepared_projections"`: evaluation reads the supplied masks and camera directions directly. They do not contain a CT volume, so they support visualization rather than the volume-based paper metrics.

## Evaluate your own annotated CT

For your own annotated RCA volume, follow the input contract below and use the annotated-CT visualization configuration.

### 1. Prepare the input

Place your files at:

```text
data/evaluation/rca/example.npz
ckpt/rca/best.pt
```

**Convert the annotation to the required coordinate system before evaluation.** The loader cannot infer arbitrary NPZ formats, patient orientation or label conventions. The file must contain one annotated RCA component, rather than raw CT intensities or both artery systems together.

| Key | Required content |
| --- | --- |
| `vol` | Binary artery mask with shape `[I, J, K]` |
| `spacing` | Three positive voxel spacings in millimetres, in array-axis order |
| `coordinate_frame` | Scalar string `LAS`: +X left, +Y anterior, +Z superior |
| `projection_center_offset_mm` | Three-coordinate projection isocentre in the same absolute LAS frame |

Use `index_to_world_affine` to provide the voxel-index-to-LAS transform. Without it, `origin_mm` defaults to zero and `direction` to identity, so the array axes must already follow LAS X/Y/Z order. The isocentre must follow the pipeline's main-artery convention. See the full [input data specification](docs/data_format.md) before converting your data.

### 2. Configure evaluation

Set `configs/rca/evaluate_visualization.json` to:

```json
{
  "mode": "visualization",
  "artery_type": "rca",
  "checkpoint": "ckpt/rca/best.pt",
  "input_dir": "data/evaluation/rca",
  "output_dir": "outputs/rca_visualization",
  "case_ids": ["example"],
  "device": "auto",
  "num_views": 2
}
```

`case_ids` selects the NPZ filename without its extension. `artery_type` is required and determines the camera preset and matching model. `num_views` can be an integer from 1 to 7; prediction uses the first N generated views automatically. Individual view indices cannot be configured.

### 3. Run

```bash
python -m vessel_code.evaluate --config configs/rca/evaluate_visualization.json
```

The output contains all seven projection masks, coarse and refined prediction NPZs, projection overlays and a 3D reconstruction figure. Choose a new `output_dir` for another run; completed output directories are not overwritten.

For an LCA case, use the corresponding configuration in `configs/lca/`, set `artery_type` to `lca`, and supply the LCA checkpoint and annotation.

## Repository layout

```text
vessel_code/               Parametric model, refiners, data loader and evaluation
configs/{rca,lca}/          Model and evaluation configurations
ckpt/                      Bundled RCA and LCA reconstruction checkpoints
examples/                  Bundled RCA/LCA evaluation cases and synthetic loader example
figures/                   Research figures and visual examples
project-page/              Offline research project page
```

## License

Unless otherwise indicated, original VesselCode materials are licensed under the [MIT License](LICENSE). Third-party code, pretrained weights, datasets and media retain their respective licenses, including VGGT.
