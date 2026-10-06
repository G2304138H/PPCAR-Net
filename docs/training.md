# Training and supplied checkpoints

Run from the repository root. `configs/rca` and `configs/lca` each contain `train_coarse.json`, `train_geometry.json`, and `train_radius.json`. The public `python -m vessel_code.train --config ...` command selects the matching engine. JSON `extends` is resolved relative to the configuration file; data and output paths are relative to the working directory.

## What the supplied files establish

`ckpt/{rca,lca}/best.pt` stores the full trained reconstruction model, including both refiners and the frozen centreline-evidence network. It does not store the frozen VGGT-1B image backbone. The full model loads strictly with seven RCA or thirteen LCA queries.

The supplied resolved configurations describe the final dedicated radius-refiner runs. They include grouped stenosis/counterfactual training, candidate sampling, spatial VGGT evidence, a frozen centreline-probability encoder, and `all_views` backbone context. The transfer preserves these semantics. They must not be replaced with simpler paper-default settings while loading the supplied weights.

`configs/{rca,lca}/resolved_config.json` records selected architecture and training settings extracted from the exact resolved run configurations supplied with the checkpoints. This public summary omits multi-megabyte run manifests and machine-specific paths. The complete original `ckpt/{rca,lca}/resolved_config.json` files remain unchanged locally.

The final radius manifests provide explicit patient assignments, exported as `radius_split_from_checkpoint.json`. They contain 601/75/76 RCA and 602/75/76 LCA training/validation/test patient groups. These differ from counts reported elsewhere in the original run metadata and from the paper's nominal 600/75/75 protocol. Use the actual run records for provenance; do not relabel these as the paper cohort. Original coarse and geometry split files have not been supplied, so those recipes require `data/splits/{rca,lca}.json` from the original runs or an explicitly defined new dataset split.

## Prepared training samples

The CT evaluation schema is not a replacement for supervised branch labels. The existing parametric training loader consumes prepared NPZs. Core fields include:

| Field | Shape / convention |
| --- | --- |
| `images` | `[V,256,256]`, mask values in [0,1] |
| `theta_deg`, `phi_deg` | `[V]`, source projector angles |
| `view_features` | `[V,4]`, sine/cosine angle encoding from the transferred camera utility |
| `artery` | `[M,200,4]`, absolute XYZ and radius in **metres** |
| `projection_center_offset` | `[3]`, camera isocentre in **metres** |
| `raw_vessel_code_mm` | `[M,200,4]`, curated raw target in absolute millimetres |
| `reconstructed_vessel_code_mm` | `[M,200,4]`, fitted B-spline XYZ with dense radii in millimetres |
| `centerline_control_points_mm` | `[M,20,3]`, endpoint-constrained controls in global millimetres |
| `branch_exists` | `[M]`, active branch mask |
| `point_valid_mask` | `[M,200]`, target-validity mask |
| `centerline_control_point_frame` | `[M]` strings `global_mm`, one frame declaration per branch |

The loader projects/centres targets consistently using the supplied offset. Preserve source preprocessing metadata when transferring prepared NPZs; do not rename or discard variant fields. The authoritative validation lives in `vessel_code/parametric/data.py` (`load_parametric_target`, `load_paired_items`, and `load_branch_variant_metadata`).

The default coarse/geometry recipes group branch-prefix variants by patient: one case directory contains its complete tree and progressive branch-removal variants, with matching masks and source `branch_subset_*` metadata. All variants stay in the same split and use shared selected views. For a deliberate single-variant dataset, set `branch_variant_group_training=false`; this changes the augmentation protocol.

The radius stage uses `vessel_code/parametric/radius_refiner_data.py`, which expects the source Stage-3/3.5 original/stenosis-removed/stenosis-strengthened group schema and visibility metadata. Set `raw_image_dataset_dir` and `radius_refiner_stage3_1_visibility_report` to those prepared inputs. Preserve all geometry, variant, candidate, and visibility fields; a binary CT mask alone cannot reconstruct that curated training supervision. The scope of this release is training/evaluation and the CT evaluation adapter, not reimplementation of the source annotation/curation tools.

## Stage behaviour

1. **Coarse:** train the independent branch-query predictor, existence heads, B-spline controls, and dense radii. Refiners are disabled. Output is `runs/{artery}_coarse`.
2. **Geometry:** initialise from that coarse checkpoint, freeze it, and train the geometry refiner. A small temporary standalone checkpoint is extracted from `ckpt/{artery}/best.pt` to initialise the frozen centreline-evidence encoder; this uses already-trained evidence weights rather than claiming fully independent training from scratch. Output is `runs/{artery}_geometry`.
3. **Radius:** initialise from the geometry checkpoint; freeze the predictor and geometry refiner; train the dedicated radius module with the transferred group objective. Output is `runs/{artery}_radius`.

Resume a stage using `--resume /path/to/latest.pt`, or `--resume` for that stage's latest checkpoint. Training checkpoints are trusted local PyTorch files with optimizer/config/RNG state. The transferred loading calls explicitly support this complete format.

The supplied checkpoint configurations are the resolved settings of the actual runs. The separate `train_coarse.json`, `train_geometry.json`, and `train_radius.json` files are the release training entry points, with local data/output paths and stage-specific settings. Validation selects checkpoints; automatic test-set evaluation is disabled in these entry points.
