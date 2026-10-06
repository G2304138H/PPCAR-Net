# Annotated CT input contract

Specify the artery explicitly: `--artery-type rca` or `--artery-type lca` for preparation, and `artery_type` in the evaluation JSON. This selection determines the seven camera views.

This loader accepts a defined NPZ schema. Users must convert their own files to it; the loader does not discover patient position, anatomical axes, artery identity, or label values.

## Coordinates

All physical values are millimetres in a LAS basis: X toward the patient's left, Y anterior, Z superior. Do not add a standing/supine rotation. A `coordinate_frame="LAS"` string declares the convention; it cannot prove that the array was oriented correctly. Verify orientation against the source imaging metadata and landmarks before export.

`vol[i,j,k]` is a binary annotation of a single RCA or LCA component. It contains only 0 and 1. `spacing` follows these three array axes. The affine maps voxel centres to absolute LAS coordinates:

```text
xyz_mm = index_to_world_affine @ [i, j, k, 1]
```

If an affine is omitted, `direction @ diag(spacing)` and `origin_mm` define it. Direction defaults to identity and origin to zero, which is valid only for an annotation already stored along LAS X/Y/Z axes. The affine must be finite, invertible, and orthogonal apart from scaling. If spacing and affine are both present, they must agree. Reorient/resample data in other frames before use. Raw CT intensities, multiple artery labels, and implicit axis inference are unsupported.

## Isocentre

`projection_center_offset_mm` is required. It is the absolute LAS position subtracted before projection; the camera receives `(xyz_mm - offset_mm) * 0.001` in metres. The same offset is added back to model outputs for volume evaluation.

For compatibility with the research data, the isocentre is the reference artery surface centroid: the main RCA surface for RCA; the combined LM–LAD and LCX surfaces for LCA. Determine it during annotation preparation, or reuse the stored projection offset of the prepared case. Do not substitute the full-volume midpoint or a centroid of all side branches and expect identical inputs. The loader cannot infer which voxels belong to the required main paths.

## Optional fields

| Key | Contract |
| --- | --- |
| `index_to_world_affine` | `[4,4]` voxel-centre-to-LAS transform in mm |
| `origin_mm` | `[3]`, used without affine |
| `direction` | `[3,3]` orthonormal columns, used without affine |
| `artery_type` | Scalar `rca` or `lca`; checked against the selected model |
| `vessel_code_mm` | `[M,N,4]` absolute LAS XYZ and radii; zero-padded absent branches |
| `theta_deg`, `phi_deg` | Both `[7]`, explicit projector angles; override the fixed defaults together |

Every active vessel-code branch must have positive radii at all sample points. Preserve the research ordering: RCA main path first; LCA LM–LAD first and LCX second, then optional branches in canonical proximal-to-distal order. The paper uses N=200, M=7 for RCA and M=13 for LCA. Centreline annotations must occupy the same physical frame as the volume.

## Projection protocol

The renderer extracts a closed surface from the annotation with marching cubes, converts vertices to LAS millimetres, centres it, and renders seven masks through the transferred cone-beam camera. Output size is 256×256, source-to-detector distance 0.90 m, source-to-isocentre distance 0.75 m; pixel spacing is 0.55 mm for RCA and 0.65 mm for LCA. The source mesh rasterizer includes its original closing and thresholded smoothing steps.

| View index | RCA theta / phi (degrees) | LCA theta / phi (degrees) |
| --- | --- | --- |
| 0 | 20 / 70 | 115 / 125 |
| 1 | −60 / 90 | 50 / 70 |
| 2 | −40 / 80 | 85 / 120 |
| 3 | 75 / 80 | 90 / 100 |
| 4 | 0 / 65 | 130 / 65 |
| 5 | −20 / 90 | 70 / 60 |
| 6 | 75 / 110 | 85 / 50 |

The table specifies the projector theta/phi angles used for each artery. No random jitter is applied during evaluation.

## Example export

```python
np.savez_compressed(
    "case_0001.npz",
    vol=artery_mask.astype(np.uint8),
    spacing=np.asarray([sx, sy, sz], dtype=np.float32),
    coordinate_frame=np.asarray("LAS"),
    index_to_world_affine=voxel_to_las_mm,
    projection_center_offset_mm=reference_surface_centroid_mm,
    artery_type=np.asarray("rca"),
    vessel_code_mm=annotated_branches_xyzr_mm,  # optional except full paper metrics
)
```

Only real numeric/string arrays are supported; object arrays and pickle payloads are not accepted by the CT loader.
