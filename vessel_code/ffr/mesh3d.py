"""Watertight lumen surface of a vessel code for 3D CFD.

The lumen is the zero level set of a signed distance field, the union of spheres
centred on a densely resampled centreline (radius = local lumen radius), truncated
by planes at the inlet (main ostium) and at every branch end so that inlet/outlet
caps are flat and perpendicular to the centreline. Marching cubes gives a closed
triangle surface; faces on a cap plane are labelled with that patch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LumenSurface:
    vertices_mm: np.ndarray          # [V,3]
    faces: np.ndarray                # [F,3]
    face_patch: np.ndarray           # [F] patch id
    patch_names: list                # id -> name ("wall", "inlet", "outlet_b<k>")
    ends: dict                       # name -> {"point_mm","normal","radius_mm","branch"}
    voxel_mm: float
    checks: dict = field(default_factory=dict)


def _resample(xyz: np.ndarray, r: np.ndarray, step: float) -> tuple:
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(np.ceil(s[-1] / step)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(t, s, xyz[:, k]) for k in range(3)], axis=1)
    return out, np.interp(t, s, r)


def _end_frame(xyz: np.ndarray, at_start: bool, reach_mm: float = 1.5) -> tuple:
    """Point and outward unit normal at a branch end (tangent averaged over ``reach_mm``)."""
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if at_start:
        k = int(np.searchsorted(s, reach_mm))
        k = min(max(k, 1), len(xyz) - 1)
        n = xyz[0] - xyz[k]
        p = xyz[0]
    else:
        k = int(np.searchsorted(s, s[-1] - reach_mm))
        k = min(max(k, 0), len(xyz) - 2)
        n = xyz[-1] - xyz[k]
        p = xyz[-1]
    return p, n / max(np.linalg.norm(n), 1e-12)


def build_lumen_surface(result, *, voxel_mm: float | None = None, cap_trim_mm: float | None = None) -> LumenSurface:
    """Surface from an :class:`FFRResult` (uses its centrelines and lumen radii)."""
    from skimage.measure import marching_cubes

    branches = list(result.branches.values())
    r_min = min(float(b.radius_mm[b.start:].min()) for b in branches)
    voxel = float(voxel_mm) if voxel_mm else float(np.clip(r_min / 4.0, 0.08, 0.35))
    # Ends: trim each end slightly inwards so the cap sits on a regular tube section.
    trim = voxel * 2 if cap_trim_mm is None else cap_trim_mm
    ends = {}
    main = result.branches[0]
    p, n = _end_frame(main.xyz_mm, at_start=True)
    ends["inlet"] = {"point_mm": p - n * trim, "normal": n, "radius_mm": float(main.radius_mm[0]), "branch": 0}
    for b in branches:
        p, n = _end_frame(b.xyz_mm, at_start=False)
        ends[f"outlet_b{b.index}"] = {"point_mm": p - n * trim, "normal": n,
                                      "radius_mm": float(b.radius_mm[-1]), "branch": b.index}
    centres, radii = [], []
    for b in branches:
        c, r = _resample(b.xyz_mm, b.radius_mm, 0.5 * voxel)
        centres.append(c)
        radii.append(r)
    centres = np.concatenate(centres)
    radii = np.concatenate(radii)
    pad = float(radii.max()) + 4 * voxel
    lo = centres.min(0) - pad
    hi = centres.max(0) + pad
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    if np.prod(shape) > 250_000_000:
        raise MemoryError(f"surface grid {tuple(shape)} too large; pass a larger voxel_mm")
    far = 4.0 * pad
    sdf = np.full(shape, far, dtype=np.float32)
    axes = [lo[k] + voxel * np.arange(shape[k]) for k in range(3)]
    for c, r in zip(centres, radii):
        reach = r + 2 * voxel
        i0 = np.maximum(np.floor((c - reach - lo) / voxel).astype(int), 0)
        i1 = np.minimum(np.ceil((c + reach - lo) / voxel).astype(int) + 1, shape)
        gx = axes[0][i0[0]:i1[0]][:, None, None] - c[0]
        gy = axes[1][i0[1]:i1[1]][None, :, None] - c[1]
        gz = axes[2][i0[2]:i1[2]][None, None, :] - c[2]
        d = np.sqrt(gx * gx + gy * gy + gz * gz) - r
        block = sdf[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        np.minimum(block, d.astype(np.float32), out=block)
    # Planar caps: remove material beyond each end plane, locally around the end.
    for e in ends.values():
        p, n, r = e["point_mm"], e["normal"], e["radius_mm"]
        reach = 2.0 * r + 4 * voxel
        i0 = np.maximum(np.floor((p - reach - lo) / voxel).astype(int), 0)
        i1 = np.minimum(np.ceil((p + reach - lo) / voxel).astype(int) + 1, shape)
        gx = axes[0][i0[0]:i1[0]][:, None, None] - p[0]
        gy = axes[1][i0[1]:i1[1]][None, :, None] - p[1]
        gz = axes[2][i0[2]:i1[2]][None, None, :] - p[2]
        along = gx * n[0] + gy * n[1] + gz * n[2]
        dist = np.sqrt(gx * gx + gy * gy + gz * gz)
        # Only inside a cylinder around the end axis, so neighbouring vessels are untouched.
        radial = np.sqrt(np.maximum(dist ** 2 - along ** 2, 0.0))
        local = (radial < 1.5 * r + 2 * voxel) & (along > -0.5 * r) & (dist < reach)
        block = sdf[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        cut = np.where(local, np.maximum(block, along.astype(np.float32)), block)
        sdf[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = cut
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(voxel, voxel, voxel))
    verts = verts + lo
    del sdf
    # Patch labels
    names = ["wall"] + list(ends.keys())
    patch = np.zeros(len(faces), dtype=np.int32)
    centroid = verts[faces].mean(axis=1)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    for k, e in enumerate(ends.values(), start=1):
        p, n, r = e["point_mm"], e["normal"], e["radius_mm"]
        rel = centroid - p
        along = rel @ n
        radial = np.linalg.norm(rel - along[:, None] * n, axis=1)
        on = (np.abs(along) < 0.6 * voxel) & (radial < 1.3 * r + voxel) & (np.abs(fn @ n) > 0.7)
        patch[on & (patch == 0)] = k
    surface = LumenSurface(verts, faces.astype(np.int64), patch, names, ends, voxel)
    surface.checks = check_surface(surface)
    return surface


def check_surface(surface: LumenSurface) -> dict:
    """Watertightness, cap areas vs pi r^2, enclosed volume."""
    v, f = surface.vertices_mm, surface.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    volume = float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)
    caps = {}
    for k, name in enumerate(surface.patch_names):
        if k == 0:
            continue
        e = surface.ends[name]
        a = float(area[surface.face_patch == k].sum())
        caps[name] = {"area_mm2": a, "expected_mm2": float(np.pi * e["radius_mm"] ** 2),
                      "faces": int(np.sum(surface.face_patch == k))}
    return {
        "watertight": bool(np.all(counts == 2)),
        "non_manifold_edges": int(np.sum(counts != 2)),
        "faces": int(len(f)), "vertices": int(len(v)),
        "volume_mm3": abs(volume), "wall_area_mm2": float(area[surface.face_patch == 0].sum()),
        "voxel_mm": surface.voxel_mm, "caps": caps,
        "all_caps_found": bool(all(c["faces"] > 0 for c in caps.values())),
    }


def write_stl(surface: LumenSurface, path, *, scale: float = 1e-3) -> None:
    """Multi-solid ASCII STL (one solid per patch). ``scale`` 1e-3 writes metres."""
    v = surface.vertices_mm * scale
    with open(path, "w") as fh:
        for k, name in enumerate(surface.patch_names):
            faces = surface.faces[surface.face_patch == k]
            if len(faces) == 0:
                continue
            fh.write(f"solid {name}\n")
            a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
            n = np.cross(b - a, c - a)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            for i in range(len(faces)):
                fh.write(f"facet normal {n[i,0]:.6e} {n[i,1]:.6e} {n[i,2]:.6e}\n outer loop\n")
                for p in (a[i], b[i], c[i]):
                    fh.write(f"  vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n")
                fh.write(" endloop\nendfacet\n")
            fh.write(f"endsolid {name}\n")
