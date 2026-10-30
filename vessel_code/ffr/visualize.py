"""Figures for FFR along the centreline.

* ``ffr_3d.png``         lumen surface coloured by FFR (two viewpoints), lesion and distal labels
* ``ffr_pullback.png``   per branch: FFR vs distance from the ostium (with Monte Carlo band) and
                         lumen vs reference diameter (lesions shaded) - a virtual pressure-wire pullback
* ``ffr_projections.png`` FFR-coloured centrelines overlaid on the input projections (optional)
* ``ffr_interactive.html`` rotatable 3D view with hover read-out (plotly from cdnjs)
* ``ffr_centreline.csv``  every centreline sample with xyz, radii, pressure, flow, FFR

Colour map: diverging around the clinical threshold - red below 0.80 (ischaemic),
neutral grey at 0.80, blue towards 1.0.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FFR_THRESHOLD = 0.80
FFR_RANGE = (0.50, 1.00)
# Diverging pair (red <-> blue) with a neutral grey midpoint at the threshold.
_STOPS = [(0.50, "#8f1d1c"), (0.65, "#c63837"), (0.75, "#e98a89"), (0.80, "#d9d8d3"),
          (0.85, "#9ec5f4"), (0.92, "#3987e5"), (1.00, "#184f95")]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BAND = "#b7d3f6"
LESION = "#f6d4d3"


def ffr_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    lo, hi = FFR_RANGE
    return LinearSegmentedColormap.from_list(
        "ffr", [((v - lo) / (hi - lo), c) for v, c in _STOPS])


def ffr_norm():
    from matplotlib.colors import Normalize
    return Normalize(*FFR_RANGE, clip=True)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------
# Tube geometry


def tube_rings(xyz: np.ndarray, radius: np.ndarray, sides: int = 16) -> np.ndarray:
    """[N, sides, 3] ring vertices around a centreline using parallel-transported frames."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    t = np.gradient(xyz, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t[0], ref)
    u /= np.linalg.norm(u)
    us = np.zeros((n, 3))
    us[0] = u
    for i in range(1, n):
        u = u - np.dot(u, t[i]) * t[i]
        norm = np.linalg.norm(u)
        u = us[i - 1] if norm < 1e-9 else u / norm
        us[i] = u
    vs = np.cross(t, us)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return xyz[:, None, :] + radius[:, None, None] * (
        np.cos(ang)[None, :, None] * us[:, None, :] + np.sin(ang)[None, :, None] * vs[:, None, :])


def _tube_quads(rings: np.ndarray) -> np.ndarray:
    n, m, _ = rings.shape
    quads = []
    for i in range(n - 1):
        for j in range(m):
            k = (j + 1) % m
            quads.append([rings[i, j], rings[i, k], rings[i + 1, k], rings[i + 1, j]])
    return np.asarray(quads)


# --------------------------------------------------------------------------------------
# Figures


def _lesion_list(result):
    out = []
    for b in result.branches.values():
        for les in b.lesions:
            out.append((b, les))
    return out


def _label_points(result):
    """Distal FFR at every branch end and a short tag (L1, L2, ...) at every lesion throat."""
    labels = []
    for b in result.branches.values():
        labels.append((b.xyz_mm[-1], f"{b.distal_ffr:.2f}", b.distal_ffr))
    for k, (b, les) in enumerate(_lesion_list(result), 1):
        labels.append((b.xyz_mm[les.imin], f"L{k}", None))
    return labels


def plot_3d(result, path: Path, *, title: str = "", stride: int = 2) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cmap, norm = ffr_cmap(), ffr_norm()
    polys, colors = [], []
    pts = []
    for b in result.branches.values():
        sel = np.arange(0, len(b.xyz_mm), stride)
        if sel[-1] != len(b.xyz_mm) - 1:
            sel = np.append(sel, len(b.xyz_mm) - 1)
        rings = tube_rings(b.xyz_mm[sel], b.radius_mm[sel], sides=14)
        quads = _tube_quads(rings)
        seg_ffr = 0.5 * (b.ffr[sel][:-1] + b.ffr[sel][1:])
        polys.append(quads)
        colors.append(np.repeat(seg_ffr, rings.shape[1]))
        pts.append(b.xyz_mm)
    quads = np.concatenate(polys)
    facec = cmap(norm(np.concatenate(colors)))
    pts = np.concatenate(pts)
    centre = 0.5 * (pts.min(0) + pts.max(0))
    half = 0.5 * float((pts.max(0) - pts.min(0)).max()) + 5.0
    labels = _label_points(result)

    fig = plt.figure(figsize=(12, 6.2), layout="constrained")
    # Elevation/azimuth in the LAS frame: z superior; looking from anterior means looking along -y.
    views = (("Anterior (AP)", 0, 90), ("Left lateral", 0, 0)) if result.artery_type != "rca" else \
        (("Anterior (AP)", 0, 90), ("Right lateral", 0, 180))
    for j, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, j, projection="3d")
        coll = Poly3DCollection(quads, facecolors=facec, edgecolors="none", linewidths=0, alpha=1.0)
        ax.add_collection3d(coll)
        for xyz, text, value in labels:
            colour = INK if value is None else INK_2
            ax.text(*(xyz + np.array([0, 0, 2.5])), text, fontsize=7.5, color=colour,
                    ha="center", va="bottom", zorder=10)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("L (mm)", fontsize=8, color=INK_2)
        ax.set_ylabel("A (mm)", fontsize=8, color=INK_2)
        ax.set_zlabel("S (mm)", fontsize=8, color=INK_2)
        ax.tick_params(labelsize=7, colors=MUTED)
        depth = ax.yaxis if azim in (90, -90) else ax.xaxis  # axis pointing at the viewer
        depth.set_ticklabels([])
        depth.label.set_visible(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02)
    cb.set_label("FFR (Pd / Pa)", fontsize=9, color=INK)
    cb.ax.axhline(FFR_THRESHOLD, color=INK, lw=1.0)
    cb.ax.tick_params(labelsize=8)
    lesions = _lesion_list(result)
    if lesions:
        text = "\n".join(f"L{k}  {b.name}: {les.ds_percent:.0f}% DS, MLD {les.mld_mm:.1f} mm, "
                         f"ΔFFR {les.dffr:.2f}" for k, (b, les) in enumerate(lesions, 1))
        fig.text(0.01, 0.01, text, fontsize=8.5, color=INK, va="bottom", ha="left", family="monospace")
    s = result.summary()
    vessel = ", ".join(f"{k} {v:.2f}" for k, v in s["vessel_ffr"].items())
    fig.suptitle((title + " - " if title else "") + f"FFR along the reconstructed artery (distal: {vessel})",
                 fontsize=11, color=INK)
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_pullback(result, path: Path, *, bands: dict | None = None, max_branches: int = 6,
                  title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(result.branches.values(), key=lambda b: (b.index > 1, -(b.distance_mm[-1] - b.distance_mm[b.start])))
    order = order[:max_branches]
    order.sort(key=lambda b: b.index)
    cols = len(order)
    fig, axes = plt.subplots(2, cols, figsize=(3.4 * cols + 0.6, 5.6), sharex="col", squeeze=False,
                             layout="constrained", gridspec_kw={"height_ratios": [1.5, 1.0]})
    for j, b in enumerate(order):
        ax, axd = axes[0, j], axes[1, j]
        x = b.distance_mm
        sl = slice(b.start, None)
        for les in b.lesions:
            for a in (ax, axd):
                a.axvspan(x[les.i0], x[les.i1], color=LESION, lw=0, zorder=0)
        ax.axhspan(0.75, FFR_THRESHOLD, color="#efeee9", lw=0, zorder=0)
        ax.axhline(FFR_THRESHOLD, color=MUTED, lw=1.0, ls="--", zorder=1)
        if bands and b.index in bands:
            band = bands[b.index]
            ax.fill_between(x[sl], band["p05"][sl], band["p95"][sl], color=BAND, lw=0, zorder=2,
                            label="90% interval")
        ax.plot(x[sl], b.ffr[sl], color=INK, lw=2.0, zorder=3, label="FFR")
        ax.plot(x[-1], b.distal_ffr, "o", ms=6, color=INK, mec="white", mew=1.5, zorder=4)
        ax.annotate(f"{b.distal_ffr:.2f}", (x[-1], b.distal_ffr), textcoords="offset points",
                    xytext=(-4, 7), ha="right", fontsize=9, color=INK)
        for les in b.lesions:
            ax.text(x[les.imin], 0.02, f"ΔFFR {les.dffr:.2f}", ha="center", va="bottom", fontsize=8,
                    color=INK_2, transform=ax.get_xaxis_transform())
        ax.set_ylim(min(0.5, float(b.ffr[sl].min()) - 0.03), 1.01)
        ax.set_title(f"{b.name}" + (f"  (from {result.branches[b.parent].name})" if b.parent >= 0 else ""),
                     fontsize=10, color=INK, loc="left")
        axd.plot(x[sl], 2 * b.ref_radius_mm[sl], color=MUTED, lw=1.2, ls="--", label="reference")
        axd.plot(x[sl], 2 * b.radius_mm[sl], color="#2a78d6", lw=2.0, label="lumen")
        for les in b.lesions:
            axd.annotate(f"{les.ds_percent:.0f}% DS", (x[les.imin], 2 * b.radius_mm[les.imin]),
                         textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=INK_2)
        axd.set_ylim(0, 1.15 * float(2 * max(b.ref_radius_mm[sl].max(), b.radius_mm[sl].max())))
        axd.set_xlabel("distance from ostium (mm)", fontsize=8, color=INK_2)
        for a in (ax, axd):
            _style(a)
        if j == 0:
            ax.set_ylabel("FFR (Pd / Pa)", fontsize=9, color=INK)
            axd.set_ylabel("diameter (mm)", fontsize=9, color=INK)
            handles = ax.get_legend_handles_labels()
            ax.legend(*handles, fontsize=7.5, frameon=False, loc="center left")
            axd.legend(fontsize=7.5, frameon=False, loc="lower left")
    s = result.summary()
    fig.suptitle((title + " - " if title else "") +
                 f"Virtual pullback (hyperemic flow {s['flow_ml_s']['diseased_tree']:.1f} ml/s, "
                 f"Pa {result.config.pa_mmhg:.0f} mmHg, {result.config.bc_mode} BC)",
                 fontsize=11, color=INK)
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def load_projections(path: Path) -> dict:
    """evaluate.py ``projections.npz`` or a prepared example NPZ -> camera dict."""
    data = np.load(path, allow_pickle=False)
    if "projection_center_offset_mm" in data.files:
        centre = np.asarray(data["projection_center_offset_mm"], dtype=np.float64)
        sid = float(data["sid_m"]) if "sid_m" in data.files else 0.9
        spacing = float(data["pixel_spacing_mm"])
    else:
        scale = float(data["input_scale_to_mm"]) if "input_scale_to_mm" in data.files else 1000.0
        centre = np.asarray(data["projection_center_offset"], dtype=np.float64) * scale
        sid = float(data["sid"])
        spacing = float(data["imager_pixel_spacing"])
    return {"images": np.asarray(data["images"], dtype=np.float32), "theta": np.asarray(data["theta_deg"]),
            "phi": np.asarray(data["phi_deg"]), "centre_mm": centre, "sid_m": sid, "spacing_mm": spacing,
            "clinical": [str(v) for v in data["clinical_views"]] if "clinical_views" in data.files else None}


def plot_projections(result, projections: dict, path: Path, *, views=None, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from vessel_code.geometry.projection import _stage2_camera_parameters, project_points_to_image_rc

    cmap, norm = ffr_cmap(), ffr_norm()
    images = projections["images"]
    views = list(range(len(images))) if views is None else list(views)
    cams = _stage2_camera_parameters(projections["theta"], projections["phi"], sid_m=projections["sid_m"])
    dim = images.shape[-1]
    fig, axes = plt.subplots(1, len(views), figsize=(3.0 * len(views), 3.3), squeeze=False, layout="constrained")
    for ax, v in zip(axes[0], views):
        ax.imshow(1.0 - images[v], cmap="gray", vmin=0, vmax=1)  # vessels dark, as in an angiogram
        for b in result.branches.values():
            pts = (b.xyz_mm - projections["centre_mm"]) * 1e-3
            rc = project_points_to_image_rc(pts, view_index=v, sensor=cams[0], source=cams[1], local_x=cams[2],
                                            local_y=cams[3], image_dim=dim, pixel_spacing_mm=projections["spacing_mm"])
            seg = np.stack([rc[:-1, ::-1], rc[1:, ::-1]], axis=1)
            lc = LineCollection(seg, colors=cmap(norm(0.5 * (b.ffr[:-1] + b.ffr[1:]))), linewidths=2.2)
            ax.add_collection(lc)
        name = projections["clinical"][v] if projections.get("clinical") else f"view {v}"
        ax.set_title(name, fontsize=9, color=INK)
        ax.set(xlim=(0, dim - 1), ylim=(dim - 1, 0))
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("FFR", fontsize=9)
    cb.ax.axhline(FFR_THRESHOLD, color=INK, lw=1.0)
    fig.suptitle((title + " - " if title else "") + "FFR-coloured centrelines on the projections", fontsize=10)
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Tables and interactive view


def write_csv(result, path: Path, bands: dict | None = None) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["branch", "name", "sample", "inside_parent", "distance_mm", "x_mm", "y_mm", "z_mm",
                  "radius_mm", "ref_radius_mm", "diameter_stenosis_pct", "flow_ml_s", "pressure_mmhg",
                  "static_pressure_mmhg", "ffr", "ffr_static"]
        if bands:
            header += ["ffr_p05", "ffr_p50", "ffr_p95"]
        w.writerow(header)
        for b in result.branches.values():
            ds = 100 * (1 - b.radius_mm / np.maximum(b.ref_radius_mm, 1e-6))
            for i in range(len(b.ffr)):
                row = [b.index, b.name, i, int(i < b.start), *np.round([b.distance_mm[i], *b.xyz_mm[i],
                       b.radius_mm[i], b.ref_radius_mm[i], ds[i], b.flow_ml_s[i], b.pressure_mmhg[i],
                       b.static_mmhg[i], b.ffr[i], b.ffr_static[i]], 5)]
                if bands:
                    band = bands.get(b.index)
                    row += [round(float(band[q][i]), 5) for q in ("p05", "p50", "p95")] if band else ["", "", ""]
                w.writerow(row)
    return path


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px} p{margin:0 0 12px;color:var(--ink2)}
#plot{width:100%;height:min(72vh,720px);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums;width:100%}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left} th{color:var(--ink2);font-weight:600}
.wrap{overflow-x:auto}
</style></head><body><main>
<h1>__TITLE__</h1><p>__SUBTITLE__</p>
<div id="plot"></div>
<div class="wrap"><table id="tab"><thead><tr><th>Branch</th><th>Parent</th><th>Distal FFR</th><th>Min lumen ⌀ (mm)</th><th>Outlet flow (ml/s)</th><th>Lesions (DS %, ΔFFR)</th></tr></thead><tbody></tbody></table></div>
</main>
<script>
const DATA = __DATA__;
const stops = __STOPS__;
const lo = __LO__, hi = __HI__;
const colorscale = stops.map(([v,c]) => [(v-lo)/(hi-lo), c]);
const dark = matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light';
const ink = dark ? '#f0efec' : '#0b0b0b';
const traces = [];
DATA.branches.forEach((b, k) => {
  traces.push({type:'mesh3d', x:b.vx, y:b.vy, z:b.vz, i:b.fi, j:b.fj, k:b.fk, intensity:b.vffr,
    colorscale, cmin:lo, cmax:hi, showscale:k===0, flatshading:false, lighting:{ambient:0.6,diffuse:0.7,specular:0.15},
    colorbar:{title:{text:'FFR', font:{color:ink}}, tickfont:{color:ink}, len:0.7},
    hoverinfo:'skip', name:b.name});
  traces.push({type:'scatter3d', mode:'lines', x:b.cx, y:b.cy, z:b.cz, line:{width:2, color:'rgba(0,0,0,0)'},
    customdata:b.cd, name:b.name,
    hovertemplate:'<b>%{customdata[0]}</b><br>FFR %{customdata[1]:.3f}<br>distance %{customdata[2]:.1f} mm'+
      '<br>lumen ⌀ %{customdata[3]:.2f} mm (ref %{customdata[4]:.2f})<br>flow %{customdata[5]:.2f} ml/s<extra></extra>'});
});
DATA.labels.forEach(l => traces.push({type:'scatter3d', mode:'text', x:[l[0]], y:[l[1]], z:[l[2]+3], text:[l[3]],
  textfont:{color:ink, size:12}, hoverinfo:'skip', showlegend:false}));
const axis = t => ({title:t, color:ink, gridcolor: dark?'#383835':'#e4e3df', zeroline:false, showbackground:false});
Plotly.newPlot('plot', traces, {paper_bgcolor:'rgba(0,0,0,0)', margin:{l:0,r:0,t:0,b:0}, showlegend:false,
  scene:{aspectmode:'data', xaxis:axis('L (mm)'), yaxis:axis('A (mm)'), zaxis:axis('S (mm)'),
  camera:{eye:{x:0, y:2.0, z:0.3}, up:{x:0,y:0,z:1}}}}, {responsive:true, displaylogo:false});
const tb = document.querySelector('#tab tbody');
DATA.table.forEach(r => { const tr = document.createElement('tr');
  r.forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td); }); tb.appendChild(tr); });
</script></body></html>
"""


def write_html(result, path: Path, *, title: str = "FFR along the centreline", stride: int = 2) -> Path:
    branches = []
    for b in result.branches.values():
        sel = np.arange(0, len(b.xyz_mm), stride)
        if sel[-1] != len(b.xyz_mm) - 1:
            sel = np.append(sel, len(b.xyz_mm) - 1)
        sides = 16
        rings = tube_rings(b.xyz_mm[sel], b.radius_mm[sel], sides=sides)
        n = len(sel)
        verts = rings.reshape(-1, 3)
        fi, fj, fk = [], [], []
        for i in range(n - 1):
            for j in range(sides):
                a, c = i * sides + j, i * sides + (j + 1) % sides
                a2, c2 = a + sides, c + sides
                fi += [a, a]; fj += [c, c2]; fk += [c2, a2]
        vffr = np.repeat(b.ffr[sel], sides)
        cd = [[b.name, round(float(b.ffr[i]), 4), round(float(b.distance_mm[i]), 2), round(float(2 * b.radius_mm[i]), 3),
               round(float(2 * b.ref_radius_mm[i]), 3), round(float(b.flow_ml_s[i]), 3)] for i in sel]
        r = lambda a: np.round(a, 3).tolist()  # noqa: E731
        branches.append({"name": b.name, "vx": r(verts[:, 0]), "vy": r(verts[:, 1]), "vz": r(verts[:, 2]),
                         "fi": fi, "fj": fj, "fk": fk, "vffr": r(vffr), "cx": r(b.xyz_mm[sel, 0]),
                         "cy": r(b.xyz_mm[sel, 1]), "cz": r(b.xyz_mm[sel, 2]), "cd": cd})
    labels = []
    for b in result.branches.values():
        labels.append([*np.round(b.xyz_mm[-1], 2).tolist(), f"{b.distal_ffr:.2f}"])
        for les in b.lesions:
            labels.append([*np.round(b.xyz_mm[les.imin], 2).tolist(), f"ΔFFR {les.dffr:.2f}"])
    table = [[b.name, result.branches[b.parent].name if b.parent >= 0 else "-", f"{b.distal_ffr:.2f}",
              f"{2 * b.radius_mm[b.start:].min():.2f}", f"{b.flow_ml_s[-1]:.2f}",
              "; ".join(f"{l.ds_percent:.0f}%, {l.dffr:.2f}" for l in b.lesions) or "-"]
             for b in result.branches.values()]
    s = result.summary()
    subtitle = (f"{result.artery_type.upper()} · hyperemic flow {s['flow_ml_s']['diseased_tree']:.2f} ml/s "
                f"(healthy {s['flow_ml_s']['healthy_tree']:.2f}) · Pa {result.config.pa_mmhg:.0f} mmHg · "
                f"{result.config.bc_mode} BC · red ≤ 0.80 · hover the vessel for values")
    html = (_HTML.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)
            .replace("__DATA__", json.dumps({"branches": branches, "labels": labels, "table": table}))
            .replace("__STOPS__", json.dumps(_STOPS)).replace("__LO__", str(FFR_RANGE[0]))
            .replace("__HI__", str(FFR_RANGE[1])))
    path = Path(path)
    path.write_text(html)
    return path
