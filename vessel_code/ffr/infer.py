"""FFR inference for a reconstructed 3D artery (vessel code).

Examples
--------
# a refined reconstruction written by vessel_code.evaluate
python -m vessel_code.ffr.infer --input outputs/lca_example/lca_0457/refined.npz \
    --projections outputs/lca_example/lca_0457/projections.npz --output outputs/ffr/lca_0457

# every refined.npz under a run directory, with Monte Carlo bands
python -m vessel_code.ffr.infer --input outputs/lca_example --output outputs/ffr/lca_example --mc 200

# bundled ground-truth example with a virtual 60% lesion in the main branch, plus a 3D CFD case
python -m vessel_code.ffr.infer --input examples/lca/lca_0457.npz --output outputs/ffr/demo \
    --virtual-lesion branch=0,t=0.3,sigma=0.03,severity=0.6 --projections examples/lca/lca_0457.npz \
    --export-3d openfoam
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vessel_code.ffr.io import load_vessel
from vessel_code.ffr.reference import apply_virtual_lesion, parse_virtual_lesion
from vessel_code.ffr.solver1d import FFRConfig, run_ffr
from vessel_code.ffr.uncertainty import UncertaintyConfig, monte_carlo
from vessel_code.ffr import visualize as viz


def _find_inputs(path: Path, pattern: str) -> list:
    if path.is_file():
        return [path]
    files = sorted(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {path}")
    return files


def _case_name(path: Path, root: Path) -> str:
    if path == root or root.is_file():
        stem = path.stem
        return path.parent.name if stem in ("refined", "coarse", "prediction") else stem
    rel = path.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts if p not in ("refined", "coarse", "prediction")]
    return "_".join(parts) or path.stem


def _save_npz(result, path: Path, bands: dict | None) -> None:
    arrays = {}
    for i, b in result.branches.items():
        for key in ("xyz_mm", "radius_mm", "ref_radius_mm", "distance_mm", "pressure_mmhg", "static_mmhg",
                    "flow_ml_s", "ffr", "ffr_static"):
            arrays[f"b{i}_{key}"] = getattr(b, key)
        arrays[f"b{i}_meta"] = np.array([b.parent, b.attach, b.start])
        if bands and i in bands:
            for q, v in bands[i].items():
                arrays[f"b{i}_ffr_{q}"] = v
    arrays["branch_indices"] = np.array(sorted(result.branches))
    np.savez_compressed(path, **arrays)


def run_case(path: Path, out: Path, cfg: FFRConfig, args, *, title: str) -> dict:
    vin = load_vessel(path, key=args.key, artery_type=args.artery)
    vessel = vin.vessel_mm
    lesions_added = []
    for spec in args.virtual_lesion or []:
        kw = parse_virtual_lesion(spec)
        vessel = apply_virtual_lesion(vessel, **kw)
        lesions_added.append(kw)
    out.mkdir(parents=True, exist_ok=True)
    result = run_ffr(vessel, vin.exists, vin.artery_type, cfg)
    bands = None
    summary = result.summary()
    summary.update({"input": str(path), "input_key": vin.key, "virtual_lesions": lesions_added})
    if args.mc and args.mc > 0:
        ucfg = UncertaintyConfig(draws=args.mc, seed=args.seed)
        mc = monte_carlo(vessel, vin.exists, vin.artery_type, cfg, ucfg, result)
        bands = mc["bands"]
        summary["uncertainty"] = {"config": asdict(ucfg), "failures": mc["failures"],
                                  "branches": {result.branches[i].name: s for i, s in mc["stats"].items()}}
    (out / "ffr_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    viz.write_csv(result, out / "ffr_centreline.csv", bands)
    _save_npz(result, out / "ffr_result.npz", bands)
    if args.figures != "none":
        viz.plot_pullback(result, out / "ffr_pullback.png", bands=bands, title=title)
        viz.plot_3d(result, out / "ffr_3d.png", title=title)
        viz.write_html(result, out / "ffr_interactive.html", title=f"FFR - {title}")
        proj = args.projections
        if proj is None and (path.parent / "projections.npz").exists():
            proj = path.parent / "projections.npz"
        if proj is not None:
            views = [int(v) for v in args.views.split(",")] if args.views else None
            viz.plot_projections(result, viz.load_projections(Path(proj)), out / "ffr_projections.png",
                                 views=views, title=title)
    if args.export_3d:
        from vessel_code.ffr.mesh3d import build_lumen_surface
        from vessel_code.ffr.openfoam import write_openfoam_case
        surface = build_lumen_surface(result, voxel_mm=args.voxel_mm)
        case_dir = write_openfoam_case(result, surface, out / "cfd3d")
        summary["cfd3d"] = {"case": str(case_dir), "surface_checks": surface.checks}
        (out / "ffr_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="vessel-code NPZ, or a directory to search")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pattern", default="refined.npz", help="file pattern when --input is a directory")
    p.add_argument("--key", default=None, help="array key holding the [M,N,4] vessel code")
    p.add_argument("--artery", choices=("lca", "rca"), default=None)
    p.add_argument("--config", type=Path, default=None, help="JSON with FFRConfig fields")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    p.add_argument("--mc", type=int, default=0, help="Monte Carlo draws for uncertainty bands (e.g. 200)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--virtual-lesion", action="append", metavar="branch=0,t=0.3,sigma=0.03,severity=0.6")
    p.add_argument("--projections", type=Path, default=None,
                   help="projections.npz from evaluate.py or a prepared example (overlay figure)")
    p.add_argument("--views", default=None, help="comma-separated view indices for the overlay")
    p.add_argument("--figures", choices=("all", "none"), default="all")
    p.add_argument("--export-3d", choices=("openfoam",), default=None, help="also write a 3D CFD case")
    p.add_argument("--voxel-mm", type=float, default=None, help="3D surface voxel size (default r_min/5)")
    return p


def config_from_args(args) -> FFRConfig:
    values = json.loads(args.config.read_text()) if args.config else {}
    values = {k: v for k, v in values.items() if not k.startswith("_")}
    defaults = FFRConfig()
    for item in args.set:
        key, value = item.split("=", 1)
        current = getattr(defaults, key, None)
        if isinstance(current, bool):
            values[key] = value.lower() in ("1", "true", "yes")
        elif isinstance(current, str):
            values[key] = value
        else:
            values[key] = float(value)
    return FFRConfig.from_dict(values)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    files = _find_inputs(args.input, args.pattern)
    rows = []
    for path in files:
        name = _case_name(path, args.input)
        out = args.output if len(files) == 1 else args.output / name
        s = run_case(path, out, cfg, args, title=name)
        rows.append({"case": name, **{f"ffr_{k}": round(v, 4) for k, v in s["vessel_ffr"].items()},
                     "min_ffr": round(s["min_ffr"], 4), "num_lesions": s["num_lesions"],
                     "flow_ml_s": round(s["flow_ml_s"]["diseased_tree"], 3), "converged": s["converged"]})
        print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in s["vessel_ffr"].items())
              + f" | lesions {s['num_lesions']} | {out}", flush=True)
    if len(files) > 1:
        import csv
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
        with (args.output / "ffr_cases.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
