"""FFR estimation along the centreline of a reconstructed coronary artery (vessel code).

Pipeline (see docs/ffr.md):

    vessel code [M,N,4] mm  ->  VesselTree (topology, effective side-branch starts)
                            ->  reference (healthy) radius + lesion detection
                            ->  boundary conditions (hyperemic flow, Murray split,
                                microvascular resistances on the healthy tree)
                            ->  steady 1D network model (default) or 3D CFD export
                            ->  FFR(s) = P(s) / Pa along every branch + figures

Only numpy / scipy / matplotlib / scikit-image are required (no torch), so the
step runs on CPU nodes after reconstruction.
"""
from vessel_code.ffr.io import load_vessel
from vessel_code.ffr.tree import VesselTree, build_tree
from vessel_code.ffr.solver1d import FFRConfig, FFRResult, solve_ffr_1d

__all__ = [
    "load_vessel",
    "VesselTree",
    "build_tree",
    "FFRConfig",
    "FFRResult",
    "solve_ffr_1d",
]
