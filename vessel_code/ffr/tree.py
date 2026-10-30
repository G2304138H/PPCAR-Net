"""Branch topology of a vessel code: parents, attachment points and effective side-branch starts.

The model stores side branches as separate curves whose first sample lies on (inside)
the parent lumen. For flow modelling a side branch begins where its centreline leaves
the parent lumen; the samples before that are "inside parent" and inherit the
parent's pressure for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

LCA_NAMES = {0: "LM-LAD", 1: "LCX"}
RCA_NAMES = {0: "RCA"}


@dataclass
class Branch:
    index: int                 # index in the vessel code (0 = main)
    name: str
    xyz: np.ndarray            # [N,3] mm
    radius: np.ndarray         # [N] mm
    s: np.ndarray              # [N] arc length from the branch's first sample, mm
    parent: int = -1           # parent branch index (-1 for the main branch)
    attach: int = 0            # sample index on the parent where this branch leaves it
    start: int = 0             # first sample outside the parent lumen
    children: list = field(default_factory=list)  # [(attach_index_on_this_branch, child_index)]

    @property
    def length(self) -> float:
        return float(self.s[-1])


@dataclass
class VesselTree:
    branches: dict             # index -> Branch, topologically ordered (parents first)
    artery_type: str

    @property
    def order(self) -> list:
        return list(self.branches.keys())

    def distance_from_ostium(self, index: int) -> np.ndarray:
        """Arc length along the flow path from the ostium to every sample of a branch (mm)."""
        b = self.branches[index]
        if b.parent < 0:
            return b.s - b.s[0]
        offset = float(self.distance_from_ostium(b.parent)[b.attach])
        return offset + (b.s - b.s[0])

    def outlets(self) -> list:
        return list(self.branches.keys())  # every branch ends in a terminal outlet


def _arc_length(xyz: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _branch_name(index: int, artery: str) -> str:
    names = LCA_NAMES if artery == "lca" else RCA_NAMES if artery == "rca" else {0: "main"}
    return names.get(index, f"side {index}")


def build_tree(vessel_mm: np.ndarray, exists: np.ndarray, artery_type: str = "unknown",
               *, max_start_fraction: float = 0.35) -> VesselTree:
    """Infer parent/attachment for every active branch.

    The parent of branch j is the earlier active branch whose centreline passes closest
    to j's first sample (vessel-code order is canonical: main first, then proximal-to-
    distal side branches). The effective start is the first sample whose distance to the
    parent centreline exceeds the local parent radius.
    """
    vessel_mm = np.asarray(vessel_mm, dtype=np.float64)
    active = [int(i) for i in np.flatnonzero(exists)]
    if not active or active[0] != 0:
        raise ValueError("branch 0 (main) must be active")
    branches: dict = {}
    for i in active:
        xyz = vessel_mm[i, :, :3]
        r = vessel_mm[i, :, 3]
        branches[i] = Branch(i, _branch_name(i, artery_type), xyz, r, _arc_length(xyz))
    for j in active[1:]:
        b = branches[j]
        best = (np.inf, -1, 0)
        for i in active:
            if i >= j:
                break
            d = np.linalg.norm(branches[i].xyz - b.xyz[0], axis=1)
            k = int(np.argmin(d))
            if d[k] < best[0]:
                best = (float(d[k]), i, k)
        _, parent, attach = best
        p = branches[parent]
        # Effective start: leave the parent lumen.
        kd = cKDTree(p.xyz)
        dist, nearest = kd.query(b.xyz)
        inside = dist < p.radius[nearest]
        limit = max(1, int(max_start_fraction * len(b.xyz)))
        start = 0
        while start < limit and inside[start]:
            start += 1
        b.parent, b.attach, b.start = parent, attach, start
        p.children.append((attach, j))
    for b in branches.values():
        b.children.sort()
    return VesselTree(branches, artery_type)
