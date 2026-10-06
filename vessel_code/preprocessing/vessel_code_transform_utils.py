# Transferred from methods/data/vessel_code_transform_utils.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from typing import Any
import numpy as np

"""Shared utilities for interpretable vessel-code compression experiments."""

VESSEL_KEY_CANDIDATES = (
    "branches_xyzr_resampled",
    "artery",
    "raw_vessel_code_mm",
    "reconstructed_vessel_code_mm",
)

VESSEL_FILE_SUFFIXES = frozenset({".npy", ".npz", ".mpd"})

def resolve_save_prediction_npz_files(config: dict[str, Any]) -> bool:
    """Resolve the shared per-case evaluation NPZ-export switch strictly."""

    value = config.get("save_prediction_npz_files", True)
    if not isinstance(value, bool):
        raise ValueError("save_prediction_npz_files must be boolean")
    return value

def open_uniform_knot_vector(num_coefficients: int, degree: int) -> np.ndarray:
    num_coefficients = int(num_coefficients)
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"Spline degree must be positive, got {degree}")
    if num_coefficients < degree + 1:
        raise ValueError(
            f"num_coefficients must be at least degree + 1, got {num_coefficients} and {degree}"
        )
    num_interior = num_coefficients - degree - 1
    interior = (
        np.linspace(0.0, 1.0, num_interior + 2, dtype=np.float64)[1:-1]
        if num_interior > 0
        else np.zeros((0,), dtype=np.float64)
    )
    return np.concatenate(
        [
            np.zeros((degree + 1,), dtype=np.float64),
            interior,
            np.ones((degree + 1,), dtype=np.float64),
        ]
    )

def bspline_basis_matrix(
    t: np.ndarray,
    num_coefficients: int,
    degree: int = 3,
    knot_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.clip(np.asarray(t, dtype=np.float64).reshape(-1), 0.0, 1.0)
    knots = (
        open_uniform_knot_vector(num_coefficients, degree)
        if knot_vector is None
        else np.asarray(knot_vector, dtype=np.float64).reshape(-1)
    )
    expected = int(num_coefficients) + int(degree) + 1
    if knots.size != expected:
        raise ValueError(f"Expected {expected} knots, got {knots.size}")

    basis = np.zeros((samples.size, knots.size - 1), dtype=np.float64)
    for index in range(knots.size - 1):
        basis[:, index] = ((samples >= knots[index]) & (samples < knots[index + 1])).astype(float)
    basis[np.isclose(samples, 1.0), -1] = 1.0

    for current_degree in range(1, int(degree) + 1):
        next_basis = np.zeros((samples.size, basis.shape[1] - 1), dtype=np.float64)
        for index in range(next_basis.shape[1]):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = knots[index + current_degree + 1] - knots[index + 1]
            if left_denominator > 0.0:
                next_basis[:, index] += (
                    (samples - knots[index]) / left_denominator
                ) * basis[:, index]
            if right_denominator > 0.0:
                next_basis[:, index] += (
                    (knots[index + current_degree + 1] - samples) / right_denominator
                ) * basis[:, index + 1]
        basis = next_basis

    basis[np.isclose(samples, 1.0), :] = 0.0
    basis[np.isclose(samples, 1.0), -1] = 1.0
    if basis.shape != (samples.size, int(num_coefficients)):
        raise RuntimeError(f"Unexpected B-spline basis shape {basis.shape}")
    return basis, knots
