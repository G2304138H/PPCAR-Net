# Transferred from methods/losses/regression.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import torch
import torch.nn.functional as F

def elementwise_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    *,
    option_name: str,
) -> torch.Tensor:
    """Return an unreduced elementwise regression loss.

    The aliases make JSON configuration forgiving while keeping ``mse``,
    ``l1``, and ``smooth_l1`` as the documented canonical values.
    """
    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized in {"mse", "squared", "squared_loss", "l2"}:
        return F.mse_loss(prediction, target, reduction="none")
    if normalized in {"l1", "mae", "absolute", "absolute_error"}:
        return F.l1_loss(prediction, target, reduction="none")
    if normalized in {"smooth_l1", "smoothl1", "huber"}:
        return F.smooth_l1_loss(prediction, target, reduction="none")
    raise ValueError(
        f"{option_name} must be one of 'mse', 'l1', or 'smooth_l1', "
        f"got {loss_type!r}."
    )

__all__ = ["elementwise_regression_loss", "pointwise_xyz_regression_loss"]
