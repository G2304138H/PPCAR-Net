# Transferred from methods/multiview/model_realdata.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import torch

def _fpn_2d_sine_position_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a fixed [1,C,H,W] detector-position encoding."""
    height = int(height)
    width = int(width)
    channels = int(channels)
    if height < 1 or width < 1 or channels < 1:
        raise ValueError(
            f"FPN position encoding requires positive H/W/C, got {(height, width, channels)}."
        )

    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=work_dtype)
    x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=work_dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    def encode_coordinate(values: torch.Tensor, out_channels: int) -> torch.Tensor:
        if out_channels <= 0:
            return values.new_zeros((*values.shape, 0))
        num_frequencies = (int(out_channels) + 1) // 2
        frequency_index = torch.arange(num_frequencies, device=device, dtype=work_dtype)
        denominator = torch.pow(
            values.new_tensor(10000.0),
            frequency_index / float(max(num_frequencies - 1, 1)),
        )
        phase = values.unsqueeze(-1) * (2.0 * torch.pi) / denominator
        encoded = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)
        return encoded[..., : int(out_channels)]

    y_channels = channels // 2
    x_channels = channels - y_channels
    position = torch.cat(
        [encode_coordinate(yy, y_channels), encode_coordinate(xx, x_channels)],
        dim=-1,
    )
    return position.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
