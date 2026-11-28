"""Differentiable RGB <-> YCbCr transform for the colour-aware extension (CLAUDE.md §14), float32 throughout.

ITU-R BT.601 full-range ("JPEG / JFIF") YCbCr for images in [0, 1]:

    Y  =  0.299    R + 0.587    G + 0.114    B
    Cb = -0.168736 R - 0.331264 G + 0.5      B + 0.5
    Cr =  0.5      R - 0.418688 G - 0.081312 B + 0.5

Y carries brightness and texture; Cb and Cr carry colour (blue-difference and red-difference), centred at 0.5.
Both directions are affine maps, so they are exactly invertible and differentiable (the DAT generator is trained
through them). The inverse matrix is computed from the forward one, so the round trip is exact up to float32
rounding (tests/test_colour_aware.py).
"""
import torch

from .common import fp32_ctx

_M = torch.tensor([[0.299, 0.587, 0.114],
                   [-0.168736, -0.331264, 0.5],
                   [0.5, -0.418688, -0.081312]], dtype=torch.float64)
_OFFSET = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
_M_INV = torch.linalg.inv(_M)


def _apply(m, x):
    """m: 3x3, x: B x 3 x H x W -> B x 3 x H x W (per-pixel matrix product)."""
    return torch.einsum("ij,bjhw->bihw", m.to(device=x.device, dtype=torch.float32), x)


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """B x 3 x H x W RGB in [0,1] -> YCbCr (Y in [0,1], Cb/Cr in [0,1] centred at 0.5). float32."""
    with fp32_ctx(x.device):
        x = x.float()
        return _apply(_M, x) + _OFFSET.to(device=x.device, dtype=torch.float32).view(1, 3, 1, 1)


def ycbcr_to_rgb(y: torch.Tensor) -> torch.Tensor:
    """Exact inverse of rgb_to_ycbcr (no clipping; callers clip to [0,1] when they need an image)."""
    with fp32_ctx(y.device):
        y = y.float() - _OFFSET.to(device=y.device, dtype=torch.float32).view(1, 3, 1, 1)
        return _apply(_M_INV, y)


def chroma_weights(chroma_protect: float, device=None) -> torch.Tensor:
    """Per-channel mixing strength in YCbCr order, shape 1 x 3 x 1 x 1: Y is always mixed fully (weight 1);
    Cb and Cr are mixed with weight (1 - alpha). alpha = chroma_protect in [0, 1]:
        alpha = 1  colour untouched (only the luminance spectrum is perturbed)  -> E5-CA / E6-CA
        alpha = 0  all three YCbCr channels are perturbed like E5/E6 perturb the three RGB channels
    """
    a = float(chroma_protect)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"chroma_protect must be in [0, 1], got {a}")
    return torch.tensor([1.0, 1.0 - a, 1.0 - a], device=device, dtype=torch.float32).view(1, 3, 1, 1)
