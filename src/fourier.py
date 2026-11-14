"""FFT helpers (all float32, autocast disabled) and the Fo-Mixup segment partition (CLAUDE.md §6.1-6.2).

Two spectrum conventions are used in this project:
  * Fo-Mixup (§6.1):  F = fftshift(fft2(x)); A = |F|, P = angle(F); DC at (16, 16).
  * DAT (official code `get_fft` / `inverse_fft`): A = fftshift(|fft2(x)|) but P = angle(fft2(x)) (unshifted);
    inverse: ifftshift(A) * exp(iP) -> ifft2 -> real -> clip[0,1].
"""
import math

import torch

from .common import fp32_ctx

N = 32
CENTER = N // 2
RING_WIDTH = 2.0          # [adaptation] rings 0-11
SECTOR_DEG = 15.0         # [adaptation] sectors 0-11 (angle folded by pi)
N_RINGS, N_SECTORS = 12, 12
N_SEGMENTS = N_RINGS * N_SECTORS  # 144


def fft2_shift(x: torch.Tensor):
    """§6.1: per image & channel, F = fftshift(fft2(x_c)); returns amplitude A and phase P (float32)."""
    with fp32_ctx(x.device):
        Fx = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
        return Fx.abs(), Fx.angle()


def ifft2_shift_complex(A: torch.Tensor, P: torch.Tensor):
    """Inverse of fft2_shift WITHOUT taking the real part (so tests can check the imaginary residue)."""
    with fp32_ctx(A.device):
        Fx = torch.polar(A.float(), P.float())
        return torch.fft.ifft2(torch.fft.ifftshift(Fx, dim=(-2, -1)))


def ifft2_shift(A, P):
    return ifft2_shift_complex(A, P).real


def fft2_dat(x: torch.Tensor):
    """DAT official code `get_fft`: shifted amplitude, UNshifted phase (full spectrum, not rfft2; see NOTES.md)."""
    with fp32_ctx(x.device):
        Fx = torch.fft.fftn(x.float(), dim=(-2, -1))
        A = torch.fft.fftshift(Fx.abs(), dim=(-2, -1))
        return A, Fx.angle()


def ifft2_dat(A: torch.Tensor, P: torch.Tensor):
    """DAT official code `inverse_fft`: ifftshift(A) * exp(iP) -> ifftn -> real -> clip to [0,1]."""
    with fp32_ctx(A.device):
        A = torch.fft.ifftshift(A.float(), dim=(-2, -1))
        img = torch.fft.ifftn(A * torch.exp(1j * P.float()), dim=(-2, -1))
        return torch.clamp(img.real.float(), 0.0, 1.0)


def rfft_roundtrip(x):
    """rfft2 -> irfft2 round trip (used by the FFT test)."""
    with fp32_ctx(x.device):
        return torch.fft.irfft2(torch.fft.rfft2(x.float()), s=x.shape[-2:])


# ----------------------------------------------------------------------------------------------
# Conjugate symmetry helpers (spectra are in fftshift-ed layout)
# ----------------------------------------------------------------------------------------------
def conj_mirror_shifted(m: torch.Tensor) -> torch.Tensor:
    """For a shifted spectrum tensor m[..., u, v], return m evaluated at the conjugate-mirror position
    (-k_u mod N, -k_v mod N) of every unshifted frequency k. A real image has A(k) = A(-k)."""
    u = torch.fft.ifftshift(m, dim=(-2, -1))
    u = torch.roll(torch.flip(u, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))
    return torch.fft.fftshift(u, dim=(-2, -1))


def canonical_half_mask(n: int = N, device=None) -> torch.Tensor:
    """Boolean n x n (shifted layout): True on one representative of each conjugate pair (and self-mirrors).
    Used to sample conjugate-symmetric random fields (Fo-Mixup Eq. 2 noise, see NOTES.md)."""
    k = torch.arange(n, device=device)
    ku = k.view(n, 1).expand(n, n)
    kv = k.view(1, n).expand(n, n)
    mu, mv = (-ku) % n, (-kv) % n
    canon = (ku < mu) | ((ku == mu) & (kv <= mv))
    return torch.fft.fftshift(canon, dim=(-2, -1))


def symmetrize_field(p: torch.Tensor) -> torch.Tensor:
    """Make a random field conjugate-symmetric by copying the canonical half onto its mirror."""
    canon = canonical_half_mask(p.shape[-1], p.device)
    return torch.where(canon, p, conj_mirror_shifted(p))


# ----------------------------------------------------------------------------------------------
# Segment partition (§6.2)
# ----------------------------------------------------------------------------------------------
def segment_index(n: int = N) -> torch.Tensor:
    """LongTensor n x n of segment IDs = ring * 12 + sector, in the fftshift-ed layout (DC at (16,16)).

    ring   = floor(r / 2),  r = sqrt(du^2 + dv^2),  du = u - 16, dv = v - 16
    sector = floor((atan2(dv, du) mod pi) / 15deg)
    On the Nyquist row/column (du = -16 or dv = -16, which are their own conjugate mirrors) the angle is taken
    from (|du|, |dv|) so that the partition stays conjugate-symmetric [adaptation, see NOTES.md].
    DC position -> ring 0, sector 0.
    """
    c = n // 2
    idx = torch.arange(n, dtype=torch.float64)
    du = (idx - c).view(n, 1).expand(n, n)
    dv = (idx - c).view(1, n).expand(n, n)
    r = torch.sqrt(du ** 2 + dv ** 2)
    ring = torch.clamp(torch.floor(r / RING_WIDTH), max=N_RINGS - 1).long()
    nyq = (du == -c) | (dv == -c)
    du_a = torch.where(nyq, du.abs(), du)
    dv_a = torch.where(nyq, dv.abs(), dv)
    theta = torch.remainder(torch.atan2(dv_a, du_a), math.pi)
    sector = torch.clamp(torch.floor(theta / math.radians(SECTOR_DEG)), max=N_SECTORS - 1).long()
    seg = ring * N_SECTORS + sector
    seg[c, c] = 0
    return seg


def segment_counts(seg: torch.Tensor, n_segments: int = N_SEGMENTS) -> torch.Tensor:
    return torch.bincount(seg.flatten(), minlength=n_segments)
