"""Fo-Mixup for 32x32 signs (CLAUDE.md §6; FreqDebias, Kashiani et al., CVPR 2025).

All operations in float32 (autocast disabled inside the FFT helpers).
"""
import torch
import torch.nn.functional as F

from .color import chroma_weights, rgb_to_ycbcr, ycbcr_to_rgb
from .common import fp32_ctx
from .fourier import (N, N_SEGMENTS, fft2_shift, ifft2_shift_complex, segment_index, segment_counts,
                      symmetrize_field)

K_CLUSTERS = 8   # [paper: FreqDebias, k = 8]
T_KEEP = 3       # [paper: FreqDebias, t = 3]
INVALID = 255    # cluster label for segments with no frequency position

_SEG_CACHE = {}


def seg_index(device):
    key = str(device)
    if key not in _SEG_CACHE:
        seg = segment_index(N).to(device)
        counts = segment_counts(seg)
        _SEG_CACHE[key] = (seg, (counts > 0).to(device))
    return _SEG_CACHE[key]


# ----------------------------------------------------------------------------------------------
# §6.3 step 1: per-segment mean of log(1 + A), averaged over channels
# ----------------------------------------------------------------------------------------------
def segment_means(A: torch.Tensor) -> torch.Tensor:
    """A: B x 3 x 32 x 32 amplitude -> mu_seg: B x 144 (NaN for empty segments)."""
    with fp32_ctx(A.device):
        seg, valid = seg_index(A.device)
        la = torch.log1p(A.float()).mean(1).flatten(1)                 # B x 1024, channel-averaged
        idx = seg.flatten().unsqueeze(0).expand(la.shape[0], -1)
        sums = torch.zeros(la.shape[0], N_SEGMENTS, device=A.device).scatter_add_(1, idx, la)
        counts = segment_counts(seg).to(A.device).float().clamp(min=1)
        mu = sums / counts
        mu[:, ~valid] = float("nan")
        return mu


# ----------------------------------------------------------------------------------------------
# §6.3 step 2: batched, vectorised 1-D K-means (20 Lloyd iterations, centres at the 8 quantiles)
# ----------------------------------------------------------------------------------------------
def quantile_init(vals: torch.Tensor, k: int = K_CLUSTERS) -> torch.Tensor:
    q = (torch.arange(k, dtype=torch.float32, device=vals.device) + 0.5) / k
    return torch.quantile(vals.float(), q, dim=1).T.contiguous()       # B x k


def kmeans1d(vals: torch.Tensor, k: int = K_CLUSTERS, iters: int = 20, init=None):
    """vals: B x S finite values. Returns (labels B x S long, centres B x k). Lloyd iterations on GPU/CPU."""
    with fp32_ctx(vals.device):
        vals = vals.float()
        centres = quantile_init(vals, k) if init is None else init.float().clone()
        labels = None
        for _ in range(iters):
            d = (vals.unsqueeze(2) - centres.unsqueeze(1)).abs()        # B x S x k
            labels = d.argmin(dim=2)                                     # B x S
            onehot = F.one_hot(labels, k).float()                        # B x S x k
            counts = onehot.sum(1)                                       # B x k
            sums = torch.einsum("bsk,bs->bk", onehot, vals)
            new = torch.where(counts > 0, sums / counts.clamp(min=1), centres)  # empty cluster keeps centre
            centres = new
        d = (vals.unsqueeze(2) - centres.unsqueeze(1)).abs()
        labels = d.argmin(dim=2)
        return labels, centres


def cluster_segments(A: torch.Tensor):
    """Full §6.3 step 1-2 for a batch of amplitudes. Returns seg_cluster: B x 144 (uint8, INVALID for empty)."""
    mu = segment_means(A)
    _, valid = seg_index(A.device)
    labels, _ = kmeans1d(mu[:, valid])
    out = torch.full((A.shape[0], N_SEGMENTS), INVALID, dtype=torch.long, device=A.device)
    out[:, valid] = labels
    return out.to(torch.uint8)


# ----------------------------------------------------------------------------------------------
# §6.3 step 3: binary band masks
# ----------------------------------------------------------------------------------------------
def band_mask(seg_cluster: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """seg_cluster: B x 144 (cluster id per segment), z: B cluster ids -> mask B x 1 x 32 x 32 (float 0/1),
    identical for all channels."""
    seg, _ = seg_index(seg_cluster.device)
    cmap = seg_cluster.long().gather(1, seg.flatten().unsqueeze(0).expand(seg_cluster.shape[0], -1))
    mask = (cmap == z.view(-1, 1)).float().view(-1, 1, N, N)
    return mask


def band_filtered_image(A, P, mask):
    """x_z = clip( real( ifft2( ifftshift( A * B_z * exp(iP) ) ) ), 0, 1 )  (§6.3 step 3)."""
    return torch.clamp(ifft2_shift_complex(A * mask, P).real, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# §6.4 the mix [paper: FreqDebias Eq. 1-2]
# ----------------------------------------------------------------------------------------------
def fomixup(x_i, x_j, mask, xi=None, p_std=0.1, p=None, return_imag=False):
    """Fo-Mixup of x_i with same-class partner x_j, keeping x_i's phase.

        A_hat = A_i * B + ((1 - xi) * A_i + xi * A_j) * (1 - B)          Eq. 1  (A_i kept INSIDE B)
        A_hat = max(0, p * A_hat),  p ~ N(1, p_std^2) element-wise        Eq. 2  [adaptation: paper prints N(1,0)]
        x_t   = clip_[0,1]( real( ifft2( ifftshift( A_hat * exp(i P_i) ) ) ) )

    p is sampled conjugate-symmetrically so the spectrum stays Hermitian and x_t real (NOTES.md).
    xi: None -> U(0,1) per image; p: None -> sampled; pass p=torch.ones for the identity test.
    """
    with fp32_ctx(x_i.device):
        B = x_i.shape[0]
        A_i, P_i = fft2_shift(x_i)
        A_j, _ = fft2_shift(x_j)
        if xi is None:
            xi = torch.rand(B, 1, 1, 1, device=x_i.device)
        elif not torch.is_tensor(xi):
            xi = torch.full((B, 1, 1, 1), float(xi), device=x_i.device)
        mask = mask.float()
        A_hat = A_i * mask + ((1.0 - xi) * A_i + xi * A_j) * (1.0 - mask)        # Eq. 1
        if p is None:
            p = symmetrize_field(1.0 + p_std * torch.randn_like(A_hat))
        A_hat = F.relu(p * A_hat)                                                  # Eq. 2
        z = ifft2_shift_complex(A_hat, P_i)
        x_t = torch.clamp(z.real, 0.0, 1.0)
        if return_imag:
            return x_t, z.imag.abs().max()
        return x_t


def fomixup_color(x_i, x_j, mask, xi=None, p_std=0.1, p=None, chroma_protect=1.0, return_imag=False):
    """Colour-aware Fo-Mixup (CLAUDE.md §14, E5-CA): Eq. 1-2 applied in YCbCr instead of RGB.

        Y_i, Cb_i, Cr_i = YCbCr(x_i);  same for x_j                          (src/color.py, BT.601 full range)
        per YCbCr channel c with strength w_c = (1, 1 - alpha, 1 - alpha):
            A_hat_c = A_i,c * B + ((1 - w_c xi) A_i,c + w_c xi A_j,c) * (1 - B)    Eq. 1, xi scaled by w_c
            A_hat_c = max(0, p_c * A_hat_c),  p_c = 1 + w_c (p - 1)                Eq. 2, noise scaled by w_c
        channels with w_c = 0 are passed through unchanged (exactly)
        x_t = clip_[0,1]( RGB( real( ifft2( ifftshift( A_hat * exp(i P_i) ) ) ) ) )

    alpha = chroma_protect: 1 -> only the luminance spectrum is mixed and the sign's colour is kept;
    0 -> all three YCbCr channels are mixed. The band mask B, xi and p are the same objects as in fomixup(), so
    E5-CA differs from E5 only in the colour space the mix is applied in. The masks were selected from the RGB
    spectrum (precompute_fomixup.py) and are reused unchanged (NOTES.md N23).
    """
    with fp32_ctx(x_i.device):
        Bn = x_i.shape[0]
        w = chroma_weights(chroma_protect, x_i.device)                                 # 1 x 3 x 1 x 1
        if xi is None:
            xi = torch.rand(Bn, 1, 1, 1, device=x_i.device)
        elif not torch.is_tensor(xi):
            xi = torch.full((Bn, 1, 1, 1), float(xi), device=x_i.device)
        y_i, y_j = rgb_to_ycbcr(x_i), rgb_to_ycbcr(x_j)
        A_i, P_i = fft2_shift(y_i)
        A_j, _ = fft2_shift(y_j)
        if p is None:
            p = symmetrize_field(1.0 + p_std * torch.randn_like(A_i))
        xi_c = xi * w
        p_c = 1.0 + w * (p - 1.0)
        mask = mask.float()
        A_hat = A_i * mask + ((1.0 - xi_c) * A_i + xi_c * A_j) * (1.0 - mask)        # Eq. 1 per YCbCr channel
        A_hat = F.relu(p_c * A_hat)                                                   # Eq. 2
        z = ifft2_shift_complex(A_hat, P_i)
        untouched = (w == 0).view(1, 3, 1, 1)
        y_t = torch.where(untouched, y_i, z.real)                                     # exact pass-through
        x_t = torch.clamp(ycbcr_to_rgb(y_t), 0.0, 1.0)
        if return_imag:
            return x_t, z.imag.abs().max()
        return x_t


def choose_masks(seg_cluster, top_clusters, mode, n_clusters=K_CLUSTERS):
    """Pick B for each image according to the experiment (§5.8):
        'ohem'    one of the image's t=3 stored OHEM clusters, uniformly at random (E4/E5)      [paper]
        'uniform' a cluster chosen uniformly from all 8 (A2)
        'all'     B = 0, i.e. mix the whole spectrum (A1)
    """
    Bn = seg_cluster.shape[0]
    dev = seg_cluster.device
    if mode == "all":
        return torch.zeros(Bn, 1, N, N, device=dev)
    if mode == "ohem":
        pick = torch.randint(0, top_clusters.shape[1], (Bn,), device=dev)
        z = top_clusters.gather(1, pick.view(-1, 1)).squeeze(1)
    elif mode == "uniform":
        z = torch.randint(0, n_clusters, (Bn,), device=dev)
    else:
        raise ValueError(mode)
    return band_mask(seg_cluster, z)
