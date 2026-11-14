"""FFT round trips, Fo-Mixup properties, segment symmetry, K-means vs scikit-learn (CLAUDE.md §11)."""
import numpy as np
import torch

from src import fourier, fomixup


def test_fft_roundtrip_both_paths():
    x = torch.rand(4, 3, 32, 32)
    A, P = fourier.fft2_shift(x)
    assert (fourier.ifft2_shift(A, P) - x).abs().max() < 1e-5
    assert (fourier.rfft_roundtrip(x) - x).abs().max() < 1e-5
    A2, P2 = fourier.fft2_dat(x)
    assert (fourier.ifft2_dat(A2, P2) - x).abs().max() < 1e-5


def test_segment_partition_shape_and_dc():
    seg = fourier.segment_index()
    assert seg.shape == (32, 32)
    assert seg[16, 16] == 0
    assert seg.max() < fourier.N_SEGMENTS and seg.min() >= 0
    rings = seg // 12
    assert rings.max() == 11


def test_segment_masks_conjugate_symmetric():
    seg = fourier.segment_index()
    for s in torch.unique(seg):
        m = (seg == s).float()
        assert torch.equal(m, fourier.conj_mirror_shifted(m)), f"segment {int(s)} is not conjugate-symmetric"
    # band masks built from any clustering are therefore symmetric too
    sc = torch.randint(0, 8, (3, fourier.N_SEGMENTS), dtype=torch.uint8)
    B = fomixup.band_mask(sc, torch.tensor([0, 3, 7]))
    assert torch.equal(B, fourier.conj_mirror_shifted(B))


def test_fomixup_identity_real_and_in_range():
    x = torch.rand(6, 3, 32, 32)
    xj = torch.rand(6, 3, 32, 32)
    sc = torch.randint(0, 8, (6, fourier.N_SEGMENTS), dtype=torch.uint8)
    B = fomixup.band_mask(sc, torch.randint(0, 8, (6,)))
    # identity when xi = 0 and p = 1
    xt, imag = fomixup.fomixup(x, xj, B, xi=0.0, p=torch.ones(6, 3, 32, 32), return_imag=True)
    assert imag < 1e-4
    assert (xt - x).abs().max() < 1e-5
    # generic mix: real-valued, in [0, 1]
    xt, imag = fomixup.fomixup(x, xj, B, return_imag=True)
    assert imag < 1e-4, "spectrum lost conjugate symmetry"
    assert xt.min() >= 0 and xt.max() <= 1
    assert torch.isfinite(xt).all()


def test_fomixup_eq1_keeps_A_inside_B():
    """Inside B the amplitude of x_i is kept; outside B it is mixed (CLAUDE.md §6.4, do not swap)."""
    x = torch.rand(2, 3, 32, 32)
    xj = torch.rand(2, 3, 32, 32)
    B = torch.ones(2, 1, 32, 32)             # everything inside B -> no mixing at all
    xt = fomixup.fomixup(x, xj, B, xi=0.9, p=torch.ones_like(x))
    assert (xt - x).abs().max() < 1e-5
    B0 = torch.zeros(2, 1, 32, 32)           # nothing inside B, xi = 1 -> amplitude of x_j with phase of x_i
    xt = fomixup.fomixup(x, xj, B0, xi=1.0, p=torch.ones_like(x))
    A_t, _ = fourier.fft2_shift(xt)
    A_j, _ = fourier.fft2_shift(xj)
    # clipping to [0,1] perturbs the spectrum slightly, so compare loosely
    assert torch.corrcoef(torch.stack([A_t.flatten(), A_j.flatten()]))[0, 1] > 0.9


def test_kmeans1d_matches_sklearn():
    from sklearn.cluster import KMeans
    x = torch.rand(20, 3, 32, 32)
    A, _ = fourier.fft2_shift(x)
    mu = fomixup.segment_means(A)
    _, valid = fomixup.seg_index(A.device)
    vals = mu[:, valid]
    init = fomixup.quantile_init(vals, 8)
    labels, centres = fomixup.kmeans1d(vals, 8, iters=20, init=init)
    for b in range(20):
        v = vals[b].numpy().reshape(-1, 1).astype(np.float64)
        km = KMeans(n_clusters=8, init=init[b].numpy().reshape(-1, 1).astype(np.float64), n_init=1,
                    max_iter=20, tol=0.0, algorithm="lloyd").fit(v)
        ours = labels[b].numpy()
        # compare up to label permutation: each of our clusters must map to exactly one sklearn cluster
        mapping = {}
        ok = True
        for o, s in zip(ours, km.labels_):
            if mapping.setdefault(o, s) != s:
                ok = False
        assert ok and len(set(mapping.values())) == len(mapping), f"image {b}: assignments differ"


def test_cluster_segments_valid_labels():
    x = torch.rand(5, 3, 32, 32)
    A, _ = fourier.fft2_shift(x)
    sc = fomixup.cluster_segments(A)
    _, valid = fomixup.seg_index(A.device)
    assert (sc[:, valid] < 8).all()
    assert (sc[:, ~valid] == fomixup.INVALID).all()
