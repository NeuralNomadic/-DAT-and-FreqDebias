"""Colour-aware extension (CLAUDE.md §14): colour map, YCbCr transform, colour-aware Fo-Mixup and DAT,
and the Phase 0 colour-group diagnosis."""
import os

import numpy as np
import pytest
import torch

from src import colour_groups, data
from src.color import chroma_weights, rgb_to_ycbcr, ycbcr_to_rgb
from src.fomixup import band_mask, fomixup, fomixup_color
from src.fourier import N


# ---------------------------------------------------------------- Phase 0: colour map
def test_colour_groups_partition_and_match_class_names():
    G, names = data.COLOUR_GROUPS, data.CLASS_NAMES
    assert sorted(c for v in G.values() for c in v) == list(range(43)), "every class in exactly one group"
    assert all(names[c].startswith("End of") for c in G["white"])
    assert names[G["yellow"][0]] == "Priority road"
    blue_words = ("ahead", "go straight", "keep", "roundabout")
    assert all(any(w in names[c].lower() for w in blue_words) for c in G["blue"])
    # red: every prohibitory / warning sign, including Stop, Yield, No entry and the speed limits
    assert {13, 14, 17, 0, 1, 2, 3, 4, 5, 7, 8} <= set(G["red"])
    assert not any(names[c].startswith("End of") for c in G["red"])


def _fake_out(tmp_path, labels, runs):
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "data_cache"))
    torch.save({"labels": torch.tensor(labels)}, data.cache_path_for(out, "test"))
    for exp, (clean, pgd) in runs.items():
        d = os.path.join(out, "runs", exp, "seed0")
        os.makedirs(d)
        torch.save({"clean": torch.tensor(clean), "pgd20": torch.tensor(pgd)}, os.path.join(d, "eval_flags.pt"))
    return out


def test_colour_group_diagnosis_detects_a_planted_gap(tmp_path):
    rng = np.random.default_rng(0)
    groups = data.COLOUR_GROUPS
    labels = np.concatenate([np.repeat(groups[g][0], n) for g, n in
                             (("red", 3000), ("blue", 1500), ("yellow", 400), ("white", 400))])
    colour = np.isin(labels, groups["red"] + groups["blue"])
    base = rng.random(len(labels)) < 0.8                                   # E3: 80 % everywhere
    worse = base & ~(colour & (rng.random(len(labels)) < 0.25))            # E5: loses 25 % of colour-coded only
    same = base.copy()                                                     # E6: identical to E3
    out = _fake_out(tmp_path, labels, {"E3_base": (base, base), "E5_fd_at": (worse, worse),
                                       "E6_dat": (same, same)})
    group_rows, drop_rows = colour_groups.analyse(out, quiet=True)
    assert [r[0] for r in group_rows] == ["E3_base", "E5_fd_at", "E6_dat"]
    verdicts = {(r[0], r[2]): r[-1] for r in drop_rows}
    assert verdicts[("E5_fd_at", "clean")].startswith("colour-coded signs lose MORE")
    assert verdicts[("E6_dat", "clean")].startswith("no clear gap")
    for name in ("table_colour_groups", "table_colour_drop"):
        assert os.path.exists(os.path.join(out, "results", name + ".md"))


def test_colour_group_diagnosis_skips_smoke_sized_flags(tmp_path):
    labels = np.arange(43).repeat(3)
    out = _fake_out(tmp_path, labels, {"E3_base": (np.ones(16, bool), np.ones(16, bool))})   # wrong length
    assert colour_groups.analyse(out, quiet=True) is None


# ---------------------------------------------------------------- Phase 1: colour transform
def test_ycbcr_round_trip_and_known_values():
    x = torch.rand(4, 3, 32, 32)
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5
    grey = torch.full((1, 3, 2, 2), 0.37)
    ycc = rgb_to_ycbcr(grey)
    assert torch.allclose(ycc[:, 0], torch.tensor(0.37), atol=1e-6)           # Y of a grey pixel = its value
    assert torch.allclose(ycc[:, 1:], torch.tensor(0.5), atol=1e-6)           # no chroma
    red = torch.zeros(1, 3, 1, 1); red[:, 0] = 1
    assert rgb_to_ycbcr(red)[0, 2, 0, 0] > 0.9                                  # Cr high for red


def test_chroma_weights():
    assert chroma_weights(1.0).flatten().tolist() == [1.0, 0.0, 0.0]
    assert chroma_weights(0.0).flatten().tolist() == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        chroma_weights(1.5)


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 0.15 + 0.7 * torch.rand(n, 3, N, N, generator=g)                      # away from 0/1: no clipping
    xj = 0.15 + 0.7 * torch.rand(n, 3, N, N, generator=g)
    seg = torch.randint(0, 8, (n, 144), generator=g).to(torch.uint8)
    return x, xj, band_mask(seg, torch.zeros(n, dtype=torch.long))


# ---------------------------------------------------------------- Phase 1: colour-aware Fo-Mixup
def test_fomixup_color_identity_real_and_range():
    x, xj, B = _batch()
    out, imag = fomixup_color(x, xj, B, xi=0.0, p=torch.ones_like(x), return_imag=True)
    assert (out - x).abs().max() < 1e-4 and imag < 1e-4                        # xi = 0, p = 1 -> x
    out = fomixup_color(x, xj, B, chroma_protect=0.5)                          # random xi and p
    assert out.min() >= 0 and out.max() <= 1


def test_fomixup_color_keeps_chroma_and_changes_luminance():
    x, xj, B = _batch()
    out = fomixup_color(x, xj, B, xi=0.9, p=torch.ones_like(x), chroma_protect=1.0)
    ycc_in, ycc_out = rgb_to_ycbcr(x), rgb_to_ycbcr(out)
    unclipped = ((out > 1e-4) & (out < 1 - 1e-4)).all(1, keepdim=True)
    assert unclipped.float().mean() > 0.9
    dc = (ycc_out[:, 1:] - ycc_in[:, 1:]).abs() * unclipped
    assert dc.max() < 1e-4, "alpha = 1 must leave Cb/Cr unchanged"
    assert (ycc_out[:, 0] - ycc_in[:, 0]).abs().mean() > 1e-3, "the luminance must change"
    rgb = fomixup(x, xj, B, xi=0.9, p=torch.ones_like(x))                      # the original RGB mix
    assert (rgb_to_ycbcr(rgb)[:, 1:] - ycc_in[:, 1:]).abs().mean() > 1e-3, "RGB mixing changes colour"


def test_fomixup_color_alpha_zero_mixes_chroma():
    x, xj, B = _batch()
    out = fomixup_color(x, xj, B, xi=0.9, p=torch.ones_like(x), chroma_protect=0.0)
    assert (rgb_to_ycbcr(out)[:, 1:] - rgb_to_ycbcr(x)[:, 1:]).abs().mean() > 1e-3


# ---------------------------------------------------------------- Phase 1: colour-aware DAT
def _dat(cfg_extra, num_classes=43):
    from src.methods.dat import DAT
    from src.models import ResNet18
    cfg = {"method": "dat", "dual_bn": True, "generator": "learned", "num_classes": num_classes, "eps": 8,
           "alpha": 2, "eaeg_steps": 1, "beta": 15.0, "label_smoothing": 0.5, **cfg_extra}
    model = ResNet18(num_classes=num_classes, dual_bn=True)
    return DAT(cfg, model, torch.device("cpu"), {})


def test_dat_color_generator_channels():
    assert _dat({}).G.c == 3                                                    # E6
    assert _dat({"color_space": "ycbcr", "chroma_protect": 1.0}).G.c == 1      # E6-CA: Y only
    assert _dat({"color_space": "ycbcr", "chroma_protect": 0.5}).G.c == 3


def test_dat_color_mix_keeps_chroma_and_trains_generator():
    torch.manual_seed(0)
    m = _dat({"color_space": "ycbcr", "chroma_protect": 1.0, "dat_post_crop": False})
    x = 0.15 + 0.7 * torch.rand(4, 3, N, N)
    x_hat = m.make_x_hat(x)
    ycc_in, ycc_out = rgb_to_ycbcr(x), rgb_to_ycbcr(x_hat)
    unclipped = ((x_hat > 1e-4) & (x_hat < 1 - 1e-4)).all(1, keepdim=True)
    assert ((ycc_out[:, 1:] - ycc_in[:, 1:]).abs() * unclipped).max() < 1e-4
    assert (ycc_out[:, 0] - ycc_in[:, 0]).abs().mean() > 1e-4
    x_hat.sum().backward()                                                      # gradient reaches the generator
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.G.parameters())


def test_rgb_dat_path_unchanged_by_default():
    m = _dat({"dat_post_crop": False})
    assert m.color_space == "rgb" and m.gen_channels == 3
