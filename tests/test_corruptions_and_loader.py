"""Corruption implementations (src/corruptions.py) and the official-archive GTSRB loader (src/data.py)."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from src import corruptions as corr
from src import data


@pytest.mark.parametrize("name", [n for _, n in corr.CORRUPTIONS])
def test_corruptions_shape_dtype_determinism(name):
    rs = np.random.RandomState(0)
    img = rs.randint(0, 256, (27, 25, 3)).astype(np.uint8)       # GTSRB-sized, non-square, < 32 px
    for sev in corr.SEVERITIES:
        a = corr.corrupt_image(img, name, sev, np.random.RandomState(7))
        b = corr.corrupt_image(img, name, sev, np.random.RandomState(7))
        assert a.dtype == np.uint8 and a.shape == img.shape
        assert np.array_equal(a, b), "same seed must give the same corruption"
        assert corr.to_32(a).shape == (3, 32, 32)


def test_corruption_strength_increases_with_severity():
    rs = np.random.RandomState(1)
    img = rs.randint(40, 216, (40, 40, 3)).astype(np.uint8)
    for name in ["gaussian_noise", "gaussian_blur", "contrast", "fog", "jpeg_compression"]:
        d = [np.abs(corr.corrupt_image(img, name, s, np.random.RandomState(3)).astype(float) - img).mean()
             for s in (1, 5)]
        assert d[1] > d[0], name


def test_build_cache_files_and_rebuild_wrong_size(tmp_path):
    out = str(tmp_path / "out")
    rs = np.random.RandomState(0)
    originals = ([rs.randint(0, 256, (30 + i, 28, 3)).astype(np.uint8) for i in range(6)], np.arange(6))
    labels = corr.build(None, out, originals=originals, verbose=False, n_expected=6)
    assert list(labels) == list(range(6))
    for p in corr.all_cache_files(out):
        assert os.path.exists(p)
    assert np.load(corr.cache_file(out, "fog", 3)).shape == (6, 3, 32, 32)
    # a cache of the wrong size (e.g. a leftover smoke cache) is rebuilt
    small = ([o for o in originals[0][:2]], np.arange(2))
    corr.build(None, out, originals=small, verbose=False, n_expected=2)
    assert np.load(corr.cache_file(out, "jpeg", 1)).shape[0] == 2


def _fake_gtsrb(root, n_train_per_class=4, classes=(0, 14), n_test=5):
    tr = data.split_dir(root, "train")
    for c in classes:
        d = os.path.join(tr, f"{c:05d}")
        os.makedirs(d)
        for i in range(n_train_per_class):
            Image.fromarray(np.full((30, 31, 3), 10 * c + i, np.uint8)).save(os.path.join(d, f"{i // 2:05d}_{i % 2:05d}.ppm"))
        open(os.path.join(d, f"GT-{c:05d}.csv"), "w").close()       # ignored (not a .ppm)
    te = data.split_dir(root, "test")
    os.makedirs(te)
    with open(data.test_gt_csv(root), "w") as f:
        f.write("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n")
        for i in range(n_test):
            Image.fromarray(np.full((33, 29, 3), i, np.uint8)).save(os.path.join(te, f"{i:05d}.ppm"))
            f.write(f"{i:05d}.ppm;29;33;5;5;24;28;{(i * 7) % 43}\n")


def test_list_split_layout_and_count_check(tmp_path, monkeypatch):
    root = str(tmp_path / "data")
    _fake_gtsrb(root)
    monkeypatch.setitem(data.EXPECTED_COUNT, "train", 8)
    monkeypatch.setitem(data.EXPECTED_COUNT, "test", 5)
    assert data.gtsrb_present(root)
    tr = data.list_split(root, "train", download=False)
    assert [r for r, _, _ in tr][:2] == ["00000/00000_00000.ppm", "00000/00000_00001.ppm"]
    assert {l for _, l, _ in tr} == {0, 14}
    assert data.track_id(tr[2][0]) == 1
    te = data.list_split(root, "test", download=False)
    assert [l for _, l, _ in te] == [0, 7, 14, 21, 28]
    monkeypatch.setitem(data.EXPECTED_COUNT, "train", 39209)
    with pytest.raises(RuntimeError, match="expected 39209"):
        data.list_split(root, "train", download=False)


def test_build_cache_rejects_stale_cache(tmp_path, monkeypatch):
    root = str(tmp_path / "data")
    _fake_gtsrb(root)
    monkeypatch.setitem(data.EXPECTED_COUNT, "train", 8)
    path = str(tmp_path / "out" / "data_cache" / "train32.pt")
    os.makedirs(os.path.dirname(path))
    torch.save({"imgs": torch.zeros(3, 3, 32, 32, dtype=torch.uint8), "labels": torch.zeros(3),
                "relpaths": ["a", "b", "c"]}, path)                  # old (version 1) cache
    cache = data.build_cache(root, "train", path, workers=0)
    assert cache["version"] == data.CACHE_VERSION and len(cache["labels"]) == 8
    assert cache["imgs"].shape == (8, 3, 32, 32) and cache["imgs"].dtype == torch.uint8
    again = data.build_cache(root, "train", path, workers=0)      # valid cache is reused
    assert again["relpaths"] == cache["relpaths"]


def test_missing_split_files_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_split"):
        data.load_train_val(str(tmp_path / "data"), str(tmp_path / "out"))
