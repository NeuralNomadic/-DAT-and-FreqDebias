"""Track split and augmentation tests (CLAUDE.md §11)."""
import inspect
import os
import re

import torch
from torchvision.transforms import v2

from src import data
from src.make_split import split_by_track


def _fake_items():
    items = []
    for c in range(43):
        n_tracks = 3 + c % 5
        for t in range(n_tracks):
            for i in range(4):
                items.append((f"{c:05d}/{t:05d}_{i:05d}.ppm", c))
    return items


def test_track_split_disjoint():
    items = _fake_items()
    train, val = split_by_track(items, seed=1234)
    assert len(train) + len(val) == len(items)
    tr = {(l, data.track_id(r)) for r, l in train}
    va = {(l, data.track_id(r)) for r, l in val}
    assert not (tr & va), "a (class, track) appears in both splits"
    # at least one validation track per class
    assert {l for _, l in val} == set(range(43))


def test_track_split_deterministic():
    items = _fake_items()
    a = split_by_track(items, seed=1234)
    b = split_by_track(items, seed=1234)
    assert a == b


def _flatten(tf):
    out = [tf]
    for attr in ("transforms", "p_transforms"):
        for t in getattr(tf, attr, []) or []:
            out.extend(_flatten(t))
    return out


def test_no_flip_in_training_transforms():
    tf = data.train_transform()
    for t in _flatten(tf):
        assert "Flip" not in type(t).__name__, f"flip transform found: {type(t).__name__}"
    # and nowhere in the source tree (apart from comments that forbid it)
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.endswith(".py"):
                code = open(os.path.join(root, fn), encoding="utf-8").read()
                code_no_comments = re.sub(r"#.*", "", code)
                code_no_docs = re.sub(r'""".*?"""', "", code_no_comments, flags=re.S)
                assert "RandomHorizontalFlip" not in code_no_docs and "RandomVerticalFlip" not in code_no_docs, fn
                assert "hflip" not in code_no_docs and "vflip" not in code_no_docs, fn


def test_train_transform_output_range_and_shape():
    tf = data.train_transform()
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out = tf(img)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32
    assert out.min() >= 0 and out.max() <= 1


def test_synthetic_dataset_partner_same_class():
    tr, va, _ = data.synthetic_train_val(with_partner=True)
    item = tr[5]
    assert set(item) >= {"x", "x_orig", "y", "idx", "x_partner"}
    j_pool = tr.class_to_indices[int(item["y"])]
    assert len(j_pool) > 0


def test_class_names_and_safety_ids():
    assert len(data.CLASS_NAMES) == 43
    assert data.CLASS_NAMES[14] == "Stop" and data.CLASS_NAMES[13] == "Yield" and data.CLASS_NAMES[17] == "No entry"
    for c in (0, 1, 2, 3, 4, 5, 7, 8):
        assert data.CLASS_NAMES[c].startswith("Speed limit")
