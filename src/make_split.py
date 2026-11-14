"""Create the track-aware train/validation split and shared split artefacts (CLAUDE.md §3.1, §4.1, §7).

Writes OUT/split/{train.txt, val.txt, stats.json, aa_subset.txt, class_names.json} and builds the 32x32 caches.
Run once; every experiment reads these files and none recomputes the split.
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch

from .data import (CLASS_NAMES, NUM_CLASSES, build_cache, cache_path_for, track_id, write_split_file,
                   subset_from_split, read_split_file)

SPLIT_SEED = 1234          # [adaptation]
VAL_TRACK_FRACTION = 0.10  # [adaptation]
AA_SUBSET_N = 2000         # [adaptation] §7


def split_by_track(items, seed=SPLIT_SEED, frac=VAL_TRACK_FRACTION):
    """Per class, hold out `frac` of the tracks (at least 1) for validation. No track is in both splits."""
    rng = random.Random(seed)
    tracks = defaultdict(lambda: defaultdict(list))   # class -> track -> [items]
    for rel, lab in items:
        tracks[lab][track_id(rel)].append((rel, lab))
    train, val = [], []
    for lab in sorted(tracks):
        tids = sorted(tracks[lab])
        n_val = max(1, int(round(frac * len(tids))))
        val_tids = set(rng.sample(tids, n_val))
        for t in tids:
            (val if t in val_tids else train).extend(tracks[lab][t])
    return train, val


def compute_stats(imgs_uint8: torch.Tensor):
    x = imgs_uint8.float() / 255.0
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    split_dir = os.path.join(args.out, "split")
    os.makedirs(split_dir, exist_ok=True)

    cache = build_cache(args.data, "train", cache_path_for(args.out, "train"), args.workers)
    items = list(zip(cache["relpaths"], cache["labels"].tolist()))
    assert len(items) == 39209, f"expected 39,209 official training images, found {len(items)}"

    tr_path, va_path = os.path.join(split_dir, "train.txt"), os.path.join(split_dir, "val.txt")
    if os.path.exists(tr_path) and os.path.exists(va_path):
        print("split files already exist; not recomputing (CLAUDE.md §3.1)")
        train, val = read_split_file(tr_path), read_split_file(va_path)
    else:
        train, val = split_by_track(items)
        write_split_file(train, tr_path)
        write_split_file(val, va_path)
    # track IDs restart per class, so uniqueness is checked per (class, track)
    tr_ct = {(l, track_id(r)) for r, l in train}
    va_ct = {(l, track_id(r)) for r, l in val}
    assert not (tr_ct & va_ct), "a (class, track) appears in both splits"
    print(f"train {len(train)} images / {len(tr_ct)} tracks; val {len(val)} images / {len(va_ct)} tracks")

    tr_imgs, _ = subset_from_split(cache, train)
    mean, std = compute_stats(tr_imgs)
    with open(os.path.join(split_dir, "stats.json"), "w") as f:
        json.dump({"mean": mean, "std": std, "computed_on": "train split, 32x32, [0,1]"}, f, indent=2)
    print("mean", mean, "std", std)

    with open(os.path.join(split_dir, "aa_subset.txt"), "w") as f:
        f.write("\n".join(str(i) for i in range(AA_SUBSET_N)) + "\n")
    with open(os.path.join(split_dir, "class_names.json"), "w") as f:
        json.dump({str(i): CLASS_NAMES[i] for i in range(NUM_CLASSES)}, f, indent=2)

    test_cache = build_cache(args.data, "test", cache_path_for(args.out, "test"), args.workers)
    assert len(test_cache["labels"]) == 12630, "expected 12,630 official test images"
    print("done:", split_dir)


if __name__ == "__main__":
    main()
