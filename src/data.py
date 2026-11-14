"""GTSRB data: 32x32 cache, track-aware split files, training augmentation (CLAUDE.md §3)."""
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

IMG_SIZE = 32  # [adaptation]

# Official GTSRB class names (German Traffic Sign Recognition Benchmark, class IDs 0-42).
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons",
]
NUM_CLASSES = 43
# CLAUDE.md §7: Stop (14), Yield (13), No entry (17), speed limits (0,1,2,3,4,5,7,8).
SAFETY_CLASSES = [14, 13, 17, 0, 1, 2, 3, 4, 5, 7, 8]


# ----------------------------------------------------------------------------------------------
# Raw dataset access
# ----------------------------------------------------------------------------------------------
def get_gtsrb(root: str, split: str, download: bool = True):
    import torchvision
    return torchvision.datasets.GTSRB(root=root, split=split, download=download)


def gtsrb_present(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "gtsrb", "GTSRB"))


def samples_with_relpaths(ds) -> List[Tuple[str, int]]:
    """(relative path, label) for every sample; relative to the split folder, e.g. 00014/00003_00021.ppm."""
    base = str(ds._target_folder)
    out = []
    for path, label in ds._samples:
        rel = os.path.relpath(str(path), base).replace(os.sep, "/")
        out.append((rel, int(label)))
    return out


def track_id(relpath: str) -> int:
    """Training files are TTTTT_IIIII.ppm; TTTTT is the track (one physical sign, ~30 photos). §3.1"""
    name = os.path.basename(relpath)
    return int(name.split("_")[0])


def resize32(img_uint8_chw: torch.Tensor) -> torch.Tensor:
    """Bilinear, antialias=True, to 32x32 [adaptation]. Input uint8 CxHxW, output uint8 3x32x32."""
    return v2.functional.resize(img_uint8_chw, [IMG_SIZE, IMG_SIZE],
                                interpolation=InterpolationMode.BILINEAR, antialias=True)


def load_image_uint8(path: str) -> torch.Tensor:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        return v2.functional.pil_to_tensor(im)


class _DecodeDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return resize32(load_image_uint8(self.paths[i]))


def build_cache(root: str, split: str, cache_path: str, workers: int = 4) -> dict:
    """Decode every image of a split once, resize to 32x32, store uint8 tensor + labels + relpaths."""
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    ds = get_gtsrb(root, split, download=True)
    items = samples_with_relpaths(ds)
    base = str(ds._target_folder)
    paths = [os.path.join(base, rel) for rel, _ in items]
    loader = DataLoader(_DecodeDataset(paths), batch_size=256, num_workers=workers, shuffle=False)
    imgs = torch.cat([b for b in loader], 0).contiguous()
    cache = {"imgs": imgs, "labels": torch.tensor([l for _, l in items], dtype=torch.long),
             "relpaths": [r for r, _ in items]}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def cache_path_for(out: str, split: str) -> str:
    return os.path.join(out, "data_cache", f"{split}32.pt")


# ----------------------------------------------------------------------------------------------
# Split files
# ----------------------------------------------------------------------------------------------
def read_split_file(path: str) -> List[Tuple[str, int]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, lab = line.rsplit(" ", 1)
            items.append((rel, int(lab)))
    return items


def write_split_file(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rel, lab in items:
            f.write(f"{rel} {lab}\n")


# ----------------------------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------------------------
def train_transform():
    """§3.2. Operates on uint8 3x32x32 tensors, returns float32 in [0,1]. NO FLIPS, EVER (sign semantics)."""
    return v2.Compose([
        v2.RandomCrop(IMG_SIZE, padding=3, padding_mode="reflect"),
        v2.RandomRotation(10),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])


def eval_transform():
    return v2.Compose([v2.ToDtype(torch.float32, scale=True)])


# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------
class GTSRB32(Dataset):
    """In-memory 32x32 GTSRB subset.

    Each item is a dict:
      x         augmented (train) or plain (eval) float image in [0,1]
      x_orig    the un-augmented image (the DAT generator branch uses it, see NOTES.md)
      y         label
      idx       index into this dataset (used to look up precomputed Fo-Mixup masks)
      x_partner (only if with_partner) a random training image of the SAME class, same transform (§5.6 step 1)
    """

    def __init__(self, imgs: torch.Tensor, labels: torch.Tensor, train: bool, with_partner: bool = False):
        assert imgs.dtype == torch.uint8 and imgs.ndim == 4 and imgs.shape[1:] == (3, IMG_SIZE, IMG_SIZE)
        self.imgs = imgs
        self.labels = labels.long()
        self.train = train
        self.with_partner = with_partner
        self.tf = train_transform() if train else eval_transform()
        self.to_float = eval_transform()
        self.class_to_indices = [torch.nonzero(self.labels == c).flatten() for c in range(NUM_CLASSES)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.imgs[i]
        item = {"x": self.tf(img), "x_orig": self.to_float(img), "y": self.labels[i], "idx": i}
        if self.with_partner:
            pool = self.class_to_indices[int(self.labels[i])]
            j = int(pool[torch.randint(len(pool), (1,))])
            item["x_partner"] = self.tf(self.imgs[j])
            item["x_partner_orig"] = self.to_float(self.imgs[j])
        return item


def subset_from_split(cache: dict, items: List[Tuple[str, int]]):
    pos = {r: i for i, r in enumerate(cache["relpaths"])}
    idx = torch.tensor([pos[r] for r, _ in items], dtype=torch.long)
    labels = torch.tensor([l for _, l in items], dtype=torch.long)
    assert torch.equal(labels, cache["labels"][idx]), "split file labels disagree with the dataset"
    return cache["imgs"][idx].contiguous(), labels


def load_train_val(data_root: str, out: str, with_partner: bool = False):
    """Datasets for the saved split (OUT/split/train.txt, val.txt). Never recomputes the split (§3.1)."""
    cache = build_cache(data_root, "train", cache_path_for(out, "train"))
    tr_items = read_split_file(os.path.join(out, "split", "train.txt"))
    va_items = read_split_file(os.path.join(out, "split", "val.txt"))
    tr_imgs, tr_lab = subset_from_split(cache, tr_items)
    va_imgs, va_lab = subset_from_split(cache, va_items)
    return (GTSRB32(tr_imgs, tr_lab, train=True, with_partner=with_partner),
            GTSRB32(va_imgs, va_lab, train=False))


def load_test(data_root: str, out: str):
    cache = build_cache(data_root, "test", cache_path_for(out, "test"))
    return GTSRB32(cache["imgs"], cache["labels"], train=False)


def load_stats(out: str):
    import json
    with open(os.path.join(out, "split", "stats.json")) as f:
        s = json.load(f)
    return s["mean"], s["std"]


# ----------------------------------------------------------------------------------------------
# Synthetic stand-in for --smoke runs on machines without the dataset
# ----------------------------------------------------------------------------------------------
def synthetic_cache(n: int = 256, seed: int = 0) -> dict:
    """Random 'signs' with the GTSRB file-name structure so every script runs offline in --smoke mode."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % NUM_CLASSES
    imgs = torch.randint(0, 256, (n, 3, IMG_SIZE, IMG_SIZE), generator=g, dtype=torch.uint8)
    # paint a class-dependent blob so the model can learn something in a few steps
    for i in range(n):
        c = int(labels[i])
        imgs[i, c % 3, 8:24, 8:24] = 255
    relpaths = [f"{int(labels[i]):05d}/{(i // 30):05d}_{(i % 30):05d}.ppm" for i in range(n)]
    return {"imgs": imgs, "labels": labels, "relpaths": relpaths}


def synthetic_train_val(with_partner: bool = False, n: int = 256):
    cache = synthetic_cache(n)
    k = int(0.8 * n)
    tr = GTSRB32(cache["imgs"][:k], cache["labels"][:k], train=True, with_partner=with_partner)
    va = GTSRB32(cache["imgs"][k:], cache["labels"][k:], train=False)
    return tr, va, {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}


def synthetic_test(n: int = 64):
    cache = synthetic_cache(n, seed=1)
    return GTSRB32(cache["imgs"], cache["labels"], train=False)
