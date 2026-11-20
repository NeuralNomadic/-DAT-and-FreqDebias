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
# The official GTSRB archives (same server torchvision uses). torchvision.datasets.GTSRB downloads
# GTSRB-Training_fixed.zip, the 26,640-image *competition* training set; CLAUDE.md §3 requires the official
# 39,209-image final training set, so the archives are downloaded here directly (NOTES.md N18).
GTSRB_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = {
    "train": [("GTSRB_Final_Training_Images.zip", "f33fd80ac59bff73c82d25ab499e03a3")],
    "test": [("GTSRB_Final_Test_Images.zip", "c7e4e6327067d32654124b0fe9e82185"),
             ("GTSRB_Final_Test_GT.zip", "fe31e9c9270bbcd7b84b7f21a9d9d9e5")],
}
EXPECTED_COUNT = {"train": 39209, "test": 12630}


def split_dir(root: str, split: str) -> str:
    return os.path.join(root, "GTSRB", "Final_Training" if split == "train" else "Final_Test", "Images")


def test_gt_csv(root: str) -> str:
    return os.path.join(root, "GT-final_test.csv")


def gtsrb_present(root: str) -> bool:
    return os.path.isdir(split_dir(root, "train")) and os.path.isdir(split_dir(root, "test")) \
        and os.path.isfile(test_gt_csv(root))


def have_real_data(root: str, out: str) -> bool:
    """True when real GTSRB can be used without a download: the split files exist and either the 32x32 caches
    (restored from storage) or the raw images are present. --smoke runs fall back to synthetic data otherwise."""
    split_ok = all(os.path.exists(os.path.join(out, "split", n)) for n in ("train.txt", "val.txt", "stats.json"))
    caches = all(os.path.exists(os.path.join(out, "data_cache", f"{s}32.pt")) for s in ("train", "test"))
    return split_ok and (caches or gtsrb_present(root))


def download_gtsrb(root: str, split: str):
    """Download + extract the official archives for `split` into `root` (skipped when already present)."""
    from torchvision.datasets.utils import download_and_extract_archive
    ready = os.path.isdir(split_dir(root, split)) and (split == "train" or os.path.isfile(test_gt_csv(root)))
    if ready:
        return
    os.makedirs(root, exist_ok=True)
    for name, md5 in ARCHIVES[split]:
        download_and_extract_archive(GTSRB_URL + name, download_root=root, md5=md5, remove_finished=True)


def list_split(root: str, split: str, download: bool = True) -> List[Tuple[str, int, str]]:
    """(relpath, label, full path) for every image of the official split, in a fixed order.
    train: relpath = 'CCCCC/TTTTT_IIIII.ppm' (class folder / track_image), sorted.
    test:  relpath = 'NNNNN.ppm', in GT-final_test.csv order."""
    if download:
        download_gtsrb(root, split)
    base = split_dir(root, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"GTSRB {split} images not found at {base}. Run with download enabled, or check "
                                f"--data (it must be the folder that contains 'GTSRB/').")
    items = []
    if split == "train":
        for cls in sorted(d for d in os.listdir(base) if d.isdigit()):
            for fn in sorted(f for f in os.listdir(os.path.join(base, cls)) if f.endswith(".ppm")):
                items.append((f"{cls}/{fn}", int(cls), os.path.join(base, cls, fn)))
    else:
        import csv
        with open(test_gt_csv(root), newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                items.append((row["Filename"], int(row["ClassId"]), os.path.join(base, row["Filename"])))
    if len(items) != EXPECTED_COUNT[split]:
        raise RuntimeError(f"GTSRB {split}: found {len(items)} images in {base}, expected {EXPECTED_COUNT[split]}. "
                           f"Delete {os.path.join(root, 'GTSRB')} and re-run to download the official archive again.")
    return items


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


CACHE_VERSION = 2   # v1 caches were built from torchvision's 26,640-image archive and are rejected


def _cache_ok(cache, split):
    return (isinstance(cache, dict) and cache.get("version") == CACHE_VERSION
            and len(cache.get("labels", [])) == EXPECTED_COUNT[split])


def build_cache(root: str, split: str, cache_path: str, workers: int = 2) -> dict:
    """Decode every image of a split once, resize to 32x32, store uint8 tensor + labels + relpaths.
    An existing cache is reused only if it has the expected version and image count; otherwise it is rebuilt.
    The raw images are needed (and downloaded) only when the cache has to be built."""
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        if _cache_ok(cache, split):
            return cache
        print(f"cache {cache_path} is stale (version/count mismatch); rebuilding from the official archive")
    items = list_split(root, split, download=True)
    loader = DataLoader(_DecodeDataset([p for _, _, p in items]), batch_size=256, num_workers=workers,
                        shuffle=False)
    imgs = torch.cat([b for b in loader], 0).contiguous()
    cache = {"imgs": imgs, "labels": torch.tensor([l for _, l, _ in items], dtype=torch.long),
             "relpaths": [r for r, _, _ in items], "version": CACHE_VERSION, "split": split}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, cache_path)
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
    missing = [r for r, _ in items if r not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} split-file entries (e.g. {missing[0]}) are not in the image cache: the "
                           f"split was made from a different copy of GTSRB. Delete OUT/split/ and re-run make_split.")
    idx = torch.tensor([pos[r] for r, _ in items], dtype=torch.long)
    labels = torch.tensor([l for _, l in items], dtype=torch.long)
    if not torch.equal(labels, cache["labels"][idx]):
        raise RuntimeError("split file labels disagree with the image cache; delete OUT/split/ and re-run make_split")
    return cache["imgs"][idx].contiguous(), labels


def load_train_val(data_root: str, out: str, with_partner: bool = False):
    """Datasets for the saved split (OUT/split/train.txt, val.txt). Never recomputes the split (§3.1)."""
    for name in ("train.txt", "val.txt", "stats.json"):
        if not os.path.exists(os.path.join(out, "split", name)):
            raise FileNotFoundError(f"{os.path.join(out, 'split', name)} is missing: run "
                                    f"`python -m src.make_split --data DATA --out OUT` first (notebook data cell)")
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
