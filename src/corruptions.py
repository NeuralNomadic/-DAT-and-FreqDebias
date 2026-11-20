"""Corrupted GTSRB test set (CLAUDE.md §3.3): 6 corruptions x severities 1-5, applied to the ORIGINAL-resolution
test image, then resized to 32x32. Deterministic (seed = 1000 * corruption_index + severity). Cached as uint8
.npy under OUT/data_cache/gtsrb_c/ (regenerated if absent).

The six corruptions are implemented here directly with the ImageNet-C formulas and severity constants of
Hendrycks & Dietterich (ICLR 2019) as packaged in `imagecorruptions` 1.1.2. The package itself is not used: it
calls skimage.filters.gaussian(multichannel=True), removed in scikit-image 0.19, imports pkg_resources, and
pulls in opencv-python, which conflicts with Colab's preinstalled OpenCV (NOTES.md N13).
"""
import argparse
import io
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .data import list_split, resize32, synthetic_cache

# (metrics.json key, ImageNet-C name)
CORRUPTIONS = [("gaussian_noise", "gaussian_noise"), ("gaussian_blur", "gaussian_blur"),
               ("brightness", "brightness"), ("contrast", "contrast"), ("fog", "fog"), ("jpeg", "jpeg_compression")]
SEVERITIES = [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------------------------
# ImageNet-C corruptions (constants identical to imagecorruptions 1.1.2)
# ----------------------------------------------------------------------------------------------
def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


def plasma_fractal(mapsize, wibbledecay, rng):
    """Diamond-square plasma fractal in [0,1] (fog texture), as in the ImageNet-C code. mapsize: power of 2."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize, 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, s, rng):
    c = [.08, .12, 0.18, 0.26, 0.38][s - 1]
    x = np.asarray(x, dtype=np.float64) / 255.
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, s, rng):
    """skimage.filters.gaussian(x, sigma=c, multichannel=True) == per-channel ndimage filter, mode='nearest'."""
    c = [1, 2, 3, 4, 6][s - 1]
    x = np.asarray(x, dtype=np.float64) / 255.
    x = gaussian_filter(x, sigma=(c, c, 0), mode="nearest", truncate=4.0)
    return np.clip(x, 0, 1) * 255


def brightness(x, s, rng):
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv   # same formulas as skimage.color.rgb2hsv / hsv2rgb
    c = [.1, .2, .3, .4, .5][s - 1]
    x = np.asarray(x, dtype=np.float64) / 255.
    hsv = rgb_to_hsv(x)
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
    return np.clip(hsv_to_rgb(hsv), 0, 1) * 255


def contrast(x, s, rng):
    c = [0.4, .3, .2, .1, .05][s - 1]
    x = np.asarray(x, dtype=np.float64) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def fog(x, s, rng):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][s - 1]
    shape = np.asarray(x).shape
    map_size = _next_power_of_2(int(max(shape[:2])))
    x = np.asarray(x, dtype=np.float64) / 255.
    max_val = x.max()
    x = x + c[0] * plasma_fractal(map_size, c[1], rng)[:shape[0], :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def jpeg_compression(x, s, rng):
    c = [25, 18, 15, 10, 7][s - 1]
    buf = io.BytesIO()
    Image.fromarray(np.asarray(x, dtype=np.uint8)).save(buf, "JPEG", quality=c)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


_FUNCS = {"gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur, "brightness": brightness,
          "contrast": contrast, "fog": fog, "jpeg_compression": jpeg_compression}


def corrupt_image(img_hwc_uint8, name, severity, rng):
    """Corrupt one original-resolution HxWx3 uint8 image; returns uint8 (truncation, as imagecorruptions)."""
    return np.uint8(_FUNCS[name](img_hwc_uint8, severity, rng))


def to_32(img_hwc_uint8):
    t = torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1)
    return resize32(t)


# ----------------------------------------------------------------------------------------------
# Build / load
# ----------------------------------------------------------------------------------------------
def cache_dir(out):
    return os.path.join(out, "data_cache", "gtsrb_c")


def cache_file(out, key, sev):
    return os.path.join(cache_dir(out), f"{key}_s{sev}.npy")


def all_cache_files(out):
    return [cache_file(out, k, s) for k, _ in CORRUPTIONS for s in SEVERITIES] + \
        [os.path.join(cache_dir(out), "labels.npy")]


def load_test_originals(data_root):
    """All 12,630 official test images at their original resolution (downloaded if needed), in GT order."""
    items = list_split(data_root, "test", download=True)
    imgs = []
    for _, _, path in items:
        with Image.open(path) as im:
            imgs.append(np.asarray(im.convert("RGB")))
    return imgs, np.asarray([lab for _, lab, _ in items])


def _valid(path, n_expected):
    if not os.path.exists(path):
        return False
    return n_expected is None or np.load(path, mmap_mode="r").shape[0] == n_expected


def build(data_root, out, originals=None, verbose=True, n_expected=None):
    """Create every missing (or wrongly sized) cache file. Returns labels (np array)."""
    os.makedirs(cache_dir(out), exist_ok=True)
    labels_path = os.path.join(cache_dir(out), "labels.npy")
    todo = [(ci, key, name, sev) for ci, (key, name) in enumerate(CORRUPTIONS) for sev in SEVERITIES
            if not _valid(cache_file(out, key, sev), n_expected)]
    if not todo and _valid(labels_path, n_expected):
        return np.load(labels_path)
    if originals is None:
        originals, labels = load_test_originals(data_root)
    else:
        originals, labels = originals
    for ci, key, name, sev in todo:
        rng = np.random.RandomState(1000 * ci + sev)                    # §3.3 determinism
        t0 = __import__("time").time()
        arr = np.stack([to_32(corrupt_image(im, name, sev, rng)).numpy() for im in originals])
        path = cache_file(out, key, sev)
        tmp = path + ".tmp.npy"
        np.save(tmp, arr.astype(np.uint8))
        os.replace(tmp, path)
        if verbose:
            print(f"built {key} severity {sev}: {arr.shape} ({__import__('time').time() - t0:.0f}s)", flush=True)
    np.save(labels_path, np.asarray(labels))
    return np.asarray(labels)


def load_corrupted(out, key, sev):
    return torch.from_numpy(np.load(cache_file(out, key, sev)))


def make_figures(out, n_signs=5):
    """5 signs x 5 severities per corruption -> OUT/figures/corruptions/<key>.png (§3.3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(out, "figures", "corruptions")
    os.makedirs(fig_dir, exist_ok=True)
    for key, _ in CORRUPTIONS:
        fig, axes = plt.subplots(n_signs, 5, figsize=(7, 7))
        for si, sev in enumerate(SEVERITIES):
            arr = np.load(cache_file(out, key, sev), mmap_mode="r")
            for r in range(n_signs):
                ax = axes[r, si]
                ax.imshow(np.transpose(arr[r], (1, 2, 0)))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"sev {sev}")
        fig.suptitle(key)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{key}.png"), dpi=120)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        c = synthetic_cache(16, seed=1)
        originals = ([img.permute(1, 2, 0).numpy() for img in c["imgs"]], c["labels"].numpy())
        build(args.data, args.out, originals=originals)
    else:
        build(args.data, args.out)
    make_figures(args.out)
    print("corruption cache ready:", cache_dir(args.out))


if __name__ == "__main__":
    main()
