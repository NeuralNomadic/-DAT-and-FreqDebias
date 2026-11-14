"""Corrupted GTSRB test set (CLAUDE.md §3.3): 6 corruptions x severities 1-5, applied to the ORIGINAL-resolution
test image, then resized to 32x32. Deterministic (seed = 1000 * corruption_index + severity). Cached as uint8
.npy under OUT/data_cache/gtsrb_c/ (regenerated if absent).

Primary implementation: the `imagecorruptions` package. If it is missing or fails on an image, the fallback
implementations below (ImageNet-C formulas) are used and the event is logged to OUT/data_cache/gtsrb_c/fallbacks.txt
so it can be recorded in NOTES.md.
"""
import argparse
import io
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

from .data import get_gtsrb, resize32, synthetic_cache

# (metrics.json key, imagecorruptions name)
CORRUPTIONS = [("gaussian_noise", "gaussian_noise"), ("gaussian_blur", "gaussian_blur"),
               ("brightness", "brightness"), ("contrast", "contrast"), ("fog", "fog"), ("jpeg", "jpeg_compression")]
SEVERITIES = [1, 2, 3, 4, 5]
MIN_SIDE = 32  # images smaller than this are upsampled before corrupting (see NOTES.md)

try:
    from imagecorruptions import corrupt as _ic_corrupt
except Exception:  # pragma: no cover
    _ic_corrupt = None


# ----------------------------------------------------------------------------------------------
# Fallbacks (ImageNet-C parameterisation)
# ----------------------------------------------------------------------------------------------
def _plasma_fractal(mapsize=256, wibbledecay=3.0, rng=np.random):
    """Diamond-square plasma fractal in [0,1] (fog texture), after Hendrycks & Dietterich's ImageNet-C code."""
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
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def _fallback(img, name, severity):
    x = img.astype(np.float32) / 255.0
    s = severity - 1
    if name == "gaussian_noise":
        c = [.08, .12, 0.18, 0.26, 0.38][s]
        x = x + np.random.normal(size=x.shape, scale=c)
    elif name == "gaussian_blur":
        c = [1, 2, 3, 4, 6][s]
        x = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=c)), dtype=np.float32) / 255.0
    elif name == "brightness":
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        c = [.1, .2, .3, .4, .5][s]
        hsv = rgb_to_hsv(x)
        hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
        x = hsv_to_rgb(hsv)
    elif name == "contrast":
        c = [0.4, .3, .2, .1, .05][s]
        means = x.mean(axis=(0, 1), keepdims=True)
        x = (x - means) * c + means
    elif name == "fog":
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][s]
        h, w = x.shape[:2]
        size = 256
        while size < max(h, w):
            size *= 2
        mx = x.max()
        x = x + c[0] * _plasma_fractal(mapsize=size, wibbledecay=c[1])[:h, :w][..., None]
        x = x * mx / (mx + c[0])
    elif name == "jpeg_compression":
        c = [25, 18, 15, 10, 7][s]
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=c)
        x = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    else:
        raise ValueError(name)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def corrupt_image(img_hwc_uint8, ic_name, severity, log=None):
    img = img_hwc_uint8
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:                      # upsample tiny signs so every corruption is well defined
        scale = MIN_SIDE / min(h, w)
        img = np.asarray(Image.fromarray(img).resize((max(MIN_SIDE, round(w * scale)),
                                                      max(MIN_SIDE, round(h * scale))), Image.BILINEAR))
    if _ic_corrupt is not None:
        try:
            out = _ic_corrupt(img, corruption_name=ic_name, severity=severity)
            return np.asarray(out, dtype=np.uint8)
        except Exception as e:  # pragma: no cover
            if log is not None:
                log.append(f"{ic_name} s{severity}: imagecorruptions failed ({e!r}); used fallback")
    return _fallback(img, ic_name, severity)


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


def load_test_originals(data_root, n=None):
    ds = get_gtsrb(data_root, "test", download=True)
    imgs, labels = [], []
    for path, lab in ds._samples[:n]:
        with Image.open(path) as im:
            imgs.append(np.asarray(im.convert("RGB")))
        labels.append(int(lab))
    return imgs, np.asarray(labels)


def build(data_root, out, n=None, originals=None, verbose=True):
    """Create every missing cache file. Returns labels (np array)."""
    os.makedirs(cache_dir(out), exist_ok=True)
    if originals is None:
        originals, labels = load_test_originals(data_root, n)
    else:
        originals, labels = originals
    log = []
    for ci, (key, ic_name) in enumerate(CORRUPTIONS):
        for sev in SEVERITIES:
            path = cache_file(out, key, sev)
            if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == len(originals):
                continue
            np.random.seed(1000 * ci + sev)                             # §3.3 determinism
            arr = np.stack([to_32(corrupt_image(im, ic_name, sev, log)).numpy() for im in originals])
            tmp = path + ".tmp.npy"
            np.save(tmp, arr.astype(np.uint8))
            os.replace(tmp, path)
            if verbose:
                print(f"built {key} severity {sev}: {arr.shape}", flush=True)
    if log:
        with open(os.path.join(cache_dir(out), "fallbacks.txt"), "a") as f:
            f.write("\n".join(sorted(set(log))) + "\n")
    np.save(os.path.join(cache_dir(out), "labels.npy"), labels)
    return labels


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
