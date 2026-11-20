"""Per-image dominant-band masks for Fo-Mixup (CLAUDE.md §6.3) [paper: FreqDebias Alg. 1, steps 3-10].

Requires the trained E0 best.pt as the frozen OHEM model. For every TRAINING image (train.txt order):
  1. mu_seg = mean of log(1 + A) over each segment's positions, averaged over channels
  2. 1-D K-means (k = 8) of the segment means
  3. for each cluster z: band-only image x_z = clip(real(ifft2(ifftshift(A * B_z * exp(iP)))))
  4. CE of the frozen OHEM model (BN-A, eval) on all 8 band-only images
  5. keep the t = 3 clusters with the HIGHEST loss
  6. store cluster index of each segment (uint8[144]) and the 3 chosen cluster IDs
Output: OUT/data_cache/fomixup_masks.pt  (+ OUT/figures/fomixup_grid.png via src.make_grids)
"""
import argparse
import os

import torch
import torch.nn.functional as F

from .common import fp32_ctx, get_device, atomic_save
from .data import gtsrb_present, load_stats, load_train_val, synthetic_train_val
from .fomixup import K_CLUSTERS, T_KEEP, band_filtered_image, band_mask, cluster_segments
from .fourier import fft2_shift
from .models import build_model
from .storage import get_syncer, rel_to_out


def load_ohem_model(out, mean, std, device, smoke=False):
    path = os.path.join(out, "runs", "E0_standard", "seed0", "checkpoints", "best.pt")
    get_syncer(out).ensure(rel_to_out(out, path))   # fetch from storage in a new session
    if not os.path.exists(path):
        if smoke:
            print("smoke: E0 best.pt not found, using a randomly initialised OHEM model")
            return build_model({"dual_bn": False}, mean, std).to(device).eval()
        raise FileNotFoundError(f"{path} missing: train E0 first (CLAUDE.md §9 dependency)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"], mean, std).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def compute_masks(model, imgs_uint8, labels, device, bs=256):
    N = imgs_uint8.shape[0]
    seg_cluster = torch.empty(N, 144, dtype=torch.uint8)
    top = torch.empty(N, T_KEEP, dtype=torch.long)
    for i in range(0, N, bs):
        x = imgs_uint8[i:i + bs].to(device).float() / 255.0
        y = labels[i:i + bs].to(device)
        with fp32_ctx(device):
            A, P = fft2_shift(x)
            sc = cluster_segments(A)                                            # steps 1-2
            losses = []
            for z in range(K_CLUSTERS):
                xz = band_filtered_image(A, P, band_mask(sc, torch.full((len(y),), z, device=device)))  # step 3
                losses.append(F.cross_entropy(model(xz, bn="A"), y, reduction="none"))                 # step 4
            losses = torch.stack(losses, 1)                                     # B x 8
            top[i:i + bs] = losses.topk(T_KEEP, dim=1).indices.cpu()            # step 5: highest loss
            seg_cluster[i:i + bs] = sc.cpu()
        if (i // bs) % 20 == 0:
            print(f"  {i + len(y)}/{N}", flush=True)
    return seg_cluster, top


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    device = get_device(force_cpu=args.cpu or args.smoke)
    if args.smoke and not gtsrb_present(args.data):
        train_ds, _, stats = synthetic_train_val()
        mean, std = stats["mean"], stats["std"]
    else:
        train_ds, _ = load_train_val(args.data, args.out)
        mean, std = load_stats(args.out)
    model = load_ohem_model(args.out, mean, std, device, smoke=args.smoke)
    n = len(train_ds)
    n_real = min(n, 64) if args.smoke else n       # smoke: real OHEM masks for 64 images, random for the rest
    seg_cluster, top = compute_masks(model, train_ds.imgs[:n_real], train_ds.labels[:n_real], device)
    if n_real < n:
        from .fourier import fft2_shift
        A, _ = fft2_shift(train_ds.imgs[n_real:].float() / 255.0)
        seg_cluster = torch.cat([seg_cluster, cluster_segments(A).cpu()])
        top = torch.cat([top, torch.stack([torch.randperm(K_CLUSTERS)[:T_KEEP] for _ in range(n - n_real)])])
    path = os.path.join(args.out, "data_cache", "fomixup_masks.pt")
    atomic_save({"seg_cluster": seg_cluster, "top_clusters": top, "n": n, "smoke": bool(args.smoke),
                 "k": K_CLUSTERS, "t": T_KEEP, "ohem_model": "E0_standard/seed0/best.pt"}, path)
    print("saved", path, seg_cluster.shape, top.shape)
    from .make_grids import fomixup_grid
    fomixup_grid(args.out, train_ds, {"seg_cluster": seg_cluster, "top_clusters": top}, device)
    get_syncer(args.out).push_now()


if __name__ == "__main__":
    main()
