"""Visual checks (CLAUDE.md §6.5): fomixup_grid.png and dat_grid.png under OUT/figures/."""
import argparse
import os

import torch

from .aag import AmplitudeGenerator
from .common import get_device
from .data import gtsrb_present, load_stats, load_train_val, synthetic_train_val
from .fomixup import band_mask, fomixup
from .fourier import fft2_dat, ifft2_dat
from .models import build_model


def _save_grid(rows, col_titles, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, k = len(rows), len(rows[0])
    fig, axes = plt.subplots(n, k, figsize=(1.3 * k, 1.3 * n))
    for r in range(n):
        for c in range(k):
            ax = axes[r, c]
            img = rows[r][c]
            if img.shape[0] == 1:
                ax.imshow(img[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def _pick(ds, n=16):
    """n images of distinct classes plus a same-class partner for each."""
    idx, seen = [], set()
    for i in range(len(ds)):
        c = int(ds.labels[i])
        if c not in seen:
            seen.add(c); idx.append(i)
        if len(idx) == n:
            break
    partners = []
    for i in idx:
        pool = ds.class_to_indices[int(ds.labels[i])]
        partners.append(int(pool[(pool != i).nonzero().flatten()[0]] if len(pool) > 1 else i))
    return torch.tensor(idx), torch.tensor(partners)


@torch.no_grad()
def fomixup_grid(out, ds, masks, device):
    """16 signs: original, x_tilde at xi = 0.25/0.5/0.75 (p = 1 so only xi varies), and the mask B (§6.5)."""
    idx, pidx = _pick(ds)
    x = ds.imgs[idx].to(device).float() / 255.0
    xj = ds.imgs[pidx].to(device).float() / 255.0
    sc = masks["seg_cluster"][idx].to(device)
    z = masks["top_clusters"][idx][:, 0].to(device)
    B = band_mask(sc, z)
    rows = []
    cols = [x] + [fomixup(x, xj, B, xi=xi, p=torch.ones_like(x)) for xi in (0.25, 0.5, 0.75)] + [B]
    for r in range(len(idx)):
        rows.append([c[r] for c in cols])
    _save_grid(rows, ["original", "xi=0.25", "xi=0.5", "xi=0.75", "mask B"],
               os.path.join(out, "figures", "fomixup_grid.png"), "Fo-Mixup (phase of x kept)")


@torch.no_grad()
def dat_grid(out, ds, model, device, generator=None):
    """16 signs: original, x_hat at lambda = 0.25/0.5/0.75 (§6.5). Uses the E6 generator if a checkpoint exists."""
    idx, _ = _pick(ds)
    x = ds.imgs[idx].to(device).float() / 255.0
    G = generator or AmplitudeGenerator().to(device)
    G.eval()
    A, P = fft2_dat(x)
    logits = model(x, bn="A")
    A_G = G(torch.randn(len(idx), G.z_dim, device=device), logits)
    cols = [x] + [ifft2_dat(lam * A_G + (1 - lam) * A, P) for lam in (0.25, 0.5, 0.75)]
    rows = [[c[r] for c in cols] for r in range(len(idx))]
    _save_grid(rows, ["original", "lam=0.25", "lam=0.5", "lam=0.75"],
               os.path.join(out, "figures", "dat_grid.png"), "DAT amplitude mix (phase kept)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", choices=["fomixup", "dat", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = get_device(force_cpu=args.smoke)
    if args.smoke and not gtsrb_present(args.data):
        ds, _, stats = synthetic_train_val()
        mean, std = stats["mean"], stats["std"]
    else:
        ds, _ = load_train_val(args.data, args.out)
        mean, std = load_stats(args.out)
    if args.which in ("fomixup", "both"):
        mp = os.path.join(args.out, "data_cache", "fomixup_masks.pt")
        if os.path.exists(mp):
            fomixup_grid(args.out, ds, torch.load(mp, weights_only=False), device)
        else:
            print("no fomixup_masks.pt yet; run src.precompute_fomixup first")
    if args.which in ("dat", "both"):
        model = build_model({"dual_bn": True}, mean, std).to(device).eval()
        G = None
        for name in ("E6_dat",):
            ck = os.path.join(args.out, "runs", name, "seed0", "checkpoints", "last.pt")
            if os.path.exists(ck):
                st = torch.load(ck, map_location="cpu", weights_only=False)
                model.load_state_dict(st["model"])
                if "generator" in st.get("method_extra", {}):
                    G = AmplitudeGenerator().to(device)
                    G.load_state_dict(st["method_extra"]["generator"])
                print("using E6 checkpoint", ck)
        dat_grid(args.out, ds, model, device, G)


if __name__ == "__main__":
    main()
