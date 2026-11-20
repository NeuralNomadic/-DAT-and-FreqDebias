"""Resumable training for every experiment (CLAUDE.md §4, §8.2, §10.3).

    python -m src.train --config configs/E3_base.yaml --seed 0 --data $DATA --out $OUT --resume

Writes OUT/runs/<EXP>/seed<k>/{config.yaml, env.json, train_log.csv, checkpoints/last.pt, checkpoints/best.pt}.
`last.pt` is written atomically after every epoch and holds everything needed to continue bit-for-bit
(model, optimizer, scheduler, GradScaler, generator + its optimizer for DAT, RNG states, best metric).
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from .attacks import pgd
from .common import (atomic_save, env_info, fp32_ctx, get_device, load_config, rng_state, run_dir, save_yaml,
                     set_rng_state, set_seed, write_json)
from .data import gtsrb_present, load_stats, load_train_val, synthetic_train_val
from .metrics import accuracy, correct_flags, dataset_tensors
from .methods import build_method, method_needs_partner
from .models import build_model
from .storage import get_syncer, rel_to_out

LOG_COLUMNS = ["epoch", "lr", "train_loss", "train_clean_acc", "val_clean_acc", "val_pgd20_acc",
               "sec_per_epoch", "is_best"]


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt if present")
    ap.add_argument("--smoke", action="store_true", help="CPU, 1 epoch, 2 batches, synthetic data if needed")
    ap.add_argument("--epochs", type=int, default=None, help="override (tests only; see NOTES.md)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", help="cudnn.deterministic (resume test only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def load_fomixup_masks(out, smoke, train_ds, device):
    path = os.path.join(out, "data_cache", "fomixup_masks.pt")
    if os.path.exists(path):
        masks = torch.load(path, map_location="cpu", weights_only=False)
        if not smoke and masks.get("smoke"):
            raise RuntimeError(f"{path} was produced by a --smoke run; re-run `python -m src.precompute_fomixup` "
                               f"without --smoke before training E4/E5/A1-A4")
        if len(masks["seg_cluster"]) != len(train_ds):
            raise RuntimeError(f"{path} has masks for {len(masks['seg_cluster'])} images but the training split has "
                               f"{len(train_ds)}; re-run src.precompute_fomixup")
        return masks
    if not smoke:
        raise FileNotFoundError(f"{path} missing: run `python -m src.precompute_fomixup` after E0 (CLAUDE.md §9)")
    # smoke without the real masks: cluster the segments of the training images, pick random bands
    from .fomixup import cluster_segments, K_CLUSTERS, T_KEEP
    from .fourier import fft2_shift
    X = train_ds.imgs.float().div(255.0).to(device)
    A, _ = fft2_shift(X)
    seg_cluster = cluster_segments(A).cpu()
    top = torch.stack([torch.randperm(K_CLUSTERS)[:T_KEEP] for _ in range(len(train_ds))])
    return {"seg_cluster": seg_cluster, "top_clusters": top, "smoke": True}


@torch.no_grad()
def validate(model, X, Y, device, eps, alpha, with_pgd, bs=256, max_n=None):
    """Clean accuracy every epoch; PGD-20 (full validation split, BN-A, float32) when `with_pgd` (§4.4)."""
    if max_n is not None:
        X, Y = X[:max_n], Y[:max_n]
    clean = accuracy(correct_flags(model, X, Y, None, bs, device))
    pgd20 = None
    if with_pgd:
        with torch.enable_grad():
            atk = lambda m, x, y: pgd(m, x, y, eps=eps, alpha=alpha, steps=20, random_start=True, loss="ce")
            pgd20 = accuracy(correct_flags(model, X, Y, atk, bs, device))
    return clean, pgd20


def write_log(path, rows, columns):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    os.replace(tmp, path)


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    smoke = args.smoke
    max_batches = args.max_batches
    if smoke:
        cfg["batch_size"] = 8
        cfg["eval_every"] = 1
        cfg["amp"] = False
        cfg["epochs"] = args.epochs or 1
        max_batches = max_batches or 2
        args.workers = 0
    if args.epochs is not None and not smoke:
        cfg["epochs"] = args.epochs
        print(f"WARNING: epochs overridden to {args.epochs}; CLAUDE.md §4.3 requires the same schedule for all")
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)
    syncer = get_syncer(args.out)
    done_path = os.path.join(run_dir(args.out, cfg["exp"], args.seed), "DONE")
    if args.resume and not smoke and os.path.exists(done_path):
        done = open(done_path).read().strip()
        if done.split()[:2] == ["epochs", str(cfg["epochs"])]:
            print(f"training already complete ({done}); nothing to do")
            syncer.push_now()
            return os.path.dirname(done_path)
        print(f"DONE marker is for a different schedule ({done}); continuing to {cfg['epochs']} epochs")
    device = get_device(force_cpu=args.cpu or smoke)
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    # ---------------- data ----------------
    with_partner = method_needs_partner(cfg)
    if smoke and not gtsrb_present(args.data):
        train_ds, val_ds, stats = synthetic_train_val(with_partner)
        mean, std = stats["mean"], stats["std"]
    else:
        train_ds, val_ds = load_train_val(args.data, args.out, with_partner)
        mean, std = load_stats(args.out)
    Xv, Yv = dataset_tensors(val_ds)

    # ---------------- model / method / optimizer ----------------
    model = build_model(cfg, mean, std).to(device)
    extra = {}
    if cfg["method"] == "fd":
        extra["fomixup_masks"] = load_fomixup_masks(args.out, smoke, train_ds, device)
    method = build_method(cfg, model, device, extra)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])                       # §4.3 [paper: DAT]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    rd = run_dir(args.out, cfg["exp"], args.seed)
    ck_dir = os.path.join(rd, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(rd, "config.yaml"))
    write_json(env_info(device), os.path.join(rd, "env.json"))
    log_path = os.path.join(rd, "train_log.csv")
    columns = LOG_COLUMNS + list(method.log_keys)

    start_epoch, best_metric, best_epoch, train_hours = 0, float("-inf"), -1, 0.0
    rows = []
    last_path, best_path = os.path.join(ck_dir, "last.pt"), os.path.join(ck_dir, "best.pt")
    if args.resume:
        syncer.ensure(rel_to_out(args.out, last_path))     # new Colab session: fetch the checkpoint from storage
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        method.load_state_dict_extra(ck.get("method_extra", {}))
        start_epoch = ck["epoch"]
        best_metric, best_epoch, train_hours = ck["best_metric"], ck["best_epoch"], ck.get("train_hours", 0.0)
        set_rng_state(ck["rng"])
        rows = [r for r in read_log(log_path) if int(r["epoch"]) <= start_epoch]
        print(f"resumed from epoch {start_epoch} (best {best_metric:.2f} @ {best_epoch})")
    elif os.path.exists(last_path) and not args.resume:
        print("WARNING: last.pt exists but --resume not given; starting from scratch and overwriting it")

    select = cfg.get("select_metric", "pgd20")        # §4.4: 'pgd20' (AT methods) or 'clean' (E0, E4)
    eps, alpha = cfg["eps"] / 255.0, cfg["alpha"] / 255.0
    val_max_n = 32 if smoke else None
    epochs = cfg["epochs"]
    if start_epoch >= epochs:
        print("training already complete")
        return rd

    for epoch in range(start_epoch, epochs):
        g = torch.Generator().manual_seed(args.seed * 100003 + epoch)   # shuffle order depends only on (seed, epoch)
        loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g,
                            num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=False)
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n, nb = 0.0, 0, 0, 0
        extra_sums = {k: 0.0 for k in method.log_keys}
        lr_now = optimizer.param_groups[0]["lr"]
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logs = method.train_step(batch, optimizer, scaler, amp)
            if not torch.isfinite(torch.tensor(logs["loss"])):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} batch {bi}: {logs}")
            tot_loss += logs["loss"] * logs["n"]
            tot_correct += logs["correct"]
            tot_n += logs["n"]
            nb += 1
            for k in method.log_keys:
                extra_sums[k] += float(logs.get(k, 0.0))
        scheduler.step()
        sec = time.time() - t0
        train_hours += sec / 3600.0

        with_pgd = ((epoch + 1) % int(cfg.get("eval_every", 2)) == 0) or (epoch + 1 == epochs)
        val_clean, val_pgd = validate(model, Xv, Yv, device, eps, alpha, with_pgd, max_n=val_max_n)
        metric = val_pgd if select == "pgd20" else val_clean
        is_best = metric is not None and metric > best_metric
        if is_best:
            best_metric, best_epoch = metric, epoch + 1
            atomic_save({"model": model.state_dict(), "epoch": epoch + 1, "metric": metric, "select": select,
                         "cfg": cfg, "method_extra": method.state_dict_extra()}, best_path)

        row = {"epoch": epoch + 1, "lr": lr_now, "train_loss": tot_loss / max(tot_n, 1),
               "train_clean_acc": 100.0 * tot_correct / max(tot_n, 1), "val_clean_acc": val_clean,
               "val_pgd20_acc": val_pgd, "sec_per_epoch": sec, "is_best": int(is_best)}
        for k in method.log_keys:
            row[k] = extra_sums[k] / max(nb, 1)
        rows.append(row)
        write_log(log_path, rows, columns)

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "method_extra": method.state_dict_extra(), "epoch": epoch + 1,
                     "best_metric": best_metric, "best_epoch": best_epoch, "rng": rng_state(),
                     "train_hours": train_hours, "cfg": cfg}, last_path)
        pgd_str = "" if val_pgd is None else f" val_pgd20 {val_pgd:.2f}"
        print(f"[{cfg['exp']} s{args.seed}] epoch {epoch + 1}/{epochs} lr {lr_now:.4f} loss {row['train_loss']:.4f} "
              f"train_acc {row['train_clean_acc']:.2f} val_clean {val_clean:.2f}{pgd_str} "
              f"{'*' if is_best else ''} ({sec:.0f}s)", flush=True)
        syncer.maybe_push()                                # throttled background upload (OUT_SYNC_INTERVAL)

    with open(os.path.join(rd, "DONE"), "w") as f:
        f.write(f"epochs {epochs} best_epoch {best_epoch} best_{select} {best_metric:.4f} "
                f"train_hours {train_hours:.3f}\n")
    syncer.push_now(squash=True)
    return rd


if __name__ == "__main__":
    main()
