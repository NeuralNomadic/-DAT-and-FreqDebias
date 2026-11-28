"""Phase 0 of the colour-aware extension (CLAUDE.md §14): accuracy per sign-colour group, and whether the
frequency arms lose more on colour-coded signs than the shared base does. No training, no GPU.

    python -m src.colour_groups --out $OUT

For every evaluated run (seed 0 by default) it reads the per-image results already saved by src.evaluate
(`eval_flags.pt`: clean and PGD-20 correctness of each of the 12,630 test images, from best.pt) and the test labels
(`data_cache/test32.pt`), and writes to OUT/results/:

  table_colour_groups.md/.csv   clean and PGD-20 accuracy per colour group (red / blue / yellow / white) per method
  table_colour_drop.md/.csv     difference from the reference run (E3 for E5/E6, the parent E5/E6 for the
                                colour-aware variants) per group, with a 95 % paired bootstrap interval, and the
                                "colour gap" = drop on red+blue minus drop on white

Why paired: every method is evaluated on the same test images, so resampling images jointly (the same bootstrap
indices for both runs) removes the image-difficulty noise that two independent accuracies would carry.

Reading the result (Phase 0 decision): the motivation for colour-aware mixing is that RGB amplitude mixing HURTS
colour-coded signs. It is supported for E5 and/or E6 only if BOTH hold (see `verdict`):
  1. the colour-coded signs (red + blue together) actually drop: Δ red+blue < 0 with an interval that excludes 0;
  2. they drop more than the achromatic signs (white): colour gap < 0 with an interval that excludes 0.
A negative gap alone is not enough: if red+blue GAIN accuracy but white gains even more, the gap is negative although
nothing was lost on colour-coded signs (this happened for GTSRB E6 at seed 0, NOTES.md N24).
For the colour-aware rows (E5-CA vs E5, E6-CA vs E6) the question is the reverse: did red+blue recover?
Caveats printed with the table: the white group has only 360 test images (wide intervals) and the yellow group is a
single class; the intervals reflect test-image noise only, not seed-to-seed variation.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

from .data import COLOUR_GROUPS, cache_path_for

GROUPS = ["red", "blue", "yellow", "white"]
METRICS = ["clean", "pgd20"]
# (run, reference) pairs for the drop table
REFERENCE = {"E5_fd_at": "E3_base", "E6_dat": "E3_base", "E4_fd_clean": "E0_standard",
             "E5_fd_at_ca": "E5_fd_at", "E6_dat_ca": "E6_dat"}
ORDER = ["E0_standard", "E1_pgdat", "E2_trades", "E3_base", "E4_fd_clean", "E5_fd_at", "E6_dat",
         "E5_fd_at_ca", "E6_dat_ca"]
N_BOOT = 2000


def group_index(labels: np.ndarray):
    """{group: boolean mask over test images}."""
    return {g: np.isin(labels, COLOUR_GROUPS[g]) for g in GROUPS}


def load_flags(out, exp, seed, n_expected):
    p = os.path.join(out, "runs", exp, f"seed{seed}", "eval_flags.pt")
    if not os.path.exists(p):
        return None
    f = torch.load(p, map_location="cpu", weights_only=False)
    res = {}
    for k in METRICS:
        v = f.get(k)
        if v is None or len(v) != n_expected:          # e.g. a --smoke evaluation on 16 images
            return None
        res[k] = np.asarray(v, dtype=bool)
    return res


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng, n_boot=N_BOOT):
    """Mean difference (a - b, in %) and 95 % percentile interval, resampling images jointly."""
    d = a.astype(np.float64) - b.astype(np.float64)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1) * 100
    return d.mean() * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def gap_bootstrap(a, b, masks, rng, n_boot=N_BOOT):
    """Colour gap = drop(red+blue) - drop(white), each drop = acc(a) - acc(b) in %. Stratified resampling
    within each of the two image sets, paired across runs."""
    col = masks["red"] | masks["blue"]
    wht = masks["white"]
    dc = (a[col].astype(float) - b[col].astype(float))
    dw = (a[wht].astype(float) - b[wht].astype(float))
    ic = rng.integers(0, len(dc), size=(n_boot, len(dc)))
    iw = rng.integers(0, len(dw), size=(n_boot, len(dw)))
    boots = (dc[ic].mean(1) - dw[iw].mean(1)) * 100
    return (dc.mean() - dw.mean()) * 100, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def verdict(colour, gap, colour_aware=False):
    """Phase 0 / Phase 3 decision from two (mean, lo, hi) triples, in percentage points:
    colour = Δ on red+blue (method − reference), gap = Δ red+blue − Δ white.

    Phase 0 (method = E5/E6, reference = E3): the colour-aware motivation needs a REAL drop on the colour-coded
    signs (colour interval below 0) that is also larger than on white signs (gap interval below 0).
    Phase 3 (method = E5-CA/E6-CA, reference = their parent): did the colour-coded signs recover (colour interval
    above 0)?"""
    _, c_lo, c_hi = colour
    _, g_lo, g_hi = gap
    if colour_aware:
        if c_lo > 0:
            return "colour-coded signs recover (CA helps)"
        if c_hi < 0:
            return "colour-coded signs get worse (CA hurts)"
        return "no clear change on colour-coded signs"
    if c_hi < 0 and g_hi < 0:
        return "colour-coded signs drop, more than white (motivation SUPPORTED)"
    if c_hi < 0:
        return "colour-coded signs drop, but not clearly more than white (not supported)"
    if g_hi < 0:
        return "no drop on colour-coded signs; white gains more (not supported)"
    if g_lo > 0:
        return "colour-coded signs fare better than white (not supported)"
    return "no clear difference (not supported)"


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _write(res_dir, name, header, rows, note=""):
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = _md_table(header, rows) + (("\n" + note + "\n") if note else "")
    with open(os.path.join(res_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def analyse(out, seed=0, quiet=False):
    """Build both tables. Returns (group_rows, drop_rows) or None if no run has full-size flags."""
    cp = cache_path_for(out, "test")
    if not os.path.exists(cp):
        if not quiet:
            print("no test cache at", cp, "- run the data cell (src.make_split) first")
        return None
    labels = torch.load(cp, weights_only=False)["labels"].numpy()
    masks = group_index(labels)
    flags = {}
    for exp in ORDER:
        f = load_flags(out, exp, seed, len(labels))
        if f is not None:
            flags[exp] = f
    if not flags:
        if not quiet:
            print("no evaluated run with per-image results (eval_flags.pt) found under", os.path.join(out, "runs"))
        return None
    res_dir = os.path.join(out, "results")
    n = {g: int(masks[g].sum()) for g in GROUPS}

    # table 1: accuracy per group
    header = ["Method"] + [f"{g} (n={n[g]}) {m}" for g in GROUPS for m in ("clean", "PGD-20")] + ["all clean", "all PGD-20"]
    group_rows = []
    for exp, f in flags.items():
        row = [exp]
        for g in GROUPS:
            for m in METRICS:
                row.append(f"{f[m][masks[g]].mean() * 100:.2f}")
        row += [f"{f['clean'].mean() * 100:.2f}", f"{f['pgd20'].mean() * 100:.2f}"]
        group_rows.append(row)
    md1 = _write(res_dir, "table_colour_groups", header, group_rows)

    # table 2: drops vs reference, paired bootstrap
    rng = np.random.default_rng(0)
    header2 = (["Method", "vs", "metric"] + [f"Δ {g}" for g in GROUPS]
               + ["Δ red+blue", "colour gap (red+blue − white)", "verdict"])
    colour_mask = masks["red"] | masks["blue"]
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
    drop_rows = []
    for exp, ref in REFERENCE.items():
        if exp not in flags or ref not in flags:
            continue
        for m in METRICS:
            a, b = flags[exp][m], flags[ref][m]
            row = [exp, ref, "clean" if m == "clean" else "PGD-20"]
            for g in GROUPS:
                row.append(fmt(paired_bootstrap(a[masks[g]], b[masks[g]], rng)))
            colour = paired_bootstrap(a[colour_mask], b[colour_mask], rng)
            gap = gap_bootstrap(a, b, masks, rng)
            row += [fmt(colour), fmt(gap), verdict(colour, gap, colour_aware=exp.endswith("_ca"))]
            drop_rows.append(row)
    note = ("Δ = accuracy of the method minus its reference, in percentage points, on the same test images, with a "
            "95 % paired bootstrap interval (2,000 resamples). Δ red+blue < 0 means the colour-coded signs actually "
            "lose accuracy; colour gap < 0 means they fare worse than the white signs. The colour-aware motivation "
            "needs BOTH intervals below 0; a negative gap with Δ red+blue ≥ 0 only means white gained more. "
            f"Caveats: the white group has only {n['white']} test images and yellow is a single class (Priority "
            "road), so their intervals are wide; intervals cover test-image noise only, one seed.")
    md2 = _write(res_dir, "table_colour_drop", header2, drop_rows, note) if drop_rows else ""
    if not quiet:
        print("=== accuracy per colour group (seed %d) ===" % seed)
        print(md1)
        if drop_rows:
            print("=== difference from the reference run ===")
            print(md2)
        else:
            print("(the drop table needs E3 plus E5 and/or E6 evaluated)")
    return group_rows, drop_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    analyse(args.out, args.seed)
    from .storage import get_syncer
    get_syncer(args.out).push_now()


if __name__ == "__main__":
    main()
