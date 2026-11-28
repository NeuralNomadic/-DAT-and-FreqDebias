"""Aggregate every metrics.json into the thesis tables and figures (CLAUDE.md §8.4-8.5).

    python -m src.aggregate --out $OUT
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# E0-E6 in the CLAUDE.md §8.4 order, then the §14 colour-aware variants (rows appear only once they exist)
MAIN_ORDER = ["E0_standard", "E1_pgdat", "E2_trades", "E3_base", "E4_fd_clean", "E5_fd_at", "E6_dat",
              "E5_fd_at_ca", "E6_dat_ca"]
ABLATIONS = {"A1": "E5_fd_at", "A2": "E5_fd_at", "A3": "E5_fd_at", "A4": "E5_fd_at", "A5": "E6_dat", "A6": "E6_dat"}
COLOUR_AWARE = {"E5_fd_at_ca": "E5_fd_at", "E6_dat_ca": "E6_dat"}                     # §14 variant -> parent
PRETTY = {"E0_standard": "E0 Standard", "E1_pgdat": "E1 PGD-AT", "E2_trades": "E2 TRADES", "E3_base": "E3 Base-AT",
          "E4_fd_clean": "E4 FreqDebias-Clean", "E5_fd_at": "E5 FreqDebias-AT (Arm A)", "E6_dat": "E6 DAT (Arm B)",
          "A1": "A1 E5, B=0 (whole spectrum)", "A2": "A2 E5, uniform band", "A3": "A3 E5, delta=0",
          "A4": "A4 E5, no confidence sampling", "A5": "A5 E6, single BN", "A6": "A6 E6, partner amplitude",
          "E5_fd_at_ca": "E5-CA colour-aware FreqDebias-AT", "E6_dat_ca": "E6-CA colour-aware DAT"}
CORR_KEYS = ["gaussian_noise", "gaussian_blur", "brightness", "contrast", "fog", "jpeg"]


def load_all(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "runs", "*", "seed*", "metrics.json"))):
        with open(p) as f:
            m = json.load(f)
        m["_path"] = p
        rows.append(m)
    return rows


def _get(m, *keys):
    cur = m
    for k in keys:
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


def summarise(rows, exp, getter):
    vals = [getter(m) for m in rows if m.get("exp") == exp]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, len(vals)


def fmt(mean, std, n, bold=False):
    if mean is None:
        return "-"
    s = f"{mean:.2f}" if n <= 1 else f"{mean:.2f} ± {std:.2f}"
    return f"**{s}**" if bold else s


def write_table(df: pd.DataFrame, bold_cols, path_base, higher_better=True):
    # CSV: flat columns  <name>_mean, <name>_std, <name>_n
    flat = {}
    for col in df.columns:
        if any(isinstance(v, tuple) for v in df[col]):
            flat[col + "_mean"] = [v[0] if isinstance(v, tuple) else v for v in df[col]]
            flat[col + "_std"] = [v[1] if isinstance(v, tuple) else None for v in df[col]]
            flat[col + "_n"] = [v[2] if isinstance(v, tuple) else None for v in df[col]]
        else:
            flat[col] = list(df[col])
    pd.DataFrame(flat).to_csv(path_base + ".csv", index=False)
    md = df.copy()
    for col in bold_cols:
        if col not in md.columns:
            continue
        means = {i: v[0] for i, v in md[col].items() if v[0] is not None}
        best = None
        if means:
            best = max(means, key=means.get) if higher_better else min(means, key=means.get)
        md[col] = [fmt(*v, bold=(i == best)) for i, v in md[col].items()]
    for col in md.columns:
        md[col] = [v if not isinstance(v, tuple) else fmt(*v) for v in md[col]]
    with open(path_base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(md.columns) + " |\n")
        f.write("|" + "---|" * len(md.columns) + "\n")
        for _, r in md.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")


def table_main(rows, res):
    cols = {"Clean": lambda m: m.get("clean"), "FGSM": lambda m: m.get("fgsm"), "PGD-20": lambda m: m.get("pgd20"),
            "PGD-100": lambda m: m.get("pgd100"), "CW∞": lambda m: m.get("cw_inf"),
            "AutoAttack": lambda m: m.get("autoattack"), "Patch-6px": lambda m: _get(m, "patch", "6")}
    recs = []
    for exp in MAIN_ORDER:
        if not any(m.get("exp") == exp for m in rows):
            continue
        rec = {"Method": PRETTY[exp], "Seeds": summarise(rows, exp, lambda m: m.get("clean"))[2]}
        for c, g in cols.items():
            rec[c] = summarise(rows, exp, g)
        recs.append(rec)
    if recs:
        write_table(pd.DataFrame(recs), list(cols), os.path.join(res, "table_main"))


def table_corruption(rows, res):
    recs = []
    for exp in MAIN_ORDER:
        if not any(m.get("exp") == exp for m in rows):
            continue
        rec = {"Method": PRETTY[exp], "Clean": summarise(rows, exp, lambda m: m.get("clean")),
               "Corruption mean": summarise(rows, exp, lambda m: m.get("corruption_mean"))}
        for k in CORR_KEYS:
            rec[k] = summarise(rows, exp, lambda m, k=k: (np.mean(m["corruptions"][k]) if _get(m, "corruptions", k) else None))
        recs.append(rec)
    if recs:
        write_table(pd.DataFrame(recs), ["Clean", "Corruption mean"] + CORR_KEYS, os.path.join(res, "table_corruption"))


def table_safety(rows, res):
    recs = []
    for exp in MAIN_ORDER:
        if not any(m.get("exp") == exp for m in rows):
            continue
        rec = {"Method": PRETTY[exp]}
        any_m = next(m for m in rows if m.get("exp") == exp)
        for cid, info in sorted((_get(any_m, "safety_classes") or {}).items(), key=lambda kv: int(kv[0])):
            name = f"{info['name']} ({cid})"
            rec[name + " clean"] = summarise(rows, exp, lambda m, cid=cid: _get(m, "safety_classes", cid, "clean"))
            rec[name + " PGD-20"] = summarise(rows, exp, lambda m, cid=cid: _get(m, "safety_classes", cid, "pgd20"))
        recs.append(rec)
    if recs:
        df = pd.DataFrame(recs)
        write_table(df, [c for c in df.columns if c != "Method"], os.path.join(res, "table_safety_classes"))


def table_cost(rows, res):
    recs = []
    for exp in MAIN_ORDER + list(ABLATIONS):
        if not any(m.get("exp") == exp for m in rows):
            continue
        recs.append({"Method": PRETTY[exp],
                     "sec/epoch": summarise(rows, exp, lambda m: m.get("sec_per_epoch")),
                     "train hours": summarise(rows, exp, lambda m: m.get("train_hours")),
                     "GPU latency (ms, bs1)": summarise(rows, exp, lambda m: m.get("latency_ms_bs1_gpu")),
                     "CPU latency (ms, bs1)": summarise(rows, exp, lambda m: m.get("latency_ms_bs1_cpu")),
                     "GPU throughput (img/s, bs256)": summarise(rows, exp, lambda m: m.get("throughput_bs256_gpu"))})
    if recs:
        write_table(pd.DataFrame(recs), [], os.path.join(res, "table_cost"))


def table_ablation(rows, res, mapping=None, name="table_ablation"):
    """Each variant vs its parent's seed-0 run (ablations: §5.8; colour-aware variants: §14)."""
    recs = []
    cols = {"Clean": lambda m: m.get("clean"), "PGD-20": lambda m: m.get("pgd20"),
            "AutoAttack": lambda m: m.get("autoattack"), "Corruption mean": lambda m: m.get("corruption_mean")}
    for exp, parent in (ABLATIONS if mapping is None else mapping).items():
        if not any(m.get("exp") == exp for m in rows):
            continue
        # compare with the parent's seed-0 run (ablations are seed 0 only)
        p0 = [m for m in rows if m.get("exp") == parent and m.get("seed") == 0]
        rec = {"Variant": PRETTY[exp], "Parent": PRETTY[parent]}
        for c, g in cols.items():
            v = summarise(rows, exp, g)
            pv = g(p0[0]) if p0 else None
            rec[c] = v
            rec[c + " Δ vs parent"] = (None, None, 0) if (v[0] is None or pv is None) else (v[0] - pv, 0.0, 1)
        recs.append(rec)
    if recs:
        write_table(pd.DataFrame(recs), [], os.path.join(res, name))


def figures(rows, out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # eps curve
    fig, ax = plt.subplots(figsize=(6, 4))
    eps = [2, 4, 8, 12, 16]
    for exp in MAIN_ORDER:
        runs = [m for m in rows if m.get("exp") == exp and m.get("eps_sweep")]
        if not runs:
            continue
        arr = np.array([[m["eps_sweep"].get(str(e), np.nan) for e in eps] for m in runs], dtype=float)
        mu, sd = np.nanmean(arr, 0), (np.nanstd(arr, 0) if len(runs) > 1 else np.zeros(len(eps)))
        ax.plot(eps, mu, marker="o", label=PRETTY[exp])
        ax.fill_between(eps, mu - sd, mu + sd, alpha=0.2)
    ax.set_xlabel("ε (/255)"); ax.set_ylabel("PGD-20 accuracy (%)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig_eps_curve.png"), dpi=150); plt.close(fig)
    # corruption severity
    fig, ax = plt.subplots(figsize=(6, 4))
    for exp in MAIN_ORDER:
        runs = [m for m in rows if m.get("exp") == exp and m.get("corruptions")]
        if not runs:
            continue
        arr = np.array([[np.mean([m["corruptions"][k][s] for k in CORR_KEYS]) for s in range(5)] for m in runs])
        ax.plot([1, 2, 3, 4, 5], arr.mean(0), marker="o", label=PRETTY[exp])
    ax.set_xlabel("severity"); ax.set_ylabel("accuracy (%) averaged over corruptions"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig_corruption_severity.png"), dpi=150); plt.close(fig)
    # train curves (seed 0 of E3, E5, E6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for exp in ["E3_base", "E5_fd_at", "E6_dat"]:
        log = os.path.join(out, "runs", exp, "seed0", "train_log.csv")
        if not os.path.exists(log):
            continue
        df = pd.read_csv(log)
        axes[0].plot(df["epoch"], df["val_clean_acc"], label=PRETTY[exp])
        d = df.dropna(subset=["val_pgd20_acc"])
        axes[1].plot(d["epoch"], d["val_pgd20_acc"], marker=".", label=PRETTY[exp])
    axes[0].set_title("validation clean accuracy"); axes[1].set_title("validation PGD-20 accuracy")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig_train_curves.png"), dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:   # tables contain non-ASCII (CW∞, ±); never crash on a non-UTF-8 console
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    rows = load_all(args.out)
    res = os.path.join(args.out, "results")
    os.makedirs(res, exist_ok=True)
    if not rows:
        print("no metrics.json found under", os.path.join(args.out, "runs"))
        return
    table_main(rows, res)
    table_corruption(rows, res)
    table_safety(rows, res)
    table_cost(rows, res)
    table_ablation(rows, res)
    table_ablation(rows, res, COLOUR_AWARE, "table_colour_aware")                    # §14 Phase 3, overall
    try:                                                                              # §14 Phase 0/3, per colour
        from .colour_groups import analyse
        analyse(args.out, quiet=True)
    except Exception as e:                                                            # never block the tables
        print("colour-group tables skipped:", type(e).__name__, e)
    figures(rows, args.out, res)
    print(f"aggregated {len(rows)} runs ->", res)
    from .storage import get_syncer
    get_syncer(args.out).push_now()
    tm = os.path.join(res, "table_main.md")
    if os.path.exists(tm):
        print(open(tm, encoding="utf-8").read())


if __name__ == "__main__":
    main()
