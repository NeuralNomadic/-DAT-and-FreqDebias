"""Full evaluation protocol for one trained run (CLAUDE.md §7, §8.3). Resumable: metrics already present in
metrics.json are skipped, and AutoAttack is check-pointed per attack.

    python -m src.evaluate --exp E6_dat --seed 0 --data $DATA --out $OUT
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from . import corruptions as corr
from .attacks import fgsm, patch_attack, pgd
from .common import env_info, fp32_ctx, get_device, read_json, run_dir, set_seed, write_json
from .storage import get_syncer, rel_to_out
from .data import (CLASS_NAMES, NUM_CLASSES, SAFETY_CLASSES, gtsrb_present, load_stats, load_test, synthetic_test)
from .metrics import accuracy, correct_flags, dataset_tensors
from .models import build_model

EPS, ALPHA = 8 / 255, 2 / 255
AA_ATTACKS = ["apgd-ce", "apgd-t", "fab-t", "square"]


def r2(x):
    return None if x is None else round(float(x), 2)


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.rd = run_dir(args.out, args.exp, args.seed)
        self.metrics_path = os.path.join(self.rd, "metrics.json")
        self.flags_path = os.path.join(self.rd, "eval_flags.pt")
        self.m = read_json(self.metrics_path, {}) if not args.force else {}
        self.flags = torch.load(self.flags_path, weights_only=False) if (os.path.exists(self.flags_path) and not args.force) else {}
        self.device = get_device(force_cpu=args.cpu or args.smoke)
        set_seed(0)

        self.syncer = get_syncer(args.out)
        ck_path = os.path.join(self.rd, "checkpoints", "best.pt")
        for name in ("best.pt", "last.pt"):        # new Colab session: fetch checkpoints from storage
            self.syncer.ensure(rel_to_out(args.out, os.path.join(self.rd, "checkpoints", name)))
        if not os.path.exists(ck_path):
            raise FileNotFoundError(f"{ck_path} not found locally or in storage: train {args.exp} seed {args.seed} first")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        if args.smoke and not gtsrb_present(args.data):
            mean, std = [0.5] * 3, [0.25] * 3
            ds = synthetic_test(64)
        else:
            mean, std = load_stats(args.out)
            ds = load_test(args.data, args.out)
        self.model = build_model(self.cfg, mean, std).to(self.device).float().eval()
        self.model.load_state_dict(ck["model"])
        self.X, self.Y = dataset_tensors(ds)
        if args.smoke:
            self.X, self.Y = self.X[:16], self.Y[:16]
        self.n_sub = min(2000, len(self.Y)) if not args.smoke else min(16, len(self.Y))
        aa_file = os.path.join(args.out, "split", "aa_subset.txt")
        if os.path.exists(aa_file) and not args.smoke:
            idx = [int(l) for l in open(aa_file) if l.strip()]
            self.sub_idx = torch.tensor(idx)
        else:
            self.sub_idx = torch.arange(self.n_sub)

        self.m.update({"exp": args.exp, "seed": args.seed, "epochs": self.cfg["epochs"],
                       "best_epoch": ck["epoch"], "select_metric": ck.get("select", "pgd20")})
        env = read_json(os.path.join(self.rd, "env.json"), None) or env_info(self.device)
        for k in ("git_commit", "torch", "torchvision", "python", "gpu"):
            self.m[k] = env.get(k)
        self._training_cost()
        self.bs = 64 if args.smoke else args.batch_size
        if args.smoke:
            self.m["smoke"] = True   # attack steps are divided by 20 in smoke mode; numbers are NOT reportable
        self.save()

    def steps(self, n):
        """Full attack strength normally; in --smoke mode steps are divided by 20 so the script runs in < 1 min."""
        return max(2, n // 20) if self.args.smoke else n

    # ------------------------------------------------------------------------------------------
    def save(self):
        write_json(self.m, self.metrics_path)
        tmp = self.flags_path + ".tmp"
        torch.save(self.flags, tmp)
        os.replace(tmp, self.flags_path)
        if hasattr(self, "syncer"):
            self.syncer.maybe_push()

    def have(self, key):
        return key in self.m and self.m[key] is not None

    def _training_cost(self):
        log = os.path.join(self.rd, "train_log.csv")
        if os.path.exists(log):
            with open(log, newline="") as f:
                rows = list(csv.DictReader(f))
            secs = [float(r["sec_per_epoch"]) for r in rows if r.get("sec_per_epoch")]
            self.m["sec_per_epoch"] = r2(np.mean(secs)) if secs else None
        last = os.path.join(self.rd, "checkpoints", "last.pt")
        if os.path.exists(last):
            ck = torch.load(last, map_location="cpu", weights_only=False)
            self.m["train_hours"] = round(float(ck.get("train_hours", 0.0)), 3)

    def flags_for(self, key, attack, X=None, Y=None, desc=None):
        if key in self.flags:
            return self.flags[key]
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        f = correct_flags(self.model, X, Y, attack, self.bs, self.device, desc=desc or key)
        self.flags[key] = f
        return f

    # ------------------------------------------------------------------------------------------
    def white_box(self):
        specs = {
            "clean": None,
            "fgsm": lambda m, x, y: fgsm(m, x, y, eps=EPS),
            "pgd20": lambda m, x, y: pgd(m, x, y, EPS, ALPHA, self.steps(20), True, "ce"),
            "pgd100": lambda m, x, y: pgd(m, x, y, EPS, ALPHA, self.steps(100), True, "ce"),
            "cw_inf": lambda m, x, y: pgd(m, x, y, EPS, ALPHA, self.steps(30), True, "cw"),
        }
        for key, atk in specs.items():
            if self.have(key):
                continue
            self.m[key] = r2(accuracy(self.flags_for(key, atk)))
            print(f"{key}: {self.m[key]}", flush=True)
            self.save()

    def eps_sweep(self):
        if self.have("eps_sweep") and len(self.m["eps_sweep"]) == 5:
            return
        res = self.m.get("eps_sweep", {}) or {}
        for e in [2, 4, 8, 12, 16]:
            if str(e) in res:
                continue
            eps = e / 255
            atk = lambda m, x, y, eps=eps: pgd(m, x, y, eps, eps / 4, self.steps(20), True, "ce")
            res[str(e)] = r2(accuracy(self.flags_for(f"pgd20_eps{e}", atk)))
            self.m["eps_sweep"] = res
            self.save()
        print("eps sweep:", res, flush=True)

    def autoattack(self):
        if self.have("autoattack") and self.have("autoattack_per_attack") and len(self.m["autoattack_per_attack"]) == 4:
            return
        if self.args.smoke or self.args.skip_aa:
            self.m["autoattack"] = None
            self.m["autoattack_n"] = 0
            self.m["autoattack_per_attack"] = {}
            self.m["autoattack_note"] = "skipped (smoke / --skip-aa)"
            self.save()
            return
        try:
            from autoattack import AutoAttack
        except ImportError:
            print("autoattack not installed; pip install git+https://github.com/fra31/auto-attack", file=sys.stderr)
            return
        x = self.X[self.sub_idx].to(self.device)
        y = self.Y[self.sub_idx].to(self.device)
        per = self.m.get("autoattack_per_attack", {}) or {}
        with fp32_ctx(self.device):
            for atk in AA_ATTACKS:
                key = f"aa_{atk}"
                if key not in self.flags:
                    adv = AutoAttack(self.model, norm="Linf", eps=EPS, version="standard", verbose=True)
                    adv.attacks_to_run = [atk]                 # standard-version attack, run alone -> standalone acc
                    x_adv = adv.run_standard_evaluation(x, y, bs=self.args.aa_bs)
                    with torch.no_grad():
                        pred = torch.cat([self.model(x_adv[i:i + 256]).argmax(1) for i in range(0, len(y), 256)])
                    self.flags[key] = (pred == y).cpu()
                per[atk] = r2(accuracy(self.flags[key]))
                self.m["autoattack_per_attack"] = per
                self.save()
        robust = torch.stack([self.flags[f"aa_{a}"] for a in AA_ATTACKS]).all(0)   # robust to every attack
        self.flags["aa_all"] = robust
        self.m["autoattack"] = r2(accuracy(robust))
        self.m["autoattack_n"] = int(len(y))
        self.save()
        print("autoattack:", self.m["autoattack"], per, flush=True)

    def corruptions(self):
        if self.have("corruptions") and len(self.m["corruptions"]) == 6 and all(len(v) == 5 for v in self.m["corruptions"].values()):
            return
        n = len(self.Y)
        # smoke runs keep their tiny synthetic corruption cache apart from the real one
        corr_out = os.path.join(self.args.out, ".smoke") if self.args.smoke else self.args.out
        self.corr_out = corr_out
        if self.args.smoke:
            c = corr.synthetic_cache(n, seed=1)
            labels = corr.build(self.args.data, corr_out, n_expected=n,
                                originals=([im.permute(1, 2, 0).numpy() for im in c["imgs"]], c["labels"].numpy()),
                                verbose=False)
        else:
            for p in corr.all_cache_files(corr_out):    # reuse arrays built in an earlier session
                self.syncer.ensure(rel_to_out(self.args.out, p))
            labels = corr.build(self.args.data, corr_out, n_expected=n)
            if not os.path.exists(os.path.join(self.args.out, "figures", "corruptions", "fog.png")):
                corr.make_figures(self.args.out)
            self.syncer.push_now()
        Y = torch.from_numpy(labels).long()
        res = self.m.get("corruptions", {}) or {}
        rows = []
        for key, _ in corr.CORRUPTIONS:
            accs = []
            for sev in corr.SEVERITIES:
                Xc = corr.load_corrupted(self.corr_out, key, sev).float() / 255.0
                f = self.flags_for(f"corr_{key}_s{sev}", None, Xc, Y)
                accs.append(r2(accuracy(f)))
                rows.append({"corruption": key, "severity": sev, "acc": accs[-1]})
            res[key] = accs
            self.m["corruptions"] = res
            self.save()
        self.m["corruption_mean"] = r2(np.mean([a for v in res.values() for a in v]))
        with open(os.path.join(self.rd, "corruptions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "severity", "acc"])
            w.writeheader(); w.writerows(rows)
        self.save()
        print("corruption mean:", self.m["corruption_mean"], flush=True)

    def patch(self):
        if self.have("patch") and all(str(s) in self.m["patch"] for s in (4, 6, 8)):
            return
        idx = self.sub_idx
        X, Y = self.X[idx], self.Y[idx]
        res = self.m.get("patch", {}) or {}
        steps = self.steps(100)
        for side in (4, 6, 8):
            if str(side) in res:
                continue
            pos = 0
            flags = []
            for i in range(0, len(Y), self.bs):
                x = X[i:i + self.bs].to(self.device); y = Y[i:i + self.bs].to(self.device)
                ids = idx[i:i + self.bs].tolist()
                x_adv = patch_attack(self.model, x, y, ids, side, steps=steps, step=0.05)
                with torch.no_grad(), fp32_ctx(self.device):
                    flags.append((self.model(x_adv).argmax(1) == y).cpu())
            f = torch.cat(flags)
            self.flags[f"patch_{side}"] = f
            res[str(side)] = r2(accuracy(f))
            res["n"] = int(len(Y))
            self.m["patch"] = res
            self.save()
        with open(os.path.join(self.rd, "patch.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["side_px", "acc", "n"])
            for s in (4, 6, 8):
                w.writerow([s, res[str(s)], res["n"]])
        print("patch:", res, flush=True)

    def per_class(self):
        if self.have("safety_classes") and os.path.exists(os.path.join(self.rd, "per_class.csv")):
            return
        clean, p20 = self.flags["clean"], self.flags["pgd20"]
        rows, safety = [], {}
        for c in range(NUM_CLASSES):
            sel = self.Y == c
            n = int(sel.sum())
            ca = r2(accuracy(clean[sel])) if n else None
            pa = r2(accuracy(p20[sel])) if n else None
            rows.append({"class_id": c, "name": CLASS_NAMES[c], "n": n, "clean": ca, "pgd20": pa})
            if c in SAFETY_CLASSES:
                safety[str(c)] = {"name": CLASS_NAMES[c], "clean": ca, "pgd20": pa, "n": n}
        with open(os.path.join(self.rd, "per_class.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class_id", "name", "n", "clean", "pgd20"])
            w.writeheader(); w.writerows(rows)
        self.m["safety_classes"] = safety
        self.save()

    @torch.no_grad()
    def cost(self):
        if self.have("latency_ms_bs1_gpu") and self.have("latency_ms_bs1_cpu"):
            return
        warm, runs = (2, 5) if self.args.smoke else (20, 200)
        x1 = torch.rand(1, 3, 32, 32)
        x256 = torch.rand(256, 3, 32, 32)

        def bench(model, dev):
            m = model.to(dev).eval()
            a, b = x1.to(dev), x256.to(dev)
            sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
            for _ in range(warm):
                m(a)
            sync(); t = time.perf_counter()
            for _ in range(runs):
                m(a)
            sync(); lat = (time.perf_counter() - t) / runs * 1000
            nb = 2 if self.args.smoke else 20
            for _ in range(2):
                m(b)
            sync(); t = time.perf_counter()
            for _ in range(nb):
                m(b)
            sync(); thr = nb * 256 / (time.perf_counter() - t)
            return lat, thr

        if self.device.type == "cuda":
            lat, thr = bench(self.model, self.device)
            self.m["latency_ms_bs1_gpu"], self.m["throughput_bs256_gpu"] = r2(lat), r2(thr)
        else:
            self.m["latency_ms_bs1_gpu"], self.m["throughput_bs256_gpu"] = None, None
        lat, thr = bench(self.model, torch.device("cpu"))
        self.m["latency_ms_bs1_cpu"], self.m["throughput_bs256_cpu"] = r2(lat), r2(thr)
        self.model.to(self.device)
        self.save()

    def sanity(self):
        """§7.1 gradient-masking checks. Failures are stored and printed in red."""
        m = self.m
        s = {}
        if m.get("pgd100") is not None and m.get("pgd20") is not None:
            s["pgd100_le_pgd20"] = m["pgd100"] <= m["pgd20"] + 0.5
        if m.get("autoattack") is not None:
            pgd20_sub = accuracy(self.flags["pgd20"][self.sub_idx])
            s["aa_le_pgd20"] = m["autoattack"] <= pgd20_sub + 0.5
            sq = m["autoattack_per_attack"].get("square")
            s["square_not_stronger"] = sq is not None and sq >= m["autoattack"] - 0.5
        if m.get("eps_sweep"):
            vals = [m["eps_sweep"][k] for k in ["2", "4", "8", "12", "16"] if k in m["eps_sweep"]]
            s["eps_monotonic"] = all(a >= b for a, b in zip(vals, vals[1:]))
        m["sanity"] = s
        self.save()
        failed = [k for k, v in s.items() if not v]
        if failed:
            print("\033[91mSANITY CHECK FAILED: " + ", ".join(failed) + " - do not report as robust without "
                  "investigation (CLAUDE.md §7.1)\033[0m", flush=True)
        else:
            print("sanity checks passed:", s, flush=True)

    def run(self):
        self.white_box()
        self.eps_sweep()
        self.per_class()
        self.patch()
        self.corruptions()
        self.autoattack()
        self.cost()
        self.sanity()
        self.syncer.push_now()
        print(f"metrics written to {self.metrics_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute everything")
    ap.add_argument("--skip-aa", action="store_true")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--aa-bs", type=int, default=250)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    Evaluator(args).run()


if __name__ == "__main__":
    main()
