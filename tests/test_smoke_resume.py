"""End-to-end --smoke runs for every config, resume equivalence, evaluate/aggregate smoke (CLAUDE.md §11)."""
import glob
import os

import pytest
import torch

from src import train, evaluate, aggregate, precompute_fomixup, make_grids, corruptions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "[EA]*.yaml")))


@pytest.mark.parametrize("config", CONFIGS, ids=[os.path.basename(c) for c in CONFIGS])
def test_smoke_every_config(config, tmp_out, tmp_data):
    rd = train.main(["--config", config, "--seed", "0", "--data", tmp_data, "--out", tmp_out, "--smoke"])
    assert os.path.exists(os.path.join(rd, "checkpoints", "last.pt"))
    assert os.path.exists(os.path.join(rd, "checkpoints", "best.pt"))
    assert os.path.exists(os.path.join(rd, "train_log.csv"))
    log = open(os.path.join(rd, "train_log.csv")).read().splitlines()
    assert len(log) == 2  # header + 1 epoch
    header = log[0].split(",")
    for col in ["epoch", "lr", "train_loss", "train_clean_acc", "val_clean_acc", "val_pgd20_acc", "sec_per_epoch", "is_best"]:
        assert col in header
    if "E5" in config or "E4" in config or os.path.basename(config).startswith("A") and int(os.path.basename(config)[1]) <= 4:
        assert {"l_at_x", "l_at_xtilde", "l_consistency", "l_att"} <= set(header)
    if "E6" in config or os.path.basename(config) in ("A5.yaml", "A6.yaml"):
        assert {"l_at_x", "l_at_xhat", "l_js", "gen_loss"} <= set(header)


def test_resume_equals_straight_run(tmp_out, tmp_data):
    cfg = os.path.join(ROOT, "configs", "E6_dat.yaml")   # the method with the most state (generator + optimizer)
    common = ["--config", cfg, "--seed", "3", "--data", tmp_data, "--smoke", "--deterministic", "--max-batches", "2"]
    out_a = os.path.join(tmp_out, "straight")
    out_b = os.path.join(tmp_out, "resumed")
    rd_a = train.main(common + ["--out", out_a, "--epochs", "3"])
    train.main(common + ["--out", out_b, "--epochs", "2"])
    rd_b = train.main(common + ["--out", out_b, "--epochs", "3", "--resume"])
    a = torch.load(os.path.join(rd_a, "checkpoints", "last.pt"), weights_only=False)
    b = torch.load(os.path.join(rd_b, "checkpoints", "last.pt"), weights_only=False)
    assert a["epoch"] == b["epoch"] == 3
    for k in a["model"]:
        assert torch.allclose(a["model"][k].float(), b["model"][k].float(), atol=1e-6, rtol=1e-5), k
    ga, gb = a["method_extra"]["generator"], b["method_extra"]["generator"]
    for k in ga:
        assert torch.allclose(ga[k].float(), gb[k].float(), atol=1e-6, rtol=1e-5), k
    la = open(os.path.join(rd_a, "train_log.csv")).read().splitlines()
    lb = open(os.path.join(rd_b, "train_log.csv")).read().splitlines()
    assert len(la) == len(lb) == 4


def test_evaluate_precompute_grids_aggregate_smoke(tmp_out, tmp_data):
    cfg = os.path.join(ROOT, "configs", "E0_standard.yaml")
    train.main(["--config", cfg, "--seed", "0", "--data", tmp_data, "--out", tmp_out, "--smoke"])
    evaluate.main(["--exp", "E0_standard", "--seed", "0", "--data", tmp_data, "--out", tmp_out, "--smoke"])
    mp = os.path.join(tmp_out, "runs", "E0_standard", "seed0", "metrics.json")
    assert os.path.exists(mp)
    import json
    m = json.load(open(mp))
    for k in ["clean", "fgsm", "pgd20", "pgd100", "cw_inf", "eps_sweep", "corruptions", "corruption_mean", "patch",
              "safety_classes", "latency_ms_bs1_cpu", "sanity", "torch", "python"]:
        assert k in m, k
    assert len(m["corruptions"]) == 6
    # resumable: a second call must not fail and keeps the values
    evaluate.main(["--exp", "E0_standard", "--seed", "0", "--data", tmp_data, "--out", tmp_out, "--smoke"])
    assert json.load(open(mp))["clean"] == m["clean"]
    precompute_fomixup.main(["--data", tmp_data, "--out", tmp_out, "--smoke"])
    assert os.path.exists(os.path.join(tmp_out, "data_cache", "fomixup_masks.pt"))
    assert os.path.exists(os.path.join(tmp_out, "figures", "fomixup_grid.png"))
    make_grids.main(["--data", tmp_data, "--out", tmp_out, "--which", "dat", "--smoke"])
    assert os.path.exists(os.path.join(tmp_out, "figures", "dat_grid.png"))
    aggregate.main(["--out", tmp_out])
    assert os.path.exists(os.path.join(tmp_out, "results", "table_main.md"))
    corruptions.make_figures(os.path.join(tmp_out, ".smoke"))   # smoke corruption cache lives apart
    assert not os.path.exists(os.path.join(tmp_out, "data_cache", "gtsrb_c")), "smoke must not touch the real cache"
    tmp_out = os.path.join(tmp_out, ".smoke")
    assert os.path.exists(os.path.join(tmp_out, "figures", "corruptions", "fog.png"))
