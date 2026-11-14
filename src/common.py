"""Shared helpers: seeding, config loading, atomic checkpoints, environment record."""
import json
import os
import platform
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

EXP_NAMES = ["E0_standard", "E1_pgdat", "E2_trades", "E3_base", "E4_fd_clean",
             "E5_fd_at", "E6_dat", "A1", "A2", "A3", "A4", "A5", "A6"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS_PATH = os.path.join(REPO_ROOT, "configs", "_defaults.yaml")


def set_seed(seed: int):
    """Seed Python, NumPy and torch (CPU + CUDA). CLAUDE.md §4.6."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def load_config(path: str) -> dict:
    """Experiment config = shared defaults (configs/_defaults.yaml) overridden by the experiment file.
    The shared file holds every setting that must be identical across methods (CLAUDE.md §4, §13)."""
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    with open(path) as f:
        exp = yaml.safe_load(f) or {}
    cfg.update(exp)
    cfg["config_path"] = os.path.abspath(path)
    return cfg


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def atomic_save(obj, path: str):
    """torch.save to a temp file in the same directory, then os.replace (CLAUDE.md §10.3)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(text: str, path: str):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info(device=None) -> dict:
    import torchvision
    gpu = "cpu"
    if device is not None and device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    elif torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "gpu": gpu,
    }


def run_dir(out: str, exp: str, seed: int) -> str:
    return os.path.join(out, "runs", exp, f"seed{seed}")


def rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    atomic_write_text(json.dumps(obj, indent=2), path)


def fp32_ctx(device):
    """Context that disables autocast: all FFT / amplitude / phase arithmetic and all attacks run in float32
    (CLAUDE.md §4.3)."""
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return torch.autocast(device_type=dev_type, enabled=False)
