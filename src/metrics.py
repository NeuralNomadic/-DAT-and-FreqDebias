"""Accuracy helpers shared by training-time validation and the final evaluation. Always float32, eval mode, BN-A."""
import torch

from .common import fp32_ctx


@torch.no_grad()
def predict(model, x, bs=512):
    out = []
    with fp32_ctx(x.device):
        for i in range(0, x.shape[0], bs):
            out.append(model(x[i:i + bs].float(), bn="A").argmax(1))
    return torch.cat(out)


def correct_flags(model, X, Y, attack=None, bs=256, device=None, desc=None):
    """Per-sample correctness (bool tensor on CPU). `attack(model, x, y) -> x_adv` is applied per batch.
    X, Y may live on CPU; batches are moved to `device`."""
    was = model.training
    model.eval()
    device = device or next(model.parameters()).device
    flags = []
    rng = range(0, X.shape[0], bs)
    if desc:
        try:
            from tqdm import tqdm
            rng = tqdm(rng, desc=desc, leave=False)
        except ImportError:
            pass
    for i in rng:
        x = X[i:i + bs].to(device, non_blocking=True).float()
        y = Y[i:i + bs].to(device, non_blocking=True)
        if attack is not None:
            x = attack(model, x, y)
        with torch.no_grad(), fp32_ctx(device):
            pred = model(x, bn="A").argmax(1)
        flags.append((pred == y).cpu())
    model.train(was)
    return torch.cat(flags)


def accuracy(flags) -> float:
    return 100.0 * flags.float().mean().item() if flags.numel() else float("nan")


def dataset_tensors(ds):
    """Stack an eval-mode GTSRB32 dataset into (X float [0,1], Y long) on CPU."""
    X = ds.imgs.float() / 255.0
    return X, ds.labels.clone()
