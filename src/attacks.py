"""White-box attacks: FGSM, PGD (CE / CW-margin), TRADES inner max, EAEG (DAT Alg. 1), patch attack.

Every attack runs in float32 with autocast disabled and with the model in eval mode (BN uses running
statistics; the previous train/eval state is restored afterwards). Inputs/outputs are [0,1] images; the L-inf
ball is enforced with min/max projection followed by a [0,1] clamp (CLAUDE.md §4.2, §4.5).
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .common import fp32_ctx

EPS_DEFAULT = 8 / 255     # [paper: DAT; Madry]
ALPHA_DEFAULT = 2 / 255   # [paper: DAT; Madry]


@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _project(x_adv, x, eps):
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    return torch.clamp(x_adv, 0.0, 1.0)


def cw_margin_loss(logits, y, kappa=0.0):
    """CW-inf margin objective (Carlini & Wagner f6), kappa = 0 [CLAUDE.md §7]. To be MAXIMISED by the attacker:
    loss = -relu(z_y - max_{j!=y} z_j + kappa)  (ascent lowers the true-class margin until it is negative)."""
    onehot = F.one_hot(y, logits.shape[1]).bool()
    z_y = logits[onehot]
    z_other = logits.masked_fill(onehot, float("-inf")).max(dim=1).values
    return -F.relu(z_y - z_other + kappa).mean()


def pgd(model, x, y, eps=EPS_DEFAULT, alpha=ALPHA_DEFAULT, steps=20, random_start=True, loss="ce", bn="A"):
    """PGD-k, L-inf [paper: Madry]. loss in {'ce', 'cw'}. Random start = uniform in the eps ball."""
    with eval_mode(model), fp32_ctx(x.device):
        x = x.detach().float()
        if random_start:
            x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
        else:
            x_adv = x.clone()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv, bn=bn)
            if loss == "ce":
                l = F.cross_entropy(logits, y)
            elif loss == "cw":
                l = cw_margin_loss(logits, y, kappa=0.0)
            else:
                raise ValueError(loss)
            grad = torch.autograd.grad(l, x_adv)[0]
            x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def fgsm(model, x, y, eps=EPS_DEFAULT, bn="A"):
    """FGSM = one signed-gradient step of size eps from x (no random start)."""
    return pgd(model, x, y, eps=eps, alpha=eps, steps=1, random_start=False, loss="ce", bn=bn)


def trades_inner(model, x, eps=EPS_DEFAULT, alpha=ALPHA_DEFAULT, steps=10, bn="A"):
    """TRADES inner maximisation [paper: TRADES]: start x + 0.001 N(0,I), maximise KL(f(x) || f(x')).
    KL written as in the TRADES code: kl_div(log_softmax(f(x')), softmax(f(x)), reduction='sum')."""
    with eval_mode(model), fp32_ctx(x.device):
        x = x.detach().float()
        with torch.no_grad():
            p_nat = F.softmax(model(x, bn=bn), dim=1)
        x_adv = x + 0.001 * torch.randn_like(x)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            l = F.kl_div(F.log_softmax(model(x_adv, bn=bn), dim=1), p_nat, reduction="sum")
            grad = torch.autograd.grad(l, x_adv)[0]
            x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def eaeg(model, x, y, eps=EPS_DEFAULT, alpha=ALPHA_DEFAULT, steps=5, beta=15.0, bn="A", label_smoothing=0.0):
    """EAEG, DAT Alg. 1 [paper: DAT] as implemented in the official code (`perturb_input`):

        v' = clip(v + 0.001 N(0, I))
        K times:  L_AE = CE(f(v'), y) + beta * KL(f(v) || f(v'))      (kl_div(log_softmax(f(v')), softmax(f(v)),
                                                                        reduction='batchmean'))
                  v'   = clip_[0,1]( Proj_eps( v' + alpha * sign(grad_{v'} L_AE) ) )
    K = 5, beta = 15 [paper: DAT]. f(v) carries no gradient to v' (equivalent to the code, computed once).
    The official code uses label_smoothing = 0.5 in this CE; the project follows it via configs/_defaults.yaml (NOTES.md N4)."""
    with eval_mode(model), fp32_ctx(x.device):
        x = x.detach().float()
        with torch.no_grad():
            p_nat = F.softmax(model(x, bn=bn), dim=1)
        x_adv = torch.clamp(x + 0.001 * torch.randn_like(x), 0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits_adv = model(x_adv, bn=bn)
            l = (F.cross_entropy(logits_adv, y, label_smoothing=label_smoothing)
                 + beta * F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction="batchmean"))
            grad = torch.autograd.grad(l, x_adv)[0]
            x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def patch_locations(indices, side, img=32, central=24):
    """Top-left corner of a side x side patch inside the central 24x24 region; location seed = image index."""
    lo = (img - central) // 2
    hi = lo + central - side
    tl = []
    for i in indices:
        rs = np.random.RandomState(int(i))
        tl.append((rs.randint(lo, hi + 1), rs.randint(lo, hi + 1)))
    return tl


def patch_attack(model, x, y, indices, side, steps=100, step=0.05, bn="A"):
    """Square patch attack [CLAUDE.md §7]: patch pixels are free in [0,1] (initialised at the image's own
    pixels), optimised with signed-gradient steps of 0.05 for 100 steps to maximise CE. Returns x_adv."""
    with eval_mode(model), fp32_ctx(x.device):
        x = x.detach().float()
        mask = torch.zeros_like(x[:, :1])
        for b, (r, c) in enumerate(patch_locations(indices, side)):
            mask[b, :, r:r + side, c:c + side] = 1.0
        patch = x.clone()
        for _ in range(steps):
            patch.requires_grad_(True)
            x_adv = x * (1 - mask) + patch * mask
            l = F.cross_entropy(model(x_adv, bn=bn), y)
            grad = torch.autograd.grad(l, patch)[0]
            patch = torch.clamp(patch.detach() + step * grad.sign(), 0.0, 1.0)
        x_adv = x * (1 - mask) + patch.detach() * mask
    return x_adv.detach()
