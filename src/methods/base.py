"""Common pieces for all training methods: the shared adversarial-training base L_AT (CLAUDE.md §4.5),
KL / JS helpers written exactly as in the DAT official code, and the Method interface."""
import torch
import torch.nn.functional as F

from ..attacks import eaeg


def _kl_from_logits(logits_p, log_q):
    """KL( softmax(logits_p) || exp(log_q) ) per sample = sum_c p_c (log p_c - log q_c), with log p from
    log_softmax. Identical to F.kl_div(log_q, softmax(logits_p)) in value, but its gradient stays finite when a
    probability underflows to exactly 0 (F.kl_div's xlogy backward gives 0/0 = NaN there; see NOTES.md N17)."""
    log_p = F.log_softmax(logits_p.float(), dim=1)
    return (log_p.exp() * (log_p - log_q.float())).sum(1)


def kl_adv_vs_clean(logits_adv, logits_clean, reduction="none"):
    """beta-term of L_AT, DAT official code: F.kl_div(log_softmax(f(v')), softmax(f(v)), 'batchmean').
    Mathematically KL( f(v) || f(v') ). Neither side is detached (as in the code). reduction='none' returns
    one value per sample (sum over classes) so callers can mask / renormalise."""
    kl = _kl_from_logits(logits_clean, F.log_softmax(logits_adv.float(), dim=1))
    return kl if reduction == "none" else kl.mean()


def js_div(logits_a, logits_b, reduction="mean"):
    """JS divergence as in the DAT official code (`one_step`), written on logits:
        p = softmax(a), q = softmax(b), m = clamp((p + q) / 2, 1e-7, 1).log();  JS = ( KL(p || m) + KL(q || m) ) / 2
    reduction='none' -> per-sample."""
    p, q = F.softmax(logits_a.float(), dim=1), F.softmax(logits_b.float(), dim=1)
    m = torch.clamp((p + q) / 2.0, 1e-7, 1.0).log()
    per = 0.5 * (_kl_from_logits(logits_a, m) + _kl_from_logits(logits_b, m))
    return per if reduction == "none" else per.mean()


def l_at_per_sample(logits_clean, logits_adv, y, beta, label_smoothing=0.0):
    """L_AT(v) = CE(f(v), y) + beta * KL(f(v) || f(v'))   [paper: DAT, Eq. 6 definition of L_AT], per sample."""
    ce = F.cross_entropy(logits_clean, y, label_smoothing=label_smoothing, reduction="none")
    return ce + beta * kl_adv_vs_clean(logits_adv, logits_clean, reduction="none")


class Method:
    """One optimizer update per `train_step`. Methods own their attack generation and backward pass."""
    needs_partner = False       # dataset must supply a same-class partner image
    needs_masks = False         # dataset indices must map to precomputed Fo-Mixup masks
    log_keys = []               # extra train_log.csv columns

    def __init__(self, cfg, model, device, extra):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.eps = cfg["eps"] / 255.0
        self.alpha = cfg["alpha"] / 255.0
        self.beta = float(cfg.get("beta", 15.0))                 # [paper: DAT]
        self.eaeg_steps = int(cfg.get("eaeg_steps", 5))          # [paper: DAT] K = 5
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))
        self.amp_dtype = torch.float16

    # -- optional state for extra modules (DAT generator) -----------------------------------------
    def state_dict_extra(self):
        return {}

    def load_state_dict_extra(self, state):
        pass

    def extra_modules(self):
        return []

    # -- helpers -----------------------------------------------------------------------------------
    def autocast(self, enabled):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                              enabled=enabled and self.device.type == "cuda")

    def eaeg(self, x, y, bn):
        return eaeg(self.model, x, y, eps=self.eps, alpha=self.alpha, steps=self.eaeg_steps,
                    beta=self.beta, bn=bn, label_smoothing=self.label_smoothing)

    def step(self, loss, optimizer, scaler):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def train_step(self, batch, optimizer, scaler, amp):
        raise NotImplementedError
