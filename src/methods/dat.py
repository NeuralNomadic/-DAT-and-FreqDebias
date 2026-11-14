"""Arm B - DAT (E6) and ablations A5 (single BN), A6 (no generator) [paper: DAT, Alg. 2; official code
train_cifar.py::adv_train / one_step]. CLAUDE.md §5.7. Every paper/code difference is listed in NOTES.md.

Per batch, following the official training loop:
  1. x' = EAEG(x) with BN-A.                                                  (perturb_input, 'base')
  2. Stage I:  A, P = fft(x_orig) on the UN-augmented image (code: img_o); generator conditioned on the
     classifier's logits for x_orig (eval mode, BN-A, detached); z ~ N(0, I) (100 dims);
     A_mix = lam * A_G + (1 - lam) * A, lam ~ U(0, 1) per image; x_hat = clip(real(ifft(A_mix, P)));
     then a random crop (the code applies crop + flip after mixing; no flips here, §3.2).
  3. x_hat' = EAEG(x_hat) with BN-B.                                          (perturb_input, 'auto')
  4. Stage III: L_DAT = 0.5 (L_AT(x) + L_AT(x_hat)) + 2 JS(f(x), f(x_hat)), omega = 2 [paper: DAT].
     Classifier: SGD step decreasing L_DAT (x_hat detached).
     Generator: fresh forward with x_hat attached, g_loss = -L_DAT, its own SGD step
     (lr 0.1, momentum 0.9, wd 5e-4 [paper: DAT]). The classifier's parameters receive no update from this
     pass (its grads are zeroed before the next classifier step, as in the code); they are frozen here only
     to save compute.
"""
import torch
import torch.nn.functional as F

from ..aag import AmplitudeGenerator
from ..fourier import fft2_dat, ifft2_dat
from ..common import fp32_ctx
from .base import Method, js_div, l_at_per_sample


def batch_random_crop(x, pad=3):
    """Per-image random crop after reflect padding (project augmentation §3.2, no flip). Differentiable in x."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (oy.view(B, 1) + torch.arange(H, device=x.device).view(1, H))      # B x H
    cols = (ox.view(B, 1) + torch.arange(W, device=x.device).view(1, W))      # B x W
    bidx = torch.arange(B, device=x.device).view(B, 1, 1)
    out = xp.permute(0, 2, 3, 1)[bidx, rows.view(B, H, 1), cols.view(B, 1, W)]   # B x H x W x C
    return out.permute(0, 3, 1, 2).contiguous()


class DAT(Method):
    log_keys = ["l_at_x", "l_at_xhat", "l_js", "gen_loss"]

    def __init__(self, cfg, model, device, extra):
        super().__init__(cfg, model, device, extra)
        self.dual_bn = bool(cfg.get("dual_bn", True))
        self.bn_b = "B" if self.dual_bn else "A"                         # A5: single BN for both branches
        self.omega = float(cfg.get("js_weight", 2.0))                    # [paper: DAT] omega = 2
        self.generator_mode = cfg.get("generator", "learned")            # 'learned' (E6) | 'partner' (A6)
        self.post_crop = bool(cfg.get("dat_post_crop", True))
        self.G = None
        self.opt_g = None
        if self.generator_mode == "learned":
            self.G = AmplitudeGenerator(num_classes=cfg.get("num_classes", 43)).to(device)
            self.opt_g = torch.optim.SGD(self.G.parameters(), lr=float(cfg.get("gen_lr", 0.1)),
                                         momentum=float(cfg.get("momentum", 0.9)),
                                         weight_decay=float(cfg.get("weight_decay", 5e-4)))   # [paper: DAT]

    needs_partner = property(lambda self: self.generator_mode == "partner")

    # -- checkpointing of the generator and its optimizer -----------------------------------------
    def state_dict_extra(self):
        if self.G is None:
            return {}
        return {"generator": self.G.state_dict(), "opt_g": self.opt_g.state_dict()}

    def load_state_dict_extra(self, state):
        if self.G is not None and "generator" in state:
            self.G.load_state_dict(state["generator"])
            self.opt_g.load_state_dict(state["opt_g"])

    def extra_modules(self):
        return [self.G] if self.G is not None else []

    # -- Stage I ------------------------------------------------------------------------------------
    def make_x_hat(self, x_orig, x_partner_orig=None):
        """Returns x_hat (graph attached to the generator when learned)."""
        B = x_orig.shape[0]
        with fp32_ctx(x_orig.device):
            A, P = fft2_dat(x_orig)                                         # shifted amplitude, raw phase
            if self.generator_mode == "learned":
                with torch.no_grad():
                    was = self.model.training
                    self.model.eval()
                    logits_o = self.model(x_orig, bn="A").float()           # code: latent_na = model(img_o)
                    self.model.train(was)
                self.G.train()
                A_G = self.G(self.G.sample_z(B, x_orig.device), logits_o.detach()).float()
            else:                                                           # A6: same-class image amplitude
                A_G, _ = fft2_dat(x_partner_orig)
            lam = torch.rand(B, 1, 1, 1, device=x_orig.device)              # code: beta1 ~ U(0, mix_beta=1)
            A_mix = lam * A_G + (1.0 - lam) * A                              # Alg. 2 Stage I mix
            x_hat = ifft2_dat(A_mix, P)                                     # real part, clipped (code)
            if self.post_crop:
                x_hat = batch_random_crop(x_hat, pad=3)
        return x_hat

    # -- Stage III loss -----------------------------------------------------------------------------
    def dat_loss(self, x, x_adv, x_hat, x_hat_adv, y):
        logits_x = self.model(x, bn="A")
        logits_x_adv = self.model(x_adv, bn="A")
        logits_h = self.model(x_hat, bn=self.bn_b)
        logits_h_adv = self.model(x_hat_adv, bn=self.bn_b)
        l_at_x = l_at_per_sample(logits_x, logits_x_adv, y, self.beta, self.label_smoothing).mean()
        l_at_h = l_at_per_sample(logits_h, logits_h_adv, y, self.beta, self.label_smoothing).mean()
        l_js = js_div(logits_x, logits_h, "mean")
        loss = 0.5 * (l_at_x + l_at_h) + self.omega * l_js                  # Alg. 2 Stage III, omega = 2
        return loss, logits_x, (l_at_x, l_at_h, l_js)

    def train_step(self, batch, optimizer, scaler, amp):
        x, y, x_orig = batch["x"], batch["y"], batch["x_orig"]
        x_adv = self.eaeg(x, y, bn="A")                                                     # 1.
        x_hat = self.make_x_hat(x_orig, batch.get("x_partner_orig"))                        # 2.
        x_hat_adv = self.eaeg(x_hat.detach(), y, bn=self.bn_b)                              # 3.

        self.model.train()
        with self.autocast(amp):                                                            # 4. classifier
            loss, logits_x, (l_at_x, l_at_h, l_js) = self.dat_loss(x, x_adv, x_hat.detach(), x_hat_adv, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        gen_loss = float("nan")
        if self.G is not None:                                                              # 4. generator
            for p in self.model.parameters():
                p.requires_grad_(False)
            with self.autocast(amp):
                g_loss = -self.dat_loss(x, x_adv, x_hat, x_hat_adv, y)[0]                 # maximise L_DAT
            self.opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(self.opt_g)
            for p in self.model.parameters():
                p.requires_grad_(True)
            gen_loss = g_loss.item()
        scaler.update()
        return {"loss": loss.item(), "correct": (logits_x.argmax(1) == y).sum().item(), "n": y.numel(),
                "l_at_x": l_at_x.item(), "l_at_xhat": l_at_h.item(), "l_js": l_js.item(), "gen_loss": gen_loss}
