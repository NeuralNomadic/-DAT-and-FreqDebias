"""Arm A - FreqDebias-AT (E5), FreqDebias-Clean (E4) and ablations A1-A4 (CLAUDE.md §5.5-5.6, §5.8).

Per batch (x, y) with same-class partner x_j and the precomputed Fo-Mixup masks:
  1. x_t = Fo-Mixup(x, x_j, B)                                                     (§6.4)
  2. Confidence sampling [paper: FreqDebias, lambda = 0.5]: keep the 50% lowest-entropy x_t (BN-B, no grad)
  3. L_FD = 0.5 (L_AT(x) + L_AT(x_t))
            + 1.0 KL( softmax(f(x_t)/tau) || softmax(f(x)/tau) )      tau = 4   [paper: FreqDebias Eq. 3]
            + delta L_att(x, x_t)                                      delta = 0.1 [paper: FreqDebias Eq. 13]
     x, x' use BN-A; x_t, x_t' use BN-B [adaptation: matches DAT].
     L_att = JS( softmax(CAM(x)/tau), softmax(CAM(x_t)/tau) ), CAM = W_y^T F(v) flattened to 16 values.
  E4: every L_AT(v) is replaced by CE(f(v), y) and no attacks are generated.
Config switches: adversarial (E4 vs E5), band_mode ('ohem' | 'all' = A1 | 'uniform' = A2), delta (A3: 0),
confidence_sampling (A4: false), color_space ('rgb' = E5 | 'ycbcr' = E5-CA, CLAUDE.md §14) with chroma_protect
(alpha in [0, 1], used only with 'ycbcr').
"""
import torch
import torch.nn.functional as F

from ..fomixup import choose_masks, fomixup, fomixup_color
from .base import Method, _kl_from_logits, js_div, l_at_per_sample


class FreqDebias(Method):
    needs_partner = True
    needs_masks = True
    log_keys = ["l_at_x", "l_at_xtilde", "l_consistency", "l_att"]

    def __init__(self, cfg, model, device, extra):
        super().__init__(cfg, model, device, extra)
        self.adversarial = bool(cfg.get("adversarial", True))
        self.band_mode = cfg.get("band_mode", "ohem")
        self.delta = float(cfg.get("delta", 0.1))                       # [paper: FreqDebias Eq. 13]
        self.tau = float(cfg.get("tau", 4.0))                           # [paper: FreqDebias Eq. 3]
        self.w_cons = float(cfg.get("consistency_weight", 1.0))
        self.conf_sampling = bool(cfg.get("confidence_sampling", True))
        self.keep_frac = float(cfg.get("confidence_keep", 0.5))         # [paper: FreqDebias lambda = 0.5]
        self.p_std = float(cfg.get("p_std", 0.1))                       # [adaptation] Eq. 2 noise
        self.color_space = cfg.get("color_space", "rgb")                 # §14: 'rgb' (E5) | 'ycbcr' (E5-CA)
        if self.color_space not in ("rgb", "ycbcr"):
            raise ValueError(f"color_space must be 'rgb' or 'ycbcr', got {self.color_space!r}")
        self.chroma_protect = float(cfg.get("chroma_protect", 1.0))      # §14 alpha (ycbcr only)
        masks = extra.get("fomixup_masks")
        if masks is None:
            raise RuntimeError("Fo-Mixup masks are required (run src.precompute_fomixup after E0)")
        self.seg_cluster = masks["seg_cluster"].to(device)              # N_train x 144 uint8
        self.top_clusters = masks["top_clusters"].to(device)            # N_train x 3

    def _branch_loss(self, logits, logits_adv, y):
        """L_AT(v) in E5; CE(f(v), y) in E4 (§5.5). Per sample."""
        if self.adversarial:
            return l_at_per_sample(logits, logits_adv, y, self.beta, self.label_smoothing)
        return F.cross_entropy(logits, y, label_smoothing=self.label_smoothing, reduction="none")

    @torch.no_grad()
    def _confidence_keep(self, x_t):
        """§5.6 step 2: entropy of f(x_t) with BN-B (no grad, eval mode so BN-B statistics are untouched);
        keep the lowest-entropy half. Returns the kept indices (sorted)."""
        n = x_t.shape[0]
        if not self.conf_sampling:
            return torch.arange(n, device=x_t.device)
        was = self.model.training
        self.model.eval()
        p = F.softmax(self.model(x_t, bn="B").float(), dim=1)
        self.model.train(was)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
        k = max(1, int(round(self.keep_frac * n)))
        return torch.sort(torch.topk(-ent, k).indices).values

    def train_step(self, batch, optimizer, scaler, amp):
        x, y, idx, x_j = batch["x"], batch["y"], batch["idx"], batch["x_partner"]
        # 1. Fo-Mixup (float32 inside)
        mask = choose_masks(self.seg_cluster[idx], self.top_clusters[idx], self.band_mode)
        if self.color_space == "ycbcr":                                   # §14 colour-aware (E5-CA)
            x_t = fomixup_color(x, x_j, mask, p_std=self.p_std, chroma_protect=self.chroma_protect).detach()
        else:
            x_t = fomixup(x, x_j, mask, p_std=self.p_std).detach()
        # 2. confidence sampling
        keep = self._confidence_keep(x_t)
        x_tk, y_k = x_t[keep], y[keep]
        # attacks (model in eval mode inside eaeg)
        x_adv = self.eaeg(x, y, bn="A") if self.adversarial else None
        x_tk_adv = self.eaeg(x_tk, y_k, bn="B") if self.adversarial else None
        # 3. losses
        self.model.train()
        with self.autocast(amp):
            logits_x, feat_x = self.model(x, bn="A", return_feat=True)
            logits_x_adv = self.model(x_adv, bn="A") if self.adversarial else None
            l_at_x = self._branch_loss(logits_x, logits_x_adv, y).mean()

            logits_t, feat_t = self.model(x_tk, bn="B", return_feat=True)
            logits_t_adv = self.model(x_tk_adv, bn="B") if self.adversarial else None
            l_at_t = self._branch_loss(logits_t, logits_t_adv, y_k).mean()        # renormalised by #kept

            # consistency, Eq. 3: KL( softmax(f(x_t)/tau) || softmax(f(x)/tau) ) over kept samples
            log_p_x = F.log_softmax(logits_x[keep].float() / self.tau, dim=1)
            l_cons = _kl_from_logits(logits_t.float() / self.tau, log_p_x).mean()

            # CAM consistency, adaptation of Eq. 4-6: JS between softmax(CAM/tau) of x and x_t
            if self.delta > 0:
                cam_x = self.model.cam(feat_x[keep], y_k) / self.tau
                cam_t = self.model.cam(feat_t, y_k) / self.tau
                l_att = js_div(cam_x, cam_t, "mean")
            else:
                l_att = torch.zeros((), device=x.device)

            loss = 0.5 * (l_at_x + l_at_t) + self.w_cons * l_cons + self.delta * l_att     # §5.6 step 3
        self.step(loss, optimizer, scaler)
        return {"loss": loss.item(), "correct": (logits_x.argmax(1) == y).sum().item(), "n": y.numel(),
                "l_at_x": l_at_x.item(), "l_at_xtilde": l_at_t.item(), "l_consistency": l_cons.item(),
                "l_att": l_att.detach().item()}
