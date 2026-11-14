"""E2 - TRADES [paper: TRADES] (CLAUDE.md §5.3).
Inner: K = 10 steps maximising KL(f(x') || f(x)) from x + 0.001 N(0, I).
Outer: L = CE(f(x), y) + 6 * KL(f(x') || f(x))   (kl_div(log_softmax(f(x')), softmax(f(x)), 'batchmean'),
as in the TRADES reference implementation)."""
import torch.nn.functional as F

from ..attacks import trades_inner
from .base import Method, kl_adv_vs_clean


class TRADES(Method):
    def train_step(self, batch, optimizer, scaler, amp):
        x, y = batch["x"], batch["y"]
        beta = float(self.cfg.get("trades_beta", 6.0))                     # [paper: TRADES] beta = 6
        x_adv = trades_inner(self.model, x, eps=self.eps, alpha=self.alpha,
                             steps=int(self.cfg.get("pgd_steps", 10)), bn="A")
        self.model.train()
        with self.autocast(amp):
            logits = self.model(x, bn="A")
            logits_adv = self.model(x_adv, bn="A")
            loss = F.cross_entropy(logits, y) + beta * kl_adv_vs_clean(logits_adv, logits, "mean")  # §5.3
        self.step(loss, optimizer, scaler)
        return {"loss": loss.item(), "correct": (logits.argmax(1) == y).sum().item(), "n": y.numel()}
