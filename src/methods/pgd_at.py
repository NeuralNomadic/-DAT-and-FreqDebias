"""E1 - PGD-AT [paper: Madry] (CLAUDE.md §5.2).
Inner: PGD, K = 10, alpha = 2/255, uniform random start in the eps-ball, maximise CE.
Outer: L = CE(f(x'), y).
The logged train accuracy is measured on x' (the only forward pass this method makes; NOTES.md)."""
import torch.nn.functional as F

from ..attacks import pgd
from .base import Method


class PGDAT(Method):
    def train_step(self, batch, optimizer, scaler, amp):
        x, y = batch["x"], batch["y"]
        x_adv = pgd(self.model, x, y, eps=self.eps, alpha=self.alpha, steps=int(self.cfg.get("pgd_steps", 10)),
                    random_start=True, loss="ce", bn="A")
        self.model.train()
        with self.autocast(amp):
            logits = self.model(x_adv, bn="A")
            loss = F.cross_entropy(logits, y)                       # §5.2
        self.step(loss, optimizer, scaler)
        return {"loss": loss.item(), "correct": (logits.argmax(1) == y).sum().item(), "n": y.numel()}
