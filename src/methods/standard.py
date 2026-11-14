"""E0 - Standard training: L = CE(f(x), y)  (CLAUDE.md §5.1)."""
import torch.nn.functional as F

from .base import Method


class Standard(Method):
    def train_step(self, batch, optimizer, scaler, amp):
        x, y = batch["x"], batch["y"]
        self.model.train()
        with self.autocast(amp):
            logits = self.model(x, bn="A")
            loss = F.cross_entropy(logits, y)                       # §5.1
        self.step(loss, optimizer, scaler)
        return {"loss": loss.item(), "correct": (logits.argmax(1) == y).sum().item(), "n": y.numel()}
