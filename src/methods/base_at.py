"""E3 - Base-AT (control), CLAUDE.md §5.4: L = L_AT(x) with x' = EAEG(x) (§4.5).
This is DAT's own ablation baseline [paper: DAT, Table 4] and the common base of both arms."""
from .base import Method, l_at_per_sample


class BaseAT(Method):
    def train_step(self, batch, optimizer, scaler, amp):
        x, y = batch["x"], batch["y"]
        x_adv = self.eaeg(x, y, bn="A")                               # §4.5 EAEG, model in eval mode inside
        self.model.train()
        with self.autocast(amp):
            logits = self.model(x, bn="A")
            logits_adv = self.model(x_adv, bn="A")
            loss = l_at_per_sample(logits, logits_adv, y, self.beta, self.label_smoothing).mean()  # §4.5 L_AT
        self.step(loss, optimizer, scaler)
        return {"loss": loss.item(), "correct": (logits.argmax(1) == y).sum().item(), "n": y.numel()}
