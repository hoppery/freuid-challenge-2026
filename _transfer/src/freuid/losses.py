"""Operating-point-aligned training losses.

Motivation (Iteration 1 of the contribution loop): the official FREUID Score is
1 − harmonic-mean(1−AuDET, 1−APCER@1%BPCER), and our baseline's bottleneck is
APCER@1%BPCER (~0.50 dominates the score). Plain BCE weights every operating
point equally, so the model never explicitly optimizes the decision region the
metric actually measures: the bona-fide upper tail (the 99th-percentile threshold)
and the attack scores just below it.

TailMarginBCE = BCE (global ranking / AuDET shaping)
              + lambda_tail * hinge pushing ATTACK logits ABOVE the running
                bona-fide 99% quantile tau (+ margin)
              + mu_bona * compression of the bona-fide RIGHT tail above tau.

tau is estimated from a detached memory bank of recent bona-fide logits (EMA-free,
quantile of a sliding window), so it tracks the score distribution as it shifts
during training without extra forward passes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TailMarginBCE(nn.Module):
    """v2 lesson (Iteration 1 failure): with `space="logit"` the hinge target tau chases
    the rising logit scale (attacks pushed above tau -> shared features lift bona logits
    -> tau rises -> repeat) and the loss DIVERGES (observed 101->767). Default is now
    `space="prob"`: tail terms operate on sigmoid(score), every term bounded in [0,1]
    and gradients saturate at extreme logits — no scale arms race. `warmup_steps`
    delays the tail terms until the bank reflects a settled score distribution."""

    def __init__(self, quantile: float = 0.99, margin: float = 0.1,
                 lambda_tail: float = 1.0, mu_bona: float = 0.2,
                 bank_size: int = 8192, min_bank: int = 256,
                 space: str = "prob", warmup_steps: int = 200):
        super().__init__()
        if space not in ("prob", "logit"):
            raise ValueError(f"space must be 'prob' or 'logit', got {space!r}")
        self.quantile = quantile
        self.margin = margin
        self.lambda_tail = lambda_tail
        self.mu_bona = mu_bona
        self.bank_size = bank_size
        self.min_bank = min_bank
        self.space = space
        self.warmup_steps = warmup_steps
        # sliding bank of recent bona-fide scores (detached, fp32; prob- or logit-space)
        self.register_buffer("bank", torch.full((bank_size,), float("nan")))
        self.register_buffer("bank_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("bank_filled", torch.zeros((), dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update_bank(self, bona_logits: torch.Tensor):
        v = bona_logits.detach().float().flatten()
        if v.numel() == 0:
            return
        n = v.numel()
        ptr = int(self.bank_ptr)
        idx = (torch.arange(n, device=v.device) + ptr) % self.bank_size
        self.bank[idx] = v
        self.bank_ptr.fill_((ptr + n) % self.bank_size)
        self.bank_filled.fill_(min(int(self.bank_filled) + n, self.bank_size))

    def _tau(self) -> torch.Tensor | None:
        filled = int(self.bank_filled)
        if filled < self.min_bank:
            return None
        return torch.quantile(self.bank[:filled], self.quantile)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits/targets: (B,1) or (B,). targets in {0,1}; 1=attack."""
        logits = logits.flatten().float()
        targets = targets.flatten().float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        scores = torch.sigmoid(logits) if self.space == "prob" else logits
        is_attack = targets > 0.5
        bona = scores[~is_attack]
        attack = scores[is_attack]
        self._update_bank(bona)
        if self.training:
            self.step += 1
        tau = self._tau()
        if tau is not None and int(self.step) >= self.warmup_steps:
            if attack.numel() > 0 and self.lambda_tail > 0:
                # attacks must clear the bona-fide 99% quantile by `margin`
                loss = loss + self.lambda_tail * F.relu(tau + self.margin - attack).mean()
            if bona.numel() > 0 and self.mu_bona > 0:
                # compress the bona-fide right tail (scores above tau pull down)
                loss = loss + self.mu_bona * F.relu(bona - tau).mean()
        return loss


def build_loss(name: str = "bce", **kw) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "tail_margin":
        return TailMarginBCE(**kw)
    raise ValueError(f"unknown loss: {name!r}")
