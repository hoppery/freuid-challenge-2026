"""Doc-Type-Classifier (DTC) head with gradient reversal — domain-invariant training.

Iteration 4 contribution. 5-fold LODO showed DINOv2 partial-unfreeze has huge fold
variance (FREUID 0.003 on MOZAMBIQUE but 0.975 on MAURITIUS/ID): the features stay
document-type-specific, so generalization to a held-out type is unreliable. A
gradient-reversal doc-type classifier forces the backbone to produce features that are
predictive of fraud but NOT of document type → domain(doc-type)-invariant fraud
features that transfer to unseen types (the private-test objective). The parallel
pipeline corroborates: DTC × dinov2-unfreeze2 reached 5-fold avg ~0.11 vs ~0.38 without.

Inference forward(x) returns ONLY the fraud logit (so eval/infer/evaluate are unchanged);
training uses forward_dtc(x, alpha) returning (fraud_logit, doctype_logits).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha: float):
    return _GradReverse.apply(x, alpha)


class DTCClassifier(nn.Module):
    def __init__(self, backbone: str, n_doctypes: int, pretrained: bool = True,
                 drop_rate: float = 0.1, img_size: int | None = None, hidden: int = 256):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        if img_size is not None:
            try:
                self.backbone = timm.create_model(backbone, img_size=img_size, **kw)
            except TypeError:
                self.backbone = timm.create_model(backbone, **kw)
        else:
            self.backbone = timm.create_model(backbone, **kw)
        d = self.backbone.num_features
        self.fraud = nn.Linear(d, 1)
        self.doctype = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop_rate), nn.Linear(hidden, n_doctypes))

    def forward(self, x):
        """Inference: fraud logit only (keeps eval/infer code path unchanged)."""
        return self.fraud(self.backbone(x))

    def forward_dtc(self, x, alpha: float):
        feat = self.backbone(x)
        fraud = self.fraud(feat)
        dt = self.doctype(grad_reverse(feat, alpha))
        return fraud, dt


def build_dtc_model(backbone: str = "vit_base_patch14_reg4_dinov2.lvd142m",
                    n_doctypes: int = 4, pretrained: bool = True,
                    drop_rate: float = 0.1, img_size: int | None = None):
    return DTCClassifier(backbone, n_doctypes, pretrained=pretrained,
                         drop_rate=drop_rate, img_size=img_size)
