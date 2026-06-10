from __future__ import annotations
import timm


def build_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                pretrained: bool = True, drop_rate: float = 0.1):
    """Binary classifier: a single logit (attack probability via sigmoid)."""
    model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=1, drop_rate=drop_rate
    )
    return model
