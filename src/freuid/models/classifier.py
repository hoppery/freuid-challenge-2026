from __future__ import annotations
import timm


def build_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                pretrained: bool = True, drop_rate: float = 0.1):
    """Binary classifier: a single logit (attack probability via sigmoid)."""
    model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=1, drop_rate=drop_rate
    )
    return model


def build_classifier(model_type: str = "rgb",
                     backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                     pretrained: bool = True, drop_rate: float = 0.1):
    """Dispatch by model_type. 'rgb' = single timm backbone (baseline);
    'freq_dual' = RGB + spectral dual-stream (see models/freq_classifier.py)."""
    if model_type == "freq_dual":
        from freuid.models.freq_classifier import build_freq_model
        return build_freq_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "hpf_dual":
        from freuid.models.freq_classifier import build_hpf_model
        return build_hpf_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "rgb":
        return build_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    raise ValueError(f"unknown model_type: {model_type!r}")
