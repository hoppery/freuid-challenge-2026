from __future__ import annotations
import timm


def build_model(backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                pretrained: bool = True, drop_rate: float = 0.1, img_size: int | None = None):
    """Binary classifier: a single logit (attack probability via sigmoid).

    img_size is passed to timm for size-strict backbones (e.g. DINOv2 ViTs default to
    518 and assert on input size); ConvNeXt etc. don't accept it, so we fall back.
    """
    kw = dict(pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
    if img_size is not None:
        try:
            return timm.create_model(backbone, img_size=img_size, **kw)
        except TypeError:
            pass
    return timm.create_model(backbone, **kw)


def build_classifier(model_type: str = "rgb",
                     backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                     pretrained: bool = True, drop_rate: float = 0.1,
                     img_size: int | None = None):
    """Dispatch by model_type. 'rgb' = single timm backbone (baseline);
    'freq_dual'/'hpf_dual' = dual-stream (see models/freq_classifier.py)."""
    if model_type == "freq_dual":
        from freuid.models.freq_classifier import build_freq_model
        return build_freq_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "hpf_dual":
        from freuid.models.freq_classifier import build_hpf_model
        return build_hpf_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "rgb":
        return build_model(backbone, pretrained=pretrained, drop_rate=drop_rate, img_size=img_size)
    raise ValueError(f"unknown model_type: {model_type!r}")
