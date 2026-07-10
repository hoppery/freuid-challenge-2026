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


def enable_grad_checkpointing(model) -> bool:
    """Turn on activation checkpointing (timm API) to fit hi-res ViTs in VRAM.
    Works on plain timm models and wrappers exposing `.backbone`. Returns True if enabled."""
    target = model
    if not hasattr(target, "set_grad_checkpointing") and hasattr(model, "backbone"):
        target = model.backbone
    if hasattr(target, "set_grad_checkpointing"):
        try:
            target.set_grad_checkpointing(enable=True)
            return True
        except Exception:
            return False
    return False


def build_classifier(model_type: str = "rgb",
                     backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                     pretrained: bool = True, drop_rate: float = 0.1,
                     img_size: int | None = None, n_doctypes: int = 0):
    """Dispatch by model_type. 'rgb' = single timm backbone (baseline);
    'freq_dual'/'hpf_dual' = dual-stream; 'dtc' = doc-type-adversarial (domain-invariant)."""
    if model_type == "freq_dual":
        from freuid.models.freq_classifier import build_freq_model
        return build_freq_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "hpf_dual":
        from freuid.models.freq_classifier import build_hpf_model
        return build_hpf_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "dtc":
        from freuid.models.dtc import build_dtc_model
        return build_dtc_model(backbone, n_doctypes=max(1, n_doctypes), pretrained=pretrained,
                               drop_rate=drop_rate, img_size=img_size)
    if model_type == "rgb":
        return build_model(backbone, pretrained=pretrained, drop_rate=drop_rate, img_size=img_size)
    raise ValueError(f"unknown model_type: {model_type!r}")


def freeze_backbone(model) -> int:
    """Freeze all params except the final classifier head (linear-probe / adapter mode).
    For timm rgb models the head is model.get_classifier(); returns # trainable params."""
    import torch.nn as nn
    for p in model.parameters():
        p.requires_grad = False
    head = None
    if hasattr(model, "get_classifier"):
        try:
            head = model.get_classifier()
        except Exception:
            head = None
    if head is None:                      # dual-stream models: train the fusion head only
        head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        for p in head.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_last_k_blocks(model, k: int) -> int:
    """Partial unfreeze for ViT/DINOv2: train only the last-k transformer blocks +
    final norm + ALL task heads (the DG sweet spot between frozen-probe and full-FT).
    Works on plain timm models and on wrapper models exposing `.backbone` (e.g. DTC).
    Returns # trainable params."""
    import torch.nn as nn
    for p in model.parameters():
        p.requires_grad = False
    vit = getattr(model, "backbone", model)   # DTC wrapper -> inner ViT
    # task heads: plain timm (get_classifier) OR wrapper heads (fraud/doctype/head)
    heads = []
    if hasattr(model, "get_classifier"):
        try:
            heads.append(model.get_classifier())
        except Exception:
            pass
    for nm in ("fraud", "doctype", "head"):
        h = getattr(model, nm, None)
        if isinstance(h, nn.Module):
            heads.append(h)
    for h in heads:
        for p in h.parameters():
            p.requires_grad = True
    # last-k transformer blocks of the ViT
    blocks = getattr(vit, "blocks", None)
    if blocks is not None and k > 0:
        for blk in list(blocks)[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
    # final norms (on the ViT)
    for nm in ("norm", "fc_norm", "norm_pre"):
        m = getattr(vit, nm, None)
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
