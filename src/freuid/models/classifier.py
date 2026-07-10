from __future__ import annotations
import timm
import torch
import torch.nn as nn


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


class FrozenBackboneClassifier(nn.Module):
    """Frozen pretrained backbone + trainable adapter head, with optional partial
    unfreezing of the last N transformer blocks (+ final norm).

    unfreeze_blocks=0: fully frozen (preserves the pretrained manifold).
    unfreeze_blocks=N: last N blocks adapt. Motivation: the fully-frozen DINOv2 failed
    with doc-type-novelty→attack conflation (unseen-type bona-fides all scored ~1.0,
    APCER@1% 0.95); semantic features must shift toward forensic texture, which needs
    some backbone plasticity. See memory/docs 2026-06-13 root-cause analysis.
    """

    def __init__(self, backbone: str, pretrained: bool = True,
                 drop_rate: float = 0.1, img_size: int | None = None,
                 unfreeze_blocks: int = 0):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0)
        if img_size is not None:
            try:
                self.backbone = timm.create_model(backbone, img_size=img_size, **kw)
            except TypeError:
                self.backbone = timm.create_model(backbone, **kw)
        else:
            self.backbone = timm.create_model(backbone, **kw)
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            if not hasattr(self.backbone, "blocks"):
                raise ValueError(f"unfreeze_blocks needs a ViT-style backbone with "
                                 f".blocks; {backbone!r} has none")
            for p in self.backbone.blocks[-unfreeze_blocks:].parameters():
                p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
        self._fully_frozen = unfreeze_blocks == 0
        dim = self.backbone.num_features
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        if self._fully_frozen:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        return self.adapter(f)


def build_classifier(model_type: str = "rgb",
                     backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
                     pretrained: bool = True, drop_rate: float = 0.1,
                     img_size: int | None = None, freeze_backbone: bool = False,
                     unfreeze_blocks: int = 0, use_prototype: bool = False,
                     mixstyle_p: float = 0.0, use_clip: bool = False,
                     use_chroma: bool = False, use_spectral: bool = False,
                     use_gsd: bool = False, gsd_r: int = 3,
                     lora_rank: int = 0, lora_alpha: int = 16, cma_chroma: bool = False,
                     cdc_theta: float = 0.0):
    """Dispatch by model_type.
    'rgb'      = single timm backbone (baseline); with freeze_backbone=True it becomes
                 FrozenBackboneClassifier (optionally unfreeze_blocks last ViT blocks).
    'dtc'      = DTCNet; freeze_backbone/unfreeze_blocks apply to its RGB host.
    'freq_dual'/'hpf_dual' = dual-stream (see models/freq_classifier.py).
    """
    if model_type == "dtc":
        from freuid.models.dtc import build_dtc_model
        return build_dtc_model(backbone, pretrained=pretrained, drop_rate=drop_rate,
                               img_size=img_size, freeze_backbone=freeze_backbone,
                               unfreeze_blocks=unfreeze_blocks,
                               use_prototype=use_prototype, mixstyle_p=mixstyle_p,
                               use_clip=use_clip, use_chroma=use_chroma,
                               use_spectral=use_spectral, use_gsd=use_gsd, gsd_r=gsd_r,
                               lora_rank=lora_rank, lora_alpha=lora_alpha, cma_chroma=cma_chroma,
                               cdc_theta=cdc_theta)
    if freeze_backbone:
        return FrozenBackboneClassifier(backbone, pretrained=pretrained,
                                        drop_rate=drop_rate, img_size=img_size,
                                        unfreeze_blocks=unfreeze_blocks)
    if model_type == "freq_dual":
        from freuid.models.freq_classifier import build_freq_model
        return build_freq_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "hpf_dual":
        from freuid.models.freq_classifier import build_hpf_model
        return build_hpf_model(backbone, pretrained=pretrained, drop_rate=drop_rate)
    if model_type == "rgb":
        return build_model(backbone, pretrained=pretrained, drop_rate=drop_rate, img_size=img_size)
    raise ValueError(f"unknown model_type: {model_type!r}")
