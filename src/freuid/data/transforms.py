import albumentations as A
from albumentations.pytorch import ToTensorV2
from freuid.data.forensic_aug import recapture_transforms, fhag_transforms

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(mode: str, size: int = 384):
    """Image transforms.
    - 'train': light, fraud-preserving augmentation (honest baseline).
    - 'heavy': baseline aug + forensic recapture block (print-scan/moiré/recompression)
      applied to BOTH classes for cross-domain robustness (see docs/sota-research.md).
    - 'eval'/other: deterministic resize+normalize.
    """
    if mode in ("train", "heavy"):
        ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(0.0, 0.03), scale=(0.95, 1.05), rotate=(-3, 3), p=0.3),
        ]
        if mode == "heavy":
            ops += recapture_transforms(p_scale=1.0)
        ops += [A.Normalize(_MEAN, _STD), ToTensorV2()]
        return A.Compose(ops)
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.Normalize(_MEAN, _STD),
        ToTensorV2(),
    ])


def build_patch_transforms(size: int = 384, scale=(0.08, 0.35)):
    """(pre, post) for PATCH-based DTC training (iteration #11). RandomResizedCrop on a
    small region destroys the global layout/script identity (the EGYPT-fold wall: unseen
    Arabic-script bona-fides read as anomalies) while preserving forensic micro-texture.
    Published precedent: patch-DINOv2 (FakeIDet) holds 0% EER on unseen datasets."""
    pre = A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=scale, ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
    ])
    post = A.Compose([A.Normalize(_MEAN, _STD), ToTensorV2()])
    return pre, post


def build_patch_eval_transform(size: int = 384):
    """Deterministic per-crop eval transform (crops produced by the dataset)."""
    return A.Compose([A.Normalize(_MEAN, _STD), ToTensorV2()])


def appearance_invariance_ops(mild: bool = False):
    """Color/palette randomization (D10) to break the doc-type COLOR identity — the measured
    difference among the 5 train types is mainly palette/saturation (GUINEA 0.20 vs MOZ 0.06),
    and the 2 private types differ chiefly in appearance. Forces the model onto type-invariant
    forensic micro-texture instead of memorizing each type's colors. Applied to BOTH classes
    on uint8 RGB BEFORE TraceMix (cannot become a class shortcut; TraceMix's region-wise
    coherence labels are unaffected by a global color shift). All ops preserve spatial
    structure / high-freq forensic detail (no blur, no resample).

    mild=True (D10b): the strong version won EGYPT 8x but HURT BENIN 2.4x (its genuine/attack
    separation lives partly in chroma). Gentler variant keeps real color most of the time —
    drops the fully color-destroying ops (ToGray, ChannelShuffle) and halves probabilities /
    strengths — seeking a net-positive sweet spot across both folds."""
    if mild:
        return [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.12, p=0.5),
            A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=0.3),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=12, p=0.3),
        ]
    return [
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.25, p=0.9),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.4),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=20, p=0.4),
        A.ToGray(p=0.15),
        A.ChannelShuffle(p=0.10),
    ]


def build_dtc_transforms(size: int = 384, heavy_recapture: bool = False,
                         appearance_inv: bool = False, appearance_inv_mild: bool = False,
                         fhag: bool = False):
    """(pre, post) pair for the DTC pipeline: geometric augs on uint8, then TraceMix is
    applied by the dataset BETWEEN the two, then normalize+tensorize. TraceMix must see
    the final geometry (it synthesizes capture-time traces) but raw uint8 pixels.

    heavy_recapture=True (plan_v2 E1): also apply the print-and-capture block
    (multi-stage JPEG / moiré / down-up / noise / blur) to BOTH classes, simulating the
    physical print-and-capture domain that the train set (99.97% born-digital) lacks but
    the test set is dominated by. Applied BEFORE TraceMix so consistency labels still hold."""
    ops = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(0.0, 0.03), scale=(0.95, 1.05), rotate=(-3, 3), p=0.3),
    ]
    if appearance_inv or appearance_inv_mild:
        ops += appearance_invariance_ops(mild=appearance_inv_mild)
    if heavy_recapture:
        ops += recapture_transforms(p_scale=1.0)
    if fhag:                               # FHAG frequency-band amplitude perturbation (E3)
        ops += fhag_transforms(p_scale=1.0)
    pre = A.Compose(ops)
    post = A.Compose([A.Normalize(_MEAN, _STD), ToTensorV2()])
    return pre, post
