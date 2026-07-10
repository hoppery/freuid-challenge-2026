import albumentations as A
from albumentations.pytorch import ToTensorV2
from freuid.data.forensic_aug import (recapture_transforms, recapture_transforms_medium,
                                      recapture_transforms_recap)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(mode: str, size: int = 384, extra=None):
    """Image transforms.
    - 'train': light, fraud-preserving augmentation (honest baseline).
    - 'medium': light + GENTLE recapture aug (preserves fine digital-edit cues; the
      working cross-domain aug for FREUID).
    - 'heavy': light + FULL recapture block. WARNING: too strong for FREUID — destroys
      the fine forgery cue (AUC collapses to 0.50). Kept for reference only.
    - 'eval'/other: deterministic resize+normalize.
    `extra`: list of ops inserted just before Normalize (e.g. FDA domain-adaptation aug).
    """
    if mode in ("train", "heavy", "medium", "recap"):
        ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(0.0, 0.03), scale=(0.95, 1.05), rotate=(-3, 3), p=0.3),
        ]
        if mode == "heavy":
            ops += recapture_transforms(p_scale=1.0)
        elif mode == "medium":
            ops += recapture_transforms_medium()
        elif mode == "recap":   # capture-realistic (print-and-capture + lighting) — host private emphasis
            ops += recapture_transforms_recap()
        if extra:
            ops += list(extra)
        ops += [A.Normalize(_MEAN, _STD), ToTensorV2()]
        return A.Compose(ops)
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.Normalize(_MEAN, _STD),
        ToTensorV2(),
    ])
