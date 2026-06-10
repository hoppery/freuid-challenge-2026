import albumentations as A
from albumentations.pytorch import ToTensorV2
from freuid.data.forensic_aug import recapture_transforms

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
