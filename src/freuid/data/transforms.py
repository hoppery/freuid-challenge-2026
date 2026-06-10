import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(mode: str, size: int = 384):
    """Image transforms. 'train' adds light, fraud-preserving augmentation."""
    if mode == "train":
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(0.0, 0.03), scale=(0.95, 1.05), rotate=(-3, 3), p=0.3),
            A.Normalize(_MEAN, _STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0),
        A.Normalize(_MEAN, _STD),
        ToTensorV2(),
    ])
