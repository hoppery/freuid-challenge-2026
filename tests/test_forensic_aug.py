import numpy as np
import torch
from freuid.data.forensic_aug import JpegRecompress, MoireArtifact, DownscaleUpscale
from freuid.data.transforms import build_transforms


def _img():
    return (np.random.rand(96, 64, 3) * 255).astype("uint8")


def test_jpeg_recompress_preserves_shape():
    out = JpegRecompress(p=1.0)(image=_img())["image"]
    assert out.shape == (96, 64, 3) and out.dtype == np.uint8


def test_moire_preserves_shape_and_range():
    out = MoireArtifact(p=1.0)(image=_img())["image"]
    assert out.shape == (96, 64, 3)
    assert out.min() >= 0 and out.max() <= 255


def test_downscale_upscale_preserves_shape():
    out = DownscaleUpscale(p=1.0)(image=_img())["image"]
    assert out.shape == (96, 64, 3)


def test_heavy_transform_yields_tensor():
    t = build_transforms("heavy", size=64)
    x = t(image=_img())["image"]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 64, 64)
