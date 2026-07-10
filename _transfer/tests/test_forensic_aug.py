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


def test_fda_preserves_shape_no_nan(tmp_path):
    import cv2, numpy as np
    from freuid.data.fda import FDATransform
    paths = []
    for i in range(3):
        p = tmp_path / f"r{i}.jpg"
        cv2.imwrite(str(p), (np.random.rand(80, 80, 3) * 255).astype("uint8"))
        paths.append(str(p))
    fda = FDATransform(paths, size=64, beta=0.05, p=1.0)
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    out = fda(image=img)["image"]
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert not np.isnan(out).any()
