import numpy as np
import pandas as pd
import cv2
import torch
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms


def test_dataset_yields_tensor(tmp_path):
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)
    df = pd.DataFrame([{"path": str(p), "label": 1, "attack_type": "physical",
                        "doc_type": "d", "source": "freuid", "split": "train"}])
    ds = ManifestDataset(df, build_transforms("eval", size=32))
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 32, 32)
    assert int(y) == 1


def test_dataset_inference_returns_id(tmp_path):
    img = (np.random.rand(48, 48, 3) * 255).astype("uint8")
    p = tmp_path / "y.jpg"
    cv2.imwrite(str(p), img)
    df = pd.DataFrame([{"path": str(p), "image_id": "abc"}])
    ds = ManifestDataset(df, build_transforms("eval", size=32), with_label=False)
    x, ident = ds[0]
    assert x.shape == (3, 32, 32)
    assert ident == "abc"
