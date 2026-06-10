from __future__ import annotations
import cv2
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Reads images listed in a manifest DataFrame. Returns (tensor, label) or
    (tensor, image_id) when with_label=False (inference)."""

    def __init__(self, df, transform, with_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(image=img)["image"]
        if not self.with_label:
            ident = r["image_id"] if "image_id" in self.df.columns else i
            return x, ident
        return x, torch.tensor(float(r["label"]))
