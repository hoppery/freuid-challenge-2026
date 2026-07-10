from __future__ import annotations
import cv2
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Reads images listed in a manifest DataFrame. Returns (tensor, label) or
    (tensor, image_id) when with_label=False (inference)."""

    def __init__(self, df, transform, with_label: bool = True, with_doctype: bool = False,
                 self_blend_p: float = 0.0, field_tamper_p: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.with_label = with_label
        self.with_doctype = with_doctype  # also return doctype_id (int) for DTC adversarial head
        # self_blend_p>0: with this prob, convert a GENUINE sample into a synthetic doc-agnostic
        # forgery (label 0→1) via self-blending → teaches unseen-type-generalizable forgery cues.
        self.self_blend_p = self_blend_p
        if self_blend_p > 0:
            from freuid.data.self_blend import self_blend as _sb
            self._sb = _sb
        # field_tamper_p>0: convert a GENUINE sample into a targeted field-tamper forgery (the ROOT-line
        # public-winning aug, ported to ViT-L). Mutually exclusive with self-blend per sample.
        self.field_tamper_p = field_tamper_p
        if field_tamper_p > 0:
            from freuid.data.field_tamper import field_tamper as _ft
            self._ft = _ft

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = float(r["label"])
        if self.with_label and self.self_blend_p > 0 and label == 0.0 and \
                __import__("numpy").random.rand() < self.self_blend_p:
            img = self._sb(img)            # genuine → synthetic self-blend forgery
            label = 1.0
        elif self.with_label and self.field_tamper_p > 0 and label == 0.0 and \
                __import__("numpy").random.rand() < self.field_tamper_p:
            img = self._ft(img)            # genuine → targeted field-tamper forgery
            label = 1.0
        x = self.transform(image=img)["image"]
        if not self.with_label:
            ident = r["image_id"] if "image_id" in self.df.columns else i
            return x, ident
        if self.with_doctype:
            return x, torch.tensor(label), torch.tensor(int(r["doctype_id"]))
        return x, torch.tensor(label)
