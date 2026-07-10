"""TENT (LayerNorm-only entropy + diversity) on a ViT-Large LODO fold — does the validated TTA
lever (+43% on ViT-B) stack with the ViT-Large breakthrough? Uses the transfer code (model_type=rgb).
Run with PYTHONPATH=_transfer/src. Held-out fold = the ckpt's holdout doc_type.

Usage:
  PYTHONPATH=_transfer/src python3 scripts/tent_vitL.py --ckpt checkpoints/exp_large896_fold2/best.pt --doc-type BENIN/DL
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


def _build(ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         n_doctypes=getattr(cfg, "n_doctypes", 0)).cuda()
    m.load_state_dict(ck["model"])
    return m, cfg


def _loader(df, cfg):
    return DataLoader(ManifestDataset(df.assign(image_id=np.arange(len(df))),
                                      build_transforms("eval", cfg.img_size), with_label=False),
                      batch_size=cfg.batch_size, shuffle=False, num_workers=10, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--doc-type", required=True)
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    df = pd.read_parquet(a.manifest)
    fold = df[df["doc_type"].astype(str) == a.doc_type].reset_index(drop=True)
    y = fold["label"].to_numpy()
    print(f"ckpt={a.ckpt} fold={a.doc_type} n={len(fold)} (bona {(y==0).sum()}/attack {(y==1).sum()})")

    # no-TTA
    m, cfg = _build(a.ckpt); m.eval(); s = []
    with torch.no_grad():
        for x, _ in _loader(fold, cfg):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                s.append(torch.sigmoid(m(x.cuda()).squeeze(1)).float().cpu().numpy())
    mm = compute_metrics(y, np.concatenate(s))
    print(f"  no-TTA  FREUID={mm['freuid_score']:.4f}  APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}  AUC={mm['roc_auc']:.4f}")
    del m; torch.cuda.empty_cache()

    # TENT: LayerNorm-only, entropy + diversity
    m, cfg = _build(a.ckpt)
    ln = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in ln:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ln, lr=a.lr); m.train(); s = []
    for x, _ in _loader(fold, cfg):
        x = x.cuda(); opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            p = torch.sigmoid(m(x).squeeze(1))
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1 - 1e-6)
            marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
            loss = ent - 1.0 * marg
        if torch.isfinite(loss):
            loss.backward(); opt.step()
        with torch.no_grad():
            s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    mm = compute_metrics(y, np.concatenate(s))
    print(f"  TENT    FREUID={mm['freuid_score']:.4f}  APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}  AUC={mm['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
