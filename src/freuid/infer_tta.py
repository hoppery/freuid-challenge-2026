"""TTA inference -> Kaggle submission (conservative TENT, plan_v2 §C).

The public/private test is distribution-shifted from train (born-digital, same doc types,
but a held-out set our model overfits). TTA adapts LayerNorm affine params to the test
distribution by entropy-minimization + an anti-collapse diversity term — validated to
help on every LODO fold. This applies it to the real test images and writes a submission.

Usage:
  python3 -m freuid.infer_tta --ckpt checkpoints/<run>/best.pt --out submission_tta.csv \
          [--lr 2e-5 --div 1.0 --fill-missing 0.5]
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
from freuid.data.adapters.freuid import build_freuid_test_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="submission_tta.csv")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--div", type=float, default=1.0)
    ap.add_argument("--fill-missing", type=float, default=0.5)
    ap.add_argument("--existing-only", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    test = build_freuid_test_index(existing_only=a.existing_only or True)  # need images on disk
    ds = ManifestDataset(test.assign(label=0, attack_type="none", doc_type="u",
                                     source="freuid", split="test"),
                         build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)

    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                             use_prototype=getattr(cfg, "use_prototype", False),
                             use_clip=getattr(cfg, "use_clip", False)).cuda()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    ln = [p for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
          for p in m.parameters()]
    for p in ln:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ln, lr=a.lr)
    model.train()                       # LN uses batch stats; ViT has no BN running stats

    ids, preds = [], []
    for x, idb in dl:
        x = x.cuda()
        opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x).squeeze(1))
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1 - 1e-6)
            marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
            loss = ent - a.div * marg
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(x).squeeze(1)).float().cpu().numpy())
        ids.extend(list(idb))

    sub = pd.DataFrame({"id": ids, "label": np.concatenate(preds)})
    if a.fill_missing is not None:
        full = build_freuid_test_index(existing_only=False)["image_id"]
        miss = full[~full.isin(sub["id"])]
        if len(miss):
            sub = pd.concat([sub, pd.DataFrame({"id": miss, "label": a.fill_missing})],
                            ignore_index=True)
            sub = sub.set_index("id").loc[full].reset_index()
            print(f"filled {len(miss)} missing-image rows with {a.fill_missing}")
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows)")
    print(sub.head())


if __name__ == "__main__":
    main()
