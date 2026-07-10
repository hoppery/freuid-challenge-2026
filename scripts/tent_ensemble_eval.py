"""D18b: TENT-on-ensemble. TENT (entropy+diversity, LN-only) is better on BENIN; the 728+896
resolution-diversity ensemble is better on EGYPT — they are COMPLEMENTARY. Adapt EACH member with
conservative TENT on the held-out fold, then rank-average the adapted scores. Compare to no-TENT
ensemble and per-member TENT.

Usage:
  python3 scripts/tent_ensemble_eval.py --holdout EGYPT/DL \
     --ckpts checkpoints/exp_d8_res728_egypt/best.pt checkpoints/exp_d9_res896_egypt/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


def build(ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         freeze_backbone=cfg.freeze_backbone,
                         use_prototype=getattr(cfg, "use_prototype", False),
                         use_clip=getattr(cfg, "use_clip", False),
                         use_chroma=getattr(cfg, "use_chroma", False),
                         use_spectral=getattr(cfg, "use_spectral", False),
                         use_gsd=getattr(cfg, "use_gsd", False),
                         gsd_r=getattr(cfg, "gsd_r", 3)).cuda()
    m.load_state_dict(ck["model"])
    return m, cfg


def scores(ckpt, fold, tent, lr=2e-5):
    m, cfg = build(ckpt)
    dl = DataLoader(ManifestDataset(fold.assign(image_id=np.arange(len(fold))),
                                    build_transforms("eval", cfg.img_size), with_label=False),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    if not tent:
        m.eval(); s = []
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast("cuda", enabled=cfg.amp):
                    s.append(torch.sigmoid(m(x.cuda()).squeeze(1)).float().cpu().numpy())
        del m; torch.cuda.empty_cache(); return np.concatenate(s)
    for p in m.parameters(): p.requires_grad_(False)
    ln = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]
    for p in ln: p.requires_grad_(True)
    opt = torch.optim.Adam(ln, lr=lr); m.train(); s = []
    for x, _ in dl:
        x = x.cuda(); opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(x).squeeze(1))
            ent = -(p*torch.log(p+1e-6)+(1-p)*torch.log(1-p+1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1-1e-6); marg = -(pm*torch.log(pm)+(1-pm)*torch.log(1-pm))
            loss = ent - 1.0*marg
        if torch.isfinite(loss): loss.backward(); opt.step()
        with torch.no_grad():
            s.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(s)


def rep(name, y, s):
    mm = compute_metrics(y, s)
    print(f"  {name:30s} FREUID={mm['freuid_score']:.4f}  APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}  AUC={mm['roc_auc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.manifest)
    fold = df[df["doc_type"].astype(str) == a.holdout].reset_index(drop=True)
    y = fold["label"].to_numpy()
    print(f"holdout={a.holdout} n={len(fold)}")
    no = [scores(c, fold, tent=False) for c in a.ckpts]
    te = [scores(c, fold, tent=True) for c in a.ckpts]
    print("=== per-member ===")
    for c, s in zip(a.ckpts, no): rep(c.split('/')[-2]+" no-TENT", y, s)
    for c, s in zip(a.ckpts, te): rep(c.split('/')[-2]+" TENT", y, s)
    print("=== ensembles (rank-mean) ===")
    rep("ensemble no-TENT", y, np.stack([rankdata(s) for s in no]).mean(0))
    rep("ensemble TENT", y, np.stack([rankdata(s) for s in te]).mean(0))


if __name__ == "__main__":
    main()
