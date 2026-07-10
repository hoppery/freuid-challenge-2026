"""D16 validation: does the 5-fold LODO ensemble generalize to UNSEEN-type proxies (FantasyID,
13 types incl Arabic/Persian + capture shift) better than a single all-types model? Each LODO
member is blind to one FREUID type → the ensemble is calibrated to novelty (the private unseen
scenario). Runs each member on FantasyID, reports per-member + rank-mean ensemble FREUID/AUC.

Usage:
  python3 scripts/ensemble_fantasyid_eval.py --ckpts <m1.pt> <m2.pt> ... \
      [--baseline checkpoints/exp_deploy896_all/best.pt]
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata

from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


@torch.no_grad()
def run_one(ckpt, val):
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
    m.load_state_dict(ck["model"]); m.eval()
    dl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    s = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            s.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(s)


def rep(name, y, s):
    mm = compute_metrics(y, s)
    print(f"  {name:34s} FREUID={mm['freuid_score']:.4f}  AUC={mm['roc_auc']:.4f}  "
          f"APCER@1%={mm['apcer_at_1pct_bpcer']:.4f}")
    return mm['roc_auc']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--manifest", default="manifests/fantasyid.parquet")
    a = ap.parse_args()
    val = read_manifest(a.manifest)
    y = val["label"].to_numpy()
    print(f"FantasyID proxy n={len(val)} (bona {int((y==0).sum())}/attack {int((y==1).sum())}) "
          f"| FREUID-only single baseline AUC ~0.586")

    print("=== per-member (LODO) ===")
    member_scores = []
    for ck in a.ckpts:
        s = run_one(ck, val)
        rep(ck.split('/')[-2], y, s)
        member_scores.append(s)

    print("=== 5-fold LODO ENSEMBLE ===")
    ranks = np.stack([rankdata(s) for s in member_scores]).mean(0)
    rep("ensemble rank-mean", y, ranks)
    rep("ensemble prob-mean", y, np.stack(member_scores).mean(0))

    if a.baseline:
        print("=== single all-types baseline (deployment) ===")
        rep("896 all-types", y, run_one(a.baseline, val))


if __name__ == "__main__":
    main()
