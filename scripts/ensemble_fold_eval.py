"""Evaluate one or more checkpoints (score-averaged ensemble) on ALL images of a held-out
doc_type — the LODO fold. Answers: does ensembling complementary models (e.g. baseline +
appearance_inv) recover the best of both folds? Metric = official FREUID Score (lower=better).

Usage:
  python3 scripts/ensemble_fold_eval.py --doc-type EGYPT/DL \
      --ckpts checkpoints/exp_d9_res896_egypt/best.pt checkpoints/exp_d10_appinv_egypt/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics, freuid_score


@torch.no_grad()
def run_one(ckpt_path, df):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone,
                             use_prototype=getattr(cfg, "use_prototype", False),
                             use_clip=getattr(cfg, "use_clip", False),
                             use_chroma=getattr(cfg, "use_chroma", False),
                             use_spectral=getattr(cfg, "use_spectral", False),
                             use_gsd=getattr(cfg, "use_gsd", False),
                             gsd_r=getattr(cfg, "gsd_r", 3),
                             lora_rank=getattr(cfg, "lora_rank", 0),
                             lora_alpha=getattr(cfg, "lora_alpha", 16),
                             cma_chroma=getattr(cfg, "cma_chroma", False)).cuda()
    model.load_state_dict(ck["model"]); model.eval()
    ds = ManifestDataset(df, build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    scores = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        scores.append(p)
    del model; torch.cuda.empty_cache()
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--doc-type", required=True, help="held-out doc_type, e.g. EGYPT/DL")
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    a = ap.parse_args()

    df = pd.read_parquet(a.manifest)
    fold = df[df["doc_type"].astype(str) == a.doc_type].reset_index(drop=True)
    if "image_id" not in fold:  # ManifestDataset needs an id col for with_label=False
        fold = fold.assign(image_id=fold.index.astype(str))
    y = fold["label"].to_numpy()
    print(f"doc_type={a.doc_type}  n={len(fold)}  genuine={(y==0).sum()} attack={(y==1).sum()}")

    per_ckpt = []
    for ck in a.ckpts:
        s = run_one(ck, fold)
        m = compute_metrics(y, s)
        print(f"  [{ck.split('/')[-2]}] FREUID={m['freuid_score']:.4f} "
              f"(AuDET={m['audet']:.4f} APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f})")
        per_ckpt.append(s)

    if len(per_ckpt) > 1:
        P = np.stack(per_ckpt)
        # mean-probability ensemble
        m_prob = compute_metrics(y, P.mean(0))
        # rank-mean ensemble (metric is rank-based, so this is the natural combiner)
        ranks = np.stack([rankdata(s) for s in per_ckpt]).mean(0)
        m_rank = compute_metrics(y, ranks)
        print(f"  >> ENSEMBLE mean-prob FREUID={m_prob['freuid_score']:.4f} "
              f"(AuDET={m_prob['audet']:.4f} APCER@1%={m_prob['apcer_at_1pct_bpcer']:.4f})")
        print(f"  >> ENSEMBLE rank-mean FREUID={m_rank['freuid_score']:.4f} "
              f"(AuDET={m_rank['audet']:.4f} APCER@1%={m_rank['apcer_at_1pct_bpcer']:.4f})")


if __name__ == "__main__":
    main()
