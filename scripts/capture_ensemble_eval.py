"""Rank-mean ensemble of N checkpoints on the FULL capture proxy (fantasyid_test, all rows).
Answers the deployment question: does fusing the base FREUID model (strong unseen-type, weak
capture) with an E2 capture model recover capture robustness while the base protects unseen-type?
Metric = official FREUID Score (lower=better).

Usage:
  python3 scripts/capture_ensemble_eval.py --manifest manifests/fantasyid_test.parquet \
      --ckpts checkpoints/exp_deploy896_all/best.pt checkpoints/exp_e2early_all/epoch1.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


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
    ds = ManifestDataset(df.assign(image_id=np.arange(len(df))),
                         build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=12, pin_memory=True)
    s = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            s.append(torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--manifest", default="manifests/fantasyid_test.parquet")
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    y = df["label"].to_numpy()
    print(f"manifest={a.manifest} n={len(df)} genuine={(y==0).sum()} attack={(y==1).sum()}")
    per = []
    for ck in a.ckpts:
        s = run_one(ck, df)
        m = compute_metrics(y, s)
        print(f"  [{ck.split('/')[-2]}/{ck.split('/')[-1]}] FREUID={m['freuid_score']:.4f} "
              f"(APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f})")
        per.append(s)
    if len(per) > 1:
        ranks = np.stack([rankdata(s) for s in per]).mean(0)
        mr = compute_metrics(y, ranks)
        mp = compute_metrics(y, np.stack(per).mean(0))
        print(f"  >> ENSEMBLE rank-mean FREUID={mr['freuid_score']:.4f} "
              f"(APCER@1%={mr['apcer_at_1pct_bpcer']:.4f} AUC={mr['roc_auc']:.4f})")
        print(f"  >> ENSEMBLE mean-prob FREUID={mp['freuid_score']:.4f} "
              f"(APCER@1%={mp['apcer_at_1pct_bpcer']:.4f} AUC={mp['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
