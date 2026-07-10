"""Diagnose APCER@1%BPCER tail failures: score the domain-holdout val set with a
checkpoint and decompose the score distribution per class.

APCER@1% asks: at the threshold where 1% of bona-fides score higher, how many attacks
score lower? It fails when the TOP of the score range is contaminated by bona-fides.
This script quantifies that contamination and what those bona-fides are.

Usage: python3 scripts/analyze_score_tail.py --ckpt checkpoints/exp_dinov2_frozen/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.train import split
from freuid.metrics import compute_metrics


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="optional parquet of per-sample scores")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, freeze_backbone=cfg.freeze_backbone).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    df = read_manifest(cfg.manifest)
    _, val = split(df, cfg)
    dl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True)
    scores = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            scores.append(torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy())
    s = np.concatenate(scores)
    y = val["label"].to_numpy()
    bona, atk = s[y == 0], s[y == 1]

    print(f"ckpt={a.ckpt}  backbone={cfg.backbone}  model_type={cfg.model_type} "
          f"frozen={cfg.freeze_backbone}  img={cfg.img_size}")
    print(f"val={len(val)} (bona {len(bona)} / attack {len(atk)})")
    print({k: round(v, 4) for k, v in compute_metrics(y, s).items()})

    q = [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
    print("\nquantile      " + "  ".join(f"{x:>7.3f}" for x in q))
    print("bona-fide     " + "  ".join(f"{np.quantile(bona, x):7.4f}" for x in q))
    print("attack        " + "  ".join(f"{np.quantile(atk, x):7.4f}" for x in q))

    thr = np.quantile(bona, 0.99)   # the BPCER=1% operating threshold
    print(f"\nBPCER=1% threshold (99th pct of bona-fide): {thr:.4f}")
    print(f"attacks ABOVE it: {(atk >= thr).sum()}/{len(atk)} "
          f"({(atk >= thr).mean()*100:.1f}%)  -> APCER@1% = {(atk < thr).mean():.4f}")
    n_top = max(1, int(0.01 * len(bona)))
    print(f"top-1% bona-fides ({n_top}): score range "
          f"[{np.sort(bona)[-n_top]:.4f}, {bona.max():.4f}]")
    print(f"bona-fides scoring > attack median ({np.median(atk):.4f}): "
          f"{(bona > np.median(atk)).sum()} ({(bona > np.median(atk)).mean()*100:.2f}%)")
    print(f"saturation: bona>0.99: {(bona > 0.99).sum()}  attack>0.99: {(atk > 0.99).sum()} "
          f"| bona<0.01: {(bona < 0.01).sum()}  attack<0.01: {(atk < 0.01).sum()}")

    if "is_digital" in val.columns:
        tail = val[(y == 0) & (s >= thr)]
        print(f"\ntail bona-fides (>=thr): {len(tail)} | is_digital counts: "
              f"{tail['is_digital'].value_counts().to_dict()}")

    if a.out:
        out = val.copy()
        out["score"] = s
        out.to_parquet(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
