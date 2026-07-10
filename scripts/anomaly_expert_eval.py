"""EXP-1 (deep-research Dual-Expert plan): genuine-manifold ANOMALY expert for the BENIN
clean-GenAI tail. Model the GENUINE manifold with a FROZEN foundation backbone (never train on
fakes; CLIP-Flow/MIRROR paradigm), score held-out-fold images by one-class anomaly (Mahalanobis
+ kNN distance to genuine features). Report LODO FREUID for anomaly-alone, the 896 detector
alone, and their RANK-FUSION (the FREUID metric is rank-based → calibration-free fusion is valid).

Key open question (deep-research #1): does a frozen genuine-manifold score give ANY separation
on the trace-free BENIN tail, or is it ~0 (information-theoretic trace absence)?

Usage:
  python3 scripts/anomaly_expert_eval.py --holdout EGYPT/DL \
      --backbone vit_base_patch16_clip_224.openai \
      --detector-ckpt checkpoints/exp_d9_res896_egypt/best.pt
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from scipy.stats import rankdata

from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics

CLIP_M = (0.4815, 0.4578, 0.4082)
CLIP_S = (0.2686, 0.2613, 0.2758)
IMN_M = (0.485, 0.456, 0.406)
IMN_S = (0.229, 0.224, 0.225)


def feat_tf(size, clip_norm):
    m, s = (CLIP_M, CLIP_S) if clip_norm else (IMN_M, IMN_S)
    return A.Compose([A.LongestMaxSize(max_size=size),
                      A.PadIfNeeded(size, size, border_mode=0),
                      A.Normalize(m, s), ToTensorV2()])


@torch.no_grad()
def extract(model, df, tf, bs, nw):
    ds = ManifestDataset(df.assign(image_id=np.arange(len(df))), tf, with_label=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda"):
            f = model(x.cuda())
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def maha_score(train, test, alpha=0.1):
    mu = train.mean(0)
    S = np.cov(train.T).astype(np.float64)
    d = S.shape[0]
    S = (1 - alpha) * S + alpha * (np.trace(S) / d) * np.eye(d)
    Si = np.linalg.inv(S)
    diff = (test - mu).astype(np.float64)
    return np.einsum("ij,jk,ik->i", diff, Si, diff)


def knn_score(train, test, k=5):
    tr = torch.tensor(train, device="cuda", dtype=torch.float32)
    te = torch.tensor(test, device="cuda", dtype=torch.float32)
    out = []
    for i in range(0, len(te), 256):
        d = torch.cdist(te[i:i + 256], tr)
        out.append(d.topk(k, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def detector_scores(ckpt, df, bs_default=16):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                         freeze_backbone=cfg.freeze_backbone,
                         use_prototype=getattr(cfg, "use_prototype", False),
                         use_clip=getattr(cfg, "use_clip", False),
                         use_chroma=getattr(cfg, "use_chroma", False),
                         use_spectral=getattr(cfg, "use_spectral", False)).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    ds = ManifestDataset(df.assign(image_id=np.arange(len(df))),
                         build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    s = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            s.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(s)


def report(name, y, score):
    mm = compute_metrics(y, score)
    print(f"  {name:28s} FREUID={mm['freuid_score']:.4f} "
          f"(AuDET={mm['audet']:.4f} APCER@1%={mm['apcer_at_1pct_bpcer']:.4f} AUC={mm['roc_auc']:.4f})")
    return mm['freuid_score']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--backbone", default="vit_base_patch16_clip_224.openai")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--imagenet-norm", action="store_true", help="use ImageNet norm (DINOv2)")
    ap.add_argument("--detector-ckpt", default=None)
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    ap.add_argument("--max-train", type=int, default=20000, help="subsample genuine train feats")
    a = ap.parse_args()

    df = pd.read_parquet(a.manifest)
    val = df[df["doc_type"].astype(str) == a.holdout].reset_index(drop=True)
    train_gen = df[(df["doc_type"].astype(str) != a.holdout) & (df["label"] == 0)].reset_index(drop=True)
    if len(train_gen) > a.max_train:
        train_gen = train_gen.sample(a.max_train, random_state=0).reset_index(drop=True)
    y = val["label"].to_numpy()
    print(f"holdout={a.holdout}  backbone={a.backbone}  train-genuine={len(train_gen)}  "
          f"val={len(val)} (bona {(y==0).sum()}/attack {(y==1).sum()})")

    model = timm.create_model(a.backbone, pretrained=True, num_classes=0).cuda().eval()
    tf = feat_tf(a.size, clip_norm=not a.imagenet_norm)
    ftr = extract(model, train_gen, tf, bs=64, nw=16)
    fva = extract(model, val, tf, bs=64, nw=16)
    del model; torch.cuda.empty_cache()
    print(f"  features: train{ftr.shape} val{fva.shape}")

    s_maha = maha_score(ftr, fva)
    s_knn = knn_score(ftr, fva, k=5)
    print("=== one-class anomaly (genuine-manifold) ===")
    report("maha-alone", y, s_maha)
    report("knn-alone", y, s_knn)

    if a.detector_ckpt:
        s_det = detector_scores(a.detector_ckpt, val)
        print("=== detector (896) + rank-fusion ===")
        report("detector-alone(896)", y, s_det)
        rd = rankdata(s_det)
        for nm, sa in [("maha", s_maha), ("knn", s_knn)]:
            ra = rankdata(sa)
            for w in [0.2, 0.35, 0.5]:
                report(f"fuse det+{nm}(w={w})", y, (1 - w) * rd + w * ra)


if __name__ == "__main__":
    main()
