"""ONE-CLASS genuine-manifold POC (creative direction #1, 2026-06-19).

CONTRARIAN THESIS: stop modeling attacks (unseen/diverse/partly-unlearnable). Model only a BROAD
genuine manifold (FREUID 4 non-held types + external/captured genuine across many countries), and
score a test image by its DISTANCE from that manifold. Robust to unseen ATTACKS (off-manifold) and —
if the manifold is broad enough — to unseen-TYPE genuine (still inside → no EGYPT-style false flag).

POC: frozen DINOv2 ViT-L features (pure pretrained = maximally type-agnostic; no fine-tune that could
overfit seen types). Genuine manifold = sampled broad genuine. Test = a held-out LODO type
(genuine + attacks). Score = Mahalanobis distance to the genuine manifold (shrinkage cov). Report
FREUID vs the supervised ViT-L LODO number for that fold.

Usage: python3 scripts/oneclass_poc.py --held EGYPT --img 518 --pool 6000
"""
from __future__ import annotations
import argparse, numpy as np, torch, timm
from torch.utils.data import DataLoader
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.metrics import compute_metrics
import pandas as pd


def feats(model, df, img, bs=32):
    dl = DataLoader(ManifestDataset(df, build_transforms("eval", img)), batch_size=bs,
                    shuffle=False, num_workers=10, pin_memory=True)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", enabled=True):
                f = model(x.cuda())            # (B, D) pooled (num_classes=0)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="EGYPT")
    ap.add_argument("--img", type=int, default=518)
    ap.add_argument("--pool", type=int, default=6000, help="genuine manifold sample size")
    a = ap.parse_args()

    rng = np.random.default_rng(42)
    fr = read_manifest("manifests/freuid.parquet")
    held_mask = fr["doc_type"].str.startswith(a.held)
    test = fr[held_mask]                                   # held-out type: genuine + attacks
    # broad genuine manifold: FREUID genuine of the OTHER 4 types + external/captured genuine
    pool_parts = [fr[(~held_mask) & (fr["label"] == 0)]]
    try:
        ext = read_manifest("manifests/union_genuine_v3.parquet")
        pool_parts.append(ext[ext["source"] != "freuid"])  # idnet/docxpand/bid genuine
    except Exception:
        pass
    pool = pd.concat(pool_parts, ignore_index=True)
    pool = pool[pool["label"] == 0]
    if len(pool) > a.pool:
        pool = pool.iloc[rng.choice(len(pool), a.pool, replace=False)].reset_index(drop=True)

    print(f"held={a.held} test n={len(test)} (gen {(test['label']==0).sum()}/att {(test['label']==1).sum()}) "
          f"| genuine-pool n={len(pool)} sources={dict(pool['source'].value_counts())}")

    model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=a.img).cuda().eval()
    Fp = feats(model, pool, a.img)            # genuine manifold features
    Ft = feats(model, test, a.img)
    y = test["label"].to_numpy()

    # Mahalanobis to genuine manifold (Ledoit-Wolf-style shrinkage for stability)
    mu = Fp.mean(0)
    Xc = Fp - mu
    cov = (Xc.T @ Xc) / len(Fp)
    shrink = 0.1
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov.astype(np.float64))
    d = Ft.astype(np.float64) - mu
    maha = np.einsum("ij,jk,ik->i", d, inv, d)            # higher = more anomalous = attack-like

    # also a cosine-kNN-to-genuine score (negative mean top-k cosine sim)
    Fpn = Fp / (np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-8)
    Ftn = Ft / (np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-8)
    sim = Ftn @ Fpn.T
    k = 20
    knn = -np.sort(sim, axis=1)[:, -k:].mean(1)            # higher = farther from genuine

    for name, s in [("mahalanobis", maha), ("cos-kNN(20)", knn)]:
        m = compute_metrics(y, s)
        print(f"  [{name}] FREUID={m['freuid_score']:.4f} AUC={m['roc_auc']:.4f} "
              f"APCER@1%={m['apcer_at_1pct_bpcer']:.4f}")


if __name__ == "__main__":
    main()
