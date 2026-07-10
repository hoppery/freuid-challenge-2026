"""Cross-corpus UNSEEN eval: score a FREUID-trained model on a totally different
document corpus (e.g. IDNet EU IDs/passports — never in 896 training). A harsher
proxy for the private test's truly-alien unseen types than within-FREUID LODO.
Usage: python3 scripts/eval_crosscorpus.py --ckpt checkpoints/X/best.pt --config X/config.yaml --manifest manifests/idnet_all.parquet
"""
from __future__ import annotations
import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="manifests/idnet_all.parquet")
    ap.add_argument("--cap", type=int, default=8000, help="subsample for speed (balanced)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    amp_dtype = torch.bfloat16 if getattr(cfg, "amp_dtype", "bf16") == "bf16" else torch.float16
    df = read_manifest(a.manifest)
    if a.cap and len(df) > a.cap:
        rng = np.random.default_rng(42)
        parts = [g.sample(min(len(g), a.cap // 2), random_state=42) for _, g in df.groupby("label")]
        import pandas as pd
        df = pd.concat(parts).reset_index(drop=True)
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                             img_size=cfg.img_size, n_doctypes=getattr(cfg, "n_doctypes", 0)).cuda()
    model.load_state_dict(ck["model"]); model.eval()
    ds = ManifestDataset(df, build_transforms("eval", cfg.img_size))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    ys, ps = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        ps.append(np.nan_to_num(p, nan=0.5)); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = compute_metrics(y, p)
    print(f"CROSS-CORPUS [{a.manifest}] n={len(y)} bona={int((y==0).sum())} atk={int((y==1).sum())}")
    print(f"  FREUID={m['freuid_score']:.4f} AuDET={m['audet']:.4f} "
          f"APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
