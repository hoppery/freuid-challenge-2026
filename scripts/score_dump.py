"""Dump per-sample scores for ONE checkpoint on a held-out fold, keyed by manifest 'path'.
Line-agnostic: run with PYTHONPATH=src (root) or PYTHONPATH=_transfer/src (ViT-L). Sorts by path so
two dumps from different lines align exactly. Output CSV: path,label,score.
"""
import sys, argparse, inspect
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--holdout", required=True)
ap.add_argument("--manifest", default="manifests/freuid.parquet")
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = read_manifest(a.manifest)
col = "doc_type" if "doc_type" in df else "type"
val = df[df[col].astype(str) == a.holdout].sort_values("path").reset_index(drop=True)

ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
sig = inspect.signature(build_classifier)
kw = {}
for k in ["freeze_backbone", "use_prototype", "use_clip", "use_chroma", "use_spectral", "use_gsd",
          "gsd_r", "lora_rank", "lora_alpha", "cma_chroma", "cdc_theta", "n_doctypes"]:
    if k in sig.parameters:
        kw[k] = getattr(cfg, k, sig.parameters[k].default)
model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
model.load_state_dict(ck["model"]); model.eval()

dl = DataLoader(ManifestDataset(val, build_transforms("eval", cfg.img_size)),
                batch_size=cfg.batch_size, shuffle=False, num_workers=10, pin_memory=True)
scores = []
with torch.no_grad():
    for x, _ in dl:
        with torch.autocast("cuda", enabled=True):
            scores.append(torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy())
val = val.assign(score=np.concatenate(scores))
val[["path", "label", "score"]].to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(val)} rows (holdout {a.holdout}, backbone {cfg.backbone}, model_type {cfg.model_type})")
