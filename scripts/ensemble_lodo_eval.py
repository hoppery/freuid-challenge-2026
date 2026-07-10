"""Ensemble-hedge LODO eval: does fusing field-tamper-LODO with an appearance-robust model
recover the EGYPT collapse while keeping the BENIN gain?

For a held-out fold, score each checkpoint on the SAME held-out val set, then compute FREUID for
each model alone and for rank-fused pairs. Rank-fusion (AUC/APCER are rank metrics) via averaged ranks.
"""
import sys, argparse
import numpy as np, torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.metrics import compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", required=True)          # e.g. EGYPT/DL
ap.add_argument("--manifest", default="manifests/freuid.parquet")
ap.add_argument("--ckpts", nargs="+", required=True) # name=path pairs
args = ap.parse_args()

df = read_manifest(args.manifest)
col = "doc_type" if "doc_type" in df else "type"
val = df[df[col].astype(str) == args.holdout].reset_index(drop=True)
y = val["label"].to_numpy()
print(f"holdout {args.holdout}: val={len(val)} (genuine={int((y==0).sum())} attack={int((y==1).sum())})")

@torch.no_grad()
def score(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
    model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size,
                             freeze_backbone=cfg.freeze_backbone,
                             cma_chroma=getattr(cfg, "cma_chroma", False),
                             cdc_theta=getattr(cfg, "cdc_theta", 0.0)).cuda()
    model.load_state_dict(ck["model"]); model.eval()
    ds = ManifestDataset(val, build_transforms("eval", cfg.img_size))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        out.append(p)
    return np.concatenate(out)

named = {}
for spec in args.ckpts:
    name, path = spec.split("=", 1)
    s = score(path)
    named[name] = s
    m = compute_metrics(y, s)
    print(f"  {name:14s}: FREUID={m['freuid_score']:.4f}  AuDET={m['audet']:.4f}  APCER@1%={m['apcer_at_1pct_bpcer']:.4f}")

def fuse(*names):
    r = np.mean([rankdata(named[n]) for n in names], axis=0)
    return compute_metrics(y, r)

print("--- rank-fused pairs ---")
keys = list(named)
ft = [k for k in keys if "ft" in k.lower()]
others = [k for k in keys if k not in ft]
for f in ft:
    for o in others:
        m = fuse(f, o)
        print(f"  {f}+{o:12s}: FREUID={m['freuid_score']:.4f}  AuDET={m['audet']:.4f}  APCER@1%={m['apcer_at_1pct_bpcer']:.4f}")
if len(others) >= 2:
    m = fuse(*keys)
    print(f"  ALL({'+'.join(keys)}): FREUID={m['freuid_score']:.4f}")
