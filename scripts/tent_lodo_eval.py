"""LODO TENT eval: measure whether conservative TENT (LayerNorm-affine entropy-min + anti-collapse
diversity, the infer_tta.py recipe) helps a ViT-L model on a held-out doc-type. Transductive: adapt LN
on the held-out fold (= what the private test looks like) and score the same fold. Reports no-TENT vs TENT FREUID.
Line-agnostic (PYTHONPATH=_transfer/src for ViT-L).
"""
import sys, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
sys.path.insert(0, "src")
from freuid.metrics import compute_metrics  # root metrics (same FREUID formula)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--holdout", required=True)
ap.add_argument("--manifest", default="manifests/freuid.parquet")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--div", type=float, default=1.0)
ap.add_argument("--passes", type=int, default=1)   # online passes over the fold
a = ap.parse_args()

df = read_manifest(a.manifest)
col = "doc_type" if "doc_type" in df else "type"
val = df[df[col].astype(str) == a.holdout].sort_values("path").reset_index(drop=True)
y = val["label"].to_numpy()

ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})

def fresh_model():
    import inspect
    sig = inspect.signature(build_classifier)
    kw = {k: getattr(cfg, k, sig.parameters[k].default) for k in
          ["freeze_backbone", "n_doctypes"] if k in sig.parameters}
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
    m.load_state_dict(ck["model"]); return m

ds = ManifestDataset(val, build_transforms("eval", cfg.img_size))

@torch.no_grad()
def score_plain():
    m = fresh_model(); m.eval()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=True):
            out.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    return np.concatenate(out)

def score_tent():
    m = fresh_model()
    for p in m.parameters():
        p.requires_grad_(False)
    ln = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]
    for p in ln:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ln, lr=a.lr)
    m.train()                                  # LN uses batch stats
    for _ in range(a.passes):                  # adapt (shuffled, drop_last for stable batch stats)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=10, drop_last=True)
        for x, _ in dl:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=True):
                p = torch.sigmoid(m(x.cuda()).squeeze(1))
                ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
                pm = p.mean().clamp(1e-6, 1 - 1e-6)
                marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
                loss = ent - a.div * marg
            if torch.isfinite(loss):
                loss.backward(); opt.step()
    # final clean eval pass (ordered)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=10)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", enabled=True):
                out.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    return np.concatenate(out)

mp = compute_metrics(y, score_plain())
mt = compute_metrics(y, score_tent())
print(f"holdout {a.holdout} (n={len(val)}, lr={a.lr} div={a.div} passes={a.passes}):")
print(f"  no-TENT: FREUID={mp['freuid_score']:.4f}  AuDET={mp['audet']:.4f}  APCER@1%={mp['apcer_at_1pct_bpcer']:.4f}")
print(f"  TENT   : FREUID={mt['freuid_score']:.4f}  AuDET={mt['audet']:.4f}  APCER@1%={mt['apcer_at_1pct_bpcer']:.4f}")
print(f"  => TENT {'HELPS' if mt['freuid_score']<mp['freuid_score'] else 'HURTS'} "
      f"({mp['freuid_score']:.4f} -> {mt['freuid_score']:.4f})")
