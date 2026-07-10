"""Capture-axis eval: score a model on the WHOLE fantasyid_test manifest (unseen doc types + phone-captured),
plain and with TENT. Reports FREUID. Reference: E6 capture card = 0.423 on this proxy.
"""
import sys, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
sys.path.insert(0, "src")
from freuid.metrics import compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", default="manifests/fantasyid_test.parquet")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--div", type=float, default=1.0)
a = ap.parse_args()

val = read_manifest(a.manifest).sort_values("path").reset_index(drop=True)
y = val["label"].to_numpy()
ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
import inspect
sig = inspect.signature(build_classifier)
def fresh():
    kw = {k: getattr(cfg, k, sig.parameters[k].default) for k in ["freeze_backbone", "n_doctypes"] if k in sig.parameters}
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
    m.load_state_dict(ck["model"]); return m
ds = ManifestDataset(val, build_transforms("eval", cfg.img_size))

@torch.no_grad()
def plain():
    m = fresh(); m.eval()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    return np.concatenate([torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy()
                           for x, _ in dl])

def tent():
    m = fresh()
    for p in m.parameters(): p.requires_grad_(False)
    ln = [p for mod in m.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()]
    for p in ln: p.requires_grad_(True)
    opt = torch.optim.Adam(ln, lr=a.lr); m.train()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=10, drop_last=True)
    for x, _ in dl:
        opt.zero_grad()
        with torch.autocast("cuda", enabled=True):
            p = torch.sigmoid(m(x.cuda()).squeeze(1))
            ent = -(p*torch.log(p+1e-6)+(1-p)*torch.log(1-p+1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1-1e-6); marg = -(pm*torch.log(pm)+(1-pm)*torch.log(1-pm))
            loss = ent - a.div*marg
        if torch.isfinite(loss): loss.backward(); opt.step()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=10)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", enabled=True):
                out.append(torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy())
    return np.concatenate(out)

mp = compute_metrics(y, plain()); mt = compute_metrics(y, tent())
print(f"capture proxy fantasyid_test (n={len(val)}, ref E6 card=0.423):")
print(f"  plain: FREUID={mp['freuid_score']:.4f}  AuDET={mp['audet']:.4f}  APCER@1%={mp['apcer_at_1pct_bpcer']:.4f}")
print(f"  TENT : FREUID={mt['freuid_score']:.4f}  AuDET={mt['audet']:.4f}  APCER@1%={mt['apcer_at_1pct_bpcer']:.4f}")
