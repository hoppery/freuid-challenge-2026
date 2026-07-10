"""TENT submission for the _transfer ViT-L line. Adapts LayerNorm affines (entropy-min + anti-collapse
diversity) on the public_test images, then writes a Kaggle submission (id,label, all 142818 ids, fill-missing).
Run: PYTHONPATH=_transfer/src python3 scripts/infer_tent_transfer.py --ckpt <ckpt> --out <csv> [--lr 2e-5 --div 1.0 --passes 1]
"""
import argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.adapters.freuid import build_freuid_test_index

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--div", type=float, default=1.0)
ap.add_argument("--passes", type=int, default=1)
ap.add_argument("--fill", type=float, default=0.5)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
import inspect
sig = inspect.signature(build_classifier)
kw = {k: getattr(cfg, k, sig.parameters[k].default) for k in ["n_doctypes", "freeze_backbone"] if k in sig.parameters}
model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
model.load_state_dict(ck["model"])

full = build_freuid_test_index(existing_only=False)
have = build_freuid_test_index(existing_only=True)
print(f"test ids total={len(full)} on-disk={len(have)}")
ds = ManifestDataset(have.assign(label=0, attack_type="none", doc_type="u", source="freuid", split="test"),
                     build_transforms("eval", cfg.img_size), with_label=False)

# TENT: adapt LN affines only
for p in model.parameters():
    p.requires_grad_(False)
ln = [p for m in model.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()]
for p in ln:
    p.requires_grad_(True)
opt = torch.optim.Adam(ln, lr=a.lr)
model.train()
for _ in range(a.passes):
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    for x, _ in dl:
        opt.zero_grad()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda()).squeeze(1))
            ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean()
            pm = p.mean().clamp(1e-6, 1 - 1e-6)
            marg = -(pm * torch.log(pm) + (1 - pm) * torch.log(1 - pm))
            loss = ent - a.div * marg
        if torch.isfinite(loss):
            loss.backward(); opt.step()

# final clean predict pass (ordered)
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
ids, preds = [], []
with torch.no_grad():
    for x, idb in dl:
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        preds.append(np.nan_to_num(p, nan=a.fill)); ids.extend(list(idb))
pred_map = dict(zip(ids, np.concatenate(preds)))
out_ids = list(full["image_id"])
sub = pd.DataFrame({"id": out_ids, "label": [float(pred_map.get(i, a.fill)) for i in out_ids]})
sub.to_csv(a.out, index=False)
n = sum(1 for i in out_ids if i in pred_map)
print(f"wrote {a.out} rows={len(sub)} predicted={n} filled={len(sub)-n}")
print(sub[sub['id'].isin(pred_map)]['label'].describe())
