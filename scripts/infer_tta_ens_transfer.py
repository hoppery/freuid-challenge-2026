"""hflip-TTA + multi-seed rank-ensemble submission for the _transfer ViT-L line.
Each ckpt: predict on public_test original + horizontal flip, average. Then rank-average across ckpts.
Run: PYTHONPATH=_transfer/src python3 scripts/infer_tta_ens_transfer.py --ckpts a.pt b.pt --out sub.csv
"""
import argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.adapters.freuid import build_freuid_test_index

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fill", type=float, default=0.5)
a = ap.parse_args()

full = build_freuid_test_index(existing_only=False)
have = build_freuid_test_index(existing_only=True)
print(f"test ids total={len(full)} on-disk={len(have)}")

@torch.no_grad()
def predict(ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = Config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
    import inspect
    sig = inspect.signature(build_classifier)
    kw = {k: getattr(cfg, k, sig.parameters[k].default) for k in ["n_doctypes"] if k in sig.parameters}
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    ds = ManifestDataset(have.assign(label=0, attack_type="none", doc_type="u", source="freuid", split="test"),
                         build_transforms("eval", cfg.img_size), with_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    ids, sc = [], []
    for x, idb in dl:
        x = x.cuda()
        with torch.autocast("cuda", enabled=cfg.amp):
            p = torch.sigmoid(m(x).squeeze(1)).float()
            pf = torch.sigmoid(m(torch.flip(x, dims=[3]).contiguous()).squeeze(1)).float()  # hflip TTA
        sc.append(((p + pf) / 2).cpu().numpy()); ids.extend(list(idb))
    del m; torch.cuda.empty_cache()
    return ids, np.concatenate(sc)

# per-ckpt TTA probabilities, aligned by id, averaged (interpretable [0,1] submission)
probs = None; base_ids = None
for c in a.ckpts:
    ids, s = predict(c)
    d = dict(zip(ids, s))
    if base_ids is None:
        base_ids = ids
    aligned = np.array([d[i] for i in base_ids])
    probs = aligned if probs is None else probs + aligned
    print(f"  {c}: predicted {len(ids)}")
ens = probs / len(a.ckpts)
pred_map = dict(zip(base_ids, ens))
out_ids = list(full["image_id"])
sub = pd.DataFrame({"id": out_ids, "label": [float(pred_map.get(i, a.fill)) for i in out_ids]})
sub.to_csv(a.out, index=False)
n = sum(1 for i in out_ids if i in pred_map)
print(f"wrote {a.out} rows={len(sub)} predicted={n} filled={len(sub)-n}")
