"""Score the WHOLE fantasyid_test manifest for one checkpoint; dump path,label,score.
Line-agnostic (PYTHONPATH=src for ROOT dtc capture cards; =_transfer/src for rgb unseen cards)."""
import sys, argparse, inspect
import numpy as np, torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

val = read_manifest('manifests/fantasyid_test.parquet').sort_values('path').reset_index(drop=True)
ck = torch.load(a.ckpt, map_location='cuda', weights_only=False)
cfg = Config(**{k: v for k, v in ck.get('cfg', {}).items() if hasattr(Config, k)})
sig = inspect.signature(build_classifier)
kw = {k: getattr(cfg, k, sig.parameters[k].default) for k in ["freeze_backbone", "n_doctypes"] if k in sig.parameters}
m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size, **kw).cuda()
m.load_state_dict(ck['model']); m.eval()
dl = DataLoader(ManifestDataset(val, build_transforms('eval', cfg.img_size)),
                batch_size=cfg.batch_size, shuffle=False, num_workers=8)
with torch.no_grad():
    s = np.concatenate([torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy() for x, _ in dl])
val.assign(score=s)[['path', 'label', 'score']].to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(val)} rows ({cfg.backbone.split('.')[0]} @{cfg.img_size} {cfg.model_type})")
