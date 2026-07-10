"""Ensemble inference -> Kaggle submission. Averages sigmoid scores from several
checkpoints over the on-disk public-test images; fills missing ids with --fill.
Usage: python3 scripts/ensemble_infer.py --ckpts checkpoints/exp_fda_fold{0..4} --out sub.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from freuid.config import Config
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
from freuid.data.adapters.freuid import build_freuid_test_index


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint dirs (each with best.pt + config.yaml)")
    ap.add_argument("--out", default="submission_ensemble.csv")
    ap.add_argument("--fill", type=float, default=0.5)
    a = ap.parse_args()

    full = build_freuid_test_index(existing_only=False)
    have = build_freuid_test_index(existing_only=True)
    print(f"test ids total={len(full)} on-disk={len(have)}")

    acc = None
    ids_ref = None
    for cd in a.ckpts:
        cfg = Config.load(f"{cd}/config.yaml")
        amp_dtype = torch.bfloat16 if getattr(cfg, "amp_dtype", "bf16") == "bf16" else torch.float16
        ck = torch.load(f"{cd}/best.pt", map_location="cuda", weights_only=False)
        model = build_classifier(cfg.model_type, cfg.backbone, pretrained=False,
                                 img_size=cfg.img_size, n_doctypes=getattr(cfg, "n_doctypes", 0)).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        ds = ManifestDataset(have.assign(label=0, attack_type="none", doc_type="u",
                                         source="freuid", split="test"),
                             build_transforms("eval", cfg.img_size), with_label=False)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        ids, ps = [], []
        for x, idb in dl:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
                p = torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
            ps.append(np.nan_to_num(p, nan=a.fill))
            ids.extend(list(idb))
        p_arr = np.concatenate(ps)
        if acc is None:
            acc = p_arr.astype(np.float64)
            ids_ref = ids
        else:
            assert ids == ids_ref, "id order mismatch across models"
            acc += p_arr
        m = ck.get("metrics", {})
        print(f"  {cd}: done (holdout FREUID={m.get('freuid_score','?')})")
        del model
        torch.cuda.empty_cache()

    acc /= len(a.ckpts)
    pred_map = dict(zip(ids_ref, acc))
    out_ids = list(full["image_id"])
    scores = [float(pred_map.get(i, a.fill)) for i in out_ids]
    sub = pd.DataFrame({"id": out_ids, "label": scores})
    sub.to_csv(a.out, index=False)
    n_pred = sum(1 for i in out_ids if i in pred_map)
    print(f"wrote {a.out} rows={len(sub)} predicted={n_pred} filled={len(sub)-n_pred}")
    print(sub.head())


if __name__ == "__main__":
    main()
