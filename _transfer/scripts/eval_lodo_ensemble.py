"""LODO-validated ensemble measurement: for ONE held-out doc-type, blend the held-out
predictions of N diverse models (all trained on the OTHER types) and compare the ensemble
FREUID to each single. Logit-space mean fusion (research: better than prob-space for the
strict operating point). Uses EXISTING fold checkpoints — no training.
Usage: python3 scripts/eval_lodo_ensemble.py --holdout BENIN/DL --ckpts checkpoints/exp_tail896_fold2 checkpoints/exp_recap896_fold2 checkpoints/exp_idnet896_fold2
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


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@torch.no_grad()
def score_model(ckpt_dir, df):
    cfg = Config.load(f"{ckpt_dir}/config.yaml")
    amp_dtype = torch.bfloat16 if getattr(cfg, "amp_dtype", "bf16") == "bf16" else torch.float16
    ck = torch.load(f"{ckpt_dir}/best.pt", map_location="cuda", weights_only=False)
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
    del model; torch.cuda.empty_cache()
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--manifest", default="manifests/freuid.parquet")
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    df = df[df.doc_type == a.holdout].reset_index(drop=True)
    print(f"holdout={a.holdout} n={len(df)} bona={int((df.label==0).sum())} atk={int((df.label==1).sum())}")
    y_ref = None; logits = []
    for cd in a.ckpts:
        y, p = score_model(cd, df)
        if y_ref is None: y_ref = y
        m = compute_metrics(y, p)
        print(f"  single {cd.split('/')[-1]:24s} FREUID={m['freuid_score']:.4f} APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f}")
        logits.append(_logit(p))
    blended = 1 / (1 + np.exp(-np.mean(logits, axis=0)))   # logit-space mean → sigmoid
    me = compute_metrics(y_ref, blended)
    print(f"  ENSEMBLE (logit-mean, n={len(a.ckpts)})  FREUID={me['freuid_score']:.4f} APCER@1%={me['apcer_at_1pct_bpcer']:.4f} AUC={me['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
