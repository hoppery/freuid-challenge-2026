"""Weight-space SOUP: average N checkpoints' weights into one model, eval on a held-out
doc-type. Only last-k blocks+head differ (same frozen pretrained init) → well-defined avg.
Usage: python3 scripts/eval_soup.py --holdout BENIN/DL --ckpts checkpoints/exp_tail896_fold2 checkpoints/exp_tail896_fold2_s123"""
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
    ap=argparse.ArgumentParser(); ap.add_argument("--holdout",required=True); ap.add_argument("--ckpts",nargs="+",required=True)
    ap.add_argument("--manifest",default="manifests/freuid.parquet"); a=ap.parse_args()
    cfg=Config.load(f"{a.ckpts[0]}/config.yaml")
    sds=[torch.load(f"{c}/best.pt",map_location="cpu",weights_only=False)["model"] for c in a.ckpts]
    avg={k: sum(sd[k].float() for sd in sds)/len(sds) for k in sds[0]}   # mean weights
    df=read_manifest(a.manifest); df=df[df.doc_type==a.holdout].reset_index(drop=True)
    amp_dtype=torch.bfloat16 if getattr(cfg,"amp_dtype","bf16")=="bf16" else torch.float16
    model=build_classifier(cfg.model_type,cfg.backbone,pretrained=False,img_size=cfg.img_size,n_doctypes=getattr(cfg,"n_doctypes",0)).cuda()
    model.load_state_dict({k:v.to(next(model.parameters()).dtype) for k,v in avg.items()}); model.eval()
    ds=ManifestDataset(df,build_transforms("eval",cfg.img_size)); dl=DataLoader(ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)
    ys,ps=[],[]
    for x,y in dl:
        with torch.autocast("cuda",dtype=amp_dtype,enabled=cfg.amp): p=torch.sigmoid(model(x.cuda())).float().squeeze(1).cpu().numpy()
        ps.append(np.nan_to_num(p,nan=0.5)); ys.append(y.numpy())
    m=compute_metrics(np.concatenate(ys),np.concatenate(ps))
    print(f"SOUP({len(a.ckpts)}) holdout={a.holdout} FREUID={m['freuid_score']:.4f} APCER@1%={m['apcer_at_1pct_bpcer']:.4f} AUC={m['roc_auc']:.4f}")

if __name__=="__main__": main()
