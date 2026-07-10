import sys, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from freuid.config import Config
from freuid.data.schema import read_manifest
from freuid.data.dataset import ManifestDataset
from freuid.data.transforms import build_transforms
from freuid.models.classifier import build_classifier
sys.path.insert(0, 'src')
from freuid.metrics import compute_metrics

val = read_manifest('manifests/fantasyid_test.parquet').sort_values('path').reset_index(drop=True)
y = val['label'].to_numpy()
scores = {}
for tag, d in [('c896', 'exp_e6_fda'), ('c1120', 'exp_e6_fda1120')]:
    ck = torch.load(f'checkpoints/{d}/epoch2.pt', map_location='cuda', weights_only=False)
    cfg = Config(**{k: v for k, v in ck.get('cfg', {}).items() if hasattr(Config, k)})
    m = build_classifier(cfg.model_type, cfg.backbone, pretrained=False, img_size=cfg.img_size).cuda()
    m.load_state_dict(ck['model']); m.eval()
    dl = DataLoader(ManifestDataset(val, build_transforms('eval', cfg.img_size)),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        s = np.concatenate([torch.sigmoid(m(x.cuda())).float().squeeze(1).cpu().numpy() for x, _ in dl])
    scores[tag] = s
    print(f"  {tag}: FREUID={compute_metrics(y, s)['freuid_score']:.4f}")
    del m; torch.cuda.empty_cache()
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ens = (1 - w) * rankdata(scores['c896']) + w * rankdata(scores['c1120'])
    print(f"  ens 896:{1-w:.1f}/1120:{w:.1f} = {compute_metrics(y, ens)['freuid_score']:.4f}")
