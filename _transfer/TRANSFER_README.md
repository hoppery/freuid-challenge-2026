# FREUID — Best-Model Transfer Bundle

Snapshot to continue training on another PC. Folder structure mirrors the repo root.

## What's inside
```
src/, scripts/, configs/, manifests/, tests/, docs/, pyproject.toml   # full code + recipes + findings
checkpoints/
  exp_prod896_large/        # ★ BEST recipe, all-5-types deployable — DINOv2 ViT-Large @896 + FDA + tail_margin
  exp_large896_fold0/       # ViT-Large LODO fold (EGYPT held out)  — the breakthrough evidence
  exp_large896_fold2/       # ViT-Large LODO fold (BENIN held out)  — BENIN 0.0165 (vs Base 0.0827, -80%)
  exp_prod896_tail/         # validated Base recipe (ViT-Base @896 + FDA + tail_margin), all-5-types
```
(Large `exp_prod896_large` is included only if it finished training before the bundle was made.)

## The winning recipe (decide everything by 5-fold leave-one-doc-type-out CV)
**DINOv2 ViT-Large `vit_large_patch14_reg4_dinov2.lvd142m` + partial-unfreeze last-2 blocks + FDA + img 896 + `tail_margin` loss + medium aug + bf16/grad_clip(1.0)/EMA/best-epoch/early-stop(2).**
- Base ViT-B 5-fold LODO-CV = 0.020 (BENIN 0.083 the bottleneck). ViT-Large cracks BENIN to 0.0165 (-80%); projected Large 5-fold ≈ 0.006.
- Private test = 2 UNSEEN doc-types → LODO-CV is the honest proxy; public LB (seen 5 types) is in-domain, do NOT tune on it.

## Setup on the new PC
```bash
unset PYTHONPATH                          # ROS pollutes PYTHONPATH; clear it
conda activate <env-with-torch-cu>        # needs torch+CUDA (DINOv2 via timm)
python3 -m pip install -e ".[dev]"
python3 -m pytest -q                      # 36 tests, CPU-safe
```
**Data:** raw images are NOT in this bundle (too large). Manifests under `manifests/*.parquet` are the data contract (columns: path,label,attack_type,doc_type,source,split). Re-point `path` to your local FREUID/IDNet copies, or rebuild via `src/freuid/data/adapters/` + `scripts/build_*`.

## Continue training
```bash
# resume/extend the winning recipe (edit out_dir to avoid overwrite)
CUDA_VISIBLE_DEVICES=0 python3 -m freuid.train --config configs/exp_prod896_large.yaml
# 5-fold LODO (the decision protocol)
python3 scripts/make_folds.py exp_large896 && python3 scripts/run_sweep.py --configs exp_large896_fold0 ... --gpus 0 1
python3 scripts/agg_folds.py exp_large896
# init from a provided checkpoint: torch.load('checkpoints/<x>/best.pt')['model'] -> model.load_state_dict(...)
```
Each `checkpoints/<x>/` has `best.pt` (full state_dict + cfg + metrics), `config.yaml`, `log.csv`.

## Full findings & next frontier
See `docs/iterations.md` (every hypothesis→method→result→verdict) and `docs/unseen-robustness-plan.md`. Verdicts: resolution 896 = dominant win; tail_margin = operating-point win; ViT-Large = bottleneck win; external-data diversity = dead at 896; recap/capture-aug = digital↔captured tradeoff (needs captured val data); ensembling = robustness not mean-gain (hard folds high seed-variance). Next: finish Large 5-fold + Large production; captured-attack data (DLC-2021/MIDV-Holo) for the host's print/physical emphasis; novel contributions (masked-freq SSL, intra-doc recapture-consistency).
