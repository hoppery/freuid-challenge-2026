"""Generate 5 leave-one-doc-type-out (LODO) fold configs from a base config.
LODO-CV across all 5 FREUID doc-types is the honest proxy for the private test
(2 unseen doc-types). Usage: python3 scripts/make_folds.py <base_config_stem>
e.g. python3 scripts/make_folds.py exp_dinov2_uf2  ->  configs/exp_dinov2_uf2_fold{0..4}.yaml
Then: run_sweep over the 5 fold configs; aggregate with scripts/agg_folds.py.
"""
import sys
import yaml

DOC_TYPES = ["EGYPT/DL", "GUINEA/DL", "BENIN/DL", "MOZAMBIQUE/DL", "MAURITIUS/ID"]


def main():
    base = sys.argv[1]
    cfg = yaml.safe_load(open(f"configs/{base}.yaml"))
    for i, dt in enumerate(DOC_TYPES):
        c = dict(cfg)
        c["holdout_by"] = "doc_type"
        c["holdout_values"] = dt
        c["val_sources"] = "freuid"
        c["out_dir"] = f"checkpoints/{base}_fold{i}"
        with open(f"configs/{base}_fold{i}.yaml", "w") as f:
            yaml.safe_dump(c, f, default_flow_style=False)
        print(f"fold{i}: holdout {dt:14s} -> configs/{base}_fold{i}.yaml")
    stems = " ".join(f"{base}_fold{i}" for i in range(5))
    print(f"\nrun: CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sweep.py --configs {stems} --gpus 0 1")
    print(f"then: python3 scripts/agg_folds.py {base}")


if __name__ == "__main__":
    main()
