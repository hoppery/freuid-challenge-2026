"""Parallel experiment runner: launches one training run per GPU (4x RTX 3090),
waits for all, then collects each run's best domain-holdout metrics into a results
CSV + prints a table. Automates the Stage-1 sweep in docs/experiments.md.

Usage:
  python3 scripts/run_sweep.py --exp baseline:0 exp_heavyaug:1 exp_freq:2 exp_dinov2:3
  python3 scripts/run_sweep.py            # uses DEFAULT_SWEEP
Each token is <config-stem>:<gpu-index> (config = configs/<stem>.yaml).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import torch

DEFAULT_SWEEP = ["baseline:0", "exp_heavyaug:1", "exp_freq:2", "exp_dinov2:3"]


def launch(stem: str, gpu: int) -> tuple[str, subprocess.Popen, Path]:
    cfg = f"configs/{stem}.yaml"
    if not Path(cfg).exists():
        raise FileNotFoundError(cfg)
    logdir = Path("checkpoints") / stem
    logdir.mkdir(parents=True, exist_ok=True)
    logf = open(logdir / "train.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env.pop("PYTHONPATH", None)  # avoid ROS pollution
    p = subprocess.Popen(["python3", "-m", "freuid.train", "--config", cfg],
                         stdout=logf, stderr=subprocess.STDOUT, env=env)
    print(f"launched {stem} on GPU{gpu} (pid {p.pid}) -> {logdir}/train.log", flush=True)
    return stem, p, logdir


def collect(stem: str, logdir: Path) -> dict:
    best = logdir / "best.pt"
    if not best.exists():
        return {"exp": stem, "status": "NO_CKPT"}
    ck = torch.load(best, map_location="cpu", weights_only=False)
    m = ck.get("metrics", {})
    cfg = ck.get("cfg", {})
    return {
        "exp": stem, "status": "ok",
        "backbone": cfg.get("backbone"), "model_type": cfg.get("model_type"),
        "train_aug": cfg.get("train_aug"), "img_size": cfg.get("img_size"),
        "apcer@1%bpcer": round(float(m.get("apcer_at_1pct_bpcer", -1)), 4),
        "audet": round(float(m.get("audet", -1)), 4),
        "roc_auc": round(float(m.get("roc_auc", -1)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=DEFAULT_SWEEP)
    a = ap.parse_args()
    ngpu = torch.cuda.device_count()
    procs = []
    for tok in a.exp:
        stem, gpu = tok.rsplit(":", 1)
        gpu = int(gpu)
        if gpu >= ngpu:
            print(f"WARN {stem}: GPU{gpu} >= {ngpu} available; skipping")
            continue
        procs.append(launch(stem, gpu))
        time.sleep(5)  # stagger weight downloads / startup

    print(f"\nwaiting for {len(procs)} runs...", flush=True)
    results = []
    for stem, p, logdir in procs:
        rc = p.wait()
        print(f"[{stem}] exited rc={rc}", flush=True)
        results.append(collect(stem, logdir))

    out = Path("docs/experiments_results.csv")
    import csv
    keys = ["exp", "status", "backbone", "model_type", "train_aug", "img_size",
            "apcer@1%bpcer", "audet", "roc_auc"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\n=== SWEEP RESULTS (lower apcer@1%bpcer = better) ===")
    print(json.dumps(sorted(results, key=lambda r: r.get("apcer@1%bpcer", 9)), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
