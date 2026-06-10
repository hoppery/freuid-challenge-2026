"""Parallel experiment runner over a restricted GPU pool.

GPU POLICY (per user): train on 3 GPUs only, EXCLUDING the monitor GPU (GPU 3 has the
display attached). Allowed training GPUs = [0, 1, 2]. NOTE: GPU 0 may be shared with
another project's job — keep its batch size modest.

Schedules a list of experiment configs onto the GPU pool: at most len(pool) run at once,
the rest queue until a GPU frees. Collects each run's best domain-holdout metrics into a
results CSV + prints a ranked table.

Usage:
  python3 scripts/run_sweep.py                                  # DEFAULT_CONFIGS on [0,1,2]
  python3 scripts/run_sweep.py --configs baseline exp_freq      # subset
  python3 scripts/run_sweep.py --gpus 1 2                       # override pool
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
import torch

ALLOWED_GPUS = [0, 1, 2]   # exclude GPU 3 (monitor)
DEFAULT_CONFIGS = ["baseline", "exp_heavyaug", "exp_freq", "exp_dinov2"]


def launch(stem: str, gpu: int):
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
    print(f"[launch] {stem} on GPU{gpu} (pid {p.pid}) -> {logdir}/train.log", flush=True)
    return p, logdir


def collect(stem: str) -> dict:
    best = Path("checkpoints") / stem / "best.pt"
    if not best.exists():
        return {"exp": stem, "status": "NO_CKPT"}
    ck = torch.load(best, map_location="cpu", weights_only=False)
    m, cfg = ck.get("metrics", {}), ck.get("cfg", {})
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
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--gpus", nargs="*", type=int, default=ALLOWED_GPUS)
    a = ap.parse_args()
    ngpu = torch.cuda.device_count()
    pool = [g for g in a.gpus if g < ngpu]
    if 3 in pool:
        print("WARN: GPU3 is the monitor GPU and should be excluded; removing it.")
        pool = [g for g in pool if g != 3]
    print(f"GPU pool = {pool} | configs = {a.configs}", flush=True)

    queue = list(a.configs)
    free = list(pool)
    running = {}   # gpu -> (stem, proc, logdir)
    results = []
    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            stem = queue.pop(0)
            try:
                p, logdir = launch(stem, gpu)
                running[gpu] = (stem, p, logdir)
                time.sleep(8)  # stagger startup / weight downloads
            except FileNotFoundError as e:
                print(f"[skip] {stem}: {e}", flush=True)
                free.append(gpu)
        for gpu, (stem, p, logdir) in list(running.items()):
            if p.poll() is not None:
                print(f"[done] {stem} rc={p.returncode}", flush=True)
                results.append(collect(stem))
                del running[gpu]
                free.append(gpu)
        time.sleep(10)

    out = Path("docs/experiments_results.csv")
    keys = ["exp", "status", "backbone", "model_type", "train_aug", "img_size",
            "apcer@1%bpcer", "audet", "roc_auc"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\n=== SWEEP RESULTS (lower apcer@1%bpcer = better) ===")
    print(json.dumps(sorted(results, key=lambda r: r.get("apcer@1%bpcer", 9)), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
