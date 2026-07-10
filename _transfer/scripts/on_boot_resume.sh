#!/bin/bash
# Runs at @reboot (user crontab): resumes the aux dataset downloads and, once the GPU
# is back after the reboot, auto-launches the Stage-1 improvement sweep (once-guarded).
cd /home/hoppery/ijcai_freuid_chanllenge || exit 1
unset PYTHONPATH
export PATH=/home/hoppery/miniforge3/bin:$PATH  # cron has a minimal PATH; need conda python (has torch)
sleep 30  # let the system/network settle after boot
# resume related-dataset downloads (idempotent: .done datasets are skipped)
nohup python3 scripts/download/fetch_aux.py >> /tmp/aux_download_boot.log 2>&1 &
# wait for CUDA, then launch the Stage-1 sweep (guarded so it runs only once)
bash scripts/wait_gpu_then_sweep.sh >> /tmp/auto_resume_boot.log 2>&1
