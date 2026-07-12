# Frozen model weights — SHA-256 manifest

The submitted model is the **CAPTURE card**: a 3-model DTC probability-average. Its weights were
**trained and frozen on or before 2026-07-12** (before the 2026-07-13 code freeze) and uploaded to the
GitHub Release `weights-v1` the same day, so they can be verified **unchanged since the freeze** — no
retraining or weight edits after 2026-07-13.

Weights are not stored in git (they exceed GitHub's 100 MB/file limit). They are hosted as **GitHub Release
`weights-v1`** assets and fetched + SHA-256-verified at Docker build time (see `Dockerfile`, `README.md`).
Release: https://github.com/hoppery/freuid-challenge-2026/releases/tag/weights-v1

## CAPTURE card — the three checkpoints

| local checkpoint | release asset | size | sha256 |
|------------------|---------------|------|--------|
| `checkpoints/exp_e6_fda/epoch2.pt`        | `e6_fda_ep2.pt`        | 341 MB | `990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48` |
| `checkpoints/exp_e6_fda1120/epoch2.pt`    | `e6_fda1120_ep2.pt`    | 348 MB | `c61d1bd76080e4946b187b4330339062ad9107fb8847f73864412d497982714f` |
| `checkpoints/exp_e6reg_fda1120/epoch1.pt` | `e6reg_fda1120_ep1.pt` | 316 MB | `e30b6bcfa7636cefa7ae633884ceed290cbee11bcedbff57a77413ffc35699d0` |

## Verify a downloaded asset

```bash
# e.g. model 1:
curl -fSL https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1/e6_fda_ep2.pt -o e6_fda_ep2.pt
echo "990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48  e6_fda_ep2.pt" | sha256sum -c
```
