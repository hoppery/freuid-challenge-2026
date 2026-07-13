# Frozen model weights — SHA-256 manifest

The submitted model is the **unseen-FDA card**: a 2-model probability-average built for born-digital
identity documents, including **document types not seen in training** (Fourier domain adaptation +
architecture diversity). Its weights were **trained and frozen on or before 2026-07-12** (before the
2026-07-13 code freeze) and are hosted on the GitHub Release `weights-v1`, so they can be verified
**unchanged since the freeze** — no retraining or weight edits after 2026-07-13.

Weights are not stored in git (they exceed GitHub's 100 MB/file limit). They are hosted as **GitHub Release
`weights-v1`** assets and fetched + SHA-256-verified at Docker build time (see `Dockerfile`, `README.md`).
Release: https://github.com/hoppery/freuid-challenge-2026/releases/tag/weights-v1

## unseen-FDA card — the two checkpoints

| local checkpoint | release asset | size | sha256 |
|------------------|---------------|------|--------|
| `checkpoints/exp_fdab12_all/epoch2.pt` | `fdab12_all_ep2.pt` | 1171 MB | `e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34` |
| `checkpoints/exp_cnxfda_all/epoch2.pt` | `cnxfda_all_ep2.pt` |  749 MB | `dd4f1738583557f2fcef2f3f33f767f5ac129a6ae403323716cfc5a90231632b` |

## Verify a downloaded asset

```bash
# e.g. model 1:
curl -fSL https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1/fdab12_all_ep2.pt -o fdab12_all_ep2.pt
echo "e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34  fdab12_all_ep2.pt" | sha256sum -c
```
