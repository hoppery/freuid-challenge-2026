# Frozen model weights — SHA-256 manifest

The private-track candidate weights below were **trained and frozen on or before 2026-07-12** (before the
2026-07-13 code freeze) and uploaded to the GitHub Release `weights-v1` the same day. They are recorded here
so that whichever set is deployed for the private rows after the private test is released (2026-07-13), its
weights can be verified **unchanged since the freeze** — no retraining or weight edits after 2026-07-13.

Selection is made after the private images are released: we inspect their acquisition type (captured/physical
vs born-digital) and deploy the matching frozen model. Swapping which frozen set the `Dockerfile` fetches is a
*packaging* change (weights unchanged), which the rules allow after the freeze. Both cards ship on the same
Release and are selected at build time with a single build-arg (default = the **capture card**, the leading
bet per the organizers' stated captured/physical + unseen-type private design):

```bash
docker build -t freuid-repro .                        # Candidate 1 — capture card (default)
docker build --build-arg CARD=unseen -t freuid-repro . # Candidate 2 — unseen-FDA card
```

Weights are not stored in git (they exceed GitHub's 100 MB/file limit). They are hosted as **GitHub Release
`weights-v1`** assets and fetched + SHA-256-verified at Docker build time (see `Dockerfile`, `README.md`).
Release: https://github.com/hoppery/freuid-challenge-2026/releases/tag/weights-v1

## Candidate 1 — CAPTURE card (private default: captured / physical private) — DEPLOYED by default

| local checkpoint | release asset | size | sha256 |
|------------------|---------------|------|--------|
| `checkpoints/exp_e6_fda/epoch2.pt`        | `e6_fda_ep2.pt`        | 341 MB | `990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48` |
| `checkpoints/exp_e6_fda1120/epoch2.pt`    | `e6_fda1120_ep2.pt`    | 348 MB | `c61d1bd76080e4946b187b4330339062ad9107fb8847f73864412d497982714f` |
| `checkpoints/exp_e6reg_fda1120/epoch1.pt` | `e6reg_fda1120_ep1.pt` | 316 MB | `e30b6bcfa7636cefa7ae633884ceed290cbee11bcedbff57a77413ffc35699d0` |

## Candidate 2 — UNSEEN-FDA card (hedge: born-digital + unseen document types)

| local checkpoint | release asset | size | sha256 |
|------------------|---------------|------|--------|
| `checkpoints/exp_fdab12_all/epoch2.pt` | `fdab12_all_ep2.pt` | 1171 MB | `e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34` |
| `checkpoints/exp_cnxfda_all/epoch2.pt` | `cnxfda_all_ep2.pt` |  749 MB | `dd4f1738583557f2fcef2f3f33f767f5ac129a6ae403323716cfc5a90231632b` |

The public-track field-tamper ensemble (born-digital, seen types) is a *reference only* — it collapses on
unseen document types and captured data, so it is not a private deployment candidate and is not hosted here.

## Verify a downloaded asset

```bash
# e.g. capture card, model 1:
curl -fSL https://github.com/hoppery/freuid-challenge-2026/releases/download/weights-v1/e6_fda_ep2.pt -o e6_fda_ep2.pt
echo "990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48  e6_fda_ep2.pt" | sha256sum -c
```
