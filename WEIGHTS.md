# Frozen model weights — SHA-256 manifest

All candidate model weights below were **trained and frozen on or before 2026-07-12** (before the
2026-07-13 code freeze). They are recorded here so that whichever set is deployed for the private
rows after the private test is released (2026-07-13), its weights can be verified to be **unchanged
since the freeze** — no retraining or weight edits after 2026-07-13.

Selection is made after the private images are released: we inspect their acquisition type (captured/
physical vs born-digital) and deploy the matching frozen model. Swapping which frozen set the Docker
loads is a *packaging* change (weights unchanged), which the rules allow after the freeze. The Docker
default is the **capture card** (our leading bet, per the organizers' stated captured/physical +
unseen-type private design).

Weights are not stored in git (they exceed GitHub's 100 MB/file limit); they are hosted as GitHub
Release assets and fetched at Docker build time (see `Dockerfile`, `README.md`).

## Candidate 1 — CAPTURE card (private-track default: captured/physical private)

| file | size | sha256 |
|------|------|--------|
| `checkpoints/exp_e6_fda/epoch2.pt`        | 341 MB | `990703a5bee241747b527fb4b3aa202f94e5871c100ca6cd1d2cd51308580e48` |
| `checkpoints/exp_e6_fda1120/epoch2.pt`    | 348 MB | `c61d1bd76080e4946b187b4330339062ad9107fb8847f73864412d497982714f` |
| `checkpoints/exp_e6reg_fda1120/epoch1.pt` | 316 MB | `e30b6bcfa7636cefa7ae633884ceed290cbee11bcedbff57a77413ffc35699d0` |

## Candidate 2 — UNSEEN-FDA card (hedge: born-digital + unseen document types)

| file | size | sha256 |
|------|------|--------|
| `checkpoints/exp_fdab12_all/epoch2.pt` | 1171 MB | `e44aa6b107011f5663849ac655ef1aa197d1e82bac4be21eb875e69cd5a86b34` |
| `checkpoints/exp_cnxfda_all/epoch2.pt` |  749 MB | `dd4f1738583557f2fcef2f3f33f767f5ac129a6ae403323716cfc5a90231632b` |

## Candidate 3 — FIELD-TAMPER ensemble (public-track / born-digital seen types; public LB 0.00039)

| file | size | sha256 |
|------|------|--------|
| `checkpoints/exp_ftcnx_large/epoch2.pt`     |  749 MB | `679566ecf191222d147565ef19d64611e0621517342a9db44139e990a033d031` |
| `checkpoints/exp_ftgiant/epoch2.pt`         | 4351 MB | `ef8b2f0cbbedf3e19b8d25860bc51bf368b0e1ef2fad6b99a4e9f8bfca2c9039` |
| `checkpoints/exp_fthires1120_s43/epoch2.pt` | 1180 MB | `523873efdff0fd6b81b3b93372eccd815aebbd53bfd1635c33e7d54f1d89106f` |

## Verify (after downloading any set)

```bash
sha256sum -c <(grep -oE '[0-9a-f]{64}  `?checkpoints/[^`]*`?' WEIGHTS.md | tr -d '`')
# or check one file:
sha256sum checkpoints/exp_e6_fda/epoch2.pt   # must equal the hash above
```
