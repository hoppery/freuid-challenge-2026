# Related Datasets — Verified Sourcing (ID-document fraud / forgery only)

Scope: identity-document & document-forgery datasets only. Generic natural-image
forensics (CASIA/IMD2020/DEFACTO/etc.) are intentionally EXCLUDED as not on-task.

## (A) Programmatically downloadable

| # | Dataset | Source | Size | License | On-task relevance |
|---|---------|--------|------|---------|-------------------|
| 1 | MIDV-500 | FTP `ftp://smartengines.com/midv-500/` | ~22 GB | CC BY-SA | genuine mobile-capture IDs (bona-fide corpus) |
| 2 | MIDV-2019 | FTP `.../midv-500/extra/midv-2019/` | ~9 GB | CC BY-SA | harder capture conditions |
| 3 | MIDV-2020 | FTP `ftp://smartengines.com/midv-2020/` | ~124 GB | CC BY-SA | largest genuine ID base |
| 4 | **MIDV-Holo** | FTP `ftp://smartengines.com/midv-holo` + `github.com/SmartEngines/midv-holo` | clips=700 | CC BY-SA 2.5 | **hologram/OVD forgery** (genuine+attack) |
| 5 | **DLC-2021** | Zenodo 6466768 / 6792396 / 7467000 | 1424 clips | CC BY | **print/screen/copy recapture** liveness |
| 6 | **IDNet** | HF `cactuslab/IDNet-2025` (~125GB) · Kaggle `chitreshkr/idnet-identity-document-analysis` (~45.6GB) | big | CC-BY-4.0 | **synthetic fraud IDs w/ forgery-op labels** |
| 7 | **DocXPand-25k** | GitHub release `QuickSign/docxpand` v1.0.0 (12 parts) | ~18.3 GB | CC-BY-NC-SA | synthetic ID layouts (9 designs) |
| 8 | DocTamper | Kaggle `dinmkeljiame/doctamper` | ~21.7 GB | non-commercial; **LMDB password-gated** | document text tampering localization |
| 9 | **FantasyID** (open) | Zenodo 17063366 `FantasyID.tgz` | 2.6 GB | CC-BY-4.0 | **face-swap + text-inpaint** ID forgery |
| 10 | BID (Brazilian IDs) | GitHub `ricardobnjunior/...` Google Drive `1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6` | 28.8k imgs | research | genuine ID layouts (fake PII) |

## (B) Manual / EULA / request-required
- **SIDTD** (forged vs genuine synthetic IDs, CC-BY-4.0): loader `github.com/Oriolrt/SIDTD_Dataset` (`python setup.py install`), data on CORA/TC-11 `tc11.cvc.uab.es/datasets/SIDTD_1/`. Contact oriolrt@cvc.uab.cat. **Directly on-task** — attempt loader.
- **FakeIDet2-db** (2025, physical composite ID forgeries): EULA email `atvs@uam.es`, portal `bidalab.eps.uam.es/listdatabases`.
- **MIDV-2020 L3i mirror**: Google Form (prefer open Smart Engines FTP).
- **FantasyID restricted partition** (Zenodo 17063494): Zenodo access request + EULA.
- **PAD-IDCard 2024/2025 (IJCB)**: competition data largely private.

## Commands (programmatic)
```bash
# FantasyID (open) — Zenodo
wget https://zenodo.org/records/17063366/files/FantasyID.tgz -O FantasyID.tgz && tar xzf FantasyID.tgz
# DLC-2021 — Zenodo
zenodo_get 10.5281/zenodo.6466768 && zenodo_get 10.5281/zenodo.6792396 && zenodo_get 10.5281/zenodo.7467000
# MIDV-Holo — FTP media + git markup
git clone https://github.com/SmartEngines/midv-holo.git
lftp -e "mirror /midv-holo ./midv-holo; quit" smartengines.com
# DocXPand-25k — GitHub release (12 parts)
BASE=https://github.com/QuickSign/docxpand/releases/download/v1.0.0
for i in $(seq -w 0 11); do wget "$BASE/DocXPand-25k.tar.gz.$i"; done; cat DocXPand-25k.tar.gz.* | tar xzf -
# IDNet — HuggingFace (full) or Kaggle (subset)
huggingface-cli download cactuslab/IDNet-2025 --repo-type dataset --local-dir ./IDNet-2025
kaggle datasets download -d chitreshkr/idnet-identity-document-analysis --unzip
# DocTamper — Kaggle (LMDB password from authors needed to unlock)
kaggle datasets download -d dinmkeljiame/doctamper --unzip
# BID — Google Drive
gdown 1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6
# MIDV-500 / 2019 / 2020 — FTP
lftp -e "mirror /midv-500 ./midv-500; quit" smartengines.com
```

Non-commercial/research-only: DocXPand (CC-BY-NC-SA), DocTamper, FantasyID-restricted, FakeIDet2, all MIDV (CC BY-SA share-alike).
