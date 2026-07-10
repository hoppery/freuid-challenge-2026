"""Sequential downloader for RELATED auxiliary datasets (ID-document / document-forgery
only). Verified sources in docs/datasets-sourcing.md. Idempotent: a dataset with a
`.done` marker is skipped. Logs to docs/aux-download-report.txt.

Order = most directly on-task (forgery-labeled) first, genuine ID corpora last, the
giant ones last. Run in background:  python3 scripts/download/fetch_aux.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("data/raw")
ROOT.mkdir(parents=True, exist_ok=True)


def _run(cmd: str, cwd: str | None = None, timeout: int = 36000) -> None:
    print(f"    $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, timeout=timeout)


def _done(name: str) -> Path:
    return ROOT / name / ".done"


# ---- per-dataset fetchers (each downloads into data/raw/<name>/) ----

def fantasyid(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run("wget -c -q https://zenodo.org/records/17063366/files/FantasyID.tgz -O FantasyID.tgz", cwd=str(d))
    _run("tar xzf FantasyID.tgz && rm -f FantasyID.tgz", cwd=str(d))


def dlc2021(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for doi in ("10.5281/zenodo.6466768", "10.5281/zenodo.6792396", "10.5281/zenodo.7467000"):
        _run(f"zenodo_get {doi}", cwd=str(d))


def midv_holo(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "midv-holo-repo").exists():
        _run("git clone --depth 1 https://github.com/SmartEngines/midv-holo.git midv-holo-repo", cwd=str(d))
    _run('lftp -e "set net:timeout 30; set net:max-retries 3; mirror -c /midv-holo ./media; quit" smartengines.com', cwd=str(d))


def docxpand(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/QuickSign/docxpand/releases/download/v1.0.0"
    for i in range(12):
        _run(f"wget -c -q {base}/DocXPand-25k.tar.gz.{i:02d}", cwd=str(d))
    _run("cat DocXPand-25k.tar.gz.* | tar xzf - && rm -f DocXPand-25k.tar.gz.*", cwd=str(d))


def bid(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run("gdown 1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6 -O bid_full.zip || gdown --fuzzy 1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6 -O bid_full.zip", cwd=str(d))
    _run("unzip -q -o bid_full.zip && rm -f bid_full.zip", cwd=str(d))


def idnet(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="cactuslab/IDNet-2025", repo_type="dataset",
                      local_dir=str(d), max_workers=8)


def midv500(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run('lftp -e "set net:timeout 30; set net:max-retries 3; mirror -c /midv-500 .; quit" smartengines.com', cwd=str(d))


def midv2019(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run('lftp -e "set net:timeout 30; set net:max-retries 3; mirror -c /midv-500/extra/midv-2019 .; quit" smartengines.com', cwd=str(d))


def midv2020(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run('lftp -e "set net:timeout 30; set net:max-retries 3; mirror -c /midv-2020 .; quit" smartengines.com', cwd=str(d))


# Kaggle-hosted (same CDN as the official archive) — deferred to the END so they don't
# starve the official competition download.
def doctamper(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    _run("kaggle datasets download -d dinmkeljiame/doctamper -p . --unzip", cwd=str(d))


# priority order: forgery-labeled & moderate-size first; giants + genuine corpora last
PIPELINE = [
    ("fantasyid", fantasyid),   # 2.6GB  face-swap+text-inpaint (most on-task)
    ("dlc2021", dlc2021),       # print/screen recapture
    ("midv_holo", midv_holo),   # hologram forgery
    ("docxpand", docxpand),     # 18GB synthetic IDs
    ("bid", bid),               # Brazilian IDs
    ("idnet", idnet),           # 125GB synthetic fraud (HF)
    ("midv500", midv500),       # 22GB genuine
    ("midv2019", midv2019),     # 9GB genuine
    ("midv2020", midv2020),     # 124GB genuine (lowest priority)
    ("doctamper", doctamper),   # 21GB Kaggle (LMDB content password-gated)
]


def main():
    only = sys.argv[1:] or None
    report = Path("docs/aux-download-report.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in PIPELINE:
        if only and name not in only:
            continue
        if _done(name).exists():
            print(f"[SKIP] {name} (already done)", flush=True)
            continue
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        try:
            fn(ROOT / name)
            _done(name).write_text("ok")
            msg = f"[OK]   {name} in {int(time.time()-t0)}s"
        except Exception as e:
            msg = f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}"
        print(msg, flush=True)
        with open(report, "a") as f:
            f.write(msg + "\n")
    print("\n=== aux download pipeline finished ===", flush=True)


if __name__ == "__main__":
    main()
