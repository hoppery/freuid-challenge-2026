"""Fetch a small balanced subset of FREUID images to run a smoke test before the
full 16.3 GB archive finishes downloading. Uses single-file Kaggle API downloads."""
from __future__ import annotations
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

os.environ.setdefault(
    "KAGGLE_API_TOKEN", open(os.path.expanduser("~/.kaggle/access_token")).read().strip())
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

COMP = "the-freuid-challenge-2026-ijcai-ecai"
ROOT = Path("data/raw/freuid")
PER_TYPE_PER_LABEL = 16      # 5 types x 2 labels x 16 = 160 train images
N_TEST = 50

api = KaggleApi()
api.authenticate()


def fetch_one(remote_rel: str, _dest_dir: Path) -> tuple[str, bool]:
    """Download `remote_rel` and place it at ROOT/remote_rel exactly."""
    final = ROOT / remote_rel
    if final.exists():
        return remote_rel, True
    final.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory() as td:
                api.competition_download_file(COMP, remote_rel, path=td, quiet=True)
                found = list(Path(td).rglob("*"))
                files = [p for p in found if p.is_file()]
                if not files:
                    raise RuntimeError("no file downloaded")
                src = files[0]
                if src.suffix == ".zip":  # unwrap if Kaggle zipped a single file
                    shutil.unpack_archive(str(src), td)
                    files = [p for p in Path(td).rglob("*") if p.is_file() and p.suffix != ".zip"]
                    src = files[0]
                shutil.move(str(src), str(final))
            return remote_rel, True
        except Exception as e:
            time.sleep(2 * (attempt + 1) if "429" in str(e) else 0.5)
    return remote_rel, False


def main():
    src_csv = ROOT / "train_labels.csv"
    if not src_csv.exists():
        shutil.copy("/tmp/probe/train_labels.csv", src_csv)
        shutil.copy("/tmp/probe/sample_submission.csv", ROOT / "sample_submission.csv")
    labels = pd.read_csv(src_csv)

    # balanced subset across (type, label)
    picks = []
    for (t, lb), g in labels.groupby(["type", "label"]):
        picks.append(g.head(PER_TYPE_PER_LABEL))
    sub = pd.concat(picks).reset_index(drop=True)
    print(f"selected {len(sub)} train images across "
          f"{sub['type'].nunique()} types, labels={dict(sub['label'].value_counts())}")

    # train images: remote path is the 'image_path' column (train/<id>.jpeg)
    train_targets = [(r.image_path, ROOT / "train") for r in sub.itertuples()]
    # test images: first N from sample_submission
    ssub = pd.read_csv(ROOT / "sample_submission.csv").head(N_TEST)
    test_targets = [(f"public_test/public_test/{i}.jpeg",
                     ROOT / "public_test" / "public_test") for i in ssub["id"]]

    all_targets = train_targets + test_targets
    ok = 0
    fail = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, rel, d): rel for rel, d in all_targets}
        for f in as_completed(futs):
            rel, good = f.result()
            ok += good
            if not good:
                fail.append(rel)
    print(f"downloaded ok={ok}/{len(all_targets)} fail={len(fail)}")
    if fail:
        print("failed:", fail[:10])
    # flatten any nested 'train/' the API may create under dest
    print("train imgs on disk:", len(list((ROOT / 'train').glob('*.jpeg'))))
    print("test imgs on disk:",
          len(list((ROOT / 'public_test' / 'public_test').glob('*.jpeg'))))


if __name__ == "__main__":
    main()
