#!/usr/bin/env python3
"""Extract the AVQA clip subset from the HuggingFace VGGSound mirror.

The dataset `Loie/VGGSound` packages ~199k VGGSound clips as ~20 tarballs of
~16 GB each (~338 GB total). AVQA needs only ~57k of those clips (~100 GB),
so this script streams each tarball and extracts only the members whose clip
ID appears in the AVQA annotations, then deletes nothing it didn't create.

Workflow (login node, needs internet + ~40 GB scratch for one tar at a time):
    pip install huggingface_hub
    python extract_avqa_from_vggsound.py \
        --out ~/SOULEIMAN_repo/datasets/AVQA --scratch /tmp/vggsound_tars

Remaining clips (absent from the mirror) are listed in missing_clips.txt;
fetch those with download_avqa.py (yt-dlp fallback), or accept the holes —
AVQADataset(skip_missing=True) tolerates them.
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path

ANN_URLS = {
    "train": "https://raw.githubusercontent.com/AlyssaYoung/AVQA/main/data/annotation/train_qa.json",
    "val":   "https://raw.githubusercontent.com/AlyssaYoung/AVQA/main/data/annotation/val_qa.json",
}
HF_REPO = "Loie/VGGSound"


def fetch_annotations(out_dir: Path) -> set[str]:
    """Download annotation JSONs; return the set of needed video_names."""
    needed = set()
    for split, url in ANN_URLS.items():
        dst = out_dir / f"{split}_qa.json"
        if not dst.exists():
            print(f"downloading {split} annotations ...")
            urllib.request.urlretrieve(url, dst)
        recs = json.loads(dst.read_text())
        needed.update(r["video_name"] for r in recs)
        print(f"  {split}: {len(recs)} QA pairs")
    print(f"unique clips needed: {len(needed)}")
    return needed


def norm_key(name: str) -> str | None:
    """Normalize a clip filename to '<ytid>_<start-seconds-int>'.

    Handles both AVQA naming (<ytid>_000030) and common VGGSound tar naming
    variants (<ytid>_30.mp4, <ytid>_000030.mp4, <ytid>_30000_40000.mp4 [ms]).
    """
    stem = Path(name).stem
    m = re.match(r"^(.{11})_(\d+)(?:_(\d+))?$", stem)
    if not m:
        return None
    ytid, a, b = m.group(1), int(m.group(2)), m.group(3)
    if b is not None and a % 1000 == 0:  # millisecond ranges
        a = a // 1000
    return f"{ytid}_{a}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True,
                    help="temp dir for one tarball at a time (~20 GB)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out).expanduser()
    video_dir = out / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch).expanduser()
    scratch.mkdir(parents=True, exist_ok=True)

    needed_names = fetch_annotations(out)
    # normalized key -> canonical AVQA video_name (used as output filename)
    needed = {norm_key(n): n for n in needed_names}
    needed.pop(None, None)

    have = {norm_key(p.name) for p in video_dir.glob("*.mp4")}
    print(f"already extracted: {len(have)}")

    api = HfApi()
    tars = sorted(f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
                  if f.endswith((".tar.gz", ".tar")))
    print(f"{len(tars)} tarballs in {HF_REPO}")

    def extract_tar(local: str) -> int:
        mode = "r:gz" if local.endswith(".gz") else "r"
        n = 0
        with tarfile.open(local, mode) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key = norm_key(m.name)
                if key in needed and key not in have:
                    dst = video_dir / f"{needed[key]}.mp4"
                    with tf.extractfile(m) as src, open(dst, "wb") as f:
                        f.write(src.read())
                    have.add(key)
                    n += 1
        return n

    n_extracted = 0
    failed_tars = []
    for t in tars:
        if len(have) >= len(needed):
            break
        print(f"--- {t} ---", flush=True)
        # A cached tarball can be truncated/corrupt (interrupted download):
        # on decompression error, delete it, force a fresh download, retry once.
        ok = False
        for attempt in range(2):
            local = hf_hub_download(
                HF_REPO, t, repo_type="dataset", local_dir=scratch,
                force_download=(attempt > 0))
            try:
                n_extracted += extract_tar(local)
                ok = True
                break
            except Exception as e:  # zlib.error, tarfile.TarError, EOFError…
                print(f"  CORRUPT ({type(e).__name__}: {e}) — "
                      f"{'re-downloading' if attempt == 0 else 'giving up'}",
                      flush=True)
                Path(local).unlink(missing_ok=True)
        if not ok:
            failed_tars.append(t)
        else:
            Path(local).unlink(missing_ok=True)  # free scratch
        print(f"  extracted so far: {n_extracted} "
              f"({len(have)}/{len(needed)} needed clips present)", flush=True)

    if failed_tars:
        print(f"WARNING: {len(failed_tars)} tarball(s) skipped after retry: "
              f"{failed_tars}")

    missing = sorted(needed[k] for k in set(needed) - have)
    (out / "missing_clips.txt").write_text("\n".join(missing))
    print(f"done. present={len(have)} missing={len(missing)} "
          f"(-> missing_clips.txt; fetch via download_avqa.py or ignore)")


if __name__ == "__main__":
    main()
