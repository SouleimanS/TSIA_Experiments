#!/bin/bash
set -euo pipefail

LIST_FILE=${LIST_FILE:-runs/v6_tier1_b/eval_videos.txt}
PAIRS_FILE=${PAIRS_FILE:-runs/v6_tier1_b/swap_pairs.json}
VIDEO_ROOT=${VIDEO_ROOT:-$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all}
OUT_ROOT=${OUT_ROOT:-$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/corrupted_v6_1k}
JOBS=${JOBS:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAKE="$SCRIPT_DIR/make_corrupted.sh"
export VIDEO_ROOT OUT_ROOT MAKE

mkdir -p "$OUT_ROOT"/{mute,shift,swap}

echo "List:    $LIST_FILE  ($(wc -l < $LIST_FILE) videos)"
echo "Out:     $OUT_ROOT"
echo "Jobs:    $JOBS"

JOBS_FILE=$(mktemp)
python3 -c "
import json
pairs = json.load(open('$PAIRS_FILE'))
for vid in [l.strip() for l in open('$LIST_FILE') if l.strip()]:
    print(f'{vid} $VIDEO_ROOT/{pairs[vid]}.mp4')
" > "$JOBS_FILE"

echo "Starting $(wc -l < $JOBS_FILE) jobs at $(date)"
START=$(date +%s)

xargs -P "$JOBS" -n 2 bash -c '"$MAKE" "$1" "$VIDEO_ROOT" "$OUT_ROOT" "$2" > /dev/null' _ < "$JOBS_FILE"

echo "Done at $(date), elapsed $(( $(date +%s) - START ))s"
for kind in mute shift swap; do
    n=$(ls "$OUT_ROOT/$kind/" 2>/dev/null | wc -l)
    echo "  $kind: $n files"
done
rm -f "$JOBS_FILE"
