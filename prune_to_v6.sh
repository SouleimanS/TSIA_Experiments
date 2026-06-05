#!/bin/bash
# Prune the TSIA repo down to what's needed to study v6+.
#
# DRY RUN by default: prints every path it would remove and the total space
# reclaimed, but deletes NOTHING. Re-run with --apply to actually delete.
#
#   bash prune_to_v6.sh           # preview
#   bash prune_to_v6.sh --apply   # delete
#
# Decisions baked in (per the keep/prune triage):
#   KEEP  : av_ib/av_ib (v6+ package), scripts/qsub_{v6,pilot,probe,baseline}*,
#           runs/*/log.jsonl, BOTH old v6 checkpoints (v6_full_b, v6_tier1_b),
#           environment.yml, READMEs/MANIFEST/NOTES, inventory scripts.
#   DROP  : __pycache__, graphify-out, *.qsub.log, the v5 day1_locked_in dir,
#           the qwen_omni_* standalone dumps, old *bis*/vanilla eval csvs,
#           the whole avhbench/AVHBench-Align-FT tree, and ALL Phase-A
#           dsink results (results/dsink*, results/validate_dsink_*).

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

# Build the kill list (only paths that exist).
targets=()

# 1. Regenerable / tooling caches
while IFS= read -r d; do targets+=("$d"); done < <(find . -type d -name '__pycache__' 2>/dev/null)
[ -d ./av_ib/graphify-out ] && targets+=("./av_ib/graphify-out")

# 2. Verbose stdout logs (the jsonl in runs/ holds the real data)
while IFS= read -r f; do targets+=("$f"); done < <(find ./av_ib -maxdepth 1 -name '*.qsub.log' 2>/dev/null)

# 3. v5-era artifacts
[ -d ./av_ib/results/day1_locked_in ] && targets+=("./av_ib/results/day1_locked_in")
for f in ./av_ib/qwen_omni_baseline.json ./av_ib/qwen_omni_full.json \
         ./av_ib/qwen_omni_smoke.json ./av_ib/qwen_omni_full.csv \
         ./av_ib/qwen_omni_full_reparsed.csv; do
    [ -e "$f" ] && targets+=("$f")
done
# old naming (bis / vanilla) eval csvs
while IFS= read -r f; do targets+=("$f"); done < <(find ./av_ib/results/eval -type f \( -name '*bis*' -o -name '*vanilla*' \) 2>/dev/null)

# 4. AVHBench-Align-FT reference repo (decision: drop the tree)
[ -d ./avhbench/AVHBench-Align-FT ] && targets+=("./avhbench/AVHBench-Align-FT")

# 5. Phase-A dsink results (decision: drop all)
for d in ./av_ib/results/dsink ./av_ib/results/dsink_audio ./av_ib/results/dsink_imstart; do
    [ -d "$d" ] && targets+=("$d")
done
while IFS= read -r d; do targets+=("$d"); done < <(find ./av_ib/results -maxdepth 1 -type d -name 'validate_dsink_*' 2>/dev/null)

# --- Report + (optionally) delete ---
echo "############################################################"
echo "# Prune to v6+   ($([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN))"
echo "# root: $ROOT"
echo "############################################################"
echo

if [ ${#targets[@]} -eq 0 ]; then
    echo "Nothing to prune (already clean)."
    exit 0
fi

total_kb=0
for t in "${targets[@]}"; do
    sz=$(du -sk "$t" 2>/dev/null | cut -f1)
    sz=${sz:-0}
    total_kb=$((total_kb + sz))
    hsz=$(du -sh "$t" 2>/dev/null | cut -f1)
    printf "  %6s  %s\n" "$hsz" "$t"
done

echo
printf "  Total reclaimed: %s MB across %d paths\n" "$((total_kb / 1024))" "${#targets[@]}"
echo

if [ $APPLY -eq 1 ]; then
    echo "Deleting..."
    for t in "${targets[@]}"; do rm -rf "$t"; done
    echo "Done. Re-run inventory_repo.sh to confirm."
else
    echo "DRY RUN — nothing deleted. Re-run with:  bash prune_to_v6.sh --apply"
fi
