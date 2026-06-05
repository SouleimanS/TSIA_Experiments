#!/bin/bash
# Repo inventory for deciding what to keep for the v6+ study.
#
# Prints, to stdout (redirect to a file and paste it back):
#   1. Top-level layout + per-directory file counts (so huge data dirs are
#      visible as a single number, not thousands of lines).
#   2. Every CODE / config / script / doc file in full (these are the things
#      we actually reason about).
#   3. Data / media directories collapsed to: count + extension breakdown +
#      a HEAD of a few sample names (so we see the format, not 10k video ids).
#
# Run from the repo root:   bash inventory_repo.sh > repo_inventory.txt 2>&1

set -uo pipefail
cd "$(dirname "$0")"

HEAD_N=8          # how many sample filenames to show per data dir
ROOT="$(pwd)"

echo "############################################################"
echo "# TSIA repo inventory"
echo "# root: $ROOT"
echo "# date: $(date)"
echo "# git:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "############################################################"

# ---------------------------------------------------------------------------
# 1. Directory layout with file counts (depth-limited, sorted)
# ---------------------------------------------------------------------------
echo
echo "==================== DIRECTORY FILE COUNTS ===================="
echo "(files counted recursively per directory; .git excluded)"
echo
find . -path ./.git -prune -o -type d -print 2>/dev/null | sort | while read -r d; do
    n=$(find "$d" -path ./.git -prune -o -type f -print 2>/dev/null | wc -l)
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf "%8s files  %6s  %s\n" "$n" "$size" "$d"
done

# ---------------------------------------------------------------------------
# 2. Code / config / docs — list in full (these are study-relevant)
# ---------------------------------------------------------------------------
echo
echo "==================== CODE / CONFIG / DOCS (full list) ===================="
echo "(extensions: py sh yaml yml json toml cfg ini md txt + Makefile/Dockerfile)"
echo
find . -path ./.git -prune -o -type f \( \
        -name '*.py'   -o -name '*.sh'   -o -name '*.yaml' -o -name '*.yml' \
     -o -name '*.json' -o -name '*.toml' -o -name '*.cfg'  -o -name '*.ini'  \
     -o -name '*.md'   -o -name '*.txt'  -o -name '*.cu'   -o -name '*.cpp'  \
     -o -name 'Makefile' -o -name 'Dockerfile' -o -name 'requirements*' \
     \) -print 2>/dev/null \
  | grep -v -E '/(node_modules|__pycache__|\.ipynb_checkpoints)/' \
  | sort | while read -r f; do
        lines=$(wc -l < "$f" 2>/dev/null)
        printf "%7s L  %s\n" "$lines" "$f"
    done

# ---------------------------------------------------------------------------
# 3. Everything else (data / media / checkpoints) — collapsed summaries
# ---------------------------------------------------------------------------
echo
echo "==================== DATA / MEDIA / OTHER (collapsed) ===================="
echo "(per directory holding non-code files: count, extension histogram,"
echo " and the first $HEAD_N filenames as samples)"
echo

# Directories that contain at least one non-code file
find . -path ./.git -prune -o -type f \
     ! \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.json' -o -name '*.toml' -o -name '*.cfg' -o -name '*.ini' \
       -o -name '*.md' -o -name '*.txt' -o -name '*.cu' -o -name '*.cpp' \
       -o -name 'Makefile' -o -name 'Dockerfile' -o -name 'requirements*' \) \
     -print 2>/dev/null \
  | grep -v -E '/(node_modules|__pycache__|\.ipynb_checkpoints)/' \
  | sed 's#/[^/]*$##' | sort -u | while read -r d; do
        cnt=$(find "$d" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "---- $d  ($cnt files) ----"
        echo "  extensions:"
        find "$d" -maxdepth 1 -type f 2>/dev/null \
          | sed -E 's#.*/[^/]*\.([^./]+)$#\1#; t; s#.*#<no-ext>#' \
          | sort | uniq -c | sort -rn | sed 's/^/    /'
        echo "  sample names (head -$HEAD_N):"
        find "$d" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null \
          | sort | head -n "$HEAD_N" | sed 's/^/    /'
        echo
    done

# ---------------------------------------------------------------------------
# 4. Largest files (so big checkpoints/datasets are obvious)
# ---------------------------------------------------------------------------
echo
echo "==================== 25 LARGEST FILES ===================="
find . -path ./.git -prune -o -type f -print 2>/dev/null \
  | xargs -d '\n' du -h 2>/dev/null | sort -rh | head -25

echo
echo "==================== END INVENTORY ===================="
