#!/bin/bash
# Collect all available results into meeting/ for today's presentation.
# Run from the repo root on the cluster:
#   bash scripts/meeting_materials.sh
set -euo pipefail
PYTHON=/home/aab11336im/anaconda3/envs/av_ib/bin/python3
OUT=meeting
mkdir -p "$OUT"

echo "========================================"
echo "  Meeting materials — $(date '+%Y-%m-%d')"
echo "========================================"

# ── 1. IB attention (with vs without bottleneck) ─────────────────────────────
IB_DIR=results/ib_attention
if [ -d "$IB_DIR" ] && ls "$IB_DIR"/*_ib_metrics.json &>/dev/null; then
    echo ""
    echo "── IB Attention (with vs without bottleneck) ──"
    $PYTHON - <<'PYEOF'
import json, glob, statistics, os
files = sorted(glob.glob("results/ib_attention/*_ib_metrics.json"))
de = [json.load(open(f))["delta_norm_entropy"] for f in files]
dm = [json.load(open(f))["delta_top5pct_mass"]  for f in files]
print(f"  N clips          : {len(files)}")
print(f"  mean Δentropy    : {statistics.mean(de):+.4f}  (negative = IB more localized)")
print(f"  mean Δmass       : {statistics.mean(dm):+.4f}  (positive = IB more localized)")
print(f"  IB wins entropy  : {sum(d<0 for d in de)}/{len(de)} clips")
print(f"  IB wins mass     : {sum(d>0 for d in dm)}/{len(dm)} clips")
# pick 4 most illustrative: largest |Δmass|
ranked = sorted(zip(dm, files), key=lambda x: abs(x[0]), reverse=True)
print(f"\n  Top clips by |Δmass|:")
for dm_val, f in ranked[:4]:
    data = json.load(open(f))
    vid = data["video"]
    print(f"    {vid:40s}  Δmass={dm_val:+.3f}  Δentropy={data['delta_norm_entropy']:+.3f}")
    # write clip name to file for bash to copy PNG
    with open("/tmp/meeting_ib_picks.txt", "a") as out:
        out.write(vid + "\n")
PYEOF

    # copy the picked PNGs
    mkdir -p "$OUT/ib_attention"
    if [ -f /tmp/meeting_ib_picks.txt ]; then
        while IFS= read -r vid; do
            cp "$IB_DIR/${vid}_ib_compare.png" "$OUT/ib_attention/" 2>/dev/null && \
                echo "  copied ${vid}_ib_compare.png" || true
        done < /tmp/meeting_ib_picks.txt
        rm -f /tmp/meeting_ib_picks.txt
    fi
    # also copy all JSON for the table
    cp "$IB_DIR"/*_ib_metrics.json "$OUT/ib_attention/" 2>/dev/null || true
    echo "  → figures in $OUT/ib_attention/"
else
    echo ""
    echo "  [IB attention] No results yet (job still running?)"
fi

# ── 2. Gate training ──────────────────────────────────────────────────────────
echo ""
echo "── Modality gate training ──"
GATE_DIR=runs/gate
if [ -f "$GATE_DIR/gate_final.pt" ]; then
    echo "  Checkpoint : $GATE_DIR/gate_final.pt  ($(du -sh $GATE_DIR/gate_final.pt | cut -f1))"
    # extract summary from PBS log
    if [ -f gate_train.qsub.log ]; then
        echo "  Training summary:"
        grep -E "Training complete|final_loss|num_steps|elapsed" gate_train.qsub.log | \
            sed 's/^/    /' || true
        echo "  Gate probe (if gate_probe.qsub.log exists):"
        grep -E "video=|audio=|text=" gate_probe.qsub.log 2>/dev/null | sed 's/^/    /' || \
            echo "    (probe not yet run — submit scripts above)"
    fi
    mkdir -p "$OUT/gate"
    [ -f gate_train.qsub.log ] && cp gate_train.qsub.log "$OUT/gate/" || true
    [ -f gate_probe.qsub.log ] && cp gate_probe.qsub.log "$OUT/gate/" || true
else
    echo "  [gate] Checkpoint not found yet"
fi

# ── 3. Perceive-then-reason ───────────────────────────────────────────────────
echo ""
echo "── Perceive-then-reason training ──"
PERC_DIR=runs/perceive
if [ -f "$PERC_DIR/perceive_final.pt" ]; then
    echo "  Checkpoint : $PERC_DIR/perceive_final.pt"
    grep -E "Training complete|final_loss" perceive_train.qsub.log 2>/dev/null | sed 's/^/  /' || true
    mkdir -p "$OUT/perceive"
elif [ -f "$PERC_DIR/perceive_train_out.txt" ]; then
    echo "  Training in progress:"
    tail -5 "$PERC_DIR/perceive_train_out.txt" | sed 's/^/    /'
else
    echo "  [perceive] Job still running — no output yet"
fi

# ── 4. Perceive attention maps ────────────────────────────────────────────────
PERC_ATT=results/perceive_attention
if [ -d "$PERC_ATT" ] && ls "$PERC_ATT"/*.png &>/dev/null; then
    echo ""
    echo "── Perceive attention maps ──"
    mkdir -p "$OUT/perceive_attention"
    cp "$PERC_ATT"/*.png "$OUT/perceive_attention/" 2>/dev/null || true
    cp "$PERC_ATT"/*.json "$OUT/perceive_attention/" 2>/dev/null || true
    echo "  $(ls $PERC_ATT/*.png | wc -l) figures → $OUT/perceive_attention/"
fi

# ── Summary card ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Files ready in: $(pwd)/$OUT/"
ls "$OUT"/
echo "========================================"
echo ""
echo "Quick scp to laptop:"
echo "  scp -r $(hostname):$(pwd)/$OUT/ ~/Desktop/meeting_$(date '+%Y%m%d')/"
