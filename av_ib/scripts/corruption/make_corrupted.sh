#!/bin/bash
# Usage: make_corrupted.sh <video_id> <video_root> <out_root> <swap_source.mp4>
# Produces three corrupted .mp4s:
#   <out_root>/mute/<video_id>.mp4
#   <out_root>/shift/<video_id>.mp4
#   <out_root>/swap/<video_id>.mp4
set -euo pipefail
VID=$1
VIDEO_ROOT=$2
OUT_ROOT=$3
SWAP_SRC=$4
SRC="$VIDEO_ROOT/$VID.mp4"

mkdir -p "$OUT_ROOT"/{mute,shift,swap}

# Get original duration for length-matching swap
DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$SRC")

# --- MUTE: replace audio with silence of same length ---
ffmpeg -y -v error -i "$SRC" \
    -f lavfi -i "anullsrc=channel_layout=stereo:sample_rate=16000" \
    -map 0:v -map 1:a -c:v copy -c:a aac -shortest \
    "$OUT_ROOT/mute/$VID.mp4"

# --- SHIFT: delay audio by 1s ---
ffmpeg -y -v error -i "$SRC" \
    -filter_complex "[0:a]adelay=1000|1000,apad,atrim=duration=$DUR[a_out]" \
    -map 0:v -map "[a_out]" -c:v copy -c:a aac \
    "$OUT_ROOT/shift/$VID.mp4"

# --- SWAP: replace audio with audio from SWAP_SRC, length-matched ---
ffmpeg -y -v error -i "$SRC" -i "$SWAP_SRC" \
    -filter_complex "[1:a]apad,atrim=duration=$DUR[a_out]" \
    -map 0:v -map "[a_out]" -c:v copy -c:a aac \
    "$OUT_ROOT/swap/$VID.mp4"

echo "OK: $VID"
