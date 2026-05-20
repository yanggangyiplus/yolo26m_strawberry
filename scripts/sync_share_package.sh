#!/usr/bin/env bash
# RealSense·stem 스크립트를 share/strawberry_yolo26m_unified/ 에 동기화
# (predict.py · review_stem_pipeline.py 는 share 전용 — 이 스크립트로 덮어쓰지 않음)
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
SHR="$SRC/share/strawberry_yolo26m_unified"

cp "$SRC/scripts/realsense_live.py" "$SHR/"
cp "$SRC/scripts/realsense_stem_pipeline.py" "$SHR/"
cp "$SRC/scripts/stem_roi_utils.py" "$SHR/"
rm -rf "$SHR/__pycache__"
echo "[OK] share RealSense/stem → $SHR"
echo "[INFO] predict.py, review_stem_pipeline.py 는 share 폴더에서 별도 관리"
