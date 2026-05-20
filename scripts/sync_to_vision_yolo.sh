#!/usr/bin/env bash
# yolo_strawberry → harvesting_robot_miniproj/vision_yolo 동기화
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${VISION_YOLO_DST:-/home/user01/harvesting_robot_miniproj/vision_yolo}"

if [[ ! -d "$DST" ]]; then
  echo "[ERROR] 대상 없음: $DST" >&2
  exit 1
fi

echo "[INFO] $SRC → $DST"

rsync -av --delete --exclude='__pycache__' --exclude='archive' \
  "$SRC/scripts/" "$DST/scripts/"

rsync -av "$SRC/configs/" "$DST/configs/"
rsync -av "$SRC/share/strawberry_yolo26m_unified/" "$DST/share/strawberry_yolo26m_unified/"

for d in yolo_stem_roi yolo_unified_seg yolo_unified_farm; do
  mkdir -p "$DST/datasets/$d/labels/all"
  rsync -av "$SRC/datasets/$d/labels/all/" "$DST/datasets/$d/labels/all/"
done
[[ -f "$SRC/datasets/yolo_stem_roi/roi_meta.jsonl" ]] && \
  cp "$SRC/datasets/yolo_stem_roi/roi_meta.jsonl" "$DST/datasets/yolo_stem_roi/"

cp "$SRC/.gitignore" "$SRC/requirements.txt" "$DST/"
sed 's|yolo_strawberry/|vision_yolo/|g' "$SRC/README.md" > "$DST/README.md"

rm -rf "$DST/share/strawberry_yolo26m_unified/__pycache__" 2>/dev/null || true

echo "[OK] 동기화 완료"
