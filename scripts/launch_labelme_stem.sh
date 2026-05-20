#!/usr/bin/env bash
# labelme — 줄기(stem) 폴리곤 수동 라벨링·수정
#
# LabelImg는 박스만 가능 → seg(줄기)는 labelme 사용.
#
# 사용법:
#   bash scripts/launch_labelme_stem.sh realscene   # 7장 (먼저 권장)
#   bash scripts/launch_labelme_stem.sh val         # val 37장
#   bash scripts/launch_labelme_stem.sh sample      # qin 20장 샘플
#
# 작업:
#   1) 폴리곤으로 줄기 따라 그리기, 라벨 이름: stem
#   2) 저장(Ctrl+S) → 같은 폴더에 <이미지>.json 생성
#   3) 종료 후 변환:
#        python scripts/labelme_to_yolo_stem.py --json-dir datasets/yolo_unified_seg/_labelme/realscene
#
# YOLO seg 라벨 저장 위치: datasets/yolo_unified_seg/labels/all/<stem>.txt

set -e
cd "$(dirname "$0")/.."

MODE="${1:-realscene}"
SEG_ROOT="$(pwd)/datasets/yolo_unified_seg"
IMG_ALL="${SEG_ROOT}/images/all"
WORK="${SEG_ROOT}/_labelme/${MODE}"

if ! command -v labelme >/dev/null 2>&1; then
  echo "[INFO] labelme 설치: pip install labelme"
  echo "       또는: pip install 'labelme>=5.0'"
  exit 1
fi

mkdir -p "${WORK}"
rm -f "${WORK}"/*

copy_subset() {
  local pattern="$1"
  python3 - "${IMG_ALL}" "${pattern}" "${WORK}" <<'PY'
import glob, os, shutil, sys
from pathlib import Path
src, pat, dst = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(glob.glob(os.path.join(src, pat)))
if not paths:
    raise SystemExit(f"매칭 없음: {pat}")
for p in paths:
    shutil.copy2(p, dst)
print(len(paths), file=sys.stderr)
PY
}

case "${MODE}" in
  realscene)
    copy_subset "realscene_*"
    echo "[INFO] realscene 7장 → ${WORK}" >&2
    ;;
  val)
    VAL_IMG="${SEG_ROOT}/images/val"
    if [ ! -d "${VAL_IMG}" ]; then
      echo "[ERROR] val 분할 없음. python scripts/split_dataset.py --root datasets/yolo_unified_seg"
      exit 1
    fi
    for f in "${VAL_IMG}"/*; do
      [ -f "$f" ] && cp -f "$f" "${WORK}/"
    done
    echo "[INFO] val 이미지 → ${WORK}" >&2
    ;;
  sample)
    copy_subset "qin_0000[0-9].jpg"
    copy_subset "qin_0001[0-9].jpg"
    echo "[INFO] qin 샘플 → ${WORK}" >&2
    ;;
  *)
    echo "사용: realscene | val | sample"
    exit 1
    ;;
esac

echo ""
echo "labelme 실행 중. 폴리곤 라벨 이름은 반드시: stem"
echo "저장 폴더: ${WORK}"
echo ""

labelme "${WORK}" --labels stem --nodata
