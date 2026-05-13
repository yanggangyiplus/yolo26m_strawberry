#!/usr/bin/env bash
# LabelImg 실행 스크립트.
#
# 사용법:
#   bash scripts/launch_labelimg.sh <모드>
#
# 모드 목록:
#   unified         ★ 권장 — 전체 통합 데이터셋 (qin+uniq+3folders+신규 realscene, 934장)
#   realscene       신규 실제 장면 사진 7장 (realscene_*.png) — 가장 먼저 작업 권장
#   pickable        strawberryDataset/Pickable (straw_pick_*, 264장)
#   unpickable      strawberryDataset/UnPickable (straw_unpick_*, 259장)
#   qin             Qin2006 기존 라벨 검토 (qin_*, 325장)
#   uniq            UniqueData 기존 라벨 검토 (uniq_*, 40장)
#   user            raw_str + red_str 모형 사진 (user_raw/red_*, 39장)
#
# 저장 위치: datasets/yolo_unified/labels/all/ (YOLO 포맷 .txt)
# 라벨 없는 이미지도 표시됨 — 직접 W키로 박스 그리기

set -e
cd "$(dirname "$0")/.."

# 이전 세션의 last_open_dir 설정이 남아 있으면 올바른 경로를 무시하므로 매번 초기화
rm -f ~/.labelImgSettings.pkl

ROOT="$(pwd)/datasets/yolo_unified"
IMG_ALL="${ROOT}/images/all"
LBL_DIR="${ROOT}/labels/all"
CLASSES="${LBL_DIR}/classes.txt"

if [ ! -f "${CLASSES}" ]; then
  echo "[ERROR] classes.txt 없음: ${CLASSES}"
  exit 1
fi

MODE="${1:-unified}"

# 부분 모드: 임시 폴더에 이미지 실제 복사 (심볼릭 링크는 labelImg가 인식 못할 수 있음)
# RGBA/P 등 비-RGB 이미지는 RGB로 변환해서 복사
# ★ stdout에는 경로만 출력, 모든 INFO는 stderr(>&2)로 보내 command substitution에서 제외
make_subset() {
  local name="$1"
  local pattern="$2"
  local tmp="${ROOT}/_label_${name}"
  rm -rf "${tmp}" && mkdir -p "${tmp}"
  python3 - "${IMG_ALL}" "${pattern}" "${tmp}" <<'PYEOF'
import sys, os
from PIL import Image
src_dir, pattern, dst_dir = sys.argv[1], sys.argv[2], sys.argv[3]
import glob
files = sorted(glob.glob(os.path.join(src_dir, pattern)))
for sp in files:
    fname = os.path.basename(sp)
    dp = os.path.join(dst_dir, fname)
    try:
        img = Image.open(sp)
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P': img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dp)
    except Exception as e:
        print(f"[WARN] {fname}: {e}", file=sys.stderr)
PYEOF
  local cnt
  cnt=$(ls "${tmp}" | wc -l)
  echo "[INFO] ${name} 임시 폴더: ${cnt}장" >&2
  echo "${tmp}"   # stdout에는 경로만
}

case "${MODE}" in
  unified)
    # unified 전체: RGBA 이미지가 있으면 labelImg가 건너뜀 → RGB 변환본을 임시 폴더에 준비
    echo "[INFO] unified 모드: 이미지 RGB 변환 중 (최초 1회만 수행)..."
    UTMP="${ROOT}/_label_unified"
    mkdir -p "${UTMP}"
    python3 - "${IMG_ALL}" "${UTMP}" <<'PYEOF3'
import sys, os, glob
from PIL import Image
src_dir, dst_dir = sys.argv[1], sys.argv[2]
existing = set(os.listdir(dst_dir))
files = sorted(glob.glob(os.path.join(src_dir, "*")))
for sp in files:
    fname = os.path.basename(sp)
    if fname in existing: continue  # 이미 변환됨
    dp = os.path.join(dst_dir, fname)
    try:
        img = Image.open(sp)
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P': img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dp)
    except Exception as e:
        print(f"[WARN] {fname}: {e}", file=sys.stderr)
print(f"총 {len(os.listdir(dst_dir))}장 준비 완료")
PYEOF3
    TARGET_IMG="${UTMP}"
    ;;
  realscene)
    TARGET_IMG=$(make_subset "realscene" "realscene_*")
    ;;
  pickable)
    TARGET_IMG=$(make_subset "pickable" "straw_pick_*")
    ;;
  unpickable)
    TARGET_IMG=$(make_subset "unpickable" "straw_unpick_*")
    ;;
  qin)
    TARGET_IMG=$(make_subset "qin" "qin_*")
    ;;
  uniq)
    TARGET_IMG=$(make_subset "uniq" "uniq_*")
    ;;
  user)
    TARGET_IMG=$(make_subset "user_raw" "user_raw_*")
    # user_red도 같은 tmp 폴더에 추가
    python3 - "${IMG_ALL}" "user_red_*" "${ROOT}/_label_user_raw" <<'PYEOF2'
import sys, os, glob
from PIL import Image
src_dir, pattern, dst_dir = sys.argv[1], sys.argv[2], sys.argv[3]
for sp in sorted(glob.glob(os.path.join(src_dir, pattern))):
    fname = os.path.basename(sp)
    dp = os.path.join(dst_dir, fname)
    try:
        img = Image.open(sp)
        if img.mode != 'RGB': img = img.convert('RGB')
        img.save(dp)
    except Exception as e:
        print(f"[WARN] {fname}: {e}", file=sys.stderr)
PYEOF2
    TARGET_IMG="${ROOT}/_label_user_raw"
    ;;
  *)
    echo "[ERROR] 알 수 없는 모드: ${MODE}"
    echo "사용가능: unified | realscene | pickable | unpickable | qin | uniq | user"
    exit 1
    ;;
esac

echo ""
echo "[INFO] mode    : ${MODE}"
echo "[INFO] img dir : ${TARGET_IMG}"
echo "[INFO] labels  : ${LBL_DIR}"
echo "[INFO] classes : $(tr '\n' '  ' < "${CLASSES}")"
echo ""
echo "단축키: W=박스그리기  D=다음  A=이전  Del=박스삭제  Ctrl+S=저장"
echo "        View > Auto Save Mode 켜기 권장"
echo ""

labelImg "${TARGET_IMG}" "${CLASSES}" "${LBL_DIR}"
