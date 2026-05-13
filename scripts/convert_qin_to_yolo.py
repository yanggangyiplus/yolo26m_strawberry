"""HuggingFace `Qin2006/Strawberry-MM-Straw5` → 통합 YOLO 변환.

원본은 5-class YOLO 라벨 (`normal`, `gray_mold`, `powdery_mildew`, `black_spot`, `overripe`).
통합 스키마(2-class: 0=unripe, 1=ripe)로 매핑:

    normal     → bbox 내 빨강 비율로 자동 분류 (ripe / unripe)
    overripe   → 드롭 (수확 불가능, 학습 노이즈 방지)
    gray_mold, powdery_mildew, black_spot → 드롭 (질병, 픽 대상 아님)

옵션:
    --keep-disease   질병/overripe 박스를 'unripe' (id=0)로 학습 (회피 학습 의도, 권장 X)
    --red-threshold  빨강 비율 임계값 (기본 0.30) — 이 이상이면 ripe, 미만이면 unripe

사용 예:
    python scripts/convert_qin_to_yolo.py \
        --src datasets/raw/mm_straw5/Strawberry_Dataset \
        --dst datasets/yolo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# 통합 스키마 클래스 ID
UNRIPE_CLS = 0
RIPE_CLS = 1

# 원본 Qin2006 클래스 ID
CLS_NORMAL = 0
CLS_GRAY_MOLD = 1
CLS_POWDERY = 2
CLS_BLACK_SPOT = 3
CLS_OVERRIPE = 4
QIN_NAMES = {0: "normal", 1: "gray_mold", 2: "powdery_mildew", 3: "black_spot", 4: "overripe"}


def red_ratio(crop_rgb: np.ndarray) -> float:
    """RGB crop에서 '잘 익은 빨강' 픽셀 비율 (0~1). HSV 변환으로 안정적 판정.

    PIL(RGB)로 받아서 numpy 변환 후, HSV 임계로 빨강 영역만 카운트.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return 0.0

    # RGB → HSV (numpy 자체 구현, OpenCV 의존 없이)
    arr = crop_rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn

    # Hue 계산
    h = np.zeros_like(mx)
    mask_r = (mx == r) & (diff > 0)
    mask_g = (mx == g) & (diff > 0)
    mask_b = (mx == b) & (diff > 0)
    h[mask_r] = (60 * ((g - b) / np.where(diff == 0, 1, diff)) % 360)[mask_r]
    h[mask_g] = (60 * ((b - r) / np.where(diff == 0, 1, diff)) + 120)[mask_g]
    h[mask_b] = (60 * ((r - g) / np.where(diff == 0, 1, diff)) + 240)[mask_b]
    s = np.where(mx == 0, 0, diff / np.where(mx == 0, 1, mx))
    v = mx

    # 빨강 영역: H ∈ [0,15] ∪ [345,360], S≥0.30, V≥0.20
    red_mask = ((h <= 15) | (h >= 345)) & (s >= 0.30) & (v >= 0.20)
    return float(red_mask.mean())


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[int, int, int, int]:
    """YOLO 정규화 좌표 → 이미지 픽셀 좌표 (정수, 클램프 포함)."""
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def safe_stem(name: str, prefix: str, idx: int) -> str:
    """중국어 + 공백 파일명을 안전하게 정규화. 인덱스 기반으로 충돌 방지."""
    # YOLO/Ultralytics가 한국어/중국어 경로를 잘 처리하긴 하지만
    # symlink/PIL 경로 호환성을 위해 ASCII로 변환.
    return f"{prefix}_{idx:05d}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("datasets/raw/mm_straw5/Strawberry_Dataset"),
                    help="Qin2006 데이터셋 루트 (images/, labels/, data.yaml 포함)")
    ap.add_argument("--dst", type=Path, default=Path("datasets/yolo"))
    ap.add_argument("--prefix", type=str, default="qin")
    ap.add_argument("--red-threshold", type=float, default=0.30,
                    help="normal 박스 분류: red_ratio ≥ 임계 → ripe, 미만 → unripe")
    ap.add_argument("--keep-disease", action="store_true",
                    help="질병/overripe 박스를 unripe로 학습 (기본: 드롭)")
    ap.add_argument("--copy-images", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        print(f"[ERROR] images/ or labels/ not found under {src}", file=sys.stderr)
        return 2

    out_images = dst / "images" / "all"
    out_labels = dst / "labels" / "all"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] src       = {src}")
    print(f"[INFO] dst       = {dst}")
    print(f"[INFO] prefix    = {args.prefix}")
    print(f"[INFO] red_threshold = {args.red_threshold}")
    print(f"[INFO] keep_disease  = {args.keep_disease}")

    stats: Counter = Counter()
    # 원본 train/val 폴더를 모두 모아서 all/ 에 쌓고, 이후 split_dataset.py가 재분할
    all_images: list[Path] = []
    for split in ("train", "val"):
        all_images.extend((src / "images" / split).glob("*"))

    idx = 0
    for img_path in sorted(all_images):
        if not img_path.is_file():
            continue
        # 원본 split을 그대로 살리려면 stem 보존하면 되지만, 한자/공백 안전성을 위해 새 ID 부여
        # 라벨은 label-folder의 같은 stem에서 찾는다 (image stem == label stem)
        # 원본 split 결정: 부모 폴더가 train/val
        split = img_path.parent.name
        lbl_path = src / "labels" / split / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            stats["label_missing"] += 1
            continue

        # 이미지를 한 번만 PIL로 열어 W,H 확보 + 필요 시 crop 분석
        try:
            with Image.open(img_path) as im:
                W, H = im.size
                rgb = np.array(im.convert("RGB"))
        except Exception as e:
            stats["image_open_error"] += 1
            print(f"[WARN] cannot open {img_path}: {e}", file=sys.stderr)
            continue

        # 라벨 파싱 + 매핑
        new_lines: list[str] = []
        text = lbl_path.read_text(encoding="utf-8").strip()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                stats["bad_line"] += 1
                continue
            try:
                qin_cls = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                stats["bad_line"] += 1
                continue

            # 매핑 규칙
            if qin_cls == CLS_NORMAL:
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
                if x2 > x1 and y2 > y1:
                    crop = rgb[y1:y2, x1:x2]
                    rr = red_ratio(crop)
                else:
                    rr = 0.0
                new_cls = RIPE_CLS if rr >= args.red_threshold else UNRIPE_CLS
                stats[f"normal→{'ripe' if new_cls == RIPE_CLS else 'unripe'}"] += 1
            elif qin_cls in (CLS_GRAY_MOLD, CLS_POWDERY, CLS_BLACK_SPOT, CLS_OVERRIPE):
                if args.keep_disease:
                    new_cls = UNRIPE_CLS  # 회피 학습용으로 unripe로 처리
                    stats[f"{QIN_NAMES[qin_cls]}→unripe"] += 1
                else:
                    stats[f"drop_{QIN_NAMES[qin_cls]}"] += 1
                    continue
            else:
                stats["unknown_cls"] += 1
                continue

            new_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not new_lines:
            stats["no_valid_box"] += 1
            continue

        stem = safe_stem(img_path.name, args.prefix, idx)
        idx += 1
        with open(out_labels / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        link_or_copy(img_path, out_images / f"{stem}{img_path.suffix}", copy=args.copy_images)
        stats["ok"] += 1

    print("\n=== 변환 결과 ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\n출력:\n  images: {out_images}\n  labels: {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
