#!/usr/bin/env python3
"""seg 학습용 labels/all 골격만 생성 (이미지는 build_farm_dataset.py --also-seg 권장).

사용:
    python scripts/build_farm_dataset.py --also-seg
    python scripts/autolabel_stem_seg.py --refine-green
    python scripts/split_dataset.py --root datasets/yolo_unified_seg
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "datasets" / "yolo_unified_seg"
LBL_ALL = SEG_ROOT / "labels" / "all"
CLASSES = LBL_ALL / "classes.txt"
SEG_IMG = SEG_ROOT / "images" / "all"


def ensure_classes() -> None:
    LBL_ALL.mkdir(parents=True, exist_ok=True)
    CLASSES.write_text("stem\n", encoding="utf-8")
    print(f"[ok] {CLASSES}")


def main() -> None:
    ensure_classes()
    n_img = len([p for p in SEG_IMG.glob("*") if p.is_file()]) if SEG_IMG.is_dir() else 0
    n_lbl = sum(1 for p in LBL_ALL.glob("*.txt") if p.name != "classes.txt")
    if n_img == 0:
        print(
            "[WARN] seg 이미지 없음. 먼저:\n"
            "  python scripts/build_farm_dataset.py --also-seg"
        )
    print(
        f"\n다음:\n"
        f"  python scripts/autolabel_stem_seg.py --refine-green\n"
        f"  python scripts/split_dataset.py --root datasets/yolo_unified_seg\n"
        f"  YOLO_CONFIG_DIR=$(pwd)/.ultralytics_cfg python scripts/train_seg.py \\\n"
        f"    --weights yolo26m-seg.pt\n"
        f"\nseg 이미지 {n_img}장, stem 라벨 {n_lbl}개"
    )


if __name__ == "__main__":
    main()
