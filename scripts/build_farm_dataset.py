#!/usr/bin/env python3
"""통합 데이터에서 농장·전경 샷만 복제해 별도 YOLO 데이터셋 생성.

포함 (농장/재배 전경, CVAT·실제 장면):
  - qin_*       Qin2006 (325)
  - uniq_*      UniqueData 농장 (40)
  - realscene_* 실제 장면 (7)

제외 (클로즈업·모형):
  - straw_pick_* / straw_unpick_*  strawberryDataset 근접 촬영
  - user_raw_* / user_red_*        플라스틱 모형

사용:
    python scripts/build_farm_dataset.py
    python scripts/build_farm_dataset.py --also-seg   # 줄기 seg 폴더 이미지까지 동기 복제
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_IMG = ROOT / "datasets" / "yolo_unified" / "images" / "all"
SRC_LBL = ROOT / "datasets" / "yolo_unified" / "labels" / "all"
FARM_ROOT = ROOT / "datasets" / "yolo_unified_farm"
SEG_IMG = ROOT / "datasets" / "yolo_unified_seg" / "images" / "all"

# 농장 전경·멀리서 본 재배 화면 prefix
FARM_PREFIXES = ("qin_", "uniq_", "realscene_")


def is_farm_stem(name: str) -> bool:
    return name.startswith(FARM_PREFIXES)


def copy_pair(stem: str, dst_img: Path, dst_lbl: Path) -> bool:
    """이미지·detect 라벨 한 쌍 복제. 성공 시 True."""
    img = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = SRC_IMG / f"{stem}{ext}"
        if p.is_file():
            img = p
            break
    lbl = SRC_LBL / f"{stem}.txt"
    if img is None or not lbl.is_file():
        return False
    shutil.copy2(img, dst_img / img.name)
    shutil.copy2(lbl, dst_lbl / lbl.name)
    return True


def clear_dir(d: Path) -> None:
    """기존 복제본 제거(재실행 시 중복 방지)."""
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_symlink():
            p.unlink()


def build_farm(also_seg: bool) -> None:
    dst_img = FARM_ROOT / "images" / "all"
    dst_lbl = FARM_ROOT / "labels" / "all"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    clear_dir(dst_img)
    clear_dir(dst_lbl)
    (dst_lbl / "classes.txt").write_text(
        "unripe_strawberry\nripe_strawberry\n",
        encoding="utf-8",
    )

    n_ok = n_skip = 0
    for lbl in sorted(SRC_LBL.glob("*.txt")):
        if lbl.name == "classes.txt":
            continue
        if not is_farm_stem(lbl.stem):
            continue
        if copy_pair(lbl.stem, dst_img, dst_lbl):
            n_ok += 1
        else:
            n_skip += 1
            print(f"[skip] missing pair: {lbl.stem}")

    print(f"[farm] {FARM_ROOT}")
    print(f"  copied: {n_ok} image+label pairs, skipped: {n_skip}")

    if also_seg:
        seg_img = SEG_IMG
        seg_img.parent.mkdir(parents=True, exist_ok=True)
        if seg_img.is_symlink():
            seg_img.unlink()
        elif seg_img.is_dir():
            clear_dir(seg_img)
        else:
            seg_img.mkdir(parents=True, exist_ok=True)
        n_seg = 0
        for f in dst_img.iterdir():
            if f.is_file():
                shutil.copy2(f, seg_img / f.name)
                n_seg += 1
        print(f"[seg]  copied {n_seg} images -> {seg_img}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--also-seg",
        action="store_true",
        help="datasets/yolo_unified_seg/images/all 에도 동일 농장 이미지 복제",
    )
    args = ap.parse_args()
    if not SRC_IMG.is_dir() or not SRC_LBL.is_dir():
        raise SystemExit(f"원본 없음: {SRC_IMG} 또는 {SRC_LBL}")
    build_farm(args.also_seg)
    print(
        "\n다음:\n"
        "  python scripts/split_dataset.py --root datasets/yolo_unified_farm\n"
        "  python scripts/autolabel_stem_seg.py \\\n"
        "    --src-labels datasets/yolo_unified_farm/labels/all \\\n"
        "    --src-images datasets/yolo_unified_farm/images/all \\\n"
        "    --refine-green\n"
        "  python scripts/split_dataset.py --root datasets/yolo_unified_seg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
