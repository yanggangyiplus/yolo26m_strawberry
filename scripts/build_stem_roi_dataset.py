#!/usr/bin/env python3
"""딸기 bbox 상단 ROI 크롭 + CLAHE 데이터셋 구축 (줄기 seg 재학습용).

파이프라인:
  yolo_unified_farm (전체 농장 이미지 + detect bbox)
    → 과실마다 상단 ROI crop + preprocess
    → yolo_stem_roi (작은 패치 + 줄기 폴리곤)

사용:
  python scripts/build_stem_roi_dataset.py
  python scripts/split_dataset.py --root datasets/yolo_stem_roi
  python scripts/train_seg.py --data configs/strawberry_stem_roi_seg.yaml \\
      --name yolo26m_stem_roi_128b16 --imgsz 128 --batch 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from stem_roi_utils import (
    RoiCrop,
    clip_polygon_to_roi,
    compute_top_roi,
    format_yolo_seg_line,
    preprocess_stem_roi,
    yolo_bbox_to_xyxy,
)

ROOT = Path(__file__).resolve().parents[1]
FARM_IMG = ROOT / "datasets" / "yolo_unified_farm" / "images" / "all"
FARM_LBL = ROOT / "datasets" / "yolo_unified_farm" / "labels" / "all"
SEG_LBL = ROOT / "datasets" / "yolo_unified_seg" / "labels" / "all"
OUT_ROOT = ROOT / "datasets" / "yolo_stem_roi"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STEM_CLS = 0


def parse_bbox(line: str) -> tuple[int, float, float, float, float] | None:
    p = line.strip().split()
    if len(p) < 5:
        return None
    return int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])


def parse_seg_polys(text: str) -> list[list[tuple[float, float]]]:
    polys: list[list[tuple[float, float]]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        coords = [float(x) for x in parts[1:]]
        polys.append([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])
    return polys


def find_image(stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = FARM_IMG / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--above-ratio", type=float, default=0.55)
    ap.add_argument("--into-fruit", type=float, default=0.08)
    ap.add_argument("--pad-x", type=float, default=0.15)
    ap.add_argument("--no-preprocess", action="store_true", help="CLAHE/HSV 생략(비교용)")
    ap.add_argument("--use-full-stem-labels", action="store_true",
                    help="전체 이미지 stem 라벨을 ROI에 클립 (없으면 bbox 휴리스틱)")
    args = ap.parse_args()

    out_img = OUT_ROOT / "images" / "all"
    out_lbl = OUT_ROOT / "labels" / "all"
    meta_path = OUT_ROOT / "roi_meta.jsonl"

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    n_crop = n_lbl = n_skip = 0
    meta_lines: list[str] = []

    for lbl_path in sorted(FARM_LBL.glob("*.txt")):
        if lbl_path.name == "classes.txt":
            continue
        stem = lbl_path.stem
        img_path = find_image(stem)
        if img_path is None:
            n_skip += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            n_skip += 1
            continue
        ih, iw = bgr.shape[:2]

        full_stem_polys: list[list[tuple[float, float]]] = []
        seg_lbl = SEG_LBL / f"{stem}.txt"
        if args.use_full_stem_labels and seg_lbl.is_file():
            full_stem_polys = parse_seg_polys(seg_lbl.read_text(encoding="utf-8"))

        for fi, line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines()):
            parsed = parse_bbox(line)
            if parsed is None:
                continue
            _cls, cx, cy, w, h = parsed
            x1, y1, x2, y2 = yolo_bbox_to_xyxy(cx, cy, w, h, iw, ih)
            roi_box = compute_top_roi(
                x1, y1, x2, y2, iw, ih,
                above_ratio=args.above_ratio,
                into_fruit_ratio=args.into_fruit,
                pad_x_ratio=args.pad_x,
            )
            if roi_box is None:
                continue
            rx1, ry1, rx2, ry2 = roi_box
            roi = RoiCrop(rx1, ry1, rx2, ry2, fi)

            crop = bgr[ry1:ry2, rx1:rx2].copy()
            if not args.no_preprocess:
                crop = preprocess_stem_roi(crop)

            crop_stem = f"{stem}_f{fi:03d}"
            cv2.imwrite(str(out_img / f"{crop_stem}.jpg"), crop)

            label_lines: list[str] = []
            if full_stem_polys:
                for poly in full_stem_polys:
                    inner = clip_polygon_to_roi(poly, roi, iw, ih)
                    if inner:
                        label_lines.append(format_yolo_seg_line(STEM_CLS, inner))
            else:
                # bbox 상단 휴리스틱 (ROI 정규 좌표)
                top_in_roi = min(1.0, (y1 - ry1) / roi.height)
                stem_h = min(0.45, top_in_roi)
                label_lines.append(
                    format_yolo_seg_line(
                        STEM_CLS,
                        [(0.25, top_in_roi), (0.75, top_in_roi), (0.75, max(0, top_in_roi - stem_h)), (0.25, max(0, top_in_roi - stem_h))],
                    )
                )

            (out_lbl / f"{crop_stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            n_crop += 1
            if label_lines:
                n_lbl += 1

            meta_lines.append(json.dumps({
                "crop_id": crop_stem,
                "source": stem,
                "fruit_idx": fi,
                "roi": [rx1, ry1, rx2, ry2],
                "fruit_bbox": [x1, y1, x2, y2],
            }))

    (out_lbl / "classes.txt").write_text("stem\n", encoding="utf-8")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"[done] ROI crops: {n_crop} (labeled {n_lbl}), skipped images {n_skip}")
    print(f"  images: {out_img}")
    print(f"  labels: {out_lbl}")
    print(f"  meta:   {meta_path}")
    print("\n다음:")
    print("  python scripts/split_dataset.py --root datasets/yolo_stem_roi")
    print("  python scripts/train_seg.py --data configs/strawberry_stem_roi_seg.yaml \\")
    print("      --name yolo26m_stem_roi_128b16 --imgsz 128 --batch 16 --epochs 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
