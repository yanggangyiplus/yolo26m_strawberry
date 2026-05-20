#!/usr/bin/env python3
"""정적 이미지 ripe/unripe 판별 + 픽 후보 JSON (share 패키지용).

사용:
  cd share/strawberry_yolo26m_unified
  python predict.py --source /path/to/images --imgsz 832
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

_PKG = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _PKG / "weights" / "best.pt"


def red_ratio(crop_bgr: np.ndarray) -> float:
    """BGR crop에서 익은 빨간색 픽셀 비율 (0~1)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    import cv2

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_mask = ((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 60)
    return float(red_mask.mean())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="딸기 detect → picks/avoid JSON")
    ap.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"detect 가중치 (기본 {DEFAULT_WEIGHTS.name})",
    )
    ap.add_argument("--source", type=str, required=True, help="이미지 폴더·단일 파일·영상")
    ap.add_argument("--out", type=str, default="runs/predict_picks.json")
    ap.add_argument(
        "--occluded-meta",
        type=str,
        default="",
        help="GT occluded JSON (없으면 생략)",
    )
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=832)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--ripe-threshold", type=float, default=0.35, help="red_ratio ≥ 이면 is_red")
    return ap.parse_args()


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()

    occluded_meta: dict[str, list[dict]] = {}
    if args.occluded_meta:
        meta_path = Path(args.occluded_meta)
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                occluded_meta = json.load(f)
            print(f"[INFO] loaded occluded meta for {len(occluded_meta)} images")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save_images,
        verbose=False,
    )

    picks_per_image: list[dict[str, Any]] = []
    UNRIPE_CLS, RIPE_CLS = 0, 1

    for r in results:
        img_path = r.path
        stem = Path(img_path).stem
        orig = r.orig_img
        boxes = r.boxes
        gt_boxes = occluded_meta.get(stem, [])

        picks: list[dict[str, Any]] = []
        avoid: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(float, xyxy[i])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                crop = orig[int(max(0, y1)) : int(y2), int(max(0, x1)) : int(x2)]
                ripeness = red_ratio(crop)

                occluded: bool | None = None
                if gt_boxes:
                    best_iou = 0.0
                    best_occ = False
                    for gt in gt_boxes:
                        i_ = iou_xyxy([x1, y1, x2, y2], gt["bbox_xyxy"])
                        if i_ > best_iou:
                            best_iou = i_
                            best_occ = bool(gt.get("occluded", False))
                    if best_iou >= 0.5:
                        occluded = best_occ

                entry = {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "conf": float(confs[i]),
                    "cls": int(clss[i]),
                    "red_ratio": ripeness,
                    "is_red": bool(ripeness >= args.ripe_threshold),
                    "occluded": occluded,
                }
                if int(clss[i]) == RIPE_CLS:
                    picks.append(entry)
                else:
                    avoid.append(entry)

        def sort_key(p: dict[str, Any]) -> tuple:
            occ_score = 0 if p["occluded"] is False else (2 if p["occluded"] is True else 1)
            return (occ_score, -p["red_ratio"], -p["conf"])

        picks.sort(key=sort_key)
        picks_per_image.append({"image": img_path, "picks": picks, "avoid": avoid})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(picks_per_image, f, ensure_ascii=False, indent=2)

    pick_total = sum(len(x["picks"]) for x in picks_per_image)
    avoid_total = sum(len(x["avoid"]) for x in picks_per_image)
    print(f"[INFO] images={len(picks_per_image)}, ripe_picks={pick_total}, unripe_avoid={avoid_total}")
    print(f"[INFO] saved: {out.resolve()}")


if __name__ == "__main__":
    main()
