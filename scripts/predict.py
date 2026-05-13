"""학습된 모델로 추론하고, 수확 로봇용 '딸기 픽 후보'를 산출.

통합 2-class 스키마(0=unripe, 1=ripe) 기준:
    - 픽 후보는 ripe 박스(class==1)만 사용
    - unripe 박스는 회피 정보로 함께 출력 (덜 익은 딸기를 그리퍼가 건드리지 않게)
    - 추가 보조 지표:
        red_ratio  : bbox crop의 빨강 픽셀 비율 (휴리스틱)
        occluded   : (있다면) occluded_meta.json 의 GT를 IoU로 매칭하여 부여

정렬 우선순위 (ripe 후보 내):
    1) occluded=False  →  Unknown  →  True
    2) red_ratio 높은 순
    3) conf 높은 순

출력: JSON
    {"image": "...", "picks": [...], "avoid": [...]}
    picks  = 픽 가능한 ripe 후보
    avoid  = 회피해야 할 unripe 박스 (그리퍼가 충돌하지 않도록)

사용 예:
    python scripts/predict.py \
        --weights runs/strawberry/yolo26m_ft/weights/best.pt \
        --source datasets/yolo/images/test \
        --out runs/strawberry/predict_picks.json \
        --conf 0.35 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


def red_ratio(crop_bgr: np.ndarray) -> float:
    """BGR crop에서 '잘 익은 빨간색' 픽셀의 비율 (0~1).

    HSV에서 H ∈ [0,10] ∪ [170,180], S ≥ 80, V ≥ 60 인 픽셀 카운트.
    완벽한 ripeness 분류는 아니고, 픽 후보 ranking 보조용 휴리스틱.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    import cv2  # 지연 import

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_mask = ((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 60)
    return float(red_mask.mean())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True, help="이미지 폴더 또는 단일 이미지/비디오")
    ap.add_argument("--out", type=str, default="runs/strawberry/predict_picks.json")
    ap.add_argument("--occluded-meta", type=str, default="datasets/yolo/occluded_meta.json",
                    help="GT occluded 정보(JSON). 없거나 매칭 안 되면 휴리스틱 사용")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--save-images", action="store_true", help="시각화 결과 이미지 저장")
    ap.add_argument("--ripe-threshold", type=float, default=0.35, help="빨강 비율 임계 (≥ 면 ripe)")
    return ap.parse_args()


def iou_xyxy(a: list[float], b: list[float]) -> float:
    """두 bbox(xyxy)의 IoU. GT occluded 매칭에 사용."""
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

    # GT occluded 정보 로드 (테스트 셋에 대해서만 매칭됨)
    occluded_meta: dict[str, list[dict]] = {}
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

    # 2-class 스키마 (configs/strawberry.yaml과 일치해야 함)
    UNRIPE_CLS, RIPE_CLS = 0, 1

    for r in results:
        img_path = r.path
        stem = Path(img_path).stem
        orig = r.orig_img  # BGR ndarray
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

                # GT occluded 매칭 (UniqueData 데이터셋만 가짐)
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

        # ripe 후보 정렬: not occluded → red_ratio 높음 → conf 높음
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
    occ_known = sum(1 for x in picks_per_image for p in x["picks"] if p["occluded"] is not None)
    print(f"[INFO] images={len(picks_per_image)}, ripe_picks={pick_total}, unripe_avoid={avoid_total}, occluded_known={occ_known}")
    print(f"[INFO] saved: {out}")


if __name__ == "__main__":
    main()
