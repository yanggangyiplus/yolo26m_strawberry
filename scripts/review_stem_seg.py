#!/usr/bin/env python3
"""줄기 seg 검수 — GT(자동 라벨) vs 모델 예측을 한 장 이미지로 저장.

사용:
  python scripts/review_stem_seg.py
  python scripts/review_stem_seg.py --mode realscene
  python scripts/review_stem_seg.py --mode val --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "datasets" / "yolo_unified_seg"
DEFAULT_WEIGHTS = (
    ROOT / "runs/segment/runs/strawberry/yolo26m_stem_farm_640b8/weights/best.pt"
)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def parse_seg_lines(text: str) -> list[list[tuple[float, float]]]:
    polys: list[list[tuple[float, float]]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        coords = [float(x) for x in parts[1:]]
        polys.append([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])
    return polys


def draw_polys(
    img: np.ndarray,
    polys: list[list[tuple[float, float]]],
    color: tuple[int, int, int],
    alpha: float,
    prefix: str,
) -> None:
    h, w = img.shape[:2]
    for i, pts in enumerate(polys):
        px = np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)
        if len(px) < 3:
            continue
        overlay = img.copy()
        cv2.fillPoly(overlay, [px], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [px], True, color, 2)
        cx, cy = int(px[:, 0].mean()), int(px[:, 1].mean())
        cv2.putText(
            img, f"{prefix}{i}", (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )


def pred_polys_from_result(r, conf_min: float) -> list[list[tuple[float, float]]]:
    """Ultralytics seg 결과 → 정규화 폴리곤 리스트."""
    polys: list[list[tuple[float, float]]] = []
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return polys
    h, w = r.orig_shape
    confs = r.boxes.conf.cpu().numpy()
    # masks.xy: 픽셀 좌표 폴리곤 리스트
    for i, xy in enumerate(r.masks.xy):
        if float(confs[i]) < conf_min:
            continue
        if xy is None or len(xy) < 3:
            continue
        polys.append([(float(x) / w, float(y) / h) for x, y in xy])
    return polys


def collect_stems(mode: str) -> tuple[Path, Path, list[str]]:
    split = "all" if mode in ("realscene", "qin_sample") else mode
    img_dir = SEG_ROOT / "images" / split
    lbl_dir = SEG_ROOT / "labels" / split
    if mode == "realscene":
        stems = sorted(p.stem for p in img_dir.glob("realscene_*") if p.is_file())
    elif mode == "qin_sample":
        stems = sorted(p.stem for p in img_dir.glob("qin_*") if p.is_file())[:20]
    else:
        stems = sorted(
            p.stem for p in img_dir.iterdir()
            if p.is_file() and (lbl_dir / f"{p.stem}.txt").is_file()
        )
    return img_dir, lbl_dir, stems


def main() -> int:
    ap = argparse.ArgumentParser(description="줄기 GT vs 예측 통합 검수")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--mode", default="val", choices=("val", "test", "realscene", "qin_sample"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--out", type=Path, default=Path("runs/stem_review"))
    ap.add_argument("--max", type=int, default=0, help="0=전부, N=앞에서 N장만")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"가중치 없음: {args.weights}")

    img_dir, lbl_dir, stems = collect_stems(args.mode)
    if args.max > 0:
        stems = stems[: args.max]

    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)

    n_gt = n_pred_img = n_both = n_miss = n_false = 0
    total_gt = total_pred = 0

    for stem in stems:
        img_path = find_image(img_dir, stem)
        lbl_path = lbl_dir / f"{stem}.txt"
        if img_path is None:
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        gt_polys = parse_seg_lines(lbl_path.read_text(encoding="utf-8")) if lbl_path.is_file() else []
        results = model.predict(
            source=bgr, conf=args.conf, imgsz=args.imgsz,
            device=args.device, verbose=False,
        )
        pred_polys = pred_polys_from_result(results[0], args.conf)

        left = bgr.copy()
        right = bgr.copy()
        draw_polys(left, gt_polys, (0, 255, 0), 0.4, "gt")
        draw_polys(right, pred_polys, (255, 180, 0), 0.4, "pr")

        cv2.putText(left, f"GT ({len(gt_polys)})", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(right, f"PRED ({len(pred_polys)}) conf>={args.conf}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        cv2.putText(left, stem, (8, h := left.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        combo = np.hstack([left, right])
        cv2.imwrite(str(args.out / f"{stem}_review.jpg"), combo)

        total_gt += len(gt_polys)
        total_pred += len(pred_polys)
        if len(gt_polys) > 0:
            n_gt += 1
        if len(pred_polys) > 0:
            n_pred_img += 1
        if len(gt_polys) > 0 and len(pred_polys) > 0:
            n_both += 1
        elif len(gt_polys) > 0 and len(pred_polys) == 0:
            n_miss += 1
        elif len(gt_polys) == 0 and len(pred_polys) > 0:
            n_false += 1

    n = len(stems)
    summary = f"""# 줄기 seg 검수 요약 ({args.mode}, conf>={args.conf})

- 이미지: {n}장
- GT stem 있는 장: {n_gt} (총 {total_gt}개 폴리곤)
- 예측 stem 있는 장: {n_pred_img} (총 {total_pred}개)
- GT·예측 둘 다 있음: {n_both}
- GT는 있는데 예측 없음 (놓침): {n_miss}
- GT 없는데 예측 있음 (과검출): {n_false}

결과 이미지: {args.out.resolve()}/<stem>_review.jpg
  왼쪽=녹색 GT(자동라벨)  오른쪽=주황 예측(best.pt)
"""
    (args.out / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
