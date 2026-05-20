#!/usr/bin/env python3
"""통합 detect bbox 기준으로 줄기(stem) YOLO-seg 폴리곤 자동 생성.

과실 박스 상단 중심 위로 가는 사각(또는 녹색 마스크 정제) 폴리곤을 class 0 stem 으로 저장.
초기 학습·부트스트랩용 — 수확 각도·가림이 많은 장면은 LabelImg/CVAT 수동 보완 권장.

사용:
    python scripts/build_farm_dataset.py --also-seg
    python scripts/autolabel_stem_seg.py --refine-green
    python scripts/split_dataset.py --root datasets/yolo_unified_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

STEM_CLS = 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_bbox_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def stem_quad_from_bbox(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    """과실 bbox 상단에서 위로 뻗는 줄기 사각형(정규화 좌표)."""
    top = cy - h / 2
    stem_len = float(np.clip(h * 0.38, 0.018, 0.14))
    stem_w = float(np.clip(w * 0.32, 0.012, w * 0.55))
    y_bot = top
    y_top = max(0.0, top - stem_len)
    x0 = float(np.clip(cx - stem_w / 2, 0.0, 1.0))
    x1 = float(np.clip(cx + stem_w / 2, 0.0, 1.0))
    return [(x0, y_bot), (x1, y_bot), (x1, y_top), (x0, y_top)]


def green_stem_polygon(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> list[tuple[float, float]] | None:
    """bbox 위쪽 ROI에서 녹색 줄기 윤곽 → 정규화 폴리곤(최소 4점)."""
    if cv2 is None:
        return None
    ih, iw = bgr.shape[:2]
    top = cy - h / 2
    stem_len = float(np.clip(h * 0.45, 0.02, 0.16))
    pad_x = float(w * 0.55)
    x1 = int(np.clip((cx - pad_x / 2) * iw, 0, iw - 1))
    x2 = int(np.clip((cx + pad_x / 2) * iw, 0, iw - 1))
    y2 = int(np.clip(top * ih, 0, ih - 1))
    y1 = int(np.clip((top - stem_len) * ih, 0, ih - 1))
    if y2 <= y1 or x2 <= x1:
        return None
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 줄기·잎 녹색 (OpenCV H 0–179)
    mask = cv2.inRange(hsv, (25, 35, 25), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 30:
        return None
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    out: list[tuple[float, float]] = []
    for px, py in pts:
        nx = (x1 + px) / iw
        ny = (y1 + py) / ih
        out.append((float(np.clip(nx, 0, 1)), float(np.clip(ny, 0, 1))))
    return out


def format_seg_line(cls_id: int, pts: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"{cls_id} {coords}"


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def autolabel_file(
    lbl_path: Path,
    img_path: Path | None,
    refine_green: bool,
) -> list[str]:
    lines_out: list[str] = []
    bgr = None
    if refine_green and img_path and cv2 is not None:
        bgr = cv2.imread(str(img_path))
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_bbox_line(line)
        if parsed is None:
            continue
        _cls, cx, cy, w, h = parsed
        pts = None
        if bgr is not None:
            pts = green_stem_polygon(bgr, cx, cy, w, h)
        if pts is None:
            pts = stem_quad_from_bbox(cx, cy, w, h)
        lines_out.append(format_seg_line(STEM_CLS, pts))
    return lines_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-labels",
        type=Path,
        default=Path("datasets/yolo_unified_farm/labels/all"),
        help="detect bbox 라벨 (기본: 농장 전경 서브셋)",
    )
    ap.add_argument(
        "--src-images",
        type=Path,
        default=Path("datasets/yolo_unified_farm/images/all"),
    )
    ap.add_argument(
        "--dst-labels",
        type=Path,
        default=Path("datasets/yolo_unified_seg/labels/all"),
    )
    ap.add_argument(
        "--refine-green",
        action="store_true",
        help="bbox 위 ROI 녹색 마스크로 줄기 윤곽 정제 (opencv 필요)",
    )
    args = ap.parse_args()
    src_lbl = args.src_labels.expanduser().resolve()
    src_img = args.src_images.expanduser().resolve()
    dst_lbl = args.dst_labels.expanduser().resolve()
    dst_lbl.mkdir(parents=True, exist_ok=True)

    (dst_lbl / "classes.txt").write_text("stem\n", encoding="utf-8")

    n_files = n_stems = n_skip = 0
    for lbl in sorted(src_lbl.glob("*.txt")):
        if lbl.name == "classes.txt":
            continue
        img_path = find_image(src_img, lbl.stem)
        out_lines = autolabel_file(lbl, img_path, args.refine_green)
        if not out_lines:
            n_skip += 1
            continue
        (dst_lbl / lbl.name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        n_files += 1
        n_stems += len(out_lines)

    print(
        f"[done] stem seg labels: {n_files} images, {n_stems} stem polygons, "
        f"skipped(empty)={n_skip}\n"
        f"  out: {dst_lbl}\n"
        f"  next: python scripts/split_dataset.py --root datasets/yolo_unified_seg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
