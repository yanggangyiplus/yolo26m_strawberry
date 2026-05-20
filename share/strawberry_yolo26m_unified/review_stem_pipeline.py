#!/usr/bin/env python3
"""이미지 폴더에서 Detect→ROI→Stem 파이프라인 검수용 비교 JPG 저장 (share용)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from realsense_stem_pipeline import default_det_weights, default_stem_weights, stem_on_fruit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="검수할 이미지 폴더")
    ap.add_argument("--pattern", type=str, default="*.jpg", help="glob (예: *.png)")
    ap.add_argument("--weights-det", default="")
    ap.add_argument("--weights-stem", default="")
    ap.add_argument("--out", type=Path, default=Path("runs/stem_pipeline_review"))
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()

    det = YOLO(args.weights_det or default_det_weights())
    stem = YOLO(args.weights_stem or default_stem_weights())

    class PipeArgs:
        conf_det = 0.25
        imgsz_det = 832
        imgsz_stem = 128
        conf_stem = 0.20
        ripe_conf = 0.30
        unripe_conf = 0.20
        min_red_for_ripe = 0.10
        device = "0"
        no_roi = False
        no_preprocess = False
        above_ratio = 0.55
        grip_margin_cm = 1.0
        stem_unripe = False

    pa = PipeArgs()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.source.glob(args.pattern))[: args.max]

    for ip in paths:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            continue
        pipe_ann, n_roi = stem_on_fruit(bgr, det, stem, pa, det.names)

        full = stem.predict(bgr, conf=0.25, imgsz=640, verbose=False)[0]
        full_ann = bgr.copy()
        if full.masks is not None:
            full_ann = cv2.addWeighted(full_ann, 0.5, full.plot(), 0.5, 0)

        h = max(pipe_ann.shape[0], full_ann.shape[0])

        def pad(im: np.ndarray) -> np.ndarray:
            if im.shape[0] < h:
                im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            return im

        combo = np.hstack([pad(full_ann), pad(pipe_ann)])
        cv2.putText(combo, "FULL seg", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            combo, f"ROI pipeline stems={n_roi}", (full_ann.shape[1] + 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.imwrite(str(args.out / f"{ip.stem}_compare.jpg"), combo)

    print(f"[done] {len(paths)}장 -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
