#!/usr/bin/env python3
"""RealSense 실시간: Detect(딸기) → 상단 ROI crop → CLAHE/HSV → Stem seg.

[전체 프레임] → YOLO detect → 각 딸기 bbox 상단 ROI → 전처리 → stem seg → 전체 좌표 복원

사용:
  python scripts/realsense_stem_pipeline.py \\
    --weights-det runs/detect/runs/strawberry/yolo26m_unified_832b8/weights/best.pt \\
    --weights-stem runs/segment/runs/strawberry/yolo26m_stem_roi_128b16/weights/best.pt \\
    --imgsz-det 832 --imgsz-stem 128

  # stem ROI 모델 없으면 전체 프레임 stem 모델로 폴백 (--no-roi)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from realsense_live import (  # noqa: E402
    cv2_gui_available,
    red_ratio,
    sample_depth,
    setup_pipeline,
)
from stem_roi_utils import (  # noqa: E402
    RoiCrop,
    compute_top_roi,
    mask_centroid_in_roi,
    offset_grip_up_px,
    preprocess_stem_roi,
)

ROOT = Path(__file__).resolve().parents[1]


def default_det_weights() -> str:
    for p in (
        ROOT / "runs/detect/runs/strawberry/yolo26m_unified_832b8/weights/best.pt",
        ROOT / "runs/detect/runs/strawberry/yolo26m_unified_640b16/weights/best.pt",
    ):
        if p.is_file():
            return str(p)
    return str(ROOT / "yolo26m.pt")


def default_stem_weights() -> str:
    for p in (
        ROOT / "runs/segment/runs/strawberry/yolo26m_stem_roi_128b16/weights/best.pt",
        ROOT / "runs/segment/runs/strawberry/yolo26m_stem_farm_640b8/weights/best.pt",
    ):
        if p.is_file():
            return str(p)
    return str(ROOT / "yolo26m-seg.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect → ROI → Stem seg (RealSense)")
    ap.add_argument("--weights-det", type=str, default="")
    ap.add_argument("--weights-stem", type=str, default="")
    ap.add_argument("--imgsz-det", type=int, default=832)
    ap.add_argument("--imgsz-stem", type=int, default=128)
    ap.add_argument("--conf-det", type=float, default=0.25)
    ap.add_argument("--conf-stem", type=float, default=0.20)
    ap.add_argument("--ripe-conf", type=float, default=0.30)
    ap.add_argument("--unripe-conf", type=float, default=0.20)
    ap.add_argument(
        "--min-red-for-ripe",
        type=float,
        default=0.10,
        help="ripe 박스 red_ratio 최소. 모델이 ripe로 봐도 빨갛지 않으면 unripe 표시. "
        "0.0=비활성. 플라스틱/흰색 위양성은 0.15~0.25 권장",
    )
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--show-depth", action="store_true")
    ap.add_argument("--no-roi", action="store_true", help="전체 프레임에 stem seg (구 방식)")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--above-ratio", type=float, default=0.55)
    ap.add_argument("--save-dir", type=str, default="runs/realsense_stem")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--headless-out", type=str, default="")
    ap.add_argument("--brightness", type=float, default=None)
    ap.add_argument("--contrast", type=float, default=None)
    ap.add_argument("--saturation", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument(
        "--grip-margin-cm",
        type=float,
        default=1.0,
        help="그립점을 줄기 방향(이미지 위)으로 올리는 거리 [cm]. 기본 1cm",
    )
    ap.add_argument(
        "--stem-unripe",
        action="store_true",
        help="unripe에도 줄기 seg·그립 (기본: ripe_strawberry 만)",
    )
    return ap.parse_args()


def get_color_fy(profile: rs.pipeline_profile, ih: int) -> float:
    """RealSense 컬러 intrinsics fy (픽셀). 실패 시 해상도 기반 추정."""
    try:
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return float(intr.fy)
    except Exception:
        return float(ih) * 1.1


def draw_fruit_box(
    ann: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    name: str,
    conf: float,
    *,
    rr: float | None = None,
    overridden: bool = False,
) -> None:
    """ripe=초록 박스, unripe=주황. overridden=모델은 ripe였으나 색 체크로 unripe."""
    col = (0, 200, 0) if "ripe" in name else (0, 165, 255)
    thick = 3 if overridden else 2
    cv2.rectangle(ann, (x1, y1), (x2, y2), col, thick)
    if overridden:
        cv2.rectangle(ann, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 0, 220), 1)
    short = "ripe" if name == "ripe_strawberry" else "unripe"
    tag = f"{short} {conf:.2f}"
    if rr is not None:
        tag += f" red={rr:.2f}"
    if overridden:
        tag += " [mdl:ripe*]"
    cv2.putText(ann, tag, (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def draw_roi_rect(ann: np.ndarray, roi: RoiCrop) -> None:
    cv2.rectangle(ann, (roi.x1, roi.y1), (roi.x2, roi.y2), (255, 180, 0), 1)


def stem_on_fruit(
    color: np.ndarray,
    det_model: YOLO,
    stem_model: YOLO,
    args: argparse.Namespace,
    class_names_det: dict,
    *,
    depth_image: np.ndarray | None = None,
    depth_scale: float = 0.0,
    fy_px: float = 500.0,
) -> tuple[np.ndarray, int]:
    """한 프레임에서 detect → ROI stem → annotated."""
    ih, iw = color.shape[:2]
    ann = color.copy()
    n_stem = 0

    det_res = det_model.predict(
        source=color, conf=args.conf_det, imgsz=args.imgsz_det,
        device=args.device, verbose=False,
    )[0]

    if det_res.boxes is None or len(det_res.boxes) == 0:
        return ann, 0

    xyxy = det_res.boxes.xyxy.cpu().numpy()
    confs = det_res.boxes.conf.cpu().numpy()
    clss = det_res.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        cls_id = int(clss[i])
        conf = float(confs[i])
        name = class_names_det.get(cls_id, str(cls_id))
        if name == "ripe_strawberry" and conf < args.ripe_conf:
            continue
        if name == "unripe_strawberry" and conf < args.unripe_conf:
            continue

        crop = color[max(0, y1):y2, max(0, x1):x2]
        rr = red_ratio(crop)
        overridden = False
        # 색 sanity: 모델 ripe인데 빨간 픽셀 비율이 낮으면 unripe로 표시 (realsense_live 와 동일)
        if args.min_red_for_ripe > 0 and name == "ripe_strawberry" and rr < args.min_red_for_ripe:
            name = "unripe_strawberry"
            overridden = True

        draw_fruit_box(ann, x1, y1, x2, y2, name, conf, rr=rr, overridden=overridden)

        # 수확 그립: 기본은 ripe 과실만 줄기 seg (unripe는 검출 박스만 표시)
        if not args.stem_unripe and name != "ripe_strawberry":
            continue

        if args.no_roi:
            continue

        roi_box = compute_top_roi(
            x1, y1, x2, y2, iw, ih,
            above_ratio=args.above_ratio,
        )
        if roi_box is None:
            continue
        rx1, ry1, rx2, ry2 = roi_box
        roi = RoiCrop(rx1, ry1, rx2, ry2, i)
        draw_roi_rect(ann, roi)

        crop = color[ry1:ry2, rx1:rx2].copy()
        if not args.no_preprocess:
            crop = preprocess_stem_roi(crop)

        stem_res = stem_model.predict(
            source=crop, conf=args.conf_stem, imgsz=args.imgsz_stem,
            device=args.device, verbose=False,
        )[0]

        grip = mask_centroid_in_roi(stem_res, args.conf_stem)
        if grip is None:
            continue
        n_stem += 1
        px0 = int(roi.x1 + grip[0] * roi.width)
        py0 = int(roi.y1 + grip[1] * roi.height)

        # 꼭지/줄기 중심에서 줄기 방향(이미지 위)으로 grip_margin_cm 만큼 이동
        margin_m = float(args.grip_margin_cm) / 100.0
        depth_m = 0.0
        if depth_image is not None and depth_scale > 0:
            depth_m = sample_depth(depth_image, px0, py0, depth_scale)
        px, py = offset_grip_up_px(
            px0, py0, depth_m=depth_m, fy_px=fy_px, margin_m=margin_m, ih=ih,
        )

        cv2.circle(ann, (px, py), 6, (0, 255, 255), -1)
        cv2.circle(ann, (px0, py0), 3, (180, 180, 180), 1)  # 마스크 중심(이동 전)
        cv2.line(ann, (px, py), ((x1 + x2) // 2, y1), (0, 255, 255), 1)
        label = f"grip+{args.grip_margin_cm:.0f}cm"
        if depth_m > 0:
            label += f" d={depth_m:.2f}m"
        cv2.putText(ann, label, (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # no_roi: 전체 프레임 stem (폴백)
    if args.no_roi:
        stem_res = stem_model.predict(
            source=color, conf=args.conf_stem, imgsz=args.imgsz_stem,
            device=args.device, verbose=False,
        )[0]
        if stem_res.masks is not None:
            overlay = stem_res.plot()
            ann = cv2.addWeighted(ann, 0.6, overlay, 0.4, 0)
            n_stem = len(stem_res.boxes) if stem_res.boxes is not None else 0

    return ann, n_stem


def main() -> None:
    args = parse_args()
    w_det = args.weights_det.strip() or default_det_weights()
    w_stem = args.weights_stem.strip() or default_stem_weights()
    headless = args.headless
    with_depth = not args.no_depth

    if not headless and not cv2_gui_available():
        print("[ERROR] OpenCV GUI 불가 → --headless 또는 opencv-python-headless 제거", file=sys.stderr)
        raise SystemExit(1)

    print(f"[INFO] detect: {w_det}")
    stem_target = "ripe+unripe" if args.stem_unripe else "ripe only"
    print(f"[INFO] stem:   {w_stem}  (roi={'off' if args.no_roi else 'on'}, {stem_target})")
    det_model = YOLO(w_det)
    stem_model = YOLO(w_stem)

    pipeline, align, depth_scale = setup_pipeline(
        args.width, args.height, args.fps, with_depth,
        brightness=args.brightness, contrast=args.contrast,
        saturation=args.saturation, gain=args.gain,
        no_auto_exposure=args.no_auto_exposure, exposure_us=args.exposure,
    )
    # grip 1cm 오프셋 픽셀 환산용 fy
    profile = pipeline.get_active_profile()
    fy_px = get_color_fy(profile, args.height)
    print(f"[INFO] grip margin: +{args.grip_margin_cm} cm (stem 방향), color fy≈{fy_px:.1f}px")
    print(
        f"[INFO] min-red-for-ripe={args.min_red_for_ripe} "
        f"(0=색체크 끔, 모형/위양성 많으면 0.15~0.25)"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    win = "Detect→ROI→Stem (q=quit p=snap)"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer = None
    out_mp4 = None
    fps_disp = 0.0
    t0 = time.time()
    n_fr = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if with_depth and align:
                frames = align.process(frames)
            cf = frames.get_color_frame()
            if not cf:
                continue
            color = np.asanyarray(cf.get_data())
            depth_image = None
            if with_depth:
                df = frames.get_depth_frame()
                if df:
                    depth_image = np.asanyarray(df.get_data())

            ann, n_stem = stem_on_fruit(
                color, det_model, stem_model, args, det_model.names,
                depth_image=depth_image,
                depth_scale=depth_scale,
                fy_px=fy_px,
            )

            n_fr += 1
            if n_fr >= 10:
                fps_disp = n_fr / (time.time() - t0)
                t0, n_fr = time.time(), 0

            cv2.putText(
                ann, f"FPS {fps_disp:.1f}  stems {n_stem}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            show = ann
            if args.show_depth and depth_image is not None:
                dvis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                show = np.hstack([ann, dvis])

            if headless:
                if writer is None:
                    out_mp4 = Path(args.headless_out or save_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                    h, w = show.shape[:2]
                    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
                writer.write(show)
            else:
                cv2.imshow(win, show)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("p"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(save_dir / f"{ts}_pipeline.png"), ann)
    finally:
        pipeline.stop()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
