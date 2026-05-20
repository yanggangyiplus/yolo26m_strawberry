#!/usr/bin/env python3
"""Intel RealSense + YOLO26m-seg 실시간 줄기(또는 seg 클래스) 마스크 확인.

detect용 realsense_live.py 와 별도로, segmentation 가중치로 마스크·박스를 표시한다.

사용 예:
    # 사전학습 seg (COCO 등 — 줄기는 안 나올 수 있음, 파이프 확인용)
    python scripts/realsense_live_seg.py --weights yolo26m-seg.pt --conf 0.25

    # 줄기 파인튜닝 후 (학습 완료 시 경로 예시)
    python scripts/realsense_live_seg.py \\
        --weights runs/segment/runs/strawberry/yolo26m_stem_farm_640b8/weights/best.pt \\
        --conf 0.30 --imgsz 640

    # GUI 없이 MP4
    python scripts/realsense_live_seg.py --weights yolo26m-seg.pt --headless

키: q 종료, p 스냅샷 (runs/realsense_seg/)
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

# RealSense·GUI 공통 유틸 (detect 스크립트와 동일)
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from realsense_live import (  # noqa: E402
    cv2_gui_available,
    sample_depth,
    setup_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]

# 클래스별 마스크 색 (BGR)
MASK_COLORS: dict[str, tuple[int, int, int]] = {
    "stem": (0, 255, 0),
    "unripe_strawberry": (0, 165, 255),
    "ripe_strawberry": (0, 200, 0),
}


def resolve_default_weights() -> str:
    """프로젝트 내 seg 가중치 후보를 순서대로 탐색."""
    candidates = [
        ROOT / "runs/segment/runs/strawberry/yolo26m_stem_farm_640b8/weights/best.pt",
        ROOT / "runs/segment/yolo26m_stem_farm_640b8/weights/best.pt",
        ROOT / "yolo26m-seg.pt",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(ROOT / "yolo26m-seg.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RealSense + YOLO seg 실시간")
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="seg .pt (기본: 학습 best 또는 yolo26m-seg.pt)",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="seg confidence")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--show-depth", action="store_true")
    ap.add_argument("--save-dir", type=str, default="runs/realsense_seg")
    ap.add_argument("--mask-alpha", type=float, default=0.45, help="마스크 오버레이 투명도")
    ap.add_argument("--retina-masks", action="store_true", help="고해상도 마스크 (느릴 수 있음)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--headless-out", type=str, default="")
    ap.add_argument("--brightness", type=float, default=None)
    ap.add_argument("--contrast", type=float, default=None)
    ap.add_argument("--saturation", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--exposure", type=float, default=None)
    return ap.parse_args()


def mask_color(class_names: dict, cls_id: int) -> tuple[int, int, int]:
    name = class_names.get(cls_id, str(cls_id))
    return MASK_COLORS.get(name, (255, 200, 0))


def overlay_masks(
    bgr: np.ndarray,
    masks_tensor,
    clss,
    confs,
    class_names: dict,
    alpha: float,
) -> np.ndarray:
    """YOLO seg 마스크를 BGR 이미지 위에 합성."""
    out = bgr.copy()
    h, w = out.shape[:2]
    if masks_tensor is None:
        return out

    masks = masks_tensor.cpu().numpy()
    n = masks.shape[0]
    for i in range(n):
        m = masks[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_bool = m > 0.5
        if not m_bool.any():
            continue
        cls_id = int(clss[i]) if clss is not None else 0
        col = np.array(mask_color(class_names, cls_id), dtype=np.uint8)
        out[m_bool] = (out[m_bool].astype(np.float32) * (1 - alpha) + col * alpha).astype(
            np.uint8
        )
    return out


def draw_instances(
    annotated: np.ndarray,
    r,
    class_names: dict,
    depth_image: np.ndarray | None,
    depth_scale: float,
    with_depth: bool,
) -> None:
    """박스·라벨·마스크 중심 깊이 표시."""
    if r.boxes is None or len(r.boxes) == 0:
        return

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        cls_id = int(clss[i])
        conf = float(confs[i])
        name = class_names.get(cls_id, str(cls_id))
        col = mask_color(class_names, cls_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 3, col, -1)

        short = name if len(name) <= 12 else name.split("_")[0]
        lines = [f"{short} {conf:.2f}"]
        if with_depth and depth_image is not None:
            d_m = sample_depth(depth_image, cx, cy, depth_scale)
            if d_m > 0:
                lines.append(f"d={d_m:.2f}m")

        for j, line in enumerate(lines):
            y_text = y1 - 6 - 14 * (len(lines) - 1 - j)
            if y_text < 12:
                y_text = y2 + 14 + 14 * j
            cv2.putText(
                annotated,
                line,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )


def main() -> None:
    args = parse_args()
    weights = args.weights.strip() or resolve_default_weights()
    with_depth = not args.no_depth
    headless = bool(args.headless)

    if not headless and not cv2_gui_available():
        print(
            "\n[ERROR] OpenCV GUI 불가. pip uninstall opencv-python-headless 또는\n"
            "  PYTHONNOUSERSITE=1 python3 scripts/realsense_live_seg.py ...\n"
            "  또는 --headless\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not Path(weights).is_file():
        print(f"[ERROR] 가중치 없음: {weights}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[INFO] loading YOLO-seg: {weights}")
    model = YOLO(weights)
    class_names = model.names
    print(f"[INFO] classes: {class_names}")

    print(f"[INFO] RealSense {args.width}x{args.height}@{args.fps} depth={with_depth}")
    pipeline, align, depth_scale = setup_pipeline(
        args.width,
        args.height,
        args.fps,
        with_depth,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        gain=args.gain,
        no_auto_exposure=args.no_auto_exposure,
        exposure_us=args.exposure,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    win = "RealSense YOLO-seg (q=quit, p=snapshot)"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer: cv2.VideoWriter | None = None
    out_mp4: Path | None = None
    fps_t0 = time.time()
    fps_frames = 0
    fps_display = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if with_depth and align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())

            depth_image: np.ndarray | None = None
            if with_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())

            results = model.predict(
                source=color,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                retina_masks=args.retina_masks,
            )
            r = results[0]

            annotated = overlay_masks(
                color,
                r.masks.data if r.masks is not None else None,
                r.boxes.cls if r.boxes is not None else None,
                r.boxes.conf if r.boxes is not None else None,
                class_names,
                args.mask_alpha,
            )
            draw_instances(
                annotated,
                r,
                class_names,
                depth_image,
                depth_scale,
                with_depth,
            )

            fps_frames += 1
            if fps_frames >= 10:
                t1 = time.time()
                fps_display = fps_frames / (t1 - fps_t0)
                fps_t0, fps_frames = t1, 0
                if headless:
                    print(f"[INFO] FPS ~{fps_display:.1f}", flush=True)

            n_det = len(r.boxes) if r.boxes is not None else 0
            cv2.putText(
                annotated,
                f"FPS {fps_display:.1f}  seg={n_det}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.show_depth and depth_image is not None:
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                show = np.hstack([annotated, depth_vis])
            else:
                show = annotated

            if headless:
                if writer is None:
                    out_mp4 = Path(
                        args.headless_out.strip()
                        or save_dir / f"live_seg_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    )
                    out_mp4.parent.mkdir(parents=True, exist_ok=True)
                    hh, ww = show.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_mp4),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(args.fps),
                        (ww, hh),
                    )
                    print(f"[INFO] MP4: {out_mp4.resolve()}")
                writer.write(show)
            else:
                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(save_dir / f"{ts}_annotated.png"), annotated)
                    cv2.imwrite(str(save_dir / f"{ts}_color.png"), color)
                    if depth_image is not None:
                        np.save(str(save_dir / f"{ts}_depth.npy"), depth_image)
                    print(f"[snap] {save_dir}/{ts}_*")

    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
            if out_mp4 is not None:
                print(f"[INFO] MP4 저장: {out_mp4.resolve()}")
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
