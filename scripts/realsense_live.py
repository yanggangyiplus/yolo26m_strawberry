"""Intel RealSense (D435/D455 등) 컬러 스트림으로 YOLO 실시간 추론.

RGB-D 카메라이므로 검출 박스 중심점의 **깊이(거리 m)** 까지 함께 표시한다.
수확 로봇 관점에서:
    - 박스 + class + conf 외에 '딸기까지 거리(m)' 와 'red_ratio' 를 라이브로 확인 가능
    - 'p' 키: 현재 프레임 스냅샷 저장 (annotated + raw color + depth_colormap)
    - 'q' / ESC: 종료

사용 예:
    python scripts/realsense_live.py \
        --weights runs/detect/runs/strawberry/yolo26m_ft_v3-2/weights/best.pt \
        --imgsz 704 --conf 0.35

    # 파인튜닝 전 모델로 보고 싶으면
    python scripts/realsense_live.py --weights yolo26m.pt --conf 0.25

요구 패키지:
    pip install pyrealsense2 opencv-python ultralytics
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs/strawberry/yolo26m_ft_v3-2/weights/best.pt",
        help="YOLO 가중치 (.pt)",
    )
    ap.add_argument("--imgsz", type=int, default=704)
    # 클래스별 conf threshold (모형/조명 도메인 차이 보정).
    # 모델 예측은 낮은 base-conf 로 받고, 클래스별로 후처리 필터링.
    ap.add_argument("--base-conf", type=float, default=0.10,
                    help="모델 추론 시 1차 conf threshold (이후 클래스별 필터링)")
    ap.add_argument("--ripe-conf", type=float, default=0.45,
                    help="ripe 로 인정하기 위한 최소 conf (보수적으로 높게)")
    ap.add_argument("--unripe-conf", type=float, default=0.20,
                    help="unripe 로 인정하기 위한 최소 conf (덜 익은 모형까지 잡기 위해 낮게)")
    # 색 기반 sanity check.
    # 진짜로 익은 딸기는 보통 빨강 비율이 충분히 높음. 그렇지 않으면 모형/덜익음으로 간주.
    ap.add_argument("--min-red-for-ripe", type=float, default=0.35,
                    help="ripe로 분류된 박스의 red_ratio가 이 값 미만이면 unripe로 강제 전환 "
                         "(red_ratio는 진한 빨강 픽셀 비율; 분홍/연한 모형은 거의 0에 가까움)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0", help="cuda 인덱스 또는 'cpu'")
    ap.add_argument("--width", type=int, default=848, help="RealSense 컬러/깊이 가로 (D455 권장 848)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true", help="깊이 스트림 끄고 컬러만 사용")
    ap.add_argument("--show-depth", action="store_true", help="우측에 depth colormap 같이 표시")
    ap.add_argument("--save-dir", type=str, default="runs/realsense", help="스냅샷 저장 폴더")
    return ap.parse_args()


def red_ratio(crop_bgr: np.ndarray) -> float:
    """BGR crop에서 '진한 빨강' 픽셀의 비율.

    분홍/연한 빨강(저채도 빨강)은 익은 딸기로 인정하지 않기 위해 S(채도) 임계를
    높게(>=120) 잡는다. 이전 버전(S>=80)에서는 분홍 모형이 red_ratio 0.25~0.30 정도가
    나와 ripe 로 잘못 통과했음.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 120) & (v >= 70)
    return float(mask.mean())


def setup_pipeline(width: int, height: int, fps: int, with_depth: bool) -> tuple[rs.pipeline, rs.align | None, float]:
    """RealSense pipeline 시작.

    Returns
    -------
    pipeline, align, depth_scale
        with_depth=False 면 align=None, depth_scale=0.0
    """
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if with_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)

    align: rs.align | None = None
    depth_scale = 0.0
    if with_depth:
        # 깊이 → 컬러 좌표계로 정렬 (박스 중심에서 거리 뽑으려면 필수)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        print(f"[INFO] depth_scale = {depth_scale:.6f} m/unit")
    return pipeline, align, depth_scale


def sample_depth(depth_image: np.ndarray, cx: int, cy: int, depth_scale: float, k: int = 5) -> float:
    """박스 중심 (cx,cy) 주변 k×k 윈도우의 유효 깊이 median (단위: m). 0 이면 invalid."""
    h, w = depth_image.shape[:2]
    x1, y1 = max(0, cx - k // 2), max(0, cy - k // 2)
    x2, y2 = min(w, cx + k // 2 + 1), min(h, cy + k // 2 + 1)
    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid) * depth_scale)


def main() -> None:
    args = parse_args()
    with_depth = not args.no_depth

    print(f"[INFO] loading YOLO: {args.weights}")
    model = YOLO(args.weights)
    class_names = model.names

    print(f"[INFO] starting RealSense {args.width}x{args.height}@{args.fps}fps (depth={with_depth})")
    pipeline, align, depth_scale = setup_pipeline(args.width, args.height, args.fps, with_depth)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    win = "RealSense YOLO (q=quit, p=snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

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
            color = np.asanyarray(color_frame.get_data())  # BGR uint8

            depth_image: np.ndarray | None = None
            if with_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())  # uint16

            # YOLO 추론 — 일단 낮은 base-conf 로 다 받은 뒤, 후처리에서 클래스별 threshold 적용
            results = model.predict(
                source=color,
                conf=args.base_conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            r = results[0]
            annotated = color.copy()

            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cls_id = int(clss[i])
                    name = class_names.get(cls_id, str(cls_id))
                    conf = float(confs[i])

                    crop = color[max(0, y1):y2, max(0, x1):x2]
                    rr = red_ratio(crop)

                    # === 클래스별 conf threshold 필터 ===
                    if name == "ripe_strawberry" and conf < args.ripe_conf:
                        continue
                    if name == "unripe_strawberry" and conf < args.unripe_conf:
                        continue

                    # === 색 기반 sanity check: ripe인데 빨갛지 않으면 unripe로 강제 ===
                    # 플라스틱/폼 딸기 모형은 보통 진한 빨강이 아니라 분홍/주황/흰빛이 섞여
                    # red_ratio가 낮게 나옴. 이걸 ripe로 신뢰하면 위양성(false ripe)이 생기므로
                    # color override 로 unripe 로 보정.
                    overridden = False
                    if name == "ripe_strawberry" and rr < args.min_red_for_ripe:
                        name = "unripe_strawberry"
                        cls_id = 0
                        overridden = True

                    # 2-class 모델이면 ripe=초록, unripe=주황. 그 외 모델은 노란색.
                    if name == "ripe_strawberry":
                        col = (0, 200, 0)
                    elif name == "unripe_strawberry":
                        col = (0, 165, 255)
                    else:
                        col = (0, 255, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)

                    tag = f"{name} {conf:.2f}"
                    if overridden:
                        tag += "*"  # color override 표식
                    label_lines = [tag, f"red={rr:.2f}"]
                    if depth_image is not None:
                        dist_m = sample_depth(depth_image, cx, cy, depth_scale)
                        if dist_m > 0:
                            label_lines.append(f"d={dist_m:.2f}m")
                    cv2.circle(annotated, (cx, cy), 3, col, -1)

                    # 라벨 박스 (가독성용 검은 배경)
                    for j, line in enumerate(label_lines):
                        y_text = y1 - 6 - 14 * (len(label_lines) - 1 - j)
                        if y_text < 12:
                            y_text = y2 + 14 + 14 * j
                        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(
                            annotated,
                            (x1, y_text - th - 2),
                            (x1 + tw + 4, y_text + 2),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            annotated, line, (x1 + 2, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA,
                        )

            # FPS 계산
            fps_frames += 1
            if fps_frames >= 10:
                t1 = time.time()
                fps_display = fps_frames / (t1 - fps_t0)
                fps_t0 = t1
                fps_frames = 0
            cv2.putText(
                annotated, f"FPS {fps_display:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )

            # 표시 (옵션: 우측에 depth colormap)
            if args.show_depth and depth_image is not None:
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                show = np.hstack([annotated, depth_vis])
            else:
                show = annotated

            cv2.imshow(win, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                # 스냅샷: 시각화 이미지 + 원본 컬러 + (있다면) 깊이 raw/colormap 저장
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(save_dir / f"{ts}_annotated.png"), annotated)
                cv2.imwrite(str(save_dir / f"{ts}_color.png"), color)
                if depth_image is not None:
                    np.save(str(save_dir / f"{ts}_depth.npy"), depth_image)
                    depth_cm = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                    )
                    cv2.imwrite(str(save_dir / f"{ts}_depth.png"), depth_cm)
                print(f"[snap] saved to {save_dir} (prefix {ts})")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
