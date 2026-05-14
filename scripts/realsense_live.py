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

OpenCV 창 오류(The function is not implemented) 시:
    - 보통 `opencv-python-headless` 가 먼저 로드됨. `pip uninstall opencv-python-headless`
    - 또는 `~/.local` 의 headless 가 venv 를 덮을 때: `PYTHONNOUSERSITE=1 python3 ...`
    - GUI 없이 녹화만: `--headless`
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def cv2_gui_available() -> bool:
    """highgui(창 표시) 사용 가능 여부. headless 빌드면 False."""
    try:
        cv2.namedWindow("__cv2_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_gui_probe__")
        return True
    except cv2.error:
        return False


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
    ap.add_argument("--ripe-conf", type=float, default=0.30,
                    help="ripe 로 인정하기 위한 최소 conf")
    ap.add_argument("--unripe-conf", type=float, default=0.20,
                    help="unripe 로 인정하기 위한 최소 conf")
    # 색 기반 sanity check.
    ap.add_argument("--min-red-for-ripe", type=float, default=0.10,
                    help="ripe 박스의 red_ratio 최소값. 0.0=색 체크 완전 비활성화. "
                         "실제 딸기인데 unripe로 튕기면 0.0으로 낮춰볼 것")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0", help="cuda 인덱스 또는 'cpu'")
    ap.add_argument("--width", type=int, default=848, help="RealSense 컬러/깊이 가로 (D455 권장 848)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true", help="깊이 스트림 끄고 컬러만 사용")
    ap.add_argument("--show-depth", action="store_true", help="우측에 depth colormap 같이 표시")
    ap.add_argument("--save-dir", type=str, default="runs/realsense", help="스냅샷 저장 폴더")
    ap.add_argument("--smooth", type=int, default=7,
                    help="시간적 평활화 버퍼 크기 (프레임 수). 이 크기만큼 다수결로 클래스 결정. "
                         "클수록 안정적이지만 반응이 느려짐. 0=비활성화")
    ap.add_argument("--match-dist", type=int, default=60,
                    help="이전 프레임 박스와 같은 딸기로 매칭할 최대 중심점 거리(픽셀)")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="GUI 없이 annotated 영상을 MP4로만 저장 (opencv-python-headless·SSH 환경)",
    )
    ap.add_argument(
        "--headless-out",
        type=str,
        default="",
        help="--headless 시 출력 MP4 경로. 비우면 save_dir/live_<timestamp>.mp4",
    )
    # ── RealSense 컬러 센서 (UVC) ── 값은 기기마다 min/max 다름 → 지원 범위로 자동 클램프
    ap.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="컬러 밝기(지원 시). 예: 20~40 정도 올려보기. 미지정이면 카메라 기본값 유지",
    )
    ap.add_argument("--contrast", type=float, default=None, help="컬러 대비(지원 시)")
    ap.add_argument("--saturation", type=float, default=None, help="컬러 채도(지원 시)")
    ap.add_argument("--gain", type=float, default=None, help="컬러 게인(지원 시, 어두울 때)")
    ap.add_argument(
        "--no-auto-exposure",
        action="store_true",
        help="자동 노출 끄고 --exposure 로 고정(밝기/프레임 안정에 유리할 때)",
    )
    ap.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="수동 노출. 단위는 기기/UVC마다 다름. --no-auto-exposure 와 함께. 시작 시 로그의 [min,max] 안에서 조정",
    )
    return ap.parse_args()


def red_ratio(crop_bgr: np.ndarray) -> float:
    """BGR crop에서 빨간 픽셀(익은 딸기 색)의 비율.

    S >= 80(OpenCV 0-255): 분홍/연빨강 포함. 실제 딸기는 조명 영향으로
    채도가 낮게 측정될 수 있으므로 과거 S>=120보다 완화함.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 12) | (h >= 168)) & (s >= 80) & (v >= 60)
    return float(mask.mean())


def _set_color_option(sensor: rs.sensor, option: rs.option, value: float, label: str) -> None:
    """지원되는 UVC 옵션만 설정하고, min~max 범위로 클램프한다."""
    if not sensor.supports(option):
        print(f"[WARN] color sensor: '{label}' 옵션 미지원 → 건너뜀")
        return
    rng = sensor.get_option_range(option)
    v = float(max(rng.min, min(rng.max, value)))
    sensor.set_option(option, v)
    print(f"[INFO] color {label} = {v}  (허용 [{rng.min}, {rng.max}], step {rng.step})")


def apply_color_controls(
    profile: rs.pipeline_profile,
    *,
    brightness: float | None,
    contrast: float | None,
    saturation: float | None,
    gain: float | None,
    no_auto_exposure: bool,
    exposure_us: float | None,
) -> None:
    """스트림 시작 직후 컬러 센서 밝기·노출 등 적용."""
    try:
        color_sensor = profile.get_device().first_color_sensor()
    except RuntimeError:
        print("[WARN] first_color_sensor() 실패 — 밝기/노출 설정 생략")
        return

    # 자동 노출 끄기 → 수동 노출/게인이 먹도록 (순서 중요)
    if no_auto_exposure and color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        print("[INFO] color auto_exposure = OFF")

    if exposure_us is not None:
        _set_color_option(color_sensor, rs.option.exposure, exposure_us, "exposure (us)")

    if brightness is not None:
        _set_color_option(color_sensor, rs.option.brightness, brightness, "brightness")
    if contrast is not None:
        _set_color_option(color_sensor, rs.option.contrast, contrast, "contrast")
    if saturation is not None:
        _set_color_option(color_sensor, rs.option.saturation, saturation, "saturation")
    if gain is not None:
        _set_color_option(color_sensor, rs.option.gain, gain, "gain")


def _print_realsense_busy_help(err: BaseException) -> None:
    """EBUSY(장치 사용 중)일 때 사용자에게 할 일을 안내한다."""
    print(
        "\n[ERROR] RealSense가 다른 프로그램에 의해 사용 중입니다 (errno=16 Device or resource busy).\n"
        f"  원본 오류: {err}\n\n"
        "  확인할 것:\n"
        "  1) 이전에 실행한 realsense_live / realsense-viewer / RViz 가 아직 떠 있는지 → q 로 종료 또는 프로세스 종료\n"
        "  2) ROS2 노드가 카메라를 쓰는지 → ros2 node list / 해당 노드 중지\n"
        "  3) 다른 터미널에서 같은 스크립트가 돌고 있는지 → Ctrl+C\n"
        "  4) USB 허브 대신 PC에 직접 연결, 케이블 재연결\n\n"
        "  점유 프로세스 확인(예):\n"
        "    fuser -v /dev/video*\n"
        "    lsof /dev/video0\n",
        file=sys.stderr,
    )
    try:
        ctx = rs.context()
        devs = list(ctx.query_devices())
        if devs:
            print("[INFO] 인식된 RealSense 장치:", file=sys.stderr)
            for d in devs:
                sn = d.get_info(rs.camera_info_serial_number)
                name = d.get_info(rs.camera_info_name)
                print(f"  - {name}  (S/N {sn})", file=sys.stderr)
        else:
            print("[INFO] librealsense가 인식한 장치: 없음", file=sys.stderr)
    except Exception as e2:
        print(f"[WARN] 장치 목록 조회 실패: {e2}", file=sys.stderr)


def setup_pipeline(
    width: int,
    height: int,
    fps: int,
    with_depth: bool,
    *,
    brightness: float | None = None,
    contrast: float | None = None,
    saturation: float | None = None,
    gain: float | None = None,
    no_auto_exposure: bool = False,
    exposure_us: float | None = None,
    pipeline_retries: int = 6,
    pipeline_retry_delay_s: float = 0.5,
) -> tuple[rs.pipeline, rs.align | None, float]:
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

    # 다른 프로세스(ROS, realsense-viewer, 이전 스크립트)가 장치를 잡고 있으면 EBUSY → 짧게 재시도
    profile: rs.pipeline_profile | None = None
    last_err: BaseException | None = None
    for attempt in range(pipeline_retries):
        try:
            profile = pipeline.start(cfg)
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            is_busy = "busy" in msg or "errno=16" in msg or "resource busy" in msg
            if is_busy and attempt + 1 < pipeline_retries:
                print(
                    f"[WARN] RealSense 장치 사용 중(EBUSY) — "
                    f"{attempt + 1}/{pipeline_retries}, {pipeline_retry_delay_s:.1f}s 후 재시도…",
                    flush=True,
                )
                time.sleep(pipeline_retry_delay_s)
            else:
                if is_busy:
                    _print_realsense_busy_help(e)
                raise
    if profile is None:
        _print_realsense_busy_help(last_err if last_err else RuntimeError("unknown"))
        raise RuntimeError(str(last_err) if last_err else "pipeline.start 실패")

    apply_color_controls(
        profile,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gain=gain,
        no_auto_exposure=no_auto_exposure,
        exposure_us=exposure_us,
    )

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


class TrackedBox:
    """위치 기반 단순 추적기.

    이전 프레임의 박스들과 현재 박스를 중심점 거리로 매칭한다.
    매칭된 박스는 클래스 버퍼에 새 값을 추가하고,
    버퍼의 다수결(mode)로 표시할 클래스와 평균 confidence를 결정한다.
    """

    def __init__(self, smooth: int) -> None:
        self.smooth = smooth
        # key: track_id(int), value: dict{cx, cy, cls_buf, conf_buf}
        self._tracks: Dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections: List[Tuple[int, int, int, float]]) -> List[Tuple[int, int, int, float]]:
        """
        detections: [(cx, cy, cls_id, conf), ...]
        반환: [(cx, cy, smoothed_cls_id, smoothed_conf), ...]
        """
        if self.smooth <= 1:
            return detections

        # 이전 트랙과 현재 검출 매칭 (greedy nearest-centroid)
        matched_ids = set()
        used_det = set()
        assignments: Dict[int, int] = {}  # track_id → det_idx

        for tid, track in self._tracks.items():
            best_d, best_j = float("inf"), -1
            for j, (cx, cy, _, _) in enumerate(detections):
                if j in used_det:
                    continue
                d = ((cx - track["cx"]) ** 2 + (cy - track["cy"]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < self._match_dist:
                assignments[tid] = best_j
                matched_ids.add(tid)
                used_det.add(best_j)

        # 매칭된 트랙 업데이트
        new_tracks: Dict[int, dict] = {}
        for tid, j in assignments.items():
            cx, cy, cls_id, conf = detections[j]
            track = self._tracks[tid]
            track["cls_buf"].append(cls_id)
            track["conf_buf"].append(conf)
            track["cx"], track["cy"] = cx, cy
            new_tracks[tid] = track

        # 새 검출 → 새 트랙 생성
        for j, det in enumerate(detections):
            if j not in used_det:
                cx, cy, cls_id, conf = det
                new_tracks[self._next_id] = {
                    "cx": cx, "cy": cy,
                    "cls_buf": deque([cls_id], maxlen=self.smooth),
                    "conf_buf": deque([conf], maxlen=self.smooth),
                }
                self._next_id += 1

        self._tracks = new_tracks

        # 다수결 + 평균 conf 계산
        result = []
        for j, (cx, cy, raw_cls, raw_conf) in enumerate(detections):
            # 이 det에 해당하는 track 찾기
            matched_tid = next((tid for tid, ji in assignments.items() if ji == j), None)
            if matched_tid is not None:
                buf_cls = list(self._tracks[matched_tid]["cls_buf"])
                buf_conf = list(self._tracks[matched_tid]["conf_buf"])
                smoothed_cls = max(set(buf_cls), key=buf_cls.count)  # 다수결
                smoothed_conf = float(np.mean(buf_conf))
            else:
                smoothed_cls = raw_cls
                smoothed_conf = raw_conf
            result.append((cx, cy, smoothed_cls, smoothed_conf))
        return result

    @property
    def _match_dist(self) -> float:
        return getattr(self, "_md", 60)


def main() -> None:
    args = parse_args()
    with_depth = not args.no_depth
    headless = bool(args.headless)

    # GUI 모드인데 OpenCV 가 headless 빌드면 즉시 안내 후 종료
    if not headless and not cv2_gui_available():
        print(
            "\n[ERROR] OpenCV 고창(GUI)을 쓸 수 없습니다. 보통 `opencv-python-headless` 가 로드된 경우입니다.\n"
            f"  현재 cv2: {getattr(cv2, '__file__', '?')}  (버전 {cv2.__version__})\n\n"
            "  해결 ① (권장): headless 제거\n"
            "    pip uninstall opencv-python-headless\n\n"
            "  해결 ②: 사용자 site-packages(~/.local)가 venv를 덮을 때\n"
            "    PYTHONNOUSERSITE=1 python3 scripts/realsense_live.py ...\n\n"
            "  해결 ③: 화면 없이 MP4로만 녹화\n"
            "    python3 scripts/realsense_live.py --headless ...\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"[INFO] loading YOLO: {args.weights}")
    model = YOLO(args.weights)
    class_names = model.names

    print(f"[INFO] starting RealSense {args.width}x{args.height}@{args.fps}fps (depth={with_depth})")
    print(f"[INFO] temporal smoothing: {'OFF' if args.smooth <= 1 else f'{args.smooth}프레임 다수결'}")
    if headless:
        print("[INFO] headless 모드: 창 없이 MP4 저장 (종료: Ctrl+C)")
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

    tracker = TrackedBox(args.smooth)
    tracker._md = args.match_dist  # type: ignore[attr-defined]

    win = "RealSense YOLO (q=quit, p=snapshot)"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # headless: MP4 라이터 (첫 프레임에서 크기 확정 후 생성)
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

                # 1차 conf 필터 후 tracker에 넘길 (cx,cy,cls,conf) 목록 구성
                raw_dets: List[Tuple[int, int, int, float, int, int, int, int]] = []
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cls_id = int(clss[i])
                    conf = float(confs[i])
                    name_raw = class_names.get(cls_id, str(cls_id))
                    if name_raw == "ripe_strawberry" and conf < args.ripe_conf:
                        continue
                    if name_raw == "unripe_strawberry" and conf < args.unripe_conf:
                        continue
                    raw_dets.append((cx, cy, cls_id, conf, x1, y1, x2, y2))

                # 시간적 평활화 (smooth>1 이면 다수결 적용)
                smooth_in = [(cx, cy, cls_id, conf) for cx, cy, cls_id, conf, *_ in raw_dets]
                smooth_out = tracker.update(smooth_in)

                for idx, (cx, cy, cls_id, conf) in enumerate(smooth_out):
                    _, _, _, _, x1, y1, x2, y2 = raw_dets[idx]
                    name = class_names.get(cls_id, str(cls_id))

                    crop = color[max(0, y1):y2, max(0, x1):x2]
                    rr = red_ratio(crop)

                    # === 색 기반 sanity check: ripe인데 빨갛지 않으면 unripe로 강제 ===
                    model_name = name  # 모델 원래 예측 보존
                    overridden = False
                    if args.min_red_for_ripe > 0 and name == "ripe_strawberry" and rr < args.min_red_for_ripe:
                        name = "unripe_strawberry"
                        cls_id = 0
                        overridden = True

                    # ripe=초록, unripe=주황, override=빨강 테두리
                    if name == "ripe_strawberry":
                        col = (0, 200, 0)
                    elif name == "unripe_strawberry":
                        col = (0, 165, 255)
                    else:
                        col = (0, 255, 255)
                    thickness = 3 if overridden else 2
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), col, thickness)
                    # override됐으면 빨간 점선 내부 테두리로 표시
                    if overridden:
                        cv2.rectangle(annotated, (x1+3, y1+3), (x2-3, y2-3), (0, 0, 220), 1)

                    # 표시: "최종클래스 conf  [모델원래예측*]"
                    short = "ripe" if name == "ripe_strawberry" else "unripe"
                    tag = f"{short} {conf:.2f}"
                    if overridden:
                        # 모델은 ripe로 봤지만 색 체크로 override됨을 표시
                        tag += f" [mdl:ripe*]"
                    label_lines = [tag, f"red={rr:.2f}"]
                    if with_depth and depth_image is not None:
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

            else:
                # 검출 없을 때도 tracker 갱신 (소멸된 트랙 정리)
                tracker.update([])

            # FPS 계산
            fps_frames += 1
            if fps_frames >= 10:
                t1 = time.time()
                fps_display = fps_frames / (t1 - fps_t0)
                fps_t0 = t1
                fps_frames = 0
                if headless:
                    print(f"[INFO] FPS ~{fps_display:.1f}", flush=True)
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

            # headless: MP4 기록 (GUI 없음)
            if headless:
                if writer is None:
                    out_mp4 = Path(
                        args.headless_out
                        if args.headless_out.strip()
                        else save_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    )
                    out_mp4.parent.mkdir(parents=True, exist_ok=True)
                    h, w = show.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(args.fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter 열기 실패: {out_mp4}")
                    print(f"[INFO] MP4 저장 시작: {out_mp4.resolve()}")
                writer.write(show)
            else:
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
        if writer is not None:
            writer.release()
            if out_mp4 is not None:
                print(f"[INFO] MP4 저장 종료: {out_mp4.resolve()}")
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
