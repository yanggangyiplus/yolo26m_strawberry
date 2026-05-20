"""딸기 bbox 상단 ROI · CLAHE · HSV 줄기 강조 — 줄기 인식 파이프라인 공통 유틸.

딸기는 아래로 매달리므로 줄기는 과실 bbox 상단에 위치한다는 전제로
상단 ROI만 잘라 seg 모델 입력 노이즈를 줄인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# YOLO 정규화 bbox → 픽셀 xyxy
def yolo_bbox_to_xyxy(
    cx: float, cy: float, w: float, h: float, iw: int, ih: int
) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    return x1, y1, x2, y2


@dataclass
class RoiCrop:
    """전체 이미지 좌표계에서의 상단 ROI."""

    x1: int
    y1: int
    x2: int
    y2: int
    fruit_idx: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


def compute_top_roi(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    iw: int,
    ih: int,
    *,
    above_ratio: float = 0.55,
    into_fruit_ratio: float = 0.08,
    pad_x_ratio: float = 0.15,
    min_size: int = 32,
) -> tuple[int, int, int, int] | None:
    """과실 bbox 기준 상단 ROI (줄기·윗동 포함).

    above_ratio: 과실 높이 대비 bbox 위쪽으로 확장하는 비율
    into_fruit_ratio: 윗동 포함을 위해 과실 상단에서 아래로 살짝 겹침
    pad_x_ratio: 가로 패딩 (bbox 너비 비율)
    """
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    pad_x = int(bw * pad_x_ratio)
    extend_up = int(bh * above_ratio)
    into_fruit = int(bh * into_fruit_ratio)

    rx1 = max(0, x1 - pad_x)
    rx2 = min(iw, x2 + pad_x)
    ry2 = min(ih, y2, y1 + into_fruit)
    ry1 = max(0, y1 - extend_up)

    if rx2 - rx1 < min_size or ry2 - ry1 < min_size:
        return None
    return rx1, ry1, rx2, ry2


def apply_clahe_bgr(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """LAB L채널 CLAHE — 그늘·하이라이트에서 줄기 외곽선 강조."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def apply_hsv_stem_boost(bgr: np.ndarray, blend: float = 0.35) -> np.ndarray:
    """HSV에서 줄기·꼭지(황록~갈색) 영역을 G채널로 가중 강조."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # OpenCV H: 0–179, 줄기·잎·꼭지
    mask = cv2.inRange(hsv, (18, 35, 35), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    boost = np.zeros_like(bgr)
    boost[:, :, 1] = mask
    return cv2.addWeighted(bgr, 1.0, boost, blend, 0)


def preprocess_stem_roi(
    bgr: np.ndarray,
    *,
    clahe: bool = True,
    hsv_boost: bool = True,
) -> np.ndarray:
    """줄기 seg 모델 입력 전처리 (학습·추론 동일 적용)."""
    out = bgr
    if clahe:
        out = apply_clahe_bgr(out)
    if hsv_boost:
        out = apply_hsv_stem_boost(out)
    return out


def clip_polygon_to_roi(
    pts_norm: list[tuple[float, float]],
    roi: RoiCrop,
    iw: int,
    ih: int,
) -> list[tuple[float, float]] | None:
    """전체 이미지 정규화 폴리곤 → ROI 내부 정규화 폴리곤."""
    rx1, ry1, rw, rh = roi.x1, roi.y1, roi.width, roi.height
    inner: list[tuple[float, float]] = []
    for nx, ny in pts_norm:
        px, py = nx * iw, ny * ih
        if rx1 <= px <= roi.x2 and ry1 <= py <= roi.y2:
            inner.append(((px - rx1) / rw, (py - ry1) / rh))
    if len(inner) < 3:
        return None
    return inner


def map_point_roi_to_full_px(
    x_norm: float, y_norm: float, roi: RoiCrop
) -> tuple[int, int]:
    px = int(roi.x1 + x_norm * roi.width)
    py = int(roi.y1 + y_norm * roi.height)
    return px, py


def map_point_roi_to_full_norm(
    x_norm: float, y_norm: float, roi: RoiCrop, iw: int, ih: int
) -> tuple[float, float]:
    px, py = map_point_roi_to_full_px(x_norm, y_norm, roi)
    return px / iw, py / ih


def offset_grip_up_px(
    px: int,
    py: int,
    *,
    depth_m: float,
    fy_px: float,
    margin_m: float = 0.01,
    ih: int,
) -> tuple[int, int]:
    """그립점을 줄기 쪽(이미지 위, y 감소)으로 margin_m 만큼 올린 픽셀 좌표.

    pinhole 근사: delta_y_px = (margin_m / depth_m) * fy
    depth 미측정 시 0.5m 가정.
    """
    d = depth_m if depth_m > 0 else 0.5
    dy = int(round((margin_m / d) * fy_px))
    dy = max(1, dy) if margin_m > 0 else 0
    return px, int(np.clip(py - dy, 0, ih - 1))


def mask_centroid_in_roi(r, conf_min: float) -> tuple[float, float] | None:
    """seg 결과 마스크 중심 (ROI 정규화 0–1)."""
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return None
    h, w = r.orig_shape
    best_conf = -1.0
    best_xy: tuple[float, float] | None = None
    confs = r.boxes.conf.cpu().numpy()
    for i, xy in enumerate(r.masks.xy):
        if float(confs[i]) < conf_min or xy is None or len(xy) < 3:
            continue
        if float(confs[i]) > best_conf:
            cx = float(np.mean(xy[:, 0])) / w
            cy = float(np.mean(xy[:, 1])) / h
            best_conf = float(confs[i])
            best_xy = (cx, cy)
    return best_xy


def format_yolo_seg_line(cls_id: int, pts: Iterable[tuple[float, float]]) -> str:
    coords = " ".join(f"{float(x):.6f} {float(y):.6f}" for x, y in pts)
    return f"{cls_id} {coords}"
