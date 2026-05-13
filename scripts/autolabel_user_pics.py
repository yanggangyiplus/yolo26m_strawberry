"""사용자 사진(라벨 없음) 자동 라벨링 → YOLO 포맷.

대상 폴더 (모두 선택적):
    --raw-dir              raw_str/                                 → class 0 (분홍 모형)
    --red-dir              red_str/                                 → class 1 (빨강 모형)
    --strawberry-pickable  strawberryDataset/Pickable/              → class 1 (실제 익은 딸기)
    --strawberry-unpickable strawberryDataset/UnPickable/           → class 0 (실제 덜 익은 딸기)

라벨 생성 방법:
    1) 폴더별 색상 마스크 생성
       - ripe / real_ripe       : 빨강 HSV 영역
       - unripe                 : 연분홍 HSV (모형 raw_str 전용)
       - real_unripe            : 연두/흰색/연분홍 HSV (실제 덜 익은 딸기)
    2) 마스크에서 최대 연결 컴포넌트(BFS) bbox 추출
    3) bbox 면적이 너무 작거나 마스크가 비면 옵션에 따라 "이미지 전체"를 폴백 bbox로 사용
       (strawberryDataset 처럼 close-up classification 이미지에 유리)
    4) 시각화 이미지(bbox 그린) 함께 저장 → 사용자 검토용

옵션:
    --visualize-dir       시각화 결과 저장 폴더 (기본 runs/autolabel_vis)
    --min-area            최소 bbox 면적 비율 (기본 0.01)
    --max-area            최대 bbox 면적 비율 (기본 0.7)
    --whole-image-fallback  마스크 실패 시 이미지 전체(약간 안쪽)를 bbox로 사용 (default=True)

사용 예 (4개 폴더 모두):
    python3 scripts/autolabel_user_pics.py \
        --raw-dir datasets/raw/data/images/raw_str \
        --red-dir datasets/raw/data/images/red_str \
        --strawberry-pickable datasets/raw/data/images/strawberryDataset/Pickable \
        --strawberry-unpickable datasets/raw/data/images/strawberryDataset/UnPickable \
        --dst datasets/yolo_3folders
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


UNRIPE_CLS = 0
RIPE_CLS = 1


def color_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    """RGB 이미지에서 'ripe(빨강)' 또는 'unripe(분홍)' 영역의 binary mask.

    OpenCV 없이 numpy로 HSV 변환 (기본 라이브러리만 사용).
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn
    safe_diff = np.where(diff == 0, 1, diff)

    h = np.zeros_like(mx)
    mask_r = (mx == r) & (diff > 0)
    mask_g = (mx == g) & (diff > 0)
    mask_b = (mx == b) & (diff > 0)
    h[mask_r] = (60 * ((g - b) / safe_diff) % 360)[mask_r]
    h[mask_g] = (60 * ((b - r) / safe_diff) + 120)[mask_g]
    h[mask_b] = (60 * ((r - g) / safe_diff) + 240)[mask_b]
    s = np.where(mx == 0, 0.0, diff / np.where(mx == 0, 1, mx))
    v = mx

    if mode == "ripe":
        # 진한 빨강(모형 red_str): H ∈ [0,15] ∪ [340,360], S≥0.35, V≥0.20
        return ((h <= 15) | (h >= 340)) & (s >= 0.35) & (v >= 0.20)
    elif mode == "unripe":
        # 연분홍(모형 raw_str): 빨강 계열인데 채도 낮고 명도 높음
        # H ∈ [0,25] ∪ [330,360], S ∈ [0.10, 0.55], V ≥ 0.6
        return (((h <= 25) | (h >= 330)) & (s >= 0.10) & (s <= 0.55) & (v >= 0.60))
    elif mode == "real_ripe":
        # 실제 익은 딸기(strawberryDataset/Pickable):
        # 자연광/그림자 환경 → 빨강 임계값 약간 완화
        # H ∈ [0,18] ∪ [335,360], S ≥ 0.25, V ≥ 0.15
        return ((h <= 18) | (h >= 335)) & (s >= 0.25) & (v >= 0.15)
    elif mode == "real_unripe":
        # 실제 덜 익은 딸기(strawberryDataset/UnPickable):
        # 흰색/크림(저채도, 고명도) + 연두/녹색(딸기 자체가 녹색) + 분홍(전이 단계) 모두 포함
        # 단, 잎/배경 녹색과 구분이 어려우므로 호출부에서 whole-image fallback과 함께 사용 권장.
        white_cream = (s <= 0.25) & (v >= 0.55)
        green_berry = (h >= 60) & (h <= 150) & (s >= 0.20) & (v >= 0.25)
        pink_trans = (((h <= 25) | (h >= 330)) & (s >= 0.15) & (s <= 0.55) & (v >= 0.55))
        return white_cream | green_berry | pink_trans
    else:
        raise ValueError(f"unknown mode: {mode}")


def largest_component_bbox(mask: np.ndarray, min_area: int) -> tuple[int, int, int, int] | None:
    """이진 마스크에서 가장 큰 연결 컴포넌트의 bbox(x1,y1,x2,y2). 면적이 min_area 미만이면 None.

    OpenCV 의존을 피하려고 단순 flood-fill (BFS) 구현.
    """
    if not mask.any():
        return None
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[int, int, int, int] | None = None
    best_area = 0

    # 효율 위해 numpy로 row/col 후보 좁힘
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    # 매우 큰 이미지에서 전체 BFS는 부담이라, 다운샘플링한 마스크에서 컴포넌트 찾고 좌표 복원
    DOWN = 4 if max(H, W) > 1500 else 2 if max(H, W) > 800 else 1
    if DOWN > 1:
        small = mask[::DOWN, ::DOWN]
    else:
        small = mask
    sH, sW = small.shape

    from collections import deque

    visited_small = np.zeros_like(small, dtype=bool)
    for y0 in range(sH):
        row = small[y0]
        for x0 in np.where(row & ~visited_small[y0])[0]:
            # BFS
            q = deque([(y0, x0)])
            visited_small[y0, x0] = True
            xs_c, ys_c = [], []
            while q:
                y, x = q.popleft()
                xs_c.append(x)
                ys_c.append(y)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < sH and 0 <= nx < sW and small[ny, nx] and not visited_small[ny, nx]:
                        visited_small[ny, nx] = True
                        q.append((ny, nx))
            area = len(xs_c) * DOWN * DOWN
            if area >= min_area and area > best_area:
                best_area = area
                x1 = min(xs_c) * DOWN
                y1 = min(ys_c) * DOWN
                x2 = (max(xs_c) + 1) * DOWN
                y2 = (max(ys_c) + 1) * DOWN
                best = (x1, y1, x2, y2)
    return best


def expand_bbox(bbox: tuple[int, int, int, int], W: int, H: int, pad: float = 0.08) -> tuple[int, int, int, int]:
    """bbox를 약간 확장 (모델 학습 시 가장자리 정보 포함)."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))


def xyxy_to_yolo_norm(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> tuple[float, float, float, float]:
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def draw_bbox(rgb: np.ndarray, bbox: tuple[int, int, int, int], color=(0, 255, 0), thickness: int = 6) -> np.ndarray:
    """OpenCV 없이 numpy로 bbox 직사각형 그리기 (시각화용)."""
    out = rgb.copy()
    x1, y1, x2, y2 = bbox
    t = thickness
    out[y1:y1 + t, x1:x2] = color
    out[y2 - t:y2, x1:x2] = color
    out[y1:y2, x1:x1 + t] = color
    out[y1:y2, x2 - t:x2] = color
    return out


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def process_folder(
    src_dir: Path,
    cls_id: int,
    cls_mode: str,
    prefix: str,
    out_images: Path,
    out_labels: Path,
    vis_dir: Path,
    min_area_ratio: float,
    max_area_ratio: float,
    copy_images: bool,
    stats: Counter,
    whole_image_fallback: bool = True,
) -> None:
    from PIL import Image  # 지연 import

    if not src_dir.is_dir():
        print(f"[WARN] {src_dir} 폴더 없음, 스킵", file=sys.stderr)
        return

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for idx, img_path in enumerate(files):
        try:
            with Image.open(img_path) as im:
                # EXIF orientation 반영 (iPhone 사진 회전 처리)
                im = im.convert("RGB")
                try:
                    from PIL import ImageOps
                    im = ImageOps.exif_transpose(im)
                except Exception:
                    pass
                rgb = np.array(im)
        except Exception as e:
            stats[f"{prefix}_open_error"] += 1
            print(f"[WARN] cannot open {img_path}: {e}", file=sys.stderr)
            continue

        H, W = rgb.shape[:2]
        mask = color_mask(rgb, mode=cls_mode)
        min_area_px = int(W * H * min_area_ratio)
        bbox = largest_component_bbox(mask, min_area=min_area_px)

        used_fallback = False
        if bbox is None:
            if whole_image_fallback:
                # close-up classification 이미지(예: strawberryDataset)는 객체가 거의 전체를 차지함.
                # 마스크 검출 실패 시 이미지 안쪽 90% 영역을 bbox로 사용한다.
                pad_x, pad_y = int(W * 0.05), int(H * 0.05)
                bbox = (pad_x, pad_y, W - pad_x, H - pad_y)
                used_fallback = True
                stats[f"{prefix}_fallback_whole"] += 1
            else:
                stats[f"{prefix}_no_object"] += 1
                print(f"[WARN] {img_path.name}: 객체 미검출 — 수동 라벨 필요", file=sys.stderr)
                continue

        if used_fallback:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = expand_bbox(bbox, W, H, pad=0.08)
        area_ratio = (x2 - x1) * (y2 - y1) / (W * H)
        if area_ratio > max_area_ratio and not used_fallback:
            stats[f"{prefix}_too_large"] += 1
            print(f"[WARN] {img_path.name}: bbox 너무 큼({area_ratio:.2f}), 배경 포함 가능 — 수동 검토",
                  file=sys.stderr)
        cx, cy, w, h = xyxy_to_yolo_norm(x1, y1, x2, y2, W, H)

        stem = f"{prefix}_{idx:03d}"
        out_lbl = out_labels / f"{stem}.txt"
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        link_or_copy(img_path, out_images / f"{stem}{img_path.suffix}", copy=copy_images)

        # 시각화 저장 (fallback 사용 시 색상 다르게 → 검토 시 식별 용이)
        if used_fallback:
            color = (255, 255, 0)  # 노랑: fallback whole-image
        elif cls_id == RIPE_CLS:
            color = (0, 255, 0)
        else:
            color = (255, 128, 0)
        vis = draw_bbox(rgb, (x1, y1, x2, y2), color=color)
        vis_path = vis_dir / f"{stem}.jpg"
        vis_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(vis_path, quality=70)

        stats[f"{prefix}_ok"] += 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("datasets/raw/data/images/raw_str"),
                    help="분홍 모형 (class 0). 빈 문자열로 비활성화 가능")
    ap.add_argument("--red-dir", type=Path, default=Path("datasets/raw/data/images/red_str"),
                    help="빨강 모형 (class 1)")
    ap.add_argument("--strawberry-pickable", type=Path, default=None,
                    help="strawberryDataset/Pickable 경로 (실제 익은 딸기, class 1)")
    ap.add_argument("--strawberry-unpickable", type=Path, default=None,
                    help="strawberryDataset/UnPickable 경로 (실제 덜 익은 딸기, class 0)")
    ap.add_argument("--dst", type=Path, default=Path("datasets/yolo"))
    ap.add_argument("--visualize-dir", type=Path, default=Path("runs/autolabel_vis"))
    ap.add_argument("--min-area", type=float, default=0.01, help="최소 bbox 면적 비율")
    ap.add_argument("--max-area", type=float, default=0.7, help="최대 bbox 면적 비율")
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument("--no-fallback", action="store_true",
                    help="마스크 실패 시 이미지 전체 bbox fallback 비활성화")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dst = args.dst.expanduser().resolve()
    out_images = dst / "images" / "all"
    out_labels = dst / "labels" / "all"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    vis_dir = args.visualize_dir.expanduser().resolve()

    stats: Counter = Counter()

    # 처리할 폴더 목록 빌드: (src_dir, cls_id, cls_mode, prefix, whole_image_fallback)
    jobs: list[tuple[Path | None, int, str, str, bool]] = [
        (args.raw_dir, UNRIPE_CLS, "unripe", "user_raw", False),
        (args.red_dir, RIPE_CLS, "ripe", "user_red", False),
        (args.strawberry_pickable, RIPE_CLS, "real_ripe", "straw_pick", True),
        (args.strawberry_unpickable, UNRIPE_CLS, "real_unripe", "straw_unpick", True),
    ]

    print(f"[INFO] dst     = {dst}")
    print(f"[INFO] vis_dir = {vis_dir}")
    for src, cls, mode, prefix, fb in jobs:
        if src is None or str(src).strip() == "":
            continue
        src_resolved = Path(src).expanduser().resolve()
        if not src_resolved.is_dir():
            print(f"[SKIP] {prefix}: {src_resolved} 폴더 없음")
            continue
        fb_eff = fb and not args.no_fallback
        print(f"[INFO] {prefix:12s} cls={cls} mode={mode:12s} fallback={fb_eff}  src={src_resolved}")
        process_folder(
            src_resolved,
            cls_id=cls, cls_mode=mode, prefix=prefix,
            out_images=out_images, out_labels=out_labels,
            vis_dir=vis_dir,
            min_area_ratio=args.min_area, max_area_ratio=args.max_area,
            copy_images=args.copy_images, stats=stats,
            whole_image_fallback=fb_eff,
        )

    print("\n=== 자동 라벨링 결과 ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\n시각화: {vis_dir} 폴더에서 bbox 가 정확한지 꼭 확인하세요.")
    print("부정확한 라벨은 labels/all/ 에서 직접 수정하거나 .txt 파일을 삭제하면 됩니다.")
    print("(노란 박스 = whole-image fallback, 초록/주황 = HSV 마스크 기반)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
