#!/usr/bin/env python3
"""줄기(stem) seg 라벨을 이미지 위에 그려 확인.

사용:
  # realscene 7장만 PNG로 저장 (가장 먼저 권장)
  python scripts/vis_stem_seg.py --mode realscene --out runs/stem_label_review

  # 키보드로 한 장씩 넘기며 확인 (n 다음, p 이전, q 종료)
  python scripts/vis_stem_seg.py --mode realscene --interactive

  # val 37장 전부 저장
  python scripts/vis_stem_seg.py --mode val --out runs/stem_label_review

  # detect 박스도 같이 표시
  python scripts/vis_stem_seg.py --mode realscene --show-detect
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "datasets" / "yolo_unified_seg"
FARM_LBL = ROOT / "datasets" / "yolo_unified_farm" / "labels" / "all"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DET_NAMES = ("unripe", "ripe")


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def parse_seg_line(line: str) -> tuple[int, list[tuple[float, float]]] | None:
    parts = line.strip().split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None
    cls_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return cls_id, pts


def parse_detect_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def draw_poly(
    img: np.ndarray,
    pts_norm: list[tuple[float, float]],
    color: tuple[int, int, int],
    label: str,
) -> None:
    h, w = img.shape[:2]
    pts_px = np.array([[int(x * w), int(y * h)] for x, y in pts_norm], dtype=np.int32)
    if len(pts_px) < 3:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_px], color)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [pts_px], True, color, 2)
    cx, cy = int(pts_px[:, 0].mean()), int(pts_px[:, 1].mean())
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_detect(img: np.ndarray, lbl_path: Path) -> None:
    if not lbl_path.is_file():
        return
    h, w = img.shape[:2]
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = parse_detect_line(line)
        if p is None:
            continue
        cls_id, cx, cy, bw, bh = p
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = (0, 165, 255) if cls_id == 0 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            img,
            DET_NAMES[cls_id] if cls_id < 2 else str(cls_id),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def render_one(
    stem: str,
    img_dir: Path,
    lbl_dir: Path,
    show_detect: bool,
) -> np.ndarray | None:
    img_path = find_image(img_dir, stem)
    lbl_path = lbl_dir / f"{stem}.txt"
    if img_path is None or not lbl_path.is_file():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    if show_detect:
        draw_detect(img, FARM_LBL / f"{stem}.txt")

    for i, line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines()):
        parsed = parse_seg_line(line)
        if parsed is None:
            continue
        _cls, pts = parsed
        draw_poly(img, pts, (0, 255, 0), f"stem{i}")

    cv2.putText(
        img,
        stem,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(img, stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return img


def collect_stems(mode: str, split: str) -> list[str]:
    if mode == "realscene":
        img_dir = SEG_ROOT / "images" / "all"
        return sorted(p.stem for p in img_dir.glob("realscene_*") if p.is_file())
    if mode == "qin_sample":
        img_dir = SEG_ROOT / "images" / "all"
        return sorted(p.stem for p in img_dir.glob("qin_*") if p.is_file())[:20]
    img_dir = SEG_ROOT / "images" / split
    lbl_dir = SEG_ROOT / "labels" / split
    if not img_dir.is_dir():
        raise SystemExit(f"폴더 없음: {img_dir} (먼저 split_dataset.py 실행)")
    stems = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {e.lstrip(".") for e in IMG_EXTS}:
            if (lbl_dir / f"{p.stem}.txt").is_file():
                stems.append(p.stem)
    return stems


def main() -> int:
    ap = argparse.ArgumentParser(description="줄기 seg 라벨 시각화·수동 확인")
    ap.add_argument(
        "--mode",
        choices=("realscene", "val", "train", "test", "all", "qin_sample"),
        default="realscene",
        help="realscene=7장 권장, val=검증셋 전체",
    )
    ap.add_argument("--split", default="all", help="mode가 val/train/test일 때 사용하는 split 이름")
    ap.add_argument("--out", type=Path, default=Path("runs/stem_label_review"))
    ap.add_argument("--interactive", action="store_true", help="OpenCV 창에서 n/p/q 로 넘기기")
    ap.add_argument("--show-detect", action="store_true", help="농장 detect 박스도 표시")
    ap.add_argument("--stem", type=str, default="", help="한 장만: 예 realscene_000")
    args = ap.parse_args()

    split = args.mode if args.mode in ("val", "train", "test", "all") else "all"
    img_dir = SEG_ROOT / "images" / split
    lbl_dir = SEG_ROOT / "labels" / split

    if args.stem:
        stems = [args.stem]
    else:
        stems = collect_stems(args.mode, split)

    if not stems:
        raise SystemExit("대상 이미지가 없습니다.")

    print(f"[vis] {len(stems)}장, img={img_dir}, lbl={lbl_dir}")

    if args.interactive:
        idx = 0
        win = "stem_seg review (n=next p=prev q=quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            s = stems[idx]
            frame = render_one(s, img_dir, lbl_dir, args.show_detect)
            if frame is None:
                print(f"[skip] {s}")
                idx = (idx + 1) % len(stems)
                continue
            cv2.imshow(win, frame)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord(" ")):
                idx = (idx + 1) % len(stems)
            elif key == ord("p"):
                idx = (idx - 1) % len(stems)
        cv2.destroyAllWindows()
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for s in stems:
        frame = render_one(s, img_dir, lbl_dir, args.show_detect)
        if frame is None:
            print(f"[skip] {s}")
            continue
        out_path = args.out / f"{s}_stem_vis.jpg"
        cv2.imwrite(str(out_path), frame)
        n_ok += 1
    print(f"[done] saved {n_ok} images -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
