"""YOLO26m-seg — 딸기 과실 + 줄기 instance segmentation 학습.

사전 요건:
  - datasets/yolo_unified_seg/labels/all/*.txt 가 YOLO seg 폴리곤 형식
  - python scripts/setup_seg_dataset.py && split_dataset.py

사용 예:
    YOLO_CONFIG_DIR=$(pwd)/.ultralytics_cfg python scripts/train_seg.py \\
        --data configs/strawberry_unified_seg.yaml \\
        --epochs 100 --imgsz 640 --batch 8 --device 0 \\
        --name yolo26m_unified_seg_640b8

첫 실행 시 yolo26m-seg.pt 가 Ultralytics에서 자동 다운로드된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from train import clean_stale_caches  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO26m-seg fine-tuning (과일+줄기 마스크)")
    ap.add_argument(
        "--weights",
        type=str,
        default="yolo26m-seg.pt",
        help="사전학습 seg 가중치 (yolo26m-seg.pt)",
    )
    ap.add_argument(
        "--data",
        type=str,
        default="configs/strawberry_stem_seg.yaml",
        help="기본: 줄기 단일 클래스 (strawberry_stem_seg.yaml)",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="줄기는 가늘어 832도 고려")
    ap.add_argument("--batch", type=int, default=8, help="seg는 VRAM을 더 씀 — 16GB면 8~12")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/strawberry")
    ap.add_argument("--name", type=str, default="yolo26m_unified_seg")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--aug-strong",
        action="store_true",
        help="줄기 ROI용 강한 조명 증강 (hsv_v/s/mosaic)",
    )
    return ap.parse_args()


def count_seg_labels(data_yaml: Path) -> int:
    """labels/all 또는 train 에 있는 폴리곤 라벨 개수(대략)."""
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except Exception:
        return 0
    base = Path(cfg.get("path", "")).expanduser()
    for sub in ("all", "train"):
        d = base / "labels" / sub
        if not d.is_dir():
            continue
        n = sum(1 for p in d.glob("*.txt") if p.name != "classes.txt")
        if n:
            return n
    return 0


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    clean_stale_caches(data_path)

    n_lbl = count_seg_labels(data_path)
    if n_lbl == 0:
        print(
            "[WARN] seg 라벨이 없습니다. 먼저:\n"
            "  1) python scripts/setup_seg_dataset.py\n"
            "  2) CVAT/labelme 등으로 stem·과실 폴리곤 라벨 → labels/all/\n"
            "  3) python scripts/split_dataset.py --root datasets/yolo_unified_seg\n"
        )

    model = YOLO(args.weights)
    # 줄기 ROI: 조명 변화에 강건하게 hsv·mosaic 강화
    hsv_h, hsv_s, hsv_v = (0.02, 0.7, 0.6) if args.aug_strong else (0.015, 0.5, 0.4)
    mosaic = 1.0 if not args.aug_strong else 0.8
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=mosaic,
        mixup=0.0,
        copy_paste=0.0,
        # --- seg: 얇은 줄기 마스크 ---
        overlap_mask=True,
        mask_ratio=2,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        cache=False,
        amp=True,
        plots=True,
        save=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"\n[INFO] validating seg model: {best}")
        YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)


if __name__ == "__main__":
    main()
