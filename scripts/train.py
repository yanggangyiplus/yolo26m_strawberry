"""YOLO26m 딸기 fine-tuning 스크립트 (Ultralytics YOLO26).

통합 데이터(UniqueData + Qin2006 + 사용자 사진)의 2-class(unripe/ripe) 검출 학습.

수확 로봇 시나리오 가정:
    - 데이터 규모에 맞춰 augmentation·patience 조정
    - 조명/색감 변화 대응 → HSV/플립/모자이크 증강
    - mAP@50-95(박스 위치 정확도)도 함께 모니터링

사용 예 (GPU, 기본 가중치 yolo26m.pt):
    YOLO_CONFIG_DIR=$(pwd)/.ultralytics_cfg python scripts/train.py \
        --weights yolo26m.pt \
        --data configs/strawberry.yaml \
        --epochs 100 --imgsz 640 --batch 16 --device 0

CPU에서 빠른 테스트:
    python scripts/train.py --data configs/strawberry.yaml \
        --epochs 3 --imgsz 640 --batch 4 --device cpu

YOLO11 계속 쓰려면: --weights yolo11m.pt (동일 스크립트).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def clean_stale_caches(data_yaml: Path) -> None:
    """data.yaml 의 path/{train,val,test} 옆에 있는 손상된(0바이트) .cache 파일 제거.

    Ultralytics는 학습 중단 시 종종 빈 cache 파일을 남기는데,
    다음 실행 때 numpy.load 가 EOFError 로 실패한다. 안전하게 모두 제거 후 재생성하도록 한다.
    """
    if not data_yaml.is_file():
        return
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except Exception:
        return
    base = Path(cfg.get("path", data_yaml.parent)).expanduser()
    for split_key in ("train", "val", "test"):
        split = cfg.get(split_key)
        if not split:
            continue
        # split 값이 'images/train' 형식이면 'labels/train' 으로 치환해 cache 위치 추정
        labels_dir = base / str(split).replace("images", "labels", 1)
        cache = labels_dir.with_suffix(".cache")
        if cache.exists() and cache.stat().st_size == 0:
            print(f"[clean] empty cache removed: {cache}")
            cache.unlink()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="yolo26m.pt", help="사전학습 가중치 (예: yolo26m.pt, yolo11m.pt)")
    ap.add_argument("--data", type=str, default="configs/strawberry.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="입력 해상도. 작은 박스가 많으면 800/960으로 올려도 됨")
    ap.add_argument("--batch", type=int, default=16, help="GPU 메모리에 맞춰 조정 (-1: auto)")
    ap.add_argument("--device", type=str, default="0", help="cuda 인덱스 또는 'cpu'")
    ap.add_argument("--project", type=str, default="runs/strawberry")
    ap.add_argument("--name", type=str, default="yolo26m_ft")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30, help="early stopping 인내 epoch")
    ap.add_argument("--resume", action="store_true", help="이전 학습 이어서")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clean_stale_caches(Path(args.data))
    model = YOLO(args.weights)

    # 학습 실행. 주요 augmentation은 농작물 도메인에 맞춰 약간 보수적으로 둠.
    # - mosaic: 마지막 10 epoch 자동 close (Ultralytics 기본)
    # - hsv_*: 조명/색감 변화에 강건성 부여 (딸기 익은 정도 색감을 너무 흔들면 안 되므로 적당히)
    # - degrees/translate: 카메라 흔들림/이동 대응
    # - fliplr=0.5: 좌우 대칭은 문제 없음 / flipud=0.0: 위아래 뒤집기는 비현실적이라 끔
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
        # --- augmentation ---
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # --- optimizer ---
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 작은 객체(stem) 검출 가중치 살짝 올리기
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # --- 기타 ---
        cache=False,  # 데이터가 크면 RAM 부족 위험. 빠르게 돌리려면 'ram' 또는 'disk' 권장
        amp=True,
        plots=True,
        save=True,
    )

    # 학습 종료 후 best.pt로 val 한 번 더 (요약 메트릭 출력)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"\n[INFO] validating with best weights: {best}")
        YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)


if __name__ == "__main__":
    main()
