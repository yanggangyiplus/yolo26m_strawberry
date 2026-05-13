"""파인튜닝 전(yolo26m.pt, COCO 80-class) / 후(best.pt, 2-class strawberry) 모델 추론 비교 스크립트.

같은 입력(폴더 또는 단일 이미지)에 대해 두 모델을 각각 실행하고,
시각화 결과를 별도 폴더에 저장한다. 콘솔에는 이미지별 검출 개수를 요약 출력.

사용 예:
    # 기본: 테스트 셋 전체로 비교
    python scripts/run_compare.py

    # 가중치/입력/출력 직접 지정
    python scripts/run_compare.py \
        --pre yolo26m.pt \
        --post runs/detect/runs/strawberry/yolo26m_ft-5/weights/best.pt \
        --source datasets/yolo/images/test \
        --out runs/compare \
        --conf 0.25 --imgsz 640 --device 0

    # CPU 강제
    python scripts/run_compare.py --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱. 기본값은 본 워크스페이스의 표준 경로를 사용한다."""
    ap = argparse.ArgumentParser()
    # 파인튜닝 전 모델 (사전학습 COCO 가중치)
    ap.add_argument("--pre", type=str, default="yolo26m.pt",
                    help="파인튜닝 전 가중치 (예: yolo26m.pt, yolo11m.pt)")
    # 파인튜닝 후 모델 (학습 결과 best.pt)
    ap.add_argument("--post", type=str,
                    default="runs/detect/runs/strawberry/yolo26m_ft-5/weights/best.pt",
                    help="파인튜닝 후 가중치 (best.pt)")
    ap.add_argument("--source", type=str, default="datasets/yolo/images/test",
                    help="추론 대상 폴더 또는 단일 이미지/비디오")
    ap.add_argument("--out", type=str, default="runs/compare",
                    help="시각화 결과를 저장할 상위 폴더")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0", help="cuda 인덱스 또는 'cpu'")
    return ap.parse_args()


def run_one(weights: str, source: str, out_dir: Path, *, conf: float, iou: float,
            imgsz: int, device: str, tag: str) -> None:
    """하나의 가중치로 추론 후 시각화 이미지를 out_dir/tag 에 저장하고 검출 요약 출력."""
    if not Path(weights).exists():
        # 사전학습 가중치는 Ultralytics가 자동 다운로드 시도하므로 경고만 출력
        print(f"[WARN] '{weights}' 파일이 로컬에 없습니다. (자동 다운로드 시도)")

    print(f"\n[{tag}] weights = {weights}")
    model = YOLO(weights)

    # save=True 로 박스 그려진 이미지를 자동 저장.
    # project/name 으로 출력 폴더를 명시적으로 고정 (덮어쓰기 허용).
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(out_dir),
        name=tag,
        exist_ok=True,
        verbose=False,
    )

    names = model.names  # 클래스 id → 이름 매핑
    total = 0
    print(f"  {'image':<40} {'#det':>5}  classes")
    print("  " + "-" * 70)
    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        total += n
        # 검출된 클래스 분포 (이름:개수)
        if n > 0:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
            counts: dict[str, int] = {}
            for c in cls_ids:
                k = names.get(int(c), str(c))
                counts[k] = counts.get(k, 0) + 1
            cls_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        else:
            cls_str = "-"
        print(f"  {Path(r.path).name:<40} {n:>5}  {cls_str}")
    print(f"  → total detections: {total}")
    print(f"  → saved: {out_dir / tag}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 파인튜닝 전: COCO 80-class 모델로 추론. 보통 'strawberry' 클래스가 없어
    #    apple/orange 등으로 잘못 잡히거나 거의 검출되지 않는 것이 정상.
    run_one(args.pre, args.source, out_dir,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
            tag="pre_finetune")

    # 2) 파인튜닝 후: 2-class(unripe/ripe strawberry) 모델로 추론.
    run_one(args.post, args.source, out_dir,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
            tag="post_finetune")

    print(f"\n[DONE] 비교 결과 폴더: {out_dir.resolve()}")
    print("       - pre_finetune/   (파인튜닝 전, COCO 80-class)")
    print("       - post_finetune/  (파인튜닝 후, strawberry 2-class)")


if __name__ == "__main__":
    main()
