"""변환된 YOLO 데이터셋(images/all, labels/all)을 train/val/test로 분할.

사용 예:
    python scripts/split_dataset.py \
        --root datasets/yolo \
        --train 0.8 --val 0.1 --test 0.1 \
        --seed 42

결과: datasets/yolo/{images,labels}/{train,val,test} 에 symlink로 분배.
labels/all 에 있는 이미지만 학습 대상에 포함 (라벨 없는 이미지는 제외).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    """src 파일을 dst 경로로 symlink (또는 복사). 기존 파일/링크는 덮어씀."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="datasets/yolo 루트")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="symlink 대신 복사")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test = {total} != 1.0")

    root = args.root.expanduser().resolve()
    img_all = root / "images" / "all"
    lbl_all = root / "labels" / "all"
    if not img_all.is_dir() or not lbl_all.is_dir():
        raise SystemExit(f"images/all 또는 labels/all 폴더가 없습니다: {root}")

    # 라벨 파일을 기준으로 train 대상 집합 결정 (라벨 없는 이미지는 학습에서 제외)
    label_files = sorted(lbl_all.glob("*.txt"))
    pairs: list[tuple[Path, Path]] = []
    for lp in label_files:
        # 같은 stem을 갖는 이미지 찾기 (확장자 후보 순회)
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            ip = img_all / f"{lp.stem}{ext}"
            if ip.exists():
                pairs.append((ip, lp))
                break

    if not pairs:
        raise SystemExit("매칭되는 (이미지, 라벨) 쌍이 없습니다.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, items in splits.items():
        for img, lbl in items:
            link_or_copy(img, root / "images" / split / img.name, copy=args.copy)
            link_or_copy(lbl, root / "labels" / split / lbl.name, copy=args.copy)
        print(f"[INFO] {split}: {len(items)}장")

    print(f"\n총 {n}장을 train/val/test = {args.train}/{args.val}/{args.test}로 분할 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
