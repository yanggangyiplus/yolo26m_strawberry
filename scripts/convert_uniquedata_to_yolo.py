"""HuggingFace `UniqueData/ripe-strawberries-detection` (CVAT XML) → YOLO 변환.

통합 스키마(2-class) 기준:
    데이터셋 라벨이 "strawberry" 1종이고 모두 ripe 만 라벨링했으므로
    → 전부 `ripe_strawberry` (id=1)로 매핑.

데이터셋 포맷:
    raw/data/
      ├── annotations.xml      # CVAT v1.1 export
      ├── images.tar.gz        # 원본 이미지 (자동으로 풀어줌)
      ├── boxes.tar.gz         # bbox 시각화 (학습엔 불필요)
      └── ripe-strawberries-detection.csv

옵션:
    --image-dir   이미지 폴더 경로 직접 지정 (생략 시 raw/data/images 자동 탐색)

사용 예:
    python scripts/convert_uniquedata_to_yolo.py \
        --xml datasets/raw/data/annotations.xml \
        --dst datasets/yolo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# 통합 2-class 스키마: 0=unripe, 1=ripe.
# UniqueData는 모두 ripe만 라벨링되어 있어 항상 1로 매핑.
RIPE_CLS = 1
UNRIPE_CLS = 0


def ensure_images_extracted(data_dir: Path) -> Path:
    """data/images 폴더가 없으면 images.tar.gz를 푼다.

    Returns:
        이미지가 들어 있는 폴더 경로.
    """
    img_dir = data_dir / "images"
    if img_dir.is_dir() and any(img_dir.iterdir()):
        return img_dir

    tar_path = data_dir / "images.tar.gz"
    if not tar_path.is_file():
        raise FileNotFoundError(f"images.tar.gz 가 없습니다: {tar_path}")

    print(f"[INFO] extracting {tar_path} ...")
    img_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        # 샌드박스에서 chown이 막힐 수 있으므로 owner 정보는 무시
        for m in tar.getmembers():
            m.uid = m.gid = 0
            m.uname = m.gname = ""
            tar.extract(m, path=img_dir, set_attrs=False)
    return img_dir


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """src → dst symlink (또는 copy). 기존 파일/링크는 덮어쓴다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> tuple[float, float, float, float]:
    """CVAT (xtl, ytl, xbr, ybr) 픽셀 좌표 → YOLO (cx, cy, w, h) 정규화 좌표."""
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    # 라벨링 오차로 미세하게 범위 밖 나오는 경우 클램프
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=Path("datasets/raw/data/annotations.xml"))
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="이미지 폴더 (생략 시 xml 옆 images/ 또는 images.tar.gz 자동 처리)")
    ap.add_argument("--dst", type=Path, default=Path("datasets/yolo"))
    ap.add_argument("--prefix", type=str, default="uniq",
                    help="동일 stem 충돌 방지용 파일명 prefix (예: uniq_0.png)")
    ap.add_argument("--copy-images", action="store_true", help="symlink 대신 이미지 복사")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    xml_path: Path = args.xml.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()
    if not xml_path.is_file():
        print(f"[ERROR] xml not found: {xml_path}", file=sys.stderr)
        return 2

    # 이미지 폴더 결정 (사용자 지정 우선, 없으면 자동)
    if args.image_dir is not None:
        img_dir = args.image_dir.expanduser().resolve()
    else:
        img_dir = ensure_images_extracted(xml_path.parent)

    # 출력 경로: split_dataset.py 와 호환되도록 images/all, labels/all 에 쌓음
    out_images = dst / "images" / "all"
    out_labels = dst / "labels" / "all"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] xml       = {xml_path}")
    print(f"[INFO] image_dir = {img_dir}")
    print(f"[INFO] dst       = {dst}")
    print(f"[INFO] prefix    = {args.prefix}")
    print(f"[INFO] mapping   = strawberry → ripe_strawberry (id={RIPE_CLS})")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats: Counter = Counter()
    occluded_meta: dict[str, list[dict]] = {}  # 이미지 stem → 박스별 occluded 여부 등

    for img_el in root.findall("image"):
        name = img_el.attrib.get("name", "")
        W = int(img_el.attrib.get("width", 0))
        H = int(img_el.attrib.get("height", 0))
        if not name or W <= 0 or H <= 0:
            stats["bad_image_meta"] += 1
            continue

        src_img = img_dir / name
        if not src_img.is_file():
            stats["image_missing"] += 1
            continue

        # prefix를 붙여 다른 데이터셋과 stem 충돌 방지 (예: uniq_0)
        base_stem = Path(name).stem
        stem = f"{args.prefix}_{base_stem}"
        lines: list[str] = []
        box_meta: list[dict] = []

        for b in img_el.findall("box"):
            try:
                x1 = float(b.attrib["xtl"])
                y1 = float(b.attrib["ytl"])
                x2 = float(b.attrib["xbr"])
                y2 = float(b.attrib["ybr"])
            except KeyError:
                stats["bad_box"] += 1
                continue
            occluded = b.attrib.get("occluded", "0") == "1"

            # 통합 스키마에서는 UniqueData는 전부 ripe
            cls_id = RIPE_CLS

            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
            if w <= 0 or h <= 0:
                stats["zero_box"] += 1
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[f"cls_{cls_id}"] += 1
            box_meta.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "occluded": occluded,
                "cls_id": cls_id,
            })

        if not lines:
            stats["no_valid_box"] += 1
            continue

        with open(out_labels / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # 이미지도 prefix를 붙여 복사/symlink (동일 stem 충돌 방지)
        target_img = out_images / f"{stem}{src_img.suffix}"
        link_or_copy(src_img, target_img, copy=args.copy_images)
        occluded_meta[stem] = box_meta
        stats["ok"] += 1

    # 다른 컨버터의 메타와 병합되도록 update 방식으로 저장
    meta_path = dst / "occluded_meta.json"
    existing: dict = {}
    if meta_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(occluded_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print("\n=== 변환 결과 ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\n출력:\n  images: {out_images}\n  labels: {out_labels}\n  meta:   {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
