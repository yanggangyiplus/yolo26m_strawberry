"""AIHub 596 (지능형 수직농장 통합 데이터 - 딸기) → YOLO 포맷 변환 스크립트.

AIHub 596 데이터셋의 라벨 포맷:
- 이미지 1장당 JSON 1개 (MS COCO 준용)
- 클래스(어노테이션 객체): leaf, stem, fruit, flower
- 카테고리 정보는 JSON 내부 categories[]에 있음
- 바운딩박스 위치는 annotations[].bbox = [x, y, w, h] (픽셀, 좌상단 기준)

YOLO 포맷:
- 이미지 1장당 .txt 1개 (같은 stem 이름)
- 한 줄에 한 객체: <class_id> <cx> <cy> <w> <h>  (모두 0~1 정규화)

사용 예:
    python scripts/convert_aihub_to_yolo.py \
        --src datasets/raw \
        --dst datasets/yolo \
        --classes leaf,stem,fruit,flower

옵션:
    --copy-images       라벨 변환과 동시에 이미지를 datasets/yolo/images로 복사 (기본 symlink)
    --image-ext         탐색할 이미지 확장자 (기본: .jpg,.jpeg,.png)
    --recursive         서브폴더까지 재귀 탐색 (기본 True)
    --dry-run           실제 파일 쓰지 않고 통계만 출력
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# 사용할 클래스 순서를 고정해 두면 향후 다른 데이터셋 합쳐도 ID 호환 유지
DEFAULT_CLASSES = ["leaf", "stem", "fruit", "flower"]


def find_files(root: Path, exts: Iterable[str], recursive: bool = True) -> list[Path]:
    """주어진 확장자(소문자 비교)에 해당하는 파일을 root 이하에서 모두 찾는다."""
    exts = {e.lower().lstrip(".") for e in exts}
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.glob("*")
    return [p for p in candidates if p.is_file() and p.suffix.lower().lstrip(".") in exts]


def build_image_index(image_paths: list[Path]) -> dict[str, Path]:
    """이미지 파일을 stem(확장자 제외 파일명) 기준으로 인덱싱.

    AIHub 596은 JSON과 이미지가 같은 파일명(.json / .jpg)을 사용하므로
    stem으로 매칭한다.
    """
    idx: dict[str, Path] = {}
    for p in image_paths:
        # 동일 stem이 여러 폴더에 있을 수 있으나 일반적으로 유일하므로 마지막 것 사용
        idx[p.stem] = p
    return idx


def load_coco_json(path: Path) -> dict:
    """JSON을 안전하게 로드. 인코딩 문제 시 cp949도 시도."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949") as f:
            return json.load(f)


def extract_category_map(data: dict, class_name_to_id: dict[str, int]) -> dict[int, int]:
    """COCO JSON의 categories[]를 읽어 'COCO category_id → YOLO class_id' 매핑을 만든다.

    카테고리 이름은 영문(name) 우선, 없으면 한글(name_ko/category)에서 추론.
    """
    cat_map: dict[int, int] = {}
    for cat in data.get("categories", []):
        cat_id = cat.get("id")
        # 일반적으로 'name'에 영문, 'supercategory'에 상위 카테고리가 들어있음
        name = (cat.get("name") or cat.get("category") or "").strip().lower()
        if cat_id is None or not name:
            continue
        if name in class_name_to_id:
            cat_map[int(cat_id)] = class_name_to_id[name]
    return cat_map


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO bbox [x, y, w, h] (좌상단 기준, 픽셀) → YOLO [cx, cy, w, h] (정규화)."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # 범위 클램프 (라벨링 오차 방지)
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def convert_one(
    json_path: Path,
    image_index: dict[str, Path],
    out_labels_dir: Path,
    out_images_dir: Path,
    class_name_to_id: dict[str, int],
    copy_images: bool,
    dry_run: bool,
    stats: Counter,
) -> bool:
    """JSON 1개를 처리하여 YOLO .txt 1개와 이미지 1장을 출력 폴더로 옮긴다.

    매칭 실패/유효한 박스 없음 등은 False 반환 (skip).
    """
    try:
        data = load_coco_json(json_path)
    except Exception as e:
        stats["json_error"] += 1
        print(f"[WARN] JSON parse failed: {json_path} ({e})", file=sys.stderr)
        return False

    # 이미지 정보 추출: AIHub는 images가 list 또는 dict일 수 있음
    images_field = data.get("images")
    if isinstance(images_field, list) and images_field:
        img_info = images_field[0]
    elif isinstance(images_field, dict):
        img_info = images_field
    else:
        stats["no_image_field"] += 1
        return False

    img_w = int(img_info.get("width") or 0)
    img_h = int(img_info.get("height") or 0)
    fname = img_info.get("fname") or img_info.get("file_name") or json_path.stem
    fext = (img_info.get("fext") or "").lstrip(".")

    # stem 기준으로 실제 이미지 파일 매칭
    stem = Path(fname).stem if "." in fname else fname
    src_img = image_index.get(stem) or image_index.get(json_path.stem)
    if src_img is None:
        stats["image_not_found"] += 1
        return False

    # JSON에 width/height가 비어 있으면 실제 이미지에서 읽기 (PIL 사용)
    if img_w <= 0 or img_h <= 0:
        try:
            from PIL import Image  # 지연 import로 PIL 없는 환경 대응

            with Image.open(src_img) as im:
                img_w, img_h = im.size
        except Exception as e:
            stats["size_unknown"] += 1
            print(f"[WARN] cannot read image size: {src_img} ({e})", file=sys.stderr)
            return False

    # COCO category_id → YOLO class_id 매핑 (파일별로 만들어 카테고리 ID 충돌 방지)
    cat_map = extract_category_map(data, class_name_to_id)

    # annotations 추출 후 YOLO 라인 생성
    lines: list[str] = []
    for ann in data.get("annotations", []) or []:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        coco_cat = ann.get("category_id")
        # category_id가 매핑 안 되면, 일부 AIHub 파일은 category가 별도 키로 들어가는 경우도 있음
        yolo_cls = cat_map.get(int(coco_cat)) if coco_cat is not None else None
        if yolo_cls is None:
            # 직접 클래스명이 들어있는 변형 대응
            name = (ann.get("category") or ann.get("name") or "").strip().lower()
            yolo_cls = class_name_to_id.get(name)
        if yolo_cls is None:
            stats["unknown_class"] += 1
            continue

        cx, cy, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
        if w <= 0 or h <= 0:
            stats["zero_box"] += 1
            continue
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stats[f"class_{yolo_cls}"] += 1

    if not lines:
        stats["no_valid_box"] += 1
        return False

    # 출력 파일명은 이미지 stem과 동일하게 유지 (YOLO 규약)
    out_label = out_labels_dir / f"{stem}.txt"
    out_image = out_images_dir / src_img.name

    if not dry_run:
        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_image.parent.mkdir(parents=True, exist_ok=True)
        with open(out_label, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # 이미지는 기본 symlink (디스크 절약). copy 옵션 시 복사
        if out_image.exists() or out_image.is_symlink():
            out_image.unlink()
        if copy_images:
            shutil.copy2(src_img, out_image)
        else:
            os.symlink(src_img.resolve(), out_image)

    stats["ok"] += 1
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="AIHub 596 → YOLO 변환기")
    ap.add_argument("--src", type=Path, required=True, help="원본 데이터 루트 (이미지 + JSON)")
    ap.add_argument("--dst", type=Path, required=True, help="YOLO 출력 루트 (images/, labels/ 생성)")
    ap.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--copy-images", action="store_true", help="symlink 대신 이미지 복사")
    ap.add_argument("--image-ext", type=str, default=".jpg,.jpeg,.png")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()
    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    class_name_to_id = {name: i for i, name in enumerate(classes)}

    # 모든 변환물은 일단 *_all 폴더에 모아두고, split 단계에서 train/val/test로 나눈다
    out_images_dir = dst / "images" / "all"
    out_labels_dir = dst / "labels" / "all"

    print(f"[INFO] src = {src}")
    print(f"[INFO] dst = {dst}")
    print(f"[INFO] classes = {class_name_to_id}")

    exts = args.image_ext.split(",")
    image_paths = find_files(src, exts, recursive=not args.no_recursive)
    json_paths = find_files(src, [".json"], recursive=not args.no_recursive)
    print(f"[INFO] found {len(image_paths)} images / {len(json_paths)} jsons")
    if not image_paths or not json_paths:
        print("[ERROR] No images or labels found under --src", file=sys.stderr)
        return 2

    image_index = build_image_index(image_paths)
    stats: Counter = Counter()
    for jp in json_paths:
        convert_one(
            jp,
            image_index,
            out_labels_dir,
            out_images_dir,
            class_name_to_id,
            copy_images=args.copy_images,
            dry_run=args.dry_run,
            stats=stats,
        )

    print("\n=== 변환 결과 ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\n출력 경로:\n  images: {out_images_dir}\n  labels: {out_labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
