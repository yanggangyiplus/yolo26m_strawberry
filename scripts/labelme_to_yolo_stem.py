#!/usr/bin/env python3
"""labelme JSON → YOLO segmentation (class stem=0).

labelme에서 그린 폴리곤(label==stem)을
datasets/yolo_unified_seg/labels/all/<stem>.txt 로 저장.

사용:
  python scripts/labelme_to_yolo_stem.py \\
      --json-dir datasets/yolo_unified_seg/_labelme/realscene
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEM_CLS = 0


def json_to_yolo_seg(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = float(data["imageWidth"])
    h = float(data["imageHeight"])
    lines: list[str] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        label = (shape.get("label") or "").strip().lower()
        if label != "stem":
            continue
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        norm = []
        for x, y in pts:
            norm.append(f"{float(x) / w:.6f}")
            norm.append(f"{float(y) / h:.6f}")
        lines.append(f"{STEM_CLS} " + " ".join(norm))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="labelme JSON이 있는 폴더 (_labelme/realscene 등)",
    )
    ap.add_argument(
        "--out-labels",
        type=Path,
        default=Path("datasets/yolo_unified_seg/labels/all"),
    )
    args = ap.parse_args()
    jdir = args.json_dir.expanduser().resolve()
    out = args.out_labels.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for jp in sorted(jdir.glob("*.json")):
        lines = json_to_yolo_seg(jp)
        stem = jp.stem
        (out / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
        n += 1
        print(f"  {stem}.txt  ({len(lines)} stems)")

    (out / "classes.txt").write_text("stem\n", encoding="utf-8")
    print(f"[done] {n} files -> {out}")
    print("분할 갱신: python scripts/split_dataset.py --root datasets/yolo_unified_seg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
