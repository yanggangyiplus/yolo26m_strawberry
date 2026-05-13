"""학습 진행 결과(results.csv)를 epoch 별로 깔끔하게 출력.

학습 도중 수시로 확인하거나, 학습이 일정 epoch 이상 돌았을 때 빠르게 메트릭 보려고 사용.

사용 예:
    python scripts/show_results.py                          # 가장 최근 run 자동 탐색
    python scripts/show_results.py --run runs/strawberry/yolo26m_ft
    python scripts/show_results.py --tail 20                # 마지막 20 epoch
    python scripts/show_results.py --watch                  # 5초마다 새로 출력
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def find_latest_run(root: Path) -> Path | None:
    """results.csv 가 있는 run 폴더 중 가장 최근 수정된 것을 반환."""
    candidates = list(root.rglob("results.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent


def render(csv_path: Path, tail: int) -> None:
    """results.csv 의 epoch / loss / metrics 컬럼만 골라서 표 형태로 출력."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"[empty] {csv_path}")
        return

    # Ultralytics results.csv 컬럼 이름은 버전마다 약간 다름. 안전하게 매칭.
    def col(row: dict, candidates: list[str], default: str = "-") -> str:
        for c in candidates:
            for k, v in row.items():
                if k.strip() == c:
                    return v.strip() if isinstance(v, str) else str(v)
        return default

    rows_to_show = rows[-tail:] if tail > 0 else rows
    header = (
        f"{'ep':>4} | {'box':>6} {'cls':>6} {'dfl':>6} | "
        f"{'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9} | "
        f"{'val_box':>7} {'val_cls':>7} {'val_dfl':>7} | {'lr':>8}"
    )
    print(f"\n{csv_path}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows_to_show:
        ep = col(r, ["epoch"])
        bl = col(r, ["train/box_loss"])
        cl = col(r, ["train/cls_loss"])
        dl = col(r, ["train/dfl_loss"])
        p = col(r, ["metrics/precision(B)", "metrics/precision"])
        rec = col(r, ["metrics/recall(B)", "metrics/recall"])
        m50 = col(r, ["metrics/mAP50(B)", "metrics/mAP50"])
        m9 = col(r, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        vb = col(r, ["val/box_loss"])
        vc = col(r, ["val/cls_loss"])
        vd = col(r, ["val/dfl_loss"])
        lr = col(r, ["lr/pg0", "lr/pg1", "lr/pg2"])

        def fmt(x: str, w: int, prec: int = 4) -> str:
            try:
                return f"{float(x):>{w}.{prec}f}"
            except Exception:
                return f"{x:>{w}}"

        print(
            f"{fmt(ep,4,0)} | {fmt(bl,6)} {fmt(cl,6)} {fmt(dl,6)} | "
            f"{fmt(p,6)} {fmt(rec,6)} {fmt(m50,7)} {fmt(m9,9)} | "
            f"{fmt(vb,7)} {fmt(vc,7)} {fmt(vd,7)} | {fmt(lr,8,6)}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="run 폴더 경로 (생략 시 runs/ 아래에서 자동 탐색)")
    ap.add_argument("--root", type=str, default="runs", help="자동 탐색 루트")
    ap.add_argument("--tail", type=int, default=0, help="마지막 N epoch 만 출력 (0=전체)")
    ap.add_argument("--watch", action="store_true", help="5초마다 갱신")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        run_dir = Path(args.run) if args.run else find_latest_run(Path(args.root))
        if run_dir is None:
            print(f"[wait] results.csv 없음, root={args.root}")
        else:
            csv_path = run_dir / "results.csv"
            if csv_path.is_file():
                render(csv_path, args.tail)
            else:
                print(f"[wait] {csv_path} 아직 없음")
        if not args.watch:
            break
        print("\n... (Ctrl+C 로 종료)")
        time.sleep(5)


if __name__ == "__main__":
    main()
