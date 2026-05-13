# 딸기 검출 모델 공유 패키지 (YOLO26m, v3) — DEPRECATED

> ⚠️ **이 버전은 더 이상 권장되지 않습니다.**  
> 최신 권장 버전: [`../strawberry_yolo26m_unified/`](../strawberry_yolo26m_unified/) (unified v4, 2026-05-13)
>
> v3는 약 50장의 작은 데이터로 학습되어 모형(플라스틱) 사진 위주 환경에 과적합되어 있습니다.  
> 실제 환경에서 사용할 경우 unified v4를 사용하세요.

---

`runs/detect/runs/strawberry/yolo26m_ft_v3-2/` 의 학습 산출물 복사본입니다.
이전 `ft-5` 버전을 대체합니다.

## 학습 설정 (vs 이전 `ft-5`)

| 항목 | ft-5 (이전) | **v3 (현재)** |
|------|-------------|---------------|
| 베이스 모델 | `yolo26m.pt` | `yolo26m.pt` |
| epochs | 100 | 100 |
| **batch** | 16 | **20** |
| **imgsz** | 640 | **704** |
| optimizer | auto (AdamW) | auto (AdamW) |
| augmentation | 동일 | 동일 |
| 학습 시간 | - | 약 16분 (RTX 5080 Laptop) |

## 성능 (val 셋, last epoch 기준)

| 지표 | ft-5 | **v3** | 변화 |
|------|------|--------|------|
| Precision | 0.872 | **0.931** | +5.9%p |
| Recall    | 0.930 | **0.943** | +1.3%p |
| **mAP50**     | 0.946 | **0.964** | +1.8%p |
| **mAP50-95**  | 0.707 | **0.711** | +0.4%p |

클래스별 (v3, val):

| 클래스 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `unripe_strawberry` | 0.971 | 0.971 | **0.990** | 0.793 |
| `ripe_strawberry`   | 0.895 | 0.866 | 0.928 | 0.632 |

## 포함 파일

| 경로 | 설명 |
|------|------|
| `weights/best.pt` | 검증 기준 최적 가중치 (추론에 권장) |
| `weights/last.pt` | 마지막 에폭 가중치 |
| `args.yaml` | 학습 시 사용한 인자 기록 |
| `results.csv` | 에폭별 지표 |
| `results.png` | 학습 곡선 시각화 |
| `BoxF1_curve.png`, `BoxPR_curve.png` | PR / F1 곡선 |
| `confusion_matrix*.png` | 혼동행렬 |
| `strawberry.yaml` | 클래스 정의 (`0` 미숙, `1` 숙성). 재학습 시 `path:` 를 본인 데이터 루트로 수정 |

## 환경

- `ultralytics>=8.4` 권장 (YOLO26)
- GPU 사용 시 해당 머신에 맞는 PyTorch/CUDA 설치

## 추론 예시

```bash
# 이미지/폴더/영상
yolo predict model=weights/best.pt source=이미지또는영상경로 imgsz=704 conf=0.35

# 웹캠 (인덱스 0)
yolo predict model=weights/best.pt source=0 imgsz=704 conf=0.35 show=True
```

학습 시 `imgsz=704` 로 했으므로 **추론도 동일하게 `imgsz=704`** 를 권장합니다.
클래스 이름(`unripe_strawberry`, `ripe_strawberry`)은 체크포인트에 포함돼 있어
별도 yaml 없이도 라벨이 표시됩니다.

## RealSense 실시간 (참고)

본 패키지에는 미포함이지만, 원본 repo의 `scripts/realsense_live.py` 로
RGB-D 카메라에서 실시간 검출 + 깊이 표시가 가능합니다.

```bash
python scripts/realsense_live.py \
  --weights weights/best.pt \
  --imgsz 704 --show-depth
```

모형(플라스틱) 딸기 환경에서 발생하는 위양성(false ripe)을 줄이기 위해
`--ripe-conf`, `--unripe-conf`, `--min-red-for-ripe` 옵션이 있습니다.
