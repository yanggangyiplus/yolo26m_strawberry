# 딸기 검출 모델 공유 패키지 (YOLO26m, **unified v4**)

`runs/detect/runs/strawberry/yolo26m_unified_640b16/` 의 학습 산출물 복사본입니다.  
이전 `v3` (yolo26m_ft_v3-2) 를 대체하는 **현재 권장 버전**입니다.

> 작성일: 2026-05-13  
> 학습 환경: NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM), Ultralytics 8.4.48, PyTorch 2.12+cu128

---

## 학습 데이터 (이전 v3 대비 대폭 확장)

| 소스 | v3 | **unified v4** | 라벨링 방식 |
|---|---|---|---|
| Qin2006 (CVAT) | - | 325 | 기존 라벨 |
| UniqueData (CVAT) | 40 | 40 | 기존 라벨 |
| strawberryDataset/Pickable | - | 264 | HSV 자동 + 검토 |
| strawberryDataset/UnPickable | - | 259 | HSV 자동 + 검토 |
| 사용자 모형 (raw_str / red_str) | 일부 | 39 | HSV 자동 |
| 실제 장면 (realscene) | - | 7 | LabelImg 수동 |
| **합계** | ~50장 | **934장** | |

분할: train 747 / val 93 / test 94

---

## 학습 설정 (vs 이전 v3)

| 항목 | v3 (이전) | **unified v4 (현재)** |
|---|---|---|
| 베이스 모델 | yolo26m.pt | yolo26m.pt |
| epochs | 100 | 100 |
| imgsz | 704 | **640** |
| batch | 20 | **16** |
| optimizer | auto (AdamW) | auto (AdamW) |
| GPU 사용 | ~12GB | 8.94GB (55%) |
| 학습 시간 | ~16분 | **27분** (데이터 19배 증가) |

---

## 성능 (val 셋, best epoch 기준)

| 지표 | v3 | **unified v4** | 비고 |
|---|---|---|---|
| Precision | 0.931 | 0.884 | val 다양성 ↑ |
| Recall    | 0.943 | 0.830 | val 다양성 ↑ |
| **mAP50**     | 0.964 | **0.892** | val에 어려운 실제장면 포함 |
| **mAP50-95**  | 0.711 | **0.727** | ★ 개선 |

> **주의**: v3는 val이 단순한 모형 사진 위주였고, unified는 CVAT/실제 장면을 포함한 훨씬 어려운 val입니다. 절대 비교는 의미가 적으며, **실제 환경 일반화 성능은 unified가 큰 폭으로 향상**되었습니다.

### 클래스별 (unified v4, val 93장 / 145 객체)

| 클래스 | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| `unripe_strawberry` | 57 | 59 | 0.941 | 0.812 | **0.906** | **0.792** |
| `ripe_strawberry`   | 43 | 86 | 0.828 | 0.849 | 0.877 | 0.662 |
| **all** | 93 | 145 | 0.884 | 0.830 | **0.892** | **0.727** |

추론 속도: **0.1ms preprocess + 4.4ms inference + 0.0ms postprocess** = ~4.5ms/이미지 (RTX 5080 기준)

---

## 포함 파일

| 경로 | 설명 |
|---|---|
| `weights/best.pt` | ★ 검증 기준 최적 가중치 (추론에 권장) |
| `weights/last.pt` | 마지막 에폭 가중치 |
| `args.yaml` | 학습 시 사용한 전체 인자 |
| `strawberry_unified.yaml` | 데이터 클래스 정의 (재학습 시 `path:` 만 본인 환경으로 수정) |
| `results.csv` | 에폭별 metric (loss, P, R, mAP) |
| `results.png` | 학습 곡선 시각화 |
| `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png` | conf 임계별 P/R/F1/PR 곡선 |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | 혼동행렬 |
| `labels.jpg` | train 라벨 분포 시각화 |
| `val_batch{0,1,2}_labels.jpg` | val GT 시각화 |
| `val_batch{0,1,2}_pred.jpg` | val 모델 예측 시각화 |
| `realsense_live.py` | RealSense 실시간 추론 스크립트 (시간적 평활화 포함) |

---

## 환경 (팀원 셋업)

```bash
pip install "ultralytics>=8.4" opencv-python pillow numpy
# RealSense 사용 시
pip install pyrealsense2
```

GPU 사용 시 환경에 맞는 PyTorch/CUDA 별도 설치.

---

## 추론 예시

### 1) 정적 이미지 / 폴더 / 영상

```bash
yolo predict model=weights/best.pt source=경로 imgsz=640 conf=0.30
```

> 학습 시 `imgsz=640`이므로 추론도 동일 권장. 클래스 이름은 체크포인트에 포함됨.

### 2) 웹캠

```bash
yolo predict model=weights/best.pt source=0 imgsz=640 conf=0.30 show=True
```

### 3) Python에서 직접

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model.predict("img.jpg", imgsz=640, conf=0.30, device=0)
for r in results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        print(f"{model.names[int(cls)]}  conf={float(conf):.2f}  box={box.tolist()}")
```

### 4) Intel RealSense 실시간 (RGB-D)

```bash
python realsense_live.py \
    --weights weights/best.pt \
    --imgsz 640 --device 0 \
    --smooth 7 --min-red-for-ripe 0.10
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--smooth N` | 7 | N프레임 다수결로 클래스 안정화 (튀는 라벨 억제) |
| `--ripe-conf` | 0.30 | ripe 인정 최소 confidence |
| `--unripe-conf` | 0.20 | unripe 인정 최소 confidence |
| `--min-red-for-ripe` | 0.10 | ripe 박스 색 체크 임계값. 0.0=비활성화 |
| `--show-depth` | off | 우측에 depth colormap 표시 |
| `--save-dir` | runs/realsense | 스냅샷 저장 폴더 |

키: `q` 종료, `p` 스냅샷 저장.

> 모형(플라스틱) 딸기 환경의 위양성을 줄이려면 `--min-red-for-ripe 0.20`처럼 임계값을 올리세요. 실제 딸기에서 ripe가 unripe로 자주 튕기면 `--min-red-for-ripe 0.0`으로 색 체크를 끄세요.

---

## 클래스 정의

| id | 이름 | 색상 (RealSense 시각화) |
|---|---|---|
| 0 | `unripe_strawberry` | 주황 |
| 1 | `ripe_strawberry` | 초록 |

---

## 알려진 한계

- val의 ripe 클래스 (0.877)가 unripe (0.906)보다 약간 낮음 — 실제 ripe 박스가 더 많아(86 vs 59) 다양한 조명·가림 케이스 포함
- realscene 학습 데이터가 7장으로 적음 → 야외 다양 환경 추가 데이터 권장
- 자동 HSV 라벨로 만든 strawberryDataset 부분은 박스 정확도 다소 낮을 수 있음

---

## 원본 리포지토리

[github.com/yanggangyiplus/yolo26m_strawberry](https://github.com/yanggangyiplus/yolo26m_strawberry)

학습 재현 / 추가 데이터 학습 / 추가 스크립트는 위 리포지토리 참고.
