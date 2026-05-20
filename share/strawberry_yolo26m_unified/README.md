# 딸기 검출 + 줄기 픽(그립) 공유 패키지 (YOLO26m)

## ★ RealSense 실시간 — 한 스크립트에 같이 동작

**딸기 검출과 줄기(그립) 검출은 따로 실행하는 게 아닙니다.**  
아래 **한 명령**으로 `realsense_stem_pipeline.py` 가 프레임마다 순서대로 처리합니다.

```
컬러 프레임 → [1] 딸기 detect → [2] 각 딸기 상단 ROI → [3] 줄기 seg → [4] 그립점 표시
```

| 구분 | 설명 |
|---|---|
| **실행 파일** | `realsense_stem_pipeline.py` **하나** (리포지토리 `scripts/realsense_stem_pipeline.py` 와 동일 로직) |
| **가중치** | `weights/best.pt`(딸기) + `weights/stem_best.pt`(줄기) — 학습은 모델별로 했지만 **추론은 한 파이프라인** |
| **detect만** | `realsense_live.py` — 줄기 없이 딸기 박스만 볼 때 (디버그·비교용) |

```bash
cd share/strawberry_yolo26m_unified
python realsense_stem_pipeline.py \
  --weights-det weights/best.pt \
  --weights-stem weights/stem_best.pt \
  --imgsz-det 832 --imgsz-stem 128
```

학습 메트릭·차트만 폴더가 나뉩니다: detect → 루트·`832b8/`, 줄기 → `stem/`.  
가중치(`weights/*.pt`)는 용량상 git에 없으니 아래 **가중치 복사**를 참고하세요.

| 파일 | 용도 |
|---|---|
| `weights/best.pt` | 딸기 detect (832 권장 시 832 학습본 복사) |
| `weights/stem_best.pt` | 줄기 ROI seg (`yolo26m_stem_roi_128b16` best) |

> 작성일: 2026-05-19 (줄기 픽 파이프라인 추가)  
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

## 832×832 이어학습 (`yolo26m_unified_832b8`, batch 8)

640 `best.pt`에서 이어 학습. EarlyStopping으로 **best = epoch 54** (아래는 해당 에폭 val 집계).

| 지표 | unified 640 | **832b8 (ep.54)** |
|---|---|---|
| Precision | 0.884 | **0.906** |
| Recall    | 0.830 | 0.731 |
| mAP50     | 0.892 | 0.855 |
| mAP50-95  | 0.727 | 0.659 |

> val 인스턴스 수가 640 학습 때(145)보다 많아질 수 있어(재라벨·해상도) 숫자만으로 640 vs 832 우열을 단정하지 마세요. RealSense 등 **고해상도 추론**에는 832 가중치를 우선 시도하는 것을 권장합니다.

에폭별 전체 곡선: 루트의 `results_832b8.csv`·`args_832b8.yaml` 또는 **`832b8/`** 폴더(run과 동일 파일명)를 참고하세요.

---

## 포함 파일

| 경로 | 설명 |
|---|---|
| `weights/best.pt` | ★ 딸기 detect 최적 가중치 (로컬에서 복사) |
| `weights/stem_best.pt` | ★ 줄기 ROI seg 최적 가중치 (로컬에서 복사) |
| `args.yaml` | **640** detect 학습 인자 |
| `strawberry_unified.yaml` | detect 데이터 YAML |
| `results.csv` / `results_832b8.csv` | detect 에폭별 metric |
| `832b8/` | **832 detect** run 복제 (차트·csv·args) |
| `stem/` | **줄기 ROI seg** run 복제 (`yolo26m_stem_roi_128b16`) |
| `strawberry_stem_roi_seg.yaml` | 줄기 ROI 데이터 YAML (학습 재현용) |
| `realsense_live.py` | 딸기만 검출 — RealSense 실시간 |
| `realsense_stem_pipeline.py` | **딸기 + 줄기 그립** — RealSense 실시간 (권장) |
| `predict.py` | 정적 이미지/폴더 ripe·unripe 판별 + 픽 JSON |
| `review_stem_pipeline.py` | 이미지 폴더로 파이프라인 검수 JPG 저장 |
| `stem_roi_utils.py` | ROI·CLAHE·HSV·그립 오프셋 유틸 |

**추론 스크립트만 이 폴더에 두고 zip 공유하면 됩니다.**  
`bash scripts/sync_share_package.sh` 로 RealSense 3종을 repo 최신과 맞출 수 있습니다.

---

## 환경 (팀원 셋업)

```bash
pip install "ultralytics>=8.4" opencv-python pillow numpy
# RealSense 사용 시
pip install pyrealsense2
```

GUI용 `realsense_live.py`는 `opencv-python-headless`와 동시 설치 시 충돌할 수 있음(리포지토리 `requirements.txt` 주석 참고).

GPU 사용 시 환경에 맞는 PyTorch/CUDA 별도 설치.

---

## 추론 예시

### 1) 정적 이미지 — ripe/unripe 판별 + JSON

```bash
cd share/strawberry_yolo26m_unified
python predict.py --source /path/to/images --imgsz 832 --save-images
# → runs/predict_picks.json (picks=ripe, avoid=unripe)
```

또는 Ultralytics CLI:

```bash
yolo predict model=weights/best.pt source=경로 imgsz=832 conf=0.30
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

### 4) Intel RealSense — 딸기 + 줄기 그립 (권장, 통합)

```bash
cd share/strawberry_yolo26m_unified   # 이 README 있는 폴더

python realsense_stem_pipeline.py \
  --weights-det weights/best.pt \
  --weights-stem weights/stem_best.pt \
  --imgsz-det 832 --imgsz-stem 128 --device 0
```

### 4-1) RealSense — 딸기만 (줄기 없음, 선택)

```bash
python realsense_live.py \
    --weights weights/best.pt \
    --imgsz 832 --device 0 \
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
| `--headless` | off | GUI 없이 MP4만 저장 (`--headless-out` 선택) |
| `--brightness` 등 | None | UVC 컬러 보정(지원 시) |
| `--no-auto-exposure` / `--exposure` | - | 수동 노출 |

키: `q` 종료, `p` 스냅샷 저장.

> 모형(플라스틱) 딸기 환경의 위양성을 줄이려면 `--min-red-for-ripe 0.20`처럼 임계값을 올리세요. 실제 딸기에서 ripe가 unripe로 자주 튕기면 `--min-red-for-ripe 0.0`으로 색 체크를 끄세요.

---

## 줄기 픽(그립) — 모델·학습 정보

위 **통합 스크립트** `realsense_stem_pipeline.py` 안에서 아래 단계가 이어집니다 (별도 프로그램 아님).

```
[RealSense 컬러+깊이]
    → YOLO detect (ripe/unripe 딸기 bbox)
    → 각 bbox 상단 ROI crop (줄기·윗동)
    → CLAHE + HSV 줄기 강조 전처리
    → YOLO seg (stem 마스크, 128×128)
    → 마스크 중심 + 깊이 기반 1cm 위 오프셋 → 그립점 (노란 원)
```

### 학습 데이터 (줄기 ROI)

| 항목 | 내용 |
|---|---|
| 원본 | `yolo_unified_farm` 이미지 + detect bbox |
| 생성 | `build_stem_roi_dataset.py` — 과실마다 상단 ROI 패치 + 줄기 폴리곤 |
| 패치 수 | train 789 / val 98 / test 100 |
| 클래스 | `stem` (단일) |
| 전처리 | CLAHE(LAB) + HSV 줄기 채널 부스트 (학습·추론 동일) |

### 학습 설정 (`yolo26m_stem_roi_128b16`)

| 항목 | 값 |
|---|---|
| 베이스 | `yolo26m-seg.pt` |
| imgsz | **128** (작은 ROI 패치) |
| batch | 16 |
| epochs | 100 |

### 성능 (val, mask 기준)

| 지표 | best (ep.75) | last (ep.100) |
|---|---|---|
| mAP50 (M) | **0.648** | 0.583 |
| mAP50-95 (M) | **0.314** | 0.265 |
| Precision (M) | 0.701 | 0.654 |
| Recall (M) | 0.655 | 0.586 |

> 추론·배포에는 **`weights/stem_best.pt`** (학습 run의 `best.pt` 복사) 사용을 권장합니다. 에폭 곡선은 `stem/results.csv` 참고.

### 가중치 복사 (팀원)

```bash
# 리포지토리 루트에서
PKG=share/strawberry_yolo26m_unified

# 딸기 detect (832 권장)
cp runs/detect/runs/strawberry/yolo26m_unified_832b8/weights/best.pt \
   "$PKG/weights/best.pt"

# 줄기 ROI seg
cp runs/segment/runs/strawberry/yolo26m_stem_roi_128b16/weights/best.pt \
   "$PKG/weights/stem_best.pt"
```

### RealSense — 줄기 그립 실시간

```bash
cd share/strawberry_yolo26m_unified

python realsense_stem_pipeline.py \
  --weights-det weights/best.pt \
  --weights-stem weights/stem_best.pt \
  --imgsz-det 832 --imgsz-stem 128 \
  --device 0 \
  --grip-margin-cm 1.0
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--imgsz-det` | 832 | detect 입력 크기 (가중치 학습 해상도와 맞출 것) |
| `--imgsz-stem` | 128 | stem ROI seg 입력 크기 |
| `--conf-stem` | 0.20 | 줄기 seg 최소 confidence |
| `--above-ratio` | 0.55 | bbox 위쪽 ROI 확장 비율 |
| `--grip-margin-cm` | 1.0 | 마스크 중심에서 줄기 방향(위)으로 올리는 그립 오프셋 [cm] |
| `--stem-unripe` | off | unripe에도 줄기·그립 (기본: **ripe만**) |
| `--no-preprocess` | off | CLAHE/HSV 전처리 끄기 |
| `--no-roi` | off | ROI 없이 전체 프레임 stem (구 방식 폴백) |
| `--no-depth` | off | 깊이 없이 2D만 (오프셋은 0.5m 가정) |

시각화:

- 초록/주황 박스: ripe / unripe 딸기
- 주황 테두리 사각형: 줄기 ROI
- **노란 원**: 최종 그립점 — **`ripe_strawberry` 만** (unripe는 박스만)
- 회색 작은 점: seg 마스크 중심 (오프셋 전)

키: `q` 종료, `p` 스냅샷 (`runs/realsense_stem/`).

리포지토리 루트에서 동일 파이프라인:

```bash
python scripts/realsense_stem_pipeline.py \
  --weights-det runs/detect/runs/strawberry/yolo26m_unified_832b8/weights/best.pt \
  --weights-stem runs/segment/runs/strawberry/yolo26m_stem_roi_128b16/weights/best.pt \
  --imgsz-det 832 --imgsz-stem 128
```

---

## 클래스 정의

| id | 이름 | 색상 (RealSense 시각화) |
|---|---|---|
| 0 | `unripe_strawberry` | 주황 |
| 1 | `ripe_strawberry` | 초록 |

**줄기 seg** (`stem/`): id `0` = `stem` (그립 마스크)

---

## 알려진 한계

- val의 ripe 클래스 (0.877)가 unripe (0.906)보다 약간 낮음 — 실제 ripe 박스가 더 많아(86 vs 59) 다양한 조명·가림 케이스 포함
- realscene 학습 데이터가 7장으로 적음 → 야외 다양 환경 추가 데이터 권장
- 자동 HSV 라벨로 만든 strawberryDataset 부분은 박스 정확도 다소 낮을 수 있음
- 줄기 seg: ROI가 작고(128px) 조명·가림에 민감 — detect bbox가 틀리면 그립점도 어긋남
- 그립 1cm 오프셋은 pinhole+median depth 근사; stem이 화면 가장자리·근거리일 때 오차 증가

---

## 원본 리포지토리

[github.com/yanggangyiplus/yolo26m_strawberry](https://github.com/yanggangyiplus/yolo26m_strawberry)

학습 재현 / 추가 데이터 학습 / 추가 스크립트는 위 리포지토리 참고.
