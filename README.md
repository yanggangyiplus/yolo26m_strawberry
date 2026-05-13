# 딸기 수확 로봇용 YOLO26m 파인튜닝

HuggingFace 공개 데이터셋 + 사용자 촬영 이미지로 **Ultralytics YOLO26m**을 2-class(`unripe_strawberry`, `ripe_strawberry`) 검출용으로 fine-tuning하고, RealSense RGB-D 카메라로 실시간 추론까지 수행하는 프로젝트입니다.

- 베이스 모델: `yolo26m.pt` (Ultralytics 8.4+, 첫 실행 시 자동 다운로드)
- 학습 환경: NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM)
- 추론 카메라: Intel RealSense D4xx (color + depth)

---

## 데이터셋 요약 (통합 후)

| 소스 | 라벨링 방식 | 장수 | prefix |
| --- | --- | --- | --- |
| Qin2006 (HuggingFace) | CVAT 기존 라벨 | 325 | `qin_*` |
| UniqueData (HuggingFace) | CVAT 기존 라벨 | 40 | `uniq_*` |
| strawberryDataset/Pickable | HSV 자동라벨 + 수동검토 | 264 | `straw_pick_*` |
| strawberryDataset/UnPickable | HSV 자동라벨 + 수동검토 | 259 | `straw_unpick_*` |
| 사용자 모형 (raw_str / red_str) | HSV 자동라벨 | 39 | `user_raw_*`, `user_red_*` |
| 실제 장면 (realscene) | LabelImg 수동라벨 | 7 | `realscene_*` |
| **합계** | | **934** | |

→ `datasets/yolo_unified/` 에 통합. `labels/all/` 에 모든 라벨 보존. `split_dataset.py`로 train(747)/val(93)/test(94) 분할.

---

## 폴더 구조

```
yolo_strawberry/
├── README.md
├── requirements.txt
├── .gitignore
├── yolo26m.pt                          # 사전학습 가중치 (gitignore)
├── configs/
│   ├── strawberry.yaml                  # 초기 데이터 (yolo/)
│   ├── strawberry_3folders.yaml         # 3폴더 데이터 (yolo_3folders/)
│   └── strawberry_unified.yaml          # ★ 통합 데이터 (yolo_unified/)
├── scripts/
│   ├── train.py                         # 학습
│   ├── predict.py                       # 정적 이미지 추론 + 픽 후보 JSON
│   ├── realsense_live.py                # ★ RealSense 실시간 추론
│   ├── autolabel_user_pics.py           # HSV 기반 자동 라벨
│   ├── convert_qin_to_yolo.py           # Qin2006 → YOLO
│   ├── convert_uniquedata_to_yolo.py    # UniqueData CVAT → YOLO
│   ├── split_dataset.py                 # train/val/test 분할 (symlink)
│   ├── launch_labelimg.sh               # ★ LabelImg 수동 라벨링 런처
│   ├── run_compare.py / show_results.py # 검증 비교 도구
│   └── archive/                         # 사용 중지 스크립트
├── datasets/                            # gitignore (라벨 일부만 포함)
│   ├── raw/                             # HuggingFace 원본 (gitignore)
│   ├── yolo/                            # 초기 변환 결과 (gitignore)
│   ├── yolo_3folders/                   # 3폴더 학습용 (gitignore)
│   └── yolo_unified/                    # ★ 통합 데이터셋
│       ├── images/all/                  # 934장 (gitignore)
│       └── labels/
│           ├── all/                     # ★ 934개 라벨 (git에 포함됨)
│           └── train|val|test/          # symlink (gitignore, split_dataset.py로 재생성)
├── runs/
│   ├── strawberry/*.log                 # ★ 학습 로그 (git에 포함됨)
│   ├── detect/runs/strawberry/          # 학습 결과 weights (gitignore)
│   ├── compare/                         # gitignore
│   └── realsense/                       # 카메라 스냅샷 (gitignore)
└── share/                               # 학습 결과 차트·csv 배포용
    └── strawberry_yolo26m_ft_v3/        # weights/ 만 gitignore
```

---

## 학습 이력 (모델 비교)

| 모델 디렉토리 | 데이터 | imgsz | batch | epochs | val mAP50 | val mAP50-95 | 비고 |
|---|---|---|---|---|---|---|---|
| `yolo26m_ft-5/` | yolo (40장 only) | 640 | 16 | 100 | - | - | 초기 시도 |
| `yolo26m_ft_v3-2/` | yolo (40장 + 일부) | 640 | 16 | 100 | 0.959 | 0.715 | 어제 v3 |
| `yolo26m_3folders_640b16/` | 3폴더 (562장) | 640 | 16 | 100 | **0.959** | **0.878** | 3폴더 베이스라인 |
| `yolo26m_3folders_1280_auto/` | 3폴더 | 1280 | auto | 일부 | 0.334 | 0.111 | OOM으로 batch 축소 |
| `yolo26m_unified_640b16/` | **통합 934장** | 640 | 16 | 100 | 0.892 | 0.727 | ★ **현재 최신** |

학습 로그는 `runs/strawberry/*.log` 에 모두 보존됨. 각 모델 가중치는 `runs/detect/runs/strawberry/<name>/weights/{best,last}.pt`.

> 통합 모델의 mAP가 3폴더보다 낮아 보이지만, val 데이터가 더 다양해진 결과(realscene·CVAT 포함)이므로 일반화 성능은 향상.

---

## 0. 환경 준비

```bash
pip install -r requirements.txt
export YOLO_CONFIG_DIR="$(pwd)/.ultralytics_cfg"   # (선택)
```

추가 의존: `pyrealsense2`, `opencv-python`, `Pillow`, `labelImg` (수동 라벨링용)

---

## 1. 데이터 다운로드 (선택)

```bash
git lfs install
git clone https://huggingface.co/datasets/UniqueData/ripe-strawberries-detection datasets/raw
# Qin2006, strawberryDataset 등도 동일하게 datasets/raw/data/images/ 아래로
```

---

## 2. 라벨 변환 / 자동라벨

```bash
# Qin2006 변환
python scripts/convert_qin_to_yolo.py --xml ... --dst datasets/yolo

# UniqueData 변환
python scripts/convert_uniquedata_to_yolo.py --xml ... --dst datasets/yolo

# 사용자/strawberryDataset 자동라벨 (HSV)
python scripts/autolabel_user_pics.py \
    --raw-str datasets/raw/data/images/raw_str \
    --red-str datasets/raw/data/images/red_str \
    --strawberry-pickable datasets/raw/data/images/strawberryDataset/Pickable \
    --strawberry-unpickable datasets/raw/data/images/strawberryDataset/UnPickable \
    --dst-img datasets/yolo_unified/images/all \
    --dst-lbl datasets/yolo_unified/labels/all
```

---

## 3. 수동 라벨 보정 (LabelImg)

```bash
# 통합 전체 (RGBA → RGB 변환 후 표시)
bash scripts/launch_labelimg.sh unified

# 부분 모드 — realscene 7장부터 권장
bash scripts/launch_labelimg.sh realscene
bash scripts/launch_labelimg.sh pickable
bash scripts/launch_labelimg.sh unpickable
bash scripts/launch_labelimg.sh qin
bash scripts/launch_labelimg.sh uniq
bash scripts/launch_labelimg.sh user
```

LabelImg 단축키: `W`(박스), `D`(다음), `A`(이전), `Ctrl+S`(저장). View → Auto Save 권장.

---

## 4. train/val/test 분할

```bash
python scripts/split_dataset.py \
    --root datasets/yolo_unified \
    --train 0.8 --val 0.1 --test 0.1 --seed 42
```

→ `images|labels/{train,val,test}/` 에 절대경로 symlink 생성.  
> 다른 컴퓨터에서 clone 받았다면 split_dataset.py를 다시 실행해야 합니다 (symlink는 git에서 제외됨).

---

## 5. 학습

```bash
python scripts/train.py \
    --weights yolo26m.pt \
    --data configs/strawberry_unified.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --project runs/strawberry \
    --name yolo26m_unified_640b16
```

GPU 메모리 가이드 (RTX 5080 16GB 기준):

| imgsz | 권장 batch | VRAM 사용 | 비고 |
|---|---|---|---|
| 640 | 16 | ~9 GB | ★ 안정·빠름 |
| 832 | 8 | ~11 GB | 작은 객체 검출 향상 |
| 1024 | 4~6 | ~13 GB | 느림 |
| 1280 | 2~3 | OOM 위험 | batch가 작아 학습 불안정 |

---

## 6. RealSense 실시간 추론

```bash
python scripts/realsense_live.py \
    --weights runs/detect/runs/strawberry/yolo26m_unified_640b16/weights/best.pt \
    --imgsz 640 \
    --device 0 \
    --smooth 7 \
    --min-red-for-ripe 0.10
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--smooth N` | 7 | N프레임 다수결로 클래스 안정화 (튀는 라벨 억제) |
| `--ripe-conf` | 0.30 | ripe 인정 최소 confidence |
| `--unripe-conf` | 0.20 | unripe 인정 최소 confidence |
| `--min-red-for-ripe` | 0.10 | ripe 박스 색 체크 임계값. 0.0=비활성화 |
| `--show-depth` | off | 우측에 depth colormap 표시 |
| `--match-dist` | 60 | 같은 딸기로 매칭할 최대 픽셀 거리 |

키: `q`=종료, `p`=스냅샷 저장. 스냅샷은 `runs/realsense/`에 저장.

---

## 7. 정적 이미지 추론 + 픽 후보 JSON

```bash
python scripts/predict.py \
    --weights runs/detect/runs/strawberry/yolo26m_unified_640b16/weights/best.pt \
    --source datasets/yolo_unified/images/test \
    --out runs/strawberry/predict_picks.json \
    --conf 0.35 --imgsz 640 --save-images
```

JSON 구조: `picks` (ripe 수확 후보, occluded·red_ratio·conf 정렬) / `avoid` (unripe 회피).

---

## 다음 작업

- realscene·strawberryDataset 라벨 정확도 추가 검토
- val 분포 균형 (현재 unripe 59 / ripe 86)
- 야외 조명·복잡 배경 데이터 추가 (현재 모형/스튜디오 비중 큼)
- ONNX/TensorRT export → 로봇 임베디드 배포

---

## Repository

GitHub: <https://github.com/yanggangyiplus/yolo26m_strawberry>
