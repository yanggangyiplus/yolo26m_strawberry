# 딸기 수확 로봇용 YOLO26m 파인튜닝

HuggingFace 등에서 모은 데이터로 **Ultralytics YOLO26** (`yolo26m.pt` 기본)을 2-class(unripe / ripe) 검출용으로 fine-tuning하고, `predict.py`로 픽 후보 JSON을 만드는 파이프라인입니다.

**YOLO26:** `ultralytics>=8.4` 권장. 첫 실행 시 `yolo26m.pt`가 자동 다운로드됩니다. YOLO11을 쓰려면 `--weights yolo11m.pt`만 지정하면 됩니다.

## 데이터셋 요약

| 항목 | 값 |
| --- | --- |
| 라벨 포맷 | CVAT XML (`annotations.xml`) |
| 클래스 | `strawberry` 1종 (모두 ripe만 라벨링됨) |
| 박스 속성 | `xtl/ytl/xbr/ybr` + `occluded` (0/1) |
| 샘플 규모 | **40장 / 475 박스** (HuggingFace 공개 샘플) — 전체 데이터셋은 상업 판매 |
| 라이선스 | `cc-by-nc-nd-4.0` (비상업 / 변경 금지) |

> 위 표는 UniqueData 샘플 기준입니다. Qin2006·사용자 사진까지 합치면 `configs/strawberry.yaml`의 2-class 스키마로 통합합니다.

## 폴더 구조

```
yolo_strawberry/
├─ yolo26m.pt                       # YOLO26m 사전학습 가중치 (기본)
├─ yolo11m.pt / yolo11m/            # (선택) YOLO11 레거시
├─ configs/
│  └─ strawberry.yaml               # 2-class 데이터 설정
├─ scripts/
│  ├─ convert_uniquedata_to_yolo.py # UniqueData CVAT XML → YOLO
│  ├─ convert_qin_to_yolo.py        # Qin2006 → 통합 2-class
│  ├─ autolabel_user_pics.py        # raw_str / red_str 자동 박스
│  ├─ split_dataset.py
│  ├─ train.py                      # 기본 weights=yolo26m.pt
│  ├─ predict.py
│  └─ archive/convert_aihub_to_yolo.py
├─ datasets/
│  ├─ raw/                          # HF 데이터셋 git clone 위치
│  │  └─ data/{annotations.xml, images.tar.gz, ...}
│  └─ yolo/                         # 변환·분할 결과 (자동 생성)
│     ├─ images/{all,train,val,test}
│     ├─ labels/{all,train,val,test}
│     └─ occluded_meta.json         # GT occluded 정보 (predict.py가 활용)
├─ runs/                            # 학습/추론 산출물
├─ requirements.txt
└─ .ultralytics_cfg/                # Ultralytics 설정 (env로 지정)
```

## 0. 사전 준비

```bash
pip install -r requirements.txt

# Ultralytics 설정 디렉토리를 워크스페이스 안으로 고정 (선택)
export YOLO_CONFIG_DIR="$(pwd)/.ultralytics_cfg"
```

## 1. 데이터셋 다운로드

```bash
# git-lfs 필요
git clone https://huggingface.co/datasets/UniqueData/ripe-strawberries-detection datasets/raw
```

`datasets/raw/data/` 아래에 `annotations.xml`, `images.tar.gz` 등이 생깁니다. 이미지 압축은 변환 스크립트가 자동으로 풀어줍니다.

## 2. CVAT XML → YOLO 포맷 변환

```bash
python scripts/convert_uniquedata_to_yolo.py \
  --xml datasets/raw/data/annotations.xml \
  --dst datasets/yolo
```

- UniqueData 박스는 통합 스키마에서 **ripe (id=1)** 로 매핑됩니다.
- `datasets/yolo/occluded_meta.json` 에 GT `occluded` 정보가 병합 저장됩니다 (predict 보조).

## 3. train/val/test 분할

```bash
python scripts/split_dataset.py \
  --root datasets/yolo \
  --train 0.8 --val 0.1 --test 0.1 --seed 42
```

40장 샘플로는 val/test가 너무 작아 메트릭 변동이 큽니다. 본 데이터셋 확보 후엔 `--train 0.85 --val 0.1 --test 0.05` 정도 권장.

## 4. 학습

```bash
python scripts/train.py \
  --weights yolo26m.pt \
  --data configs/strawberry.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

- 결과: `runs/strawberry/yolo26m_ft/weights/best.pt` (기본 `--name yolo26m_ft`)

### 빠른 동작 확인 (CPU)

```bash
python scripts/train.py \
  --data configs/strawberry.yaml \
  --epochs 3 --imgsz 640 --batch 4 --device cpu --workers 0
```

## 5. 검증 / 추론

```bash
# 메트릭 확인
yolo detect val model=runs/strawberry/yolo26m_ft/weights/best.pt \
  data=configs/strawberry.yaml imgsz=640

# 수확 픽 후보 산출
python scripts/predict.py \
  --weights runs/strawberry/yolo26m_ft/weights/best.pt \
  --source datasets/yolo/images/test \
  --out runs/strawberry/predict_picks.json \
  --conf 0.35 --imgsz 640 --save-images
```

`predict_picks.json` 구조 (2-class):

- `picks`: **ripe (cls=1)** 후보만 (수확 대상)
- `avoid`: **unripe (cls=0)** 검출 (그리퍼 회피 참고)

각 항목 예:

```json
{
  "bbox_xyxy": [x1, y1, x2, y2],
  "center": [cx, cy],
  "conf": 0.999,
  "cls": 1,
  "red_ratio": 0.60,
  "is_red": true,
  "occluded": false
}
```

**ripe 후보 정렬** (`predict.py`): `occluded=False` 우선 → `red_ratio` 높음 → `conf` 높음.

## 6. 다음 단계 (선택)

- **데이터 추가**: 40장은 데모용. 자체 촬영 + LabelImg/CVAT로 라벨링하거나, AIHub 596(이미 변환기 `scripts/archive/convert_aihub_to_yolo.py` 존재)을 추가로 변환해 학습 데이터를 늘리세요.
- **가림(occluded)**: UniqueData에서만 `occluded_meta.json`으로 보조 정렬 가능.
- **Ripeness 분류기**: 현재는 색상 휴리스틱(`red_ratio`). 본 데이터셋이 ripe만 라벨링되어 있어 unripe 구분이 필요하면 별도 분류기 추가.
- **모델 export**: 실시간 로봇 추론용 → `yolo export model=best.pt format=onnx imgsz=640` 또는 `format=engine` (TensorRT).
- **Depth/PCD 결합**: RGB-D 카메라가 있다면 박스 → 3D 좌표 역투영하여 그리퍼 자세 계산.

## 참고

| 항목 | 출처 |
| --- | --- |
| 데이터셋 | <https://huggingface.co/datasets/UniqueData/ripe-strawberries-detection> |
| 라벨 도구 | CVAT (Computer Vision Annotation Tool) |
| 베이스 모델 | YOLO26m (기본), Ultralytics 8.4+ / 필요 시 `yolo11m.pt` |
| 가중치 | [ultralytics/assets](https://github.com/ultralytics/assets/releases) (`yolo26m.pt` 등) |
