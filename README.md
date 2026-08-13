# Deep Learning 실습 프로젝트 (machine_deep/deep)

PyTorch와 Ultralytics YOLO를 활용한 딥러닝 학습 프로젝트입니다.
크게 **두 가지 트랙**으로 구성되어 있습니다.

1. **이미지 분류 (Classification)** — CIFAR-10 샘플 데이터를 VGG11 / ResNet34 전이학습으로 분류
2. **객체 탐지 (Object Detection)** — 복숭아(Peach) 데이터셋을 YOLOv8로 탐지

전체 흐름은 딥러닝 표준 시퀀스를 따라갑니다.

> 데이터 수집 → 전처리(Preprocessing) → 증강(Augmentation) → 모델 선택 → 훈련(Train) → 검증(Eval) → 시각화/평가

---

## 폴더 구조

```
deep/
├── main.py                  # 실행 진입점 (현재는 albumentations 증강 실습 코드)
├── train.py                 # 분류 모델 1 에포크 훈련 함수 (train_one_epoch)
├── eval.py                  # 분류 모델 검증 함수 (evaluate, @torch.no_grad)
├── yolo_settings.yaml       # YOLO 훈련용 데이터셋 설정 (경로, 클래스: peach)
├── yolov8n.pt / yolo26n.pt  # YOLO 사전학습 가중치
├── requirements.txt         # 의존성 목록 (torch, torchvision, ultralytics, opencv, albumentations 등)
│
├── models/                  # 모델 정의
│   ├── vgg.py               #   VGG11 전이학습 (features 동결, classifier만 학습)
│   ├── resnet_pre.py        #   ResNet34 전이학습 (fc만 학습)
│   ├── resnet.py            #   ResNet18/34 직접 구현 (BasicBlock + shortcut)
│   └── yolo.py              #   YOLO 훈련/예측 함수 (yolo_train, yolo_predict)
│
├── Preprocessing/
│   └── yolo_preprocessing.py  # 원본 JSON 라벨 → YOLO txt 라벨 변환,
│                              # train/valid 폴더 구성 및 데이터 이동
│
├── Utils/
│   ├── DataLoader.py        # 커스텀 Dataset(DriveDataset) + DataLoader 생성 (7:3 split)
│   ├── augmentation.py      # OpenCV 기반 데이터 증강 (flip 등, YOLO 라벨 좌표 보정 포함)
│   ├── graph.py             # 훈련 히스토리(loss/acc) 그래프 저장
│   └── visualize.py         # YOLO 라벨 바운딩박스 시각화
│
├── Data/
│   ├── cifar10_samples/     # 분류 실습용 CIFAR-10 샘플 (클래스별 폴더 10개)
│   ├── PeachDataset/        # 복숭아 원본 이미지/라벨 + YOLO 형식 데이터셋
│   └── YoloAugmentation/    # 증강된 YOLO 데이터셋 (images/labels)
│
├── Pth/                     # 분류 모델 체크포인트 (.pth, 최고 정확도 갱신 시 저장)
├── runs/detect/             # YOLO 훈련/예측 결과 (peach_train01, predict 등)
├── image/                   # 훈련 결과 그래프 저장 위치
└── PreVersion/
    └── old_main.py          # 이전 버전 메인 (CIFAR-10 분류 전체 파이프라인)
```

---

## 트랙 1 — CIFAR-10 이미지 분류 (전이학습)

`PreVersion/old_main.py`가 전체 파이프라인을 담고 있습니다.

1. **데이터 로딩** — `Utils/DataLoader.py`의 `DriveDataset`이 `Data/cifar10_samples/` 폴더 구조(클래스별 폴더)를 읽어 라벨을 매핑하고, `random_split`으로 7:3 train/valid 분할 후 DataLoader를 생성합니다.
2. **모델 선택** — 전이학습 모델 사용:
   - `models/vgg.py` : VGG11 사전학습 가중치 로드 후 `features` 동결, `classifier`만 학습
   - `models/resnet_pre.py` : ResNet34 사전학습 가중치 로드 후 `fc`만 학습
   - `models/resnet.py` : 학습용으로 ResNet 구조(BasicBlock, shortcut)를 직접 구현한 파일
3. **훈련/검증 루프** — `train.py`의 `train_one_epoch`과 `eval.py`의 `evaluate`를 에포크마다 반복 (`CrossEntropyLoss` + `Adam`, GPU 자동 사용).
4. **체크포인트 저장** — 검증 정확도가 최고 기록을 갱신할 때마다 `CIFAR10_VGG11_{에포크}epoch_{정확도}.pth` 형식으로 저장 (`Pth/` 폴더에 누적).
5. **결과 시각화** — `Utils/graph.py`의 `draw_plot`으로 loss/accuracy 곡선을 이미지로 저장.

## 트랙 2 — 복숭아 객체 탐지 (YOLOv8)

1. **라벨 전처리** — `Preprocessing/yolo_preprocessing.py`
   - 원본 JSON 라벨에서 `(x, y, w, h)` 추출 → YOLO 포맷 txt 라벨 생성 (`create_yolo_label`)
   - `images/train`, `images/valid`, `labels/train`, `labels/valid` 표준 폴더 구조 생성 및 파일 분배
2. **라벨 검증** — `Utils/visualize.py`의 `show_yolo_label`로 바운딩박스를 그려 라벨이 올바른지 확인.
3. **데이터 증강** — `Utils/augmentation.py`
   - 원본 YOLO 데이터셋을 `Data/YoloAugmentation/`으로 복사 후, 좌우/상하 반전 등을 랜덤 적용
   - 이미지 변환에 맞춰 YOLO 라벨 좌표(cx, cy)도 함께 보정
   - `main.py`에서는 albumentations 라이브러리를 이용한 증강도 실습
4. **훈련** — `models/yolo.py`의 `yolo_train()` : `yolov8n.pt` 가중치로 `yolo_settings.yaml` 데이터셋을 50 에포크 훈련 → 결과는 `runs/detect/peach_train01/`
5. **예측** — `yolo_predict(source)` : 훈련된 `best.pt`로 임의 이미지 추론 (`runs/detect/predict*/`에 저장)

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt
```

```bash
# 현재 main.py 실행 (albumentations 증강 실습)
python main.py
```

```bash
# CIFAR-10 분류 파이프라인 실행 (이전 버전 메인)
python PreVersion/old_main.py
```

YOLO 훈련/예측은 `models/yolo.py`의 `yolo_train()` / `yolo_predict(source)` 함수를 호출하여 실행합니다.

## 참고 사항

- GPU(CUDA) 환경 기준으로 작성되어 있습니다 (`device=0`, `cuda:0`). CPU만 있는 경우 자동으로 CPU로 폴백되지만(분류 트랙), YOLO 훈련 시에는 `device` 인자 수정이 필요합니다.
- `yolo_settings.yaml`과 일부 코드의 데이터 경로가 이전 폴더 경로(`C:\kdh\머신러닝, 딥러닝\...`)로 하드코딩되어 있어, 현재 위치(`C:\kdh\machine_deep\deep`)에서 실행하려면 경로 수정이 필요할 수 있습니다.
- 주석이 학습 노트 형태로 상세히 달려 있어, 각 파일이 딥러닝 개념(전이학습, 배치 정규화, 잔차 연결, 데이터 증강 등)의 실습 자료 역할을 겸합니다.
