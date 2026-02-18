# Orchard-Noxious-Algae-ControlSys
## 과수원 유해조류 퇴치 시스템 🍎🪶

과수원에 출몰하는 유해 조류를 실시간으로 감지하고 퇴치하는 엣지 AI 시스템

## 프로젝트 구조

```
⚙️hardware/              # 엣지 디바이스별 추론 스크립트
├── 🕹️raspberry_pi/      # Raspberry Pi (CPU / Coral USB / Movidius NCS)
├── 🖲️NVIDIA® Jetson Nano™/  # Jetson Nano + GPU 모니터링
└── 🪸Google Coral Dev Board/ # Google Coral EdgeTPU

🗂️src/                   # 소스 코드
└── 🔍detection/
    ├── 🟡yolo/           # YOLOv8 커스텀 학습 & 추론
    └── 🦕grounding_dino/ # Grounding DINO zero-shot 감지

🗃️data/                  # 데이터
├── 📈benchmark/          # 디바이스별 벤치마크 결과 (CSV)
├── 🕊️sample/            # 샘플 데이터셋
└── 🦅real-time/          # 실시간 데이터셋
```

## 엣지 디바이스 벤치마크

| 디바이스 | 가속기 | 모델 | 평균 추론 시간 |
|---------|--------|------|--------------|
| Raspberry Pi | CPU only | YOLOv8m | ~0.35s |
| Raspberry Pi | Google Coral USB | YOLOv8m | ~0.35s |
| Raspberry Pi | Intel Movidius NCS | YOLOv8m | ~0.36s |

## 감지 모델

- **YOLOv8** (n/s/m) - 커스텀 데이터셋 학습, TFLite 변환 지원
- **Grounding DINO** - zero-shot 텍스트 기반 객체 감지

## 사용법

```bash
# Raspberry Pi
python rasb_performance_comparison.py \
    --video ./video/input.mp4 \
    --model-dir ./models \
    --output ./output \
    --mode cpu  # cpu | movidius | coral

# Jetson Nano
python jetson.py \
    --video ./video/input.mp4 \
    --model-dir ./models \
    --output ./output

# Google Coral Dev Board
python "Google Coral Dev Board.py" \
    --video ./video/input.mp4 \
    --model-dir ./models \
    --output ./output
```

## 데이터 전처리

- `voc_to_yolo_converter.py` - Pascal VOC → YOLO 포맷 변환
- `yolo_dataset_cleaner.py` - 빈 라벨 제거 & 데이터셋 정리
- `bird_arguementations.py` - 이미지 증강 (flip, jitter, affine)
