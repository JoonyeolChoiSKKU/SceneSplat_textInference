# SceneSplat 텍스트 쿼리 하이라이트 가이드

이 가이드는 SceneSplat을 사용하여 텍스트 쿼리(예: "책상 위에 있는 펜")로 3DGS 장면에서 특정 영역을 하이라이트하는 방법을 설명합니다.

## 전체 워크플로우

1. **환경 설치** ✅ (완료)
2. **체크포인트 다운로드** (HuggingFace에서 사전학습된 모델)
3. **3DGS 장면 준비** (PLY → NPY 변환)
4. **특징 추출** (모델로 장면 특징 계산)
5. **텍스트 쿼리 하이라이트** (쿼리로 영역 찾기)

---

## 1. 체크포인트 다운로드

### 방법 1: 자동화 스크립트 사용 (권장)

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat
bash scripts/download_checkpoint.sh
```

스크립트가 HuggingFace에서 체크포인트를 자동으로 다운로드합니다.

### 방법 2: 수동 다운로드

HuggingFace에서 사전학습된 모델을 다운로드합니다:

```bash
# HuggingFace 로그인 (필요시)
huggingface-cli login

# 체크포인트 다운로드
# https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive
# 또는 직접 웹에서 다운로드 후 압축 해제

# 예시 경로
mkdir -p /media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat
# model_best.pth를 위 경로에 저장
```

---

## 2. 3DGS 장면 준비

### 2.1 PLY 파일이 있는 경우

`scripts/preprocess_gs.py`를 사용하여 PLY를 NPY로 변환:

```bash
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python scripts/preprocess_gs.py \
    --input /path/to/your/scene.ply \
    --output /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene
```

또는 디렉토리 전체 처리:

```bash
python scripts/preprocess_gs.py \
    --input /path/to/ply_directory \
    --output /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes \
    --recursive
```

### 2.2 이미 NPY 파일이 있는 경우

다음 구조로 정리:

```
/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/
└── my_scene/
    ├── coord.npy      # [N, 3] 좌표
    ├── color.npy      # [N, 3] 색상 (0-255 uint8 또는 0-1 float)
    ├── opacity.npy    # [N] 또는 [N, 1] 불투명도
    ├── scale.npy      # [N, 3] 스케일
    └── quat.npy       # [N, 4] 쿼터니언
```

---

## 3. 특징 추출 (Inference)

커스텀 장면에서 모델 특징을 추출합니다:

```bash
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python tools/test.py \
    --config-file configs/custom/text_query_demo.py \
    --options \
        test_only=True \
        test.skip_eval=True \
        test.save_feat=True \
        data.test.data_root=/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes \
        data.test.split=my_scene \
        weight=/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth \
        save_path=/media/stevenlair/Data_exFAT/SceneSplat/runs/my_scene_features \
    --num-gpus 1
```

**결과:**
- `save_path/result_GenericGSDataset/feat/my_scene_feat.pth` - 특징 파일 (N, 768)

---

## 4. 텍스트 쿼리 하이라이트

### 방법 1: 전체 파이프라인 자동 실행 (권장)

한 번에 특징 추출부터 하이라이트까지 모두 실행:

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

bash scripts/run_text_query_pipeline.sh \
    --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene \
    --query "pen on desk" \
    --threshold 0.25 \
    --visualize
```

**옵션:**
- `--checkpoint PATH`: 체크포인트 경로 (기본값 사용 시 생략 가능)
- `--threshold FLOAT`: 유사도 임계값 (기본: 0.25)
- `--top-k-ratio FLOAT`: 상위 k%만 선택 (예: 0.1)
- `--visualize`: Open3D로 결과 시각화

### 방법 2: 수동 실행 (단계별)

#### 4.1 특징 추출

```bash
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python tools/test.py \
    --config-file configs/custom/text_query_demo.py \
    --options \
        test_only=True \
        test.skip_eval=True \
        test.save_feat=True \
        data.test.data_root=/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes \
        data.test.split=my_scene \
        weight=/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth \
        save_path=/media/stevenlair/Data_exFAT/SceneSplat/runs/my_scene_features \
    --num-gpus 1
```

#### 4.2 텍스트 쿼리 하이라이트

```bash
python tools/query_highlight_demo.py \
    --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene \
    --feat-path /media/stevenlair/Data_exFAT/SceneSplat/runs/my_scene_features/result_GenericGSDataset/feat/my_scene_feat.pth \
    --query "pen on desk" \
    --output /media/stevenlair/Data_exFAT/SceneSplat/results/pen_highlight \
    --threshold 0.25 \
    --visualize
```

**옵션:**
- `--threshold`: 유사도 임계값 (0~1, 기본값: 0.25)
- `--top-k-ratio`: 상위 k%만 선택 (예: 0.1 = 상위 10%, threshold보다 우선)
- `--visualize`: Open3D로 결과 시각화

**결과:**
- `output/highlight_mask.npy` - 하이라이트 마스크 (boolean)
- `output/similarity_scores.npy` - 유사도 점수
- `output/highlighted_color.npy` - 하이라이트된 색상

---

## 5. 고급 사용법

### 5.1 여러 쿼리 동시 처리

```bash
for query in "pen" "red mug" "chair"; do
    python tools/query_highlight_demo.py \
        --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene \
        --feat-path /path/to/feat.pth \
        --query "$query" \
        --output /media/stevenlair/Data_exFAT/SceneSplat/results/${query// /_} \
        --threshold 0.2
done
```

### 5.2 임계값 조정

- **낮은 임계값 (0.15~0.2)**: 더 많은 영역 하이라이트 (노이즈 포함 가능)
- **높은 임계값 (0.3~0.4)**: 정확한 영역만 하이라이트 (일부 누락 가능)

### 5.3 상위 k% 사용

임계값 대신 상위 k%만 선택:

```bash
python tools/query_highlight_demo.py \
    --scene-path /path/to/scene \
    --feat-path /path/to/feat.pth \
    --query "pen" \
    --top-k-ratio 0.05 \
    --output /path/to/output
```

---

## 6. 트러블슈팅

### 문제: "Feature file not found"

**해결:** 먼저 특징 추출을 실행하세요 (3단계).

### 문제: "CUDA out of memory"

**해결:** 
- `configs/custom/text_query_demo.py`에서 `test_cfg.voxelize.grid_size`를 0.02 → 0.03으로 증가
- 또는 `batch_size_test`를 1로 유지

### 문제: 하이라이트 결과가 너무 많음/적음

**해결:**
- `--threshold` 값을 조정하거나
- `--top-k-ratio`를 사용하여 상위 k%만 선택

### 문제: Open3D 시각화가 느림

**해결:**
- 큰 장면의 경우 샘플링 후 시각화:
  ```python
  import numpy as np
  mask = np.load("highlight_mask.npy")
  coord = np.load("coord.npy")
  # 샘플링
  indices = np.random.choice(len(coord), size=min(100000, len(coord)), replace=False)
  coord_sample = coord[indices]
  mask_sample = mask[indices]
  ```

---

## 7. 예제: "책상 위에 있는 펜" 찾기

### 빠른 방법 (자동화 스크립트)

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

bash scripts/run_text_query_pipeline.sh \
    --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/office_scene \
    --query "pen on desk" \
    --threshold 0.25 \
    --visualize
```

### 수동 방법 (단계별)

```bash
# 1. 특징 추출
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python tools/test.py \
    --config-file configs/custom/text_query_demo.py \
    --options \
        test_only=True test.skip_eval=True test.save_feat=True \
        data.test.data_root=/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes \
        data.test.split=office_scene \
        weight=/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth \
        save_path=/media/stevenlair/Data_exFAT/SceneSplat/runs/office_features \
    --num-gpus 1

# 2. 펜 찾기
python tools/query_highlight_demo.py \
    --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/office_scene \
    --feat-path /media/stevenlair/Data_exFAT/SceneSplat/runs/office_features/result_GenericGSDataset/feat/office_scene_feat.pth \
    --query "pen on desk" \
    --output /media/stevenlair/Data_exFAT/SceneSplat/results/pen_on_desk \
    --threshold 0.25 \
    --visualize
```

---

## 참고 자료

- [SceneSplat README](README.md)
- [HuggingFace Dataset](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k)
- [HuggingFace Checkpoint](https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive)

