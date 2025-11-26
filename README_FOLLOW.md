# SceneSplat 원본 README 따라하기 가이드

원본 README의 기본 워크플로우를 단계별로 따라하는 가이드입니다.

## 전제 조건 확인

```bash
# 1. Conda 환경 활성화
conda activate scene_splat

# 2. CUDA/GCC 설정
use-cuda12
use-gcc13

# 3. 환경 확인
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## 1단계: 체크포인트 다운로드

README의 "Reproducing the results" 섹션에 따르면, 체크포인트는 HuggingFace에서 다운로드할 수 있습니다.

### 방법 1: 자동 스크립트 (대화형)

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat
bash scripts/download_checkpoint.sh
```

스크립트가 HuggingFace 로그인을 요청할 수 있습니다. 로그인이 필요하면:
```bash
huggingface-cli login
```

### 방법 2: 수동 다운로드

1. 브라우저에서 방문: https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive
2. `model_best.pth` 파일 다운로드
3. 다음 경로에 저장:
   ```bash
   mkdir -p /media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat
   # 다운로드한 model_best.pth를 위 경로로 이동
   ```

### 방법 3: huggingface-cli로 직접 다운로드

```bash
conda activate scene_splat
mkdir -p /media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat
cd /media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat

huggingface-cli download \
    GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive \
    --local-dir . \
    --local-dir-use-symlinks False
```

## 2단계: 데이터셋 준비

README에 따르면, 사전 전처리된 데이터를 사용하는 것을 권장합니다.

### 옵션 A: 사전 전처리된 데이터 사용 (권장)

1. HuggingFace 데이터셋 페이지 방문: https://huggingface.co/datasets/GaussianWorld/scene_splat_7k
2. 라이선스 동의 (필수)
3. "Preprocessed Pretraining Data" 섹션에서 필요한 데이터셋 다운로드:
   - ScanNet
   - ScanNet++
   - Matterport3D

다운로드 후 데이터 구조:
```
/media/stevenlair/Data_exFAT/SceneSplat/datasets/
├── scannet_3dgs_mcmc_preprocessed/
│   ├── train/
│   │   └── scene0000_00/
│   │       ├── coord.npy
│   │       ├── color.npy
│   │       ├── opacity.npy
│   │       ├── quat.npy
│   │       ├── scale.npy
│   │       ├── lang_feat.npy
│   │       └── valid_feat_mask.npy
│   └── val/
└── ...
```

### 옵션 B: 원본 3DGS 장면에서 전처리

원본 PLY 파일이 있다면 `scripts/preprocess_gs.py`를 사용하여 전처리할 수 있습니다.

## 3단계: Config 파일 경로 수정

원본 config 파일의 경로를 사용자 환경에 맞게 수정해야 합니다.

예: `configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py`

```python
# 수정 전
data_root = "/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannet_3dgs_mcmc_preprocessed"
repo_root = "/home/yli7/projects/release/SceneSplat_release"

# 수정 후 (예시)
data_root = "/media/stevenlair/Data_exFAT/SceneSplat/datasets/scannet_3dgs_mcmc_preprocessed"
repo_root = "/home/stevenlair/LAIR/AGI-Memory/SceneSplat"
```

## 4단계: 테스트 실행

README의 "Testing" 섹션에 따르면, `test_only=True`로 테스트만 실행할 수 있습니다.

### ScanNet 테스트 예시

```bash
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python tools/train.py \
    --config-file configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py \
    --options \
        weight=/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth \
        test_only=True \
        data.test.data_root=/media/stevenlair/Data_exFAT/SceneSplat/datasets/scannet_3dgs_mcmc_preprocessed \
        repo_root=/home/stevenlair/LAIR/AGI-Memory/SceneSplat \
        save_path=/media/stevenlair/Data_exFAT/SceneSplat/runs/scannet_test \
    --num-gpus 1
```

### Multi-dataset 테스트 (concat)

```bash
python tools/train.py \
    --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
    --options \
        weight=/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth \
        test_only=True \
        repo_root=/home/stevenlair/LAIR/AGI-Memory/SceneSplat \
        save_path=/media/stevenlair/Data_exFAT/SceneSplat/runs/concat_test \
    --num-gpus 1
```

## 5단계: 결과 확인

테스트 완료 후 결과는 `save_path` 디렉토리에 저장됩니다:

- `save_path/result_*/` - 예측 결과
- `save_path/test.log` - 로그 파일
- `save_path/submit/` - 제출용 파일 (해당하는 경우)

## 트러블슈팅

### 문제: "No module named 'pointcept'"
→ 현재 디렉토리에서 실행: `cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat`

### 문제: "CUDA out of memory"
→ `batch_size_test`를 줄이거나 `grid_size`를 키우세요

### 문제: "FileNotFoundError: class_names.txt"
→ `repo_root` 경로가 올바른지 확인하세요

### 문제: 데이터 경로 오류
→ config 파일의 `data_root`와 `repo_root`를 사용자 환경에 맞게 수정하세요

## 다음 단계

기본 테스트가 성공하면:
- 다른 데이터셋으로 테스트
- 커스텀 데이터로 확장 (TEXT_QUERY_GUIDE.md 참고)
- 학습 실행 (필요한 경우)

