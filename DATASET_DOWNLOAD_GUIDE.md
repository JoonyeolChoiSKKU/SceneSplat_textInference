# SceneSplat-7K 데이터셋 다운로드 가이드

[HuggingFace 데이터셋 페이지](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k)에서 무엇을 다운로드해야 하는지 안내합니다.

## 필수: 라이선스 동의

먼저 데이터셋 페이지에서 **라이선스를 동의**해야 파일에 접근할 수 있습니다.

## 다운로드 옵션

### 옵션 1: 사전 전처리된 데이터 (권장 - 테스트용)

**"Preprocessed Language Pretraining Data"** 섹션에서 다운로드:

#### 최소 구성 (테스트만 실행하려면)
- **ScanNet** - 가장 작고 빠르게 테스트 가능
  - 폴더: `GaussianWorld/scannet_mcmc_3dgs_preprocessed`
  - 필요한 split: `val` (또는 `test_grid1.0cm_chunk6x6_stride3x3`)

#### 전체 구성 (학습 + 테스트)
- **ScanNet** - `GaussianWorld/scannet_mcmc_3dgs_preprocessed`
- **ScanNet++ V2** - `GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed`
- **Matterport3D** - `GaussianWorld/matterport3d_region_mcmc_3dgs_preprocessed`

### 각 데이터셋의 필수 파일 구조

각 장면 폴더에는 다음 파일들이 있어야 합니다:

```
scene_name/
├── color.npy          # [N, 3] 색상
├── coord.npy          # [N, 3] 좌표
├── opacity.npy        # [N] 불투명도
├── quat.npy           # [N, 4] 쿼터니언
├── scale.npy          # [N, 3] 스케일
├── lang_feat.npy      # [N, 768] 언어 특징 (vision-language pretraining용)
└── valid_feat_mask.npy # [N] 유효 마스크
```

### 필요한 Split 폴더

각 데이터셋마다 특정 split 폴더가 필요합니다:

**ScanNet:**
- `train_grid1.0cm_chunk6x6_stride3x3/` - 학습용
- `test_grid1.0cm_chunk6x6_stride3x3/` - 테스트용
- `val/` - 검증용

**ScanNet++ V2:**
- `train_grid1.0cm_chunk6x6_stride3x3/` - 학습용
- `test_grid1.0cm_chunk6x6_stride3x3/` - 테스트용
- `val/` - 검증용

**Matterport3D:**
- `train_grid1.0cm_chunk6x6_stride3x3_filtered/` - 학습용
- `val_grid1.0cm_chunk6x6_stride3x3_filtered/` - 검증용
- `test_grid1.0cm_chunk6x6_stride3x3_filtered/` - 테스트용
- `test/` - 테스트용

## 다운로드 방법

### 방법 1: HuggingFace CLI (권장)

```bash
conda activate scene_splat

# 1. 로그인 (처음 한 번만)
huggingface-cli login

# 2. 데이터셋 다운로드
# 예: ScanNet만 다운로드 (테스트용)
mkdir -p /media/stevenlair/Data_exFAT/SceneSplat/datasets
cd /media/stevenlair/Data_exFAT/SceneSplat/datasets

# ScanNet 다운로드
huggingface-cli download \
    GaussianWorld/scene_splat_7k \
    --repo-type dataset \
    --local-dir . \
    --local-dir-use-symlinks False \
    --include "GaussianWorld/scannet_mcmc_3dgs_preprocessed/**"
```

### 방법 2: 웹 브라우저에서 직접 다운로드

1. https://huggingface.co/datasets/GaussianWorld/scene_splat_7k 방문
2. 라이선스 동의
3. "Files and versions" 탭 클릭
4. "Preprocessed Language Pretraining Data" 섹션에서 원하는 데이터셋 폴더 찾기
5. 폴더를 클릭하고 "Download" 버튼 클릭

### 방법 3: Python 스크립트로 다운로드

```python
from huggingface_hub import snapshot_download

# 데이터셋 다운로드
snapshot_download(
    repo_id="GaussianWorld/scene_splat_7k",
    repo_type="dataset",
    local_dir="/media/stevenlair/Data_exFAT/SceneSplat/datasets",
    local_dir_use_symlinks=False,
    # 특정 폴더만 다운로드하려면 allow_patterns 사용
    allow_patterns=["GaussianWorld/scannet_mcmc_3dgs_preprocessed/**"]
)
```

## 저장 위치 권장

```
/media/stevenlair/Data_exFAT/SceneSplat/datasets/
├── scannet_mcmc_3dgs_preprocessed/
│   ├── train_grid1.0cm_chunk6x6_stride3x3/
│   ├── test_grid1.0cm_chunk6x6_stride3x3/
│   └── val/
├── scannetpp_v2_mcmc_3dgs_preprocessed/
│   └── ...
└── matterport3d_region_mcmc_3dgs_preprocessed/
    └── ...
```

## 다운로드 크기 예상

- **ScanNet (val split만)**: 약 수 GB (테스트용으로 충분)
- **ScanNet 전체**: 수십 GB
- **전체 데이터셋**: 수백 GB

## 빠른 테스트를 위한 최소 구성

테스트만 빠르게 실행하려면:

1. **ScanNet의 val split만** 다운로드
2. Config 파일에서 `data.test.split="val"` 설정
3. `data.test.data_root`를 ScanNet 경로로 설정

이렇게 하면 가장 빠르게 테스트를 실행할 수 있습니다.

## 참고사항

- **라이선스**: 모든 데이터셋은 비상업적 연구/교육 목적으로만 사용 가능
- **ScanNet/ScanNet++**: 원본 플랫폼에서 직접 접근이 필요할 수 있음 (HuggingFace에 직접 호스팅되지 않을 수 있음)
- **디스크 공간**: 충분한 저장 공간 확보 필요 (exFAT 파티션 사용 중이므로 문제없음)

## 다음 단계

데이터 다운로드 완료 후:
1. Config 파일의 `data_root` 경로 수정
2. 테스트 실행 (README_FOLLOW.md 참고)

