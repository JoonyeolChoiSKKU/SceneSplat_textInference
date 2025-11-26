# SceneSplat 텍스트 쿼리 빠른 시작 가이드

이 가이드는 SceneSplat을 사용하여 텍스트 쿼리로 3DGS 장면에서 특정 영역을 하이라이트하는 **가장 빠른 방법**을 설명합니다.

## 전제 조건

- ✅ Conda 환경 설치 완료 (`scene_splat`)
- ✅ CUDA 12.4 + GCC 13 설정 완료

## 3단계로 시작하기

### 1단계: 체크포인트 다운로드

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat
bash scripts/download_checkpoint.sh
```

또는 수동으로:
- [HuggingFace 페이지](https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive) 방문
- `model_best.pth` 다운로드
- `/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/`에 저장

### 2단계: 3DGS 장면 준비

#### PLY 파일이 있는 경우:
```bash
conda activate scene_splat
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

python scripts/preprocess_gs.py \
    --input /path/to/your/scene.ply \
    --output /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene
```

#### 이미 NPY 파일이 있는 경우:
다음 구조로 정리:
```
/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/
└── my_scene/
    ├── coord.npy      # [N, 3]
    ├── color.npy      # [N, 3] (0-255 uint8)
    ├── opacity.npy    # [N] 또는 [N, 1]
    ├── scale.npy      # [N, 3]
    └── quat.npy       # [N, 4]
```

### 3단계: 텍스트 쿼리 실행 (한 줄!)

```bash
cd /home/stevenlair/LAIR/AGI-Memory/SceneSplat

bash scripts/run_text_query_pipeline.sh \
    --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene \
    --query "pen on desk" \
    --visualize
```

**끝!** 결과는 `/media/stevenlair/Data_exFAT/SceneSplat/results/my_scene_pen_on_desk/`에 저장됩니다.

## 고급 옵션

### 임계값 조정
```bash
bash scripts/run_text_query_pipeline.sh \
    --scene-path /path/to/scene \
    --query "pen" \
    --threshold 0.3  # 더 엄격한 매칭
```

### 상위 k%만 선택
```bash
bash scripts/run_text_query_pipeline.sh \
    --scene-path /path/to/scene \
    --query "pen" \
    --top-k-ratio 0.1  # 상위 10%만 하이라이트
```

### 여러 쿼리 실행
```bash
for query in "pen" "red mug" "chair"; do
    bash scripts/run_text_query_pipeline.sh \
        --scene-path /path/to/scene \
        --query "$query" \
        --threshold 0.25
done
```

## 결과 파일

각 쿼리 실행 후 다음 파일들이 생성됩니다:

- `highlight_mask.npy` - 하이라이트 마스크 (boolean, [N])
- `similarity_scores.npy` - 유사도 점수 (float, [N])
- `highlighted_color.npy` - 하이라이트된 색상 (uint8, [N, 3])
- `coord.npy` - 좌표 (참고용)

## 문제 해결

### "체크포인트를 찾을 수 없습니다"
→ 1단계를 다시 실행하세요.

### "장면 데이터를 찾을 수 없습니다"
→ 2단계에서 PLY → NPY 변환을 확인하세요.

### CUDA out of memory
→ `configs/custom/text_query_demo.py`에서 `test_cfg.voxelize.grid_size`를 0.02 → 0.03으로 증가

### 하이라이트 결과가 너무 많음/적음
→ `--threshold` 값을 조정하거나 `--top-k-ratio` 사용

## 더 자세한 정보

전체 가이드는 [TEXT_QUERY_GUIDE.md](TEXT_QUERY_GUIDE.md)를 참고하세요.

