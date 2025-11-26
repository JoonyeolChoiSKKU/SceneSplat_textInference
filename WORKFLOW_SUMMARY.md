# SceneSplat 텍스트 쿼리 워크플로우 요약

## 전체 프로세스 개요

```
[체크포인트] → [3DGS 장면 준비] → [특징 추출] → [텍스트 쿼리] → [하이라이트 결과]
     ↓              ↓                  ↓              ↓              ↓
  model_best.pth  *.npy files    feat.pth      "pen on desk"   mask.npy
```

## 단계별 상세

### 1. 체크포인트 다운로드
**목적**: 사전학습된 모델 가중치 다운로드

**방법**:
```bash
bash scripts/download_checkpoint.sh
```

**결과**: `/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth`

---

### 2. 3DGS 장면 준비
**목적**: PLY 파일을 NPY 형식으로 변환

**방법**:
```bash
python scripts/preprocess_gs.py --input scene.ply --output /media/.../custom_scenes/my_scene
```

**필수 파일**:
- `coord.npy` - [N, 3] 좌표
- `color.npy` - [N, 3] 색상 (0-255 uint8)
- `opacity.npy` - [N] 불투명도
- `scale.npy` - [N, 3] 스케일
- `quat.npy` - [N, 4] 쿼터니언

**디렉토리 구조**:
```
custom_scenes/
└── my_scene/
    ├── coord.npy
    ├── color.npy
    ├── opacity.npy
    ├── scale.npy
    └── quat.npy
```

---

### 3. 특징 추출 (Inference)
**목적**: 3DGS 장면에서 768차원 특징 벡터 추출

**방법**:
```bash
python tools/test.py \
    --config-file configs/custom/text_query_demo.py \
    --options \
        test_only=True \
        test.skip_eval=True \
        test.save_feat=True \
        data.test.data_root=/media/.../custom_scenes \
        data.test.split=my_scene \
        weight=/media/.../checkpoints/.../model_best.pth \
        save_path=/media/.../runs/my_scene_features \
    --num-gpus 1
```

**결과**: 
- `save_path/result_GenericGSDataset/feat/my_scene_feat.pth` - [N, 768] 특징

**설정 파일**: `configs/custom/text_query_demo.py`
- `test.save_feat=True` - 특징 저장 활성화
- `test.skip_eval=True` - 평가 스킵
- `data.test.type="GenericGSDataset"` - 커스텀 데이터셋 사용

---

### 4. 텍스트 쿼리 하이라이트
**목적**: 텍스트 쿼리와 특징의 유사도를 계산하여 하이라이트 영역 찾기

**방법 (자동화)**:
```bash
bash scripts/run_text_query_pipeline.sh \
    --scene-path /media/.../custom_scenes/my_scene \
    --query "pen on desk" \
    --threshold 0.25 \
    --visualize
```

**방법 (수동)**:
```bash
python tools/query_highlight_demo.py \
    --scene-path /media/.../custom_scenes/my_scene \
    --feat-path /media/.../runs/.../feat/my_scene_feat.pth \
    --query "pen on desk" \
    --output /media/.../results/pen_highlight \
    --threshold 0.25
```

**내부 동작**:
1. 텍스트 쿼리를 SigLIP2로 임베딩 변환
2. 특징과 쿼리 임베딩의 cosine similarity 계산
3. 임계값을 넘는 점들을 마스크로 생성
4. 하이라이트 색상 적용 및 저장

**결과 파일**:
- `highlight_mask.npy` - [N] boolean 마스크
- `similarity_scores.npy` - [N] 유사도 점수
- `highlighted_color.npy` - [N, 3] 하이라이트 색상

---

## 핵심 파일 및 설정

### 설정 파일
- `configs/custom/text_query_demo.py` - 커스텀 장면용 설정
- `env.yaml` - Conda 환경 설정

### 스크립트
- `scripts/download_checkpoint.sh` - 체크포인트 다운로드
- `scripts/preprocess_gs.py` - PLY → NPY 변환
- `scripts/run_text_query_pipeline.sh` - 전체 파이프라인 자동 실행
- `tools/query_highlight_demo.py` - 텍스트 쿼리 하이라이트

### 가이드 문서
- `QUICKSTART.md` - 빠른 시작 가이드
- `TEXT_QUERY_GUIDE.md` - 상세 가이드
- `WORKFLOW_SUMMARY.md` - 이 문서

---

## 파라미터 튜닝 가이드

### 임계값 (threshold)
- **낮은 값 (0.15~0.2)**: 더 많은 영역 하이라이트 (노이즈 포함 가능)
- **기본값 (0.25)**: 균형잡힌 결과
- **높은 값 (0.3~0.4)**: 정확한 영역만 하이라이트 (일부 누락 가능)

### 상위 k% 선택 (top-k-ratio)
- 임계값 대신 상위 k%만 선택
- 예: `--top-k-ratio 0.1` = 상위 10%만 하이라이트
- 장면 크기에 관계없이 일정 비율 유지

### Grid Size (메모리 부족 시)
- `configs/custom/text_query_demo.py`에서 수정
- `test_cfg.voxelize.grid_size`: 0.02 → 0.03 (더 큰 그리드 = 더 적은 메모리)

---

## 트러블슈팅 체크리스트

- [ ] Conda 환경 활성화: `conda activate scene_splat`
- [ ] CUDA/GCC 설정: `use-cuda12 && use-gcc13`
- [ ] 체크포인트 존재 확인: `/media/.../checkpoints/.../model_best.pth`
- [ ] 장면 데이터 확인: `coord.npy`, `color.npy` 등 필수 파일 존재
- [ ] 특징 파일 확인: `*_feat.pth` 파일 생성 여부
- [ ] 로그 확인: `save_path/test.log`에서 에러 메시지 확인

---

## 다음 단계

1. **여러 쿼리 실행**: 여러 객체를 동시에 찾기
2. **결과 시각화**: Open3D로 3D 뷰어에서 확인
3. **임계값 최적화**: 장면에 맞는 최적 임계값 찾기
4. **배치 처리**: 여러 장면에 대해 자동화

더 자세한 정보는 [TEXT_QUERY_GUIDE.md](TEXT_QUERY_GUIDE.md)를 참고하세요.

