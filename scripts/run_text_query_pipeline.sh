#!/bin/bash
# 텍스트 쿼리 하이라이트 전체 파이프라인 실행 스크립트

set -e

# 기본 경로 설정
BASE_DIR="/media/stevenlair/Data_exFAT/SceneSplat"
SCENE_SPLAT_DIR="/home/stevenlair/LAIR/AGI-Memory/SceneSplat"

# 색상 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SceneSplat 텍스트 쿼리 파이프라인 ===${NC}"
echo ""

# 인자 파싱
SCENE_PATH=""
QUERY=""
CHECKPOINT_PATH="$BASE_DIR/checkpoints/lang-pretrain-concat/model_best.pth"
THRESHOLD=0.25
TOP_K_RATIO=""
VISUALIZE=false

usage() {
    echo "사용법: $0 --scene-path <경로> --query <쿼리> [옵션]"
    echo ""
    echo "필수 옵션:"
    echo "  --scene-path PATH    3DGS 장면 폴더 경로 (coord.npy 등이 있는 곳)"
    echo "  --query TEXT         검색할 텍스트 쿼리 (예: 'pen on desk')"
    echo ""
    echo "선택 옵션:"
    echo "  --checkpoint PATH    체크포인트 경로 (기본: $CHECKPOINT_PATH)"
    echo "  --threshold FLOAT    유사도 임계값 (기본: 0.25)"
    echo "  --top-k-ratio FLOAT  상위 k%만 선택 (예: 0.1)"
    echo "  --visualize          결과 시각화"
    echo "  --help               도움말 표시"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene-path)
            SCENE_PATH="$2"
            shift 2
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --top-k-ratio)
            TOP_K_RATIO="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            usage
            ;;
    esac
done

# 필수 인자 확인
if [ -z "$SCENE_PATH" ] || [ -z "$QUERY" ]; then
    echo "에러: --scene-path와 --query는 필수입니다."
    usage
fi

# 경로 정규화
SCENE_PATH=$(realpath "$SCENE_PATH")
SCENE_NAME=$(basename "$SCENE_PATH")
FEAT_OUTPUT_DIR="$BASE_DIR/runs/${SCENE_NAME}_features"
QUERY_SAFE=$(echo "$QUERY" | tr ' ' '_' | tr '/' '_')
HIGHLIGHT_OUTPUT="$BASE_DIR/results/${SCENE_NAME}_${QUERY_SAFE}"

echo -e "${GREEN}설정:${NC}"
echo "  장면 경로: $SCENE_PATH"
echo "  쿼리: $QUERY"
echo "  체크포인트: $CHECKPOINT_PATH"
echo "  임계값: $THRESHOLD"
echo ""

# 1. 체크포인트 확인
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}경고: 체크포인트를 찾을 수 없습니다: $CHECKPOINT_PATH${NC}"
    echo "다운로드 스크립트를 실행하세요:"
    echo "  bash $SCENE_SPLAT_DIR/scripts/download_checkpoint.sh"
    exit 1
fi

# 2. 장면 데이터 확인
if [ ! -f "$SCENE_PATH/coord.npy" ]; then
    echo -e "${YELLOW}에러: 장면 데이터를 찾을 수 없습니다: $SCENE_PATH/coord.npy${NC}"
    echo "PLY 파일이 있다면 먼저 전처리하세요:"
    echo "  python $SCENE_SPLAT_DIR/scripts/preprocess_gs.py --input <ply_file> --output $SCENE_PATH"
    exit 1
fi

# 3. 특징 추출 (이미 있으면 스킵)
FEAT_PATH="$FEAT_OUTPUT_DIR/result_GenericGSDataset/feat/${SCENE_NAME}_feat.pth"
if [ ! -f "$FEAT_PATH" ]; then
    echo -e "${BLUE}[1/3] 특징 추출 중...${NC}"
    cd "$SCENE_SPLAT_DIR"
    
    conda activate scene_splat
    
    python tools/test.py \
        --config-file configs/custom/text_query_demo.py \
        --options \
            test_only=True \
            test.skip_eval=True \
            test.save_feat=True \
            data.test.data_root=$(dirname "$SCENE_PATH") \
            data.test.split="$SCENE_NAME" \
            weight="$CHECKPOINT_PATH" \
            save_path="$FEAT_OUTPUT_DIR" \
        --num-gpus 1
    
    if [ ! -f "$FEAT_PATH" ]; then
        echo -e "${YELLOW}에러: 특징 추출 실패${NC}"
        exit 1
    fi
    echo -e "${GREEN}특징 추출 완료: $FEAT_PATH${NC}"
else
    echo -e "${GREEN}[1/3] 특징 파일이 이미 존재합니다: $FEAT_PATH${NC}"
fi

# 4. 텍스트 쿼리 하이라이트
echo ""
echo -e "${BLUE}[2/3] 텍스트 쿼리 하이라이트 중...${NC}"
cd "$SCENE_SPLAT_DIR"
conda activate scene_splat

VISUALIZE_FLAG=""
if [ "$VISUALIZE" = true ]; then
    VISUALIZE_FLAG="--visualize"
fi

TOP_K_FLAG=""
if [ -n "$TOP_K_RATIO" ]; then
    TOP_K_FLAG="--top-k-ratio $TOP_K_RATIO"
fi

python tools/query_highlight_demo.py \
    --scene-path "$SCENE_PATH" \
    --feat-path "$FEAT_PATH" \
    --query "$QUERY" \
    --output "$HIGHLIGHT_OUTPUT" \
    --threshold "$THRESHOLD" \
    $TOP_K_FLAG \
    $VISUALIZE_FLAG

echo ""
echo -e "${GREEN}[3/3] 완료!${NC}"
echo ""
echo -e "${GREEN}결과 파일:${NC}"
echo "  마스크: $HIGHLIGHT_OUTPUT/highlight_mask.npy"
echo "  점수: $HIGHLIGHT_OUTPUT/similarity_scores.npy"
echo "  하이라이트 색상: $HIGHLIGHT_OUTPUT/highlighted_color.npy"
echo ""

