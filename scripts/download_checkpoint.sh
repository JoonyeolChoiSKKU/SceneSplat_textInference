#!/bin/bash
# SceneSplat 체크포인트 다운로드 스크립트

set -e

CHECKPOINT_DIR="/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat"
CHECKPOINT_URL="https://huggingface.co/GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive"

echo "=== SceneSplat 체크포인트 다운로드 ==="
echo ""
echo "체크포인트 저장 경로: $CHECKPOINT_DIR"
echo "HuggingFace URL: $CHECKPOINT_URL"
echo ""

# 디렉토리 생성
mkdir -p "$CHECKPOINT_DIR"

# HuggingFace CLI 확인
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli가 설치되어 있지 않습니다."
    echo "설치 중..."
    pip install huggingface_hub[cli]
fi

# 로그인 확인
echo "HuggingFace 로그인이 필요할 수 있습니다."
echo "로그인하지 않았다면: huggingface-cli login"
echo ""
read -p "계속하시겠습니까? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "취소되었습니다."
    exit 0
fi

# 체크포인트 다운로드
echo ""
echo "체크포인트 다운로드 중..."
cd "$CHECKPOINT_DIR"

# huggingface-cli를 사용한 다운로드
huggingface-cli download \
    GaussianWorld/SceneSplat_lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive \
    --local-dir . \
    --local-dir-use-symlinks False || {
    echo ""
    echo "다운로드 실패. 수동으로 다운로드하세요:"
    echo "1. 브라우저에서 $CHECKPOINT_URL 방문"
    echo "2. model_best.pth 파일 다운로드"
    echo "3. $CHECKPOINT_DIR에 저장"
    exit 1
}

echo ""
echo "다운로드 완료!"
echo "체크포인트 위치: $CHECKPOINT_DIR/model_best.pth"

