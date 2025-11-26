#!/bin/bash
# 자신의 fork repository로 push하기 위한 설정 스크립트

set -e

echo "=== SceneSplat Fork 설정 스크립트 ==="
echo ""

# 현재 remote 확인
echo "현재 remote 설정:"
git remote -v
echo ""

# 사용자 fork URL 입력
read -p "자신의 fork repository URL을 입력하세요 (예: https://github.com/YOUR_USERNAME/SceneSplat.git): " FORK_URL

if [ -z "$FORK_URL" ]; then
    echo "에러: URL이 입력되지 않았습니다."
    exit 1
fi

# fork remote 추가 (이미 있으면 스킵)
if git remote | grep -q "^fork$"; then
    echo "이미 'fork' remote가 존재합니다. 업데이트합니다..."
    git remote set-url fork "$FORK_URL"
else
    echo "'fork' remote를 추가합니다..."
    git remote add fork "$FORK_URL"
fi

echo ""
echo "변경사항을 확인합니다..."
git status

echo ""
read -p "변경사항을 커밋하고 push하시겠습니까? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "취소되었습니다."
    exit 0
fi

# 변경사항 스테이징
echo ""
echo "변경사항을 스테이징합니다..."
git add env.yaml TEXT_QUERY_GUIDE.md configs/custom/ tools/query_highlight_demo.py install_local_packages.sh 2>/dev/null || true

# 커밋 메시지 입력
read -p "커밋 메시지를 입력하세요 (기본: 'Add text query highlight demo'): " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"Add text query highlight demo"}

echo ""
echo "커밋합니다..."
git commit -m "$COMMIT_MSG"

# 브랜치 확인
CURRENT_BRANCH=$(git branch --show-current)
echo ""
echo "현재 브랜치: $CURRENT_BRANCH"

# fork로 push
echo ""
echo "fork repository로 push합니다..."
git push fork "$CURRENT_BRANCH"

echo ""
echo "완료! 변경사항이 fork repository로 push되었습니다."
echo ""
echo "확인: git remote -v"

