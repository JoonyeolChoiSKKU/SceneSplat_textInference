"""
텍스트 쿼리로 3DGS 장면에서 특정 영역을 하이라이트하는 데모 스크립트

사용 예:
    python tools/query_highlight_demo.py \
        --scene-path /media/stevenlair/Data_exFAT/SceneSplat/custom_scenes/my_scene \
        --query "pen on desk" \
        --checkpoint /media/stevenlair/Data_exFAT/SceneSplat/checkpoints/model_best.pth \
        --output /media/stevenlair/Data_exFAT/SceneSplat/results/pen_highlight
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import open3d as o3d


def encode_query_text(query: str, model_name: str = "google/siglip2-base-patch16-512", device: str = "cuda"):
    """텍스트 쿼리를 임베딩으로 변환"""
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # "this is a {query}" 형태로 프롬프트 추가
    prompt = f"this is a {query}"
    inputs = tokenizer(prompt, padding="max_length", max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
    
    return text_embedding.cpu()


def load_scene_features(feat_path: Path):
    """저장된 특징 파일 로드"""
    if feat_path.suffix == ".pth":
        feat = torch.load(feat_path, weights_only=True)
    elif feat_path.suffix == ".npy":
        feat = torch.from_numpy(np.load(feat_path))
    else:
        raise ValueError(f"Unsupported feature format: {feat_path.suffix}")
    
    # 정규화 확인
    if feat.dim() == 2:
        feat = F.normalize(feat, p=2, dim=1)
    return feat


def compute_highlight_mask(features: torch.Tensor, query_embedding: torch.Tensor, 
                          threshold: float = 0.25, top_k_ratio: float = None):
    """
    특징과 쿼리 임베딩의 유사도를 계산하여 하이라이트 마스크 생성
    
    Args:
        features: [N, D] 특징 텐서
        query_embedding: [1, D] 쿼리 임베딩
        threshold: 유사도 임계값 (0~1)
        top_k_ratio: 상위 k%만 선택 (None이면 threshold 사용)
    
    Returns:
        mask: [N] boolean 마스크
        scores: [N] 유사도 점수
    """
    # Cosine similarity 계산
    scores = (features @ query_embedding.t()).squeeze(1)  # [N]
    
    if top_k_ratio is not None:
        k = int(len(scores) * top_k_ratio)
        _, top_indices = torch.topk(scores, k)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[top_indices] = True
    else:
        mask = scores > threshold
    
    return mask, scores


def visualize_highlight(coord: np.ndarray, color: np.ndarray, mask: np.ndarray, 
                        highlight_color: np.ndarray = np.array([255, 0, 0])):
    """
    Open3D로 하이라이트 결과 시각화
    
    Args:
        coord: [N, 3] 좌표
        color: [N, 3] 원본 색상 (0-255)
        mask: [N] boolean 마스크
        highlight_color: 하이라이트 색상 (RGB, 0-255)
    """
    # 색상 복사
    vis_color = color.copy()
    
    # 하이라이트 영역 색상 변경
    vis_color[mask] = highlight_color
    
    # Point cloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(vis_color / 255.0)
    
    # 시각화
    o3d.visualization.draw_geometries([pcd])


def save_highlight_result(coord: np.ndarray, color: np.ndarray, mask: np.ndarray,
                         scores: np.ndarray, output_dir: Path):
    """하이라이트 결과를 파일로 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 마스크 저장
    np.save(output_dir / "highlight_mask.npy", mask.astype(np.uint8))
    
    # 점수 저장
    np.save(output_dir / "similarity_scores.npy", scores)
    
    # 하이라이트된 색상 저장
    highlight_color = np.array([255, 0, 0], dtype=np.uint8)
    vis_color = color.copy()
    vis_color[mask] = highlight_color
    np.save(output_dir / "highlighted_color.npy", vis_color)
    
    # 좌표 저장 (참고용)
    np.save(output_dir / "coord.npy", coord)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Highlighted points: {mask.sum()} / {len(mask)} ({100 * mask.sum() / len(mask):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="텍스트 쿼리로 3DGS 장면 하이라이트")
    
    parser.add_argument("--scene-path", type=Path, required=True,
                       help="3DGS 장면 폴더 경로 (coord.npy, color.npy 등이 있는 곳)")
    parser.add_argument("--feat-path", type=Path, default=None,
                       help="저장된 특징 파일 경로 (.pth 또는 .npy). None이면 장면 경로에서 자동 탐색")
    parser.add_argument("--query", type=str, required=True,
                       help="검색할 텍스트 쿼리 (예: 'pen', 'red mug', 'chair on floor')")
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="모델 체크포인트 경로 (특징이 없을 때만 필요)")
    parser.add_argument("--output", type=Path, required=True,
                       help="결과 저장 디렉토리")
    
    parser.add_argument("--threshold", type=float, default=0.25,
                       help="유사도 임계값 (0~1)")
    parser.add_argument("--top-k-ratio", type=float, default=None,
                       help="상위 k%만 선택 (예: 0.1 = 상위 10%%, threshold보다 우선)")
    parser.add_argument("--visualize", action="store_true",
                       help="Open3D로 결과 시각화")
    parser.add_argument("--model-name", type=str, default="google/siglip2-base-patch16-512",
                       help="텍스트 인코더 모델 이름")
    
    args = parser.parse_args()
    
    # 1. 장면 데이터 로드
    print(f"Loading scene from {args.scene_path}...")
    coord = np.load(args.scene_path / "coord.npy")
    color = np.load(args.scene_path / "color.npy")
    
    if color.dtype != np.uint8:
        color = (color * 255).astype(np.uint8).clip(0, 255)
    
    print(f"  - Points: {len(coord)}")
    print(f"  - Color shape: {color.shape}")
    
    # 2. 특징 로드 또는 생성
    if args.feat_path is not None:
        feat_path = args.feat_path
    else:
        # 자동 탐색: scene_path의 상위 디렉토리에서 feat 폴더 찾기
        possible_feat_paths = [
            args.scene_path.parent.parent / "feat" / f"{args.scene_path.name}_feat.pth",
            args.scene_path.parent / "feat" / f"{args.scene_path.name}_feat.pth",
            args.scene_path / "feat.pth",
        ]
        feat_path = None
        for p in possible_feat_paths:
            if p.exists():
                feat_path = p
                break
        
        if feat_path is None:
            raise FileNotFoundError(
                f"Feature file not found. Please run inference first or specify --feat-path.\n"
                f"Searched: {possible_feat_paths}"
            )
    
    print(f"Loading features from {feat_path}...")
    features = load_scene_features(feat_path)
    print(f"  - Feature shape: {features.shape}")
    
    # 3. 텍스트 쿼리 임베딩 생성
    print(f"Encoding query: '{args.query}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embedding = encode_query_text(args.query, args.model_name, device)
    print(f"  - Query embedding shape: {query_embedding.shape}")
    
    # 4. 하이라이트 마스크 계산
    print("Computing highlight mask...")
    features_gpu = features.to(device)
    query_embedding_gpu = query_embedding.to(device)
    
    mask, scores = compute_highlight_mask(
        features_gpu, query_embedding_gpu, 
        threshold=args.threshold, 
        top_k_ratio=args.top_k_ratio
    )
    
    mask = mask.cpu().numpy()
    scores = scores.cpu().numpy()
    
    print(f"  - Highlighted points: {mask.sum()} / {len(mask)} ({100 * mask.sum() / len(mask):.2f}%)")
    print(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # 5. 결과 저장
    save_highlight_result(coord, color, mask, scores, args.output)
    
    # 6. 시각화 (옵션)
    if args.visualize:
        print("Visualizing result...")
        visualize_highlight(coord, color, mask)
    
    print("Done!")


if __name__ == "__main__":
    main()

