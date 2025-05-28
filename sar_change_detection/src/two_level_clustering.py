"""
2단계 클러스터링 모듈

논문 "Synthetic Aperture Radar Image Change Detection Based on Principal Component Analysis and Two-Level Clustering"
의 2단계 클러스터링 방법을 구현합니다.
"""

import numpy as np
from typing import Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def initialize_membership_matrix(n_samples: int, c: int, random_state: int = None) -> np.ndarray:
    """
    멤버십 행렬 초기화 (제약조건 만족)
    
    Args:
        n_samples: 샘플 수
        c: 클러스터 수
        random_state: 랜덤 시드
    
    Returns:
        초기화된 멤버십 행렬 (c × n_samples)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 랜덤 초기화
    u = np.random.rand(c, n_samples)
    
    # 제약조건 적용: 각 열의 합이 1
    u = u / np.sum(u, axis=0, keepdims=True)
    
    return u

def check_convergence(u_old: np.ndarray, u_new: np.ndarray, tolerance: float) -> bool:
    """
    FCM 알고리즘의 수렴 여부 확인
    
    Args:
        u_old: 이전 멤버십 행렬
        u_new: 새로운 멤버십 행렬
        tolerance: 수렴 기준값
    
    Returns:
        수렴 여부 (True/False)
    """
    # tolerance를 float로 변환
    tolerance = float(tolerance)
    return np.max(np.abs(u_new - u_old)) < tolerance

def fcm_clustering(features: np.ndarray, 
                   c: int = 3, m: float = 2.0,
                   max_iter: int = 100, 
                   tolerance: float = 1e-4,
                   random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    퍼지 C-평균 클러스터링 (논문 식 19-22)
    
    Args:
        features: Gabor 특징 벡터 (MN × V)
        c: 클러스터 수 
        m: 퍼지 계수
        max_iter: 최대 반복 횟수
        tolerance: 수렴 임계값
        random_state: 랜덤 시드
    
    Returns:
        (centers, membership_matrix, labels)
    """
    # 입력 데이터 준비
    n_samples, n_features = features.shape
    
    # 멤버십 행렬 초기화
    u = initialize_membership_matrix(n_samples, c, random_state)
    
    # 반복 최적화
    for iteration in range(max_iter):
        # 중심점 업데이트 (식 21)
        u_m = u ** m
        centers = np.dot(u_m, features) / np.sum(u_m, axis=1, keepdims=True)
        
        # 거리 계산
        distances = cdist(centers, features)
        
        # 멤버십 업데이트 (식 22)
        u_new = 1 / (distances ** (2/(m-1)))
        u_new = u_new / np.sum(u_new, axis=0, keepdims=True)
        
        # 수렴 체크
        if check_convergence(u, u_new, tolerance):
            break
            
        u = u_new
    
    # 클러스터 레이블 할당
    labels = np.argmax(u, axis=0)
    
    return centers, u, labels

def assign_initial_clusters(features: np.ndarray, membership: np.ndarray,
                           difference_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    1단계 클러스터 할당 (논문 식 23-24)
    
    Args:
        features: Gabor 특징 벡터
        membership: FCM 멤버십 행렬
        difference_image: 차분 이미지
    
    Returns:
        {'changed': indices, 'unchanged': indices, 'intermediate': indices}
    """
    # 클러스터 평균 계산 (식 24)
    cluster_means = np.zeros(3)
    for i in range(3):
        cluster_indices = np.argmax(membership, axis=0) == i
        cluster_means[i] = np.mean(difference_image.flat[cluster_indices])
    
    # 클러스터 할당 (식 23)
    max_cluster = np.argmax(cluster_means)
    min_cluster = np.argmin(cluster_means)
    mid_cluster = 3 - max_cluster - min_cluster
    
    # 클러스터 인덱스 추출
    changed_indices = np.where(np.argmax(membership, axis=0) == max_cluster)[0]
    unchanged_indices = np.where(np.argmax(membership, axis=0) == min_cluster)[0]
    intermediate_indices = np.where(np.argmax(membership, axis=0) == mid_cluster)[0]
    
    return {
        'changed': changed_indices,
        'unchanged': unchanged_indices,
        'intermediate': intermediate_indices
    }

def second_level_clustering(features: np.ndarray, 
                           initial_clusters: Dict[str, np.ndarray],
                           membership: np.ndarray,
                           m: float = 2.0) -> np.ndarray:
    """
    2단계 최근접 이웃 클러스터링 (논문 식 25-27)
    
    Args:
        features: Gabor 특징 벡터
        initial_clusters: 1단계 클러스터 결과
        membership: FCM 멤버십 행렬
        m: 퍼지 계수
    
    Returns:
        최종 변화 맵 (0: 무변화, 1: 변화)
    """
    # 중심점 재계산 (식 25-26)
    u_m = membership ** m
    
    # 변화 클러스터 중심
    changed_features = features[initial_clusters['changed']]  # (n_changed, n_features)
    changed_membership = u_m[:, initial_clusters['changed']]  # (n_clusters, n_changed)
    v_changed = np.zeros(features.shape[1])  # (n_features,)
    for i in range(features.shape[1]):
        v_changed[i] = np.sum(changed_membership * changed_features[:, i:i+1].T) / np.sum(changed_membership)
    
    # 무변화 클러스터 중심
    unchanged_features = features[initial_clusters['unchanged']]  # (n_unchanged, n_features)
    unchanged_membership = u_m[:, initial_clusters['unchanged']]  # (n_clusters, n_unchanged)
    v_unchanged = np.zeros(features.shape[1])  # (n_features,)
    for i in range(features.shape[1]):
        v_unchanged[i] = np.sum(unchanged_membership * unchanged_features[:, i:i+1].T) / np.sum(unchanged_membership)
    
    # 중간 클래스 재분류 (식 27)
    intermediate_features = features[initial_clusters['intermediate']]
    dist_to_changed = np.linalg.norm(intermediate_features - v_changed, axis=1)
    dist_to_unchanged = np.linalg.norm(intermediate_features - v_unchanged, axis=1)
    
    # 최종 변화 맵 생성 (식 28-29)
    change_map = np.zeros(len(features))
    change_map[initial_clusters['changed']] = 1
    change_map[initial_clusters['intermediate'][dist_to_changed <= dist_to_unchanged]] = 1
    
    return change_map

def two_level_clustering(features: np.ndarray, 
                        difference_image: np.ndarray,
                        c: int = 3, m: float = 2.0,
                        max_iter: int = 100,
                        tolerance: float = 1e-4,
                        return_intermediate: bool = False) -> Union[np.ndarray, Tuple]:
    """
    완전한 2단계 클러스터링 파이프라인 (논문 식 19-29)
    
    Args:
        features: Gabor 특징 벡터 (H×W×V를 H*W×V로 reshape)
        difference_image: 차분 이미지 (클러스터 할당용)
        c: 클러스터 수
        m: 퍼지 계수
        max_iter: 최대 반복 횟수
        tolerance: 수렴 임계값
        return_intermediate: 중간 결과 반환 여부
    
    Returns:
        최종 변화 맵 (H×W)
        선택적으로 중간 결과들도 반환
    """
    # 입력 데이터 준비
    original_shape = difference_image.shape
    features_2d = features.reshape(-1, features.shape[-1])
    difference_image_1d = difference_image.ravel()
    
    # 크기 확인 및 조정
    if features_2d.shape[0] != difference_image_1d.shape[0]:
        raise ValueError(f"특징 벡터 크기({features_2d.shape[0]})와 이미지 크기({difference_image_1d.shape[0]})가 일치하지 않습니다.")
    
    # 1단계: FCM 클러스터링
    centers, membership, labels = fcm_clustering(
        features_2d, c=c, m=m, max_iter=max_iter, tolerance=tolerance
    )
    
    # 초기 클러스터 할당
    initial_clusters = assign_initial_clusters(
        features_2d, membership, difference_image_1d
    )
    
    # 2단계: 최근접 이웃 클러스터링
    change_map_1d = second_level_clustering(
        features_2d, initial_clusters, membership, m=m
    )
    
    # 원래 이미지 크기로 복원
    change_map = change_map_1d.reshape(original_shape)
    
    if return_intermediate:
        intermediate_results = {
            'centers': centers,
            'membership': membership,
            'labels': labels,
            'initial_clusters': initial_clusters
        }
        return change_map, intermediate_results
    
    return change_map

def visualize_clustering_results(original_images: Tuple[np.ndarray, np.ndarray],
                               change_map: np.ndarray,
                               intermediate_results: Dict = None,
                               save_path: Optional[str] = None):
    """
    클러스터링 결과 시각화
    
    Args:
        original_images: (이미지1, 이미지2) 튜플
        change_map: 변화 맵
        intermediate_results: 중간 결과 (선택사항)
        save_path: 결과 저장 경로 (선택사항)
    """
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지들
    plt.subplot(131)
    plt.imshow(original_images[0], cmap='gray')
    plt.title('시간 t1 이미지')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(original_images[1], cmap='gray')
    plt.title('시간 t2 이미지')
    plt.axis('off')
    
    # 변화 맵
    plt.subplot(133)
    plt.imshow(change_map, cmap='hot')
    plt.title('변화 맵')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    # 중간 결과가 있는 경우 추가 시각화
    if intermediate_results is not None:
        # 멤버십 행렬 시각화
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            membership_map = intermediate_results['membership'][i].reshape(change_map.shape)
            plt.imshow(membership_map, cmap='hot')
            plt.title(f'클러스터 {i+1} 멤버십')
            plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # 이전 모듈들의 전체 파이프라인 테스트
    from difference_image import DifferenceImage
    from pca_fusion import PCAFusion
    from gabor_features import extract_gabor_features
    
    # 테스트 데이터 생성 (더 현실적인 변화 패턴)
    H, W = 100, 100
    test_img1 = np.random.rand(H, W) + 1
    test_img2 = test_img1.copy()
    # 일부 영역에 변화 추가
    test_img2[30:70, 30:70] *= 2  # 변화 영역
    
    # 전체 파이프라인 실행
    print("1. 차분 이미지 계산...")
    diff_image = DifferenceImage()
    lr_img, mr_img = diff_image.compute_difference_images(test_img1, test_img2)
    
    print("2. PCA 융합...")
    pca_fusion = PCAFusion()
    fused_img = pca_fusion.fuse_images(lr_img, mr_img)
    
    print("3. Gabor 특징 추출...")
    gabor_features = extract_gabor_features(fused_img)  # H×W×V
    
    print("4. 2단계 클러스터링...")
    change_map, intermediate = two_level_clustering(
        gabor_features, fused_img, return_intermediate=True
    )
    
    print(f"변화 맵 크기: {change_map.shape}")
    print(f"변화 픽셀 수: {np.sum(change_map == 1)}")
    print(f"무변화 픽셀 수: {np.sum(change_map == 0)}")
    
    # 결과 시각화
    visualize_clustering_results((test_img1, test_img2), change_map, intermediate)
    
    # 간단한 정확도 확인 (변화 영역이 올바르게 탐지되었는지)
    true_change_region = np.zeros((H, W))
    true_change_region[30:70, 30:70] = 1
    accuracy = np.mean(change_map == true_change_region)
    print(f"간단한 정확도: {accuracy:.3f}") 