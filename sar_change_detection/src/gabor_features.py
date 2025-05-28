"""
Gabor 웨이블릿 특징 추출 모듈

논문 "Synthetic Aperture Radar Image Change Detection Based on Principal Component Analysis and Two-Level Clustering"
의 Gabor 웨이블릿 특징 추출 방법을 구현합니다.
"""

import numpy as np
from scipy import ndimage
import cv2
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def create_gabor_kernel(k: float, theta: float, sigma: float,
                       kernel_size: int = 39) -> np.ndarray:
    """
    Gabor 웨이블릿 커널 생성 (논문 식 13)
    
    Args:
        k: 주파수
        theta: 방향 각도
        sigma: 가우시안 표준편차
        kernel_size: 커널 크기 (홀수)
    
    Returns:
        실수 Gabor 커널
    """
    # 커널 크기가 홀수인지 확인
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 좌표 그리드 생성
    x, y = np.meshgrid(np.arange(-(kernel_size//2), kernel_size//2 + 1),
                      np.arange(-(kernel_size//2), kernel_size//2 + 1))
    
    # 회전된 좌표
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gabor 웨이블릿 계산
    gb = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * \
         np.cos(2 * np.pi * k * x_theta)
    
    return gb.astype(np.float32)

def compute_gabor_magnitude(complex_response: np.ndarray) -> np.ndarray:
    """
    복소수 Gabor 응답의 크기 계산 (논문 식 16)
    
    Args:
        complex_response: 복소수 Gabor 응답
        
    Returns:
        응답의 크기
    """
    return np.sqrt(np.real(complex_response) ** 2 + np.imag(complex_response) ** 2)

def extract_maximum_response(responses: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    모든 방향에서 최대 크기 추출 (논문 식 18)
    
    Args:
        responses: 모든 방향의 Gabor 응답
        axis: 최대값을 계산할 축
        
    Returns:
        최대 응답
    """
    return np.max(responses, axis=axis)

def extract_gabor_features(image: np.ndarray,
                          U: int = 8, V: int = 5,
                          kmax: float = 0.5, f: float = 2,
                          sigma: float = 2) -> np.ndarray:
    """
    Gabor 특징 추출 (논문 식 13-18)
    
    Args:
        image: 입력 이미지
        U: 방향 수
        V: 스케일 수
        kmax: 최대 주파수
        f: 주파수 비율
        sigma: 가우시안 표준편차
    
    Returns:
        Gabor 특징 벡터 (H×W×V)
    """
    height, width = image.shape
    features = np.zeros((height, width, V))
    
    # 각 스케일에 대해
    for v in range(V):
        # 주파수 계산
        k = kmax / (f ** v)
        
        # 각 방향에 대해
        for u in range(U):
            # 방향 계산
            theta = u * np.pi / U
            
            # Gabor 커널 생성
            kernel = create_gabor_kernel(k, theta, sigma)
            
            # 컨볼루션 수행
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            
            # 크기 계산
            magnitude = np.abs(response)
            
            # 최대 응답 추출
            if u == 0:
                features[:, :, v] = magnitude
            else:
                features[:, :, v] = np.maximum(features[:, :, v], magnitude)
    
    return features

def visualize_gabor_responses(image: np.ndarray, responses: np.ndarray, 
                             U: int = 8, V: int = 5,
                             save_path: Optional[str] = None):
    """
    Gabor 응답 시각화 (방향별, 스케일별)
    
    Args:
        image: 원본 이미지
        responses: Gabor 응답 (U, V, H, W)
        U: 방향 수
        V: 스케일 수
        save_path: 결과 저장 경로 (선택사항)
    """
    # 원본 이미지 표시
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('원본 이미지')
    plt.axis('off')
    
    # 각 방향과 스케일의 응답 표시
    plt.subplot(1, 2, 2)
    response_grid = np.zeros((U*image.shape[0], V*image.shape[1]))
    
    for mu in range(U):
        for nu in range(V):
            response = responses[mu, nu]
            response = (response - response.min()) / (response.max() - response.min())
            response_grid[mu*image.shape[0]:(mu+1)*image.shape[0],
                         nu*image.shape[1]:(nu+1)*image.shape[1]] = response
    
    plt.imshow(response_grid, cmap='gray')
    plt.title('Gabor 응답 (방향 × 스케일)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # 테스트 데이터 생성
    test_img1 = np.random.rand(100, 100) + 1
    test_img2 = np.random.rand(100, 100) + 1
    
    # 차분 이미지 및 PCA 융합
    from difference_image import DifferenceImage
    from pca_fusion import PCAFusion
    
    diff_image = DifferenceImage()
    lr_img, mr_img = diff_image.compute_difference_images(test_img1, test_img2)
    
    pca_fusion = PCAFusion()
    fused_img = pca_fusion.fuse_images(lr_img, mr_img)
    
    # Gabor 특징 추출 테스트
    gabor_features = extract_gabor_features(fused_img, return_responses=True)
    features, all_responses = gabor_features
    
    print(f"특징 벡터 크기: {features.shape}")
    print(f"모든 응답 크기: {all_responses.shape}")  # (U, V, H, W)
    print(f"특징값 범위: [{features.min():.3f}, {features.max():.3f}]")
    
    # Gabor 커널 시각화
    kernel = create_gabor_kernel(0, 0, 2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(kernel.real, cmap='gray')
    plt.title('Gabor 커널 (실수부)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(kernel.imag, cmap='gray')
    plt.title('Gabor 커널 (허수부)')
    plt.axis('off')
    plt.show()
    
    # 응답 시각화
    visualize_gabor_responses(fused_img, all_responses)

    # 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
    # 또는
    plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕이 설치된 경우 