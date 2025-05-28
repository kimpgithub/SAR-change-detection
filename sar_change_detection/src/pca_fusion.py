"""
PCA 기반 이미지 융합 모듈

논문 "Synthetic Aperture Radar Image Change Detection Based on Principal Component Analysis and Two-Level Clustering"
의 PCA 융합 방법을 구현합니다.
"""

import numpy as np
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class PCAFusion:
    def __init__(self, n_components: int = 2):
        """
        PCA 융합 클래스 초기화
        
        Args:
            n_components: 사용할 주성분 수
        """
        self.n_components = n_components
    
    def _validate_images(self, image1: np.ndarray, image2: np.ndarray) -> None:
        """
        입력 이미지 유효성 검사
        
        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            
        Raises:
            ValueError: 이미지 크기가 다르거나 타입이 잘못된 경우
        """
        # 타입 검사
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise ValueError("입력은 numpy 배열이어야 합니다.")
        
        # 크기 검사
        if image1.shape != image2.shape:
            raise ValueError("두 이미지의 크기가 일치해야 합니다.")
    
    def _flatten_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지를 벡터로 변환 (논문 식 3-4)
        
        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            
        Returns:
            (image1_flat, image2_flat): 벡터화된 이미지들
        """
        return image1.flatten(), image2.flatten()
    
    def compute_covariance_matrix(self, image1_flat: np.ndarray, image2_flat: np.ndarray) -> np.ndarray:
        """
        두 이미지 간 공분산 행렬 계산 (논문 식 5)
        
        Args:
            image1_flat: 첫 번째 이미지의 벡터
            image2_flat: 두 번째 이미지의 벡터
            
        Returns:
            공분산 행렬
        """
        # 평균 계산 (논문 식 6-7)
        beta1 = np.mean(image1_flat)
        beta2 = np.mean(image2_flat)
        
        # 공분산 계산 (논문 식 5)
        cov_matrix = np.zeros((2, 2))
        cov_matrix[0, 0] = np.mean((image1_flat - beta1) ** 2)
        cov_matrix[1, 1] = np.mean((image2_flat - beta2) ** 2)
        cov_matrix[0, 1] = cov_matrix[1, 0] = np.mean((image1_flat - beta1) * (image2_flat - beta2))
        
        return cov_matrix
    
    def compute_fusion_weights(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[float, float]:
        """
        PCA 기반 융합 가중치 계산 (논문 식 8-11)
        
        Args:
            eigenvalues: 고유값 배열
            eigenvectors: 고유벡터 행렬
            
        Returns:
            (m1, m2): 융합 가중치
        """
        # 고유값과 고유벡터 정렬
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 가중치 계산 (논문 식 8-11)
        if eigenvalues[0] > eigenvalues[1]:
            m1 = eigenvectors[0, 0] / (eigenvectors[0, 0] + eigenvectors[1, 0])
            m2 = eigenvectors[1, 0] / (eigenvectors[0, 0] + eigenvectors[1, 0])
        else:
            m1 = eigenvectors[0, 1] / (eigenvectors[0, 1] + eigenvectors[1, 1])
            m2 = eigenvectors[1, 1] / (eigenvectors[0, 1] + eigenvectors[1, 1])
        
        return m1, m2
    
    def fuse_images(self, lr_image: np.ndarray, mr_image: np.ndarray,
                   return_weights: bool = False,
                   return_components: bool = False) -> Union[np.ndarray, Tuple]:
        """
        PCA를 사용한 LR과 MR 이미지 융합 (논문 식 3-12)
        
        Args:
            lr_image: Log-ratio 차분 이미지
            mr_image: Mean-ratio 차분 이미지
            return_weights: 융합 가중치 반환 여부
            return_components: PCA 성분들 반환 여부
            
        Returns:
            융합된 차분 이미지 (DI_PCA)
            선택적으로 (융합_이미지, 가중치, PCA_성분들) 튜플
        """
        self._validate_images(lr_image, mr_image)
        
        # 이미지 벡터화
        lr_flat, mr_flat = self._flatten_images(lr_image, mr_image)
        
        # 공분산 행렬 계산
        cov_matrix = self.compute_covariance_matrix(lr_flat, mr_flat)
        
        # 고유값 분해
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # 융합 가중치 계산
        m1, m2 = self.compute_fusion_weights(eigenvalues, eigenvectors)
        
        # 이미지 융합 (논문 식 12)
        fused_flat = m1 * lr_flat + m2 * mr_flat
        fused_image = fused_flat.reshape(lr_image.shape)
        
        if return_weights or return_components:
            result = [fused_image]
            if return_weights:
                result.append((m1, m2))
            if return_components:
                result.append((eigenvalues, eigenvectors))
            return tuple(result)
        
        return fused_image
    
    def visualize_results(self, lr_image: np.ndarray, mr_image: np.ndarray,
                         fused_image: np.ndarray, weights: Optional[Tuple[float, float]] = None):
        """
        PCA 융합 결과 시각화
        
        Args:
            lr_image: Log-ratio 차분 이미지
            mr_image: Mean-ratio 차분 이미지
            fused_image: 융합된 차분 이미지
            weights: 융합 가중치 (선택사항)
        """
        plt.figure(figsize=(15, 5))
        
        # Log-ratio 이미지
        plt.subplot(131)
        plt.imshow(lr_image, cmap='jet')
        plt.title('Log-Ratio Image')
        plt.axis('off')
        
        # Mean-ratio 이미지
        plt.subplot(132)
        plt.imshow(mr_image, cmap='jet')
        plt.title('Mean-Ratio Image')
        plt.axis('off')
        
        # 융합된 이미지
        plt.subplot(133)
        plt.imshow(fused_image, cmap='jet')
        title = 'Fused Image'
        if weights:
            title += f'\nWeights: ({weights[0]:.3f}, {weights[1]:.3f})'
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show() 