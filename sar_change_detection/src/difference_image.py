"""
차이 이미지 생성 모듈

논문 "Synthetic Aperture Radar Image Change Detection Based on Principal Component Analysis and Two-Level Clustering"
의 차분 이미지 생성 방법을 구현합니다.
"""

import numpy as np
import cv2
from typing import Tuple, Union

class DifferenceImage:
    def __init__(self, method: str = 'subtraction'):
        """
        차이 이미지 생성 클래스 초기화
        
        Args:
            method (str): 차이 이미지 생성 방법 ('subtraction', 'log_ratio', 'mean_ratio')
        """
        self.method = method
    
    def _validate_images(self, image1: np.ndarray, image2: np.ndarray) -> None:
        """
        입력 이미지 유효성 검사
        
        Args:
            image1: 첫 번째 SAR 이미지
            image2: 두 번째 SAR 이미지
            
        Raises:
            ValueError: 이미지 크기가 다르거나 타입이 잘못된 경우
        """
        if image1.shape != image2.shape:
            raise ValueError("두 이미지의 크기가 일치해야 합니다.")
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise ValueError("입력은 numpy 배열이어야 합니다.")
        if image1.dtype != np.float32 and image1.dtype != np.float64:
            image1 = image1.astype(np.float32)
        if image2.dtype != np.float32 and image2.dtype != np.float64:
            image2 = image2.astype(np.float32)
    
    def compute_log_ratio(self, image1: np.ndarray, image2: np.ndarray, 
                         epsilon: float = 1e-10) -> np.ndarray:
        """
        Log-Ratio 차분 이미지 계산 (논문 식 1)
        
        Args:
            image1: 시간 t1의 SAR 이미지
            image2: 시간 t2의 SAR 이미지
            epsilon: 0으로 나누기 방지를 위한 작은 값
            
        Returns:
            Log-ratio 차분 이미지
        """
        self._validate_images(image1, image2)
        
        # 0으로 나누기 방지
        image2 = np.maximum(image2, epsilon)
        image1 = np.maximum(image1, epsilon)
        
        # Log-ratio 계산
        log_ratio = np.log(image1 / image2)
        
        return np.abs(log_ratio)  # 절대값 취하기
    
    def compute_mean_ratio(self, image1: np.ndarray, image2: np.ndarray, 
                          window_size: int = 7) -> np.ndarray:
        """
        Mean-Ratio 차분 이미지 계산 (논문 식 2)
        
        Args:
            image1: 시간 t1의 SAR 이미지
            image2: 시간 t2의 SAR 이미지
            window_size: 지역 평균 계산을 위한 윈도우 크기
            
        Returns:
            Mean-ratio 차분 이미지
        """
        self._validate_images(image1, image2)
        
        # 지역 평균 계산
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        alpha1 = cv2.filter2D(image1, -1, kernel)
        alpha2 = cv2.filter2D(image2, -1, kernel)
        
        # 0으로 나누기 방지
        alpha1 = np.maximum(alpha1, 1e-10)
        alpha2 = np.maximum(alpha2, 1e-10)
        
        # Mean-ratio 계산
        ratio1 = alpha2 / alpha1
        ratio2 = alpha1 / alpha2
        min_ratio = np.minimum(ratio1, ratio2)
        
        # 조건에 따른 계산
        mask = alpha2 > alpha1
        result = np.zeros_like(image1)
        result[mask] = 1 - min_ratio[mask]
        result[~mask] = min_ratio[~mask] - 1
        
        return np.abs(result)  # 절대값 취하기
    
    def compute_difference_images(self, image1: np.ndarray, image2: np.ndarray, 
                                window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        LR과 MR 차분 이미지를 동시에 계산하는 편의 함수
        
        Args:
            image1: 시간 t1의 SAR 이미지
            image2: 시간 t2의 SAR 이미지
            window_size: 지역 평균 계산을 위한 윈도우 크기
            
        Returns:
            (lr_image, mr_image) 튜플
        """
        lr_image = self.compute_log_ratio(image1, image2)
        mr_image = self.compute_mean_ratio(image1, image2, window_size)
        return lr_image, mr_image
    
    def compute(self, image1: np.ndarray, image2: np.ndarray, 
                window_size: int = 7) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        선택된 방법으로 차이 이미지 계산
        
        Args:
            image1: 첫 번째 SAR 이미지
            image2: 두 번째 SAR 이미지
            window_size: 지역 평균 계산을 위한 윈도우 크기
            
        Returns:
            차이 이미지 또는 (lr_image, mr_image) 튜플
        """
        if self.method == 'log_ratio':
            return self.compute_log_ratio(image1, image2)
        elif self.method == 'mean_ratio':
            return self.compute_mean_ratio(image1, image2, window_size)
        elif self.method == 'both':
            return self.compute_difference_images(image1, image2, window_size)
        else:
            raise ValueError(f"지원하지 않는 방법입니다: {self.method}") 