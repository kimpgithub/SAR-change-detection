"""
이미지 입출력 유틸리티 모듈
"""

import os
import numpy as np
import cv2

def load_image(image_path):
    """
    이미지 파일 로드
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        np.ndarray: 로드된 이미지
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    return image

def save_image(image, output_path):
    """
    이미지 저장
    
    Args:
        image (np.ndarray): 저장할 이미지
        output_path (str): 저장할 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image) 