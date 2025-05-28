"""
시각화 유틸리티 모듈
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_results(original_image, difference_image, change_map, output_path=None):
    """
    변화 탐지 결과 시각화
    
    Args:
        original_image (np.ndarray): 원본 이미지
        difference_image (np.ndarray): 차이 이미지
        change_map (np.ndarray): 변화 맵
        output_path (str, optional): 저장할 경로
    """
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title('원본 이미지')
    plt.axis('off')
    
    # 차이 이미지
    plt.subplot(132)
    plt.imshow(difference_image, cmap='jet')
    plt.title('차이 이미지')
    plt.axis('off')
    
    # 변화 맵
    plt.subplot(133)
    plt.imshow(change_map, cmap='hot')
    plt.title('변화 맵')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    plt.close() 