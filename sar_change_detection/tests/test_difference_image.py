"""
DifferenceImage 모듈 단위 테스트
"""

import unittest
import numpy as np
import cv2
from src.difference_image import DifferenceImage

class TestDifferenceImage(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.diff_image = DifferenceImage()
        
        # 기본 테스트 이미지
        self.img1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.img2 = np.array([[2.0, 3.0], [4.0, 5.0]])
        
        # 동일한 이미지
        self.same_img = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # 0이 포함된 이미지
        self.zero_img = np.array([[0.0, 1.0], [2.0, 3.0]])
        
        # 음수가 포함된 이미지
        self.neg_img = np.array([[-1.0, 2.0], [3.0, -4.0]])
    
    def test_same_images(self):
        """동일한 이미지 입력 테스트"""
        # Log-Ratio
        lr_result = self.diff_image.compute_log_ratio(self.same_img, self.same_img)
        self.assertTrue(np.allclose(lr_result, 0))
        
        # Mean-Ratio
        mr_result = self.diff_image.compute_mean_ratio(self.same_img, self.same_img)
        self.assertTrue(np.allclose(mr_result, 0))
    
    def test_different_sizes(self):
        """크기가 다른 이미지 입력 테스트"""
        diff_size_img = np.array([[1.0, 2.0]])
        
        with self.assertRaises(ValueError):
            self.diff_image.compute_log_ratio(self.img1, diff_size_img)
        
        with self.assertRaises(ValueError):
            self.diff_image.compute_mean_ratio(self.img1, diff_size_img)
    
    def test_zero_values(self):
        """0이 포함된 이미지 테스트"""
        # Log-Ratio
        lr_result = self.diff_image.compute_log_ratio(self.zero_img, self.img2)
        self.assertTrue(np.all(np.isfinite(lr_result)))
        
        # Mean-Ratio
        mr_result = self.diff_image.compute_mean_ratio(self.zero_img, self.img2)
        self.assertTrue(np.all(np.isfinite(mr_result)))
    
    def test_negative_values(self):
        """음수가 포함된 이미지 테스트"""
        # Log-Ratio
        lr_result = self.diff_image.compute_log_ratio(self.neg_img, self.img2)
        self.assertTrue(np.all(np.isfinite(lr_result)))
        
        # Mean-Ratio
        mr_result = self.diff_image.compute_mean_ratio(self.neg_img, self.img2)
        self.assertTrue(np.all(np.isfinite(mr_result)))
    
    def test_log_ratio_formula(self):
        """Log-Ratio 수식 정확성 테스트"""
        lr_result = self.diff_image.compute_log_ratio(self.img1, self.img2)
        expected = np.abs(np.log(self.img1 / self.img2))
        self.assertTrue(np.allclose(lr_result, expected))
    
    def test_mean_ratio_formula(self):
        """Mean-Ratio 수식 정확성 테스트"""
        window_size = 3
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # 지역 평균 계산
        alpha1 = cv2.filter2D(self.img1, -1, kernel)
        alpha2 = cv2.filter2D(self.img2, -1, kernel)
        
        # Mean-Ratio 계산
        mr_result = self.diff_image.compute_mean_ratio(self.img1, self.img2, window_size)
        
        # 수식 검증
        ratio1 = alpha2 / alpha1
        ratio2 = alpha1 / alpha2
        min_ratio = np.minimum(ratio1, ratio2)
        
        mask = alpha2 > alpha1
        expected = np.zeros_like(self.img1)
        expected[mask] = 1 - min_ratio[mask]
        expected[~mask] = min_ratio[~mask] - 1
        expected = np.abs(expected)
        
        self.assertTrue(np.allclose(mr_result, expected, atol=1e-5))
    
    def test_compute_method(self):
        """compute 메서드 테스트"""
        # Log-Ratio
        self.diff_image.method = 'log_ratio'
        lr_result = self.diff_image.compute(self.img1, self.img2)
        expected_lr = self.diff_image.compute_log_ratio(self.img1, self.img2)
        self.assertTrue(np.allclose(lr_result, expected_lr))
        
        # Mean-Ratio
        self.diff_image.method = 'mean_ratio'
        mr_result = self.diff_image.compute(self.img1, self.img2)
        expected_mr = self.diff_image.compute_mean_ratio(self.img1, self.img2)
        self.assertTrue(np.allclose(mr_result, expected_mr))
        
        # 잘못된 방법
        self.diff_image.method = 'invalid'
        with self.assertRaises(ValueError):
            self.diff_image.compute(self.img1, self.img2)

if __name__ == '__main__':
    unittest.main() 