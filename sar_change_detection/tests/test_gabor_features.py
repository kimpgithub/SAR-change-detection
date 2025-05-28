"""
Gabor 웨이블릿 특징 추출 모듈 단위 테스트
"""

import unittest
import numpy as np
from src.gabor_features import (
    create_gabor_kernel,
    compute_gabor_magnitude,
    extract_maximum_response,
    extract_gabor_features
)

class TestGaborFeatures(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        # 테스트 이미지 생성
        self.test_image = np.random.rand(32, 32) + 1
        
        # Gabor 파라미터
        self.U = 8  # 방향 수
        self.V = 5  # 스케일 수
        self.kmax = 2 * np.pi
        self.f = np.sqrt(2)
        self.sigma = 2.8 * np.pi
    
    def test_gabor_kernel_properties(self):
        """Gabor 커널 속성 테스트"""
        # 커널 생성
        kernel = create_gabor_kernel(0, 0)
        
        # 커널 크기 확인
        self.assertEqual(kernel.shape[0], kernel.shape[1])
        self.assertTrue(kernel.shape[0] % 2 == 1)  # 홀수 크기
        
        # 복소수 타입 확인
        self.assertTrue(np.iscomplexobj(kernel))
        
        # 대칭성 확인
        self.assertTrue(np.allclose(kernel, kernel.T))
    
    def test_gabor_magnitude(self):
        """Gabor 응답 크기 계산 테스트"""
        # 복소수 응답 생성
        complex_response = np.array([[1+1j, 2+3j], [3+4j, 5+12j]])
        
        # 크기 계산
        magnitude = compute_gabor_magnitude(complex_response)
        
        # 예상값 계산
        expected = np.array([[np.sqrt(2), np.sqrt(13)], [5, 13]])
        
        # 비교
        self.assertTrue(np.allclose(magnitude, expected))
    
    def test_maximum_response(self):
        """최대 응답 추출 테스트"""
        # 테스트 데이터 생성
        responses = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ])
        
        # 최대 응답 계산
        max_response = extract_maximum_response(responses, axis=0)
        
        # 예상값
        expected = np.array([[9, 10], [11, 12]])
        
        # 비교
        self.assertTrue(np.allclose(max_response, expected))
    
    def test_gabor_feature_extraction(self):
        """Gabor 특징 추출 테스트"""
        # 특징 추출
        features, all_responses = extract_gabor_features(
            self.test_image,
            U=self.U,
            V=self.V,
            return_responses=True
        )
        
        # 크기 확인
        self.assertEqual(features.shape, (self.V, *self.test_image.shape))
        self.assertEqual(all_responses.shape, (self.U, self.V, *self.test_image.shape))
        
        # 값 범위 확인
        self.assertTrue(np.all(features >= 0))  # 음수 없음
        self.assertTrue(np.all(np.isfinite(features)))  # 유한한 값
    
    def test_different_scales(self):
        """다른 스케일에서의 특징 추출 테스트"""
        # 다른 스케일 수로 특징 추출
        V = 3
        features = extract_gabor_features(
            self.test_image,
            U=self.U,
            V=V
        )
        
        # 크기 확인
        self.assertEqual(features.shape, (V, *self.test_image.shape))
    
    def test_different_directions(self):
        """다른 방향 수에서의 특징 추출 테스트"""
        # 다른 방향 수로 특징 추출
        U = 4
        features = extract_gabor_features(
            self.test_image,
            U=U,
            V=self.V
        )
        
        # 크기 확인
        self.assertEqual(features.shape, (self.V, *self.test_image.shape))
    
    def test_invalid_input(self):
        """잘못된 입력 테스트"""
        # 잘못된 타입
        with self.assertRaises(ValueError):
            extract_gabor_features("invalid")
        
        # 잘못된 크기
        with self.assertRaises(ValueError):
            extract_gabor_features(np.array([1, 2, 3]))
    
    def test_kernel_parameters(self):
        """커널 파라미터 테스트"""
        # 다른 파라미터로 커널 생성
        kernel1 = create_gabor_kernel(0, 0, kmax=4*np.pi)
        kernel2 = create_gabor_kernel(0, 0, kmax=2*np.pi)
        
        # 크기 비교
        self.assertFalse(np.allclose(kernel1, kernel2))
        
        # 다른 방향
        kernel3 = create_gabor_kernel(1, 0)
        kernel4 = create_gabor_kernel(0, 0)
        self.assertFalse(np.allclose(kernel3, kernel4))
        
        # 다른 스케일
        kernel5 = create_gabor_kernel(0, 1)
        kernel6 = create_gabor_kernel(0, 0)
        self.assertFalse(np.allclose(kernel5, kernel6))

if __name__ == '__main__':
    unittest.main() 