"""
PCAFusion 모듈 단위 테스트
"""

import unittest
import numpy as np
from src.pca_fusion import PCAFusion

class TestPCAFusion(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.pca_fusion = PCAFusion()
        
        # 테스트용 이미지 생성
        self.lr_image = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        self.mr_image = np.array([
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0]
        ])
    
    def test_validate_images(self):
        """이미지 유효성 검사 테스트"""
        # 정상 케이스
        self.pca_fusion._validate_images(self.lr_image, self.mr_image)
        
        # 크기가 다른 이미지
        with self.assertRaises(ValueError):
            self.pca_fusion._validate_images(self.lr_image, self.mr_image[:2, :])
        
        # 잘못된 타입
        with self.assertRaises(ValueError):
            self.pca_fusion._validate_images([1, 2, 3], self.mr_image)
    
    def test_flatten_images(self):
        """이미지 벡터화 테스트"""
        lr_flat, mr_flat = self.pca_fusion._flatten_images(self.lr_image, self.mr_image)
        
        self.assertEqual(lr_flat.shape, (9,))
        self.assertEqual(mr_flat.shape, (9,))
        self.assertTrue(np.array_equal(lr_flat, self.lr_image.flatten()))
        self.assertTrue(np.array_equal(mr_flat, self.mr_image.flatten()))
    
    def test_compute_covariance_matrix(self):
        """공분산 행렬 계산 테스트"""
        lr_flat, mr_flat = self.pca_fusion._flatten_images(self.lr_image, self.mr_image)
        cov_matrix = self.pca_fusion.compute_covariance_matrix(lr_flat, mr_flat)
        
        self.assertEqual(cov_matrix.shape, (2, 2))
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))  # 대칭 행렬 확인
        self.assertTrue(np.all(np.diag(cov_matrix) >= 0))  # 분산은 항상 양수
    
    def test_compute_fusion_weights(self):
        """융합 가중치 계산 테스트"""
        # 테스트용 고유값과 고유벡터
        eigenvalues = np.array([2.0, 1.0])
        eigenvectors = np.array([[0.7071, -0.7071], [0.7071, 0.7071]])
        
        m1, m2 = self.pca_fusion.compute_fusion_weights(eigenvalues, eigenvectors)
        
        self.assertIsInstance(m1, float)
        self.assertIsInstance(m2, float)
        self.assertAlmostEqual(m1 + m2, 1.0)  # 가중치 합이 1
    
    def test_fuse_images(self):
        """이미지 융합 테스트"""
        # 기본 융합
        fused_image = self.pca_fusion.fuse_images(self.lr_image, self.mr_image)
        self.assertEqual(fused_image.shape, self.lr_image.shape)
        
        # 가중치 반환
        fused_image, weights = self.pca_fusion.fuse_images(
            self.lr_image, self.mr_image, return_weights=True
        )
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights), 1.0)
        
        # PCA 성분 반환
        fused_image, weights, components = self.pca_fusion.fuse_images(
            self.lr_image, self.mr_image, return_weights=True, return_components=True
        )
        self.assertEqual(len(components), 2)
    
    def test_edge_cases(self):
        """경계 케이스 테스트"""
        # 0으로만 이루어진 이미지
        zero_image = np.zeros_like(self.lr_image)
        fused_image = self.pca_fusion.fuse_images(zero_image, zero_image)
        self.assertTrue(np.allclose(fused_image, 0))
        
        # 동일한 이미지
        fused_image = self.pca_fusion.fuse_images(self.lr_image, self.lr_image)
        self.assertTrue(np.allclose(fused_image, self.lr_image))
        
        # 매우 큰 값
        large_image = self.lr_image * 1e10
        fused_image = self.pca_fusion.fuse_images(large_image, self.mr_image)
        self.assertTrue(np.all(np.isfinite(fused_image)))
    
    def test_visualization(self):
        """시각화 테스트"""
        fused_image = self.pca_fusion.fuse_images(self.lr_image, self.mr_image)
        # 시각화 함수가 예외를 발생시키지 않는지 확인
        self.pca_fusion.visualize_results(
            self.lr_image, self.mr_image, fused_image, weights=(0.5, 0.5)
        )

if __name__ == '__main__':
    unittest.main() 