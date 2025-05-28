"""
2단계 클러스터링 모듈의 단위 테스트
"""

import unittest
import numpy as np
from src.two_level_clustering import (
    initialize_membership_matrix,
    check_convergence,
    fcm_clustering,
    assign_initial_clusters,
    second_level_clustering,
    two_level_clustering
)

class TestTwoLevelClustering(unittest.TestCase):
    def setUp(self):
        """테스트 데이터 초기화"""
        # 테스트용 특징 벡터 생성
        self.n_samples = 100
        self.n_features = 5
        self.features = np.random.rand(self.n_samples, self.n_features)
        
        # 테스트용 차분 이미지 생성
        self.difference_image = np.random.rand(10, 10)
        
        # 테스트용 멤버십 행렬 생성
        self.membership = np.random.rand(3, self.n_samples)
        self.membership = self.membership / np.sum(self.membership, axis=0, keepdims=True)
    
    def test_initialize_membership_matrix(self):
        """멤버십 행렬 초기화 테스트"""
        c = 3
        u = initialize_membership_matrix(self.n_samples, c)
        
        # 크기 확인
        self.assertEqual(u.shape, (c, self.n_samples))
        
        # 제약조건 확인
        self.assertTrue(np.all(u >= 0))  # 음수 없음
        self.assertTrue(np.all(u <= 1))  # 1 이하
        self.assertTrue(np.allclose(np.sum(u, axis=0), 1))  # 각 열의 합이 1
    
    def test_check_convergence(self):
        """수렴 체크 테스트"""
        u_old = np.random.rand(3, 10)
        u_new = u_old + 1e-5  # 작은 변화
        u_new_large = u_old + 1e-3  # 큰 변화
        
        # 작은 변화는 수렴으로 판단
        self.assertTrue(check_convergence(u_old, u_new, 1e-4))
        # 큰 변화는 미수렴으로 판단
        self.assertFalse(check_convergence(u_old, u_new_large, 1e-4))
    
    def test_fcm_clustering(self):
        """FCM 클러스터링 테스트"""
        centers, membership, labels = fcm_clustering(
            self.features, c=3, m=2.0, max_iter=100, tolerance=1e-4
        )
        
        # 결과 크기 확인
        self.assertEqual(centers.shape, (3, self.n_features))
        self.assertEqual(membership.shape, (3, self.n_samples))
        self.assertEqual(len(labels), self.n_samples)
        
        # 제약조건 확인
        self.assertTrue(np.all(membership >= 0))
        self.assertTrue(np.all(membership <= 1))
        self.assertTrue(np.allclose(np.sum(membership, axis=0), 1))
    
    def test_assign_initial_clusters(self):
        """초기 클러스터 할당 테스트"""
        clusters = assign_initial_clusters(
            self.features, self.membership, self.difference_image.ravel()
        )
        
        # 클러스터 키 확인
        self.assertIn('changed', clusters)
        self.assertIn('unchanged', clusters)
        self.assertIn('intermediate', clusters)
        
        # 클러스터 크기 확인
        total_pixels = len(self.difference_image.ravel())
        self.assertEqual(
            len(clusters['changed']) + len(clusters['unchanged']) + len(clusters['intermediate']),
            total_pixels
        )
    
    def test_second_level_clustering(self):
        """2단계 클러스터링 테스트"""
        # 초기 클러스터 생성
        clusters = assign_initial_clusters(
            self.features, self.membership, self.difference_image.ravel()
        )
        
        # 2단계 클러스터링 수행
        change_map = second_level_clustering(
            self.features, clusters, self.membership
        )
        
        # 결과 크기 확인
        self.assertEqual(len(change_map), self.n_samples)
        
        # 값 범위 확인
        self.assertTrue(np.all(np.isin(change_map, [0, 1])))
    
    def test_two_level_clustering(self):
        """전체 2단계 클러스터링 파이프라인 테스트"""
        # 3D 특징 벡터 생성
        features_3d = np.random.rand(10, 10, 5)
        
        # 클러스터링 수행
        change_map, intermediate = two_level_clustering(
            features_3d, self.difference_image, return_intermediate=True
        )
        
        # 결과 크기 확인
        self.assertEqual(change_map.shape, self.difference_image.shape)
        
        # 값 범위 확인
        self.assertTrue(np.all(np.isin(change_map, [0, 1])))
        
        # 중간 결과 확인
        self.assertIn('centers', intermediate)
        self.assertIn('membership', intermediate)
        self.assertIn('labels', intermediate)
        self.assertIn('initial_clusters', intermediate)
    
    def test_edge_cases(self):
        """경계 케이스 테스트"""
        # 1. 매우 작은 이미지
        small_features = np.random.rand(2, 2, 5)
        small_diff = np.random.rand(2, 2)
        change_map = two_level_clustering(small_features, small_diff)
        self.assertEqual(change_map.shape, (2, 2))
        
        # 2. 동일한 특징값
        same_features = np.ones((10, 10, 5))
        change_map = two_level_clustering(same_features, self.difference_image)
        self.assertEqual(change_map.shape, (10, 10))
        
        # 3. 매우 큰 특징값
        large_features = np.random.rand(10, 10, 5) * 1e6
        change_map = two_level_clustering(large_features, self.difference_image)
        self.assertEqual(change_map.shape, (10, 10))

if __name__ == '__main__':
    unittest.main() 