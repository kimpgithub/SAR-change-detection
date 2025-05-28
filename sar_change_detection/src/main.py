"""
SAR 변화 탐지 메인 모듈

전체 파이프라인을 통합하고 실행하는 모듈입니다.
"""

import os
import yaml
import argparse
import numpy as np
from typing import Dict, Tuple, Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

from .difference_image import DifferenceImage
from .pca_fusion import PCAFusion
from .gabor_features import extract_gabor_features
from .two_level_clustering import two_level_clustering
from .evaluation import (
    evaluate_change_detection,
    visualize_evaluation_results,
    plot_roc_curve,
    compare_methods
)

class SARChangeDetector:
    """
    SAR 이미지 변화 탐지 전체 파이프라인 클래스
    """
    
    def __init__(self, config: Dict = None):
        """
        설정 초기화
        
        Args:
            config: 설정 딕셔너리 (기본값: config/config.yaml)
        """
        if config is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        
        # 모듈 초기화
        self.diff_image = DifferenceImage()
        self.pca_fusion = PCAFusion()
        
        self.gabor_params = config['gabor_features']
        self.clustering_params = config['clustering']
    
    def load_images(self, image1_path: str, image2_path: str, 
                   ground_truth_path: str = None) -> Tuple:
        """
        SAR 이미지 및 ground truth 로드
        
        Args:
            image1_path: 첫 번째 SAR 이미지 경로
            image2_path: 두 번째 SAR 이미지 경로
            ground_truth_path: ground truth 이미지 경로 (선택사항)
        
        Returns:
            (image1, image2, ground_truth) 튜플
        """
        # 이미지 로드 (numpy 배열로 변환)
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        # ground truth 로드 (제공된 경우)
        ground_truth = None
        if ground_truth_path:
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            # ground truth를 이진화 (0 또는 1)
            ground_truth = (ground_truth > 127).astype(np.uint8)
        
        return image1, image2, ground_truth
    
    def preprocess_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple:
        """
        이미지 전처리 (크기 조정, 정규화 등)
        
        Args:
            image1: 첫 번째 SAR 이미지
            image2: 두 번째 SAR 이미지
        
        Returns:
            (image1_processed, image2_processed) 튜플
        """
        # 이미지 크기 확인
        if image1.shape != image2.shape:
            raise ValueError("이미지 크기가 일치하지 않습니다.")
        
        # 정규화 (0-1 범위로)
        image1 = (image1 - image1.min()) / (image1.max() - image1.min())
        image2 = (image2 - image2.min()) / (image2.max() - image2.min())
        
        return image1, image2
    
    def detect_changes(self, image1: np.ndarray, image2: np.ndarray, 
                      return_intermediate: bool = False) -> Union[np.ndarray, Dict]:
        """
        전체 변화 탐지 파이프라인 실행
        
        Steps:
        1. 차분 이미지 계산 (LR, MR)
        2. PCA 융합
        3. Gabor 특징 추출  
        4. 2단계 클러스터링
        
        Args:
            image1: 첫 번째 SAR 이미지
            image2: 두 번째 SAR 이미지
            return_intermediate: 중간 결과 반환 여부
        
        Returns:
            변화 맵 또는 모든 중간 결과
        """
        # 1. 차분 이미지 계산
        lr_img, mr_img = self.diff_image.compute_difference_images(image1, image2)
        
        # 2. PCA 융합
        fused_img = self.pca_fusion.fuse_images(lr_img, mr_img)
        
        # 3. Gabor 특징 추출
        gabor_features = extract_gabor_features(
            fused_img,
            U=self.gabor_params['U'],
            V=self.gabor_params['V'],
            kmax=self.gabor_params['kmax'],
            f=self.gabor_params['f'],
            sigma=self.gabor_params['sigma']
        )
        
        # 4. 2단계 클러스터링
        change_map, intermediate = two_level_clustering(
            gabor_features,
            fused_img,
            c=self.clustering_params['c'],
            m=self.clustering_params['m'],
            max_iter=self.clustering_params['max_iter'],
            tolerance=self.clustering_params['tolerance'],
            return_intermediate=True
        )
        
        if return_intermediate:
            return {
                'change_map': change_map,
                'lr_img': lr_img,
                'mr_img': mr_img,
                'fused_img': fused_img,
                'gabor_features': gabor_features,
                'clustering_results': intermediate
            }
        
        return change_map
    
    def evaluate_performance(self, predicted: np.ndarray, ground_truth: np.ndarray,
                           change_scores: np.ndarray = None) -> Dict:
        """
        성능 평가 실행
        
        Args:
            predicted: 예측 변화 맵
            ground_truth: 실제 변화 맵
            change_scores: 변화 확률/점수 (선택사항)
        
        Returns:
            평가 메트릭 딕셔너리
        """
        return evaluate_change_detection(ground_truth, predicted, change_scores)
    
    def run_full_pipeline(self, image1_path: str, image2_path: str,
                         ground_truth_path: str = None,
                         save_results: bool = True,
                         output_dir: str = "results") -> Dict:
        """
        전체 파이프라인 실행 및 결과 저장
        
        Args:
            image1_path: 첫 번째 SAR 이미지 경로
            image2_path: 두 번째 SAR 이미지 경로
            ground_truth_path: ground truth 이미지 경로 (선택사항)
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리
        
        Returns:
            결과 딕셔너리
        """
        # 결과 저장 디렉토리 생성
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(output_dir) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 로드
        image1, image2, ground_truth = self.load_images(
            image1_path, image2_path, ground_truth_path
        )
        
        # 이미지 전처리
        image1, image2 = self.preprocess_images(image1, image2)
        
        # 변화 탐지
        results = self.detect_changes(image1, image2, return_intermediate=True)
        change_map = results['change_map']
        
        # 성능 평가 (ground truth가 있는 경우)
        metrics = None
        if ground_truth is not None:
            metrics = self.evaluate_performance(change_map, ground_truth)
            
            # 결과 시각화
            if save_results:
                # 평가 결과 시각화
                visualize_evaluation_results(
                    ground_truth, change_map, (image1, image2), metrics,
                    save_path=output_dir / 'evaluation_results.png'
                )
                
                # ROC 곡선 플롯 (변화 점수가 있는 경우)
                if 'change_scores' in results:
                    plot_roc_curve(
                        ground_truth, results['change_scores'],
                        save_path=output_dir / 'roc_curve.png'
                    )
        
        # 중간 결과 저장
        if save_results:
            np.save(output_dir / 'change_map.npy', change_map)
            np.save(output_dir / 'fused_img.npy', results['fused_img'])
            np.save(output_dir / 'gabor_features.npy', results['gabor_features'])
            
            # 메트릭 저장
            if metrics:
                with open(output_dir / 'metrics.txt', 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value:.2f}\n")
        
        return {
            'change_map': change_map,
            'metrics': metrics,
            'intermediate_results': results
        }

def main():
    """
    메인 실행 함수 - 명령행 인터페이스
    """
    parser = argparse.ArgumentParser(description='SAR Change Detection')
    parser.add_argument('--image1', required=True, help='First SAR image path')
    parser.add_argument('--image2', required=True, help='Second SAR image path') 
    parser.add_argument('--ground_truth', help='Ground truth image path')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # 전체 파이프라인 실행
    detector = SARChangeDetector(config_path=args.config)
    results = detector.run_full_pipeline(
        args.image1, args.image2, args.ground_truth,
        save_results=True, output_dir=args.output
    )
    
    print("=== SAR Change Detection Results ===")
    if results['metrics']:
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.2f}")

def test_paper_datasets():
    """
    논문의 5개 데이터셋으로 성능 검증
    - Village of Feltwell
    - Ottawa  
    - San Francisco
    - Yellow River
    - Sulzberger Ice Shelf
    """
    
    datasets = {
        'Village of Feltwell': {
            'image1': 'data/raw/feltwell_t1.tif',
            'image2': 'data/raw/feltwell_t2.tif', 
            'ground_truth': 'data/raw/feltwell_gt.tif',
            'expected_metrics': {  # 논문 Table 5 값들
                'OE': 360, 'PCC': 99.77, 'KC': 95.54, 'F1': 95.66
            }
        },
        'Ottawa': {
            'image1': 'data/raw/ottawa_t1.tif',
            'image2': 'data/raw/ottawa_t2.tif',
            'ground_truth': 'data/raw/ottawa_gt.tif',
            'expected_metrics': {
                'OE': 280, 'PCC': 99.72, 'KC': 94.88, 'F1': 94.95
            }
        },
        'San Francisco': {
            'image1': 'data/raw/sf_t1.tif',
            'image2': 'data/raw/sf_t2.tif',
            'ground_truth': 'data/raw/sf_gt.tif',
            'expected_metrics': {
                'OE': 320, 'PCC': 99.68, 'KC': 94.52, 'F1': 94.60
            }
        },
        'Yellow River': {
            'image1': 'data/raw/yellow_river_t1.tif',
            'image2': 'data/raw/yellow_river_t2.tif',
            'ground_truth': 'data/raw/yellow_river_gt.tif',
            'expected_metrics': {
                'OE': 340, 'PCC': 99.66, 'KC': 94.32, 'F1': 94.40
            }
        },
        'Sulzberger Ice Shelf': {
            'image1': 'data/raw/sulzberger_t1.tif',
            'image2': 'data/raw/sulzberger_t2.tif',
            'ground_truth': 'data/raw/sulzberger_gt.tif',
            'expected_metrics': {
                'OE': 300, 'PCC': 99.70, 'KC': 94.80, 'F1': 94.85
            }
        }
    }
    
    detector = SARChangeDetector()
    all_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n=== Testing {dataset_name} ===")
        
        results = detector.run_full_pipeline(
            dataset_info['image1'],
            dataset_info['image2'], 
            dataset_info['ground_truth']
        )
        
        all_results[dataset_name] = results['metrics']
        
        # 논문 결과와 비교
        expected = dataset_info['expected_metrics']
        actual = results['metrics']
        
        print("논문 vs 구현 결과 비교:")
        for metric in expected:
            exp_val = expected[metric]
            act_val = actual[metric]
            diff = abs(exp_val - act_val)
            print(f"{metric}: 논문={exp_val}, 구현={act_val:.2f}, 차이={diff:.2f}")
    
    # 전체 결과 테이블 생성 (논문 Table 5-9 형식)
    comparison_table = compare_methods(all_results)
    print("\n=== Overall Results Comparison ===")
    print(comparison_table)
    
    return all_results

def test_ottawa_dataset():
    """
    Ottawa 데이터셋으로 성능 검증
    """
    print("\n=== Ottawa 데이터셋 테스트 ===")
    
    # 데이터 경로 설정
    data_dir = Path(__file__).parent.parent.parent / 'Ottawa_data'
    image1_path = data_dir / 'ottawa_1.bmp'
    image2_path = data_dir / 'ottawa_2.bmp'
    ground_truth_path = data_dir / 'ottawa_gt.bmp'
    
    # 전체 파이프라인 실행
    detector = SARChangeDetector()
    results = detector.run_full_pipeline(
        str(image1_path),
        str(image2_path),
        str(ground_truth_path),
        save_results=True,
        output_dir='results/ottawa_test'
    )
    
    # 결과 출력
    print("\n=== Ottawa 데이터셋 평가 결과 ===")
    if results['metrics']:
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.2f}")
    
    return results

if __name__ == "__main__":
    # Ottawa 데이터셋 테스트 실행
    ottawa_results = test_ottawa_dataset() 