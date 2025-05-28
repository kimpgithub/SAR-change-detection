"""
SAR 변화 탐지 성능 평가 모듈

논문 "Synthetic Aperture Radar Image Change Detection Based on Principal Component Analysis and Two-Level Clustering"
의 평가 메트릭들을 구현합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    혼동 행렬 계산
    
    Args:
        y_true: 실제 변화 맵 (ground truth)
        y_pred: 예측 변화 맵
    
    Returns:
        {'TP': int, 'TN': int, 'FP': int, 'FN': int}
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def compute_basic_metrics(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    기본 평가 메트릭 계산 (FN, FP, OE, PCC)
    
    Args:
        confusion_matrix: 혼동 행렬
    
    Returns:
        {'FN': float, 'FP': float, 'OE': float, 'PCC': float}
    """
    TP = confusion_matrix['TP']
    TN = confusion_matrix['TN']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    
    total = TP + TN + FP + FN
    
    # 기본 메트릭 계산
    FN_rate = FN / total * 100
    FP_rate = FP / total * 100
    OE = FN + FP
    PCC = (TP + TN) / total * 100
    
    return {
        'FN': FN_rate,
        'FP': FP_rate,
        'OE': OE,
        'PCC': PCC
    }

def compute_kappa_coefficient(confusion_matrix: Dict[str, int]) -> float:
    """
    Kappa 계수 계산 (논문 Table 2 공식)
    
    Args:
        confusion_matrix: 혼동 행렬
    
    Returns:
        Kappa 계수 (%)
    """
    TP = confusion_matrix['TP']
    TN = confusion_matrix['TN']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    
    total = TP + TN + FP + FN
    
    # PCC 계산
    PCC = (TP + TN) / total
    
    # PRE 계산
    PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total * total)
    
    # Kappa 계산
    kappa = (PCC - PRE) / (1 - PRE) * 100
    
    return kappa

def compute_f1_score(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    F1-score, precision, recall 계산
    
    Args:
        confusion_matrix: 혼동 행렬
    
    Returns:
        {'precision': float, 'recall': float, 'F1': float}
    """
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    
    # precision과 recall 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1-score 계산
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'F1': F1 * 100
    }

def compute_roc_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    ROC 곡선 기반 메트릭 (AUC, Ddist) 계산
    
    Args:
        y_true: 실제 변화 맵
        y_scores: 변화 확률/점수
    
    Returns:
        {'AUC': float, 'Ddist': float}
    """
    # ROC 곡선 계산
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # AUC 계산
    roc_auc = auc(fpr, tpr) * 100
    
    # Ddist 계산 (대각선 거리)
    ddist = np.min(np.sqrt((1 - tpr)**2 + fpr**2)) * 100
    
    return {
        'AUC': roc_auc,
        'Ddist': ddist
    }

def evaluate_change_detection(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_scores: np.ndarray = None) -> Dict[str, float]:
    """
    전체 성능 평가 (논문 Table 2의 모든 메트릭)
    
    Args:
        y_true: 실제 변화 맵 (ground truth)
        y_pred: 예측 변화 맵 
        y_scores: 변화 확률/점수 (ROC용, 선택사항)
    
    Returns:
        모든 평가 메트릭을 포함한 딕셔너리
    """
    # 혼동 행렬 계산
    confusion = compute_confusion_matrix(y_true, y_pred)
    
    # 기본 메트릭 계산
    basic_metrics = compute_basic_metrics(confusion)
    
    # Kappa 계수 계산
    kappa = compute_kappa_coefficient(confusion)
    
    # F1-score 계산
    f1_metrics = compute_f1_score(confusion)
    
    # 모든 메트릭 통합
    metrics = {
        **basic_metrics,
        'KC': kappa,
        **f1_metrics
    }
    
    # ROC 메트릭 계산 (점수가 제공된 경우)
    if y_scores is not None:
        roc_metrics = compute_roc_metrics(y_true, y_scores)
        metrics.update(roc_metrics)
    
    return metrics

def compare_methods(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    여러 방법들의 성능 비교 테이블 생성 (논문 Table 5-9 형식)
    
    Args:
        results: {방법명: {메트릭명: 값}} 형식의 결과 딕셔너리
    
    Returns:
        비교 테이블 (DataFrame)
    """
    # DataFrame 생성
    df = pd.DataFrame(results).T
    
    # 메트릭 순서 지정
    metric_order = ['OE', 'PCC', 'KC', 'F1', 'AUC', 'Ddist']
    df = df[metric_order]
    
    # 소수점 자릿수 지정
    df = df.round(2)
    
    return df

def visualize_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray,
                               original_images: Tuple[np.ndarray, np.ndarray],
                               metrics: Dict[str, float],
                               save_path: Optional[str] = None):
    """
    평가 결과 시각화 (논문 Figure 13-17 스타일)
    
    Args:
        y_true: 실제 변화 맵
        y_pred: 예측 변화 맵
        original_images: (이미지1, 이미지2) 튜플
        metrics: 평가 메트릭
        save_path: 결과 저장 경로 (선택사항)
    """
    plt.figure(figsize=(15, 10))
    
    # 1. 원본 이미지들
    plt.subplot(231)
    plt.imshow(original_images[0], cmap='gray')
    plt.title('Time t1 Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(original_images[1], cmap='gray')
    plt.title('Time t2 Image')
    plt.axis('off')
    
    # 2. Ground truth와 예측 결과
    plt.subplot(233)
    plt.imshow(y_true, cmap='hot')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(y_pred, cmap='hot')
    plt.title('Prediction Result')
    plt.axis('off')
    
    # 3. 오분류 영역 시각화
    error_map = np.zeros_like(y_true)
    error_map[(y_true == 1) & (y_pred == 0)] = 1  # FN
    error_map[(y_true == 0) & (y_pred == 1)] = 2  # FP
    
    plt.subplot(235)
    plt.imshow(error_map, cmap='tab10')
    plt.title('Misclassification (Red: FN, Green: FP)')
    plt.axis('off')
    
    # 4. 메트릭 표시
    plt.subplot(236)
    plt.axis('off')
    metric_text = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
    plt.text(0.1, 0.5, metric_text, fontsize=10, va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                   method_name: str = "Proposed",
                   save_path: Optional[str] = None) -> Tuple[float, float]:
    """
    ROC 곡선 플롯 및 AUC, Ddist 계산 (논문 Figure 11 스타일)
    
    Args:
        y_true: 실제 변화 맵
        y_scores: 변화 확률/점수
        method_name: 방법 이름
        save_path: 결과 저장 경로 (선택사항)
    
    Returns:
        (AUC, Ddist) 튜플
    """
    # ROC 곡선 계산
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Ddist 계산
    ddist = np.min(np.sqrt((1 - tpr)**2 + fpr**2))
    
    # ROC 곡선 플롯
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'{method_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    
    return roc_auc, ddist

if __name__ == "__main__":
    # 간단한 테스트
    np.random.seed(42)
    
    # 테스트 데이터 생성
    y_true = np.random.randint(0, 2, (100, 100))
    y_pred = np.random.randint(0, 2, (100, 100))
    y_scores = np.random.rand(100, 100)
    
    # 성능 평가
    metrics = evaluate_change_detection(y_true, y_pred, y_scores)
    print("\n=== 평가 메트릭 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # ROC 곡선 플롯
    plot_roc_curve(y_true, y_scores) 