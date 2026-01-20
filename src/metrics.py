import numpy as np
from scipy.stats import entropy

def calculate_statistical_metrics(predictions, scores):
    # 1. Raw Confidence: 단순 점수 평균, 평균 확신도 (Mean Confidence)
    raw_mean_conf = np.mean(scores)
    
    # 2. Uncertainty: 모델의 불확실성 (낮을수록 세 모델의 의견이 일치함)
    # 0.5 근처 점수가 많으면 이 값이 올라갑니다.
    unc = [entropy([s, 1-s]) if 0 < s < 1 else 0 for s in scores]
    uncertainty = np.mean(unc)
    
    # 3. New Ratio: 전체 중 'new_individual'로 판정한 비율
    new_mask = [1 if p == 'new_individual' else 0 for p in predictions]
    new_ratio = np.mean(new_mask)
    
    # 4. Reliability: 최종 신뢰도 지표 (점수는 높고 불확실성은 낮을수록 큼)
    # 논문에서 "가중치 최적화의 근거"로 제시하기 가장 좋은 지표입니다.
    reliability_score = raw_mean_conf - (0.1 * uncertainty)

    return raw_mean_conf, uncertainty, new_ratio, reliability_score

def calculate_core_metrics(predictions, scores, ground_truth=None):
    """
    핵심 지표 계산: 정답(ground_truth)이 있을 때만 작동
    정답이 없으면 0이나 None을 반환하여 에러 방지
    """
    if ground_truth is None:
        return 0.0, 0.0, 0.0
    
    # [참고] 실제 논문용 지표 계산 로직 (정답 데이터 확보 시 사용)
    # 1. Rank-1: 첫 번째 예측이 정답과 일치하는 비율
    rank1 = np.mean([p == gt for p, gt in zip(predictions, ground_truth)])
    
    # 2. F1-Score (New 개체 판별용)
    # New를 1, Known을 0으로 두고 이진 분류 성능 측정
    from sklearn.metrics import f1_score
    y_true = [1 if gt == 'new_individual' else 0 for gt in ground_truth]
    y_pred = [1 if p == 'new_individual' else 0 for p in predictions]
    f1 = f1_score(y_true, y_pred)
    
    # 3. mAP: 순위 기반 지표 (검색 시스템에서 중요)
    # 이 부분은 전체 정답 리스트(Top-K)가 필요하므로 나중에 추가 확장 가능
    map_score = 0.0 # 임시값
    
    return rank1, map_score, f1