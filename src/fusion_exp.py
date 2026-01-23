import numpy as np
from wildlife_tools.similarity.wildfusion import WildFusion
from wildlife_tools.similarity.calibration import IsotonicCalibration, LogisticCalibration

class ExperimentalWildFusion(WildFusion):
    """
    기존 WildFusion을 확장하여 가중치 조절 및 캘리브레이션 교체가 가능하도록 만든 클래스
    """
    def __init__(self, calibrated_pipelines, priority_pipeline=None, weights=None):
        super().__init__(calibrated_pipelines, priority_pipeline)
        
        # 모델 개수에 맞춰 가중치 설정
        if weights is None:
            self.weights = [1.0 / len(calibrated_pipelines)] * len(calibrated_pipelines)
        else:
            if len(weights) != len(calibrated_pipelines):
                raise ValueError("가중치 개수와 파이프라인 개수가 일치해야 합니다.")
            self.weights = weights

    def fit_calibration(self, dataset0, dataset1, calibration_type='isotonic'):
        """
        exp 1. 캘리브레이션 알고리즘 선택
        """
        for matcher in self.calibrated_pipelines:
            # 실험할 알고리즘에 따라 객체 생성 (추후에 Platt, Binning 등 직접 구현해서 추가 실험)
            # wildlife-tools 라이브러리에는 IsotonicCalibration, LogisticCalibration 제공
            if calibration_type == 'logistic':
                matcher.calibration = LogisticCalibration()
            elif calibration_type == 'isotonic':
                matcher.calibration = IsotonicCalibration()
            else:
                # 라이브러리 외의 커스텀 방식이 필요할 경우 여기에 추가
                raise ValueError(f"Unknown calibration type: {calibration_type}")
            
            # 각 모델 별로 calibration 학습 (MegaDescriptor, ALIKED, EVA02 각각 수행됨)
            matcher.fit_calibration(dataset0, dataset1)

    def __call__(self, dataset0, dataset1, pairs=None, B=None):
        """
        exp 2. (캘리브레이션된 점수 x 가중치) 합산
        """
        # priority_pipeline을 사용하여 후보군(pairs) 선정
        if B is not None:
            pairs = self.get_priority_pairs(dataset0, dataset1, B=B)

        all_scores = []
        # 각 파이프라인(Mega, ALIKED, EVA02)의 점수에 가중치(w)를 곱함
        for matcher, w in zip(self.calibrated_pipelines, self.weights):
            score = matcher(dataset0, dataset1, pairs=pairs)
            all_scores.append(score * w)

        # 가중 합산
        score_combined = np.sum(all_scores, axis=0)
        
        # 결측값 처리
        score_combined = np.where(np.isnan(score_combined), -np.inf, score_combined)
        return score_combined

def build_experimental_fusion(calibration_query, calibration_db, *pipelines, priority_pipeline, weights=None, calib_type='isotonic'):
    """
    실험용 build_fusion 함수 (가중치 및 캘리브레이션 타입 수정 가능)
    """
    fusion = ExperimentalWildFusion(
        calibrated_pipelines=list(pipelines),
        priority_pipeline=priority_pipeline,
        weights=weights
    )

    fusion.fit_calibration(calibration_query, calibration_db, calibration_type=calib_type)
    return fusion