from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transforms_aliked, transform_tta_mega
from src.dataset import load_datasets
from src.fusion_exp import build_experimental_fusion  # experimental fusion
from src.matcher import build_megadescriptor, build_aliked, build_eva02, build_dinov3
from src.fusion_head import FusionMLP
from src.utils import set_seed

import timm
import numpy as np
import pandas as pd

from src.metrics import calculate_statistical_metrics, calculate_core_metrics  # 지표 계산 함수
import os

def main():
    set_seed(42)

    # exp 1. 캘리브레이션 알고리즘 선택: 'isotonic' 또는 'logistic'
    CURRENT_CALIB_TYPE = 'isotonic' 
    # exp 2. 모델별 가중치 설정: [ALIKED, MegaDescriptor, EVA02]
    CURRENT_WEIGHTS = [0.34, 0.33, 0.33]
    THRESHOLD = 0.35  # config.py의 THRESHOLD꺼 가져옴

    # 1. Load the full dataset
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT, calibration_size=1000)

    # 2. Load MegaDescriptor model
    model_mega = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)

    # 3. 각 모델 별 Matcher(pipeline) build
    matcher_mega = build_megadescriptor(model=model_mega, transform=transform_tta_mega, device=DEVICE)
    matcher_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)
    matcher_eva = build_eva02(device=DEVICE)  # 수정1: EVA02 matcher 추가

    # 4. Fusion 모델 빌드 (ALIKED + Mega + EVA02)
    fusion = build_experimental_fusion(
        dataset_calib, dataset_calib,
        matcher_aliked, matcher_mega, matcher_eva,   # 수정1
        priority_pipeline=matcher_mega,              # 후보 선정용 메인 파이프라인
        weights=CURRENT_WEIGHTS,
        calib_type=CURRENT_CALIB_TYPE
    )

    # 5. Compute predictions per query group
    predictions_all = []
    image_ids_all = []
    scores_all = []  # 수정2: 점수 저장용

    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_subset = dataset_query.get_subset(dataset_query.metadata["dataset"] == dataset_name)

        # 6. WildFusion 한 번의 호출로 3개 모델이 통합된 점수 산출
        # 수정3: 내부적으로 ALIKED, Mega, EVA02가 각각 계산되고 가중치 합산됨
        combined_sim = fusion(query_subset, dataset_db, B=25)

        # 결과 도출 (Top-1 + threshold): idx / score 계산
        idx_sorted = combined_sim.argsort(axis=1)
        top_idx = idx_sorted[:, -1]
        p_top1 = combined_sim[np.arange(len(query_subset)), top_idx]  # Top-1 점수

        labels = dataset_db.labels_string
        predictions = labels[top_idx].copy()
        predictions[(p_top1 < THRESHOLD)] = "new_individual"  # threshold 미달 시 new_individual

        predictions_all.extend(predictions)
        image_ids_all.extend(query_subset.metadata["image_id"])
        scores_all.extend(p_top1)   # 수정2: top-1 점수 저장
    
    # 6. 통계 지표 계산 및 출력 (수정2)
    # 현재 정답(ground_truth)이 없으므로 core_metrics는 0으로 반환됨
    raw_conf, unc, n_ratio, rel_score = calculate_statistical_metrics(predictions_all, scores_all)
    rank1, m_ap, f1 = calculate_core_metrics(predictions_all, scores_all, ground_truth=None)

    # 가중치 정수화
    w_int = [int(round(w * 100)) for w in CURRENT_WEIGHTS]

    # 실험 지표 저장용 데이터 생성
    metrics_data = {
        "calibration": CURRENT_CALIB_TYPE,
        "aliked_w": w_int[0],
        "mega_w": w_int[1],
        "eva_w": w_int[2],
        "mean_confidence": round(raw_conf, 4),
        "uncertainty": round(unc, 4),
        "new_ratio": round(n_ratio, 4),
        "reliability_score": round(rel_score, 4),
        "rank1": rank1,
        "mAP": m_ap,
        "f1_score": f1
    }

    # metrics_history.csv에 누적 저장 (없으면 생성, 있으면 추가)
    metrics_df = pd.DataFrame([metrics_data])
    metrics_log_file = "metrics_history.csv"
    
    if not os.path.exists(metrics_log_file):
        metrics_df.to_csv(metrics_log_file, index=False)
    else:
        metrics_df.to_csv(metrics_log_file, mode='a', header=False, index=False)
    
    print(f"📊 Metrics log updated in {metrics_log_file}")

    # 7. Save to CSV
    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    output_name = f"{CURRENT_CALIB_TYPE}_Aw{w_int[0]}_Mw{w_int[1]}_Ew{w_int[2]}.csv"   # 파일명에 실험 변수 포함
    df.to_csv(output_name, index=False)
    print(f"✅ {output_name} saved!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()