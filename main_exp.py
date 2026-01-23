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

import os
os.environ['KAGGLE_USERNAME'] = "sua0409"  # Kaggle Username
os.environ['KAGGLE_KEY'] = "KGAT_d7970e69f9a2c68f79a9ef5c127390bb"  # Kaggle API Key
# 해결방안1: .kaggle/kaggle.json 파일 넣고, chmod 600 설정
# 해결방안2: cmd 창에 kaggle API key 직접 입력

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

    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_subset = dataset_query.get_subset(dataset_query.metadata["dataset"] == dataset_name)

        # 6. WildFusion 한 번의 호출로 3개 모델이 통합된 점수 산출
        # 수정2: 내부적으로 ALIKED, Mega, EVA02가 각각 계산되고 가중치 합산됨
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

    # 7. Save to CSV
    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    output_name = f"{CURRENT_CALIB_TYPE}_Aw{CURRENT_WEIGHTS[0]}_Mw{CURRENT_WEIGHTS[1]}_Ew{CURRENT_WEIGHTS[2]}.csv"   # 파일명에 실험 변수 포함
    df.to_csv(output_name, index=False)
    print(f"✅ {output_name} saved!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()