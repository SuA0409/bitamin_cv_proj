from wildlife_datasets.datasets import AnimalCLEF2025
import pandas as pd
from torchvision.transforms import functional as TF

'''
AnimalCLEF2025 로드하고, query/database/calibration 분리까지 담당
'''
def salamander_orientation_transform(image, metadata):
    # Only apply to SalamanderID2025 dataset
    if metadata.get("dataset") == "SalamanderID2025":
        orientation = metadata.get("orientation", "top")
        # Align 'right' orientation to 'top' by rotating -90 degrees
        if orientation == "right":
            return TF.rotate(image, -90)
        # Align 'left' orientation to 'top' by rotating +90 degrees
        elif orientation == "left":
            return TF.rotate(image, 90)
        # 'top' orientation needs no change
    return image

def load_datasets(root, calibration_size=1000):
    AnimalCLEF2025.get_data(root)
    # Apply rotation transform for SalamanderID2025 samples during dataset loading
    dataset = AnimalCLEF2025(root, load_label=True, transform=salamander_orientation_transform)
    
    dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
    dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

    calib_meta = dataset_database.metadata.iloc[:calibration_size].copy()
    
    dataset_calibration = AnimalCLEF2025(
        root, df=calib_meta, load_label=True, transform=salamander_orientation_transform
    )
    
    return dataset, dataset_database, dataset_query, dataset_calibration


# Return database and query datasets split by species
def load_datasets_by_species(root, calibration_size=1000):
    # 모든 종을 포함하는 전체 데이터셋 로드
    dataset = AnimalCLEF2025(root, load_label=True, transform=salamander_orientation_transform)

    species_groups = {}
    # 메타데이터에 있는 모든 종(dataset 명칭)에 대해 루프
    for dataset_name in dataset.metadata['dataset'].unique():
        is_dataset = dataset.metadata['dataset'] == dataset_name

        # 해당 종의 DB와 Query 필터링
        db_df = dataset.metadata[is_dataset & (dataset.metadata['split'] == 'database')]
        query_df = dataset.metadata[is_dataset & (dataset.metadata['split'] == 'query')]

        if len(db_df) == 0:
            continue
        
        print(f"[INFO] Dataset: {dataset_name} | Total DB samples: {len(db_df)}")

        # Calibration 데이터 샘플링
        calib_df = db_df.sample(n=min(calibration_size, len(db_df)), random_state=42)
        
        # 가중치 실험을 위해 DB에서 calib 샘플을 제외할 수도 있고, 포함할 수도 있음
        # 일단은 중복 없이 분리하는 기존 방식을 유지
        db_df_remain = db_df.drop(calib_df.index)

        # 개별 종에 대한 데이터셋 객체 생성
        dataset_db = AnimalCLEF2025(root, df=db_df_remain, load_label=True, transform=salamander_orientation_transform)
        dataset_query = AnimalCLEF2025(root, df=query_df, load_label=True, transform=salamander_orientation_transform)
        dataset_calib = AnimalCLEF2025(root, df=calib_df, load_label=True, transform=salamander_orientation_transform)

        species_groups[dataset_name] = {
            'db': dataset_db,
            'query': dataset_query,
            'calib': dataset_calib
        }

    return species_groups